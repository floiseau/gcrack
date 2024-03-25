import sys
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import gmsh
import numpy as np
from scipy.optimize import minimize, differential_evolution

import ufl
import dolfinx
from dolfinx import io, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

import pyvista

### TODO
#   - [] Computations to plot the force-displacement curve
#       - [] Compute and export of the reaction force (for a unitary loading)
#       - [] Compute and export the boundary displacement (for a unitary loading)
#   - [] Compute the crack length
#   - [] Change the elastic solver
#   - [] Try to incorporate the cracks in the mesh (in the initial mesh) to avoid a complete remeshing
#       - The crack plugin in GMSH seems able to do this but does not seems to work with FEniCSx
#       - This forum post says that it works: https://fenicsproject.discourse.group/t/very-intersting-question/10276
#       - See also https://fenicsproject.discourse.group/t/interior-integration-measure-for-n-1-dimensional-boundary/8701
#       - One of the issue is that, for open cracks, one of the physical group is of dim 0 which FEniCSx does not like.

### Constants
# Geometry
L: float = 1e-3
# Mechanical
E: float = 230.77e9
nu: float = 0.43
la: float = E * nu / ((1 + nu) * (1 - 2 * nu))  # Plane strain
mu: float = E / (2 * (1 + nu))
# Numerical
R_int: float = L / 64
h_min: float = R_int / 32
h: float = L / 64
da = 1e-5


def generate_mesh(crack_points: List[Tuple[float, float, float]]) -> None:
    # Clear existing model
    gmsh.clear()
    # Points
    # Bot
    p1: int = gmsh.model.geo.addPoint(0, 0, 0, h)
    p2: int = gmsh.model.geo.addPoint(L, 0, 0, h)
    p3: int = gmsh.model.geo.addPoint(L, L / 2, 0, h)  # Mid right node
    pc_bot: List[int] = []
    pc_top: List[int] = []
    for i, p in enumerate(reversed(crack_points)):
        # The crack tip is shared
        if i == 0:
            pc_new: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
            pc_bot.append(pc_new)
            pc_top.append(pc_new)
        else:
            pc_new_bot: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
            pc_bot.append(pc_new_bot)
            pc_new_top: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
            pc_top.append(pc_new_top)
    p5: int = gmsh.model.geo.addPoint(0, L / 2, 0, h)  # Bot crack lip
    # Top
    p6: int = gmsh.model.geo.addPoint(0, L, 0, h)
    p7: int = gmsh.model.geo.addPoint(L, L, 0, h)
    # Point(13) // Mid right node
    # Point(14) // Crack tip
    p8: int = gmsh.model.geo.addPoint(0, L / 2, 0, h)  # Top crack lip

    # Lines
    # Bot
    l1: int = gmsh.model.geo.addLine(p1, p2)
    l2: int = gmsh.model.geo.addLine(p2, p3)
    l3: int = gmsh.model.geo.addLine(p3, pc_bot[0])
    crack_lines_bot: List[int] = []
    for i in range(len(pc_bot) - 1):
        l: int = gmsh.model.geo.addLine(pc_bot[i], pc_bot[i + 1])
        crack_lines_bot.append(l)
    crack_lines_bot.append(gmsh.model.geo.addLine(pc_bot[-1], p5))
    l5: int = gmsh.model.geo.addLine(p5, p1)
    # Top
    l6: int = gmsh.model.geo.addLine(p6, p7)
    l7: int = gmsh.model.geo.addLine(p7, p3)
    # Line(13)
    # Top  crack line
    crack_lines_top: List[int] = []
    for i in range(len(pc_bot) - 1):
        l: int = gmsh.model.geo.addLine(pc_top[i], pc_top[i + 1])
        crack_lines_top.append(l)
    crack_lines_top.append(gmsh.model.geo.addLine(pc_top[-1], p8))
    l9: int = gmsh.model.geo.addLine(p8, p6)

    # Surfaces
    # Bot
    cl1: int = gmsh.model.geo.addCurveLoop([l1, l2, l3] + crack_lines_bot + [l5])
    s1: int = gmsh.model.geo.addPlaneSurface([cl1])
    # Top
    cl2: int = gmsh.model.geo.addCurveLoop([l6, l7, l3] + crack_lines_top + [l9])
    s2: int = gmsh.model.geo.addPlaneSurface([cl2])

    # Physical groups
    # Domain
    domain: int = gmsh.model.addPhysicalGroup(2, [s1, s2], tag=21)
    gmsh.model.setPhysicalName(2, domain, "domain")
    # Boundaries
    bot: int = gmsh.model.addPhysicalGroup(1, [l1], tag=11)
    gmsh.model.setPhysicalName(1, bot, "bot")
    top: int = gmsh.model.addPhysicalGroup(1, [l6], tag=12)
    gmsh.model.setPhysicalName(1, top, "top")

    # Element size
    # Refine around the crack line
    field1: int = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(field1, "PointsList", [pc_bot[0]])
    gmsh.model.mesh.field.setNumber(field1, "Sampling", 100)
    field2: int = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(field2, "InField", field1)
    gmsh.model.mesh.field.setNumber(field2, "DistMin", 2 * R_int)
    gmsh.model.mesh.field.setNumber(field2, "DistMax", 4 * R_int)
    gmsh.model.mesh.field.setNumber(field2, "SizeMin", h_min)
    gmsh.model.mesh.field.setNumber(field2, "SizeMax", h)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.field.setAsBackgroundMesh(field2)
    gmsh.model.mesh.generate(2)


def show_mesh():
    gmsh.fltk.run()


def solve_elastic_problem():
    # Read the mesh with dolfinx
    gdim = 2
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    domain, cell_markers, facet_markers = io.gmshio.model_to_mesh(
        gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim
    )

    # Define the state
    element_u = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
    V_u = fem.FunctionSpace(domain, element_u)

    # Define boundary conditions
    bcs = []

    def on_bot_boundary(x):
        return np.isclose(x[1], 0)

    comp = 1
    bot_dofs = fem.locate_dofs_geometrical(
        (V_u.sub(comp), V_u.sub(comp).collapse()[0]),
        on_bot_boundary)
    u0_func = fem.Function(V_u.sub(comp).collapse()[0])
    with u0_func.vector.localForm() as bc_local:
        bc_local.set(0.0)
    bot_bc = fem.dirichletbc(u0_func, bot_dofs, V_u)

    def on_locked_point(x):
        return np.logical_and(np.isclose(x[0], 0), np.isclose(x[1], 0))

    comp = 0
    locked_dofs = fem.locate_dofs_geometrical(V_u, on_locked_point)
    locked_bc = fem.dirichletbc(np.array([0.0, 0.0]), locked_dofs, V_u)

    def on_uimp_boundary(x):
        return np.isclose(x[1], 1e-3)

    comp = 1
    uimp_dofs = fem.locate_dofs_geometrical(
        (V_u.sub(comp), V_u.sub(comp).collapse()[0]),
        on_uimp_boundary)
    uimp_func = fem.Function(V_u.sub(comp).collapse()[0])
    with uimp_func.vector.localForm() as bc_local:
        bc_local.set(1.0)
    uimp_bc = fem.dirichletbc(uimp_func, uimp_dofs, V_u)

    bcs = [bot_bc, locked_bc, uimp_bc]

    # Define the variational formulation

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return la * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

    ds = ufl.Measure("ds", domain=domain)
    T = fem.Constant(domain, default_scalar_type((0, 0)))

    u = ufl.TrialFunction(V_u)
    v = ufl.TestFunction(V_u)
    f = fem.Constant(domain, default_scalar_type((0, 0)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

    #  Define and solve the problem
    problem = LinearProblem(
        a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    uh = problem.solve()

    # Compute the reaction force
    dim = 2
    n = ufl.FacetNormal(domain)
    def on_top_boundary(x):
        return np.isclose(x[1], 1e-3)
    top_dofs = fem.locate_dofs_geometrical(V_u, on_top_boundary)

    top_facets = dolfinx.mesh.locate_entities(
        domain, dim-1, on_top_boundary)
    markers = np.full_like(top_facets, 1, dtype=np.int32),
    facet_tags = dolfinx.mesh.meshtags(domain, dim-1, top_facets, markers)
    ds = ufl.Measure(
        "ds",
        domain=domain,
        subdomain_data=facet_tags,
        subdomain_id=1,
    )
    # Add the cohtribution to the external work
    f = np.empty((2,))
    for comp in range(dim):
        # Elementary vector
        elem_vec_np = np.zeros((dim,))
        elem_vec_np[comp] = 1
        elem_vec = fem.Constant(domain, elem_vec_np)
        # Set the expression of the reaction force along direction "comp"
        expr = ufl.dot(ufl.dot(sigma(uh), n), elem_vec) * ds
        # Get the associated form
        form = fem.form(expr)
        # Store the expression
        f[comp] = fem.assemble_scalar(form)

    return domain, uh, f


def compute_theta_field(domain, crack_tip, R_int):
    # Set the parameters
    R_ext = 2 * R_int

    # Define the distance to the crack tip
    def distance_to_crack_tip(x):
        return np.sqrt((x[0] - crack_tip[0]) ** 2 + (x[1] - crack_tip[1]) ** 2)

    # Define the variational problem to define theta
    V_theta = fem.FunctionSpace(domain, ("Lagrange", 1))
    theta, theta_ = ufl.TrialFunction(V_theta), ufl.TestFunction(V_theta)
    a = ufl.dot(ufl.grad(theta), ufl.grad(theta_)) * ufl.dx
    L = fem.Constant(domain, default_scalar_type(0.0)) * theta_ * ufl.dx(domain=domain)
    # Set the boundary conditions
    # Imposing 1 in the inner circle and zero in the outer circle
    dofs_inner = fem.locate_dofs_geometrical(
        V_theta, lambda x: distance_to_crack_tip(x) <= R_int
    )
    dofs_out = fem.locate_dofs_geometrical(
        V_theta, lambda x: distance_to_crack_tip(x) >= R_ext
    )
    bc_inner = fem.dirichletbc(default_scalar_type(1.0), dofs_inner, V_theta)
    bc_out = fem.dirichletbc(default_scalar_type(0.0), dofs_out, V_theta)
    bcs = [bc_out, bc_inner]
    # Solve the problem
    problem = fem.petsc.LinearProblem(a, L, bcs=bcs)
    return problem.solve()


def G(domain, u, xc):
    # Get the theta field
    theta_field = compute_theta_field(domain, xc, R_int)
    # Compute the energy release rate form
    eps = ufl.sym(ufl.grad(u))
    sig = la * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * eps
    dx = ufl.dx(domain=domain)
    P = (
        1 / 2 * ufl.inner(sig, eps) * ufl.Identity(2)
        # - ufl.transpose(ufl.dot(sig, ufl.grad(u)))
        - ufl.dot(ufl.transpose(ufl.grad(u)), sig)
    )
    G_expr = -ufl.dot(P, ufl.grad(theta_field))
    G_1 = fem.assemble_scalar(fem.form(G_expr[0] * dx))
    G_2 = fem.assemble_scalar(fem.form(G_expr[1] * dx))
    # Initialize the
    return np.array([G_1, G_2])


def export_function(u, t, dir_name):
    # Get function info
    V = u.function_space
    mesh = V.mesh
    vtkfile = io.VTKFile(mesh.comm, dir_name / f"u_{t:04d}.pvd", "w")
    vtkfile.write_function(u, 0)
    vtkfile.close()


def Gc(gamma):
    if True:
        Gc_min = 2700
        Gc_max = Gc_min*10
        theta0 = np.pi/6
        return Gc_min + (Gc_max-Gc_min)*np.sqrt(1/2*(1-np.cos(2*gamma-theta0)))
    else:
        return 2700


def residual(x):
    gamma = x[0]
    t = np.array([np.cos(gamma), np.sin(gamma)])
    g = np.maximum(np.dot(g_vec, t), 1e-12)
    gc = Gc(gamma)
    return np.sqrt(gc / g)


def compute_load_factor(gamma0):
    return differential_evolution(
        residual,
        bounds=[(gamma0 - np.pi, gamma0 + np.pi)],
        popsize=128,
        workers=16,
    )


def export_dict_to_csv(data, filename):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header containing keys of the dictionary
        writer.writerow(data.keys())

        # Determine the length of the longest list in the dictionary
        max_length = max(len(lst) for lst in data.values())

        # Iterate through the lists and write rows to the CSV file
        for i in range(max_length):
            row = [data[key][i] if i < len(data[key]) else None for key in data.keys()]
            writer.writerow(row)


if __name__ == "__main__":
    # Initialize GMSH
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Disable terminal output
    gmsh.option.setNumber("Mesh.Algorithm", 8)  # Use Delaunay algorithm
    # Initialize export directory
    now = datetime.now()
    dir_name = Path("results_" + now.strftime("%Y-%m-%d_%H-%M-%S"))
    dir_name.mkdir(parents=True, exist_ok=True)

    # Initialize the crack points
    crack_points = [
        np.array([L / 2, L / 2, 0]),
    ]
    # Store the results
    res = {
        "gamma": [0],
        "lambda": [0],
        "xc_1": [crack_points[0][0]],
        "xc_2": [crack_points[0][1]],
        "xc_3": [crack_points[0][2]],
        "uimp_1": [0.0],
        "uimp_2": [0.0],
        "fimp_1": [0.0],
        "fimp_2": [0.0],
    }

    for t in range(40):
        # Get current crack tip
        xc = crack_points[-1]
        # Generate the mesh
        generate_mesh(crack_points)
        # show_mesh()
        # Solve the elastic problem
        domain, u, f = solve_elastic_problem()
        export_function(u, t, dir_name)

        # Compute the energy release rate vector
        g_vec = G(domain, u, xc)

        # Compute the load factor and crack angle.
        gamma0 = res["gamma"][-1]
        opti_res = compute_load_factor(gamma0)

        # Get the results
        gamma_ = opti_res.x[0]
        lambda_ = opti_res.fun
        # Add a new crack point
        da_vec = da * np.array([np.cos(gamma_), np.sin(gamma_), 0])
        xc_new = xc + da_vec
        crack_points.append(xc_new)

        print(f"==== Time step {t}")
        print(f"crack_tip={xc_new}")
        print(f"gamma={gamma_}")
        print(f"lambda={lambda_}")
        # Store the results
        res["gamma"].append(gamma_)
        res["lambda"].append(lambda_)
        res["xc_1"].append(xc_new[0])
        res["xc_2"].append(xc_new[1])
        res["xc_3"].append(xc_new[2])
        res["uimp_1"].append(0.0)
        res["uimp_2"].append(1.0)
        res["fimp_1"].append(f[0])
        res["fimp_2"].append(f[1])
    # Export the dictionary to a CSV file
    export_dict_to_csv(res, dir_name / "results.csv")
    # Clean up
    gmsh.finalize()
