from math import isnan

from domain import Domain
from models import ElasticModel

import numpy as np

import ufl
import dolfinx
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem


def solve_elastic_problem(
    domain: Domain,
    model: ElasticModel,
    gcrack_data,
) -> fem.Function:
    print("-- Find the elastic solution with FEM")

    # Define the displacement function space
    shape_u = (domain.mesh.geometry.dim,)
    V_u = fem.functionspace(domain.mesh, ("Lagrange", 1, shape_u))

    # Define the boundary conditions
    bcs = get_dirichlet_boundary_conditions(domain, V_u, gcrack_data)

    # Define the variational formulation
    u = ufl.TrialFunction(V_u)
    v = ufl.TestFunction(V_u)
    a = ufl.inner(model.sig(u), model.eps(v)) * ufl.dx
    L = compute_external_work(domain, v, gcrack_data)

    #  Define and solve the problem
    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={
            "ksp_type": "cg",
            "ksp_rtol": 1e-12,
            "ksp_atol": 1e-12,
            "ksp_max_it": 1000,
            "pc_type": "gamg",
            "pc_gamg_agg_nsmooths": 1,
            "pc_gamg_esteig_ksp_type": "cg",
        },
    )
    return problem.solve()


def get_dirichlet_boundary_conditions(
    domain: Domain, V_u: dolfinx.fem.FunctionSpace, gcrack_data
):
    # Get the dimensions
    dim = domain.mesh.geometry.dim
    fdim = dim - 1
    # Get the boundary condition definitions
    bc_defs = gcrack_data.define_imposed_displacements()
    # Get the facets markers
    facet_markers = domain.facet_markers
    # Get the facets indices
    boundary_facets = {
        facet_value: facet_markers.indices[facet_markers.values == facet_value]
        for facet_name, facet_value in gcrack_data.boundaries.items()
    }
    # Get boundary dofs (per comp)
    boundary_dofs = {
        f"{facet_id}_{comp}": fem.locate_dofs_topological(
            (V_u.sub(comp), V_u.sub(comp).collapse()[0]),
            fdim,
            boundary_facet,
        )
        for comp in range(dim)
        for facet_id, boundary_facet in boundary_facets.items()
    }
    # Create variables to store bcs and loading functions
    bcs = []
    # Iterage through the displacement loadings
    for facet_id, u_imp in bc_defs:
        # Iterate through the axis
        for comp in range(dim):
            # Check if the DOF is imposed
            if isnan(u_imp[comp]):
                continue
            # Define an FEM function (to control the BC)
            bc_func = fem.Function(V_u.sub(comp).collapse()[0])
            # Update the load
            with bc_func.vector.localForm() as bc_local:
                bc_local.set(u_imp[comp])
            # Get the DOFs
            boundary_dof = boundary_dofs[f"{facet_id}_{comp}"]
            # Create the Dirichlet boundary condition
            bc = fem.dirichletbc(bc_func, boundary_dof, V_u)
            # Add the boundary conditions to the list
            bcs.append(bc)

    # Add the locked points
    locked_points = gcrack_data.define_locked_points()
    for p in locked_points:
        # Define the location function
        def on_locked_point(x):
            return np.logical_and(np.isclose(x[0], p[0]), np.isclose(x[1], p[1]))

        # Define locked point boundary condition for the x and y components
        locked_dofs = fem.locate_dofs_geometrical(V_u, on_locked_point)
        locked_bc = fem.dirichletbc(np.array([0.0, 0.0]), locked_dofs, V_u)
        # Append the boundary condition to the list of boundary condition
        bcs.append(locked_bc)
    return bcs


def compute_external_work(
    domain: Domain, v: dolfinx.fem.Function, gcrack_data
) -> ufl.classes.Form:
    """
    Compute the external work on the boundary of the domain due to imposed forces.

    This function calculates the external work on a boundary of the given mesh by
    integrating the dot product of imposed traction forces and a test function
    over the relevant boundary entities.

    Args:
        domain (gcrack.Domain): The finite element mesh representing the domain.
        v (dolfinx.fem.Function): The test function representing the virtual displacement or velocity.
        gcrack_data: An instance containing the method `define_imposed_forces()` which
            returns the imposed forces and their corresponding marker functions.

    Returns:
        ufl.classes.Form: An UFL form representing the external work, which can be integrated
        over the domain or used in variational formulations.

    """
    # Get imposed forces
    imposed_forces = gcrack_data.define_imposed_forces()
    # Initialize the external work
    f = fem.Constant(domain.mesh, [0.0, 0.0])
    external_work = ufl.dot(f, v) * ufl.dx
    # Iterate through the forces
    for id, T_imp in imposed_forces:
        # Define the integrand
        ds = ufl.Measure(
            "ds",
            domain=domain.mesh,
            subdomain_data=domain.facet_markers,
            subdomain_id=id,
        )
        T = ufl.as_vector(T_imp)
        # Add the contribution to the external work
        external_work += ufl.dot(T, v) * ds
    return external_work

    #
