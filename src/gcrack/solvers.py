from domain import Domain
from models import ElasticModel
from boundary_conditions import BoundaryConditions, get_dirichlet_boundary_conditions

import ufl
import dolfinx
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem


def solve_elastic_problem(
    domain: Domain, model: ElasticModel, bcs: BoundaryConditions
) -> fem.Function:
    print("-- Find the elastic solution with FEM")

    # Define the displacement function space
    shape_u = (domain.mesh.geometry.dim,)
    V_u = fem.functionspace(domain.mesh, ("Lagrange", 1, shape_u))

    # Define the boundary conditions
    dirichlet_bcs = get_dirichlet_boundary_conditions(domain, V_u, bcs)

    # Define the variational formulation
    u = ufl.TrialFunction(V_u)
    v = ufl.TestFunction(V_u)
    a = ufl.inner(model.sig(u), model.eps(v)) * ufl.dx
    L = compute_external_work(domain, v, bcs)

    #  Define and solve the problem
    problem = LinearProblem(
        a,
        L,
        bcs=dirichlet_bcs,
        # petsc_options={
        #     "ksp_type": "cg",
        #     "ksp_rtol": 1e-12,
        #     "ksp_atol": 1e-12,
        #     "ksp_max_it": 1000,
        #     "pc_type": "gamg",
        #     "pc_gamg_agg_nsmooths": 1,
        #     "pc_gamg_esteig_ksp_type": "cg",
        # },
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "cholesky",
            "pc_factor_mat_solver_type": "cholmod",
        },
    )
    return problem.solve()


def compute_external_work(
    domain: Domain, v: dolfinx.fem.Function, bcs: BoundaryConditions
) -> ufl.classes.Form:
    """
    Compute the external work on the boundary of the domain due to imposed forces.

    This function calculates the external work on a boundary of the given mesh by
    integrating the dot product of imposed traction forces and a test function
    over the relevant boundary entities.

    Args:
        domain (gcrack.Domain): The finite element mesh representing the domain.
        v (dolfinx.fem.Function): The test function representing the virtual displacement or velocity.
        bcs: Object containing the boundary conditions.

    Returns:
        ufl.classes.Form: An UFL form representing the external work, which can be integrated
        over the domain or used in variational formulations.

    """
    # Initialize the external work
    f = fem.Constant(domain.mesh, [0.0, 0.0])
    external_work = ufl.dot(f, v) * ufl.dx
    # Iterate through the forces
    for f_bc in bcs.force_bcs:
        # Define the integrand
        ds = ufl.Measure(
            "ds",
            domain=domain.mesh,
            subdomain_data=domain.facet_markers,
            subdomain_id=f_bc.boundary_id,
        )
        T = ufl.as_vector(f_bc.f_imp)
        # Add the contribution to the external work
        external_work += ufl.dot(T, v) * ds
    return external_work
