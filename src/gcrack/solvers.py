import logging

from domain import Domain
from models import ElasticModel

import ufl
import dolfinx
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem


def solve_elastic_problem(
    domain: Domain,
    model: ElasticModel,
    gcrack_data,
) -> fem.Function:
    logging.info("-- Find the elastic solution with FEM")

    # Define the displacement function space
    shape_u = (domain.mesh.geometry.dim,)
    V_u = fem.functionspace(domain.mesh, ("Lagrange", 1, shape_u))

    # Define the boundary conditions
    bcs = gcrack_data.define_imposed_displacements(V_u)

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
        # Add the contribution to the external work
        external_work += ufl.dot(T_imp, v) * ds
    return external_work

    #
