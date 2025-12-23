"""
Module with solution method for the elastic problem.

This module provides functions for solving elastic problems using the finite element method.

Functions:
    solve_elastic_problem:
        Solves an elastic problem using the finite element method.
    compute_external_work:
        Computes the external work due to imposed forces and body forces.
"""

from gcrack.utils.expression_parsers import parse_expression
from gcrack.domain import Domain
from gcrack.models import ElasticModel
from gcrack.boundary_conditions import (
    BoundaryConditions,
    get_dirichlet_boundary_conditions,
)

import ufl
import dolfinx
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem


def solve_elastic_problem(
    domain: Domain, model: ElasticModel, bcs: BoundaryConditions
) -> fem.Function:
    """Solves an elastic problem using the finite element method.

    This function sets up and solves a linear elastic problem using the finite element method.
    It defines the function space, boundary conditions, and variational formulation based on the elastic energy and external work.
    The problem is solved using PETSc's LinearProblem.

    Args:
        domain (Domain): The domain object representing the physical space.
        model (ElasticModel): The elastic model defining the material properties.
        bcs (BoundaryConditions): The boundary conditions for the problem.

    Returns:
        fem.Function: The displacement solution of the elastic problem.

    Raises:
        ValueError: If the 2D assumption is unknown.
    """
    # Define the displacement function space
    if model.assumption.startswith("plane"):
        shape_u = (2,)
    elif model.assumption in ["anti_plane"]:
        shape_u = (1,)
    else:
        raise ValueError(f"Unknown 2D assumption: {model.assumption}.")
    V_u = fem.functionspace(domain.mesh, ("Lagrange", 1, shape_u))
    # Define the displacement field
    u = fem.Function(V_u, name="Displacement")
    # Define the boundary conditions
    dirichlet_bcs = get_dirichlet_boundary_conditions(domain, V_u, bcs)
    # Define the total energy
    energy = model.elastic_energy(u, domain)
    external_work = compute_external_work(domain, u, bcs)
    if external_work:
        energy -= external_work
    # Derive the energy to obtain the variational formulation
    E_u = ufl.derivative(energy, u, ufl.TestFunction(V_u))
    E_du = ufl.replace(E_u, {u: ufl.TrialFunction(V_u)})
    # Define the variational formulation
    a = ufl.lhs(E_du)
    L = ufl.rhs(E_du)

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
        petsc_options_prefix="basic_linear_problem",
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
    """Computes the external work due to imposed forces and body forces.

    This function calculates the external work on a boundary of the given mesh by integrating the dot product of imposed traction forces and a test function over the relevant boundary entities.
    It also accounts for body forces applied within the domain.

    Args:
        domain (Domain):
            The finite element mesh representing the domain.
        v (dolfinx.fem.Function):
            The test function representing the virtual displacement or velocity.
        bcs (BoundaryConditions):
            Object containing the boundary conditions, including body forces and force boundary conditions.

    Returns:
        ufl.classes.Form:
            A UFL form representing the external work, which can be integrated over the domain or used in variational formulations.
    """

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
        external_work(ufl.classes.Form):
            An UFL form representing the external work, which can be integrated over the domain or used in variational formulations.

    """
    # Get the number of components in u
    N_comp = v.function_space.value_shape[0]
    # Initialize the external work
    f = fem.Constant(domain.mesh, [0.0] * N_comp)
    external_work = ufl.dot(f, v) * ufl.dx
    # Create a function space for body forces
    bf_space = fem.functionspace(domain.mesh, ("Lagrange", 1))
    # Iterate through the body forces
    for bf in bcs.body_forces:
        # Define the integrand
        dx = ufl.Measure("dx", domain=domain.mesh)
        # Convert to ufl vector
        f_list = []
        for i, f_comp in enumerate(bf.f_imp):
            f_comp_parsed = parse_expression(f_comp, bf_space)
            f_list.append(f_comp_parsed)
        f = ufl.as_vector(f_list)
        # Add constant body force to the external work
        external_work += ufl.dot(f, v) * dx

    # Iterate through the force boundary conditions
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
