"""
Module for computing post-processing quantities in finite element simulations of elastic problems.

This module provides functions to compute various post-processing quantities in finite element simulations, including reaction forces, displacements at specific points, elastic energy, and external work.

Functions:
    compute_measured_forces:
        Computes the reaction forces on a specified boundary.
    compute_measured_displacement:
        Computes the displacement at a specified point.
    compute_elastic_energy:
        Computes the elastic energy in the domain.
    compute_external_work:
        Computes the external work done on the domain.
"""

import numpy as np

import ufl
from dolfinx import geometry, fem

from gcrack.domain import Domain
from gcrack.models import ElasticModel


def compute_measured_forces(
    domain: Domain, model: ElasticModel, uh: fem.Function, gcrack_data
) -> np.array:
    """Compute the measured forces.

    Args:
        domain (Domain): The domain of the problem.
        model (ElasticModel): The elastic model being used.
        uh (Function): The displacement solution of the elastic problem.

    Returns:
        np.array: The computed reaction forces as a numpy array.
    """
    # Get the number of components
    N_comp = uh.function_space.value_shape[0]
    # Get the normal to the boundary
    facet_normal: ufl.FacetNormal = ufl.FacetNormal(domain.mesh)
    n = ufl.as_vector([facet_normal[0], facet_normal[1], 0])
    # Get the boundary id
    boundary_id = gcrack_data.locate_measured_forces()
    # Get the integrand over the boundary
    ds = ufl.Measure(
        "ds",
        domain=domain.mesh,
        subdomain_data=domain.facet_markers,
        subdomain_id=boundary_id,
    )
    # Compute the stress
    sig = model.sig(uh)
    # Compute the traction vector
    T = ufl.dot(sig, n)
    # Initialize the force array
    f = np.empty((3,))
    for comp in range(N_comp):
        # Elementary vector for the current component
        elem_vec_np = np.zeros((3,))
        elem_vec_np[comp] = 1
        elem_vec = fem.Constant(domain.mesh, elem_vec_np)
        # Expression for the reaction force for the current component
        expr = ufl.dot(T, elem_vec) * ds
        # Form for the reaction force expression
        form = fem.form(expr)
        # Assemble the form to get the reaction force component
        f[comp] = fem.assemble_scalar(form)
    return f


def compute_measured_displacement(
    domain: Domain, uh: fem.Function, gcrack_data
) -> np.array:
    """Compute the displacement at the specified point.

    Args:
        domain (Domain): The domain of the problem.
        uh (Function): The displacement solution of the elastic problem.

    Returns:
        np.array: The computed displacement as a numpy array.
    """
    # Get the mesh
    mesh = domain.mesh
    # Get the position of the measurement
    x = gcrack_data.locate_measured_displacement()
    if len(x) == 2:
        x.append(0)
    # Store x in an array
    xs = np.array([x])
    # Generate the bounding box tree
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    # Find cells whose bounding-box collide with the points
    cell_candidates = geometry.compute_collisions_points(tree, xs)
    # For each points, choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, xs)
    cell = colliding_cells.array[0]
    # Compute the measured displacement
    u_meas = uh.eval(xs, cell)
    # Initialize the probes values
    return u_meas


def compute_elastic_energy(
    domain: Domain, model: ElasticModel, uh: fem.Function
) -> float:
    """Compute the elastic energy in the domain.

    Args:
        domain (Domain): The domain of the problem.
        model (ElasticModel): The elastic model being used.
        uh (Function): The displacement solution of the elastic problem.

    Returns:
        float: Elastic energy.
    """
    # Compute the elastic energy
    return fem.assemble_scalar(fem.form(model.elastic_energy(uh, domain)))


def compute_external_work(
    domain: Domain, model: ElasticModel, uh: fem.Function
) -> float:
    """Compute the external work.

    Args:
        domain (Domain): The domain of the problem.
        model (ElasticModel): The elastic model being used.
        uh (Function): The displacement solution of the elastic problem.

    Returns:
        float: External work.
    """
    # Get surface measure
    ds = ufl.Measure("ds", domain=domain.mesh)
    # Get the normal
    facet_normal: ufl.FacetNormal = ufl.FacetNormal(domain.mesh)
    n = ufl.as_vector([facet_normal[0], facet_normal[1], 0])
    # Convert displacement to 3D
    uh3D = model.u_to_3D(uh)
    # Define the ufl expression of the external work
    ew_ufl = ufl.dot(ufl.dot(model.sig(uh), n), uh3D) * ds
    # Compute the elastic energy
    return fem.assemble_scalar(fem.form(ew_ufl))
