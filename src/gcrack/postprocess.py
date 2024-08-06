import numpy as np

import ufl
from dolfinx import geometry, fem

from domain import Domain
from models import ElasticModel


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
    # Get the dimension of the domain
    dim: int = domain.dim
    # Get the normal to the boundary
    n: ufl.FacetNormal = ufl.FacetNormal(domain.mesh)
    # Get the boundary id
    boundary_id = gcrack_data.locate_measured_forces()
    # Get the integrand over the boundary
    ds = ufl.Measure(
        "ds",
        domain=domain.mesh,
        subdomain_data=domain.facet_markers,
        subdomain_id=boundary_id,
    )
    # Initialize the force array
    f = np.empty((2,))
    for comp in range(dim):
        # Elementary vector for the current component
        elem_vec_np = np.zeros((dim,))
        elem_vec_np[comp] = 1
        elem_vec = fem.Constant(domain.mesh, elem_vec_np)
        # Expression for the reaction force for the current component
        expr = ufl.dot(ufl.dot(model.sig(uh), n), elem_vec) * ds
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
    # Generate the bounding box tree
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    # Find cells whose bounding-box collide with the points
    cell_candidates = geometry.compute_collisions_points(tree, x)
    # For each points, choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, x)
    cell = colliding_cells[0]
    # Compute the measured displacement
    u_meas = uh.eval([x], cell)
    # Initialize the probes values
    return u_meas
