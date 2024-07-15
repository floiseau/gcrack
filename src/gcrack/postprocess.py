import numpy as np

import ufl
from dolfinx import mesh, fem

from domain import Domain
from models import ElasticModel


def compute_reaction_forces(
    domain: Domain, model: ElasticModel, uh: fem.Function, gcrack_data
) -> np.array:
    """Compute the reaction forces of the domain.

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

    # Locate the entities on the top boundary
    top_facets = mesh.locate_entities(
        domain.mesh, dim - 1, gcrack_data.locate_reaction_forces
    )
    markers = np.full_like(top_facets, 1, dtype=np.int32)
    facet_tags = mesh.meshtags(domain.mesh, dim - 1, top_facets, markers)

    # Get the integrand over the boundary
    ds = ufl.Measure(
        "ds",
        domain=domain.mesh,
        subdomain_data=facet_tags,
        subdomain_id=1,
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
