import numpy as np

import ufl
from dolfinx import mesh, fem

from domain import Domain
from models import ElasticModel


def compute_reaction_forces(
    domain: Domain, model: ElasticModel, uh: fem.Function
) -> np.array:
    # Compute the reaction force
    dim: int = domain.dim
    n: ufl.FacetNormal = ufl.FacetNormal(domain.mesh)

    def on_top_boundary(x):
        return np.isclose(x[1], 1)

    top_facets = mesh.locate_entities(domain.mesh, dim - 1, on_top_boundary)
    markers = np.full_like(top_facets, 1, dtype=np.int32)
    facet_tags = mesh.meshtags(domain.mesh, dim - 1, top_facets, markers)
    ds = ufl.Measure(
        "ds",
        domain=domain.mesh,
        subdomain_data=facet_tags,
        subdomain_id=1,
    )
    # Add the cohtribution to the external work
    f = np.empty((2,))
    for comp in range(dim):
        # Elementary vector
        elem_vec_np = np.zeros((dim,))
        elem_vec_np[comp] = 1
        elem_vec = fem.Constant(domain.mesh, elem_vec_np)
        # Set the expression of the reaction force along direction "comp"
        expr = ufl.dot(ufl.dot(model.sig(uh), n), elem_vec) * ds
        # Get the associated form
        form = fem.form(expr)
        # Store the expression
        f[comp] = fem.assemble_scalar(form)

    return f
