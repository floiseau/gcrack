from dataclasses import dataclass
from typing import List

import numpy as np

import dolfinx
from dolfinx import fem

from gcrack.domain import Domain
from gcrack.utils.expression_parsers import parse_expression


@dataclass
class DisplacementBC:
    boundary_id: int
    u_imp: List[float]


@dataclass
class ForceBC:
    boundary_id: int
    f_imp: List[float]


@dataclass
class BodyForce:
    f_imp: List[float]


@dataclass
class BoundaryConditions:
    displacement_bcs: List[DisplacementBC]
    force_bcs: List[ForceBC]
    body_forces: List[BodyForce]
    locked_points: List[List[float]]

    def is_empty(self) -> bool:
        """Check if all boundary condition lists are empty."""
        return not (self.displacement_bcs or self.force_bcs)


def get_dirichlet_boundary_conditions(
    domain: Domain,
    V_u: dolfinx.fem.FunctionSpace,
    bcs: BoundaryConditions,
):
    # Get the dimensions
    dim = domain.mesh.geometry.dim
    fdim = dim - 1
    # Get the number of components
    N_comp = V_u.value_shape[0]
    # Get the facets markers
    facet_markers = domain.facet_markers
    # Get the facets indices
    boundary_facets = {
        u_bc.boundary_id: facet_markers.indices[
            facet_markers.values == u_bc.boundary_id
        ]
        for u_bc in bcs.displacement_bcs
    }
    # Get boundary dofs (per comp)
    if N_comp == 1:  # Anti-plane
        comp = 0
        boundary_dofs = {
            f"{facet_id}_{comp}": fem.locate_dofs_topological(V_u, fdim, boundary_facet)
            for facet_id, boundary_facet in boundary_facets.items()
        }
    else:
        boundary_dofs = {
            f"{facet_id}_{comp}": fem.locate_dofs_topological(
                (V_u.sub(comp), V_u.sub(comp).collapse()[0]),
                fdim,
                boundary_facet,
            )
            for comp in range(N_comp)
            for facet_id, boundary_facet in boundary_facets.items()
        }
    # Create variables to store bcs and loading functions
    dirichlet_bcs = []
    # Iterage through the displacement loadings
    for u_bc in bcs.displacement_bcs:
        # Iterate through the axis
        for comp in range(N_comp):
            # Parse the boundary condition
            V_u_comp = V_u if N_comp == 1 else V_u.sub(comp).collapse()[0]
            bc_func = parse_expression(u_bc.u_imp[comp], V_u_comp)
            if bc_func is None:
                continue
            # Get the DOFs
            boundary_dof = boundary_dofs[f"{u_bc.boundary_id}_{comp}"]
            # Create the Dirichlet boundary condition
            if N_comp == 1:  # TODO: Clean (idk why no syntax works in both cases)
                bc = fem.dirichletbc(bc_func, boundary_dof)
            else:
                bc = fem.dirichletbc(bc_func, boundary_dof, V_u)
            # Add the boundary conditions to the list
            dirichlet_bcs.append(bc)

    # Add the locked points
    for p in bcs.locked_points:
        # Define the location function
        def on_locked_point(x):
            return np.logical_and(np.isclose(x[0], p[0]), np.isclose(x[1], p[1]))

        # Define locked point boundary condition for the x and y components
        locked_dofs = fem.locate_dofs_geometrical(V_u, on_locked_point)
        locked_bc = fem.dirichletbc(np.array([0.0] * N_comp), locked_dofs, V_u)
        # Append the boundary condition to the list of boundary condition
        dirichlet_bcs.append(locked_bc)
    return dirichlet_bcs
