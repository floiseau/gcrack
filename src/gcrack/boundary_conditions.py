from dataclasses import dataclass
from typing import List
from math import isnan

import numpy as np
import sympy as sp

import dolfinx
from dolfinx import fem

from gcrack.domain import Domain


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
    dirichlet_bcs = []
    # Iterage through the displacement loadings
    for u_bc in bcs.displacement_bcs:
        # Iterate through the axis
        for comp in range(dim):
            # Check if the DOF is imposed
            if isinstance(u_bc.u_imp[comp], str):
                # Parse the function
                x = sp.Symbol("x")
                # Parse the expression using sympy
                par_lambda = sp.utilities.lambdify(x, u_bc.u_imp[comp], "numpy")
                # Create and interpolate the fem function
                bc_func = fem.Function(V_u.sub(comp).collapse()[0])
                bc_func.interpolate(lambda xx: par_lambda(xx))
            elif isnan(u_bc.u_imp[comp]):
                continue
            else:
                # Define an FEM function (to control the BC)
                bc_func = fem.Function(V_u.sub(comp).collapse()[0])
                # Update the load
                with bc_func.x.petsc_vec.localForm() as bc_local:
                    bc_local.set(u_bc.u_imp[comp])
            # Get the DOFs
            boundary_dof = boundary_dofs[f"{u_bc.boundary_id}_{comp}"]
            # Create the Dirichlet boundary condition
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
        locked_bc = fem.dirichletbc(np.array([0.0, 0.0]), locked_dofs, V_u)
        # Append the boundary condition to the list of boundary condition
        dirichlet_bcs.append(locked_bc)
    return dirichlet_bcs
