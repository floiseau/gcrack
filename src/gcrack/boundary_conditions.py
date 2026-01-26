"""
Module for defining and processing boundary conditions in finite element simulations.

This module provides data structures and functions to handle boundary conditions,
including displacement, force, body forces, locked points, and nodal displacements.
It is designed to work with the `dolfinx` library for finite element analysis.

Classes:
    DisplacementBC: Represents a displacement boundary condition.
    ForceBC: Represents a force boundary condition.
    BodyForce: Represents a body force applied within a domain.
    NodalDisplacement: Represents a nodal displacement condition.
    BoundaryConditions: Aggregates all boundary conditions for a simulation.

Functions:
    get_dirichlet_boundary_conditions: Constructs and returns a list of Dirichlet boundary conditions for a given domain and function space.
"""

from dataclasses import dataclass
from typing import List
from math import isnan

import numpy as np

import dolfinx
from dolfinx import fem

from gcrack.domain import Domain
from gcrack.utils.expression_parsers import parse_expression


@dataclass
class DisplacementBC:
    """
    Represents a displacement boundary condition.

    Attributes:
        boundary_id (int): Unique identifier for the boundary where the displacement is applied.
        u_imp (List[float]): Imposed displacement values for the boundary.
    """

    boundary_id: int
    u_imp: List[float]


@dataclass
class ForceBC:
    """
    Represents a force boundary condition.

    Attributes:
        boundary_id (int): Unique identifier for the boundary where the force is applied.
        f_imp (List[float]): Imposed force values for the boundary.
    """

    boundary_id: int
    f_imp: List[float]


@dataclass
class BodyForce:
    """
    Represents a body force applied within a domain.

    Attributes:
        f_imp (List[float]): Imposed body force values.
    """

    f_imp: List[float]


@dataclass
class NodalDisplacement:
    """
    Represents a nodal displacement condition.

    Attributes:
        x (List[float]): Coordinates of the node.
        u_imp (List[float]): Imposed displacement values for the node.
    """

    x: List[float]
    u_imp: List[float]


@dataclass
class BoundaryConditions:
    """
    Aggregates all boundary conditions for a simulation.

    Attributes:
        displacement_bcs (List[DisplacementBC]): List of displacement boundary conditions.
        force_bcs (List[ForceBC]): List of force boundary conditions.
        body_forces (List[BodyForce]): List of body forces applied within the domain.
        locked_points (List[List[float]]): List of coordinates for points that are locked (fixed).
        nodal_displacements (List[NodalDisplacement]): List of nodal displacement conditions.
    """

    displacement_bcs: List[DisplacementBC]
    force_bcs: List[ForceBC]
    body_forces: List[BodyForce]
    locked_points: List[List[float]]
    nodal_displacements: List[NodalDisplacement]

    def is_empty(self) -> bool:
        """
        Checks if all boundary condition lists are empty or None.

        Returns:
            bool: True if all boundary condition lists are empty or None, False otherwise.
        """
        return all(
            not lst
            for lst in (
                self.displacement_bcs,
                self.force_bcs,
                self.body_forces,
                self.nodal_displacements,
            )
        )

    def _is_null_or_nan(self, value):
        if isinstance(value, (int, float)):
            return value == 0 or isnan(value)
        elif isinstance(value, str):
            return False

    def is_null(self) -> bool:
        """
        Check if all boundary conditions and forces are zero, NaN, or if all lists are empty.

        Returns:
            bool: True if all lists are empty or all their values are zero or NaN, False otherwise.
        """

        conditions = [
            all(
                self._is_null_or_nan(comp)
                for bc in self.displacement_bcs
                for comp in bc.u_imp
            ),
            all(
                self._is_null_or_nan(comp) for bc in self.force_bcs for comp in bc.f_imp
            ),
            all(
                self._is_null_or_nan(comp)
                for bc in self.body_forces
                for comp in bc.f_imp
            ),
            all(
                self._is_null_or_nan(comp)
                for bc in self.nodal_displacements
                for comp in bc.u_imp
            ),
        ]
        return all(conditions)


def get_dirichlet_boundary_conditions(
    domain: Domain,
    V_u: dolfinx.fem.FunctionSpace,
    bcs: BoundaryConditions,
) -> List[dolfinx.fem.dirichletbc]:
    """
    Constructs and returns a list of Dirichlet boundary conditions for a given domain and function space.

    This function processes displacement boundary conditions, locked points, and nodal displacements to create Dirichlet boundary conditions for use in finite element simulations.

    Args:
        domain (Domain): The computational domain, including mesh and facet markers.
        V_u (dolfinx.fem.FunctionSpace): The function space for the displacement field.
        bcs (BoundaryConditions): An object containing all boundary conditions, including displacement, locked points, and nodal displacements.

    Returns:
        A list of Dirichlet boundary conditions for the displacement field.

    Notes:
        - For each displacement boundary condition, the function locates the relevant degrees of freedom (DOFs)
          and creates a Dirichlet boundary condition for each component.
        - Locked points are treated as fixed (zero displacement) boundary conditions.
        - Nodal displacements are imposed at specific nodes, with support for multi-component fields.
        - The function handles both scalar and vector function spaces.
    """
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

    # Add the nodal displacements
    for nd in bcs.nodal_displacements:
        # Extract the quantities
        p = np.array(nd.x)
        u_imp = nd.u_imp

        # Define the location function
        def on_locked_point(x):
            return np.logical_and(np.isclose(x[0], p[0]), np.isclose(x[1], p[1]))

        for comp in range(N_comp):
            # Check if the imposed displement is nan
            if isnan(u_imp[comp]):
                continue
            # Get the locked dofs
            dof = fem.locate_dofs_geometrical(
                (V_u.sub(comp), V_u.sub(comp).collapse()[0]), on_locked_point
            )
            if not dof:
                raise ValueError(
                    f"No node found at {nd.x} to impose nodal displacement."
                )
            # Parse the nodal value
            V_u_comp = V_u if N_comp == 1 else V_u.sub(comp).collapse()[0]
            bc_func = parse_expression(u_imp[comp], V_u_comp)
            # Create the Dirichlet boundary condition
            if N_comp == 1:  # TODO: Clean (idk why no syntax works in both cases)
                bc = fem.dirichletbc(bc_func, dof)
            else:
                bc = fem.dirichletbc(bc_func, dof, V_u)
            # Append the boundary condition to the list
            dirichlet_bcs.append(bc)
    return dirichlet_bcs
