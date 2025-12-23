"""
This module provides functionality to represent the domain for the problem.

Classes:
    Domain: Represents the domain for the problem.
"""

from mpi4py import MPI

from dolfinx import io


class Domain:
    """Class representing the domain for the problem.

    This class reads the mesh from a GMSH file and locates physical groups.

    Attributes:
        dim (int): The dimension of the domain.
        mesh (dolfinx.Mesh): The mesh representing the domain.
        cell_tags (numpy.ndarray): Array containing cell tags.
        facet_tags (numpy.ndarray): Array containing facet tags.
        boundary_facets (dict): Dictionary containing boundary facets grouped by physical group name.
    """

    def __init__(self, gmsh_model):
        """
        Initialize the Domain.
        """
        # Get the dimension
        self.dim = gmsh_model.getDimension()
        # Generate mesh data from the gmsh model
        self.mesh_data = io.gmsh.model_to_mesh(
            gmsh_model, MPI.COMM_WORLD, 0, gdim=self.dim
        )
        # Extract the mesh and the tags
        self.mesh = self.mesh_data.mesh
        self.cell_markers = self.mesh_data.cell_tags
        self.facet_markers = self.mesh_data.facet_tags
