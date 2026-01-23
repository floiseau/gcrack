from pathlib import Path
from typing import List

import gmsh
import numpy as np
import jax.numpy as jnp

import gmsh
from gcrack import GCrackBase
from gcrack.boundary_conditions import DisplacementBC, ForceBC, BodyForce


class GCrackData(GCrackBase):
    def generate_mesh(self, crack_points: List[np.ndarray]) -> gmsh.model:
        # Clear existing model
        gmsh.clear()
        # Parameters
        Lx = self.pars["Lx"]
        Ly = self.pars["Ly"]
        h = Lx / 128
        h_min = self.R_int / 16
        # Points
        p1: int = gmsh.model.geo.addPoint(0, 0, 0, h)
        p2: int = gmsh.model.geo.addPoint(Lx, 0, 0, h)
        p3: int = gmsh.model.geo.addPoint(Lx, Ly, 0, h)
        p4: int = gmsh.model.geo.addPoint(0, Ly, 0, h)

        # Lines
        l1: int = gmsh.model.geo.addLine(p1, p2)
        l2: int = gmsh.model.geo.addLine(p2, p3)
        l3: int = gmsh.model.geo.addLine(p3, p4)
        l4: int = gmsh.model.geo.addLine(p4, p1)

        # Surfaces
        cl1: int = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        s1: int = gmsh.model.geo.addPlaneSurface([cl1])

        # Boundaries
        self.boundaries = {
            "left": l4,
            "right": l2,
        }
        # Physical groups
        # Domain
        domain: int = gmsh.model.addPhysicalGroup(2, [s1], tag=21)
        gmsh.model.setPhysicalName(2, domain, "domain")
        # Boundaries
        left: int = gmsh.model.addPhysicalGroup(
            1, [self.boundaries["left"]], tag=self.boundaries["left"]
        )
        gmsh.model.setPhysicalName(1, left, "left")
        right: int = gmsh.model.addPhysicalGroup(
            1, [self.boundaries["right"]], tag=self.boundaries["right"]
        )
        gmsh.model.setPhysicalName(1, right, "right")

        # Generate mesh
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

        # Return the model
        return gmsh.model()

    def locate_measured_displacement(self) -> List[float]:
        """Define the point where the displacement is measured.

        Returns:
            List: Coordinate of the point where the displacement is measured
        """
        return [self.pars["Lx"], self.pars["Ly"] / 2]

    def locate_measured_forces(self) -> int:
        """Define the boundary where the reaction force are measured.

        Returns:
            int: Identifier (id) of the boundary in GMSH.
        """
        return self.boundaries["right"]

    def define_controlled_displacements(self) -> List[DisplacementBC]:
        """Define the imposed displacement boundary conditions.

        Returns:
            List[DisplacementBC]: List of DisplacementBC(boundary_id, u_imp) where boundary_id is the boundary id (int number) in GMSH, and u_imp is the displacement vector (componements can be nan to let it free).
        """
        return [
            DisplacementBC(
                boundary_id=self.boundaries["left"], u_imp=[0, float("nan")]
            ),
        ]

    def define_controlled_body_forces(self) -> List[BodyForce]:
        """Define the controlled body forces that are affected by the load factor.

        Returns:
            List[BodyForce]: List of BodyForce (f_imp) where f_imp is the force vector.
        """
        return [BodyForce(f_imp=[0.0, "-x[0]"])]

    def define_locked_points(self) -> List[List[float]]:
        """Define the list of locked points.

        Returns:
            List[List[float]]: A list of points (list) coordinates.
        """
        return [
            [0, 0, 0],
        ]

    def Gc(self, phi):
        return self.pars["Gc"] + 0.0 * phi


if __name__ == "__main__":
    # Define user parameters
    pars = {
        "Lx": 10.0,
        "Ly": 1.0,
        "Gc": 10_000,
    }

    gcrack_data = GCrackData(
        E=1,
        nu=0.0,
        da=pars["Lx"] / 128,
        Nt=1,
        xc0=[pars["Lx"] / 2, pars["Ly"] / 2, 0],
        assumption_2D="plane_strain",
        pars=pars,
        sif_method="i-integral",  # "i-integral" "willliams"
        name="numeric",
    )
    gcrack_data.run()
