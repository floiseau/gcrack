import sys

from typing import List

import jax.numpy as jnp

import gmsh

from gcrack.gcrack import GCrackBase
from gcrack.boundary_conditions import DisplacementBC, BodyForce


class GCrack(GCrackBase):
    def generate_mesh(self, crack_points: List[jnp.ndarray]) -> gmsh.model:
        # Set the pacman size
        self.R_int = 0.5 * self.da
        self.R_ext = 1.0 * self.da
        # Clear existing model
        gmsh.clear()
        # Parameters
        a = pars["a"]
        b = pars["b"]
        # Coarse mesh
        h = b / 128
        # # Fine mesh
        # h = W / 128
        # h_min = self.R_int / 64
        # Points
        # Bot
        p0: int = gmsh.model.geo.addPoint(0, 0, 0, h)
        pa1: int = gmsh.model.geo.addPoint(-a, 0, 0, h)
        pa2: int = gmsh.model.geo.addPoint(a, 0, 0, h)
        pb1: int = gmsh.model.geo.addPoint(-b, 0, 0, h)
        pb2: int = gmsh.model.geo.addPoint(b, 0, 0, h)
        # Lines
        lbb: int = gmsh.model.geo.addCircleArc(pb1, p0, pb2)
        lbu: int = gmsh.model.geo.addCircleArc(pb2, p0, pb1)
        lab: int = gmsh.model.geo.addCircleArc(pa1, p0, pa2)
        lau: int = gmsh.model.geo.addCircleArc(pa2, p0, pa1)

        # Surfaces
        # Bot
        clb: int = gmsh.model.geo.addCurveLoop([lbb, lbu])
        cla: int = gmsh.model.geo.addCurveLoop([lab, lau])
        s1: int = gmsh.model.geo.addPlaneSurface([clb, cla])

        # Boundaries
        self.boundaries = {
            "inn": cla,
            "out": clb,
        }
        self.boundary_lines = {
            "inn": [lab, lau],
            "out": [lbb, lbu],
        }
        # Physical groups
        # Domain
        domain: int = gmsh.model.addPhysicalGroup(2, [s1], tag=21)
        gmsh.model.setPhysicalName(2, domain, "domain")
        # Boundaries
        for boundary_name in self.boundaries:
            pg: int = gmsh.model.addPhysicalGroup(
                1,
                self.boundary_lines[boundary_name],
                tag=self.boundaries[boundary_name],
            )
            gmsh.model.setPhysicalName(1, pg, boundary_name)

        # Generate the mesh
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

        # # Display and exit for debug purposes
        # # Synchronize the model
        # gmsh.model.geo.synchronize()
        # # Display the GMSH window
        # gmsh.fltk.run()
        # exit()

        # Return the model
        return gmsh.model()

    def locate_measured_displacement(self) -> List[float]:
        """Define the point where the displacement is measured.

        Returns:
            List: Coordinate of the point where the displacement is measured
        """
        return [self.pars["b"], 0.0]

    def locate_measured_forces(self) -> int:
        """Define the boundary where the reaction force are measured.

        Returns:
            int: Identifier (id) of the boundary in GMSH.
        """
        return self.boundaries["out"]

    def define_controlled_displacements(self) -> List[DisplacementBC]:
        """Define the displacement boundary conditions controlled by the load factor.

        Returns:
            List[DisplacementBC]: List of DisplacementBC(boundary_id, u_imp) where boundary_id is the boundary id (int number) in GMSH, and u_imp is the displacement vector (componements can be nan to let it free).
        """
        # Return the boundary conditions
        return [
            DisplacementBC(self.boundaries["inn"], [0.0, 0.0]),
        ]

    def define_controlled_body_forces(self) -> List[BodyForce]:
        """Define the controlled body forces that are affected by the load factor.

        Returns:
            List[BodyForce]: List of BodyForce (f_imp) where f_imp is the force vector.
        """
        # Get the parameters
        rho = self.pars["rho"]
        w = self.pars["w"]
        wd = self.pars["wd"]
        # Compute the polar coordinates
        r = "sqrt(x[0]**2 + x[1]**2)"
        th = "atan2(x[1], x[0]) - pi/2"
        # Calculate the amplitude of each forces
        fw_amp = f"({rho} * {w}**2 * {r})"
        fe_amp = f"(- {rho} * {wd} * {r})"
        # Project onto the x and y axis
        fx = f"{fw_amp} * (-sin({th})) + {fe_amp} * (-cos({th}))"
        fy = f"{fw_amp} * cos({th}) + {fe_amp} * (-sin({th}))"
        return [BodyForce(f_imp=[fx, fy])]

    def Gc(self, phi):
        return self.pars["Gc"] * jnp.ones(phi.shape)


if __name__ == "__main__":
    pars = {
        # Geometry
        "a": 0.1,
        "b": 1.0,
        # Load
        "w": 0.0,
        "wd": 1.0,
        "rho": 1.0,
        # Fracture (not used here)
        "Gc": 1.0,
    }

    gcrack = GCrack(
        # Mechanics
        assumption_2D="plane_stress",
        E=1,
        nu=0.3,
        # Initial crack
        xc0=[0, (pars["a"] + pars["b"]) / 2, 0],
        # Propagation
        da=pars["b"] / 64,
        Nt=1,
        # Fracture
        sif_method="williams",
        # Other parameters
        pars=pars,
        name="numeric",
    )
    gcrack.run()
