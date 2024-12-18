import sys

sys.path.append("/home/flavien.loiseau/sdrive/codes/gcrack/src/gcrack")

from typing import List, Tuple

import numpy as np
import sympy as sp

import gmsh

from gcrack import GCrackBaseData, gcrack


class GCrackData(GCrackBaseData):
    def generate_mesh(self, crack_points: List[np.ndarray]) -> gmsh.model:
        # Clear existing model
        gmsh.clear()
        # Parameters
        Lx = self.pars["Lx"]
        Ly = self.pars["Ly"]
        h = Lx / 128
        h_min = self.R_int / 16
        # Points
        # Bot
        p1: int = gmsh.model.geo.addPoint(0, 0, 0, h)
        p2: int = gmsh.model.geo.addPoint(Lx, 0, 0, h)
        p3: int = gmsh.model.geo.addPoint(Lx, Ly / 2, 0, h)  # Mid right node
        pc_bot: List[int] = []
        pc_top: List[int] = []
        for i, p in enumerate(reversed(crack_points)):
            # The crack tip is shared
            if i == 0:
                pc_new: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
                pc_bot.append(pc_new)
                pc_top.append(pc_new)
            else:
                pc_new_bot: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
                pc_bot.append(pc_new_bot)
                pc_new_top: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
                pc_top.append(pc_new_top)
        p5: int = gmsh.model.geo.addPoint(0, Ly / 2, 0, h)  # Bot crack lip
        # Top
        p6: int = gmsh.model.geo.addPoint(0, Ly, 0, h)
        p7: int = gmsh.model.geo.addPoint(Lx, Ly, 0, h)
        # Point(13) // Mid right node
        # Point(14) // Crack tip
        p8: int = gmsh.model.geo.addPoint(0, Ly / 2, 0, h)  # Top crack lip

        # Lines
        # Bot
        l1: int = gmsh.model.geo.addLine(p1, p2)
        l2: int = gmsh.model.geo.addLine(p2, p3)
        l3: int = gmsh.model.geo.addLine(p3, pc_bot[0])
        crack_lines_bot: List[int] = []
        for i in range(len(pc_bot) - 1):
            lb: int = gmsh.model.geo.addLine(pc_bot[i], pc_bot[i + 1])
            crack_lines_bot.append(lb)
        crack_lines_bot.append(gmsh.model.geo.addLine(pc_bot[-1], p5))
        l5: int = gmsh.model.geo.addLine(p5, p1)
        # Top
        l6: int = gmsh.model.geo.addLine(p6, p7)
        l7: int = gmsh.model.geo.addLine(p7, p3)
        # Line(13)
        # Top  crack line
        crack_lines_top: List[int] = []
        for i in range(len(pc_bot) - 1):
            lt: int = gmsh.model.geo.addLine(pc_top[i], pc_top[i + 1])
            crack_lines_top.append(lt)
        crack_lines_top.append(gmsh.model.geo.addLine(pc_top[-1], p8))
        l9: int = gmsh.model.geo.addLine(p8, p6)

        # Surfaces
        # Bot
        cl1: int = gmsh.model.geo.addCurveLoop([l1, l2, l3] + crack_lines_bot + [l5])
        s1: int = gmsh.model.geo.addPlaneSurface([cl1])
        # Top
        cl2: int = gmsh.model.geo.addCurveLoop([l6, l7, l3] + crack_lines_top + [l9])
        s2: int = gmsh.model.geo.addPlaneSurface([cl2])

        # Boundaries
        self.boundaries = {
            "left_bot": 13,
            "left_top": 14,
        }
        self.boundary_lines = {
            "left_bot": [l5],
            "left_top": [l9],
        }
        # Physical groups
        # Domain
        domain: int = gmsh.model.addPhysicalGroup(2, [s1, s2], tag=21)
        gmsh.model.setPhysicalName(2, domain, "domain")
        # Boundaries
        for boundary_name in self.boundaries:
            pg: int = gmsh.model.addPhysicalGroup(
                1,
                self.boundary_lines[boundary_name],
                tag=self.boundaries[boundary_name],
            )
            gmsh.model.setPhysicalName(1, pg, boundary_name)

        # Element size
        # Refine around the crack line
        field1: int = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field1, "PointsList", [pc_bot[0]])
        gmsh.model.mesh.field.setNumber(field1, "Sampling", 100)
        field2: int = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field2, "InField", field1)
        gmsh.model.mesh.field.setNumber(field2, "DistMin", self.R_ext)
        gmsh.model.mesh.field.setNumber(field2, "DistMax", 2 * self.R_ext)
        gmsh.model.mesh.field.setNumber(field2, "SizeMin", h_min)
        gmsh.model.mesh.field.setNumber(field2, "SizeMax", h)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.field.setAsBackgroundMesh(field2)
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
        return [self.pars["Ly"], 0]

    def locate_measured_forces(self) -> int:
        """Define the boundary where the reaction force are measured.

        Returns:
            int: Identifier (id) of the boundary in GMSH.
        """
        return self.boundaries["left_top"]

    def define_imposed_displacements(self) -> List[DisplacementBC]:
        """Define the imposed displacement boundary conditions.

        Returns:
            List: with (id, value) where id is the boundary id (int number) in GMSH, and value is the displacement vector (componements can be nan to let it free).
        """
        return [
            (self.boundaries["left_bot"], [0.0, -1.0]),
            (self.boundaries["left_top"], [0.0, 1.0]),
        ]

    def define_locked_points(self) -> List[List[float]]:
        """Define the list of locked points.

        Returns:
            List[List[float]]: A list of points (list) coordinates.
        """
        return []

    def Gc(self, phi):
        # # Get the parameters
        # Gc_min = 4225
        # return Gc_min * (1000 + 999 * sp.cos(4 * phi)) ** (1 / 4)
        Gc = 1.0
        D2 = 0
        P2 = 0
        D4 = 0.8  # 0.99
        P4 = np.pi / 6  # np.pi / 4
        # Define expression of the energy release rate
        Gc_expression = Gc * (
            1 + D2 * sp.cos(2 * (phi - P2)) + D4 * sp.cos(4 * (phi - P4))
        ) ** (1 / 4)
        return Gc_expression
        # In plotter: 1 + (2 - 1) * sqrt(1 / 2 * (1 - cos(2 * (phi - pi/6))))


if __name__ == "__main__":
    # Define user parameters
    pars = {
        "Lx": 1.0,
        "Ly": 1.0,
    }

    gcrack_data = GCrackData(
        E=1.0,
        nu=0.3,
        da=pars["Lx"] / 256,  # 128,  # 0.6e-3,
        Nt=120,  # 60,
        xc0=[pars["Lx"] / 2, pars["Ly"] / 2, 0],
        assumption_2D="plane_stress",
        pars=pars,
        sif_method="i-integral",  # "i-integral" "willliams"
        s=0.04,  # 0.6e-3,
    )
    gcrack(gcrack_data)
