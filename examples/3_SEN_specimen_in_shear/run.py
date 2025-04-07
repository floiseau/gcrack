import sys

sys.path.append("/home/flavien.loiseau/sdrive/codes/gcrack/src/gcrack")

from typing import List, Tuple

import numpy as np

import gmsh

from gcrack import GCrackBaseData
from boundary_conditions import DisplacementBC, ForceBC


class GCrackData(GCrackBaseData):
    def generate_mesh(self, crack_points: List[np.ndarray]) -> gmsh.model:
        # Clear existing model
        gmsh.clear()
        # Parameters
        L = self.pars["L"]
        h = L / 64
        h_min = self.R_int / 16
        # Points
        # Bot
        p1: int = gmsh.model.geo.addPoint(0, 0, 0, h)
        p2: int = gmsh.model.geo.addPoint(L, 0, 0, h)
        p3: int = gmsh.model.geo.addPoint(L, L / 2, 0, h)  # Mid right node
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
        p5: int = gmsh.model.geo.addPoint(0, L / 2, 0, h)  # Bot crack lip
        # Top
        p6: int = gmsh.model.geo.addPoint(0, L, 0, h)
        p7: int = gmsh.model.geo.addPoint(L, L, 0, h)
        # Point(13) // Mid right node
        # Point(14) // Crack tip
        p8: int = gmsh.model.geo.addPoint(0, L / 2, 0, h)  # Top crack lip

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
            "bot": 11,
            "top": 12,
            "sides": 13,
        }
        # Physical groups
        # Domain
        domain: int = gmsh.model.addPhysicalGroup(2, [s1, s2], tag=21)
        gmsh.model.setPhysicalName(2, domain, "domain")
        # Boundaries
        bot: int = gmsh.model.addPhysicalGroup(1, [l1], tag=self.boundaries["bot"])
        gmsh.model.setPhysicalName(1, bot, "bot")
        top: int = gmsh.model.addPhysicalGroup(1, [l6], tag=self.boundaries["top"])
        gmsh.model.setPhysicalName(1, top, "top")
        sides: int = gmsh.model.addPhysicalGroup(
            1, [l2, l5, l7, l9], tag=self.boundaries["sides"]
        )
        gmsh.model.setPhysicalName(1, sides, "sides")

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

        # Return the model
        return gmsh.model()

    def locate_measured_displacement(self) -> List[float]:
        """Define the point where the displacement is measured.

        Returns:
            List: Coordinate of the point where the displacement is measured
        """
        return [self.pars["L"], 0]

    def locate_measured_forces(self) -> int:
        """Define the boundary where the reaction force are measured.

        Returns:
            int: Identifier (id) of the boundary in GMSH.
        """
        return self.boundaries["top"]

    def define_controlled_displacements(self) -> List[DisplacementBC]:
        """Define the imposed displacement boundary conditions.

        Returns:
            List[DisplacementBC]: List of DisplacementBC(boundary_id, u_imp) where boundary_id is the boundary id (int number) in GMSH, and u_imp is the displacement vector (componements can be nan to let it free).
        """
        return [
            DisplacementBC(boundary_id=self.boundaries["bot"], u_imp=[0, 0]),
            DisplacementBC(boundary_id=self.boundaries["top"], u_imp=[1, 0]),
            DisplacementBC(
                boundary_id=self.boundaries["sides"], u_imp=[float("nan"), 0]
            ),
        ]

    def Gc(self, phi):
        return self.pars["Gc"]


if __name__ == "__main__":
    # Define user parameters
    pars = {
        "L": 1e-3,
        "Gc": 2_700,
    }

    gcrack_data = GCrackData(
        E=230.77e9,
        nu=0.43,
        da=pars["L"] / 64,
        Nt=35,
        xc0=[pars["L"] / 2, pars["L"] / 2, 0],
        assumption_2D="plane_stress",
        pars=pars,
    )
    gcrack_data.gcrack()
