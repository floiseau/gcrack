import sys

sys.path.append("/home/flavien.loiseau/sdrive/codes/gcrack/src/gcrack")

from typing import List, Tuple

import numpy as np

import gmsh

from gcrack import GCrackBaseData, gcrack
from boundary_conditions import DisplacementBC, ForceBC

SET = int(sys.argv[1])
print("\n========================")
print(f"  SIMULATION FOR SET {SET}  ")
print("========================")


class GCrackData(GCrackBaseData):
    def generate_mesh(self, crack_points: List[np.ndarray]) -> gmsh.model:
        # Clear existing model
        gmsh.clear()
        # Parameters
        R = self.pars["R"]
        S = R * 0.43
        h = R / 64
        h_min = self.R_int / 8
        czw = 1e-3  # contact zone width
        cz_angle = czw / R  # angle to obtain an contact zone of cwm
        # Points (left)
        pl1: int = gmsh.model.geo.addPoint(0, 0, 0, h)  # Left crack lip
        pl2: int = gmsh.model.geo.addPoint(-S + czw / 2, 0, 0, h)
        pl3: int = gmsh.model.geo.addPoint(-S - czw / 2, 0, 0, h)
        pl4: int = gmsh.model.geo.addPoint(-R, 0, 0, h)
        pl5: int = gmsh.model.geo.addPoint(
            R * np.sin(-cz_angle / 2), R * np.cos(-cz_angle / 2), 0, h
        )
        plr: int = gmsh.model.geo.addPoint(0, R, 0, h)  # shared point
        # Points (crack)
        pc_left: List[int] = []
        pc_right: List[int] = []
        for i, p in enumerate(reversed(crack_points)):
            # The crack tip is shared
            if i == 0:
                pc_new: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
                pc_left.append(pc_new)
                pc_right.append(pc_new)
            else:
                pc_new_left: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
                pc_left.append(pc_new_left)
                pc_new_right: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
                pc_right.append(pc_new_right)
        # Points (right)
        pr1: int = gmsh.model.geo.addPoint(0, 0, 0, h)  # Right crack lip
        pr2: int = gmsh.model.geo.addPoint(S - czw / 2, 0, 0, h)
        pr3: int = gmsh.model.geo.addPoint(S + czw / 2, 0, 0, h)
        pr4: int = gmsh.model.geo.addPoint(R, 0, 0, h)
        pr5: int = gmsh.model.geo.addPoint(
            R * np.sin(cz_angle / 2), R * np.cos(cz_angle / 2), 0, h
        )
        # plr (shared point)

        # Lines (left)
        ll1: int = gmsh.model.geo.addLine(pl1, pl2)
        ll2: int = gmsh.model.geo.addLine(pl2, pl3)
        ll3: int = gmsh.model.geo.addLine(pl3, pl4)
        ll4: int = gmsh.model.geo.addCircleArc(pl4, pl1, pl5)
        ll5: int = gmsh.model.geo.addCircleArc(pl5, pl1, plr)
        crack_lines_left: List[int] = []
        crack_lines_left.append(gmsh.model.geo.addLine(plr, pc_left[0]))
        for i in reversed(range(len(pc_left) - 1)):
            ll: int = gmsh.model.geo.addLine(pc_left[i], pc_left[i + 1])
            crack_lines_left.append(ll)
        ll6: int = gmsh.model.geo.addLine(pc_left[-1], pl1)
        # Lines (right)
        lr1: int = gmsh.model.geo.addLine(pr1, pr2)
        lr2: int = gmsh.model.geo.addLine(pr2, pr3)
        lr3: int = gmsh.model.geo.addLine(pr3, pr4)
        lr4: int = gmsh.model.geo.addCircleArc(pr4, pr1, pr5)
        lr5: int = gmsh.model.geo.addCircleArc(pr5, pr1, plr)
        crack_lines_right: List[int] = []
        # crack_lines_right.append(gmsh.model.geo.addLine(plr, pc_right[0]))
        crack_lines_right.append(crack_lines_left[0])
        for i in reversed(range(len(pc_right) - 1)):
            lr: int = gmsh.model.geo.addLine(pc_right[i], pc_right[i + 1])
            crack_lines_right.append(lr)
        lr6: int = gmsh.model.geo.addLine(pc_right[-1], pr1)

        # Surfaces (left)
        cl1: int = gmsh.model.geo.addCurveLoop(
            [ll1, ll2, ll3, ll4, ll5] + crack_lines_left + [ll6]
        )
        s1: int = gmsh.model.geo.addPlaneSurface([cl1])
        # Surfaces (right)
        cl2: int = gmsh.model.geo.addCurveLoop(
            [lr1, lr2, lr3, lr4, lr5] + crack_lines_right + [lr6]
        )
        # cl2: int = gmsh.model.geo.addCurveLoop(
        #     [-lr6]
        #     + [-l for l in reversed(crack_lines_right)]
        #     + [-lr5, -lr4, -lr3, -lr2, -lr1]
        # )
        s2: int = gmsh.model.geo.addPlaneSurface([cl2])

        # Define the boundaries
        self.boundaries = {
            "bot_left": 11,
            "bot_right": 12,
            "top": 13,
        }
        # Physical groups (domain)
        domain: int = gmsh.model.addPhysicalGroup(2, [s1, s2], tag=1)
        gmsh.model.setPhysicalName(2, domain, "domain")
        # Boundaries
        bot_left: int = gmsh.model.addPhysicalGroup(
            1, [ll2], tag=self.boundaries["bot_left"]
        )
        gmsh.model.setPhysicalName(1, bot_left, "bot_left")
        bot_right: int = gmsh.model.addPhysicalGroup(
            1, [lr2], tag=self.boundaries["bot_right"]
        )
        gmsh.model.setPhysicalName(1, bot_right, "bot_right")
        top: int = gmsh.model.addPhysicalGroup(
            1, [ll5, lr5], tag=self.boundaries["top"]
        )
        gmsh.model.setPhysicalName(1, top, "top")

        # Element size
        # Refine around the crack line
        field1: int = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field1, "PointsList", [pc_left[0]])
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

    def define_imposed_displacements(self) -> List[DisplacementBC]:
        """Define the imposed displacement boundary conditions.

        Returns:
            List[DisplacementBC]: List of DisplacementBC(boundary_id, u_imp) where boundary_id is the boundary id (int number) in GMSH, and u_imp is the displacement vector (componements can be nan to let it free).
        """
        match SET:
            case 1:
                return [
                    DisplacementBC(self.boundaries["bot_left"], [0, 0]),
                    DisplacementBC(self.boundaries["bot_right"], [float("nan"), 0]),
                    DisplacementBC(self.boundaries["top"], [float("nan"), -1.0]),
                ]
            case 2:
                return [
                    DisplacementBC(self.boundaries["bot_left"], [0, 0]),
                    DisplacementBC(self.boundaries["bot_right"], [float("nan"), 0]),
                    DisplacementBC(self.boundaries["top"], [0, -1.0]),
                ]
            case 3:
                return [
                    DisplacementBC(self.boundaries["bot_left"], [0, 0]),
                    DisplacementBC(self.boundaries["bot_right"], [0, 0]),
                    DisplacementBC(self.boundaries["top"], [float("nan"), -1.0]),
                ]
            case 4:
                return [
                    DisplacementBC(self.boundaries["bot_left"], [0, 0]),
                    DisplacementBC(self.boundaries["bot_right"], [float("nan"), 0]),
                ]
            case 5:
                return [
                    DisplacementBC(self.boundaries["bot_left"], [0, 0]),
                    DisplacementBC(self.boundaries["bot_right"], [0, 0]),
                ]

    def define_imposed_forces(self) -> List[ForceBC]:
        """Define the list of imposed forces.

        Returns:
            List[ForceBC]: List of ForceBC(boundary_id, f_imp) where boundary_id is the boundary id (int number) in GMSH, and f_imp is the force vector.
        """
        match SET:
            case 1 | 2 | 3:
                return []
            case 4 | 5:
                return [ForceBC(self.boundaries["top"], [0.0, -1.0])]

    def locate_measured_forces(self) -> int:
        """Define the boundary where the reaction force are measured.

        Returns:
            int: Identifier (id) of the boundary in GMSH.
        """
        return self.boundaries["top"]

    def locate_measured_displacement(self) -> List[float]:
        """Define the point where the displacement is measured.

        Returns:
            List: Coordinate of the point where the displacement is measured
        """
        return [0, self.pars["R"]]

    def Gc(self, phi):
        return self.pars["Gc"]


if __name__ == "__main__":
    # Define user parameters (to pass in user-defined function)
    pars = {
        "R": 50e-3,  # radius in m
        "beta": 30,  # crack angle in °
        "Gc": 380,
    }
    beta_rad = np.deg2rad(pars["beta"])
    angle_rad = np.pi / 2 - beta_rad
    xc0 = 0.3 * pars["R"] * np.array([np.cos(angle_rad), np.sin(angle_rad), 0])

    gcrack_data = GCrackData(
        E=3e9,
        nu=0.4,
        da=pars["R"] / 256,
        Nt=90 if SET != 3 else 80,
        xc0=xc0,
        assumption_2D="plane_stress",
        pars=pars,
        phi0=angle_rad,
    )
    gcrack(gcrack_data)
