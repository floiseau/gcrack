import sys

sys.path.append("/home/flavien.loiseau/sdrive/codes/gcrack/src/gcrack")

from typing import List, Tuple

import numpy as np

import gmsh

from gcrack import GCrackBaseData, gcrack
from boundary_conditions import DisplacementBC, ForceBC


class GCrackData(GCrackBaseData):
    def generate_mesh(self, crack_points: List[np.ndarray]) -> gmsh.model:
        # Clear existing model
        gmsh.clear()
        # Parameters
        W = self.pars["W"]
        H = self.pars["H"]
        R = self.pars["R"]
        e = self.pars["e"]
        beta = np.deg2rad(self.pars["beta"])
        h = H / 64
        h_min = self.R_int / 8
        cz = 40e-3  # contact zone

        # Points (bot)
        pb1: int = gmsh.model.geo.addPoint(0, 0, 0, h)
        pb2: int = gmsh.model.geo.addPoint(W, 0, 0, h)
        pb3: int = gmsh.model.geo.addPoint(W, H / 2 - cz / 2, 0, h)
        pb4: int = gmsh.model.geo.addPoint(
            W / 2 + e + R * np.cos(np.pi + beta), H / 2 + R * np.sin(np.pi + beta), 0, h
        )
        pb5: int = gmsh.model.geo.addPoint(0, H / 2 - cz / 2, 0, h)
        # Points (top)
        pt1: int = gmsh.model.geo.addPoint(0, H, 0, h)
        pt2: int = gmsh.model.geo.addPoint(W, H, 0, h)
        pt3: int = gmsh.model.geo.addPoint(W, H / 2 + cz / 2, 0, h)
        pt4a: int = gmsh.model.geo.addPoint(W / 2 + e, H / 2 + R, 0, h)
        pt4b: int = gmsh.model.geo.addPoint(
            W / 2 + e + R * np.cos(np.pi + beta), H / 2 + R * np.sin(np.pi + beta), 0, h
        )
        pt5: int = gmsh.model.geo.addPoint(0, H / 2 + cz / 2, 0, h)
        # Points (shared)
        pbt1: int = gmsh.model.geo.addPoint(W, H / 2, 0, h)
        pbt2: int = gmsh.model.geo.addPoint(W / 2 + e + R, H / 2, 0, h)
        pbt3: int = gmsh.model.geo.addPoint(0, H / 2, 0, h)
        # Points (cirlce)
        pc: int = gmsh.model.geo.addPoint(W / 2 + e, H / 2, 0, h)
        # Points (crack)
        pbc: List[int] = []
        ptc: List[int] = []
        for i, p in enumerate(crack_points):
            # The crack tip is shared
            if i == len(crack_points) - 1:
                pc_new: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
                pbc.append(pc_new)
                ptc.append(pc_new)
            else:
                pc_new_bot: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
                pbc.append(pc_new_bot)
                pc_new_top: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
                ptc.append(pc_new_top)

        # Lines (bot)
        lb1: int = gmsh.model.geo.addLine(pb1, pb2)
        lb2: int = gmsh.model.geo.addLine(pb2, pb3)
        lb3: int = gmsh.model.geo.addLine(pb3, pbt1)
        lb4: int = gmsh.model.geo.addCircleArc(pbt2, pc, pb4)
        lbc_list: List[int] = []
        lbc_list.append(gmsh.model.geo.addLine(pb4, pbc[0]))
        for i in range(len(pbc) - 1):
            ll: int = gmsh.model.geo.addLine(pbc[i], pbc[i + 1])
            lbc_list.append(ll)
        lb5: int = gmsh.model.geo.addLine(pbt3, pb5)
        lb6: int = gmsh.model.geo.addLine(pb5, pb1)
        # Lines (top)
        lt1: int = gmsh.model.geo.addLine(pt1, pt2)
        lt2: int = gmsh.model.geo.addLine(pt2, pt3)
        lt3: int = gmsh.model.geo.addLine(pt3, pbt1)
        lt4a: int = gmsh.model.geo.addCircleArc(pbt2, pc, pt4a)
        lt4b: int = gmsh.model.geo.addCircleArc(pt4a, pc, pt4b)
        ltc_list: List[int] = []
        ltc_list.append(gmsh.model.geo.addLine(pt4b, ptc[0]))
        for i in range(len(ptc) - 1):
            ll: int = gmsh.model.geo.addLine(ptc[i], ptc[i + 1])
            ltc_list.append(ll)
        lt5: int = gmsh.model.geo.addLine(pbt3, pt5)
        lt6: int = gmsh.model.geo.addLine(pt5, pt1)

        # Lines shared
        lbt1: int = gmsh.model.geo.addLine(pbt1, pbt2)
        lbt2: int = gmsh.model.geo.addLine(pbc[-1], pbt3)

        # Surfaces (bot)
        clb: int = gmsh.model.geo.addCurveLoop(
            [lb1, lb2, lb3, lbt1, lb4] + lbc_list + [lbt2, lb5, lb6]
        )
        sb: int = gmsh.model.geo.addPlaneSurface([clb])
        # Surfaces (top)
        clt: int = gmsh.model.geo.addCurveLoop(
            [lt1, lt2, lt3, lbt1, lt4a, lt4b] + ltc_list + [lbt2, lt5, lt6]
        )
        st: int = gmsh.model.geo.addPlaneSurface([clt])

        # Define the boundaries
        self.boundaries = {
            "left": 11,
            "right": 12,
        }
        # Physical groups (domain)
        domain: int = gmsh.model.addPhysicalGroup(2, [sb, st], tag=1)
        gmsh.model.setPhysicalName(2, domain, "domain")
        # Boundaries
        left: int = gmsh.model.addPhysicalGroup(
            1, [lb5, lt5], tag=self.boundaries["left"]
        )
        gmsh.model.setPhysicalName(1, left, "left")
        right: int = gmsh.model.addPhysicalGroup(
            1, [lb3, lt3], tag=self.boundaries["right"]
        )
        gmsh.model.setPhysicalName(1, right, "right")

        # Element size
        # Refine around the crack line
        field1: int = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field1, "PointsList", [pbc[-1]])
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
        return [
            DisplacementBC(self.boundaries["left"], [0, 0]),
            DisplacementBC(self.boundaries["right"], [-1, 0]),
        ]
        # return [
        #     (self.boundaries["left"], [0, 0]),
        #     (self.boundaries["right"], [-1, -0.1]),
        # ]
        # return [(self.boundaries["left"], [0, 0]), (self.boundaries["right"], [-1, float("nan")])]
        # return [(self.boundaries["left"], [0, float("nan")]), (self.boundaries["right"], [-1, float("nan")])]
        # return [(self.boundaries["left"], [0, 0])]

    # def define_imposed_forces(self) -> List[ForceBC]:
    #     return [(self.boundaries["right"], [-1, 0])]

    # def define_locked_points(self) -> List[List[float]]:
    #     return [
    #         [0, self.pars["H"]/2],
    #     ]

    def locate_measured_forces(self) -> int:
        """Define the boundary where the reaction force are measured.

        Returns:
            int: Identifier (id) of the boundary in GMSH.
        """
        return self.boundaries["right"]

    def locate_measured_displacement(self) -> List[float]:
        """Define the point where the displacement is measured.

        Returns:
            List: Coordinate of the point where the displacement is measured
        """
        return [self.pars["W"], self.pars["H"] / 2]

    def Gc(self, phi):
        return self.pars["Gc"]


if __name__ == "__main__":
    # Define user parameters (to pass in user-defined function)
    pars = {
        "Gc": 430,
        "W": 140e-3,
        "H": 70e-3,
        "R": 15e-3,
        "e": 25e-3,  # eccentricity of the circle (along x)
        "a0": 20e-3,  # 35e-3 - 15e-3                       ,
        "beta": 30,
    }
    W = pars["W"]
    H = pars["H"]
    R = pars["R"]
    e = pars["e"]
    a0 = pars["a0"]
    angle = np.pi + np.deg2rad(pars["beta"])
    xc0 = np.array(
        [W / 2 + e + (R + a0) * np.cos(angle), H / 2 + (R + a0) * np.sin(angle), 0]
    )

    gcrack_data = GCrackData(
        E=5e9,
        nu=0.32,
        da=pars["W"] / 512,
        Nt=20,
        xc0=xc0,
        assumption_2D="plane_strain",
        pars=pars,
        phi0=angle,
    )
    gcrack(gcrack_data)
