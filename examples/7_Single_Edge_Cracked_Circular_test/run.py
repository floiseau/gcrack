import sys

sys.path.append("/home/flavien.loiseau/sdrive/codes/gcrack/src/gcrack")

from typing import List

import numpy as np

import gmsh

from gcrack import GCrackBaseData, gcrack
from boundary_conditions import DisplacementBC, ForceBC

ALPHA: int = int(sys.argv[1])


class GCrackData(GCrackBaseData):
    def generate_mesh(self, crack_points: List[np.ndarray]) -> gmsh.model:
        # Clear existing model
        gmsh.clear()
        # Parameters
        Di = self.pars["Di"]  # Diameter of the specimen
        Do = self.pars["Do"]  # Location diameter of the hole centers
        d = self.pars["d"]  # Hole diameter
        nw = self.pars["nw"]  # Notch width
        a0 = self.pars["a0"]  # Pre-crack length
        alpha = self.pars["alpha"]  # Loading angle
        # Calculated parameters
        Ri = Di / 2
        Ro = Do / 2
        theta = np.arctan2(nw, Di)
        # Numerical parameters
        h = Di / 64
        h_min = self.R_int / 8

        # Points (centers)
        pc: int = gmsh.model.geo.addPoint(0, 0, 0, h)
        # Points (bot)
        pb1: int = gmsh.model.geo.addPoint(
            Ri * np.cos(np.pi + theta), Ri * np.sin(np.pi + theta), 0, h
        )
        pb2: int = gmsh.model.geo.addPoint(-a0, 0, 0, h)
        # Points (bot)
        pt1: int = gmsh.model.geo.addPoint(
            Ri * np.cos(np.pi - theta), Ri * np.sin(np.pi - theta), 0, h
        )
        pt2: int = gmsh.model.geo.addPoint(-a0, 0, 0, h)
        # Points (shared)
        pbt1: int = gmsh.model.geo.addPoint(Ri, 0, 0, h)
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
        # Points (hole A)
        angle = np.pi / 2 - alpha
        xac = Ro * np.cos(np.pi - angle)
        yac = Ro * np.sin(np.pi - angle)
        pha1: int = gmsh.model.geo.addPoint(
            xac + d / 2 * np.cos(np.pi - angle + np.pi / 2),
            yac + d / 2 * np.sin(np.pi - angle + np.pi / 2),
            0,
            h,
        )
        pha2: int = gmsh.model.geo.addPoint(xac, yac, 0, h)
        pha3: int = gmsh.model.geo.addPoint(
            xac + d / 2 * np.cos(np.pi - angle - np.pi / 2),
            yac + d / 2 * np.sin(np.pi - angle - np.pi / 2),
            0,
            h,
        )
        # Points (hole B)
        xbc = Ro * np.cos(-angle)
        ybc = Ro * np.sin(-angle)
        phb1: int = gmsh.model.geo.addPoint(
            xbc + d / 2 * np.cos(-angle + np.pi / 2),
            ybc + d / 2 * np.sin(-angle + np.pi / 2),
            0,
            h,
        )
        phb2: int = gmsh.model.geo.addPoint(xbc, ybc, 0, h)
        phb3: int = gmsh.model.geo.addPoint(
            xbc + d / 2 * np.cos(-angle - np.pi / 2),
            ybc + d / 2 * np.sin(-angle - np.pi / 2),
            0,
            h,
        )

        # Points (hole C)
        xcc = Ro * np.cos(np.pi + angle)
        ycc = Ro * np.sin(np.pi + angle)
        phc1: int = gmsh.model.geo.addPoint(
            xcc + d / 2 * np.cos(np.pi + angle + np.pi / 2),
            ycc + d / 2 * np.sin(np.pi + angle + np.pi / 2),
            0,
            h,
        )
        phc2: int = gmsh.model.geo.addPoint(xcc, ycc, 0, h)
        phc3: int = gmsh.model.geo.addPoint(
            xcc + d / 2 * np.cos(np.pi + angle - np.pi / 2),
            ycc + d / 2 * np.sin(np.pi + angle - np.pi / 2),
            0,
            h,
        )

        # Lines (bot)
        lb1: int = gmsh.model.geo.addLine(pb1, pb2)
        lbc_list: List[int] = []
        lbc_list.append(gmsh.model.geo.addLine(pb2, pbc[0]))
        for i in range(len(pbc) - 1):
            ll: int = gmsh.model.geo.addLine(pbc[i], pbc[i + 1])
            lbc_list.append(ll)
        lb2: int = gmsh.model.geo.addCircleArc(pbt1, pc, pb1)
        # Lines (top)
        lt1: int = gmsh.model.geo.addLine(pt1, pt2)
        ltc_list: List[int] = []
        ltc_list.append(gmsh.model.geo.addLine(pt2, ptc[0]))
        for i in range(len(ptc) - 1):
            ll: int = gmsh.model.geo.addLine(ptc[i], ptc[i + 1])
            ltc_list.append(ll)
        lt2: int = gmsh.model.geo.addCircleArc(pbt1, pc, pt1)
        # Line (shared)
        lbt1: int = gmsh.model.geo.addLine(pbc[-1], pbt1)
        # Lines (holes)
        lha1: int = gmsh.model.geo.addCircleArc(pha1, pha2, pha3)
        lha2: int = gmsh.model.geo.addCircleArc(pha3, pha2, pha1)
        lhb1: int = gmsh.model.geo.addCircleArc(phb1, phb2, phb3)
        lhb2: int = gmsh.model.geo.addCircleArc(phb3, phb2, phb1)
        lhc1: int = gmsh.model.geo.addCircleArc(phc1, phc2, phc3)
        lhc2: int = gmsh.model.geo.addCircleArc(phc3, phc2, phc1)

        # Surfaces (bot)
        clb: int = gmsh.model.geo.addCurveLoop([lb1] + lbc_list + [lbt1, lb2])
        clhb: int = gmsh.model.geo.addCurveLoop([lhb1, lhb2])
        clhc: int = gmsh.model.geo.addCurveLoop([lhc1, lhc2])
        sb: int = gmsh.model.geo.addPlaneSurface([clb, clhb, clhc])
        # Surfaces (top)
        clt: int = gmsh.model.geo.addCurveLoop([lt1] + ltc_list + [lbt1, lt2])
        clha: int = gmsh.model.geo.addCurveLoop([lha1, lha2])
        st: int = gmsh.model.geo.addPlaneSurface([clt, clha])

        # Define the boundaries
        self.boundaries = {
            "top": 11,
            "bot": 12,
        }
        # Physical groups (domain)
        domain: int = gmsh.model.addPhysicalGroup(2, [sb, st], tag=1)
        gmsh.model.setPhysicalName(2, domain, "domain")
        # Boundaries
        bot: int = gmsh.model.addPhysicalGroup(1, [lhb2], tag=self.boundaries["bot"])
        gmsh.model.setPhysicalName(1, bot, "bot")
        top: int = gmsh.model.addPhysicalGroup(1, [lha2], tag=self.boundaries["top"])
        gmsh.model.setPhysicalName(1, top, "top")

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

    def define_controlled_displacements(self) -> List[DisplacementBC]:
        return [
            DisplacementBC(self.boundaries["bot"], [0, 0]),
        ]

    def define_controlled_forces(self) -> List[ForceBC]:
        alpha = self.pars["alpha"]
        angle = np.pi / 2 - alpha
        return [
            ForceBC(
                self.boundaries["top"], [np.cos(np.pi - angle), np.sin(np.pi - angle)]
            )
        ]

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
        Ro = self.pars["Do"] / 2
        r = self.pars["d"] / 2
        alpha = self.pars["alpha"]
        angle = np.pi / 2 - alpha
        x = (Ro + r) * np.cos(np.pi - angle)
        y = (Ro + r) * np.sin(np.pi - angle)
        return [x, y]

    def Gc(self, phi):
        return self.pars["Gc"]


if __name__ == "__main__":
    # Define user parameters (to pass in user-defined function)
    pars = {
        "Gc": 614.79,
        "Di": 120e-3,  # Diameter of the specimen
        "Do": 78e-3,  # Location diameter of the hole centers
        "d": 12.8e-3,  # Hole diameter
        "nw": 3e-3,  # Notch width
        "a0": 2e-3,  # Pre-crack length
        "alpha": np.deg2rad(ALPHA),  # Loading angle
    }

    gcrack_data = GCrackData(
        E=2.4476e9,
        nu=0.3,
        da=pars["Di"] / 256,
        Nt=120,
        xc0=np.array([0.0, 0.0, 0.0]),
        assumption_2D="plane_stress",
        pars=pars,
        phi0=0,
    )
    gcrack(gcrack_data)
