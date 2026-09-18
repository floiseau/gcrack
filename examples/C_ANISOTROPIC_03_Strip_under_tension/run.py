from typing import List

import numpy as np
import jax.numpy as jnp

import gmsh
from gcrack import GCrackBase
from gcrack.boundary_conditions import DisplacementBC


class GCrackData(GCrackBase):
    def generate_mesh(self, crack_points: List[np.ndarray]) -> gmsh.model:
        # Clear existing model
        gmsh.clear()
        # Extract user-defined parameters
        L = self.pars["L"]
        H = self.pars["H"]
        h = pars["h"]
        h_min = pars["h_min"]
        self.R_int = pars["R_int"]
        self.R_ext = 2 * self.R_int
        # Points
        # Bot
        p1: int = gmsh.model.geo.addPoint(0, -H / 2, 0, h)
        p2: int = gmsh.model.geo.addPoint(L, -H / 2, 0, h)
        p3: int = gmsh.model.geo.addPoint(L, 0, 0, h)  # Mid right node
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
        p5: int = gmsh.model.geo.addPoint(0, 0, 0, h)  # Bot crack lip
        # Top
        p6: int = gmsh.model.geo.addPoint(0, H / 2, 0, h)
        p7: int = gmsh.model.geo.addPoint(L, H / 2, 0, h)
        # Point(13) // Mid right node
        # Point(14) // Crack tip
        p8: int = gmsh.model.geo.addPoint(0, 0, 0, h)  # Top crack lip

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

        # Element size
        # Refine around the crack line
        field1: int = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field1, "PointsList", [pc_bot[0]])
        gmsh.model.mesh.field.setNumber(field1, "Sampling", 100)
        field2: int = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field2, "InField", field1)
        gmsh.model.mesh.field.setNumber(field2, "DistMin", 1 * self.R_ext)
        gmsh.model.mesh.field.setNumber(field2, "DistMax", 4 * self.R_ext)
        gmsh.model.mesh.field.setNumber(field2, "SizeMin", h_min)
        gmsh.model.mesh.field.setNumber(field2, "SizeMax", h)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.field.setAsBackgroundMesh(field2)
        gmsh.model.mesh.generate(2)

        # # NOTE: Uncomment and move this block to display the mesh in GMSH
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
        return [0, self.pars["H"] / 2]

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
            DisplacementBC(
                boundary_id=self.boundaries["bot"],
                u_imp=[0, -1],
            ),
            DisplacementBC(
                boundary_id=self.boundaries["top"],
                u_imp=[0, 1],
            ),
        ]

    def end_simulation(self, crack_points: List[List[float]]) -> bool:
        """User-defined function to end the simulation when a condition is met.

        Args:
            crack_points (List[List[float]]): List of the crack points.

        Returns:
            bool: True if the simulation must be ended, else False.
        """
        # Get the crack tip
        xt = crack_points[-1][0]
        yt = crack_points[-1][1]
        # Get geometric parameters parameters
        L = self.pars["L"]
        H = self.pars["H"]
        # Set the condition
        return L < xt or yt < -H / 2 or H / 2 < yt

    def Gc(self, phi, xc):
        # Get the parameters
        Gc_min = self.pars["Gc_min"]
        Gc_max = self.pars["Gc_max"]
        theta0 = self.pars["theta0"]
        # Define expression of the energy release rate
        return Gc_min + (Gc_max - Gc_min) * jnp.abs(jnp.sin(2 * (phi - theta0)))


if __name__ == "__main__":
    # Define user parameters
    pars = {}

    # Geometry
    pars["H"] = 1  # Height
    pars["L"] = 6  # Length
    pars["a0"] = 1  # Initial crack length

    # Numeric
    pars["da"] = pars["L"] / 128  # Crack increment
    pars["R_int"] = pars["da"] / 4  # Size of pacman for SIF determination
    pars["h_min"] = pars["R_int"] / 8  # Mesh size around crack tip
    pars["h"] = pars["H"] / 64  # Mesh size in the bulk

    # Fracture properties
    pars["Gc_min"] = 1  # Gc in the weak direction
    pars["Gc_max"] = 2  # Gc in the strong direction
    pars["theta0"] = np.deg2rad(10)  # Anisotropy angle (angle of weak direction)

    gcrack_data = GCrackData(
        E=1,  # Young modulus
        nu=0.3,  # Poisson ratio
        da=pars["da"],  # Crack increment
        Nt=100,  # Number of load step
        # WARNING: gcrack "crashes" if the crack goes outside before Nt load steps, but the results are still preserved.
        xc0=[pars["a0"], 0, 0],  # Initial crack tip location
        assumption_2D="plane_strain",  # 2D assumption ("plane_strain"/"plane_stress")
        pars=pars,  # User-defined parameters
        sif_method="williams",  # Method to calculate SIFs ("i-integral"/"willliams")
        s=pars["da"],  # Length associated with T-stress (Amestoy-Leblond, 1992)
    )
    gcrack_data.run()
