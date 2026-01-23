from typing import List

import numpy as np
import jax.numpy as jnp

import gmsh
from gcrack import GCrackBase
from gcrack.boundary_conditions import DisplacementBC, ForceBC


class GCrackData(GCrackBase):
    def generate_mesh(self, crack_points: List[np.ndarray]) -> gmsh.model:
        # Clear existing model
        gmsh.clear()
        # Parameters
        L = self.pars["L"]
        h = L / 128
        h_min = self.R_int / 64
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
            "bot": l5,
            "top": l9,
            "all": 101,
        }
        # Physical groups
        # Domain
        domain: int = gmsh.model.addPhysicalGroup(2, [s1, s2], tag=21)
        gmsh.model.setPhysicalName(2, domain, "domain")
        # Boundaries
        bot: int = gmsh.model.addPhysicalGroup(
            1, [self.boundaries["bot"]], tag=self.boundaries["bot"]
        )
        gmsh.model.setPhysicalName(1, bot, "bot")
        top: int = gmsh.model.addPhysicalGroup(
            1, [self.boundaries["top"]], tag=self.boundaries["top"]
        )
        gmsh.model.setPhysicalName(1, top, "top")
        all: int = gmsh.model.addPhysicalGroup(
            1, [l1, l2, l5, l6, l7, l9], tag=self.boundaries["all"]
        )
        gmsh.model.setPhysicalName(1, all, "all")

        # # Display and exit for debug purposes
        # # Synchronize the model
        # gmsh.model.geo.synchronize()
        # # Display the GMSH window
        # gmsh.fltk.run()
        # exit()

        # Element size
        # Refine around the crack line
        field1: int = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field1, "PointsList", [pc_bot[0]])
        gmsh.model.mesh.field.setNumber(field1, "Sampling", 100)
        field2: int = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field2, "InField", field1)
        gmsh.model.mesh.field.setNumber(field2, "DistMin", 2 * self.R_ext)
        gmsh.model.mesh.field.setNumber(field2, "DistMax", 4 * self.R_ext)
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
        return [0, self.pars["L"]]

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
        # Parameter
        KIII = 1.0
        # Material parameters
        mu = self.E / (2 * (1 + self.nu))
        # Center coordinates around crack tip
        x0t = self.pars["L"] / 2
        y0t = self.pars["L"] / 2
        x = f"(x[0] - {x0t})"
        y = f"(x[1] - {y0t})"
        # Radial coordintes
        r = f"sqrt({x}**2 + {y}**2)"
        theta = f"atan2({y},{x})"
        # Displacement field
        u_imp = f"2*{KIII}/({mu}) * sqrt({r}/(2*pi)) * sin({theta}/2)"
        return [DisplacementBC(boundary_id=self.boundaries["all"], u_imp=[u_imp])]

    # def define_locked_points(self) -> List[List[float]]:
    #     """Define the list of locked points.

    #     Returns:
    #         List[List[float]]: A list of points (list) coordinates.
    #     """
    #     return [
    #         [0, 0, 0],
    #     ]

    def Gc(self, phi):
        return 1 + 0 * phi
        # In plotter: 1 + (2 - 1) * sqrt(1 / 2 * (1 - cos(2 * (phi - pi/6))))


if __name__ == "__main__":
    # Define user parameters
    pars = {"L": 1.0}

    gcrack_data = GCrackData(
        E=1.0,
        nu=0.3,
        da=pars["L"] / 64,
        Nt=1,
        xc0=[pars["L"] / 2, pars["L"] / 2, 0],
        assumption_2D="anti_plane",
        pars=pars,
        sif_method="williams",  # "i-integral" "willliams"
        no_propagation=True,
    )
    gcrack_data.run()
