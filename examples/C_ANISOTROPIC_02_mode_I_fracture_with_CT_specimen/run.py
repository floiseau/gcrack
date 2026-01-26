from typing import List

import jax.numpy as jnp

import gmsh
from gcrack import GCrackBase
from gcrack.boundary_conditions import NodalDisplacement, ForceBC


class GCrackData(GCrackBase):
    def generate_mesh(self, crack_points: List[jnp.ndarray]) -> gmsh.model:
        # Clear existing model
        gmsh.clear()

        # Enforce the pacman size
        self.R_int = 1 * pars["da"]
        self.R_ext = 4 * pars["da"]

        # Geometric parameters
        L = self.pars["L"]
        H = self.pars["H"]
        xh = pars["xh"]
        yh = pars["yh"]
        rh = pars["rh"]
        yp = pars["yp"]
        psi = jnp.deg2rad(pars["angular_region"])

        # # NOTE: Add a crack point to test crack implementation
        # crack_points.append([L / 5, H / 5, 0])

        # Numerical parameters
        h = L / 128
        h_min = self.R_int / 16

        # Store the crack tip to impose nodal displacements
        self.crack_tip = crack_points[-1]

        ### Points
        # Bot
        p11: int = gmsh.model.geo.addPoint(-L / 2, 0, 0, h)
        p12: int = gmsh.model.geo.addPoint(-L / 2, -H / 2, 0, h)
        p13: int = gmsh.model.geo.addPoint(L / 2, -H / 2, 0, h)
        p14: int = gmsh.model.geo.addPoint(L / 2, 0, 0, h)
        # Crack
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
        # Top
        p21: int = gmsh.model.geo.addPoint(-L / 2, 0, 0, h)
        p22: int = gmsh.model.geo.addPoint(-L / 2, H / 2, 0, h)
        p23: int = gmsh.model.geo.addPoint(L / 2, H / 2, 0, h)
        p24: int = p14
        # Bot hole
        p31: int = gmsh.model.geo.addPoint(
            -L / 2 + xh - rh * jnp.sin(psi / 2),
            -H / 2 + yh - rh * jnp.cos(psi / 2),
            0,
            h,
        )
        p32: int = gmsh.model.geo.addPoint(-L / 2 + xh, -H / 2 + yh, 0, h)
        p33: int = gmsh.model.geo.addPoint(
            -L / 2 + xh + rh * jnp.sin(psi / 2),
            -H / 2 + yh - rh * jnp.cos(psi / 2),
            0,
            h,
        )
        p34: int = gmsh.model.geo.addPoint(-L / 2 + xh, -H / 2 + yh + rh, 0, h)
        # Top hole
        p41: int = gmsh.model.geo.addPoint(
            -L / 2 + xh - rh * jnp.sin(psi / 2),
            H / 2 - yh + rh * jnp.cos(psi / 2),
            0,
            h,
        )
        p42: int = gmsh.model.geo.addPoint(-L / 2 + xh, H / 2 - yh, 0, h)
        p43: int = gmsh.model.geo.addPoint(
            -L / 2 + xh + rh * jnp.sin(psi / 2),
            H / 2 - yh + rh * jnp.cos(psi / 2),
            0,
            h,
        )
        p44: int = gmsh.model.geo.addPoint(-L / 2 + xh, H / 2 - yh - rh, 0, h)
        # Point for displacement probes
        p51: int = gmsh.model.geo.addPoint(-L / 2 + xh, -yp, 0, h)
        p52: int = gmsh.model.geo.addPoint(-L / 2 + xh, yp, 0, h)

        ### Lines
        # # Bot
        l11: int = gmsh.model.geo.addLine(p11, p12)
        l12: int = gmsh.model.geo.addLine(p12, p13)
        l13: int = gmsh.model.geo.addLine(p13, p14)
        l14: int = gmsh.model.geo.addLine(p14, pc_bot[0])
        crack_lines_bot: List[int] = []
        for i in range(len(pc_bot) - 1):
            lb: int = gmsh.model.geo.addLine(pc_bot[i], pc_bot[i + 1])
            crack_lines_bot.append(lb)
        l16: int = gmsh.model.geo.addLine(pc_bot[-1], p11)
        # Top
        l21: int = gmsh.model.geo.addLine(p21, p22)
        l22: int = gmsh.model.geo.addLine(p22, p23)
        l23: int = gmsh.model.geo.addLine(p23, p24)
        l24: int = l14
        crack_lines_top: List[int] = []
        for i in range(len(pc_top) - 1):
            lb: int = gmsh.model.geo.addLine(pc_top[i], pc_top[i + 1])
            crack_lines_top.append(lb)
        l26: int = gmsh.model.geo.addLine(pc_top[-1], p21)

        # Bot hole
        l31: int = gmsh.model.geo.addCircleArc(p31, p32, p33)
        l32: int = gmsh.model.geo.addCircleArc(p33, p32, p34)
        l33: int = gmsh.model.geo.addCircleArc(p34, p32, p31)
        # Top hole
        l41: int = gmsh.model.geo.addCircleArc(p41, p42, p43)
        l42: int = gmsh.model.geo.addCircleArc(p43, p42, p44)
        l43: int = gmsh.model.geo.addCircleArc(p44, p42, p41)

        # Make the boundary line transfinite to keep a constant number of element.
        gmsh.model.geo.mesh.setTransfiniteCurve(l31, 256)
        gmsh.model.geo.mesh.setTransfiniteCurve(l41, 256)

        ### Surfaces
        # Bot
        cl11: int = gmsh.model.geo.addCurveLoop(
            [l11, l12, l13, l14] + crack_lines_bot + [l16]
        )
        cl12: int = gmsh.model.geo.addCurveLoop([l31, l32, l33])
        s1: int = gmsh.model.geo.addPlaneSurface([cl11, cl12])
        # Top
        cl21: int = gmsh.model.geo.addCurveLoop(
            [l21, l22, l23, l24] + crack_lines_top + [l26]
        )
        cl22: int = gmsh.model.geo.addCurveLoop([l41, l42, l43])
        s2: int = gmsh.model.geo.addPlaneSurface([cl21, cl22])

        # Embed the probes in the surfaces
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(0, [p51], 2, s1)
        gmsh.model.mesh.embed(0, [p52], 2, s2)

        ### Boundaries
        self.boundaries = {
            "bot": l31,
            "top": l41,
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

        # Element size
        # Refine around the crack line
        field1: int = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field1, "PointsList", [pc_bot[0]])
        gmsh.model.mesh.field.setNumber(field1, "Sampling", 100)
        field2: int = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field2, "InField", field1)
        gmsh.model.mesh.field.setNumber(field2, "DistMin", 1 * self.R_ext)
        gmsh.model.mesh.field.setNumber(field2, "DistMax", 5 * self.R_ext)
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
        # Set one of the point of the top boundary where the displacement is imposed
        psi = pars["angular_region"]
        x = -pars["L"] / 2 + pars["xh"] + pars["rh"] * jnp.sin(psi / 2)
        y = pars["H"] / 2 - pars["yh"] + pars["rh"] * jnp.cos(psi / 2)
        return [x, y, 0]

    def locate_measured_forces(self) -> int:
        """Define the boundary where the reaction force are measured.

        Returns:
            int: Identifier (id) of the boundary in GMSH.
        """
        return self.boundaries["top"]

    def define_nodal_displacements(self) -> List[NodalDisplacement]:
        """Define a list of imposed nodal displacements.

        Returns:
            List[NodalDisplacements]: A list of NodalDisplacement.
        """
        L = self.pars["L"]
        return [
            NodalDisplacement(self.crack_tip, [0.0, 0.0, 0.0]),
            NodalDisplacement([L / 2, 0.0, 0.0], [float("nan"), 0.0, 0.0]),
        ]

    def define_controlled_forces(self) -> List[ForceBC]:
        """Define the list of imposed forces.

        Returns:
            List[ForceBC]: List of ForceBC(boundary_id, f_imp) where boundary_id is the boundary id (int number) in GMSH, and f_imp is the force vector.
        """
        return [
            ForceBC(self.boundaries["bot"], [0.0, -1.0]),
            ForceBC(self.boundaries["top"], [0.0, 1.0]),
        ]

    def Gc(self, phi):
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
    pars["L"] = 50e-3
    pars["H"] = 48e-3
    pars["xh"] = 10e-3
    pars["yh"] = 13e-3
    pars["rh"] = 10.4e-3 / 2
    # Material
    pars["E"] = 2e9
    pars["nu"] = 0.3
    pars["Gc_min"] = 4.5e3
    pars["Gc_max"] = 1e4
    pars["theta0"] = jnp.deg2rad(25)
    # Load
    pars["angular_region"] = 90
    # Probes
    pars["yp"] = pars["H"] / 100
    # Material
    pars["nu"] = 0.34
    # Numerical
    pars["da"] = pars["L"] / 256
    # Initialize the simulation
    gcrack_data = GCrackData(
        E=pars["E"],
        nu=pars["nu"],
        da=pars["da"],
        Nt=120,
        xc0=[0.0, 0.0, 0.0],
        assumption_2D="plane_stress",
        pars=pars,
        sif_method="williams",  # "i-integral" "willliams"
        s=pars["da"],
    )
    # Run the simulation
    gcrack_data.run()
