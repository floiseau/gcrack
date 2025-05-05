from typing import List
import itertools

import gmsh
import numpy as np
import dolfinx
import ufl

from gcrack import GCrackBase
from gcrack.domain import Domain
from gcrack.models import ElasticModel
from gcrack.boundary_conditions import (
    DisplacementBC,
    ForceBC,
    BoundaryConditions,
)
from gcrack.solvers import solve_elastic_problem
from gcrack.sif import compute_SIFs
from gcrack.lefm import G_star


class GCrackData(GCrackBase):
    def generate_mesh(self, crack_points: List[np.ndarray]) -> gmsh.model:
        # Clear existing model
        gmsh.clear()
        # Parameters
        L = 1
        h = L / 256
        h_min = self.R_int / 32
        # Points
        # Bot
        p1: int = gmsh.model.geo.addPoint(-L / 2, -L / 2, 0, h)
        p2: int = gmsh.model.geo.addPoint(L / 2, -L / 2, 0, h)
        p3: int = gmsh.model.geo.addPoint(L / 2, 0, 0, h)  # Mid right node
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
        p5: int = gmsh.model.geo.addPoint(-L / 2, 0, 0, h)  # Bot crack lip
        # Top
        p6: int = gmsh.model.geo.addPoint(-L / 2, L / 2, 0, h)
        p7: int = gmsh.model.geo.addPoint(L / 2, L / 2, 0, h)
        # Point(13) // Mid right node
        # Point(14) // Crack tip
        p8: int = gmsh.model.geo.addPoint(-L / 2, 0, 0, h)  # Top crack lip

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
        """Define the imposed displacement boundary conditions.

        Returns:
            List[DisplacementBC]: List of DisplacementBC(boundary_id, u_imp) where boundary_id is the boundary id (int number) in GMSH, and u_imp is the displacement vector (componements can be nan to let it free).
        """
        return [
            DisplacementBC(boundary_id=self.boundaries["bot"], u_imp=[float("nan"), 0]),
            DisplacementBC(boundary_id=self.boundaries["top"], u_imp=[float("nan"), 1]),
        ]

    def define_locked_points(self) -> List[List[float]]:
        """Define the list of locked points.

        Returns:
            List[List[float]]: A list of points (list) coordinates.
        """
        return [
            [-self.pars["L"] / 2, -self.pars["L"] / 2, 0],
        ]

    def locate_measured_displacement(self):
        return []

    def locate_measured_forces(self):
        return []

    def Gc(self, phi):
        return 0.0


def run_SIF_computation(data: GCrackData, phi1: float = None):
    # Crack points
    crack_points = [data.xc0]
    # Increment the crack if necessary
    if phi1 is not None:
        da = data.da
        # Add the new crack point
        xc1 = data.xc0 + da * np.array([np.cos(phi1), np.sin(phi1), 0])
        crack_points.append(xc1)
        # Update the initial crack angle
        data.phi0 = phi1

    # Generate the new mesh
    gmsh_model = data.generate_mesh(crack_points)
    domain = Domain(gmsh_model)
    # Define an elastic model
    ela_pars = {
        "E": data.E,
        "nu": data.nu,
        "2D_assumption": data.assumption_2D,
    }
    model = ElasticModel(ela_pars, domain)
    #  Run the elastic simulation
    bcs = BoundaryConditions(
        displacement_bcs=data.define_controlled_displacements(),
        force_bcs=data.define_controlled_forces(),
        locked_points=data.define_locked_points(),
    )
    # Solve the elastic problem
    u = solve_elastic_problem(domain, model, bcs)
    # Compute the SIFs
    return model, compute_SIFs(
        domain,
        model,
        u,
        crack_points[-1],
        data.phi0,
        data.R_int,
        data.R_ext,
        data.sif_method,
    )


def verify_kinked_crack_SIFs():
    # Define user parameters
    pars = {"L": 1.0}
    data = GCrackData(
        E=1.0,
        nu=0.0,
        da=pars["L"] / 256,
        Nt=1,
        xc0=np.array([0, 0, 0]),
        assumption_2D="plane_strain",
        pars=pars,
        # sif_method="williams",
        sif_method="i-integral",
        s=0.0,
    )
    # Intialize GMSH
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Disable terminal output
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    # Generate the list of angle
    N = 4
    phi1s = [i / N * np.pi / 2 for i in range(0, N + 1)]

    # Run SIFs computation without kinked increment
    print("Initial")
    model_init, SIF_init = run_SIF_computation(data)
    # Compute G_star for each values of phi1
    gs_AM = [
        float(
            G_star(
                phi1,
                data.phi0,
                SIF_init["KI"],
                SIF_init["KII"],
                SIF_init["T"],
                model_init.Ep,
                data.s,
            )
        )
        for phi1 in phi1s
    ]

    # Initialize G_star for
    gs_FEM = []
    for phi1 in phi1s:
        print(f"Kinked phi1={np.rad2deg(phi1)}°")
        # Run SIFs computation with kinked increment (angle phi1)
        model_kink, SIF_kinked = run_SIF_computation(data, phi1)
        # Compute the energy release rate
        gs_FEM.append(
            float(1 / model_kink.Ep * (SIF_kinked["KI"] ** 2 + SIF_kinked["KII"] ** 2))
        )
    print(f"{gs_AM=}")
    print(f"{gs_FEM=}")

    gmsh.finalize()


if __name__ == "__main__":
    verify_kinked_crack_SIFs()
