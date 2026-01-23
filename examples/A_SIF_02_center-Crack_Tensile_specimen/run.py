from typing import List
from pathlib import Path

import gmsh
import numpy as np

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
        # Parameters (geometry)
        L = self.pars["L"]
        W = self.pars["W"]
        # Parameters (discretization)
        h = L / 128
        h_min = self.R_int / 32
        # Points
        # Bot
        p1: int = gmsh.model.geo.addPoint(-W / 2, -L / 2, 0, h)
        p2: int = gmsh.model.geo.addPoint(W / 2, -L / 2, 0, h)
        # Mid
        p3: int = gmsh.model.geo.addPoint(-W / 2, 0, 0, h)  # Mid right node
        p4: int = gmsh.model.geo.addPoint(W / 2, 0, 0, h)  # Mid left node
        # Top
        p5: int = gmsh.model.geo.addPoint(-W / 2, L / 2, 0, h)
        p6: int = gmsh.model.geo.addPoint(W / 2, L / 2, 0, h)
        # Cracks
        pc_bot = []
        pc_top = []
        # Left crack
        for i, p in enumerate(reversed(crack_points)):
            # The crack tip is shared
            if i == 0:
                pc_new: int = gmsh.model.geo.addPoint(-p[0], -p[1], -p[2], h)
                pc_bot.append(pc_new)
                pc_top.append(pc_new)
            else:
                pc_new_bot: int = gmsh.model.geo.addPoint(-p[0], -p[1], -p[2], h)
                pc_bot.append(pc_new_bot)
                pc_new_top: int = gmsh.model.geo.addPoint(-p[0], -p[1], -p[2], h)
                pc_top.append(pc_new_top)

        # Right crack
        for i, p in enumerate(crack_points):
            # The crack tip is shared
            if i == len(crack_points) - 1:
                pc_new: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
                pc_bot.append(pc_new)
                pc_top.append(pc_new)
            else:
                pc_new_bot: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
                pc_bot.append(pc_new_bot)
                pc_new_top: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
                pc_top.append(pc_new_top)

        # Lines
        # Bot
        lb1: int = gmsh.model.geo.addLine(p1, p3)
        l2: int = gmsh.model.geo.addLine(p3, pc_bot[0])
        crack_lines_bot: List[int] = []
        for i in range(len(pc_bot) - 1):
            lb: int = gmsh.model.geo.addLine(pc_bot[i], pc_bot[i + 1])
            crack_lines_bot.append(lb)
        l3: int = gmsh.model.geo.addLine(pc_bot[-1], p4)
        lb4: int = gmsh.model.geo.addLine(p4, p2)
        lb5: int = gmsh.model.geo.addLine(p2, p1)
        # Top
        lt1: int = gmsh.model.geo.addLine(p5, p3)
        crack_lines_top: List[int] = []
        for i in range(len(pc_top) - 1):
            lt: int = gmsh.model.geo.addLine(pc_top[i], pc_top[i + 1])
            crack_lines_top.append(lt)
        lt4: int = gmsh.model.geo.addLine(p4, p6)
        lt5: int = gmsh.model.geo.addLine(p6, p5)

        # Surfaces
        # Bot
        cl1: int = gmsh.model.geo.addCurveLoop(
            [lb1, l2] + crack_lines_bot + [l3, lb4, lb5]
        )
        s1: int = gmsh.model.geo.addPlaneSurface([cl1])
        # Top
        cl2: int = gmsh.model.geo.addCurveLoop(
            [lt1, l2] + crack_lines_top + [l3, lt4, lt5]
        )
        s2: int = gmsh.model.geo.addPlaneSurface([cl2])

        # Boundaries
        self.boundaries = {
            "bot": lb5,
            "top": lt5,
        }
        # Physical groups
        # Domain
        domain: int = gmsh.model.addPhysicalGroup(2, [s1, s2], tag=21)
        gmsh.model.setPhysicalName(2, domain, "domain")
        # Boundaries
        for name, line in self.boundaries.items():
            pg: int = gmsh.model.addPhysicalGroup(1, [line], tag=self.boundaries[name])
            gmsh.model.setPhysicalName(1, pg, name)

        # Element size
        # Refine around the crack line
        field1: int = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field1, "PointsList", [pc_bot[0], pc_bot[-1]])
        gmsh.model.mesh.field.setNumber(field1, "Sampling", 100)
        field2: int = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field2, "InField", field1)
        gmsh.model.mesh.field.setNumber(field2, "DistMin", 1 * self.R_ext)
        gmsh.model.mesh.field.setNumber(field2, "DistMax", 16 * self.R_ext)
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

    def define_controlled_forces(self) -> List[ForceBC]:
        """Define the force boundary conditions controlled by the load factor.

        Returns:
            List[ForceBC]: List of ForceBC(boundary_id, f_imp) where boundary_id is the boundary id (int number) in GMSH, and f_imp is the force vector.
        """
        return [
            ForceBC(boundary_id=self.boundaries["bot"], f_imp=[0.0, -1.0]),
            ForceBC(boundary_id=self.boundaries["top"], f_imp=[0.0, 1.0]),
        ]

    # def define_controlled_displacements(self) -> List[ForceBC]:
    #     """Define the force boundary conditions controlled by the load factor.

    #     Returns:
    #         List[ForceBC]: List of ForceBC(boundary_id, f_imp) where boundary_id is the boundary id (int number) in GMSH, and f_imp is the force vector.
    #     """
    #     return [
    #         DisplacementBC(boundary_id=self.boundaries["bot"], u_imp=[0.0, -1.0]),
    #         DisplacementBC(boundary_id=self.boundaries["top"], u_imp=[0.0, 1.0]),
    #     ]

    def define_locked_points(self) -> List[List[float]]:
        """Define the list of locked points.

        Returns:
            List[List[float]]: A list of points (list) coordinates.
        """
        return [
            [-self.pars["W"] / 2, -self.pars["L"] / 2, 0],
            [self.pars["W"] / 2, -self.pars["L"] / 2, 0],
        ]

    def locate_measured_displacement(self):
        return []

    def locate_measured_forces(self):
        return []

    def Gc(self, phi):
        return 0.0


def run_SIF_computation(data: GCrackData, alpha: float):
    # Get parameters
    a0 = data.pars["a0"]
    # Set the initial crack angle
    data.phi0 = alpha
    # Set the crack tip position
    data.xc0 = np.array([a0 * np.cos(alpha), a0 * np.sin(alpha), 0])
    crack_points = [data.xc0]
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
        body_forces=[],
        nodal_displacements=[],
    )
    # Solve the elastic problem
    u = solve_elastic_problem(domain, model, bcs)

    # from gcrack.exporters import export_function
    # export_function(u, 0, Path("."))

    # Compute the SIFs
    return compute_SIFs(
        domain,
        model,
        u,
        crack_points[-1],
        data.phi0,
        data.R_int,
        data.R_ext,
        data.sif_method,
    )


def compute_sif_evolution_with_angle_alpha(sif_method: str):
    # Define user parameters
    pars = {}
    pars["L"] = 1.0
    pars["W"] = 0.5 * pars["L"]
    pars["a0"] = 0.11 * pars["W"]
    data = GCrackData(
        E=1.0,
        nu=0.3,
        da=pars["L"] / 128,
        Nt=1,
        xc0=np.array([pars["a0"], 0, 0]),
        assumption_2D="plane_strain",
        pars=pars,
        sif_method=sif_method,
        s=0.0,
    )
    # Intialize GMSH
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Disable terminal output
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    # Generate the list of angles
    dalpha = 5
    alphas_deg = [alpha for alpha in range(0, 91, dalpha)]
    alphas = [np.deg2rad(alpha) for alpha in alphas_deg]
    res = np.empty((len(alphas), 4))

    for i, alpha in enumerate(alphas):
        print(f"Case alpha={alphas_deg[i]}°")
        # Run SIFs computation
        SIFs = run_SIF_computation(data, alpha)
        # Compute dimensionless SIFs
        c = np.sqrt(np.pi * pars["a0"])
        KI = SIFs["KI"] / c
        KII = SIFs["KII"] / c
        B = SIFs["T"] * c / np.sqrt(SIFs["KI"] ** 2 + SIFs["KII"] ** 2)
        print(f"{KI=}")
        print(f"{KII=}")
        print(f"{B=}")
        # Store the dimensionless SIFs
        res[i] = alphas_deg[i], KI, KII, B

    # Export the results
    np.savetxt(
        f"sif_{sif_method}.csv",
        res,
        delimiter=",",
        header="alpha, KI, KII, B",
        comments="",
    )

    gmsh.finalize()


if __name__ == "__main__":
    compute_sif_evolution_with_angle_alpha(sif_method="i-integral")
    compute_sif_evolution_with_angle_alpha(sif_method="williams")
