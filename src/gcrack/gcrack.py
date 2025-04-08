from pathlib import Path
from typing import List, Optional
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass

import gmsh
import numpy as np

from gcrack.domain import Domain
from gcrack.models import ElasticModel
from gcrack.boundary_conditions import (
    DisplacementBC,
    ForceBC,
    BoundaryConditions,
)
from gcrack.solvers import solve_elastic_problem
from gcrack.sif import compute_SIFs
from gcrack.optimization_solvers import LoadFactorSolver
from gcrack.postprocess import compute_measured_forces, compute_measured_displacement
from gcrack.exporters import export_function, export_res_to_csv, clean_vtk_files


@dataclass
class GCrackBase(ABC):
    E: float
    nu: float
    da: float
    Nt: int
    xc0: np.array
    assumption_2D: str
    pars: dict  # User defined parameters (passed to user-defined functions)
    phi0: Optional[float] = 0.0
    s: Optional[float] = 0  # Internal length associated with T-stress
    sif_method: Optional[str] = "I-integral"
    criterion: Optional[str] = "gmerr"

    def __post_init__(self):
        # Compute the radii for the SIF evaluation
        self.R_int = 2 * self.da
        self.R_ext = 4 * self.da

    @abstractmethod
    def generate_mesh(self, crack_points) -> gmsh.model:
        pass

    @abstractmethod
    def locate_measured_displacement(self) -> List[float]:
        """Define the point where the displacement is measured.

        Returns:
            List: Coordinate of the point where the displacement is measured
        """
        pass

    @abstractmethod
    def locate_measured_forces(self) -> int:
        """Define the boundary where the reaction force are measured.

        Returns:
            int: Identifier (id) of the boundary in GMSH.
        """
        pass

    @abstractmethod
    def Gc(self, phi: float | np.ndarray) -> float | np.ndarray:
        pass

    def define_locked_points(self) -> List[List[float]]:
        """Define the list of locked points.

        Returns:
            List[List[float]]: A list of points (list) coordinates.
        """
        return []

    def define_controlled_displacements(self) -> List[DisplacementBC]:
        """Define the displacement boundary conditions controlled by the load factor.

        Returns:
            List[DisplacementBC]: List of DisplacementBC(boundary_id, u_imp) where boundary_id is the boundary id (int number) in GMSH, and u_imp is the displacement vector (componements can be nan to let it free).
        """
        return []

    def define_controlled_forces(self) -> List[ForceBC]:
        """Define the force boundary conditions controlled by the load factor.

        Returns:
            List[ForceBC]: List of ForceBC(boundary_id, f_imp) where boundary_id is the boundary id (int number) in GMSH, and f_imp is the force vector.
        """
        return []

    def define_prescribed_displacements(self) -> List[DisplacementBC]:
        """Define the prescribed displacement boundary conditions that are not affected by the load factor.

        Returns:
            List[DisplacementBC]: List of DisplacementBC(boundary_id, u_imp) where boundary_id is the boundary id (int number) in GMSH, and u_imp is the displacement vector (componements can be nan to let it free).
        """
        return []

    def define_prescribed_forces(self) -> List[ForceBC]:
        """Define the prescribed force boundary conditions that are not affected by the load factor.

        Returns:
            List[ForceBC]: List of ForceBC(boundary_id, f_imp) where boundary_id is the boundary id (int number) in GMSH, and f_imp is the force vector.
        """
        return []

    def run(self):
        # Initialize GMSH
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # Disable terminal output
        gmsh.option.setNumber("Mesh.Algorithm", 1)
        # 1: meshadapt; 5: delaunay, 6: frontal-delaunay

        # Initialize export directory
        now = datetime.now()
        dir_name = Path("results_" + now.strftime("%Y-%m-%d_%H-%M-%S"))
        dir_name.mkdir(parents=True, exist_ok=True)

        # Get the elastic parameters
        ela_pars = {
            "E": self.E,
            "nu": self.nu,
            "2D_assumption": self.assumption_2D,
        }
        # Initialize the crack points
        crack_points = [self.xc0]
        # Initialize results storage
        res = {
            "t": 0,
            "a": 0,
            "phi": self.phi0,
            "lambda": 0,
            "xc_1": crack_points[-1][0],
            "xc_2": crack_points[-1][1],
            "xc_3": crack_points[-1][2],
            "uimp_1": 0.0,
            "uimp_2": 0.0,
            "fimp_1": 0.0,
            "fimp_2": 0.0,
            "KI": 0.0,
            "KII": 0.0,
            "T": 0.0,
        }
        export_res_to_csv(res, dir_name / "results.csv")

        for t in range(1, self.Nt + 1):
            print(f"\nLOAD STEP {t}")
            # Get current crack properties
            phi0 = res["phi"]

            print("│  Meshing the cracked domain")
            gmsh_model = self.generate_mesh(crack_points)

            # Get the controlled boundary conditions
            controlled_bcs = BoundaryConditions(
                displacement_bcs=self.define_controlled_displacements(),
                force_bcs=self.define_controlled_forces(),
                locked_points=self.define_locked_points(),
            )

            # Get the controlled boundary conditions
            prescribed_bcs = BoundaryConditions(
                displacement_bcs=self.define_prescribed_displacements(),
                force_bcs=self.define_prescribed_forces(),
                locked_points=self.define_locked_points(),
            )

            # Define the domain
            domain = Domain(gmsh_model)

            # Define an elastic model
            model = ElasticModel(ela_pars, domain)

            print("│  Solve the controlled elastic problem with FEM")
            # Solve the controlled elastic problem
            u_controlled = solve_elastic_problem(domain, model, controlled_bcs)

            print(f"│  Compute the SIFs for the controlled problem ({self.sif_method})")
            # Compute the SIFs for the controlled problem
            SIFs_controlled = compute_SIFs(
                domain,
                model,
                u_controlled,
                crack_points[-1],
                phi0,
                self.R_int,
                self.R_ext,
                self.sif_method,
            )

            # Tackle the prescribed problem
            if not prescribed_bcs.is_empty():
                print("│  Solve the prescribed elastic problem with FEM")
                # Solve the prescribed elastic problem
                u_prescribed = solve_elastic_problem(domain, model, prescribed_bcs)
                # Compute the SIFs for the prescribed problem
                SIFs_prescribed = compute_SIFs(
                    domain,
                    model,
                    u_prescribed,
                    crack_points[-1],
                    phi0,
                    self.R_int,
                    self.R_ext,
                    self.sif_method,
                )
            else:
                # Set the prescribed displacement to 0
                u_prescribed = u_controlled.copy()
                u_prescribed.x.array[:] = 0.0
                # Set the SIFs to 0
                SIFs_prescribed = {
                    "KI": 0.0,
                    "KII": 0.0,
                    "T": 0.0,
                }
                print("│  No prescribed BCs")

            # Compute the load factor and crack angle.

            print("│  Determination of propagation angle and load factor")
            load_factor_solver = LoadFactorSolver(model, self.Gc)
            opti_res = load_factor_solver.solve(
                phi0, SIFs_controlled, SIFs_prescribed, self.s
            )
            # Get the results
            phi_ = opti_res[0]
            lambda_ = opti_res[1]
            # # NOTE: DEBUG
            # load_factor_solver.export_minimization_plots(
            #     phi_,
            #     lambda_,
            #     phi0,
            #     SIFs_controlled,
            #     SIFs_prescribed,
            #     self.s,
            #     t,
            #     dir_name,
            # )
            # Add a new crack point
            da_vec = self.da * np.array([np.cos(phi_), np.sin(phi_), 0])
            crack_points.append(crack_points[-1] + da_vec)

            print("│  Results of the step")
            print(
                f"│  │  Crack propagation angle : {phi_:.3f} rad / {phi_ * 180 / np.pi:.3f}°"
            )
            print(f"│  │  Load factor             : {lambda_:.3g}")
            print(f"│  │  New crack tip position  : {crack_points[-1]}")

            print("│  Postprocess")
            # Scale the displacement field
            u_scaled = u_controlled.copy()
            u_scaled.x.array[:] = lambda_ * u_controlled.x.array + u_prescribed.x.array
            u_scaled.name = "Displacement"
            # Compute the reaction force
            fimp = compute_measured_forces(domain, model, u_scaled, self)
            uimp = compute_measured_displacement(domain, u_scaled, self)

            print("│  Export the results")
            # Export the elastic solution
            export_function(u_scaled, t, dir_name)
            # Store and export the results
            res["t"] = t + 1
            res["a"] += self.da
            res["phi"] = phi_
            res["lambda"] = lambda_
            res["xc_1"] = crack_points[-1][0]
            res["xc_2"] = crack_points[-1][1]
            res["xc_3"] = crack_points[-1][2]
            res["uimp_1"] = uimp[0]
            res["uimp_2"] = uimp[1]
            res["fimp_1"] = fimp[0]
            res["fimp_2"] = fimp[1]
            res["KI"] = lambda_ * SIFs_controlled["KI"] + SIFs_prescribed["KI"]
            res["KII"] = lambda_ * SIFs_controlled["KII"] + SIFs_prescribed["KII"]
            res["T"] = lambda_ * SIFs_controlled["T"] + SIFs_prescribed["T"]
            export_res_to_csv(res, dir_name / "results.csv")
        print("\nFinalize exports")
        # Group clean the results directory
        clean_vtk_files(dir_name)
        # Clean up
        gmsh.finalize()
