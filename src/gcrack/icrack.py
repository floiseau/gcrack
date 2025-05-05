from pathlib import Path
from typing import List, Optional
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass

import gmsh
import numpy as np

from dolfinx.fem import Constant

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
class ICrackBase(ABC):
    E: float
    nu: float
    da: float
    l0: int
    dl: float
    l_max: float
    xc0: np.array
    assumption_2D: str
    pars: dict  # User defined parameters (passed to user-defined functions)
    phi0: Optional[float] = 0.0
    s: Optional[float] = 0  # Internal length associated with T-stress
    sif_method: Optional[str] = "I-integral"
    criterion: Optional[str] = "gmerr"

    def __post_init__(self):
        # Compute the radii for the SIF evaluation
        self.R_int = 1 / 8 * self.da
        self.R_ext = 1 / 4 * self.da

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

    def define_boundary_displacements(self, l: float) -> List[DisplacementBC]:
        """Define the displacement boundary conditions as a function of the load factor.

        Returns:
            List[DisplacementBC]: List of DisplacementBC(boundary_id, u_imp) where boundary_id is the boundary id (int number) in GMSH, and u_imp is the displacement vector (componements can be nan to let it free).
        """
        return []

    def define_boundary_forces(self, l: float) -> List[ForceBC]:
        """Define the force boundary conditions as a function of the load factor.

        Returns:
            List[ForceBC]: List of ForceBC(boundary_id, f_imp) where boundary_id is the boundary id (int number) in GMSH, and f_imp is the force vector.
        """
        return []

    def run(self):
        # Initialize GMSH
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # Disable terminal output
        gmsh.option.setNumber("Mesh.Algorithm", 5)
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
        # Initialize a time counter
        t = -1
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

        # Initialize the load factor
        self.l = self.l0
        # Initialize the previous crack angle
        phi_nm1 = self.phi0

        while self.l < self.l_max:
            print(f"\nLOAD FACTOR {self.l:.3g}")
            # Increment time
            t += 1
            # Initialize the crack propagation flag as true
            crack_propagates = True

            while crack_propagates:
                print("│  Meshing the cracked domain")
                gmsh_model = self.generate_mesh(crack_points)
                # Define the domain
                domain = Domain(gmsh_model)

                # Get the controlled boundary conditions
                bcs = BoundaryConditions(
                    displacement_bcs=self.define_boundary_displacements(self.l),
                    force_bcs=self.define_boundary_forces(self.l),
                    locked_points=self.define_locked_points(),
                )

                # Define an elastic model
                model = ElasticModel(ela_pars, domain)

                print("│  Solve the elastic problem with FEM")
                # Solve the elastic problem
                u = solve_elastic_problem(domain, model, bcs)

                print(f"│  Compute the SIFs ({self.sif_method})")
                # Compute the SIFs for the controlled problem
                SIFs = compute_SIFs(
                    domain,
                    model,
                    u,
                    crack_points[-1],
                    phi_nm1,
                    self.R_int,
                    self.R_ext,
                    self.sif_method,
                )
                # Set the SIFs of the prescribed loading to 0
                SIFs_prescribed = {"KI": 0.0, "KII": 0.0, "T": 0.0}

                print("│  Check if the crack propagates")
                # Compute the load factor and crack angle.
                load_factor_solver = LoadFactorSolver(model, self.Gc)
                opti_res = load_factor_solver.solve(
                    phi_nm1, SIFs, SIFs_prescribed, self.s
                )

                # NOTE: DEBUG
                load_factor_solver.export_minimization_plots(
                    opti_res[0],
                    opti_res[1],
                    phi_nm1,
                    SIFs,
                    SIFs_prescribed,
                    self.s,
                    t,
                    dir_name,
                )
                # Get the results
                phi_ = opti_res[0]
                crack_propagates = opti_res[1] <= 1
                # Add a new crack point
                if crack_propagates:
                    print("│  │  The crack propagates: continue the propagation")
                    # Increment the crack
                    da_vec = self.da * np.array([np.cos(phi_), np.sin(phi_), 0])
                    crack_points.append(crack_points[-1] + da_vec)
                    print(f"│  │  New crack tip position  : {crack_points[-1]}")
                    # Update the previous crack angle
                    phi_nm1 = phi_
                else:
                    print("│  │  The crack does not propagates: end of the load step")

                print("│  Export the results")
                # Export the elastic solution
                u.name = "Displacement"
                export_function(u, t, dir_name)
                # Compute the reaction force
                fimp = compute_measured_forces(domain, model, u, self)
                uimp = compute_measured_displacement(domain, u, self)
                # Store and export the results
                print("│  Results of the step")
                res["t"] = t
                res["a"] += self.da
                res["phi"] = phi_
                res["lambda"] = self.l
                res["xc_1"] = crack_points[-1][0]
                res["xc_2"] = crack_points[-1][1]
                res["xc_3"] = crack_points[-1][2]
                res["uimp_1"] = uimp[0]
                res["uimp_2"] = uimp[1]
                res["fimp_1"] = fimp[0]
                res["fimp_2"] = fimp[1]
                for sif_name in SIFs:
                    res[sif_name] = SIFs[sif_name]
                    export_res_to_csv(res, dir_name / "results.csv")

                if crack_propagates:
                    print("-  Next crack increment")
            # Increment the load
            self.l += self.dl

        print("\nFinalize exports")
        # Group clean the results directory
        clean_vtk_files(dir_name)
        # Clean up
        gmsh.finalize()
