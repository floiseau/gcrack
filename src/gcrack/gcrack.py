from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass

import gmsh
import numpy as np

from domain import Domain
from models import ElasticModel

from boundary_conditions import (
    DisplacementBC,
    ForceBC,
    BoundaryConditions,
)
from solvers import solve_elastic_problem
from sif import compute_SIFs
from optimization_solvers import compute_load_factor
from postprocess import compute_measured_forces, compute_measured_displacement
from exporters import export_function, export_res_to_csv, clean_vtk_files


@dataclass
class GCrackBaseData(ABC):
    E: float
    nu: float
    da: float
    Nt: int
    xc0: np.array
    assumption_2D: str
    pars: dict  # User defined parameters (passed to user-defined functions)
    phi0: Optional[float] = 0
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

    def define_imposed_displacements(self) -> List[DisplacementBC]:
        """Define the imposed displacement boundary conditions.

        Returns:
            List[DisplacementBC]: List of DisplacementBC(boundary_id, u_imp) where boundary_id is the boundary id (int number) in GMSH, and u_imp is the displacement vector (componements can be nan to let it free).
        """
        return []

    def define_imposed_forces(self) -> List[ForceBC]:
        """Define the list of imposed forces.

        Returns:
            List[ForceBC]: List of ForceBC(boundary_id, f_imp) where boundary_id is the boundary id (int number) in GMSH, and f_imp is the force vector.
        """
        return []


def gcrack(gcrack_data: GCrackBaseData):
    # Initialize GMSH
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Disable terminal output
    gmsh.option.setNumber("Mesh.Algorithm", 1)
    # 1: meshadapt; 5: delaunay, 6: frontal-delaunay

    # Initialize export directory
    now = datetime.now()
    dir_name = Path("results_" + now.strftime("%Y-%m-%d_%H-%M-%S"))
    dir_name.mkdir(parents=True, exist_ok=True)

    # Initialize the crack points
    crack_points = [gcrack_data.xc0]
    # Initialize results storage
    res = {
        "t": 0,
        "a": 0,
        "phi": gcrack_data.phi0,
        "lambda": 0,
        "xc_1": crack_points[0][0],
        "xc_2": crack_points[0][1],
        "xc_3": crack_points[0][2],
        "uimp_1": 0.0,
        "uimp_2": 0.0,
        "fimp_1": 0.0,
        "fimp_2": 0.0,
        "KI": 0.0,
        "KII": 0.0,
        "T": 0.0,
    }
    export_res_to_csv(res, dir_name / "results.csv")

    for t in range(1, gcrack_data.Nt + 1):
        print(f"\n==== Time step {t}")
        # Get current crack properties
        xc = crack_points[-1]
        phi0 = res["phi"]

        print("-- Meshing the cracked domain")
        gmsh_model = gcrack_data.generate_mesh(crack_points)

        # Get the boundary conditions
        controlled_bcs = BoundaryConditions(
            displacement_bcs=gcrack_data.define_imposed_displacements(),
            force_bcs=gcrack_data.define_imposed_forces(),
            locked_points=gcrack_data.define_locked_points(),
        )

        # Define the domain
        domain = Domain(gmsh_model)

        # Define an elastic model
        ela_pars = {
            "E": gcrack_data.E,
            "nu": gcrack_data.nu,
            "2D_assumption": gcrack_data.assumption_2D,
        }
        model = ElasticModel(ela_pars, domain)

        # Solve the elastic problem
        u = solve_elastic_problem(domain, model, controlled_bcs)

        # Compute the energy release rate vector
        SIFs = compute_SIFs(
            domain,
            model,
            u,
            crack_points[-1],
            phi0,
            gcrack_data.R_int,
            gcrack_data.R_ext,
            gcrack_data.sif_method,
        )

        # Compute the load factor and crack angle.
        opti_res = compute_load_factor(
            phi0, model, SIFs, gcrack_data.Gc, gcrack_data.s, gcrack_data.criterion
        )

        # Get the results
        phi_ = opti_res[0]
        lambda_ = opti_res[1]
        # Add a new crack point
        da_vec = gcrack_data.da * np.array([np.cos(phi_), np.sin(phi_), 0])
        crack_points.append(crack_points[-1] + da_vec)

        print("-- Results of the step")
        print(f"Crack propagation angle : {phi_:.3f} rad / {phi_*180/np.pi:.3f}°")
        print(f"Load factor             : {lambda_:.3g}")
        print(f"New crack tip position  : {crack_points[-1]}")

        print("-- Postprocess")
        # Scale the displacement field
        u_scaled = u.copy()
        u_scaled.x.array[:] = lambda_ * u_scaled.x.array
        u.scaled = "Displacement"
        # Compute the reaction force
        fimp = compute_measured_forces(domain, model, u_scaled, gcrack_data)
        uimp = compute_measured_displacement(domain, u_scaled, gcrack_data)

        print("-- Export the results")
        # Export the elastic solution
        export_function(u_scaled, t, dir_name)
        # Store and export the results
        res["t"] = t + 1
        res["a"] += gcrack_data.da
        res["phi"] = phi_
        res["lambda"] = lambda_
        res["xc_1"] = crack_points[-1][0]
        res["xc_2"] = crack_points[-1][1]
        res["xc_3"] = crack_points[-1][2]
        res["uimp_1"] = uimp[0]
        res["uimp_2"] = uimp[1]
        res["fimp_1"] = fimp[0]
        res["fimp_2"] = fimp[1]
        res["KI"] = SIFs["KI"]
        res["KII"] = SIFs["KII"]
        res["T"] = SIFs["T"]
        export_res_to_csv(res, dir_name / "results.csv")
    print("-- Finalize the exports")
    # Group clean the results directory
    clean_vtk_files(dir_name)
    # Clean up
    gmsh.finalize()
