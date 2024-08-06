from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass

import gmsh
import numpy as np

from domain import Domain
from models import ElasticModel

from solvers import solve_elastic_problem
from sif import compute_SIFs
from optimization_solvers import compute_load_factor
from postprocess import compute_measured_forces, compute_measured_displacement
from exporters import export_function, export_dict_to_csv, clean_vtk_files


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

    def __post_init__(self):
        # Compute the radii for the SIF evaluation
        self.R_int = self.da
        self.R_ext = 2 * self.da

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

    def define_imposed_displacements(self) -> List[Tuple[int, List[float]]]:
        """Define the imposed displacement boundary conditions.

        Returns:
            Tuple: with (id, value) where id is the boundary id (int number) in GMSH, and value is the displacement vector (componements can be nan to let it free).
        """
        return []

    def define_imposed_forces(self) -> List[Tuple[int, List[float]]]:
        """
        Define the list of imposed forces.
        Each element of the list is a tuple.

        Returns:
            Tuple:  with (id, value) where id is the boundary id (int number) in GMSH, and value is the force vector.
        """
        return []


def gcrack(gcrack_data: GCrackBaseData):
    # Initialize GMSH
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Disable terminal output
    gmsh.option.setNumber("Mesh.Algorithm", 5)  # Use meshadapt algorithm

    # Initialize export directory
    now = datetime.now()
    dir_name = Path("results_" + now.strftime("%Y-%m-%d_%H-%M-%S"))
    dir_name.mkdir(parents=True, exist_ok=True)

    # Initialize the crack points
    crack_points = [gcrack_data.xc0]
    # Store the results (TODO: Change the initialization to be more generic)
    res = {
        "a": [0],
        "phi": [gcrack_data.phi0],
        "lambda": [0],
        "xc_1": [crack_points[0][0]],
        "xc_2": [crack_points[0][1]],
        "xc_3": [crack_points[0][2]],
        "uimp_1": [0.0],
        "uimp_2": [0.0],
        "fimp_1": [0.0],
        "fimp_2": [0.0],
    }

    for t in range(gcrack_data.Nt):
        print(f"\n==== Time step {t}")
        # Get current crack properties
        xc = crack_points[-1]
        phi0 = res["phi"][-1]

        print("-- Meshing the cracked domain")
        gmsh_model = gcrack_data.generate_mesh(crack_points)

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
        u = solve_elastic_problem(domain, model, gcrack_data)

        # Compute the energy release rate vector
        K = compute_SIFs(
            domain, model, u, xc, phi0, gcrack_data.R_int, gcrack_data.R_ext
        )

        # Compute the load factor and crack angle.
        opti_res = compute_load_factor(phi0, model, K, gcrack_data.Gc)

        # Get the results
        phi_ = opti_res[0]
        lambda_ = opti_res[1]
        # Add a new crack point
        da_vec = gcrack_data.da * np.array([np.cos(phi_), np.sin(phi_), 0])
        xc_new = xc + da_vec
        crack_points.append(xc_new)

        print("-- Results of the step")
        print(f"Crack propagation angle : {phi_:.3f} rad / {phi_*180/np.pi:.3f}°")
        print(f"Load factor             : {lambda_:.3g}")
        print(f"New crack tip position  : {xc_new}")

        print("-- Postprocess")
        # Compute the reaction force
        fimp = compute_measured_forces(domain, model, u, gcrack_data)
        uimp = compute_measured_displacement(domain, u, gcrack_data)
        # Scale the displacement field
        u_scaled = u.copy()
        u_scaled.x.array[:] = lambda_ * u_scaled.x.array
        u.scaled = "Displacement"

        print("-- Export the results")
        # Export the elastic solution
        export_function(u_scaled, t, dir_name)
        # Store the results
        res["a"].append(res["a"][-1] + gcrack_data.da)
        res["phi"].append(phi_)
        res["lambda"].append(lambda_)
        res["xc_1"].append(xc_new[0])
        res["xc_2"].append(xc_new[1])
        res["xc_3"].append(xc_new[2])
        res["uimp_1"].append(uimp[0])
        res["uimp_2"].append(uimp[1])
        res["fimp_1"].append(fimp[0])
        res["fimp_2"].append(fimp[1])
    print("-- Finalize the exports")
    # Export the dictionary to a CSV file
    export_dict_to_csv(res, dir_name / "results.csv")
    # Group clean the results directory
    clean_vtk_files(dir_name)
    # Clean up
    gmsh.finalize()
