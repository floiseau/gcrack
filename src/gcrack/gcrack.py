from pathlib import Path
import logging
from typing import List
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass

import gmsh
import numpy as np

from dolfinx import fem

from domain import Domain
from models import ElasticModel

from solvers import solve_elastic_problem
from sif import compute_SIFs
from optimization_solvers import compute_load_factor
from exporters import export_function, export_dict_to_csv, clean_vtk_files

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
)


@dataclass
class GCrackBaseData(ABC):
    E: float
    nu: float
    R_int: float
    R_ext: float
    da: float
    Nt: int
    xc0: np.array
    assumption_2D: str
    pars: dict

    @abstractmethod
    def generate_mesh(self, crack_points) -> gmsh.model:
        pass

    @abstractmethod
    def define_imposed_displacements(
        self, V_u: fem.FunctionSpace
    ) -> List[fem.DirichletBC]:
        """Define the imposed displacement boundary conditions.

        Args:
            V_u (fem.FunctionSpace): The function space for the displacement field.

        Returns:
            List[fem.DirichletBC]: A list of Dirichlet boundary conditions.
        """
        pass

    @abstractmethod
    def compute_reaction_forces(
        domain: Domain, model: ElasticModel, uh: fem.Function
    ) -> np.array:
        """Compute the reaction forces of the domain.

        Args:
            domain (Domain): The domain of the problem.
            model (ElasticModel): The elastic model being used.
            uh (Function): The displacement solution of the elastic problem.

        Returns:
            np.array: The computed reaction forces as a numpy array.
        """
        pass

    @abstractmethod
    def Gc(self, phi: float | np.ndarray) -> float | np.ndarray:
        pass


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
        "phi": [0],
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
        logging.info(f"\n==== Time step {t}")
        # Get current crack properties
        xc = crack_points[-1]
        phi0 = res["phi"][-1]

        logging.info("-- Meshing the cracked domain")
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
        # Export the elastic solution
        export_function(u, t, dir_name)

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

        # Postprocess
        fimp = gcrack_data.compute_reaction_forces(domain, model, u)

        logging.info("-- Results of the step")
        logging.info(
            f"Crack propagation angle : {phi_:.3f} rad / {phi_*180/np.pi:.3f}°"
        )
        logging.info(f"Load factor             : {lambda_:.3g}")
        logging.info(f"New crack tip position  : {xc_new}")
        # Store the results
        res["phi"].append(phi_)
        res["lambda"].append(lambda_)
        res["xc_1"].append(xc_new[0])
        res["xc_2"].append(xc_new[1])
        res["xc_3"].append(xc_new[2])
        res["uimp_1"].append(0.0)
        res["uimp_2"].append(1.0)
        res["fimp_1"].append(fimp[0])
        res["fimp_2"].append(fimp[1])
    logging.info("-- Finalize the exports")
    # Export the dictionary to a CSV file
    export_dict_to_csv(res, dir_name / "results.csv")
    # Group clean the results directory
    clean_vtk_files(dir_name)
    # Clean up
    gmsh.finalize()
