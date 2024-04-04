from pathlib import Path
from datetime import datetime

from abc import ABC, abstractmethod

import gmsh
import numpy as np

import ufl
from dolfinx import fem

from domain import Domain
from models import ElasticModel
from solvers import solve_elastic_problem
from gtheta import G
from optimization_solvers import compute_load_factor
from postprocess import compute_reaction_forces
from exporters import export_function, export_dict_to_csv


class GCrackBaseData(ABC):
    def __init__(
        self,
        E: float,
        nu: float,
        R_int: float,
        da: float,
        xc0: np.array,
        assumption_2D: str = "",
        pars: dict = {},
    ):
        self.E = E
        self.nu = nu
        self.assumption_2D = assumption_2D
        self.R_int = R_int
        self.xc0 = xc0
        self.da = da
        # Add other parameters that can be used in the functions
        self.pars = pars

    @abstractmethod
    def generate_mesh(self, crack_points) -> gmsh.model:
        pass

    @abstractmethod
    def define_dirichlet_bcs(self, crack_points):
        pass

    @abstractmethod
    def Gc(self, gamma: float | np.ndarray) -> float | np.ndarray:
        pass


def gcrack(gcrack_data: GCrackBaseData):
    # Initialize GMSH
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Disable terminal output
    gmsh.option.setNumber("Mesh.Algorithm", 8)  # Use Delaunay algorithm
    # Initialize export directory
    now = datetime.now()
    dir_name = Path("results_" + now.strftime("%Y-%m-%d_%H-%M-%S"))
    dir_name.mkdir(parents=True, exist_ok=True)

    # Initialize the crack points
    crack_points = [gcrack_data.xc0]
    # Store the results
    res = {
        "gamma": [0],
        "lambda": [0],
        "xc_1": [crack_points[0][0]],
        "xc_2": [crack_points[0][1]],
        "xc_3": [crack_points[0][2]],
        "uimp_1": [0.0],
        "uimp_2": [0.0],
        "fimp_1": [0.0],
        "fimp_2": [0.0],
    }

    for t in range(40):
        # Get current crack tip
        xc = crack_points[-1]
        # Generate the mesh
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
        # Define the displacement function space
        element_u = ufl.VectorElement("Lagrange", domain.mesh.ufl_cell(), 1)
        V_u = fem.FunctionSpace(domain.mesh, element_u)
        # Define the boundary conditions
        dirichlet_bcs = gcrack_data.define_dirichlet_bcs(V_u)
        # Solve the elastic problem
        u = solve_elastic_problem(domain, model, V_u, dirichlet_bcs)
        export_function(u, t, dir_name)

        # Compute the energy release rate vector
        g_vec = G(domain, u, xc, model, gcrack_data.R_int)

        # Compute the load factor and crack angle.
        gamma0 = res["gamma"][-1]
        opti_res = compute_load_factor(gamma0, g_vec, gcrack_data.Gc)

        # Get the results
        gamma_ = opti_res.x[0]
        lambda_ = opti_res.fun
        # Add a new crack point
        da_vec = gcrack_data.da * np.array([np.cos(gamma_), np.sin(gamma_), 0])
        xc_new = xc + da_vec
        crack_points.append(xc_new)

        # Postprocess
        fimp = compute_reaction_forces(domain, model, u)

        print(f"==== Time step {t}")
        print(f"crack_tip={xc_new}")
        print(f"gamma={gamma_}")
        print(f"lambda={lambda_}")
        # Store the results
        res["gamma"].append(gamma_)
        res["lambda"].append(lambda_)
        res["xc_1"].append(xc_new[0])
        res["xc_2"].append(xc_new[1])
        res["xc_3"].append(xc_new[2])
        res["uimp_1"].append(0.0)
        res["uimp_2"].append(1.0)
        res["fimp_1"].append(fimp[0])
        res["fimp_2"].append(fimp[1])
    # Export the dictionary to a CSV file
    export_dict_to_csv(res, dir_name / "results.csv")
    # Clean up
    gmsh.finalize()
