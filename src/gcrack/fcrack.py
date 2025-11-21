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
    BodyForce,
    NodalDisplacement,
    BoundaryConditions,
)
from gcrack.solvers import solve_elastic_problem
from gcrack.sif import compute_SIFs
from gcrack.postprocess import (
    compute_measured_forces,
    compute_measured_displacement,
    compute_elastic_energy,
    compute_external_work,
)
from gcrack.exporters import export_function, export_res_to_csv, clean_vtk_files


@dataclass
class FCrackBase(ABC):
    """
    FCrackBase is an abstract base class for simulating fatigue crack propagation in elastic materials using the Finite Element Method (FEM).
    """

    E: float
    """E (float): Young's modulus of the material."""
    nu: float
    """nu (float): Poisson's ratio of the material."""
    C: float
    """C (float): Factor parameter in Paris law."""
    m: float
    """m (float): Exponent parameter in Paris law."""
    dG: float
    """dG (float): Threshold in Paris law."""
    dN: float
    """dN (float): Cycle increment."""
    Nt: int
    """Nt (int): Number of crack increment."""
    xc0: np.array
    """xc0 (np.array): Initial crack tip coordinates."""
    assumption_2D: str
    """assumption_2D (str): Assumption for 2D elasticity (e.g., 'plane stress', 'plane strain')."""
    pars: dict
    """pars (dict): User-defined parameters passed to user-defined functions."""
    R_int: float
    """R_int (float): Raduis for the determination of SIFs (pacman inner raduis)."""
    lmax: float
    """lmax (float): Maximal load factor, defaults to 0.0."""
    lmin: Optional[float] = 0.0
    """pars (dict): User-defined parameters passed to user-defined functions."""
    phi0: Optional[float] = 0.0
    """phi0 (Optional[float]): Initial crack propagation angle, defaults to 0.0."""
    sif_method: Optional[str] = "Williams"
    """sif_method (Optional[str]): Method for computing Stress Intensity Factors (SIFs), defaults to "Williams"."""
    # criterion: Optional[str] = "gmerr"
    # """criterion (Optional[str]): Criterion for crack propagation, defaults to "gmerr"."""
    name: Optional[str] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    """name (Optional[str]): Name of the simulation used to name the results directory."""
    no_meshing: Optional[bool] = False
    """no_meshing (Optional[bool]): Flag to skip mesh generation when a mesh has already been created."""

    def __post_init__(self):
        # Compute the radii for the SIF evaluation
        self.R_ext = 2 * self.R_int

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

    def define_locked_points(self) -> List[List[float]]:
        """Define the list of locked points.

        Returns:
            List[List[float]]: A list of points (list) coordinates.
        """
        return []

    def define_nodal_displacements(self) -> List[NodalDisplacement]:
        """Define a list of imposed nodal displacements.

        Returns:
            List[NodalDisplacements]: A list of NodalDisplacement.
        """
        return []

    def define_displacements(self) -> List[DisplacementBC]:
        """Define the displacement boundary conditions.

        Returns:
            List[DisplacementBC]: List of DisplacementBC(boundary_id, u_imp) where boundary_id is the boundary id (int number) in GMSH, and u_imp is the displacement vector (componements can be nan to let it free).
        """
        return []

    def define_forces(self) -> List[ForceBC]:
        """Define the force boundary conditions.

        Returns:
            List[ForceBC]: List of ForceBC(boundary_id, f_imp) where boundary_id is the boundary id (int number) in GMSH, and f_imp is the force vector.
        """
        return []

    def define_body_forces(self) -> List[BodyForce]:
        """Define the body forces.

        Returns:
            List[BodyForce]: List of BodyForce (f_imp) where f_imp is the force vector.
        """
        return []

    def user_load_step_initialization(self, res: dict) -> None:
        """User-defined load step initialization.

        Customize load step initialization using results from the previous step.
        This can include creating custom variables for boundary conditions.
        This function is called at the very beginning of the load step.
        """
        ...

    def run(self):
        # Initialize GMSH
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # Disable terminal output
        gmsh.option.setNumber("Mesh.Algorithm", 5)
        # 1: meshadapt; 5: delaunay, 6: frontal-delaunay

        # Initialize export directory
        dir_name = Path("results_" + self.name)
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
            "da": 0,
            "phi": self.phi0,
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
            "elastic_energy": 0.0,
            "fracture_dissipation": 0.0,
            "external_work": 0.0,
        }

        for t in range(1, self.Nt + 1):
            print(f"\nLOAD STEP {t}")
            # Get current crack properties
            phi0 = res["phi"]

            # Run the user defined load step initialization
            self.user_load_step_initialization(res)

            # Generate the mesh
            if not self.no_meshing:
                print("│  Meshing the cracked domain")
                gmsh_model = self.generate_mesh(crack_points)
            else:
                print("│  Skip meshing (no_meshing = True)")

            # Get the controlled boundary conditions
            bcs = BoundaryConditions(
                displacement_bcs=self.define_displacements(),
                force_bcs=self.define_forces(),
                body_forces=self.define_body_forces(),
                locked_points=self.define_locked_points(),
                nodal_displacements=self.define_nodal_displacements(),
            )

            # Define the domain
            if not self.no_meshing:
                self.domain = Domain(gmsh_model)

            # Define an elastic model
            model = ElasticModel(ela_pars, self.domain)

            print("│  Solve the controlled elastic problem with FEM")
            # Solve the controlled elastic problem
            u = solve_elastic_problem(self.domain, model, bcs)

            print(f"│  Compute the SIFs for the controlled problem ({self.sif_method})")
            # Compute the SIFs for the controlled problem
            SIFs = compute_SIFs(
                self.domain,
                model,
                u,
                crack_points[-1],
                phi0,
                self.R_int,
                self.R_ext,
                self.sif_method,
            )

            # Compute the load factor and crack angle.
            print("│  Determination of crack increment (Paris law)")
            # Compute the crack increment (foward euler scheme
            G_bar = 1 / model.Ep * (SIFs["KI"] ** 2 + SIFs["KII"] ** 2) + SIFs[
                "KIII"
            ] ** 2 / (2 * model.mu)
            da = (
                self.C
                * max((self.lmax**2 - self.lmin**2) * G_bar - self.dG, 0) ** self.m
                * self.dN
            )
            print(da)
            # Add a new crack point
            da_vec = da * np.array([np.cos(phi0), np.sin(phi0), 0])
            crack_points.append(crack_points[-1] + da_vec)

            print("│  Results of the step")
            print(
                f"│  │  Crack propagation angle : {phi0:.3f} rad / {phi0 * 180 / np.pi:.3f}°"
            )
            print(f"│  │  Crack increment         : {da:.3g}")
            print(f"│  │  New crack tip position  : {crack_points[-1]}")

            print("│  Postprocess")
            # Compute and postprocess the max displacement field
            u_max = u.copy()
            u_max.x.array[:] = self.lmax * u.x.array
            u_max.name = "umax"
            fimp_max = compute_measured_forces(self.domain, model, u_max, self)
            uimp_max = compute_measured_displacement(self.domain, u_max, self)
            elastic_energy_max = compute_elastic_energy(self.domain, model, u_max)
            external_work_max = compute_external_work(self.domain, model, u_max)
            # Compute the min displacement field (when lmin is not zero)
            u_min = u.copy()
            u_min.x.array[:] = self.lmin * u.x.array
            u_min.name = "umin"
            fimp_min = compute_measured_forces(self.domain, model, u_min, self)
            uimp_min = compute_measured_displacement(self.domain, u_min, self)
            elastic_energy_min = compute_elastic_energy(self.domain, model, u_min)
            external_work_min = compute_external_work(self.domain, model, u_min)

            print("│  Export the results")
            # Export the elastic solution
            export_function(u_max, t, dir_name)
            if self.lmin != 0:
                export_function(u_min, t, dir_name)
            # Store the results
            res["t"] = t
            res["a"] += da
            res["da"] = da
            res["phi"] = phi0
            res["xc_1"] = crack_points[-1][0]
            res["xc_2"] = crack_points[-1][1]
            res["xc_3"] = crack_points[-1][2]
            for vname, vec in [
                ("f_min", fimp_min),
                ("u_min", uimp_min),
                ("f_max", fimp_max),
                ("u_max", uimp_max),
            ]:
                for comp, vec_comp in enumerate(vec):
                    res[f"{vname}_{comp + 1}"] = vec_comp

            for sif_name in SIFs:
                res[f"{sif_name}_min"] = self.lmin * SIFs[sif_name]
                res[f"{sif_name}_max"] = self.lmax * SIFs[sif_name]
            res["elastic_energy_min"] = elastic_energy_min
            res["elastic_energy_max"] = elastic_energy_max
            res["fracture_dissipation"] = "TODO"
            res["external_work_min"] = external_work_min
            res["external_work_max"] = external_work_max
            # At first load step, also export the initial state
            if t == 1:
                res_init = res.copy()
                for key in res_init:
                    res_init[key] = 0.0
                export_res_to_csv(res_init, dir_name / "results.csv")
                del res_init
            # Export the current results to csv
            export_res_to_csv(res, dir_name / "results.csv")
        print("\nFinalize exports")
        # Group clean the results directory
        clean_vtk_files(dir_name)
        # Clean up
        gmsh.finalize()
