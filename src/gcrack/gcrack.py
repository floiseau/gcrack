"""
Main entry point for the GCrack simulation framework.

This module serves as the entry point for simulating crack propagation in elastic materials using the Finite Element Method (FEM).
It contains the entire simulation workflow, including mesh generation, boundary condition application, solving the elastic problem, computing Stress Intensity Factors (SIFs), and post-processing results.

The simulation is driven by the `GCrackBase` abstract base class, which users must subclass to define problem-specific parameters and behaviors.
The workflow includes:

- Generating the mesh for the cracked domain.
- Applying boundary conditions (displacements, forces, body forces, locked points, etc.).
- Solving the elastic problem for both controlled and prescribed boundary conditions.
- Computing SIFs using the specified method (e.g., I-integral).
- Determining the crack propagation angle and load factor.
- Post-processing results, including computing energies, reaction forces, and displacements.
- Exporting results to VTK and CSV formats.

Classes:
    GCrackBase:
        Abstract base class for defining and running crack propagation simulations.
        Users must subclass this class and implement abstract methods to define problem-specific parameters and behaviors.

Attributes:
    None: All attributes and methods are encapsulated within the `GCrackBase` class.

Usage:
    To run a simulation, users must:

    1. Subclass `GCrackBase` and implement all abstract methods.
    2. Instantiate the subclass with the required parameters.
    3. Call the `run()` method to execute the simulation.

Example:

    from gcrack.main import GCrackBase
    class MySimulation(GCrackBase):
        def generate_mesh(self, crack_points):
            # Implement mesh generation logic
        def locate_measured_displacement(self):
            # Implement logic to locate measured displacement
        # Implement other abstract methods

    simulation = MySimulation(E=1e5, nu=0.3, da=0.1, Nt=10, xc0=np.array([0, 0, 0]))
    simulation.run()
"""

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
from gcrack.optimization_solvers import LoadFactorSolver
from gcrack.postprocess import (
    compute_measured_forces,
    compute_measured_displacement,
    compute_elastic_energy,
    compute_external_work,
)
from gcrack.exporters import export_function, export_res_to_csv, clean_vtk_files


@dataclass
class GCrackBase(ABC):
    """
    GCrackBase is an abstract base class for simulating crack propagation in elastic materials using the Finite Element Method (FEM).
    """

    E: float
    """E (float): Young's modulus of the material."""
    nu: float
    """nu (float): Poisson's ratio of the material."""
    da: float
    """da (float): Crack increment length."""
    Nt: int
    """Nt (int): Number of crack increment."""
    xc0: np.array
    """xc0 (np.array): Initial crack tip coordinates."""
    assumption_2D: str
    """assumption_2D (str): Assumption for 2D elasticity (e.g., 'plane stress', 'plane strain')."""
    pars: dict
    """pars (dict): User-defined parameters passed to user-defined functions."""
    l0: Optional[float] = 0.0
    """l0 (Optional[float]): Initial load factor, defaults to 0.0."""
    phi0: Optional[float] = 0.0
    """phi0 (Optional[float]): Initial crack propagation angle, defaults to 0.0."""
    s: Optional[float] = 0
    """s (Optional[float]): Internal length associated with T-stress, defaults to 0."""
    sif_method: Optional[str] = "I-integral"
    """sif_method (Optional[str]): Method for computing Stress Intensity Factors (SIFs), defaults to "I-integral"."""
    criterion: Optional[str] = "gmerr"
    """criterion (Optional[str]): Criterion for crack propagation, defaults to "gmerr"."""
    name: Optional[str] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    """name (Optional[str]): Name of the simulation used to name the results directory."""
    no_propagation: Optional[bool] = False
    """no_propagation (Optional[bool]): Flag to only run skip the crack propagation phase."""
    no_meshing: Optional[bool] = False
    """no_meshing (Optional[bool]): Flag to skip mesh generation when a mesh has already been created."""

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
        """Define the critical energy release rate.

        To account for material anisotropy, the critical energy release rate can depend on the crack orientation $\\varphi$

        Args:
            phi (np.ndarray): Crack angle.

        Returns:
            np.ndarray: Value of the critical energy release rate.

        Note:
            The intput and output should be arrays for practical details in the minimization of the load factor.
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

    def define_controlled_body_forces(self) -> List[BodyForce]:
        """Define the controlled body forces that are affected by the load factor.

        Returns:
            List[BodyForce]: List of BodyForce (f_imp) where f_imp is the force vector.
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

    def define_prescribed_body_forces(self) -> List[BodyForce]:
        """Define the prescribed body forces that are not affected by the load factor.

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
        """Executes the crack propagation simulation workflow.

        This method manages the entire simulation process, including:

        - Initializing the GMSH environment and results directory.
        - Generating the mesh for the cracked domain (unless `no_meshing` is True).
        - Applying boundary conditions (displacements, forces, body forces, locked points, etc.).
        - Solving the elastic problem for both controlled and prescribed boundary conditions.
        - Computing Stress Intensity Factors (SIFs) using the specified method.
        - Determining the crack propagation angle and load factor.
        - Post-processing results, including computing energies, reaction forces, and displacements.
        - Exporting results to VTK and CSV formats.

        The simulation runs for `self.Nt` load steps, updating the crack tip position and
        boundary conditions at each step. Results are stored in a dictionary and exported
        to the results directory.

        Steps:

        1. Initialize GMSH and create the results directory.
        2. For each load step:
            1. Generate the mesh (if `no_meshing` is False).
            2. Define and apply boundary conditions.
            3. Solve the elastic problem for controlled and prescribed boundary conditions.
            4. Compute SIFs for both controlled and prescribed problems.
            5. Determine the crack propagation angle and load factor.
            6. Post-process results (e.g., compute energies, reaction forces, displacements).
            g. Export results to VTK and CSV files.
        3. Clean up and finalize the simulation.

        Note:
            - The `no_meshing` flag can be set to skip mesh generation if a mesh already exists.
            - The `no_propagation` flag can be set to skip crack propagation and use arbitrary load factor/crack propagation angle.
        """
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
            "phi": self.phi0,
            "lambda": self.l0,
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
            controlled_bcs = BoundaryConditions(
                displacement_bcs=self.define_controlled_displacements(),
                force_bcs=self.define_controlled_forces(),
                body_forces=self.define_controlled_body_forces(),
                locked_points=self.define_locked_points(),
                nodal_displacements=self.define_nodal_displacements(),
            )

            # Get the controlled boundary conditions
            prescribed_bcs = BoundaryConditions(
                displacement_bcs=self.define_prescribed_displacements(),
                force_bcs=self.define_prescribed_forces(),
                body_forces=self.define_prescribed_body_forces(),
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
            u_controlled = solve_elastic_problem(self.domain, model, controlled_bcs)

            print(f"│  Compute the SIFs for the controlled problem ({self.sif_method})")
            # Compute the SIFs for the controlled problem
            SIFs_controlled = compute_SIFs(
                self.domain,
                model,
                u_controlled,
                crack_points[-1],
                phi0,
                self.R_int,
                self.R_ext,
                self.sif_method,
            )

            # Tackle the prescribed problem
            if not prescribed_bcs.is_null():
                print("│  Solve the prescribed elastic problem with FEM")
                # Solve the prescribed elastic problem
                u_prescribed = solve_elastic_problem(self.domain, model, prescribed_bcs)
                # Compute the SIFs for the prescribed problem
                SIFs_prescribed = compute_SIFs(
                    self.domain,
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
                SIFs_prescribed = {key: 0.0 for key in SIFs_controlled}
                print("│  No prescribed BCs")

            # Compute the load factor and crack angle.
            print("│  Determination of propagation angle and load factor")
            if not self.no_propagation:
                load_factor_solver = LoadFactorSolver(model, self.Gc, crack_points[-1])
                opti_res = load_factor_solver.solve(
                    phi0, SIFs_controlled, SIFs_prescribed, self.s
                )
                # Get the results
                phi_ = opti_res[0]
                lambda_ = opti_res[1]
                # NOTE: DEBUG
                load_factor_solver.export_minimization_plots(
                    phi_,
                    lambda_,
                    phi0,
                    SIFs_controlled,
                    SIFs_prescribed,
                    self.s,
                    t,
                    dir_name,
                )
                # Add a new crack point
                da_vec = self.da * np.array([np.cos(phi_), np.sin(phi_), 0])
                crack_points.append(crack_points[-1] + da_vec)
            else:
                # Display a warning message
                print("│  Running in no propagation mode (set arbitrary results).")
                # Set arbitrary results
                phi_ = 0
                lambda_ = 1

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
            fimp = compute_measured_forces(self.domain, model, u_scaled, self)
            uimp = compute_measured_displacement(self.domain, u_scaled, self)
            # COmpute energies
            elastic_energy = compute_elastic_energy(self.domain, model, u_scaled)
            external_work = compute_external_work(self.domain, model, u_scaled)

            print("│  Export the results")
            # Export the elastic solution
            export_function(u_scaled, t, dir_name)
            # Store the results
            res["t"] = t
            res["a"] += self.da
            res["phi"] = phi_
            res["lambda"] = lambda_
            res["xc_1"] = crack_points[-1][0]
            res["xc_2"] = crack_points[-1][1]
            res["xc_3"] = crack_points[-1][2]
            for comp, uimp_comp in enumerate(uimp):
                res[f"uimp_{comp + 1}"] = uimp[comp]
            for comp, fimp_comp in enumerate(fimp):
                res[f"fimp_{comp + 1}"] = fimp[comp]
            for sif_name in SIFs_controlled:
                res[sif_name] = (
                    lambda_ * SIFs_controlled[sif_name] + SIFs_prescribed[sif_name]
                )
            res["elastic_energy"] = elastic_energy
            res["fracture_dissipation"] += self.da * self.Gc(np.array([phi_]))[0]
            res["external_work"] = external_work
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
