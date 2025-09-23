from typing import List
from pathlib import Path

import numpy as np

import gmsh
from gcrack import GCrackBase
from gcrack.domain import Domain
from gcrack.models import ElasticModel
from gcrack.boundary_conditions import DisplacementBC, BodyForce, BoundaryConditions
from gcrack.solvers import solve_elastic_problem
from gcrack.exporters import export_function


class GCrackData(GCrackBase):
    def generate_mesh(self, crack_points: List[np.ndarray]) -> gmsh.model:
        # Clear existing model
        gmsh.clear()
        # Parameters
        R = self.pars["R"]
        h = R / 128
        # Points
        p1: int = gmsh.model.geo.addPoint(-R, 0, 0, h)
        p2: int = gmsh.model.geo.addPoint(0, 0, 0, h)
        p3: int = gmsh.model.geo.addPoint(R, 0, 0, h)

        # Lines
        l1: int = gmsh.model.geo.addCircleArc(p1, p2, p3)
        l2: int = gmsh.model.geo.addCircleArc(p3, p2, p1)

        # Surface
        cl1: int = gmsh.model.geo.addCurveLoop([l1, l2])
        s1: int = gmsh.model.geo.addPlaneSurface([cl1])

        # Boundaries
        self.boundaries = {"all": 10}
        # Physical groups
        # Domain
        domain: int = gmsh.model.addPhysicalGroup(2, [s1], tag=1)
        gmsh.model.setPhysicalName(2, domain, "domain")
        # Border
        all: int = gmsh.model.addPhysicalGroup(1, [l1, l2], tag=self.boundaries["all"])
        gmsh.model.setPhysicalName(1, all, "all")

        # Generate the mesh
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

        # # Display and exit for debug purposes
        # # Synchronize the model
        # gmsh.model.geo.synchronize()
        # # Display the GMSH window
        # gmsh.fltk.run()
        # exit()

        # Return the model
        return gmsh.model()

    def locate_measured_displacement(self) -> List[float]:
        """Define the point where the displacement is measured.

        Returns:
            List: Coordinate of the point where the displacement is measured
        """
        return [0, 0]

    def locate_measured_forces(self) -> int:
        """Define the boundary where the reaction force are measured.

        Returns:
            int: Identifier (id) of the boundary in GMSH.
        """
        return self.boundaries["all"]

    def define_controlled_displacements(self) -> List[DisplacementBC]:
        """Define the imposed displacement boundary conditions.

        Returns:
            List[DisplacementBC]: List of DisplacementBC(boundary_id, u_imp) where boundary_id is the boundary id (int number) in GMSH, and u_imp is the displacement vector (componements can be nan to let it free).
        """
        return [DisplacementBC(boundary_id=self.boundaries["all"], u_imp=[0.0])]

    def define_controlled_body_forces(self) -> List[BodyForce]:
        """Define the controlled body forces that are affected by the load factor.

        Returns:
            List[BodyForce]: List of BodyForce (f_imp) where f_imp is the force vector.
        """
        rho = self.pars["rho"]
        g = self.pars["g"]
        return [BodyForce(f_imp=[-rho * g])]

    def Gc(self, phi):
        return 1 + 0 * phi
        # In plotter: 1 + (2 - 1) * sqrt(1 / 2 * (1 - cos(2 * (phi - pi/6))))


if __name__ == "__main__":
    # Define user parameters
    pars = {"R": 1.0, "rho": 1.0, "g": 9.81}

    gcrack_data = GCrackData(
        E=1.0,
        nu=0.3,
        da=0.0,
        Nt=1,
        xc0=[0, 0, 0],
        assumption_2D="anti_plane",
        pars=pars,
        sif_method="williams",  # "i-integral" "willliams"
        no_propagation=True,
    )

    # Initialize GMSH
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Disable terminal output
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    print("│  Meshing the cracked domain")
    gmsh_model = gcrack_data.generate_mesh([])

    # Get the controlled boundary conditions
    controlled_bcs = BoundaryConditions(
        displacement_bcs=gcrack_data.define_controlled_displacements(),
        force_bcs=gcrack_data.define_controlled_forces(),
        body_forces=gcrack_data.define_controlled_body_forces(),
        locked_points=gcrack_data.define_locked_points(),
    )

    # Get the controlled boundary conditions
    prescribed_bcs = BoundaryConditions(
        displacement_bcs=gcrack_data.define_prescribed_displacements(),
        force_bcs=gcrack_data.define_prescribed_forces(),
        body_forces=gcrack_data.define_prescribed_body_forces(),
        locked_points=gcrack_data.define_locked_points(),
    )

    # Define the domain
    domain = Domain(gmsh_model)
    # Get the elastic parameters
    ela_pars = {
        "E": gcrack_data.E,
        "nu": gcrack_data.nu,
        "2D_assumption": gcrack_data.assumption_2D,
    }
    # Define an elastic model
    model = ElasticModel(ela_pars, domain)

    print("│  Solve the controlled elastic problem with FEM")
    # Solve the controlled elastic problem
    u = solve_elastic_problem(domain, model, controlled_bcs)
    print("│  Export the results")
    # Export the elastic solution
    export_function(u, 1, Path("./"))
