from typing import List
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import gmsh

from gcrack.gcrack import GCrackBase
from gcrack.boundary_conditions import (
    BodyForce,
    NodalDisplacement,
    BoundaryConditions,
)
from gcrack.domain import Domain
from gcrack.models import ElasticModel
from gcrack.solvers import solve_elastic_problem


import dolfinx
from dolfinx import fem


class GCrack(GCrackBase):
    def generate_mesh(self, crack_points: List[jnp.ndarray]) -> gmsh.model:
        # Set the pacman size
        self.R_int = 0.5 * self.da
        self.R_ext = 1.0 * self.da
        # Clear existing model
        gmsh.clear()
        # Parameters
        b = pars["b"]
        # Coarse mesh
        h = b / 128
        # # Fine mesh
        # h = W / 128
        # h_min = self.R_int / 64
        # Points
        # Bot
        p0: int = gmsh.model.geo.addPoint(0, 0, 0, h)
        pb1: int = gmsh.model.geo.addPoint(-b, 0, 0, h)
        pb2: int = gmsh.model.geo.addPoint(b, 0, 0, h)
        # Lines
        lbb: int = gmsh.model.geo.addCircleArc(pb1, p0, pb2)
        lbu: int = gmsh.model.geo.addCircleArc(pb2, p0, pb1)

        # Surfaces
        # Bot
        clb: int = gmsh.model.geo.addCurveLoop([lbb, lbu])
        s1: int = gmsh.model.geo.addPlaneSurface([clb])

        # Boundaries
        self.boundaries = {
            "out": clb,
        }
        self.boundary_lines = {
            "out": [lbb, lbu],
        }
        # Physical groups
        # Domain
        domain: int = gmsh.model.addPhysicalGroup(2, [s1], tag=21)
        gmsh.model.setPhysicalName(2, domain, "domain")
        # Boundaries
        for boundary_name in self.boundaries:
            pg: int = gmsh.model.addPhysicalGroup(
                1,
                self.boundary_lines[boundary_name],
                tag=self.boundaries[boundary_name],
            )
            gmsh.model.setPhysicalName(1, pg, boundary_name)

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
        return [self.pars["b"], 0.0]

    def locate_measured_forces(self) -> int:
        """Define the boundary where the reaction force are measured.

        Returns:
            int: Identifier (id) of the boundary in GMSH.
        """
        return self.boundaries["out"]

    def define_nodal_displacements(self) -> List[NodalDisplacement]:
        """Define a list of imposed nodal displacements.

        Returns:
            List[NodalDisplacements]: A list of NodalDisplacement.
        """

        return [
            NodalDisplacement(x=[0.0, 0.0], u_imp=[0.0, 0.0]),
            NodalDisplacement(x=[pars["b"], 0.0], u_imp=[float("nan"), 0.0]),
        ]

    def define_controlled_body_forces(self) -> List[BodyForce]:
        """Define the controlled body forces that are affected by the load factor.

        Returns:
            List[BodyForce]: List of BodyForce (f_imp) where f_imp is the force vector.
        """
        # Get the parameters
        rho = self.pars["rho"]
        w = self.pars["w"]
        wd = self.pars["wd"]
        # Compute the polar coordinates
        r = "sqrt((x[0])**2 + x[1]**2)"
        th = "atan2(x[1], x[0])"
        # Calculate the amplitude of each forces
        fw_amp = f"({rho} * {w}**2 * {r})"
        fe_amp = f"(- {rho} * {wd} * {r})"
        # Project onto the x and y axis
        fx = f"{fw_amp} * (+cos({th})) + {fe_amp} * (-sin({th}))"
        fy = f"{fw_amp} * (+sin({th})) + {fe_amp} * (+cos({th}))"
        return [BodyForce(f_imp=[fx, fy])]

    def Gc(self, phi):
        return self.pars["Gc"] * jnp.ones(phi.shape)


if __name__ == "__main__":
    pars = {
        # Geometry
        "b": 1.0,
        # Load
        "w": 1.0,
        "wd": 0.0,
        "rho": 1.0,
        # Fracture (not used here)
        "Gc": 1.0,
    }

    gcrack = GCrack(
        # Mechanics
        assumption_2D="plane_stress",
        E=1,
        nu=0.3,
        # Initial crack
        xc0=[0, pars["b"] / 2, 0],
        # Propagation
        da=pars["b"] / 64,
        Nt=1,
        # Fracture
        sif_method="williams",
        # Other parameters
        pars=pars,
        name="numeric_centrifugal",
        no_propagation=True,
    )

    # Initialize GMSH
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Disable terminal output
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    # Initialize export directory
    dir_name = Path("results_" + gcrack.name)
    dir_name.mkdir(parents=True, exist_ok=True)
    # Get the elastic parameters
    ela_pars = {
        "E": gcrack.E,
        "nu": gcrack.nu,
        "2D_assumption": gcrack.assumption_2D,
    }
    # Initialize the crack points
    crack_points = [gcrack.xc0]
    print("│  Meshing the cracked domain")
    gmsh_model = gcrack.generate_mesh(crack_points)

    # Get the controlled boundary conditions
    bcs = BoundaryConditions(
        displacement_bcs=gcrack.define_controlled_displacements(),
        force_bcs=gcrack.define_controlled_forces(),
        body_forces=gcrack.define_controlled_body_forces(),
        locked_points=gcrack.define_locked_points(),
        nodal_displacements=gcrack.define_nodal_displacements(),
    )

    # Define the domain
    domain = Domain(gmsh_model)

    # Define an elastic model
    model = ElasticModel(ela_pars, domain)

    print("│  Solve the controlled elastic problem with FEM")
    # Solve the controlled elastic problem
    u = solve_elastic_problem(domain, model, bcs)

    print("│  Extract the stress field along a radial line")
    # Compute the stress field along a line
    sig_ufl = model.sig(u)
    # Generate FEM space for stress
    shape = sig_ufl.ufl_shape
    V_sig = fem.functionspace(domain.mesh, ("DG", 0, shape))
    # Convert the stress into an expression
    sig_expr = fem.Expression(sig_ufl, V_sig.element.interpolation_points)
    # Set the stress function
    sig = fem.Function(V_sig, name="Stress")
    sig.interpolate(sig_expr)
    # Evaluate along a line
    bb_tree = dolfinx.geometry.bb_tree(domain.mesh, domain.mesh.topology.dim)
    rs = np.linspace(0, pars["b"], num=64)
    points = np.array([[r, 0.0, 0.0] for r in rs])
    potential_colliding_cells = dolfinx.geometry.compute_collisions_points(
        bb_tree, points
    )
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        domain.mesh, potential_colliding_cells, points
    )
    cells = [colliding_cells.links(i)[0] for i, _ in enumerate(points)]
    cells = np.array(cells, dtype=np.int32)
    sig_line = sig.eval(points, cells)
    # Exctract the components
    sig_rr_num = sig_line[:, 0]
    sig_tt_num = sig_line[:, 4]

    # Compute the analytical solution
    sig_rr_ana = (
        pars["rho"]
        * pars["w"] ** 2
        * (3 + ela_pars["nu"])
        / 8
        * (pars["b"] ** 2 - rs**2)
    )
    sig_tt_ana = (
        pars["rho"] * pars["w"] ** 2 * (3 + ela_pars["nu"]) / 8 * pars["b"] ** 2
        - (1 + 3 * ela_pars["nu"]) / 8 * rs**2
    )

    # Display the comparison
    plt.figure()
    plt.xlabel(r"Radial coordinate $r$")
    plt.ylabel(r"Radial stress component $\sigma_{rr}$")
    plt.scatter(rs, sig_rr_ana, marker="x", label="Analytic")
    plt.scatter(rs, sig_rr_num, marker="+", label="Numeric")
    plt.grid()
    plt.legend()
    plt.savefig("comparison_num_ana_radial_stress.svg")

    plt.figure()
    plt.xlabel(r"Radial coordinate $r$")
    plt.ylabel(r"Orthoradial stress component $\sigma_{\theta \theta}$")
    plt.scatter(rs, sig_tt_ana, marker="x", label="Analytic")
    plt.scatter(rs, sig_tt_num, marker="+", label="Numeric")
    plt.grid()
    plt.legend()
    plt.savefig("comparison_num_ana_orthoradial_stress.svg")

    plt.show()

    # Clean up
    gmsh.finalize()
