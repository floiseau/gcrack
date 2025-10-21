from typing import List
import itertools

import gmsh
import numpy as np
import dolfinx
import ufl

from gcrack import GCrackBase
from gcrack.domain import Domain
from gcrack.models import ElasticModel
from gcrack.sif import compute_SIFs


class GCrackData(GCrackBase):
    def generate_mesh(self, crack_points: List[np.ndarray]) -> gmsh.model:
        # Clear existing model
        gmsh.clear()
        # Parameters
        L = 1
        h = L / 128
        h_min = self.R_int / 64
        # Points
        # Bot
        p1: int = gmsh.model.geo.addPoint(-L / 2, -L / 2, 0, h)
        p2: int = gmsh.model.geo.addPoint(L / 2, -L / 2, 0, h)
        p3: int = gmsh.model.geo.addPoint(L / 2, 0, 0, h)  # Mid right node
        pc_bot: List[int] = []
        pc_top: List[int] = []
        for i, p in enumerate(reversed(crack_points)):
            # The crack tip is shared
            if i == 0:
                pc_new: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
                pc_bot.append(pc_new)
                pc_top.append(pc_new)
            else:
                pc_new_bot: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
                pc_bot.append(pc_new_bot)
                pc_new_top: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
                pc_top.append(pc_new_top)
        p5: int = gmsh.model.geo.addPoint(-L / 2, 0, 0, h)  # Bot crack lip
        # Top
        p6: int = gmsh.model.geo.addPoint(-L / 2, L / 2, 0, h)
        p7: int = gmsh.model.geo.addPoint(L / 2, L / 2, 0, h)
        # Point(13) // Mid right node
        # Point(14) // Crack tip
        p8: int = gmsh.model.geo.addPoint(-L / 2, 0, 0, h)  # Top crack lip

        # Lines
        # Bot
        l1: int = gmsh.model.geo.addLine(p1, p2)
        l2: int = gmsh.model.geo.addLine(p2, p3)
        l3: int = gmsh.model.geo.addLine(p3, pc_bot[0])
        crack_lines_bot: List[int] = []
        for i in range(len(pc_bot) - 1):
            lb: int = gmsh.model.geo.addLine(pc_bot[i], pc_bot[i + 1])
            crack_lines_bot.append(lb)
        crack_lines_bot.append(gmsh.model.geo.addLine(pc_bot[-1], p5))
        l5: int = gmsh.model.geo.addLine(p5, p1)
        # Top
        l6: int = gmsh.model.geo.addLine(p6, p7)
        l7: int = gmsh.model.geo.addLine(p7, p3)
        # Line(13)
        # Top  crack line
        crack_lines_top: List[int] = []
        for i in range(len(pc_bot) - 1):
            lt: int = gmsh.model.geo.addLine(pc_top[i], pc_top[i + 1])
            crack_lines_top.append(lt)
        crack_lines_top.append(gmsh.model.geo.addLine(pc_top[-1], p8))
        l9: int = gmsh.model.geo.addLine(p8, p6)

        # Surfaces
        # Bot
        cl1: int = gmsh.model.geo.addCurveLoop([l1, l2, l3] + crack_lines_bot + [l5])
        s1: int = gmsh.model.geo.addPlaneSurface([cl1])
        # Top
        cl2: int = gmsh.model.geo.addCurveLoop([l6, l7, l3] + crack_lines_top + [l9])
        s2: int = gmsh.model.geo.addPlaneSurface([cl2])

        # Boundaries
        self.boundaries = {
            "bot": 11,
            "top": 12,
        }
        # Physical groups
        # Domain
        domain: int = gmsh.model.addPhysicalGroup(2, [s1, s2], tag=21)
        gmsh.model.setPhysicalName(2, domain, "domain")
        # Boundaries
        bot: int = gmsh.model.addPhysicalGroup(1, [l1], tag=self.boundaries["bot"])
        gmsh.model.setPhysicalName(1, bot, "bot")
        top: int = gmsh.model.addPhysicalGroup(1, [l6], tag=self.boundaries["top"])
        gmsh.model.setPhysicalName(1, top, "top")

        # Element size
        # Refine around the crack line
        field1: int = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field1, "PointsList", [pc_bot[0]])
        gmsh.model.mesh.field.setNumber(field1, "Sampling", 100)
        field2: int = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field2, "InField", field1)
        gmsh.model.mesh.field.setNumber(field2, "DistMin", 2 * self.R_ext)
        gmsh.model.mesh.field.setNumber(field2, "DistMax", 4 * self.R_ext)
        gmsh.model.mesh.field.setNumber(field2, "SizeMin", h_min)
        gmsh.model.mesh.field.setNumber(field2, "SizeMax", h)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.field.setAsBackgroundMesh(field2)
        gmsh.model.mesh.generate(2)

        # # Display and exit for debug purposes
        # # Synchronize the model
        # gmsh.model.geo.synchronize()
        # # Display the GMSH window
        # gmsh.fltk.run()
        # exit()

        # Return the model
        return gmsh.model()

    def locate_measured_displacement(self):
        return []

    def locate_measured_forces(self):
        return []

    def Gc(self, phi):
        return 0.0


def u_Ki(domain, model, KI, KII, KIII, T, phi0, model_assumption):
    # Initialize the displacement function
    if model_assumption.startswith("plane"):
        u_shape = (2,)
    elif model_assumption == "anti_plane":
        u_shape = (1,)
    V_u = dolfinx.fem.functionspace(domain.mesh, ("Lagrange", 1, u_shape))
    u = dolfinx.fem.Function(V_u, name="Displacement")
    # Define the rotation matrix
    R_phi0 = ufl.as_matrix(
        [[ufl.cos(phi0), -ufl.sin(phi0)], [ufl.sin(phi0), ufl.cos(phi0)]]
    )
    # Compute the KI displacement field
    x = ufl.SpatialCoordinate(domain.mesh)
    x_rot = ufl.dot(ufl.transpose(R_phi0), x)
    r = ufl.sqrt(ufl.dot(x_rot, x_rot))
    th = ufl.atan2(x_rot[1], x_rot[0])
    # Get the elastic parameters
    ka = model.ka
    mu = model.mu
    # Compute the factor
    u_fac = 1 / (2 * mu) * ufl.sqrt(r / (2 * np.pi))
    if model_assumption.startswith("plane"):
        # Compute the spatial functions
        fx_I = ufl.cos(th / 2) * (ka - 1 + 2 * ufl.sin(th / 2) ** 2)
        fy_I = ufl.sin(th / 2) * (ka + 1 - 2 * ufl.cos(th / 2) ** 2)
        fx_II = ufl.sin(th / 2) * (ka + 1 + 2 * ufl.cos(th / 2) ** 2)
        fy_II = -ufl.cos(th / 2) * (ka - 1 - 2 * ufl.sin(th / 2) ** 2)
        fx_T = (1 + ka) * ufl.cos(th)
        fy_T = (ka - 3) * ufl.sin(th)
        # Compute the terms of the displacement field
        ux_I = KI * u_fac * fx_I
        uy_I = KI * u_fac * fy_I
        ux_II = KII * u_fac * fx_II
        uy_II = KII * u_fac * fy_II
        ux_T = T / (8 * mu) * r * fx_T
        uy_T = T / (8 * mu) * r * fy_T
        # Define the expression of the displacement field
        u_ufl = ufl.as_vector([ux_I + ux_II + ux_T, uy_I + uy_II + uy_T])
        # Finish the rotation of the displacement field
        u_ufl_rotated = ufl.dot(R_phi0, u_ufl)
        u_expr = dolfinx.fem.Expression(
            u_ufl_rotated, V_u.element.interpolation_points()
        )
    elif model_assumption == "anti_plane":
        # Compute the spatial function
        fz_III = 4 * ufl.sin(th / 2)
        # Comupte the terms of the displacement field
        uz_III = KIII * u_fac * fz_III
        # Define the expression of the displacement field
        u_ufl = ufl.as_vector([uz_III])
        u_expr = dolfinx.fem.Expression(u_ufl, V_u.element.interpolation_points())
    # Interpolate on the displacement function
    u.interpolate(u_expr)
    # Return the displacement field
    return u


def test_compute_SIFs_2D_plane():
    # Define user parameters
    pars = {"L": 1.0}
    data = GCrackData(
        E=1.0,
        nu=0.3,
        da=pars["L"] / 32,
        Nt=1,
        xc0=[0, 0, 0],
        assumption_2D="plane_strain",
        pars=pars,
        sif_method="williams",  # NOTE: This is overwritten afterward.
        # sif_method="i-integral",
        s=0.0,
    )
    # Generate the domain
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Disable terminal output
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh_model = data.generate_mesh([data.xc0])
    domain = Domain(gmsh_model)
    # Define an elastic model
    ela_pars = {
        "E": data.E,
        "nu": data.nu,
        "2D_assumption": data.assumption_2D,
    }
    model = ElasticModel(ela_pars, domain)
    # # Parameters
    KIs = [0, 1]
    KIIs = [0, 1]
    Ts = [0, 1]
    phi0s = [0, np.deg2rad(30), np.deg2rad(45)]
    methods = ["williams"]  # ["i-integral"]
    # Test all combinations of parameters
    for KI, KII, T, phi0, method in itertools.product(KIs, KIIs, Ts, phi0s, methods):
        # Pass when all the SIFs are null
        if KI == 0 and KII == 0 and T == 0:
            continue
        print(f"{KI=}, {KII=}, {T=}, {phi0=}, {method=}")
        # Generate a displacement field
        u = u_Ki(domain, model, KI, KII, 0, T, phi0, data.assumption_2D)
        # # Debug : Plot the displacement field
        # plot_displacement_field(domain, u)
        # Compute the SIF associated with each displacament field
        data.phi0 = phi0
        SIF = compute_SIFs(
            domain, model, u, data.xc0, data.phi0, data.R_int, data.R_ext, method
        )
        assert np.isclose(SIF["KI"], KI, atol=1e-2), (
            f"Error in KI for : {KI=}, {KII=}, {T=}, {phi0=}, {method=}."
        )
        assert np.isclose(SIF["KII"], KII, atol=1e-2), (
            f"Error in KII for : {KI=}, {KII=}, {T=}, {phi0=}, {method=}."
        )
        assert np.isclose(SIF["T"], T, atol=1e-2), (
            f"Error in T for : {KI=}, {KII=}, {T=}, {phi0=}, {method=}."
        )

    gmsh.finalize()


def test_compute_SIFs_anti_plane():
    # Define user parameters
    pars = {"L": 1.0}
    data = GCrackData(
        E=1.0,
        nu=0.3,
        da=pars["L"] / 32,
        Nt=1,
        xc0=[0, 0, 0],
        assumption_2D="anti_plane",
        pars=pars,
        sif_method="williams",  # NOTE: This is not used here
        # sif_method="i-integral",
        s=0.0,
    )
    # Generate the domain
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Disable terminal output
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh_model = data.generate_mesh([data.xc0])
    domain = Domain(gmsh_model)
    # Define an elastic model
    ela_pars = {
        "E": data.E,
        "nu": data.nu,
        "2D_assumption": data.assumption_2D,
    }
    model = ElasticModel(ela_pars, domain)
    # # Parameters
    KI = 0
    KII = 0
    KIII = 1
    T = 0
    phi0s = [0, np.deg2rad(30), np.deg2rad(45)]
    methods = ["williams", "i-integral"]  # ["williams"]  # ["i-integral"]
    # Test all combinations of parameters
    for phi0, method in itertools.product(phi0s, methods):
        print(f"{KI=}, {KII=}, {KIII=}, {T=}, {phi0=}, {method=}")
        # Generate a displacement field
        u = u_Ki(domain, model, KI, KII, KIII, T, phi0, data.assumption_2D)
        # Compute the SIF associated with each displacament field
        data.phi0 = phi0
        SIF = compute_SIFs(
            domain, model, u, data.xc0, data.phi0, data.R_int, data.R_ext, method
        )
        assert np.isclose(SIF["KI"], KI, atol=1e-2), (
            f"Error in KI for : {KI=}, {KII=}, {KIII=}, {T=}, {phi0=}, {method=}."
        )
        assert np.isclose(SIF["KII"], KII, atol=1e-2), (
            f"Error in KII for : {KI=}, {KII=}, {KIII=}, {T=}, {phi0=}, {method=}."
        )
        assert np.isclose(SIF["KIII"], KIII, atol=1e-2), (
            f"Error in KII for : {KI=}, {KII=}, {KIII=}, {T=}, {phi0=}, {method=}."
        )
        assert np.isclose(SIF["T"], T, atol=1e-2), (
            f"Error in T for : {KI=}, {KII=}, {KIII=}, {T=}, {phi0=}, {method=}."
        )

    gmsh.finalize()


def plot_displacement_field(domain, u):
    # DEBUG Check the generated displacement fields
    import matplotlib.pyplot as plt

    # Get the initial coordinates
    x_ufl = ufl.SpatialCoordinate(domain.mesh)
    V_x = dolfinx.fem.functionspace(domain.mesh, ("Lagrange", 1, (2,)))
    x = dolfinx.fem.Function(V_x, name="Coordinates")
    x_expr = dolfinx.fem.Expression(x_ufl, V_x.element.interpolation_points())
    x.interpolate(x_expr)

    # Generate masks to extract the components
    N = int(len(u.x.array) / 2)
    x_comp = list(range(0, 2 * N, 2))
    y_comp = list(range(1, 2 * N + 1, 2))

    # Plot the initial domain and its deformed
    s = 0.1  # scale factor
    plt.figure()
    plt.scatter(x.x.array[x_comp], x.x.array[y_comp], marker=".", label="Initial pos")
    plt.scatter(
        x.x.array[x_comp] + s * u.x.array[x_comp],
        x.x.array[y_comp] + s * u.x.array[y_comp],
        marker="s",
        label="Generated field",
    )
    plt.grid()
    plt.legend()
    plt.axis("equal")
    plt.show()
    # END DEBUG
