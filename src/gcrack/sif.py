import numpy as np

import ufl
import dolfinx
from dolfinx import fem, default_scalar_type

from domain import Domain
from models import ElasticModel
from utils.geometry import distance_point_to_segment
from utils.williams_series import Gamma_I, Gamma_II


def compute_theta_field(domain, crack_tip, R_int, R_ext):
    # Define the distance to the crack tip
    def distance_to_crack_tip(x):
        return np.sqrt((x[0] - crack_tip[0]) ** 2 + (x[1] - crack_tip[1]) ** 2)

    # Define the variational problem to define theta
    V_theta = fem.functionspace(domain.mesh, ("Lagrange", 1))
    theta, theta_ = ufl.TrialFunction(V_theta), ufl.TestFunction(V_theta)
    a = ufl.dot(ufl.grad(theta), ufl.grad(theta_)) * ufl.dx
    L = (
        fem.Constant(domain.mesh, default_scalar_type(0.0))
        * theta_
        * ufl.dx(domain=domain.mesh)
    )
    # Set the boundary conditions
    # Imposing 1 in the inner circle and zero in the outer circle
    dofs_inner = fem.locate_dofs_geometrical(
        V_theta, lambda x: distance_to_crack_tip(x) <= R_int
    )
    dofs_out = fem.locate_dofs_geometrical(
        V_theta, lambda x: distance_to_crack_tip(x) >= R_ext
    )
    bc_inner = fem.dirichletbc(default_scalar_type(1.0), dofs_inner, V_theta)
    bc_out = fem.dirichletbc(default_scalar_type(0.0), dofs_out, V_theta)
    bcs = [bc_out, bc_inner]
    # Solve the problem
    problem = fem.petsc.LinearProblem(a, L, bcs=bcs)
    return problem.solve()


def compute_auxiliary_displacement_field(
    domain: Domain,
    model: ElasticModel,
    xc: np.ndarray,
    phi0: float,
    K_I_aux: float,
    K_II_aux: float,
):
    # Get the cartesian coordinates
    x = ufl.SpatialCoordinate(domain.mesh)
    x_tip = ufl.as_vector(xc[:2])
    # Translate the domain to set the crack tip as origin
    r_vec_init = x - x_tip
    # Rotate the spatial coordinates to match the crack direction
    R = ufl.as_tensor([[ufl.cos(phi0), -ufl.sin(phi0)], [ufl.sin(phi0), ufl.cos(phi0)]])
    r_vec = ufl.transpose(R) * r_vec_init
    # Get the polar coordinates
    r = ufl.sqrt(ufl.dot(r_vec, r_vec))
    theta = ufl.atan2(r_vec[1], r_vec[0])
    # Get the elastic parameters
    mu = model.mu
    # Get kappa
    ka = model.ka
    # Compute the functions f
    f_I, f_II = [0, 0], [0, 0]
    f_I[0] = (ka - 1 + 2 * ufl.sin(theta / 2) ** 2) * ufl.cos(theta / 2)
    f_I[1] = (ka + 1 - 2 * ufl.cos(theta / 2) ** 2) * ufl.sin(theta / 2)
    f_II[0] = (ka + 1 + 2 * ufl.cos(theta / 2) ** 2) * ufl.sin(theta / 2)
    f_II[1] = -(ka - 1 - 2 * ufl.sin(theta / 2) ** 2) * ufl.cos(theta / 2)
    # Compute the displacement field
    ui = (
        ufl.sqrt(r / (2 * np.pi))
        / (2 * mu)
        * ufl.as_vector(
            [
                K_I_aux * f_I[0] + K_II_aux * f_II[0],
                K_I_aux * f_I[1] + K_II_aux * f_II[1],
            ]
        )
    )
    # Rotate the displacement vectors
    return R * ui


def compute_I_integral(
    domain: Domain,
    model: ElasticModel,
    u: fem.Function,
    u_aux: fem.Function,
    theta: fem.Function,
):
    # Compute the gradients
    grad_u = ufl.grad(u)
    grad_u_aux = ufl.grad(u_aux)
    # Compute the strains
    eps = ufl.sym(grad_u)
    eps_aux = ufl.sym(grad_u_aux)
    # Compute the stresses
    sig = model.sig(u)
    sig_aux = model.sig(u_aux)
    # Compute theta gradient and div
    div_theta = ufl.div(theta)
    grad_theta = ufl.grad(theta)
    # Compute the terms of the interaction integral
    dx = ufl.dx(domain=domain.mesh)
    Iw12 = 1 / 2 * ufl.inner(sig, eps_aux) * div_theta * dx
    Iw21 = 1 / 2 * ufl.inner(sig_aux, eps) * div_theta * dx
    Ig12 = ufl.inner(sig, grad_u_aux * grad_theta) * dx
    Ig21 = ufl.inner(sig_aux, grad_u * grad_theta) * dx
    # Compute the interaction integral expression
    I_expr = Ig12 + Ig21 - Iw12 - Iw21
    return fem.assemble_scalar(fem.form(I_expr))


def compute_SIFs_with_I_integral(
    domain: Domain,
    model: ElasticModel,
    u: fem.Function,
    xc: np.ndarray,
    phi0: float,
    R_int: float,
    R_ext: float,
):
    # Get the theta field
    theta_field = compute_theta_field(domain, xc, R_int, R_ext)
    theta = ufl.as_vector([ufl.cos(phi0), ufl.sin(phi0)]) * theta_field
    # # DEBUG
    # from dolfinx import io
    # new_file = io.VTKFile(theta_field.function_space.mesh.comm, "theta.pvd", "w")
    # new_file.write_function(theta_field, 0)
    # new_file.close()
    # input("Press enter to continue.")
    # # END DEBUG
    # Compute auxialiary displacement fields
    u_I_aux = compute_auxiliary_displacement_field(
        domain, model, xc, phi0, K_I_aux=1, K_II_aux=0
    )
    u_II_aux = compute_auxiliary_displacement_field(
        domain, model, xc, phi0, K_I_aux=0, K_II_aux=1
    )

    # # DEBUG
    # from dolfinx import io
    # V = u.function_space
    # uIaux = fem.Function(V, dtype=default_scalar_type)
    # uIaux.interpolate(fem.Expression(u_I_aux,V.element.interpolation_points()))
    # vtkfile = io.VTKFile(V.mesh.comm, "u_I_aux.pvd", "w")
    # vtkfile.write_function(uIaux, 0)
    # vtkfile.close()
    # input("Press enter to continue.")
    # # END DEBUG

    # Compute the I-integrals
    I_I = compute_I_integral(domain, model, u, u_I_aux, theta)
    I_II = compute_I_integral(domain, model, u, u_II_aux, theta)
    # Compute the SIF vector
    K_I = model.Ep / 2 * I_I
    K_II = model.Ep / 2 * I_II
    # Return SIF array
    return np.array([K_I, K_II])


def compute_SIFs_from_William_series_interpolation(
    domain: Domain,
    model: ElasticModel,
    u: fem.Function,
    xc: np.ndarray,
    phi0: float,
    R_int: float,
    R_ext: float,
):
    ### Extract x and u in the pacman from the FEM results
    def in_pacman(x):
        xc1 = np.array(xc)
        # Center coordinate on crack tip
        dx = x - xc1[:, np.newaxis]
        # Compute the distance to crack tip
        r = np.linalg.norm(dx, axis=0)
        # Keep the elements in the external radius
        in_pacman = r < R_ext
        # Remove the nodes that are too close to crack line
        xc2 = xc1 + R_ext * np.array([np.cos(np.pi + phi0), np.sin(np.pi + phi0), 0])
        far_from_crack = distance_point_to_segment(x, xc1, xc2) > R_int
        return np.logical_and(in_pacman, far_from_crack)

    # Get the entity ids
    entities_ids = dolfinx.mesh.locate_entities(domain.mesh, 2, in_pacman)
    # Get the dof of each element
    dof_ids = dolfinx.mesh.entities_to_geometry(domain.mesh, 2, entities_ids)
    # Generate the list of nodes (without any duplicated nodes)
    dof_unique_ids = np.unique(dof_ids.flatten())
    # Get the node coordinates (and set the crack tip as the origin)
    xs = domain.mesh.geometry.x[dof_unique_ids] - np.array(xc)[np.newaxis, :]

    # Get the displacement values
    us = np.empty((xs.shape[0], 2))
    us[:, 0] = u.x.array[2 * dof_unique_ids]
    us[:, 1] = u.x.array[2 * dof_unique_ids + 1]
    # Find crack tip element
    xs_all = domain.mesh.geometry.x - xc
    crack_tip_id = np.argmin(np.linalg.norm(xs_all, axis=1))
    # Remove crack tip motion
    us[:, 0] -= u.x.array[2 * crack_tip_id]
    us[:, 1] -= u.x.array[2 * crack_tip_id + 1]

    # # DEBUG Check the Williams series displacement fields
    # # Compute the theoretical displacement fields
    # K_I = 4e11 # 1e9
    # K_II = 0
    # zs = (xs[:, 0] + 1j * xs[:, 1]) * np.exp(-1j*phi0)
    # us_comp = K_I * Gamma_I(1, zs, model.mu, model.ka) + K_II * Gamma_II(
    #     1, zs, model.mu, model.ka
    # )
    # us_comp *= np.exp(1j*phi0)
    # us_williams = np.empty(us.shape)
    # us_williams[:, 0] = np.real(us_comp)
    # us_williams[:, 1] = np.imag(us_comp)

    # s = 0.01  # scale factor
    # plt.figure()
    # plt.scatter(xs[:, 0], xs[:, 1], marker=".", label="Initial pos")
    # plt.scatter(xs[:, 0] + s * us_williams[:, 0], xs[:, 1] + s * us_williams[:, 1], marker="s", label="Williams")
    # plt.scatter(xs[:, 0] + s * us[:, 0], xs[:, 1] + s * us[:, 1], marker="s", label="FEM")
    # plt.grid()
    # plt.legend()
    # plt.show()

    # Define the Williams series field
    N_min = -3
    N_max = 9

    # Get the complex coordinates around crack tip
    zs = xs[:, 0] + 1j * xs[:, 1]
    zs *= np.exp(-1j * phi0)
    # Compute the sizes
    Nn = us.shape[0]  # Number of nodes
    Ndof = us.shape[0] * us.shape[1]  # Number of dof
    xaxis = np.array([2 * n for n in range(Nn)])  # Mask to isolate x axis
    yaxis = np.array([2 * n + 1 for n in range(Nn)])  # Mask to isolate y axis
    # Get the displacement vector (from FEM)
    UF = us.flatten()
    # Get the Gamma matrix
    Gamma = np.empty((Ndof, 2 * (N_max - N_min + 1)))
    for i, n in enumerate(range(N_min, N_max + 1)):
        GI = Gamma_I(n, zs, model.mu, model.ka)
        GII = Gamma_II(n, zs, model.mu, model.ka)
        Gamma[xaxis, 2 * i] = np.real(GI)
        Gamma[yaxis, 2 * i] = np.imag(GI)
        Gamma[xaxis, 2 * i + 1] = np.real(GII)
        Gamma[yaxis, 2 * i + 1] = np.imag(GII)
    # Define the linear system
    GT_G = np.matmul(Gamma.T, Gamma)
    GT_UF = np.matmul(Gamma.T, UF)
    # Solve the linear system
    sol = np.linalg.solve(GT_G, GT_UF)
    # Extract KI and KII
    KI = sol[2 * (1 - N_min)]
    KII = sol[2 * (1 - N_min) + 1]
    # Extract KI and KII
    return np.array([KI, KII])


def compute_SIFs(
    domain: Domain,
    model: ElasticModel,
    u: fem.Function,
    xc: np.ndarray,
    phi0: float,
    R_int: float,
    R_ext: float,
    method: str,
):
    """
    Computes the Stress Intensity Factors (SIFs) for a given elastic model and
    displacement field using the specified method.

    Args:
        domain (Domain): The domain object representing the physical space in
            which the problem is defined.
        model (ElasticModel): The elastic model defining the material properties
            and behavior.
        u (fem.Function): The displacement field function obtained from the
            finite element method (FEM) analysis.
        xc (np.ndarray): A 1D array representing the coordinates of the crack tip.
        phi0 (float): The angle defining the crack orientation.
        R_int (float): The internal radius for the contour or region of interest.
        R_ext (float): The external radius for the contour or region of interest.
        method (str): The method used for calculating the SIFs. Can be either
            "i-integral" or "williams".

    Returns:
        tuple: A tuple containing the calculated Mode I and Mode II Stress
            Intensity Factors (K_I, K_II).
    """
    print(f"-- Calculation of the SIFs ({method})")
    match method.lower():
        case "i-integral":
            res = compute_SIFs_with_I_integral(
                domain,
                model,
                u,
                xc,
                phi0,
                R_int,
                R_ext,
            )
        case "williams":
            res = compute_SIFs_from_William_series_interpolation(
                domain,
                model,
                u,
                xc,
                phi0,
                R_int,
                R_ext,
            )

    # Display informations
    print(f"K_I  : {res[0]:.3g}")
    print(f"K_II : {res[1]:.3g}")
    return res
