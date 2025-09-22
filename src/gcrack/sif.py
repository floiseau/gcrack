import numpy as np
import ufl
import dolfinx
from dolfinx import fem
from gcrack.domain import Domain
from gcrack.models import ElasticModel
from gcrack.utils.geometry import distance_point_to_segment
from gcrack.utils.williams_series import Gamma_I, Gamma_II, Gamma_III


def compute_theta_field(domain, crack_tip, R_int, R_ext):
    # Get the cartesian coordinates
    x = ufl.SpatialCoordinate(domain.mesh)
    # Get the crack tip
    x_tip = ufl.as_vector(crack_tip[:2])
    # Get the polar coordinates
    r = ufl.sqrt(ufl.dot(x - x_tip, x - x_tip))
    # Define the ufl expression of the theta field
    theta_temp = (R_ext - r) / (R_ext - R_int)
    # Clip the value and return
    return ufl.max_value(0.0, ufl.min_value(theta_temp, 1.0))


def compute_auxiliary_displacement_field(
    domain: Domain,
    model: ElasticModel,
    xc: np.ndarray,
    phi0: float,
    K_I_aux: float = 0,
    K_II_aux: float = 0,
    K_III_aux: float = 0,
    T_aux: float = 0,
):
    # Get the cartesian coordinates
    x_2D = ufl.SpatialCoordinate(domain.mesh)
    x = ufl.as_vector([x_2D[0], x_2D[1], 0])
    x_tip = ufl.as_vector(xc)
    # Translate the domain to set the crack tip as origin
    r_vec_init = x - x_tip
    # Rotate the spatial coordinates to match the crack direction
    R = ufl.as_tensor(
        [
            [ufl.cos(phi0), -ufl.sin(phi0), 0.0],
            [ufl.sin(phi0), ufl.cos(phi0), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    r_vec = ufl.transpose(R) * r_vec_init
    # Get the polar coordinates
    r = ufl.sqrt(ufl.dot(r_vec, r_vec))
    theta = ufl.atan2(r_vec[1], r_vec[0])
    # Get the elastic parameters
    mu = model.mu
    # Get kappa
    ka = model.ka
    # Compute the functions f
    fx_I = ufl.cos(theta / 2) * (ka - 1 + 2 * ufl.sin(theta / 2) ** 2)
    fy_I = ufl.sin(theta / 2) * (ka + 1 - 2 * ufl.cos(theta / 2) ** 2)
    fx_II = ufl.sin(theta / 2) * (ka + 1 + 2 * ufl.cos(theta / 2) ** 2)
    fy_II = -ufl.cos(theta / 2) * (ka - 1 - 2 * ufl.sin(theta / 2) ** 2)
    fz_III = 4 * ufl.sin(theta / 2)
    # Introduce the factor u_fac
    u_fac = ufl.sqrt(r / (2 * np.pi)) / (2 * mu)
    # Compute the displacement field for mode I
    u_I = K_I_aux * u_fac * ufl.as_vector([fx_I, fy_I, 0])
    # Compute the displacement field for mode II
    u_II = K_II_aux * u_fac * ufl.as_vector([fx_II, fy_II, 0])
    u_III = K_III_aux * u_fac * ufl.as_vector([0, 0, fz_III])
    # Compute the displacement field for mode T
    ux_T = (
        -1 / np.pi * (ka + 1) / (8 * mu) * ufl.ln(r)
        - 1 / np.pi * 1 / (4 * mu) * ufl.sin(theta) ** 2
    )
    uy_T = -1 / np.pi * (ka - 1) / (8 * mu) * theta + 1 / np.pi * 1 / (
        4 * mu
    ) * ufl.sin(theta) * ufl.cos(theta)
    u_T = T_aux * ufl.as_vector([ux_T, uy_T, 0])
    # Compute the total displacement field and rotate it
    u_tot = R * (u_I + u_II + u_III + u_T)
    # Rotate the displacement vectors
    return u_tot


def compute_I_integral(
    domain: Domain,
    model: ElasticModel,
    u: fem.Function,
    u_aux: ufl.core.expr.Expr,
    theta: ufl.core.expr.Expr,
) -> float:
    # Compute the gradients
    grad_u = model.grad_u(u)
    gua_2D = ufl.grad(u_aux)
    # Compute theta gradient and div
    div_theta = ufl.div(theta)
    gt_2D = ufl.grad(theta)
    # Convert the 2D gradient to 3D
    grad_u_aux = ufl.as_tensor(
        [
            [gua_2D[0, 0], gua_2D[0, 1], 0],
            [gua_2D[1, 0], gua_2D[1, 1], 0],
            [gua_2D[2, 0], gua_2D[2, 1], 0],
        ]
    )
    grad_theta = ufl.as_tensor(
        [
            [gt_2D[0, 0], gt_2D[0, 1], 0],
            [gt_2D[1, 0], gt_2D[1, 1], 0],
            [0, 0, 0],
        ]
    )
    # Compute the strains
    eps = model.eps(u)
    eps_aux = ufl.sym(grad_u_aux)
    # Compute the stresses
    sig = model.sig(u)
    sig_aux = model.la * ufl.tr(eps_aux) * ufl.Identity(3) + 2 * model.mu * eps_aux
    # Compute the interaction integral (reduce the quadrature degree for faster evaluation)
    dx = ufl.Measure("dx", domain=domain.mesh, metadata={"quadrature_degree": 4})
    I_expr = (
        ufl.inner(sig, grad_u_aux * grad_theta)
        + ufl.inner(sig_aux, grad_u * grad_theta)
        - 1.0 / 2.0 * ufl.inner(sig, eps_aux) * div_theta
        - 1.0 / 2.0 * ufl.inner(sig_aux, eps) * div_theta
    ) * dx
    I_integral = fem.assemble_scalar(fem.form(I_expr))
    return I_integral


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
    theta = theta_field * ufl.as_vector([ufl.cos(phi0), ufl.sin(phi0)])
    # Compute auxiliary displacement fields
    u_I_aux = compute_auxiliary_displacement_field(
        domain, model, xc, phi0, K_I_aux=1.0, K_II_aux=0.0, K_III_aux=0.0, T_aux=0.0
    )
    u_II_aux = compute_auxiliary_displacement_field(
        domain, model, xc, phi0, K_I_aux=0.0, K_II_aux=1.0, K_III_aux=0.0, T_aux=0.0
    )
    u_III_aux = compute_auxiliary_displacement_field(
        domain, model, xc, phi0, K_I_aux=0.0, K_II_aux=0.0, K_III_aux=1.0, T_aux=0.0
    )
    u_T_aux = compute_auxiliary_displacement_field(
        domain, model, xc, phi0, K_I_aux=0.0, K_II_aux=0.0, K_III_aux=0.0, T_aux=1.0
    )
    # Compute the I-integrals
    I_I = compute_I_integral(domain, model, u, u_I_aux, theta)
    I_II = compute_I_integral(domain, model, u, u_II_aux, theta)
    I_III = compute_I_integral(domain, model, u, u_III_aux, theta)
    I_T = compute_I_integral(domain, model, u, u_T_aux, theta)
    # Compute the SIF
    K_I = model.Ep / 2 * I_I
    K_II = model.Ep / 2 * I_II
    K_III = model.E / (2 * (1 + model.nu)) * I_III
    T = model.Ep * I_T
    # Return SIF array
    return {"KI": K_I, "KII": K_II, "KIII": K_III, "T": T}


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

    # Construct the displacement vector
    N_comp = u.function_space.value_shape[0]
    us = np.empty((xs.shape[0], N_comp))
    # Get the displacements
    if model.assumption.startswith("plane"):
        # Get the displacement values
        us[:, 0] = u.x.array[2 * dof_unique_ids]
        us[:, 1] = u.x.array[2 * dof_unique_ids + 1]
        # Find crack tip element
        xs_all = domain.mesh.geometry.x - xc
        crack_tip_id = np.argmin(np.linalg.norm(xs_all, axis=1))
        # Remove crack tip motion
        us[:, 0] -= u.x.array[2 * crack_tip_id]
        us[:, 1] -= u.x.array[2 * crack_tip_id + 1]
    elif model.assumption == "anti_plane":
        # Get the displacement values
        us[:, 0] = u.x.array[dof_unique_ids]

    # Define the Williams series field
    N_min = -1  # -3
    N_max = 7  # 9

    # Get the complex coordinates around crack tip
    zs = xs[:, 0] + 1j * xs[:, 1]
    zs *= np.exp(-1j * phi0)
    # Compute the sizes
    Nn = us.shape[0]  # Number of nodes
    Ndof = us.shape[0] * us.shape[1]  # Number of dof

    # Construct the matrix Gamma
    if model.assumption.startswith("plane"):
        xaxis = np.array([2 * n for n in range(Nn)])  # Mask to isolate x axis
        yaxis = np.array([2 * n + 1 for n in range(Nn)])  # Mask to isolate y axis
        # Get the displacement vector (from FEM)
        UF = us.flatten()
        # Get the Gamma matrix
        Gamma = np.empty((Ndof, 2 * (N_max - N_min + 1)))
        for i, n in enumerate(range(N_min, N_max + 1)):
            GI = Gamma_I(n, zs, model.mu, model.ka) * np.exp(1j * phi0)
            GII = Gamma_II(n, zs, model.mu, model.ka) * np.exp(1j * phi0)
            Gamma[xaxis, 2 * i] = np.real(GI)
            Gamma[yaxis, 2 * i] = np.imag(GI)
            Gamma[xaxis, 2 * i + 1] = np.real(GII)
            Gamma[yaxis, 2 * i + 1] = np.imag(GII)
    elif model.assumption == "anti_plane":
        # Get the displacement vector (from FEM)
        UF = us.flatten()
        # Get the Gamma matrix
        Gamma = np.empty((Ndof, 2 * (N_max - N_min + 1)))
        for i, n in enumerate(range(N_min, N_max + 1)):
            GIII = Gamma_III(n, zs, model.mu, model.ka)
            Gamma[:, i] = GIII
    # Solve the least square problem
    sol, res, _, _ = np.linalg.lstsq(Gamma, UF)
    # Create the SIF dictionary
    SIFs = {}
    if model.assumption.startswith("plane"):
        # Extract KI, KII and T
        SIFs["KI"] = sol[2 * (1 - N_min)]
        SIFs["KII"] = sol[2 * (1 - N_min) + 1]
        SIFs["KIII"] = 0
        SIFs["T"] = 2 * np.sqrt(2) / np.sqrt(np.pi) * sol[2 * (2 - N_min)]
        # Store the other coefficients of the seriess
        for i, n in enumerate(range(N_min, N_max + 1)):
            SIFs[f"aI_{n}"] = sol[2 * i]
            SIFs[f"aII_{n}"] = sol[2 * i + 1]
    elif model.assumption == "anti_plane":
        SIFs["KI"] = 0
        SIFs["KII"] = 0
        SIFs["KIII"] = sol[0 - N_min]
        SIFs["T"] = 0
        # Store the other coefficients of the seriess
        for i, n in enumerate(range(N_min, N_max + 1)):
            SIFs[f"aIII_{n}"] = sol[i]
    # Return the SIFs
    return SIFs


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
        dict: A dict containing the calculated Stress Intensity Factors.
    """
    # Compute the SIFs
    match method.lower():
        case "i-integral":
            SIFs = compute_SIFs_with_I_integral(
                domain,
                model,
                u,
                xc,
                phi0,
                R_int,
                R_ext,
            )
        case "williams":
            SIFs = compute_SIFs_from_William_series_interpolation(
                domain,
                model,
                u,
                xc,
                phi0,
                R_int,
                R_ext,
            )
        case _:
            raise NotImplementedError(
                f"SIF method '{method}' is not implemented. Existing methods are: 'I-integral' and 'Williams'."
            )

    # Display informations
    for name, val in SIFs.items():
        print(f"│  │  {name: <3}: {val:.3g}")
    print("│  │  End of SIF calculations")
    return SIFs
