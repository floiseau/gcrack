import logging

import numpy as np

import ufl
from dolfinx import fem, default_scalar_type

from domain import Domain
from models import ElasticModel


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
    nu = model.nu
    match model.assumption:
        case "plane_stress":
            ka = (3 - nu) / (1 + nu)
        case "plane_strain":
            ka = 3 - 4 * nu
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


def compute_SIFs(
    domain: Domain,
    model: ElasticModel,
    u: fem.Function,
    xc: np.ndarray,
    phi0: float,
    R_int: float,
    R_ext: float,
):
    logging.info("-- Calculation of the SIFs")
    # Get the theta field
    theta_field = compute_theta_field(domain, xc, R_int, R_ext)
    theta = ufl.as_vector([ufl.cos(phi0), ufl.sin(phi0)]) * theta_field
    # DEBUG
    # new_file = io.VTKFile(theta_field.function_space.mesh.comm, "theta.pvd", "w")
    # new_file.write_function(theta_field, 0)
    # new_file.close()
    # END DEBUG
    # Compute auxialiary displacement fields
    u_I_aux = compute_auxiliary_displacement_field(
        domain, model, xc, phi0, K_I_aux=1, K_II_aux=0
    )
    u_II_aux = compute_auxiliary_displacement_field(
        domain, model, xc, phi0, K_I_aux=0, K_II_aux=1
    )
    # Compute the I-integrals
    I_I = compute_I_integral(domain, model, u, u_I_aux, theta)
    I_II = compute_I_integral(domain, model, u, u_II_aux, theta)
    # Compute the SIF vector
    K_I = model.Ep / 2 * I_I
    K_II = model.Ep / 2 * I_II
    # Display informations
    logging.info(f"K_I  : {K_I:.3g}")
    logging.info(f"K_II : {K_II:.3g}")
    # Return SIF array
    return np.array([K_I, K_II])
