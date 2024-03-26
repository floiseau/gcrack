import numpy as np

import ufl
from dolfinx import fem, default_scalar_type

from domain import Domain
from models import ElasticModel


def compute_theta_field(domain, crack_tip, R_int):
    # Set the parameters
    R_ext = 2 * R_int

    # Define the distance to the crack tip
    def distance_to_crack_tip(x):
        return np.sqrt((x[0] - crack_tip[0]) ** 2 + (x[1] - crack_tip[1]) ** 2)

    # Define the variational problem to define theta
    V_theta = fem.FunctionSpace(domain.mesh, ("Lagrange", 1))
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


def G(
    domain: Domain, u: fem.Function, xc, model: ElasticModel, R_int: float
) -> np.array:
    # Get the theta field
    theta_field = compute_theta_field(domain, xc, R_int)
    # Compute the energy release rate form
    eps = ufl.sym(ufl.grad(u))
    sig = model.la * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * model.mu * eps
    dx = ufl.dx(domain=domain.mesh)
    P = (
        1 / 2 * ufl.inner(sig, eps) * ufl.Identity(2)
        # - ufl.transpose(ufl.dot(sig, ufl.grad(u)))
        - ufl.dot(ufl.transpose(ufl.grad(u)), sig)
    )
    G_expr = -ufl.dot(P, ufl.grad(theta_field))
    G_1 = fem.assemble_scalar(fem.form(G_expr[0] * dx))
    G_2 = fem.assemble_scalar(fem.form(G_expr[1] * dx))
    # Initialize the
    return np.array([G_1, G_2])
