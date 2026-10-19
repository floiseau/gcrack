"""
Module for solving the optimization problem to determine the load factor in crack propagation simulations.

This module provides the `LoadFactorSolver` class, which implements the GMERR (Generalized Maximum Energy Release Rate) criterion for crack propagation.
The solver is based on the work of Amestoy and Leblond (1992) and is designed to work with the `gcrack.lefm` module.

References:
    Amestoy, M., & Leblond, J. B. (1992). A new numerical method for crack growth prediction.
    International Journal of Solids and Structures, 29(21), 2619-2638.
    https://doi.org/10.1016/0020-7683(92)90210-K
"""

from collections.abc import Callable
from typing import List


import jax.numpy as jnp
from jax import jit, grad, hessian, random
from scipy.optimize import root_scalar

from gcrack.models import ElasticModel
from gcrack.lefm import G_star, G_star_coupled

prng_key = random.key(0)


class LoadFactorSolver:
    """Solver for load factors and crack propagation angles using the GMERR criterion.

    This class implements the GMERR (Generalized Maximum Energy Release Rate) criterion for crack propagation.
    It uses automatic differentiation to compute gradients and Hessians of the objective function, and employs gradient descent with line search to find the optimal crack propagation angle and load factor.

    Attributes:
        model (gcrack.models.ElasticModel): The elastic model used for the simulation.
        Gc (Callable): The critical energy release rate function.
        grad (Callable): The gradient of the objective function.
        hess (Callable): The Hessian of the objective function.
        grad_pert (Callable): The gradient of the perturbed objective function.
        hess_pert (Callable): The Hessian of the perturbed objective function.
    """

    def __init__(self, model: ElasticModel, Gc_func: Callable, xc: List):
        """Initializes the LoadFactorSolver.

        Args:
            model (gcrack.models.ElasticModel): The elastic model used for the simulation.
            Gc_func (Callable): The critical energy release rate function.
            xc (List): Position of the crack tip.
        """
        # Store the model
        self.model = model
        # Store the crack tip position
        self.xc = xc
        # Set the critical energy release rate function
        self.Gc = jit(lambda phi: Gc_func(phi, xc))
        # Automatic differentiation of the objective function
        self.grad = jit(grad(self.objective))
        self.hess = jit(hessian(self.objective))
        # Automatic differentiation of the perturbed objective function
        self.grad_pert = jit(grad(self.objective_pert))
        self.hess_pert = jit(hessian(self.objective_pert))

    def objective(
        self,
        x: jnp.ndarray,
        Ep: float,
        s: float,
        KIc: float,
        KIIc: float,
        Tc: float,
        KIp: float,
        KIIp: float,
        Tp: float,
        phi0: float,
    ) -> float:
        """Computes the objective function for the GMERR criterion.

        This function computes the objective function for the GMERR criterion, which is used to find the optimal crack propagation angle and load factor.

        Args:
            x (jnp.ndarray): The optimization variables (crack propagation angle).
            Ep (float): Plane strain modulus.
            s (float): Internal length associated with T-stress.
            KIc (float): Mode I stress intensity factor for the controlled problem.
            KIIc (float): Mode II stress intensity factor for the controlled problem.
            Tc (float): T-stress for the controlled problem.
            KIp (float): Mode I stress intensity factor for the prescribed problem.
            KIIp (float): Mode II stress intensity factor for the prescribed problem.
            Tp (float): T-stress for the prescribed problem.
            phi0 (float): Initial crack angle.

        Returns:
            float: The value of the objective function.
        """
        # NOTE: The KIc (etc.) means controlled (not critical!)
        phi = x[0]
        # Compute the G star
        Gs_cc = G_star(phi, phi0, KIc, KIIc, Tc, Ep, s)
        Gs_cp = G_star_coupled(phi, phi0, KIc, KIIc, Tc, KIp, KIIp, Tp, Ep, s)
        Gs_pp = G_star(phi, phi0, KIp, KIIp, Tp, Ep, s)
        # Compute the Gc from phi
        gc = self.Gc(phi)
        # Compute and return the load factor
        delta = Gs_cp**2 - 4 * Gs_cc * (Gs_pp - gc)
        return (-Gs_cp + jnp.sqrt(delta)) / (2 * Gs_cc)

    def objective_pert(
        self,
        x: jnp.ndarray,
        Ep: float,
        s: float,
        KIc: float,
        KIIc: float,
        Tc: float,
        KIp: float,
        KIIp: float,
        Tp: float,
        phi0: float,
    ) -> float:
        """Computes the perturbed objective function for the GMERR criterion.

        This function adds a small perturbation to the objective function to avoid
        convergence to a maximum instead of a minimum.

        Args:
            x (jnp.ndarray): The optimization variables (crack propagation angle).
            Ep (float): Plane strain modulus.
            s (float): Internal length associated with T-stress.
            KIc (float): Mode I stress intensity factor for the controlled problem.
            KIIc (float): Mode II stress intensity factor for the controlled problem.
            Tc (float): T-stress for the controlled problem.
            KIp (float): Mode I stress intensity factor for the prescribed problem.
            KIIp (float): Mode II stress intensity factor for the prescribed problem.
            Tp (float): T-stress for the prescribed problem.
            phi0 (float): Initial crack angle.

        Returns:
            float: The value of the perturbed objective function.
        """
        return (
            self.objective(x, Ep, s, KIc, KIIc, Tc, KIp, KIIp, Tp, phi0) - 1e-5 * x[0]
        )

    def solve(
        self, phi0: float, SIFs_controlled: dict, SIFs_prescribed: dict, s: float
    ):
        """Solves for the optimal crack propagation angle and load factor.

        This function uses gradient descent with line search to find the optimal
        crack propagation angle and load factor.

        Args:
            phi0 (float): Initial crack angle.
            SIFs_controlled (dict): Stress intensity factors for the controlled problem.
            SIFs_prescribed (dict): Stress intensity factors for the prescribed problem.
            s (float): Internal length associated with T-stress.

        Returns:
            Tuple[float, float]: The optimal crack propagation angle and load factor.
        """
        KIc, KIIc, Tc = (
            SIFs_controlled["KI"],
            SIFs_controlled["KII"],
            SIFs_controlled["T"],
        )
        KIp, KIIp, Tp = (
            SIFs_prescribed["KI"],
            SIFs_prescribed["KII"],
            SIFs_prescribed["T"],
        )

        # Perform the minimization
        kwargs = {
            "Ep": self.model.Ep_func(self.xc),
            "s": s,
            "KIc": KIc,
            "KIIc": KIIc,
            "Tc": Tc,
            "KIp": KIp,
            "KIIp": KIIp,
            "Tp": Tp,
            "phi0": phi0,
        }

        phi = find_local_min(phi0, self.grad, kwargs=kwargs)

        # Check the stability of the solution (i.e., check if solution is a max)
        hess = self.hess([phi], **kwargs)[0][0]
        solution_is_max = hess < 0
        if solution_is_max:
            print("Found a maximum instead of minimum -> perturbating the objective")
            print("Note: this test might also be triggered by cups!")
            # Perform another gradient descent on the perturbed objective
            phi = find_local_min(phi0, self.grad_pert, kwargs=kwargs)

        # Compute the load factor
        load_factor = self.objective([phi], **kwargs)

        return float(phi), float(load_factor)


def find_local_min(
    phi0: float,
    grad_func: Callable,
    kwargs: dict,
    tol: float = 1e-10,
    max_expand: int = 60,
) -> float:
    """Finds the local minimum reached by descending -grad(phi0).

    1. Walk downhill with geometrically growing steps until grad changes sign.
    2. Refine the sign-change bracket with brentq (robust to discontinuous grad:
       a jump from - to + across a kink is bracketed just like a smooth zero).
    """
    # Initialize the phi and the gradient
    phi = float(phi0)
    g = float(grad_func([jnp.asarray(phi)], **kwargs)[0])

    # Check if the point is already stationary
    if abs(g) < 1e-14:
        return phi

    # Downhill walk: expand steps until grad flips sign
    d = -jnp.sign(g)
    step = 1e-3
    for _ in range(max_expand):
        phi_new = phi + d * step
        g_new = float(grad_func([jnp.asarray(phi_new)], **kwargs)[0])
        if g_new == 0.0 or g_new * g < 0:  # sign change -> bracketed
            break
        phi, g, step = phi_new, g_new, 2.0 * step

    # Brent on the bracket: cheap, superlinear, kink-safe
    res = root_scalar(
        lambda p: float(grad_func([jnp.asarray(p)], **kwargs)[0]),
        bracket=(phi, phi_new),
        method="brentq",
        xtol=tol,
        rtol=tol,
    )
    return float(res.root)
