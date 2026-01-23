"""
Module for solving the optimization problem to determine the load factor in crack propagation simulations.

This module provides the `LoadFactorSolver` class, which implements the GMERR (Generalized Maximum Energy Release Rate) criterion for crack propagation.
The solver is based on the work of Amestoy and Leblond (1992) and is designed to work with the `gcrack.lefm` module.
It uses gradient descent with line search to find the optimal crack propagation angle and load factor.

The module also includes a custom gradient descent optimizer with line search for robust convergence, even in the presence of discontinuities or "cups" in critical energy release rate Gc.

References:
    Amestoy, M., & Leblond, J. B. (1992). A new numerical method for crack growth prediction.
    International Journal of Solids and Structures, 29(21), 2619-2638.
    https://doi.org/10.1016/0020-7683(92)90210-K
"""

from pathlib import Path
from collections.abc import Callable
from typing import List

from math import pi

import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, grad, hessian, random

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
        self.Gc = jit(Gc_func)
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
            self.objective(x, Ep, s, KIc, KIIc, Tc, KIp, KIIp, Tp, phi0) + 1e-5 * x[0]
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

        phi = gradient_descent_with_line_search(phi0, self.grad, kwargs=kwargs)

        # Check the stability of the solution (i.e., check if solution is a max)
        hess = self.hess([phi], **kwargs)[0][0]
        solution_is_max = hess < 0
        if solution_is_max:
            print("Found a maximum instead of minimum -> perturbating the objective")
            print("Note: this test might also be triggered by cups!")
            # Perform another gradient descent on the perturbed objective
            phi = gradient_descent_with_line_search(phi0, self.grad_pert, kwargs=kwargs)

        # Compute the load factor
        load_factor = self.objective([phi], **kwargs)

        return float(phi), float(load_factor)

    def export_minimization_plots(
        self,
        phi: float,
        load_factor: float,
        phi0: float,
        SIFs_controlled: dict,
        SIFs_prescribed: dict,
        s: float,
        t: int,
        dir_name: Path,
    ):
        """Exports plots of the objective function and its gradient during minimization.

        This function generates and saves plots of the objective function, its perturbed
        version, and the gradient of the objective function during the minimization process.
        This function is mainly use to illustrate the minimization.

        Args:
            phi (float): Optimal crack propagation angle.
            load_factor (float): Optimal load factor.
            phi0 (float): Initial crack angle.
            SIFs_controlled (dict): Stress intensity factors for the controlled problem.
            SIFs_prescribed (dict): Stress intensity factors for the prescribed problem.
            s (float): Internal length associated with T-stress.
            t (int): Current time step.
            dir_name (Path): Directory to save the plots.
        """
        # Extract the SIFs
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
        # Construct the kwargs
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

        # Display the objective function (and its minimum)
        plt.figure()
        plt.xlabel(r"Bifurcation angle $\varphi$ (rad)")
        plt.ylabel(r"Load factor $\sqrt{\frac{G_c(\varphi)}{G^*(\varphi)}}$")
        phis = jnp.linspace(phi0 - pi / 2, phi0 + pi / 2, num=180).__array__()
        objs = [self.objective([phi], **kwargs) for phi in phis]
        objs_pert = [self.objective_pert([phi], **kwargs) for phi in phis]
        plt.plot(phis, objs, label="Objective")
        plt.plot(phis, objs_pert, label="Perturbated objective")
        plt.scatter([phi], [self.objective([phi], **kwargs)], c="r")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(dir_name / f"objective_function_{t:08d}.svg")

        plt.figure()
        plt.xlabel(r"Bifurcation angle $\varphi$ (rad)")
        plt.ylabel(r"Derivative of the load factor")
        grads = [self.grad([phi_], **kwargs)[0] for phi_ in phis]
        plt.scatter([phi], [self.grad([phi], **kwargs)[0]], c="r")
        plt.plot(phis, grads)
        plt.grid()
        plt.tight_layout()
        plt.savefig(dir_name / f"residual_function_{t:08d}.svg")

        plt.close("all")


def gradient_descent_with_line_search(
    phi0: float,
    gra: Callable,
    tol: float = 1e-6,
    max_iter: int = 10_000,
    kwargs: dict = {},
) -> float:
    """Performs gradient descent with line search to minimize an objective function.

    This function implements gradient descent with a custom line search to find the minimum
    of an objective function. It is designed to handle discontinuities and "cups" in the
    objective function.

    Args:
        phi0 (float): Initial guess for the crack propagation angle.
        gra (Callable): The gradient of the objective function.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.
        kwargs (dict): Additional arguments to pass to the gradient function.

    Returns:
        float: The optimal crack propagation angle.

    Raises:
        RuntimeError: If the gradient descent fails to converge.
    """
    print("│  │  Running the gradient descent with custom line search")
    # Initialization
    phi = float(phi0)
    converged = False
    for i in range(max_iter):
        # Determine the direction
        direction = -gra([phi], **kwargs)[0]
        # Check if the direction is close to 0
        if jnp.isclose(direction, 0):
            # Set a null increment
            dphi = 0
            # Set a null idx
            idx = 0
        else:
            # Apply line-search
            cs = [0.0] + [(jnp.pi / 2) ** k for k in range(-29, 2)]
            phis_test = jnp.array([phi + c * jnp.sign(direction) for c in cs])
            # Get the index associated with the first increase of the objective
            diff = jnp.array([gra([phi_test], **kwargs)[0] for phi_test in phis_test])
            # Create an array with the slope "in the direction of minimization"
            slope = jnp.sign(direction) * diff
            if all(slope < 0):  # If the slope is always negative, take the largest step
                idx = -1
            elif all(slope > 0):  # If the slope is always positive, take no step
                idx = 0
            else:  # If the slope increases after a decrease, then local minimum
                idx = jnp.where(slope > 0)[0][0] - 1
                # If the first grad is positive, we are at the solution
                # This case only occurs when the grad is discontinuous (cups)
                if idx == -1:
                    idx = 0
            # Calculate the increment
            dphi = phis_test[idx] - phi
        # Update the solution
        phi += dphi
        # Generate an info message
        msg = "│  │  │  "
        msg += f"Step: {i + 1:06d} | "
        msg += f"phi: {jnp.rad2deg(phi):+7.2f}° | "
        msg += f"dphi: {abs(dphi):8.3g}"
        print(msg)
        # Check the convergence
        converged = idx == 0 or abs(dphi) <= tol
        if converged:
            print("│  │  │  Converged")
            break
        else:
            # Clip the angle phi
            phi = min(max(phi0 - 2 * jnp.pi / 3, phi), phi0 + 2 * jnp.pi / 3)

    # Check the convergence
    if not converged:
        raise RuntimeError(" └─ Gradient descent failed to converge!")
    return phi
