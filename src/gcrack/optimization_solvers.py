from math import pi

import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, grad, hessian, random

from gcrack.lefm import G_star, G_star_coupled
# NOTE: Use from lefm import KII_star for PLS

prng_key = random.key(0)


### GMERR
class LoadFactorSolver:
    def __init__(self, model, Gc_func):
        # Store the model
        self.model = model
        # Set the critical energy release rate function
        self.Gc = jit(Gc_func)
        # Automatic differentiation of the objective function
        self.grad = jit(grad(self.objective))
        self.hess = jit(hessian(self.objective))
        # Automatic differentiation of the perturbated objective function
        self.grad_pert = jit(grad(self.objective_pert))
        self.hess_pert = jit(hessian(self.objective_pert))

    def objective(self, x, Ep, s, KIc, KIIc, Tc, KIp, KIIp, Tp, phi0):
        # NOTE : The KIc (etc.) means controlled (not critical !)
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

    def objective_pert(self, x, Ep, s, KIc, KIIc, Tc, KIp, KIIp, Tp, phi0):
        return (
            self.objective(x, Ep, s, KIc, KIIc, Tc, KIp, KIIp, Tp, phi0) + 1e-6 * x[0]
        )

    def solve(self, phi0: float, SIFs_controlled, SIFs_prescribed, s):
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
            "Ep": self.model.Ep,
            "s": s,
            "KIc": KIc,
            "KIIc": KIIc,
            "Tc": Tc,
            "KIp": KIp,
            "KIIp": KIIp,
            "Tp": Tp,
            "phi0": phi0,
        }

        # print(f"phi  = {float(jnp.rad2deg(phi))}")
        phi = gradient_descent_with_line_search(phi0, self.grad, kwargs=kwargs)

        # Check the stability of the solution (i.e., check if solution is a max)
        hess = self.hess([phi], **kwargs)[0][0]
        solution_is_max = hess < 0
        if solution_is_max:
            print("Found a maximum instead of minimum -> perturbating the objective")
            print("Note: this test might also be triggered by cups!")
            # Perform another gradient descent on the perturbated objective
            phi = gradient_descent_with_line_search(phi0, self.grad_pert, kwargs=kwargs)

        # Compute the load factor
        load_factor = self.objective([phi], **kwargs)

        return float(phi), float(load_factor)

    def export_minimization_plots(
        self, phi, load_factor, phi0, SIFs_controlled, SIFs_prescribed, s, t, dir_name
    ):
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
            "Ep": self.model.Ep,
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
        # plt.scatter([phi], [self.grad([phi], **kwargs)[0]], c="r")
        plt.grid()
        plt.tight_layout()
        plt.savefig(dir_name / f"residual_function_{t:08d}.svg")

        # phis = jnp.linspace(-pi / 2, pi / 2, num=180)
        # Gcs_inv = 1 / self.Gc(phis)
        # Gss_inv = 1 / (
        #     load_factor**2 * G_star(phis, phi0, KIc, KIIc, Tc, self.model.Ep, s)
        # )

        # # NOTE: This plot is invalid if there is a prescribed loading !
        # # To make it valid, see the calculation with superimposition of the controlled and prescribed loads.
        # fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        # ax.plot(phis, Gcs_inv, label=r"$G_c^{-1} (\varphi)$")
        # ax.plot(phis, Gss_inv, label=r"$G^{* -1} (\varphi)$")
        # ax.scatter([phi], [1 / self.Gc(phi)], label="Min", color="r")
        # ax.grid(True)
        # ax.legend()
        # ax.set_title("Wulff plot")
        # plt.savefig(dir_name / f"wulff_plot_{t:08d}.svg")

        plt.close("all")


### Optimizers


def gradient_descent_with_line_search(
    phi0, gra, tol: float = 1e-6, max_iter: int = 100, kwargs={}
):
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
        else:
            # Apply line-seach
            cs = [0.0] + [0.9**k for k in reversed(range(-31, 32))]
            phis_test = jnp.array([phi + c * direction for c in cs])
            # Get the index associated with the first increase of the objective
            diff = jnp.array([-gra([phi_test], **kwargs)[0] for phi_test in phis_test])
            if all(diff < 0):  # If it only decreases, take the largest step
                idx = -1
            else:  # If it increases after a decrease, then local minimum
                idx = jnp.where(diff > 0)[0][0] - 1
            # Calculate the increment
            dphi = cs[idx] * direction
        # Update the solution
        phi += dphi
        # Generate an info message
        msg = "│  │  │  "
        msg += f"Step: {i + 1:06d} | "
        msg += f"phi: {jnp.rad2deg(phi):+7.2f}° | "
        msg += f"dphi: {abs(dphi):8.3g}"
        print(msg)
        # Check the convergence
        converged = abs(dphi) <= tol
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
