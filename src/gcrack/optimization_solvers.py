from math import pi

import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, jacobian, hessian, random

from lefm import G_star, G_star_coupled
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
        self.grad = jit(jacobian(self.objective))
        self.hess = jit(hessian(self.objective))
        # Automatic differentiation of the perturbated objective function
        self.grad_pert = jit(jacobian(self.objective_pert))
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
        print("├─ Determination of propagation angle (GMERR) and load factor (GMERR)")
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

        # There are two cases: continuous vs discontinuous derivative
        # TODO: Make it a user-specified parameter
        case = "continuous"  # "discontinuous"
        match case:
            case "continuous":
                # TODO: Replace the gradient descent with a line search !!!
                phi = gradient_descent(phi0, self.grad, kwargs=kwargs)
            case "discontinuous":
                phi = bisection(
                    self.grad, phi0 - jnp.pi / 2, phi0 + jnp.pi / 2, kwargs=kwargs
                )

        # Check the stability of the solution (i.e., check if solution is a max)
        hess = self.hess([phi], **kwargs)[0][0]
        solution_is_max = hess < 0
        if solution_is_max:
            # TODO: Check is this case really occurs
            print("Found a maximum instead of minimum -> perturbating the objective")

            match case:
                case "continuous":
                    phi = gradient_descent(phi0, self.grad_pert, kwargs=kwargs)
                case "discontinuous":
                    # Choose an arbitrary side ???
                    # TODO: Check the bounds
                    # NOTE: Maybe need to try both sides
                    phi = bisection(
                        self.grad, phi0 - jnp.pi / 2, phi0 - 1e-6, kwargs=kwargs
                    )

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
        phis = jnp.linspace(-pi / 2, pi / 2, num=180).__array__()
        objs = self.objective([phis], **kwargs)
        objs_pert = self.objective_pert([phis], **kwargs)
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
        plt.plot(phis, grads)
        # plt.scatter([phi], [self.grad([phi], **kwargs)[0]], c="r")
        plt.grid()
        plt.tight_layout()
        plt.savefig(dir_name / f"residual_function_{t:08d}.svg")

        phis = jnp.linspace(-pi / 2, pi / 2, num=180)
        Gcs_inv = 1 / self.Gc(phis)
        Gss_inv = 1 / (
            load_factor**2 * G_star(phis, phi0, KIc, KIIc, Tc, self.model.Ep, s)
        )

        # NOTE: This plot is invalid if there is a prescribed loading !
        # To make it valid, see the calculation with superimposition of the controlled and prescribed loads.
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.plot(phis, Gcs_inv, label=r"$G_c^{-1} (\varphi)$")
        ax.plot(phis, Gss_inv, label=r"$G^{* -1} (\varphi)$")
        ax.scatter([phi], [1 / self.Gc(phi)], label="Min", color="r")
        ax.grid(True)
        ax.legend()
        ax.set_title("Wulff plot")
        plt.savefig(dir_name / f"wulff_plot_{t:08d}.svg")

        plt.close("all")


### Optimizers


def gradient_descent(phi0, f, tol: float = 1e-6, max_iter=100_000, kwargs={}):
    print("│  └─ Running the gradient descent")
    # Initialization
    phi = float(phi0)
    converged = False
    for i in range(max_iter):
        inc = -f([phi], **kwargs)[0]
        phi += inc
        phi = min(max(phi0 - 2 * jnp.pi / 3, phi), phi0 + 2 * jnp.pi / 3)
        print(
            f"│     ├─ Step: {i + 1:06d} | Phi: {jnp.rad2deg(phi):+7.2f}° | Error: {abs(inc):8.3g}",
            end="\r",
        )
        if abs(inc) < tol:
            converged = True
            print(
                f"│     ├─ Step: {i + 1:06d} | Phi: {jnp.rad2deg(phi):+7.2f}° | Error: {abs(inc):8.3g}",
            )
            print("│     └─ Converged")
            break

    # Check the convergence
    if not converged:
        raise RuntimeError(" └─ Gradient descent failed to converge!")
    return phi


def bisection(f, a, b, tol: float = 1e-6, kwargs: dict = {}):
    # Evaluate f at bounds
    fa = f([a], **kwargs)[0]
    fb = f([b], **kwargs)[0]
    # check if a and b bound a root
    if jnp.sign(fa) == jnp.sign(fb):
        raise Exception("The scalars a and b do not bound a root")

    # get midpoint
    m = (a + b) / 2
    print(m)
    # evaluate f at midpoint
    fm = f([m], **kwargs)[0]

    # if jnp.abs(fm) < tol:
    if b - a < tol:
        # stopping condition, report m as root
        return m
    elif jnp.sign(fa) == jnp.sign(fm):
        # case where m is an improvement on a.
        # Make recursive call with a = m
        return bisection(f, m, b, tol, kwargs)
    elif jnp.sign(fb) == jnp.sign(fm):
        # case where m is an improvement on b.
        # Make recursive call with b = m
        return bisection(f, a, m, tol, kwargs)


def newton(phi0, f, df, tol: float = 1e-6, max_iter: int = 1000, kwargs: dict = {}):
    print("│  └─ Running the Newton method")
    # Initialization
    phi = float(phi0)
    converged = False
    for i in range(max_iter):
        inc = -f([phi], **kwargs)[0] / df([phi], **kwargs)[0][0]
        phi += inc
        # phi = min(max(phi0 - jnp.pi / 2, phi), phi0 + jnp.pi / 2)
        print(
            f"│     ├─ Step: {i + 1:03d} | Phi: {jnp.rad2deg(phi):+7.2f}° | Error: {abs(inc):.3g}"
        )
        if abs(inc) < tol:
            converged = True
            print("│     └─ Converged")
            break

    # Check the convergence
    if not converged:
        raise RuntimeError(" └─ Newton method failed to converge!")
    return phi
