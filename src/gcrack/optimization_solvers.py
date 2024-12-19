from enum import Enum
from math import pi

import jax.numpy as jnp
from jax import jit, jacobian, hessian


class Criterion(Enum):
    GMERR = 1
    PLS = 2


def compute_load_factor(
    phi0: float, model, SIFs_controlled, SIFs_prescribed, gc_func, s, criterion: str
):
    match Criterion[criterion.upper()]:
        case Criterion.GMERR:
            return compute_load_factor_with_gmerr(
                phi0, model, SIFs_controlled, SIFs_prescribed, gc_func, s
            )
        case Criterion.PLS:
            return compute_load_factor_with_pls(
                phi0, model, SIFs_controlled, SIFs_prescribed, gc_func, s
            )
        case _:
            raise NotImplementedError(f"The criterion {criterion} is not implemented")


### GMERR
def gmerr_objective(x, Gc, Ep, s, KIc, KIIc, Tc, KIp, KIIp, Tp, phi0):
    # NOTE : The KIc (etc.) means controlled (not critical !)
    phi = x[0]
    m = (phi - phi0) / pi
    # Compute the star SIFs
    KIc_star = F11(m) * KIc + F12(m) * KIIc + Tc * jnp.sqrt(s) * G1(m)
    KIIc_star = F21(m) * KIc + F22(m) * KIIc + Tc * jnp.sqrt(s) * G2(m)
    KIp_star = F11(m) * KIp + F12(m) * KIIp + Tp * jnp.sqrt(s) * G1(m)
    KIIp_star = F21(m) * KIp + F22(m) * KIIp + Tp * jnp.sqrt(s) * G2(m)
    # Compute the G star
    Gs_cc = 1 / Ep * (KIc_star**2 + KIIc_star**2)
    Gs_cp = 2 / Ep * (KIc_star * KIp_star + KIIc_star * KIIp_star)
    Gs_pp = 1 / Ep * (KIp_star**2 + KIIp_star**2)
    # Compute and return the load factor
    delta = Gs_cp**2 - 4 * Gs_cc * (Gs_pp - Gc)
    return (-Gs_cp + jnp.sqrt(delta)) / (2 * Gs_cc)


gmerr_residual = jit(jacobian(gmerr_objective))
gmerr_hess = hessian(gmerr_objective)


def compute_load_factor_with_gmerr(
    phi0: float, model, SIFs_controlled, SIFs_prescribed, gc_func, s
):
    print("├─ Determination of propagation angle (GMERR) and load factor (GMERR)")
    KIc, KIIc, Tc = SIFs_controlled["KI"], SIFs_controlled["KII"], SIFs_controlled["T"]
    KIp, KIIp, Tp = SIFs_prescribed["KI"], SIFs_prescribed["KII"], SIFs_prescribed["T"]

    # Perform the minimization
    kwargs = {
        "Gc": gc_func(0),
        "Ep": model.Ep,
        "s": s,
        "KIc": KIc,
        "KIIc": KIIc,
        "Tc": Tc,
        "KIp": KIp,
        "KIIp": KIIp,
        "Tp": Tp,
        "phi0": phi0,
    }
    phi = newton(phi0, gmerr_residual, gmerr_hess, kwargs=kwargs, gc_func=gc_func)
    # Compute the load factor
    kwargs["Gc"] = gc_func(phi)
    load_factor = gmerr_objective([phi], **kwargs)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.xlabel(r"Bifurcation angle $\varphi$ (rad)")
    # plt.ylabel(r"Load factor $\sqrt{\frac{G_c(\varphi)}{G^*(\varphi)}}$")
    # phis = np.linspace(-pi / 2, pi / 2, num=180)
    # plt.plot(phis, obj_func(phis))
    # plt.scatter([phi_val], [obj_func(phi_val)], c="r")
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig("test.pdf")
    # plt.close()

    return float(phi), float(load_factor)


### PLS
@jit
def pls_residual(x, Ep, s, KI, KII, T, phi0):
    phi = x[0]
    m = (phi - phi0) / pi
    KII_star = F21(m) * KI + F22(m) * KII + T * jnp.sqrt(s) * G2(m)
    return jnp.array([KII_star])


pls_jac = jit(jacobian(pls_residual))
pls_hess = jit(hessian(pls_residual))


def newton(
    phi0, f, df, tol: float = 1e-6, max_iter: int = 100, kwargs={}, gc_func=None
):
    print("│  └─ Running the Newton method")
    # Initialization
    phi = float(phi0)
    converged = False
    for i in range(max_iter):
        if gc_func is not None:
            kwargs["Gc"] = gc_func(phi)
        inc = -f([phi], **kwargs)[0] / df([phi], **kwargs)[0][0]
        phi += inc
        print(f"│     ├─ Step: {i+1} | Error: {abs(inc)}")
        if abs(inc) < tol:
            converged = True
            print("│     └─ Converged")
            break

    # Check the convergence
    if not converged:
        raise RuntimeError(" └─ Newton method failed to converge!")
    return phi


def compute_load_factor_with_pls(
    phi0: float, model, SIFs_controlled, SIFs_prescribed, gc_func, s
):
    print("├─ Determination of propagation angle (PLS) and load factor (GMERR)")
    KIc, KIIc, Tc = SIFs_controlled["KI"], SIFs_controlled["KII"], SIFs_controlled["T"]
    # Find a root of KII
    kwargs = {"Ep": model.Ep, "s": s, "KI": KIc, "KII": KIIc, "T": Tc, "phi0": phi0}
    phi_val = newton(phi0, pls_residual, pls_jac, kwargs=kwargs)

    # Check if the result is a local minimum
    hes_sol = pls_hess([phi_val], **kwargs)
    print(f"hessian of solution: {hes_sol}")
    if hes_sol[0][0] < 0:
        print("WARNING : The gmerr_hessian of the solution is local maximum")

    # Compute the load factor
    KIp, KIIp, Tp = SIFs_prescribed["KI"], SIFs_prescribed["KII"], SIFs_prescribed["T"]
    kwargs_gmerr = {
        "Gc": gc_func(0),
        "Ep": model.Ep,
        "s": s,
        "KIc": KIc,
        "KIIc": KIIc,
        "Tc": Tc,
        "KIp": KIp,
        "KIIp": KIIp,
        "Tp": Tp,
        "phi0": phi0,
    }
    load_factor = gmerr_objective([phi_val], **kwargs_gmerr)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.xlabel(r"Bifurcation angle $\varphi$ (rad)")
    # plt.ylabel(r"Load factor $\sqrt{\frac{G_c(\varphi)}{G^*(\varphi)}}$")
    # phis = np.linspace(-pi / 2, pi / 2, num=180)
    # plt.plot(phis, obj_func(phis))
    # plt.scatter([phi_val], [obj_func(phi_val)], c="r")
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig("test.pdf")
    # plt.close()

    return float(phi_val), float(load_factor)


# Define the Amestoy-Leblond F functions
@jit
def F11(m):
    return (
        1
        - 3 * pi**2 / 8 * m**2
        + (pi**2 - 5 * pi**4 / 128) * m**4
        + (pi**2 / 9 - 11 * pi**4 / 72 + 119 * pi**6 / 15_360) * m**6
        + 5.07790 * m**8
        - 2.88312 * m**10
        - 0.0925 * m**12
        + 2.996 * m**14
        - 4.059 * m**16
        + 1.63 * m**18
        + 4.1 * m**20
    )


@jit
def F12(m):
    return (
        -3 * pi / 2 * m
        + (10 * pi / 3 + pi**3 / 16) * m**3
        + (-2 * pi - 133 * pi**3 / 180 + 59 * pi**5 / 1280) * m**5
        + 12.313906 * m**7
        - 7.32433 * m**9
        + 1.5793 * m**11
        + 4.0216 * m**13
        - 6.915 * m**15
        + 4.21 * m**17
        + 4.56 * m**19
    )


@jit
def F21(m):
    return (
        pi / 2 * m
        - (4 * pi / 3 + pi**3 / 48) * m**3
        + (-2 * pi / 3 + 13 * pi**3 / 30 - 59 * pi**5 / 3840) * m**5
        - 6.176023 * m**7
        + 4.44112 * m**9
        - 1.5340 * m**11
        - 2.0700 * m**13
        + 4.684 * m**15
        - 3.95 * m**17
        - 1.32 * m**19
    )


@jit
def F22(m):
    return (
        1
        - (4 + 3 / 8 * pi**2) * m**2
        + (8 / 3 + 29 / 18 * pi**2 - 5 / 128 * pi**4) * m**4
        + (-32 / 15 - 4 / 9 * pi**2 - 1159 / 7200 * pi**4 + 119 / 15_360 * pi**6) * m**6
        + 10.58254 * m**8
        - 4.78511 * m**10
        - 1.8804 * m**12
        + 7.280 * m**14
        - 7.591 * m**16
        + 0.25 * m**18
        + 12.5 * m**20
    )


@jit
def G1(m):
    return (
        (2 * pi) ** (3 / 2) * m**2
        - 47.933390 * m**4
        + 63.665987 * m**6
        - 50.70880 * m**8
        + 26.66807 * m**10
        - 6.0205 * m**12
        - 7.314 * m**14
        + 10.947 * m**16
        - 2.85 * m**18
        - 13.7 * m**20
    )


@jit
def G2(m):
    return (
        -2 * jnp.sqrt(2 * pi) * m
        + 12 * jnp.sqrt(2 * pi) * m**3
        - 59.565733 * m**5
        + 61.174444 * m**7
        - 39.90249 * m**9
        + 15.6222 * m**11
        + 3.0343 * m**13
        - 12.781 * m**15
        + 9.69 * m**17
        + 6.62 * m**19
    )
