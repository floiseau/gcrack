from math import pi

import jax.numpy as jnp
from jax import hessian
from jax.scipy.optimize import minimize


# Define the Amestoy-Leblond F functions
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


def compute_load_factor(phi0: float, model, SIFs, gc_func, s):
    print("-- Determination of propagation angle and load factor")
    KI, KII, T = SIFs["KI"], SIFs["KII"], SIFs["T"]

    # Define
    def objective(x):
        phi = x[0]
        m = (phi - phi0) / pi
        KI_star = F11(m) * KI + F12(m) * KII + T * jnp.sqrt(s) * G1(m)
        KII_star = F21(m) * KI + F22(m) * KII + T * jnp.sqrt(s) * G2(m)
        gs = 1 / model.Ep * (KI_star**2 + KII_star**2)
        gc = gc_func(phi)
        return jnp.sqrt(gc / gs)

    # Perform the minimization
    res = minimize(
        objective,
        x0=jnp.array([float(phi0)]),  # + pi / 90 * np.random.uniform(-1, 1),
        method="BFGS",
    )
    phi_val = res.x[0]

    # Check if the result is a local minimum
    hess = hessian(objective)
    hes_sol = hess(res.x)
    print(f"Hessian of solution: {hes_sol}")
    if hes_sol < 0:
        print("WARNING : The hessian of the solution is local maximum")

    # Compute the load factor
    load_factor = objective(res.x)

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
