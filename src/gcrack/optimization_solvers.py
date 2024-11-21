import numpy as np
import sympy as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# NOTE https://www.google.com/search?q=polynomial+fractional+optimization&sca_esv=eec4b8c5dff95975&sca_upv=1&hl=en&ei=9PcPZuWeG7SNkdUPp4arkAE&ved=0ahUKEwjl6PbJkquFAxW0RqQEHSfDChIQ4dUDCBA&uact=5&oq=polynomial+fractional+optimization&gs_lp=Egxnd3Mtd2l6LXNlcnAiInBvbHlub21pYWwgZnJhY3Rpb25hbCBvcHRpbWl6YXRpb24yCBAAGIAEGKIEMggQABiABBiiBDIIEAAYgAQYogRIxAhQmARYjQVwAXgAkAEAmAFfoAHUAaoBATO4AQPIAQD4AQGYAgOgAqcBwgIOEAAYgAQYigUYhgMYsAPCAgsQABiABBiiBBiwA5gDAIgGAZAGBpIHATOgB-UJ&sclient=gws-wiz-serp


# Declare the m symbol
phi_symb = sp.symbols("phi")
phi0_symb = sp.symbols("phi0")
m = (phi_symb - phi0_symb) / sp.pi
# Define the Amestoy-Leblond F functions
F11 = (
    1
    - 3 * np.pi**2 / 8 * m**2
    + (np.pi**2 - 5 * np.pi**4 / 128) * m**4
    + (np.pi**2 / 9 - 11 * np.pi**4 / 72 + 119 * np.pi**6 / 15_360) * m**6
    + 5.07790 * m**8
    - 2.88312 * m**10
    - 0.0925 * m**12
    + 2.996 * m**14
    - 4.059 * m**16
    + 1.63 * m**18
    + 4.1 * m**20
)
F12 = (
    -3 * np.pi / 2 * m
    + (10 * np.pi / 3 + np.pi**3 / 16) * m**3
    + (-2 * np.pi - 133 * np.pi**3 / 180 + 59 * np.pi**5 / 1280) * m**5
    + 12.313906 * m**7
    - 7.32433 * m**9
    + 1.5793 * m**11
    + 4.0216 * m**13
    - 6.915 * m**15
    + 4.21 * m**17
    + 4.56 * m**19
)
F21 = (
    np.pi / 2 * m
    - (4 * np.pi / 3 + np.pi**3 / 48) * m**3
    + (-2 * np.pi / 3 + 13 * np.pi**3 / 30 - 59 * np.pi**5 / 3840) * m**5
    - 6.176023 * m**7
    + 4.44112 * m**9
    - 1.5340 * m**11
    - 2.0700 * m**13
    + 4.684 * m**15
    - 3.95 * m**17
    - 1.32 * m**19
)
F22 = (
    1
    - (4 + 3 / 8 * np.pi**2) * m**2
    + (8 / 3 + 29 / 18 * np.pi**2 - 5 / 128 * np.pi**4) * m**4
    + (-32 / 15 - 4 / 9 * np.pi**2 - 1159 / 7200 * np.pi**4 + 119 / 15_360 * np.pi**6)
    * m**6
    + 10.58254 * m**8
    - 4.78511 * m**10
    - 1.8804 * m**12
    + 7.280 * m**14
    - 7.591 * m**16
    + 0.25 * m**18
    + 12.5 * m**20
)

G1 = (
    (2 * np.pi) ** (3 / 2) * m**2
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
G2 = (
    -2 * np.sqrt(2 * np.pi) * m
    + 12 * np.sqrt(2 * np.pi) * m**3
    - 59.565733 * m**5
    + 61.174444 * m**7
    - 39.90249 * m**9
    + 15.6222 * m**11
    + 3.0343 * m**13
    - 12.781 * m**15
    + 9.69 * m**17
    + 6.62 * m**19
)


def compute_load_factor(phi0: float, model, SIFs, gc_expr, s):
    print("-- Determination of propagation angle and load factor")
    # Define phi as a SymPy symbol
    phi = sp.Symbol("phi")
    # Get the SIFs (star)
    KI, KII, T = SIFs["KI"], SIFs["KII"], SIFs["T"]
    KI_star = F11 * KI + F12 * KII + T * np.sqrt(s) * G1
    KII_star = F21 * KI + F22 * KII + T * np.sqrt(s) * G2
    # Compute the G*
    gs = 1 / model.Ep * (KI_star**2 + KII_star**2)
    # Get the Gc
    gc = gc_expr(phi)
    # Get the symbolic expressions
    obj_symb = sp.sqrt(gc / gs)
    jac_symb = sp.diff(obj_symb, phi)
    hes_symb = sp.diff(jac_symb, phi)
    # Compute the python functions
    obj_symb = obj_symb.subs({"phi0": phi0})
    jac_symb = jac_symb.subs({"phi0": phi0})
    hes_symb = hes_symb.subs({"phi0": phi0})
    obj_func = sp.lambdify(phi, obj_symb, "numpy")
    jac_func = sp.lambdify(phi, jac_symb, "numpy")
    hes_func = sp.lambdify(phi, hes_symb, "numpy")
    print("Replace numpy as scipy with JAX !!!!")
    # Perform the minimization
    res = minimize(
        obj_func,
        x0=phi0,  # + np.pi / 90 * np.random.uniform(-1, 1),
        jac=jac_func,
        hess=hes_func,
        bounds=[
            (
                phi0 - np.pi / 2,
                phi0 + np.pi / 2,
            )
        ],
        method="Newton-CG",  # "Nelder-Mead",  # "Newton-CG",
    )
    phi_val = res.x[0]
    print(res)

    hes_sol = hes_func(phi_val)
    print(f"Hessian of solution: {hes_sol}")
    if hes_sol < 0:
        print("WARNING : The hessian of the solution is local maximum")

    # plt.figure()
    # plt.xlabel(r"Bifurcation angle $\varphi$ (rad)")
    # plt.ylabel(r"Load factor $\sqrt{\frac{G_c(\varphi)}{G^*(\varphi)}}$")
    # phis = np.linspace(-np.pi / 2, np.pi / 2, num=180)
    # plt.plot(phis, obj_func(phis))
    # plt.scatter([phi_val], [obj_func(phi_val)], c="r")
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig("test.pdf")
    # plt.close()

    # Compute the load factor
    gs_func = sp.lambdify(phi, gs.subs({"phi0": phi0}), "numpy")
    gc_func = sp.lambdify(phi, gc.subs({"phi0": phi0}), "numpy")
    load_factor = np.sqrt(gc_func(phi_val) / gs_func(phi_val))
    # load_factor = np.sqrt(obj_func(phi_val))
    return phi_val, load_factor
