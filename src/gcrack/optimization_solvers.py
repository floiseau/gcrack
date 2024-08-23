import numpy as np
import sympy as sp
from scipy.optimize import minimize

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


def compute_load_factor(phi0: float, model, K, gc_expr):
    print("-- Determination of propagation angle and load factor")
    # Define phi as a SymPy symbol
    phi = sp.Symbol("phi")
    # Get the SIFs (star)
    KI, KII = K
    KI_star = F11 * KI + F12 * KII
    KII_star = F21 * KI + F22 * KII
    # Compute the G*
    gs = 1 / model.Ep * (KI_star**2 + KII_star**2)
    # Get the Gc
    gc = gc_expr(phi)
    # Get the expression
    obj_symb = gc / gs  # sp.sqrt(gc/gs)
    # obj_symb = gc / gs  # sp.sqrt(gc/gs)
    # obj_symb = -(gs - gc)
    obj_symb = obj_symb.subs({"phi0": phi0})
    obj_func = sp.lambdify(phi, obj_symb, "numpy")
    # Perform the (local) minimization
    res = minimize(
        obj_func,
        x0=[phi0],
        tol=1e-12,
        bounds=[
            (
                phi0 - np.pi / 2,
                phi0 + np.pi / 2,
            )
        ],
    )
    phi_val = res.x[0]
    # Compute the load factor
    gs_func = sp.lambdify(phi, gs.subs({"phi0": phi0}), "numpy")
    gc_func = sp.lambdify(phi, gc.subs({"phi0": phi0}), "numpy")
    load_factor = np.sqrt(gc_func(phi_val) / gs_func(phi_val))
    # load_factor = np.sqrt(obj_func(phi_val))
    return phi_val, load_factor
