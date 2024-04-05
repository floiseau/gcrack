import numpy as np
from scipy.optimize import differential_evolution

from utils.wrap_to_pi import wrap_to_pi

# NOTE https://www.google.com/search?q=polynomial+fractional+optimization&sca_esv=eec4b8c5dff95975&sca_upv=1&hl=en&ei=9PcPZuWeG7SNkdUPp4arkAE&ved=0ahUKEwjl6PbJkquFAxW0RqQEHSfDChIQ4dUDCBA&uact=5&oq=polynomial+fractional+optimization&gs_lp=Egxnd3Mtd2l6LXNlcnAiInBvbHlub21pYWwgZnJhY3Rpb25hbCBvcHRpbWl6YXRpb24yCBAAGIAEGKIEMggQABiABBiiBDIIEAAYgAQYogRIxAhQmARYjQVwAXgAkAEAmAFfoAHUAaoBATO4AQPIAQD4AQGYAgOgAqcBwgIOEAAYgAQYigUYhgMYsAPCAgsQABiABBiiBBiwA5gDAIgGAZAGBpIHATOgB-UJ&sclient=gws-wiz-serp


def F_AL92(dphi):
    wrapped_dphi = wrap_to_pi(dphi)
    m = wrapped_dphi / np.pi
    f_AL = np.empty((2, 2))
    f_AL[0, 0] = (
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
    f_AL[0, 1] = (
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
    f_AL[1, 0] = (
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
    f_AL[1, 1] = (
        1
        - (4 + 3 / 8 * np.pi**2) * m**2
        + (8 / 3 + 29 / 18 * np.pi**2 - 5 / 128 * np.pi**4) * m**4
        + (
            -32 / 15
            - 4 / 9 * np.pi**2
            - 1159 / 7200 * np.pi**4
            + 119 / 15_360 * np.pi**6
        )
        * m**6
        + 10.58254 * m**8
        - 4.78511 * m**10
        - 1.8804 * m**12
        + 7.280 * m**14
        - 7.591 * m**16
        + 0.25 * m**18
        + 12.5 * m**20
    )
    return f_AL


def residual(x, model, K, gc_func, phi0):
    phi = x[0]
    K_star = np.dot(F_AL92(phi - phi0), K)
    g = 1 / model.Ep * (np.dot(K_star, K_star))
    gc = gc_func(phi)
    return np.sqrt(gc / g)


def compute_load_factor(phi0: float, model, K, gc_func):
    return differential_evolution(
        residual,
        bounds=[(phi0 - np.pi * 2 / 3, phi0 + np.pi * 2 / 3)],
        args=(model, K, gc_func, phi0),
        popsize=128,
    )
