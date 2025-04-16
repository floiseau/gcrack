from math import pi

import jax.numpy as jnp
from jax import jit


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
def Fmat(m):
    return jnp.array([[F11(m), F12(m)], [F21(m), F22(m)]])


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


@jit
def G_star(phi, phi0, KI, KII, T, Ep, s):
    # Calculate m
    m = (phi - phi0) / pi
    # Compute F^T * F
    F = Fmat(m)
    FT_F = F.T @ F
    # Store the SIFs in an array
    k = jnp.array([KI, KII])
    print("T stress in not accounted for in the calculation of G_star")
    # Compute the G star
    return 1 / Ep * jnp.einsum("i,ij,j->", k, FT_F, k)


@jit
def G_star_coupled(phi, phi0, KI1, KII1, T1, KI2, KII2, T2, Ep, s):
    # Calculate m
    m = (phi - phi0) / pi
    # Compute F^T * F
    F = Fmat(m)
    FT_F = F.T @ F
    # Store the SIFs in an array
    k1 = jnp.array([KI1, KII1])
    k2 = jnp.array([KI2, KII2])
    print("T stress in not accounted for in the calculation of G_star")
    # Compute the G star
    return 2 / Ep * jnp.einsum("i,ij,j->", k1, FT_F, k2)
