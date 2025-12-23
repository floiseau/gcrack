"""
Module for computing stress intensity factor (SIF) influence functions and energy release rates.

This module provides functions to compute the functions F_ij and G_i as described in Amestoy and Leblond (1992).
These functions are used to calculate the SIFs at the tip of an infinitesimal straight crack extension with a given bifurcation angle.

References:
    Amestoy, M., & Leblond, J. B. (1992). Crack paths in plane situations—II. Detailed form of the expansion of the stress intensity factors. International Journal of Solids and Structures, 29(4), 465–501. [https://doi.org/10.1016/0020-7683(92)90210-K](https://doi.org/10.1016/0020-7683(92)90210-K)
"""

from math import pi

import jax.numpy as jnp
from jax import jit


@jit
def F11(m: float) -> float:
    """Computes F11 function for a given normalized crack angle.

    Args:
        m (float): Normalized crack angle, defined as (phi - phi0) / pi.

    Returns:
        float: The value of the F11 for the given normalized crack angle.
    """
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
def F12(m: float) -> float:
    """Computes F12 function for a given normalized crack angle.

    Args:
        m (float): Normalized crack angle, defined as (phi - phi0) / pi.

    Returns:
        float: The value of the F12 for the given normalized crack angle.
    """
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
def F21(m: float) -> float:
    """Computes F21 function for a given normalized crack angle.

    Args:
        m (float): Normalized crack angle, defined as (phi - phi0) / pi.

    Returns:
        float: The value of the F21 for the given normalized crack angle.
    """
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
def F22(m: float) -> float:
    """Computes F22 function for a given normalized crack angle.

    Args:
        m (float): Normalized crack angle, defined as (phi - phi0) / pi.

    Returns:
        float: The value of the F22 for the given normalized crack angle.
    """
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
def Fmat(m: float) -> jnp.ndarray:
    """Construct the matrix F containing the Fij functions of Amestoy-Leblond.

    Args:
        m (float): Normalized crack angle, defined as (phi - phi0) / pi.

    Returns:
        jnp.ndarray: The 2x2 matrix F for the given normalized crack angle.
    """
    return jnp.array([[F11(m), F12(m)], [F21(m), F22(m)]])


@jit
def G1(m: float) -> float:
    """Computes G1 function for a given normalized crack angle.

    Args:
        m (float): Normalized crack angle, defined as (phi - phi0) / pi.

    Returns:
        float: The value of the G1 for the given normalized crack angle.
    """
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
def G2(m: float) -> float:
    """Computes G2 function for a given normalized crack angle.

    Args:
        m (float): Normalized crack angle, defined as (phi - phi0) / pi.

    Returns:
        float: The value of the G2 for the given normalized crack angle.
    """
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
def Gvec(m: float) -> jnp.ndarray:
    """Construct the vector G containing the Gi functions of Amestoy-Leblond.

    Args:
        m (float): Normalized crack angle, defined as (phi - phi0) / pi.

    Returns:
        jnp.ndarray: The vector G for the given normalized crack angle.
    """
    return jnp.array([G1(m), G2(m)])


@jit
def G_star(
    phi: float, phi0: float, KI: float, KII: float, T: float, Ep: float, s: float
) -> float:
    """Computes the energy release rate G* after a infinitesimal kink of angle.

    This function computes the energy release rate G* using the Irwin formula.
    The SIFs are calculated as described in Amestoy and Leblond (1992).

    Args:
        phi (float): Current crack angle.
        phi0 (float): Initial crack angle.
        KI (float): Mode I stress intensity factor.
        KII (float): Mode II stress intensity factor.
        T (float): T-stress.
        Ep (float): Plane strain/stress modulus.
        s (float): Internal length associated with T-stress.

    Returns:
        float: The energy release rate G* for the given crack angle and stress intensity factors.
    """
    # Store the SIFs in an array
    k = jnp.array([KI, KII])
    # Calculate m
    m = (phi - phi0) / pi
    # Compute the Amestoy-Leblond functions
    f_mat = Fmat(m)
    g_vec = Gvec(m)
    # Apply Amestoy-Leblond formula
    ks = f_mat @ k + g_vec * T * jnp.sqrt(s)
    # Compute the G star
    return 1 / Ep * jnp.dot(ks, ks)


@jit
def G_star_coupled(
    phi: float,
    phi0: float,
    KI1: float,
    KII1: float,
    T1: float,
    KI2: float,
    KII2: float,
    T2: float,
    Ep: float,
    s: float,
) -> float:
    """Computes the coupled energy release rate G* for two sets of stress intensity factors.

    It is used to evaluate the energy release rate for two interacting loading conditions.

    Args:
        phi (float): Current crack angle.
        phi0 (float): Initial crack angle.
        KI1 (float): Mode I stress intensity factor for the first load.
        KII1 (float): Mode II stress intensity factor for the first load.
        T1 (float): T-stress for the first load.
        KI2 (float): Mode I stress intensity factor for the second load.
        KII2 (float): Mode II stress intensity factor for the second load.
        T2 (float): T-stress for the second load.
        Ep (float): Plane strain modulus.
        s (float): Internal length associated with T-stress.

    Returns:
        float: The coupled energy release rate G* for the given crack angle and stress intensity factors.
    """
    # Calculate m
    m = (phi - phi0) / pi
    # Compute F^T * F
    F = Fmat(m)
    FT_F = F.T @ F
    # Store the SIFs in an array
    k1 = jnp.array([KI1, KII1])
    k2 = jnp.array([KI2, KII2])
    # Compute the G star
    return 2 / Ep * jnp.einsum("i,ij,j->", k1, FT_F, k2)
