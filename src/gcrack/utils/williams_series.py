"""
Module for computing Williams series functions for crack tip fields.

This module provides functions to compute the Williams series functions (Gamma_I, Gamma_II, Gamma_III) for crack tip displacement fields in linear elastic fracture mechanics.
These functions are used to represent the displacement fields around a crack tip using complex variable methods.

Functions:
    Gamma_I:
        Computes the Williams series function for mode I.
    Gamma_II:
        Computes the Williams series function for mode II.
    Gamma_III:
        Computes the Williams series function for mode III.
"""

import jax
import jax.numpy as jnp


@jax.jit
def Gamma_I(n: int, z: complex, mu: float, ka: float):
    """Computes the Williams series function for mode I crack tip displacement fields.

    This function calculates the function for the mode I (opening mode) term in the Williams series expansion of the displacement field around a crack tip.
    The function is expressed in terms of complex coordinates and material properties.

    Args:
        n (int): The order of the term in the Williams series expansion.
        z (complex): Complex coordinate relative to the crack tip.
        mu (float): Shear modulus of the material.
        ka (float): Kolosov constant, which depends on the material properties and the 2D assumption.

    Returns:
        complex: The Williams series function for mode I at the given point.

    Notes:
        - The function uses polar coordinates derived from the complex coordinate z.
        - The result is a complex number representing the function in the series expansion.
        - The function is JIT-compiled using JAX for efficient execution.
    """
    # Compute the polar coordinates
    r = jnp.abs(z)
    theta = jnp.angle(z)
    # Compute the factor
    return (
        r ** (n / 2)
        / (2 * mu * jnp.sqrt(2 * jnp.pi))
        * (
            ka * jnp.exp(1j * theta * n / 2)
            - n / 2 * jnp.exp(1j * theta * (4 - n) / 2)
            + (n / 2 + (-1) ** n) * jnp.exp(-1j * n * theta / 2)
        )
    )


@jax.jit
def Gamma_II(n: int, z: complex, mu: float, ka: float):
    """Computes the Williams series function for mode II crack tip displacement fields.

    This function calculates the function for the mode II (opening mode) term in the Williams series expansion of the displacement field around a crack tip.
    The function is expressed in terms of complex coordinates and material properties.

    Args:
        n (int): The order of the term in the Williams series expansion.
        z (complex): Complex coordinate relative to the crack tip.
        mu (float): Shear modulus of the material.
        ka (float): Kolosov constant, which depends on the material properties and the 2D assumption.

    Returns:
        complex: The Williams series function for mode II at the given point.

    Notes:
        - The function uses polar coordinates derived from the complex coordinate z.
        - The result is a complex number representing the function in the series expansion.
        - The function includes a negative sign to recover the classic mode II displacement field.
        - The function is JIT-compiled using JAX for efficient execution.
    """
    # Compute the polar coordinates
    r = jnp.abs(z)
    theta = jnp.angle(z)
    # Compute the factor
    # NOTE: Minus sign to recover the classic mode II (+ux above the crack and -ux below the crack)
    return (
        -1j
        * r ** (n / 2)
        / (2 * mu * jnp.sqrt(2 * jnp.pi))
        * (
            ka * jnp.exp(1j * theta * n / 2)
            + n / 2 * jnp.exp(1j * theta * (4 - n) / 2)
            - (n / 2 - (-1) ** n) * jnp.exp(-1j * n * theta / 2)
        )
    )


@jax.jit
def Gamma_III(n: int, z: complex, mu: float, ka: float):
    """Computes the Williams series function for mode III crack tip displacement fields.

    This function calculates the function for the mode III (opening mode) term in the Williams series expansion of the displacement field around a crack tip.
    The function is expressed in terms of complex coordinates and material properties.

    Args:
        n (int): The order of the term in the Williams series expansion.
        z (complex): Complex coordinate relative to the crack tip.
        mu (float): Shear modulus of the material.
        ka (float): Kolosov constant, which depends on the material properties and the 2D assumption.

    Returns:
        complex: The Williams series function for mode III at the given point.

    Notes:
        - The function uses polar coordinates derived from the complex coordinate z.
        - The result is a complex number representing the function in the series expansion.
        - The function is JIT-compiled using JAX for efficient execution.
        - The ka parameter is included for consistency with the other modes but is not used in the calculation.
    """
    # Compute the polar coordinates
    r = jnp.abs(z)
    theta = jnp.angle(z)
    # Compute the factor
    return (
        r ** (n / 2)
        / (2 * mu * jnp.sqrt(2 * jnp.pi))
        * jnp.sin(n / 2 * theta + (1 + (-1) ** n) / 2 * jnp.pi / 2)
    )
