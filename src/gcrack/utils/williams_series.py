import jax
import jax.numpy as jnp


@jax.jit
def Gamma_I(n: int, z: complex, mu: float, ka: float):
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
    # Compute the polar coordinates
    r = jnp.abs(z)
    theta = jnp.angle(z)
    # Compute the factor
    return (
        r ** (n / 2)
        / (2 * mu * jnp.sqrt(2 * jnp.pi))
        * jnp.sin(n / 2 * theta + (1 + (-1) ** n) / 2 * jnp.pi / 2)
    )
