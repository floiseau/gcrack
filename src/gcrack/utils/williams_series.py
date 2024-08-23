import numpy as np


def Gamma_I(n: int, z: complex, mu: float, ka: float):
    # Compute the polar coordinates
    r = np.abs(z)
    theta = np.angle(z)
    # Compute the factor
    return (
        r ** (n / 2)
        / (2 * mu * np.sqrt(2 * np.pi))
        * (
            ka * np.exp(1j * theta * n / 2)
            - n / 2 * np.exp(1j * theta * (2 - n / 2))
            + (n / 2 + (-1) ** n) * np.exp(-1j * n * theta / 2)
        )
    )


def Gamma_II(n: int, z: complex, mu: float, ka: float):
    # Compute the polar coordinates
    r = np.abs(z)
    theta = np.angle(z)
    # Compute the factor
    return (
        1j
        * r ** (n / 2)
        / (2 * mu * np.sqrt(2 * np.pi))
        * (
            ka * np.exp(1j * theta * n / 2)
            + n / 2 * np.exp(1j * theta * (2 - n / 2))
            + (-n / 2 + (-1) ** n) * np.exp(-1j * n * theta / 2)
        )
    )
