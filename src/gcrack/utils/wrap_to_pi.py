from math import pi


def wrap_to_pi(phi):
    """
    Wrap phi to the range [-π, π].

    Parameters
    ----------
    phi : array_like
        Angle(s) to be wrapped, in radians.

    Returns
    -------
    ndarray
        Angle(s) wrapped to the range [-π, π].
    """
    return ((phi + pi) % (2 * pi)) - pi
