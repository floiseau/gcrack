import numpy as np
from scipy.optimize import differential_evolution


def residual(x, g_vec, gc_func):
    gamma = x[0]
    t = np.array([np.cos(gamma), np.sin(gamma)])
    g = np.maximum(np.dot(g_vec, t), 1e-12)
    gc = gc_func(gamma)
    return np.sqrt(gc / g)


def compute_load_factor(gamma0: float, g_vec, gc_func):
    return differential_evolution(
        residual,
        bounds=[(gamma0 - np.pi, gamma0 + np.pi)],
        args=(g_vec, gc_func),
        popsize=128,
    )
