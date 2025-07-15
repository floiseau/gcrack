from math import isnan
import sympy as sp

from dolfinx import fem


def parse_expression(value, space):
    # Check if the DOF is imposed
    if isinstance(value, str):
        # Parse the function
        x = sp.Symbol("x")
        # Parse the expression using sympy
        par_lambda = sp.utilities.lambdify(x, value, "numpy")
        # Create and interpolate the fem function
        func = fem.Function(space)
        func.interpolate(lambda xx: par_lambda(xx))
    elif isnan(value):
        return None
    else:
        # Define an FEM function (to control the BC)
        func = fem.Function(space)
        # Update the load
        with func.x.petsc_vec.localForm() as local_func:
            local_func.set(value)
    return func
