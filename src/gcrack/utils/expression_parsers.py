"""
Module for parsing mathematical expressions into finite element functions.

This module provides a utility function to parse string expressions or numerical values into finite element functions.
It supports both symbolic expressions (as strings) and numerical values, converting them into appropriate FEniCSx finite element functions.

Functions:
    parse_expression(value, space): Parses a value into a finite element function.
"""

from math import isnan
import sympy as sp

from dolfinx import fem


def parse_expression(value, space: fem.FunctionSpace, export_func: bool = False):
    """Parses a value into a finite element function.

    This function converts a string expression or numerical value into a finite element function.
    If the input is a string, it is parsed as a symbolic expression and interpolated onto the provided function space.
    If the input is a numerical value, it is used to create a constant finite element function.
    If the input is NaN, the function returns None.

    Args:
        value:
            The value to parse.
            Can be a string expression (e.g., "x**2 + 1") or a numerical value.
            If NaN, the function returns None.
        space (fem.FunctionSpace):
            The finite element function space onto which the expression or value should be interpolated.
        export_func (bool):
            Flag indicating if the parameter function must be exported.


    Returns:
        func (fem.Function or None):
            A finite element function representing the parsed expression or value.
            Returns None if the input value is NaN.

    Example:

        from dolfinx import fem
        from gcrack.utils.expression_parsers import parse_expression
        mesh = ...  # Create a mesh
        V = fem.functionspace(mesh, ("Lagrange", 1))
        # Parse a string expression
        f1 = parse_expression("x**2 + 1", V)
        # Parse a numerical value
        f2 = parse_expression(5.0, V)
    """
    if isinstance(value, (int, float)):
        # Check if the DOF is imposed
        if isnan(value):
            return None
        # Define an FEM function (to control the BC)
        func = fem.Function(space)
        # Update the load
        with func.x.petsc_vec.localForm() as local_func:
            local_func.set(value)
        # Create the par_func if necessary
        if export_func:

            def par_func(xx):
                return value
    elif isinstance(value, str):
        # Parse the function
        x = sp.Symbol("x")
        # Parse the expression using sympy
        par_func = sp.utilities.lambdify(x, value, "numpy")
        # Create and interpolate the fem function
        func = fem.Function(space)
        func.interpolate(lambda xx: par_func(xx))
    else:
        raise ValueError("Unknown type passed to a parsed expression.")

    if not export_func:
        return func
    else:
        return func, par_func
