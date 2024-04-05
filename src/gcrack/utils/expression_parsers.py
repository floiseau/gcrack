import sympy as sp


def get_gc_function(gc_pars):
    # Get the expression
    gc_str = gc_pars["Gc"]
    # Perform the substitution of the variables
    for symb, val in gc_pars.items():
        if symb == "Gc":
            continue
        gc_str = gc_str.replace(symb, str(val))
    # Convert it into an expression
    phi_symb = sp.symbols("phi")
    gc_expr = sp.sympify(gc_str, locals={"phi": phi_symb})
    return sp.lambdify([phi_symb], gc_expr)
