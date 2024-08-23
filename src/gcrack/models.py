import sympy as sp

from dolfinx import fem
import ufl


class ElasticModel:
    """
    Class for defining an elastic material models.

    Parameters
    ----------
    pars : dict
        Dictionary containing parameters of the material model.

    Attributes
    ----------
    la : dolfinx.Constant
        Lame coefficient lambda.
    mu : dolfinx.Constant
        Lame coefficient mu.
    """

    def __init__(self, pars, domain=None):
        """
        Initialize the ElasticModel.

        Parameters
        ----------
        pars : dict
            Dictionary containing parameters of the material model.
        domain : fragma.Domain.domain
            Domain object used to initialize heterogeneous properties.
        """
        # Get elastic parameters
        self.E = self.parse_parameter(pars["E"], domain)
        self.nu = self.parse_parameter(pars["nu"], domain)
        # Compute Lame coefficient
        self.la = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))
        # Check the 2D assumption
        if domain.dim == 2:
            self.assumption = pars["2D_assumption"]
            match self.assumption:
                case "plane_stress":
                    self.la = 2 * self.mu * self.la / (self.la + 2 * self.mu)
                    self.Ep = self.E
                    self.ka = (3 - self.nu) / (1 + self.nu)
                case "plane_strain":
                    self.Ep = self.E / (1 - self.nu**2)
                    self.ka = 3 - 4 * self.nu
                case _:
                    raise ValueError(
                        f'The 2D assumption "{self.assumption}" in unknown'
                    )

    def parse_parameter(self, par, domain):
        """
        Parse the given parameter.

        If the parameter is a number (integer or float), returns the raw number.
        Otherwise, it interprets the parameter as a mathematical expression,
        parses it using SymPy, and creates a finite element function representing
        the parsed expression on the given domain.

        Parameters
        ----------
        par : int, float, or sympy.Expr
            The parameter to parse. If it's a number, it will be returned as is.
            If it's a SymPy expression, it will be parsed and represented as a
            finite element function.
        domain : fragma.Domain.domain
            The domain on which to interpolate the parsed parameter.

        Returns
        -------
        par_value : int, float, or dolfinx.Function
            The parsed parameter. If the parameter is a number, it will be returned
            as is. If it's a SymPy expression, it will be represented as a finite
            element function.
        """
        # Check if the parameter is a number
        if isinstance(par, (int, float)):
            # Return the parameter as is
            return par
        else:
            # Declare the coordinate symbol
            x = sp.Symbol("x")
            # Parse the expression using sympy
            par_lambda = sp.utilities.lambdify(x, par, "numpy")
            # Define the function space
            par_elem = ufl.FiniteElement("DG", domain.mesh.ufl_cell(), 0)
            V_par = fem.FunctionSpace(domain.mesh, par_elem)
            # Create the fem function
            par_func = fem.Function(V_par)
            par_func.interpolate(par_lambda)
            # Return the fem function
            return par_func

    def eps(self, u: fem.Function) -> ufl.classes.Expr:
        """
        Compute the strain tensor.

        Parameters
        ----------
        u : fem.Function
            FEM function of the displacement field.

        Returns
        -------
        ufl.form.Expression
            Strain tensor.
        """
        return ufl.sym(ufl.grad(u))

    def sig(self, u: fem.Function) -> ufl.classes.Expr:
        """
        Compute the stress tensor.

        Parameters
        ----------
        u : fem.Function
            FEM function of the displacement field.

        Returns
        -------
        ufl.form.Expression
            Stress tensor.
        """
        # Get elastic parameters
        mu, la = self.mu, self.la
        # Compute the stess
        return la * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2.0 * mu * self.eps(u)
