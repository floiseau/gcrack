"""
Module for defining the elastic model.

This module provides the `ElasticModel` class, which encapsulates the material properties and mechanical behavior of elastic materials.
It supports both homogeneous and heterogeneous material properties, as well as different 2D assumptions (plane stress, plane strain, anti-plane).
The class also provides methods for computing displacement gradients, strain tensors, stress tensors, and elastic energy.
"""

import sympy as sp

from dolfinx import fem
import ufl


class ElasticModel:
    """Class for defining an elastic material model in finite element simulations.

    This class encapsulates the material properties and mechanical behavior of elastic materials.
    It supports both homogeneous and heterogeneous material properties, as well as different 2D assumptions (plane stress, plane strain, anti-plane).
    The class provides methods for computing displacement gradients, strain tensors, stress tensors, and elastic energy.

    Attributes:
        E (float or dolfinx.Function): Young's modulus.
        nu (float or dolfinx.Function): Poisson's ratio.
        la (float or dolfinx.Function): Lame coefficient lambda.
        mu (float or dolfinx.Function): Lame coefficient mu.
        assumption (str): 2D assumption for the simulation (e.g., "plane_stress", "plane_strain", "anti_plane").
        Ep (float): Plane strain modulus.
        ka (float): Kolosov constant.
    """

    def __init__(self, pars, domain=None):
        """Initializes the ElasticModel.

        Args:
            pars (dict): Dictionary containing parameters of the material model.
                Required keys: "E" (Young's modulus), "nu" (Poisson's ratio), and "2D_assumption" (2D assumption).
            domain (fragma.Domain.domain, optional): Domain object, it is only used to initialize heterogeneous properties.
                Defaults to None.
        """
        # Get elastic parameters
        self.E = self.parse_parameter(pars["E"], domain)
        self.nu = self.parse_parameter(pars["nu"], domain)
        # Compute Lame coefficient
        self.la = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))
        # Check the 2D assumption
        if domain is not None and domain.dim == 2:
            self.assumption = pars["2D_assumption"]
            match self.assumption:
                case "plane_stress":
                    self.Ep = self.E
                    self.ka = (3 - self.nu) / (1 + self.nu)
                case "plane_strain":
                    self.Ep = self.E / (1 - self.nu**2)
                    self.ka = 3 - 4 * self.nu
                case "anti_plane":
                    print(
                        "For anti-plane, we assume plane strain for SIF calculations."
                    )
                    self.Ep = self.E
                    self.ka = (3 - self.nu) / (1 + self.nu)
                case _:
                    raise ValueError(
                        f'The 2D assumption "{self.assumption}" is unknown.'
                    )

    def parse_parameter(self, par, domain):
        """Parses a material parameter.

        If the parameter is a number (integer or float), it is returned as is.
        If the parameter is a mathetematical expression (str), it is parsed using sympy and represented as a finite element function.

        Args:
            par (int, float, or str): The parameter to parse.
            domain (fragma.Domain.domain): The domain on which to interpolate the parsed parameter.

        Returns:
            int, float, or dolfinx.Function: The parsed parameter.
                If the parameter is a number, it is returned as is.
                If it is a SymPy expression, it is represented as a finite element function.
        """
        # Check if the parameter is a number
        if isinstance(par, (int, float)):
            # Return the parameter as is
            return par
        else:
            # Declare the coordinate symbol
            x = sp.Symbol("x")
            # Parse the expression using SymPy
            par_lambda = sp.utilities.lambdify(x, par, "numpy")
            # Define the function space
            V_par = fem.FunctionSpace(domain.mesh, ("DG", 0))
            # Create the finite element function
            par_func = fem.Function(V_par)
            par_func.interpolate(par_lambda)
            # Return the finite element function
            return par_func

    def u_to_3D(self, u: fem.Function) -> ufl.classes.Expr:
        """Converts a 2D displacement field to its 3D version.

        The conversion depends on the 2D assumption (plane stress, plane strain, or anti-plane).

        Args:
            u (fem.Function): FEM function of the displacement field.

        Returns:
            ufl.classes.Expr: Displacement in 3D.
        """
        if self.assumption.startswith("plane"):
            return ufl.as_vector([u[0], u[1], 0])
        elif self.assumption == "anti_plane":
            return ufl.as_vector([0.0, 0.0, u[0]])
        else:
            raise ValueError(f"Unknown 2D assumption: {self.assumption}.")

    def grad_u(self, u: fem.Function) -> ufl.classes.Expr:
        """Computes the gradient of the displacement field.

        Args:
            u (fem.Function): FEM function of the displacement field.

        Returns:
            ufl.classes.Expr: Gradient of the displacement field in 3D.
        """
        # Convert the displacement to 3D
        u3D = self.u_to_3D(u)
        # Compute the 2D gradient of the field
        g_u3D = ufl.grad(u3D)
        # Construct the strain tensor
        if self.assumption.startswith("plane"):
            eps_zz = (
                -self.la / (2 * self.mu + self.la) * (g_u3D[0, 0] + g_u3D[1, 1])
                if self.assumption == "plane_stress"
                else 0
            )
            grad_u3D = ufl.as_tensor(
                [
                    [g_u3D[0, 0], g_u3D[0, 1], 0],
                    [g_u3D[1, 0], g_u3D[1, 1], 0],
                    [0, 0, eps_zz],
                ]
            )
        elif self.assumption == "anti_plane":
            grad_u3D = ufl.as_tensor(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [g_u3D[2, 0], g_u3D[2, 1], 0],
                ]
            )
        # Return the gradient
        return grad_u3D

    def eps(self, u: fem.Function) -> ufl.classes.Expr:
        """Computes the strain tensor.

        Args:
            u (fem.Function): FEM function of the displacement field.

        Returns:
            ufl.classes.Expr: Strain tensor.
        """
        # Symmetrize the gradient
        return ufl.sym(self.grad_u(u))

    def sig(self, u: fem.Function) -> ufl.classes.Expr:
        """Computes the stress tensor.

        Args:
            u (fem.Function): FEM function of the displacement field.

        Returns:
            ufl.classes.Expr: Stress tensor.
        """
        # Get elastic parameters
        mu, la = self.mu, self.la
        # Compute the stress
        eps = self.eps(u)
        return la * ufl.tr(eps) * ufl.Identity(3) + 2 * mu * eps

    def elastic_energy(self, u, domain):
        """Computes the elastic energy.

        Args:
            u (fem.Function): FEM function of the displacement field.
            domain (fragma.Domain.domain): The domain object representing the computational domain.

        Returns:
            ufl.classes.Expr: Elastic energy.
        """
        # Get the integration measure
        dx = ufl.Measure("dx", domain=domain.mesh, metadata={"quadrature_degree": 12})
        # Compute the stress
        sig = self.sig(u)
        # Compute the strain
        eps = self.eps(u)
        # Define the total energy
        return 1 / 2 * ufl.inner(sig, eps) * dx
