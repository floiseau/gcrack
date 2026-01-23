"""
Module for defining the elastic model.

This module provides the `ElasticModel` class, which encapsulates the material properties and mechanical behavior of elastic materials.
It supports both homogeneous and heterogeneous material properties, as well as different 2D assumptions (plane stress, plane strain, anti-plane).
The class also provides methods for computing displacement gradients, strain tensors, stress tensors, and elastic energy.
"""

import sympy as sp

from dolfinx import fem
import ufl

from gcrack.utils.expression_parsers import parse_expression


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
        # Display warnings if necessary
        self.displays_warnings(pars)
        # Define a function space for parameter parsing
        V_par = fem.functionspace(domain.mesh, ("DG", 0))
        # Get elastic parameters
        self.E, self.E_func = parse_expression(pars["E"], V_par, export_func=True)
        self.nu, self.nu_func = parse_expression(pars["nu"], V_par, export_func=True)
        # Compute Lame coefficient
        self.la = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))
        self.mu_func = lambda xx: self.E_func(xx) / (2 * (1 + self.nu_func(xx)))
        # Check the 2D assumption
        if domain is not None and domain.dim == 2:
            self.assumption = pars["2D_assumption"]
            match self.assumption:
                case "plane_stress" | "anti_plane":
                    self.Ep = self.E
                    self.Ep_func = lambda xx: self.E_func(xx)
                    self.ka = (3 - self.nu) / (1 + self.nu)
                    self.ka_func = lambda xx: (3 - self.nu_func(xx)) / (
                        1 + self.nu_func(xx)
                    )
                    if self.assumption == "anti_plane":
                        print(
                            "│  For anti-plane, we assume plane stress for SIF calculations."
                        )
                case "plane_strain":
                    self.Ep = self.E / (1 - self.nu**2)
                    self.Ep_func = lambda xx: self.E_func(xx) / (
                        1 - self.nu_func(xx) ** 2
                    )
                    self.ka = 3 - 4 * self.nu
                    self.ka_func = lambda xx: 3 - 4 * self.nu_func(xx)
                case _:
                    raise ValueError(
                        f'The 2D assumption "{self.assumption}" is unknown.'
                    )

    def displays_warnings(self, pars: dict):
        """Check the parameters and display warnings if necessary.

        Args:
            pars (dict): Dictionary of the model parameters.
        """
        # Check potential triggers
        heterogeneous_properties = isinstance(pars["E"], str) or isinstance(
            pars["nu"], str
        )
        # Display the warning in crack of heterogeneous properties
        if heterogeneous_properties:
            print("""│  WARNING: USE OF HETEROGENEOUS ELASTIC PROPERTIES
│  │  A string has been passed for the elastic properties (E or nu).
│  │  It means the simulation includes heterogeneous elastic properties.
│  │  Note that, when calculating the SIFs, the elastic properties are assumed to be:
│  │      (1) homogeneous, and
│  │      (2) equal to the elastic properties at the crack tip.
│  │  The elastic properties variations must be negligible or null in the pacman.
│  │  If the elastic properties  are constant, use floats to disable this message.""")

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
        match self.assumption:
            case "plane_strain":
                grad_u3D = ufl.as_tensor(
                    [
                        [g_u3D[0, 0], g_u3D[0, 1], 0],
                        [g_u3D[1, 0], g_u3D[1, 1], 0],
                        [0, 0, 0],
                    ]
                )
            case "plane_stress":
                eps_zz = -self.nu / (1 - self.nu) * (g_u3D[0, 0] + g_u3D[1, 1])
                grad_u3D = ufl.as_tensor(
                    [
                        [g_u3D[0, 0], g_u3D[0, 1], 0],
                        [g_u3D[1, 0], g_u3D[1, 1], 0],
                        [0, 0, eps_zz],
                    ]
                )
            case "anti_plane":
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
        dx = ufl.Measure("dx", domain=domain.mesh, metadata={"quadrature_degree": 6})
        # Compute the stress
        sig = self.sig(u)
        # Compute the strain
        eps = self.eps(u)
        # Define the total energy
        return 1 / 2 * ufl.inner(sig, eps) * dx
