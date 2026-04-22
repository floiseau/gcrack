"""
Module for defining the elastic model.

This module provides the `ElasticModel` class, which encapsulates the material properties and mechanical behavior of elastic materials.
It supports both homogeneous and heterogeneous material properties, as well as different 2D assumptions (plane stress, plane strain, anti-plane).
The class also provides methods for computing displacement gradients, strain tensors, stress tensors, and elastic energy.
"""

import numpy as np
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
        ka (float or dolfinx.Function): Bulk modulus.
        mu (float or dolfinx.Function): Shear modulus.
        assumption (str): 2D assumption for the simulation (e.g., "plane_stress", "plane_strain", "anti_plane").
        Ep (float): Plane strain modulus.
        ko (float): Kolosov constant.
    """

    def __init__(self, pars, domain):
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

        # Define the elastic properties
        if not (pars.get("E", None) is None and pars.get("nu", None) is None):
            # Determine the elasticity tensor from E and nu
            # Get elastic parameters
            self.E, self.E_func = parse_expression(pars["E"], V_par, export_func=True)
            self.nu, self.nu_func = parse_expression(
                pars["nu"], V_par, export_func=True
            )
            # Compute the harmonic components
            self.ka = self.E / (3 - 6 * self.nu)
            self.mu = self.E / (2 * (1 + self.nu))
            self.d = 0
            self.theta_d = 0
            self.h = 0
            self.theta_h = 0
            # Compute the functions
            self.ka_func = lambda xx: self.E_func(xx) / (3 - 6 * self.nu_func(xx))
            self.mu_func = lambda xx: self.E_func(xx) / (2 * (1 + self.nu_func(xx)))
            self.d_func = lambda xx: self.d
            self.theta_d_func = lambda xx: self.theta_d
            self.h_func = lambda xx: self.h
            self.theta_h_func = lambda xx: self.theta_h
        else:  # From the harmonic components
            print("")
            # Extract the parameters
            self.mu, self.mu_func = parse_expression(
                pars["mu"], V_par, export_func=True
            )
            self.ka, self.ka_func = parse_expression(
                pars["ka"], V_par, export_func=True
            )
            self.d, self.d_func = parse_expression(pars["d"], V_par, export_func=True)
            self.theta_d, self.theta_d_func = parse_expression(
                pars["theta_d"], V_par, export_func=True
            )
            self.h, self.h_func = parse_expression(pars["h"], V_par, export_func=True)
            self.theta_h, self.theta_h_func = parse_expression(
                pars["theta_h"], V_par, export_func=True
            )
            # Define E and nu
            self.E = 9 * self.ka * self.mu / (3 * self.ka + self.mu)
            self.E_func = lambda xx: (
                (9 * self.ka_func(xx) * self.mu_func(xx))
                / (3 * self.ka_func(xx) + self.mu_func(xx))
            )
            self.nu = (3 * self.ka - 2 * self.mu) / (6 * self.ka + 2 * self.mu)
            self.nu_func = lambda xx: (
                (3 * self.ka_func(xx) - 2 * self.mu_func(xx))
                / (6 * self.ka_func(xx) + 2 * self.mu_func(xx))
            )

        # Generate the elasticity tensor
        self.ela = self.elasticity_tensor(domain)

        # Check the 2D assumption for LEFM formulas
        self.assumption = pars["2D_assumption"]
        match self.assumption:
            case "plane_stress" | "anti_plane":
                self.Ep = self.E
                self.Ep_func = lambda xx: self.E_func(xx)
                self.ko = (3 - self.nu) / (1 + self.nu)
                self.ko_func = lambda xx: (
                    (3 - self.nu_func(xx)) / (1 + self.nu_func(xx))
                )
                if self.assumption == "anti_plane":
                    print(
                        "│  For anti-plane, we assume plane stress for SIF calculations."
                    )
            case "plane_strain":
                self.Ep = self.E / (1 - self.nu**2)
                self.Ep_func = lambda xx: self.E_func(xx) / (1 - self.nu_func(xx) ** 2)
                self.ko = 3 - 4 * self.nu
                self.ko_func = lambda xx: 3 - 4 * self.nu_func(xx)
            case _:
                raise ValueError(f'The 2D assumption "{self.assumption}" is unknown.')

    def displays_warnings(self, pars: dict):
        """Check the parameters and display warnings if necessary.

        Args:
            pars (dict): Dictionary of the model parameters.
        """
        # Check potential triggers
        heterogeneous_properties = (
            isinstance(pars.get("E", None), str)
            or isinstance(pars.get("nu", None), str)
            or isinstance(pars.get("mu", None), str)
            or isinstance(pars.get("ka", None), str)
            or isinstance(pars.get("d", None), str)
            or isinstance(pars.get("theta_d", None), str)
            or isinstance(pars.get("h", None), str)
            or isinstance(pars.get("theta_h", None), str)
        )
        anisotropic_elasticity = not ("E" in pars and "nu" in pars)
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
        if anisotropic_elasticity:
            print("""|  WARNING: USE OF ANISOTROPIC ELASTICITY
│  │  The results of gcrack are invalid if the crack propagates in a region with anisotropic
│  │  elastic properties !
│  │  Typical use case : bi-material specimen with an isotropic region (containing the crack) 
│  │                     and an anisotropic region (e.g., structure repaired by 3D printing).""")

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

    def elasticity_tensor(self, domain) -> ufl.classes.Expr:
        print("""
To extend the elasticity tensor in 3D, we assume that it is transversely isotropic,
with the 3rd dimension acting as the first dimension.
              """)
        # Initialize the FEM function for the elasticity tensor
        V_E = fem.functionspace(domain.mesh, ("DG", 0, (3, 3, 3, 3)))
        ela_func = fem.Function(V_E)
        # Define the constants
        Id2 = np.eye(2)  # ufl.Identity(2)
        Id2xId2 = np.einsum("ij,kl->ijkl", Id2, Id2)  # ufl.outer(Id2, Id2)
        Id4 = (1 / 2) * (
            np.einsum("ij,kl->ikjl", Id2, Id2) + np.einsum("ij,kl->iljk", Id2, Id2)
        )
        J = Id4 - 1 / 2 * Id2xId2

        # Calculate the isotropic part
        def iso_func(xx):
            test = np.einsum("a,ijkl->ijkla", 2 * self.mu_func(xx), J) + np.einsum(
                "a,ijkl->ijkla", self.ka_func(xx), Id2xId2
            )
            test_bis = test.reshape(16, xx.shape[1])
            print(test_bis.shape)
            return test_bis

        # Calculate the dilatation part
        def dp(xx):
            d_prim = np.empty((xx.shape[1], 2, 2))
            d_prim[:, 0, 0] = self.d_func(xx) * np.cos(self.theta_d_func(xx))
            d_prim[:, 0, 1] = self.d_func(xx) * np.sin(self.theta_d_func(xx))
            d_prim[:, 1, 0] = d_prim[:, 0, 1]
            d_prim[:, 1, 1] = -d_prim[:, 1, 1]
            return d_prim

        def dil_func(xx):
            return (1 / 2) * (
                np.einsum("ij,akl->aijkl", Id2, dp(xx))
                + np.einsum("ij,akl->aklij", Id2, dp(xx))
            )

        # Calculate the harmonic part
        def har_func(xx):
            # Initialize the array
            H = np.empty((xx.shape[1], 2, 2, 2, 2))

            H[:, 0, 0, 0, 0] = self.h_func(xx) * np.cos(self.theta_h_func(xx))
            H[:, 0, 0, 0, 1] = self.h_func(xx) * np.sin(self.theta_h_func(xx))

            # Account for the total symmetry and the nullity of the traces
            H[:, 1, 1, 1, 1] = H[:, 0, 0, 0, 0]

            H[:, 0, 0, 1, 0] = H[:, 0, 0, 0, 1]
            H[:, 0, 1, 0, 0] = H[:, 0, 0, 0, 1]
            H[:, 1, 0, 0, 0] = H[:, 0, 0, 0, 1]

            H[:, 1, 1, 0, 0] = -H[:, 0, 0, 0, 0]
            H[:, 0, 0, 1, 1] = -H[:, 0, 0, 0, 0]
            H[:, 0, 1, 0, 1] = -H[:, 0, 0, 0, 0]
            H[:, 1, 0, 1, 0] = -H[:, 0, 0, 0, 0]
            H[:, 0, 1, 1, 0] = -H[:, 0, 0, 0, 0]
            H[:, 1, 0, 0, 1] = -H[:, 0, 0, 0, 0]

            H[:, 1, 1, 1, 0] = -H[:, 0, 0, 0, 1]
            H[:, 1, 1, 0, 1] = -H[:, 0, 0, 0, 1]
            H[:, 1, 0, 1, 1] = -H[:, 0, 0, 0, 1]
            H[:, 0, 1, 1, 1] = -H[:, 0, 0, 0, 1]

            return H

        # Interpolate the expression
        print(ela_func.x.array.shape)
        ela_func.interpolate(lambda xx: iso_func(xx))  # + dil_func(xx) + har_func(xx))
        return ela_func

    def sig(self, u: fem.Function) -> ufl.classes.Expr:
        """Computes the stress tensor.

        Args:
            u (fem.Function): FEM function of the displacement field.

        Returns:
            ufl.classes.Expr: Stress tensor.
        """
        # Generate indices
        i, j, k, l = ufl.indices(4)
        # Compute the stress
        eps = self.eps(u)
        return ufl.as_tensor(self.ela[i, j, k, l] * eps[k, l], (i, j))

    def elastic_energy(self, u, domain):
        """Computes the elastic energy.

        Args:
            u (fem.Function): FEM function of the displacement field.
            domain (fragma.Domain.domain): The domain object representing the computational domain.

        Returns:
            ufl.classes.Expr: Elastic energy.
        """
        # Get the integration measure
        dx = ufl.Measure("dx", domain=domain.mesh)
        # , metadata={"quadrature_degree": 6})
        # Compute the stress
        sig = self.sig(u)
        # Compute the strain
        eps = self.eps(u)
        print(sig.ufl_shape)
        print(eps.ufl_shape)
        # Define the total energy
        return 1 / 2 * ufl.inner(sig, eps) * dx
