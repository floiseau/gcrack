import numpy as np
import matplotlib.pyplot as plt

# Aide-mémoire Mécanique des structures - 3e édition
# Résistance des matériaux - IUT-Licence-Master
# Arnaud Delaplace, Fabrice Gatuingt, Frédéric Ragueneau
# Page 59 : https://www.reeflexion.fr/wp-content/uploads/articles/cuveaqua1/Mecaniquedesstructures.pdf


# Geometry
Lx = 10
Ly = 1
Lz = 1
# Mechanical parameters
E = 1
I = Lz * Ly**3 / 12
# Max of the load
P = -10.0


# X coordinates
x = np.linspace(0, Lx, 101)

# Displacement field
u = P * x**2 * (20 * Lx**3 - 10 * Lx**2 * x + x**3) / (120 * E * I * Lx)

# Display the solution
if False:
    plt.figure()
    plt.plot(x, u)
    plt.xlabel(r"Coordinate $x$ [m]")
    plt.ylabel(r"Displacement $u$ [m]")
    plt.show()

# Displacement at end point
print(f"Vertical displacement at end point (analytical): {u[-1]}m")

# Check if the numerical solution matches
sol = np.loadtxt(
    "results_numeric/results.csv",
    delimiter=",",
    dtype=float,
    skiprows=1,
)
# The displacement is scaled with a wrong scale factor
# (as there is not crack in the domain)
lamb = sol[1, 3]
uy_scaled = sol[1, 8]
uy_num = uy_scaled / lamb
print(f"Vertical displacement at end point  (numerical): {uy_num}m")
