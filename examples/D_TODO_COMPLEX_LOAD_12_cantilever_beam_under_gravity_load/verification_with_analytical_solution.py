import numpy as np
import matplotlib.pyplot as plt

# Geometry
Lx = 10
Ly = 1
Lz = 1
# Mechanical parameters
E = 1
I = Lz * Ly**3 / 12
# Load
q = -1.0


# X coordinates
x = np.linspace(0, Lx, 101)

# Displacement field
u = q * x**2 * (6 * Lx**2 - 4 * Lx * x + x**2) / (24 * E * I)

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
