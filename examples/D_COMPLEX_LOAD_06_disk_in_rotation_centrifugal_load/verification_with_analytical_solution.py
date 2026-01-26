import numpy as np


# Geometry
a = 0.1
b = 1.0
# Mechanical parameters
rho = 1.0
E = 1.0
nu = 0.3
G = E / (2 * (1 + nu))
# Load
wd = 1.0

print("=== Verification of centrifugal load")
# Read the numerical solution
sol = np.loadtxt(
    "results_numeric_centrifugal/results.csv",
    delimiter=",",
    dtype=float,
    skiprows=1,
)
lamb = sol[1, 3]
ur_scaled = sol[1, 7]  # ux is ur
ut_scaled = sol[1, 8]  # uy is ut
ut_num = ut_scaled / lamb
print(f"Numerical  maximum tangential displacement is : {ut_num:.3g}m")

# Compute the analytical solution
# Equation 13 for r=b in
# Tang, S. (1970). Note on acceleration stress in a rotating disk. International Journal of Mechanical Sciences, 12(2), 205–207. https://doi.org/10.1016/0020-7403(70)90020-2
ut_ana = -rho * wd * b**3 / (8 * G) * ((b**2 / a**2 + a**2 / b**2) - 2)

print(f"Analytical maximum tangential displacement is : {ut_ana:.3g}m")
