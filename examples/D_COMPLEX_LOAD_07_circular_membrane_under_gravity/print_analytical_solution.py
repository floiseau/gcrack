# Parameters
E = 1
nu = 0.3
mu = E / (2 * (1 + nu))
R = 1
rho = 1.0
g = -9.81


# Display the analytical solution
print("The analytical solution is")
print("uz(r) = 1/4 * rho * g / mu * (r**2 - R**2).")

print("The maximum deflection is at r=0 for a displacement")
uz_max = -1 / 4 * rho * g / mu * R**2
print(f"uz(r=0) = {uz_max}.")
print("This value is to be compared to the max in the VTK files")
