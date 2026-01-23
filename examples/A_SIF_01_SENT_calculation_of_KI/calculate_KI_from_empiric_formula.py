from math import sqrt, pi

# NOTE : This is based on equations 34 and 35 in https://onlinelibrary.wiley.com/doi/epdf/10.1111/ffe.13994.

### Parameters
# Geometric
W = 1.0
a = 0.5 * W
# Mechanical
E = 1.0
# Loading
fimp = 1
sig = fimp / W

### Compute KIc
aW = a / W
YI = 1.122 - 0.231 * aW + 10.55 * aW**2 - 21.71 * aW**3 + 30.382 * aW**4

print(f"Empiric model : KI = {sig * sqrt(pi * a) * YI:.3g}")
print("Note that the KII in the simulation is due to the asymmetry of the BCs.")
