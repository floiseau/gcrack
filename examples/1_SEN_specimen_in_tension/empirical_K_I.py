from math import sqrt, pi

### Parameters
# Geometric
W = 1e-3
a = W / 2
# Mechanical
E = 230.77e9
# Loading
uimp = 1
sig = E * uimp / W

### Compute KIc
aW = a / W
f = 1.122 - 0.231 * aW + 10.55 * aW**2 - 21.71 * aW**3 + 30.382 * aW**4

print(f"KIc = {sig * sqrt(pi*a) * f:.3g}")
