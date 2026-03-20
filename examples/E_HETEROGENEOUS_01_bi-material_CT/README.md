# Heterogeneous Compact Tension specimen (bi-material)
This example illustrates crack propagation in an heterogeneous material.
The primary change compared to classic simulation is the introduction of a dependency on the space coordinates in the elastic properties.

In this example, we consider a bi-material specimen where each material has its own Young modulus.
This heterogeneous Young modulus is passed as a string (which that is latter parsed using `sympy`):
```python
pars["E"] = f"1e9 * heaviside(x[1]-{pars['H'] / 4}, 0) + 1e9"
```
Note that this heterogeneity induces a disymmetry in the elastic problem 

**IMPORTANT REMARK.** As explained in the warning when running the simulation, the elastic properties used in the calculation of the SIFs are the ones at the crack tip.
It means that heterogeneous simulations are valid only when elastic properties are constant in the pacman.

