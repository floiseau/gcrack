# Single Edge Notched Specimen in Tension
This example illustrates crack propagation in an anisotropic material with strong fracture anisotropy.
The primary change compared to isotropic fracture is the introduction of a dependency on the crack propagation angle in the critical energy release rate $G_c$.

In this example, we consider a four-fold anisotropy with cusps (spikes in \( G_c \)) defined as follows:
```python
def Gc(self, phi):
    # Retrieve parameters
    Gc_min = self.pars["Gc_min"]
    Gc_max = self.pars["Gc_max"]
    theta0 = self.pars["theta0"]

    # Define the expression for the energy release rate
    return Gc_min + (Gc_max - Gc_min) * jnp.abs(jnp.sin(2 * (phi - theta0)))
```
For this simulation, we selected the following parameters :

- $G_{c,\mathrm{min}} = 10^5$ J/m$^2$,
- $G_{c,\mathrm{max}} = 2 \times 10^5$ J/m$^2$,
- $\theta_0 = 25°$ (material orientation).

Hence, the crack is expected to propagate with an angle $\varphi=25°$.

