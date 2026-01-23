# Stress Intensity Factor (SIF) Analysis for the Center Crack Tensile specimen

## Physical Problem
This example solves the 2D elastic fracture mechanics problem of a rectangular domain with a central crack, subjected to tensile loading.
The crack is modeled at varying angles $\alpha$, and the Stress Intensity Factors $K_{I}, K_{II}, and $T$-stress are computed for each configuration.

### Key Aspects:
- **Domain**: Rectangular plate with length `L` and width `W`.
- **Crack**: Centered, symmetric, with initial length `a0` and variable angle `α`.
- **Loading**: Uniform tension applied at top/bottom boundaries.
- **Methods**: SIFs are computed using both the interaction integral and Williams series interpolation.

### Outputs:
- Normalized SIFs (`KI`, `KII`, `T-stress`) vs. crack angle.
- Comparison with analytical solutions for validation.

---

## Running this example

```shell
conda activate gcrack
make simulation
python display_results.py
```
