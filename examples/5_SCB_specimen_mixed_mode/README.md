# Cracked semi-circular bend (SCB) specimen

## Experimental study of Ayatollahi *et al.* (2006)

The reference for this study is the work of Ayatollahi *et al.* (2006).
They studied mixed-mode fracture using cracked semi-circular bend (SCB) specimen.
The geometry of the specimen and the boundary conditions are specified in their work.
Their experimental results are used here as a reference.

## Objective

Initially, this example only aimed at validating the `gcrack` code.
When performing those tests, I found that the choice of boundary conditions seems to highly influence the results.
Thus, this example is a study the influence of the choice of boundary conditions on the resulting crack path.

## Sets of boundary conditions

As specified above, different sets of boundary conditions are used.
Each simulation is run for 90 load steps (except the set 3 which fails at step 88).
They are described in the table below.

| Set | Folder name | Bottom left               | Bottom right                       | Top                                 |
|-----|-------------|---------------------------|------------------------------------|-------------------------------------|
| 1   | `set_1`     | $\boldsymbol{u} = [0, 0]$ | $\boldsymbol{u} = [\text{nan}, 0]$ | $\boldsymbol{u} = [\text{nan}, -1]$ |
| 2   | `set_2`     | $\boldsymbol{u} = [0, 0]$ | $\boldsymbol{u} = [\text{nan}, 0]$ | $\boldsymbol{u} = [0, -1]$          |
| 3   | `set_3`     | $\boldsymbol{u} = [0, 0]$ | $\boldsymbol{u} = [0, 0]$          | $\boldsymbol{u} = [\text{nan}, -1]$ |
| 4   | `set_4`     | $\boldsymbol{u} = [0, 0]$ | $\boldsymbol{u} = [\text{nan}, 0]$ | $\boldsymbol{f} = [0, -1]$          |
| 5   | `set_5`     | $\boldsymbol{u} = [0, 0]$ | $\boldsymbol{u} = [0, 0]$          | $\boldsymbol{f} = [0, -1]$          |

The output folder is specified so that the script to display results (`display_results.py`) can read those results.

*Note 1: The simulation with set 3 fails *

*Note 2: During the simulation, the exact value of the boundary condition are mutliplied by the load factor. The load factor is obtained from the GMERR criterion.*

## How to use

The set of boundary conditions can be chosen directly in `run.py` by commenting and uncommenting in the `define_imposed_...` functions.

To start the simulation, one must load the environment and run the command `make simulation`.

The results are displayed and compared to the experimental results when running the command `make display_results`.

## Results

In the results, three plots are displayed.
First, the evolution of the load factor $\lambda$ with the crack length $a$.
Note that the load factor is higher for the force boundary condition which is totally fine (an displacement of 1 creates a reaction  $F \approx 1 \times E$).
The force-displacement curve is also plotted.
Yet it is complicated to analyze.
The main comparison point is the crack path, which is the 3rd displayed figure.
It can be observed that the set of boundary conditions 3, 4, and 5 lead to incorrect crack path.
The sets 1 and 2 leads to more reasonable results that compares well with numerical results of Ayatollahi et al. (2006).
However, a small discrepancy with the experimental results is observed.

Those results seems to be sufficient to partially validate this code.
Those results could be completed with the other pre-crack orientations ($\beta$) and using the set 1 of boundary conditions.

## References

- Ayatollahi, M. R., Aliha, M. R. M., & Hassani, M. M. (2006). Mixed mode brittle fracture in PMMA—An experimental study using SCB specimens. Materials Science and Engineering: A, 417(1), 348–356. [https://doi.org/10.1016/j.msea.2005.11.002](https://doi.org/10.1016/j.msea.2005.11.002)
