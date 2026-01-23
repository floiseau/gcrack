# Single edge cracked circular (Hao et al., 2023)

## Presentation

This example is based on the work of (Hao et al., 2023).
It serves as a validation for this code.
The problem configuration (geometry, loading, etc.) is given in (Hao et al. 2023).

## Mesh

An illustration of the geometry of the mesh (and the name of the points and lines) is available in the document `mesh.pdf`.

## Experimental results

The images with the crack path, extracted from Hao et al. (2023), are given in the `reference` folder.
The crack paths have been extracted using the [https://plotdigitizer.com/app](https://plotdigitizer.com/app).

## Running the simulation and displaying the results

The simulations can be run using the makefile with the command

```shell
make simulation_alpha_N
```
where `N` is to be replace with the loading angle value (15, 30, 45, 60 or 75).
To use the scripts to display the results, the results folders must be renamed to `alpha_N`, where`N` is to be replaced with the loading angle value. 

Two scripts are available to display the results.
The first one compares the simulation crack path for a loading angle of 60° to the experimentals ones from the work of Hao et al. (2023).
It can be run using
```shell
make display_comparison_60
```
The second script compared the simulated crack path to the experimental ones for the angles 15, 30, 45, 60, and 75.
This script can be run using the command
```shell
make display_comparison_all
```

## Observations

We observe that the early crack path is well-predicted.
However, the error grows when we get close to the boundary.

## References

Hao, L., Yu, H., Shen, Z., Zhu, S., Wang, B., Huang, C., & Guo, L. (2023). Determination of mode-II critical energy release rate using mixed-mode phase-field model. Theoretical and Applied Fracture Mechanics, 125, 103840. https://doi.org/10.1016/j.tafmec.2023.103840
