# Mixed-mode crack propagation (Réthoré, 2018)

## Presentation

This example is based on the open dataset of Réthoré (2018).
It serves as a validation for this code.
The geometry is given in Zenodo page of the dataset.

## Mesh

The generation of the mesh is briefly illustrated in the document `geo_mat_mesh.pdf`.

## Experimental results

The experimental results from Réthoré (2018) are given in the reference folder.

## Running the simulation and displaying the results

The simulation can be run using the available makefile with the command

```shell
make simulation
```

Once the simulation is run, the results can be shown using the command `make display_results`.

## Observations

We observe that the simulated crack is similar to the experimental one.
However, a kind of offset is observed.
Different causes are possible:

- The numerical boundary conditions does not match the experimental ones (contact) .
- The crack increment $\Delta a$ (`da` in the code) is to large.
- Another criterion should be used.

## References

Réthoré, J. (2018). PMMA Mixed mode fracture (Version 1.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.1473126
