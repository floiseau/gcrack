# Overview

## Description

[`gcrack`](https://github.com/floiseau/fragma) is a 2D **finite element solver** for simulating crack propagation using **Linear Elastic Fracture Mechanics**.
It is built on top of [`fenicsx`](https://fenicsproject.org/).

It has several non-classic features :

- crack propagation in anistropic media,
- indirect load control using path-following methods.

---

## Installation

To run [`gcrack`](https://github.com/floiseau/gcrack), install the required Python modules in a dedicated environment:

1. **Create and activate a new conda environment**:

        conda create -n gcrack
        conda activate gcrack

2. **Install the required dependencies**:

        conda install -c conda-forge numpy sympy mpich python-gmsh fenics-dolfinx pyvista jax jaxlib=*=*cpu*

3. **Install gcrack**:

        pip install .

---

## Hands-On: Test with Examples

The [`examples`](https://github.com/floiseau/gcrack/tree/main/examples) directory contains ready-to-run scripts and parameter files.
Just follow these steps:

1. Navigate to the [`examples`](https://github.com/floiseau/gcrack/tree/main/examples) directory.

        cd gcrack/examples/example_name

2. Activate the conda environment:

        conda activate gcrack

3. Run the provided `run.py` script using the provided `makefile`:

        make simulation

4. Vizualize the results in the directory `results_YYY-MM-DD-HH-mm-ss/`.

---

## Usage

### Requirements
To run a simulation, the only required file is a Python script (named `run.py` in the examples).
It contains the definition of a class inheriting from `GCrackBase`.
In this class, the methods to define the mesh (`generate_mesh`), to define the boundary conditions (*e.g.*, `define_controlled_displacements`), and define the critical energy release rate (`Gc`) must be defined.

**TODO: Add the link to the functions in the references!!!!**

### Running the Solver

Once the Python script is define, one only need to call the method `GCrackBase.run` to start the resolution.

1. Activate the `gcrack` environment:

        conda activate gcrack

3. Run the solver:

        python run.py

!!! Notes
    - On some Linux distributions, set `OMP_NUM_THREADS=1` to prevent FEniCSx from using all available threads:

            OMP_NUM_THREADS=1 python run.py

### Outputs

Results are saved in the `results` subdirectory.
Two types of file are exported:

- The displacement fields are saved in `VTK` files (with the extension `.pvd `).
- The scalar outputs (displacement/force measures, stress intensity factors, etc.) are saved in the `results.csv` CSV file.

!!! Vizualization tools
    To open the `VTK` files, vizualization tools such as [Paraview](https://www.paraview.org/) or [pyvsita](https://docs.pyvista.org/index.html) can be employed.
