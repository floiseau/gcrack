# gcrack

**gcrack** is a simulation tool designed to model crack propagation in anisotropic media.
It is based on linear elastic fracture mechanics, specifically the GMERR (Generalized Maximum Energy Release Rate) criterion, to simulate incremental crack propagation with remeshing.

## Features

- **Incremental Propagation**: Simulates crack growth step-by-step, with an explicit remeshing of the crack.
- **Linear Elastic Fracture Mechanics**: Utilizes GMERR for determine the direction of crack propagation.
- **Anisotropic Media Simulation**: Accurately models crack propagation in materials with directionally dependent properties.

## Installation

To install gcrack, follow these steps:

1. **Create and activate a new conda environment**:
    ```shell
    conda create -n gcrack
    conda activate gcrack
    ```

2. **Install the required dependencies**:
    ```shell
    conda install -c conda-forge numpy sympy mpich python-gmsh fenics-dolfinx pyvista jax jaxlib=*=*cpu*
    ```

3. **Install gcrack**:
    ```shell
    pip install .
    ```

## How to Use

Examples are provided in the `gcrack/examples/` directory. Each example typically includes:

- **`run.py`**: Contains the problem definition and simulation setup.
- **`display_results.py`**: Script to visualize the simulation results.
- **Makefile**: Automates the simulation process.

To run a simulation, follow these steps:

1. **Activate the conda environment**:
    ```shell
    conda activate gcrack
    ```

2. **Navigate to the example directory** and run the simulation:
    ```shell
    cd gcrack/examples/example_name
    make simulation
    ```
