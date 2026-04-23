# Parallelizing `gcrack` simulations

While the `gcrack` simulation package does not offer direct support for parallel execution, it is possible to run multiple simulations concurrently.

!!! info

    The main difficulty is caused by the Python API for GMSH as it relies on global variables.
    Consequently, directly implementing standard parallelization techniques, such as `multiprocessing`, within the `gcrack` scripts is not feasible.

To circumvent this limitation, we propose the following approach designed to handle the majority of typical use cases.

The recommended method consists of the following two steps:

1. **Parameterize the Run Script:**
   Modify the primary `gcrack` execution script (`run.py`) to accept command-line arguments, enabling modular calls from the terminal (e.g., `python run.py -L 10 -H 5`).

2. **External Parallel Runner:**
   Develop a dedicated script (`run_in_parallel.py`) to invoke `run.py` in parallel. This wrapper manages the execution flow, ensuring each instance of `run.py` receives its required arguments and runs independently.

## Argument parser (using `argparse`)

The first step involves modifying `run.py` to accept arguments via the `argparse` module.
For more information on the `argparse` package, check its documentation at [https://docs.python.org/3/library/argparse.html](https://docs.python.org/3/library/argparse.html).
In the following example, we consider a simulation with two parameters : the length `L` and the height `H`.

### Before: Classic `gcrack` Script

```python
from gcrack import GCrackBase


class GCrackData(GCrackBase):
    ...


if __name__ == "__main__":
    # Define user parameters
    pars = {}
    pars["L"] = 5.0     # Length
    pars["H"] = 1.0     # Height
    pars["Gc"] = 4.5e3  # Critical energy release rate
    # Initialize the simulation
    gcrack_data = GCrackData(
        E=2e9,
        nu=0.3,
        da=pars["L"] / 128,
        Nt=120,
        xc0=[0.0, 0.0, 0.0],
        assumption_2D="plane_stress",
        pars=pars,
        sif_method="williams",
    )
    # Run the simulation
    gcrack_data.run()
```

### After: Parameterized Script with `argparse`

```python
import argparse
from gcrack import GCrackBase


class GCrackData(GCrackBase):
    ...


if __name__ == "__main__":

    # Initialize the parser
    parser = argparse.ArgumentParser(
        prog='ParametricStudy',   
        description='Run a parametrized gcrack simulation'
    )
    # Add each of the arguments
    parser.add_argument(
        '-L',       # Short name of the argument
        '--length', # Name of the argument
        type=float, # Type of the argument (optional)
        default=5.0 # Default value when unspecified (optional)
    )
    parser.add_argument(
        '-H',       # Short name of the argument
        '--height', # Name of the argument
        type=float, # Type of the argument (optional)
        default=1.0 # Default value when unspecified (optional)
    )
    # NOTE: Add other arguments here

    # Parse the arguments
    args = parser.parse_args()
    # NOTE: The input arguments are now available in args.name
    #       Example: args.length gives the value of L
    
    # (Optional) Display the value of the argument L
    print(f"L={args.length}")
    print(f"H={args.height}")

    # Define user parameters
    pars = {}
    pars["L"] = args.length # Length
    pars["H"] = args.height # Height
    pars["Gc"] = 4.5e3      # Critical energy release rate
    # Initialize the simulation
    gcrack_data = GCrackData(
        E=2e9,
        nu=0.3,
        da=pars["L"] / 128,
        Nt=120,
        xc0=[0.0, 0.0, 0.0],
        assumption_2D="plane_stress",
        pars=pars,
        sif_method="williams",
    )
    # Run the simulation
    gcrack_data.run()
```

**Example Usage:**
```shell
python run.py -L 10
```
Here, `L` is set to `10`, while `H` retains its default value of `1`.

!!! note

    On some platforms, setting the environment variable `OMP_NUM_THREADS=1` is required to avoid performance issues.
    Use the command:
    ```shell
    OMP_NUM_THREADS=1 python run.py -L 10
    ```

## Calling `run.py` in parallel

The second step is to invoke `run.py` with varying arguments in parallel.
To this aim, the `multiprocessing` standard module is employed.
For more information on the `multiprocessing` package, check its documentation at [https://docs.python.org/3/library/multiprocessing.html](https://docs.python.org/3/library/multiprocessing.html).

### Script: `run_in_parallel.py`

```python
from multiprocessing import Pool
import subprocess, os

def run(args : dict):
    """Launch a simulation for a given set of parameters
    """

    # Set OMP_NUM_THREADS to 1 to avoid performance issues
    my_env = os.environ.copy()
    my_env["OMP_NUM_THREADS"] = "1"

    # Build and run the command
    command = ["python", "run.py"]
    command += ["-L", str(args["L"])]
    command += ["-H", str(args["H"])]
    # NOTE: This can also be done with a for loop on the dictionary
    subprocess.run(command, env=my_env)

if __name__ == '__main__':

    # Number of parallel processes
    N_proc = 4
    # Sets of parameters for simulations
    args = [
        {"L": 1, "H": 1},
        {"L": 1, "H": 5},
        {"L": 1, "H": 10},
        {"L": 5, "H": 1},
        {"L": 5, "H": 5},
        {"L": 5, "H": 10},
        {"L": 10, "H": 1},
        {"L": 10, "H": 5},
        {"L": 10, "H": 10},
    ]
    #  Launch simulations in parallel
    with Pool(N_proc) as p:
        p.map(run, args)
```
**Example Usage:**
```shell
python run_in_parallel.py
```
This script runs 9 simulations in parallel, with up to `N_proc=4` running simultaneously.




