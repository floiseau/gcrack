"""
This script visualizes displacement fields from a `gcrack` simulation and saves them as a series of images.

### Description

The script reads displacement data, applies potential transformations (warping), and saves the resulting image at each load step.

### Usage

To visualize the displacement fields, run the following command in the `results_<name>` directory:

```sh
gcrack_displacement_plotter
```

This command generates one image file per load step.

### Arguments

You can pass different arguments to the script. For detailed information on available options, use:

```sh
gcrack_displacement_plotter --help
```

### Creating a Video Animation

The generated images can be used to create a video animation of the simulation. Here's an example using `ffmpeg`:

```sh
ffmpeg -framerate 25 -pattern_type glob -i 'Displacement*.png' Displacement.mp4
```
"""

import argparse

from pathlib import Path
import pyvista as pv


def export_displacement_figures(factor: float, extension: str):
    """Export the displacement fields into image files.

    Attributes:
        factor (float): Amplitude of the displacement warping.
        extension (str): Extension of the image file (e.g., `"jpg"`, `"png"`).
    """

    # Find the pvtu files
    pvtu_files = list(sorted(Path(".").glob("Displacement*.pvtu")))

    # Create a plotter
    pl = pv.Plotter(off_screen=True)

    # Read each pvtu file
    for t, pvtu_file in enumerate(pvtu_files, start=1):
        # Read the file
        data = pv.read(pvtu_file)
        # Warp by displacement
        data = data.warp_by_vector("Displacement", factor=factor)
        # Display the results
        pl.add_mesh(data, scalars="Displacement")
        pl.view_xy()
        # Keep a constant camera position
        if t == 1:
            cpos = pl.camera_position

        pl.camera_position = cpos
        # Add frame to the gif
        pl.show(screenshot=f"Displacement_{t:04d}.{extension}", auto_close=False)
        # Clear the plot
        pl.clear()
    # Close the plotter
    pl.close()


def main():
    """Entry point of `gcrack_displacement_plotter`."""
    # Create the parser
    parser = argparse.ArgumentParser(
        prog="gcrack_displacement_plotter",
        description="Generate the displacement field images from a gcrack simulation",
    )
    parser.add_argument(
        "-e",
        "--extension",
        help="extension of the figures (see matplotlib.pypyplot.savefig)",
        default="png",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--factor",
        help="scaling factor (0.0 for no warping)",
        default=0.0,
        type=float,
    )

    # Parse and extract the arguments
    args = parser.parse_args()
    factor = args.factor
    extension = args.extension

    # Generate the figures
    export_displacement_figures(factor, extension)


if __name__ == "__main__":
    main()
