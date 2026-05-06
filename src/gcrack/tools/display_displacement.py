import argparse

from pathlib import Path
import pyvista as pv


def export_displacement_figures(args):
    # Find the pvtu files
    pvtu_files = list(sorted(Path(".").glob("Displacement*.pvtu")))

    # Create a plotter
    pl = pv.Plotter(off_screen=True)

    # Read each pvtu file
    for t, pvtu_file in enumerate(pvtu_files, start=1):
        # Read the file
        data = pv.read(pvtu_file)
        # Warp by displacement
        data = data.warp_by_vector("Displacement", factor=args.factor)
        # Display the results
        pl.add_mesh(data, scalars="Displacement")
        pl.view_xy()
        # Keep a constant camera position
        if t == 1:
            cpos = pl.camera_position

        pl.camera_position = cpos
        # Add frame to the gif
        pl.show(screenshot=f"Displacement_{t:04d}.{args.extension}", auto_close=False)
        # Clear the plot
        pl.clear()
    # Close the plotter
    pl.close()


def main():
    # Create the parser
    parser = argparse.ArgumentParser(
        prog="gcrack_wulff_plotter",
        description="Generate the Wulff plot from a gcrack simulation",
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
    args = parser.parse_args()

    # Generate the figures
    export_displacement_figures(args)

    print("Generate an mp4 video from the png files using :")
    print(
        "ffmpeg -framerate 25 -pattern_type glob -i 'Displacement*.png' Displacement.mp4"
    )


if __name__ == "__main__":
    main()
