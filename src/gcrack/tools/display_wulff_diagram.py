import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# NOTE:
# convert -delay 5 -loop 0 wulff_diagram_00000???.svg wulff_diagram.gif

# Parameters
fontsize = 18


def export_wulff_diagram(csv_file: Path, extension: str):

    # Read the csv file using numpy
    data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)

    # Extract the data
    phi = data[:, 0]
    Gs = data[:, 1]
    Gc = data[:, 2]

    # Find the intersection point
    idx = np.argmin(abs(Gs - Gc))

    # Create the plot
    fig, ax = plt.subplots(
        1, 1, figsize=(5, 8), subplot_kw={"projection": "polar"}, layout="tight"
    )

    # Add the intersection point
    ax.scatter([phi[idx]], 1 / Gc[idx], s=1e2, marker="o", color="k")
    # Add the intersection line
    ax.plot([phi[idx], phi[idx]], [0, 1.5 / Gc[idx]], color="gray", ls="dashed")
    ax.annotate(
        f"$\\varphi = {np.rad2deg(phi[idx]):.2g}^\\circ$",
        xy=(phi[idx], 1.45 / Gc[idx]),
        # xy=(np.pi, 0.6 / Gc[idx]),
        fontsize=fontsize,
        annotation_clip=False,
        backgroundcolor="white",
        rotation=np.rad2deg(phi[idx]),
    )

    # Add 1/Gc and 1/Gs
    ax.plot(phi, 1 / Gs, "b", label=r"${G^*}^{-1}(\varphi)$")
    ax.plot(phi, 1 / Gc, "r", label=r"$G_c^{-1}(\varphi)$")

    # Set the r-axis options
    ax.set_rmax(1.4 * np.max(1 / Gc))
    ax.set_rticks(
        [0.5 / Gc[idx], 1 / Gc[idx]],
        labels=[r"$G_{c,\mathrm{min}}^{-1} /2$", r"$G_{c,\mathrm{min}}^{-1}$"],
    )
    # Set the theta limits
    ax.set_thetamin(-90)
    ax.set_thetamax(90)

    # Set the font size
    ax.tick_params(labelsize=fontsize)
    # Add the grid
    ax.grid(True)
    ax.legend(loc=(0.25, 1.1), fontsize=fontsize)

    # Save the figure
    img_file = csv_file.with_suffix(f".{extension}")
    plt.savefig(img_file, dpi=150)

    # Close the figure
    plt.close()


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
        default="svg",
        type=str,
    )
    args = parser.parse_args()

    # Get all the wulff csv file in the current directory
    csv_files = list(sorted(Path(".").glob("wulff*.csv")))
    # Generate the Wulff diagram for each csv file
    for csv_file in csv_files:
        export_wulff_diagram(csv_file, args.extension)


if __name__ == "__main__":
    main()
