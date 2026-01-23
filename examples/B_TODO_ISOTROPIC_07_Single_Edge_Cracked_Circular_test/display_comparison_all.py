from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt

# Set parameters
angles = [15, 30, 45, 60, 75]
colors = {15: "#00002c", 30: "#003162", 45: "#007e9f", 60: "#ff950a", 75: "#f1443a"}

# Read and store the simulation results
sim_res = {}
for angle in angles:
    sim_csv = Path(f"alpha_{angle}") / "results.csv"
    sim_res[angle] = pl.read_csv(sim_csv)

# Read the experimental results
exp_res = {}
for angle in angles:
    exp_csv = Path(f"reference/hao_et_al_2023_alpha_{angle}.csv")
    exp_res[angle] = pl.read_csv(exp_csv)

# Crack path
plt.figure()
for angle in angles:
    # Get the dataframes
    sim_df = sim_res[angle]
    exp_df = exp_res[angle]
    # Display the simulation data
    plt.plot(
        sim_df["xc_1"] - sim_df["xc_1"][0],
        sim_df["xc_2"] - sim_df["xc_2"][0],
        color=colors[angle],
        marker=None,
        label=f"Sim ({angle}°)",
    )
    # Display the experimental data
    plt.plot(
        exp_df["x"] - exp_df["x"][0],
        exp_df["y"] - exp_df["y"][0],
        ":",
        color=colors[angle],
        marker=None,
        label=f"Exp ({angle}°)",
    )
plt.xlabel("$x$-coordinate (m)")
plt.ylabel("$y$-coordinate (m)")
plt.legend()
plt.grid()
plt.axis("equal")

# Show the figures
plt.show()
