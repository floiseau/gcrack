import os
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt

# Parameters
B = 10e-3

# Find the last modified subdirectory of the current directory
current_dir = Path(".")
subdirectories = [x for x in current_dir.iterdir() if x.is_dir()]
last_modified_subdir = max(subdirectories, key=os.path.getmtime)
res_dir = last_modified_subdir

# Read the simulation results (with uy free on the right BC)
sim_csv = last_modified_subdir/"results.csv"
sim_df = pl.read_csv(sim_csv)

# Read the experimental results
exp_cp_csv = Path("reference/crack-path.csv")
exp_fu_csv = Path("reference/force-displacement.csv")
exp_cp_df = pl.read_csv(exp_cp_csv, separator=";")
exp_fu_df = pl.read_csv(exp_fu_csv, separator=";")
print(exp_cp_df)
print(exp_fu_df)


# Load factor evolution
plt.figure()
plt.plot(sim_df["a"], sim_df["lambda"], label="Simulation")
plt.xlabel("Crack length (m)")
plt.ylabel("Load factor")
plt.legend()
plt.grid()

# Force displacement curve
plt.figure()
plt.plot(
    -sim_df["lambda"] * sim_df["uimp_1"],
    -sim_df["lambda"] * sim_df["fimp_1"] * B,
    marker=".",
    label="Simulation",
)
plt.plot(
    exp_fu_df["Displacement [mm]"] * 1e-3,
    exp_fu_df["Force [N]"],
    marker=".",
    label="Experiment",
)

plt.xlabel("Top displacement $-u$ [m]")
plt.ylabel("Top force $-F$ [N]")
plt.legend()
plt.grid()

# Crack path
plt.figure()
plt.plot(
    sim_df["xc_1"] - sim_df["xc_1"][0],
    sim_df["xc_2"] - sim_df["xc_2"][0],
    marker=".", label="Simulation")
plt.plot(
    (exp_cp_df["x [mm]"] - exp_cp_df["x [mm]"][1]) * 1e-3,
    (exp_cp_df["y [mm]"] - exp_cp_df["y [mm]"][1]) * 1e-3,
    marker=".",
    label="Experiment",
)
plt.xlabel("$x$-coordinate (m)")
plt.ylabel("$y$-coordinate (m)")
plt.legend()
plt.grid()

# Show the figures
plt.show()
