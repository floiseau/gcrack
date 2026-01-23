from pathlib import Path

import polars as pl
import numpy as np
import matplotlib.pyplot as plt


# Read the simulation results (with uy free on the right BC)
sim_csv = Path("alpha_60") / "results.csv"
sim_df = pl.read_csv(sim_csv)

# Read the experimental results
exp_df_list = []
for i in range(4):
    exp_csv = Path(f"reference/hao_et_al_2023_alpha_60_{i+1}.csv")
    exp_df = pl.read_csv(exp_csv)
    exp_df_list.append(exp_df)

# Crack path
plt.figure()
plt.plot(
    sim_df["xc_1"] - sim_df["xc_1"][0],
    sim_df["xc_2"] - sim_df["xc_2"][0],
    marker="+",
    color="r",
    label="Simulation",
)
for i, exp_df in enumerate(exp_df_list):
    xc = np.array(
        [
            (exp_df["x"] - exp_df["x"][0]).to_numpy(),
            (exp_df["y"] - exp_df["y"][0]).to_numpy(),
        ]
    )
    alpha = np.deg2rad(60)
    R = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
    xcr = np.matmul(R.T, xc)
    plt.plot(
        xcr[0],
        xcr[1],
        marker="x",
        color="k",
        linewidth=0.5,
        label=f"Experiment {i}",
    )
plt.xlabel("$x$-coordinate (m)")
plt.ylabel("$y$-coordinate (m)")
plt.legend()
plt.grid()
plt.axis("equal")

# Show the figures
plt.show()
