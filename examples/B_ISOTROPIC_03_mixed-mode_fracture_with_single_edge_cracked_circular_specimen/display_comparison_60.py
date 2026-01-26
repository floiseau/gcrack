from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


# Function to read CSV as a dictionary of lists
def read_csv_as_dict(file_path):
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        data = {col: [] for col in reader.fieldnames}
        for row in reader:
            for col in reader.fieldnames:
                data[col].append(float(row[col]))
        return data


# Read the simulation results
sim_csv = Path("results_alpha_60") / "results.csv"
sim_df = read_csv_as_dict(sim_csv)

# Read the experimental results
exp_df_list = []
for i in range(4):
    exp_csv = Path(f"reference/hao_et_al_2023_alpha_60_{i + 1}.csv")
    exp_df = read_csv_as_dict(exp_csv)
    exp_df_list.append(exp_df)

# Crack path
plt.figure()
plt.plot(
    [x - sim_df["xc_1"][0] for x in sim_df["xc_1"]],
    [y - sim_df["xc_2"][0] for y in sim_df["xc_2"]],
    marker="+",
    color="r",
    label="Simulation",
)

for i, exp_df in enumerate(exp_df_list):
    x = np.array(exp_df["x"])
    y = np.array(exp_df["y"])
    xc = np.array([x - x[0], y - y[0]])
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
