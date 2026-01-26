from pathlib import Path
import csv
import matplotlib.pyplot as plt

# Set parameters
angles = [15, 30, 45, 60, 75]
colors = {15: "#00002c", 30: "#003162", 45: "#007e9f", 60: "#ff950a", 75: "#f1443a"}


# Function to read CSV as a dictionary of lists
def read_csv_as_dict(file_path):
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        data = {col: [] for col in reader.fieldnames}
        for row in reader:
            for col in reader.fieldnames:
                data[col].append(float(row[col]))
        return data


# Read and store the simulation results
sim_res = {}
for angle in angles:
    sim_csv = Path(f"results_alpha_{angle}/results.csv")
    sim_res[angle] = read_csv_as_dict(sim_csv)

# Read the experimental results
exp_res = {}
for angle in angles:
    exp_csv = Path(f"reference/hao_et_al_2023_alpha_{angle}.csv")
    exp_res[angle] = read_csv_as_dict(exp_csv)

# Crack path
plt.figure()
for angle in angles:
    # Get the data
    sim_df = sim_res[angle]
    exp_df = exp_res[angle]
    # Display the simulation data
    plt.plot(
        [x - sim_df["xc_1"][0] for x in sim_df["xc_1"]],
        [y - sim_df["xc_2"][0] for y in sim_df["xc_2"]],
        color=colors[angle],
        marker=None,
        label=f"Sim ({angle}°)",
    )
    # Display the experimental data
    plt.plot(
        [x - exp_df["x"][0] for x in exp_df["x"]],
        [y - exp_df["y"][0] for y in exp_df["y"]],
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
