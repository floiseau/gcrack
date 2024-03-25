from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Parameters
res_dir = Path("results_2024-03-25_15-49-58")
B = 0.001

# Read the CSV file
df = pd.read_csv(res_dir / "results.csv")

# Load factor evolution
plt.figure()
plt.plot(df["lambda"])
plt.xlabel("Load step")
plt.ylabel("Load factor")
plt.grid()

# Force displacement curve
plt.figure()
plt.plot(
    df["lambda"]*df["uimp_2"],
    df["lambda"]*df["fimp_2"]*B,
    marker="x",
)
plt.xlabel("Load step")
plt.ylabel("Load factor")
plt.grid()

# Show the figures
plt.show()
