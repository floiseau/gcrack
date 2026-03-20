import os
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Parameters
B = 0.001

# Find the last modified subdirectory of the current directory
current_dir = Path(".")
subdirectories = [x for x in current_dir.iterdir() if x.is_dir()]
last_modified_subdir = max(subdirectories, key=os.path.getmtime)
res_dir = last_modified_subdir

# Read the CSV file
data = {}
with open(res_dir / "results.csv", newline="") as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)  # Read the header line
    for header in headers:
        data[header] = []
    for row in reader:
        for idx, value in enumerate(row):
            data[headers[idx]].append(float(value))

# Convert lists to numpy arrays
for key, value in data.items():
    data[key] = np.array(value)

# Load factor evolution
plt.figure()
plt.plot(data["lambda"])
plt.xlabel("Load step")
plt.ylabel("Load factor")
plt.grid()

# Force displacement curve
plt.figure()
plt.plot(
    data["uimp_2"],
    data["fimp_2"] * B,
    marker="x",
)
plt.xlabel("Displacement [m]")
plt.ylabel("Force [N]")
plt.grid()

# Show the figures
plt.show()
