from pathlib import Path
import csv

from dolfinx import io, fem


def export_dict_to_csv(data, filename):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header containing keys of the dictionary
        writer.writerow(data.keys())
        # Determine the length of the longest list in the dictionary
        max_length = max(len(lst) for lst in data.values())
        # Iterate through the lists and write rows to the CSV file
        for i in range(max_length):
            row = [data[key][i] if i < len(data[key]) else None for key in data.keys()]
            writer.writerow(row)


def export_function(u: fem.Function, t: int, dir_path: Path):
    # Get function info
    V = u.function_space
    vtkfile = io.VTKFile(V.mesh.comm, dir_path / f"u_{t:04d}.pvd", "w")
    vtkfile.write_function(u, 0)
    vtkfile.close()
