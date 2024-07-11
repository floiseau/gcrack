import logging
from pathlib import Path
import csv

from dolfinx import io, fem


def export_dict_to_csv(data, filename):
    """
    Export a dictionary to a CSV file.

    This function writes the contents of a dictionary to a CSV file. The keys of the dictionary
    become the column headers in the CSV file, and the values, which are expected to be lists,
    become the rows. The function handles lists of different lengths by filling missing values with None.

    Args:
        data (dict): The dictionary containing data to be exported. The keys are column headers,
                     and the values are lists representing the rows.
        filename (str): The name of the CSV file to be created.

    Returns:
        None
    """
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
    logging.info("Created results.csv.")


def export_function(u: fem.Function, t: int, dir_path: Path):
    """
    Export a FEniCS function to a VTK file.

    This function writes the given FEniCS function to a VTK file for visualization. The filename
    is constructed using the provided time step `t`, and the file is saved in the specified directory.

    Args:
        u (fem.Function): The FEniCS function to be exported.
        t (int): The time step used to construct the filename.
        dir_path (Path): The path to the directory where the VTK file will be saved.

    Returns:
        None
    """
    # Get function info
    V = u.function_space
    vtkfile = io.VTKFile(V.mesh.comm, dir_path / f"u_{t:04d}_.pvd", "w")
    vtkfile.write_function(u, 0)
    vtkfile.close()


def clean_vtk_files(res_dir: Path):
    """
    Clean the specified directory by removing existing .pvd files and create a new .pvd file listing all .pvtu files.

    This function removes all existing .pvd files in the given directory and creates a new .pvd file that lists all .pvtu files
    with their corresponding timesteps. The new .pvd file is named 'displacement.pvd'.

    Args:
        res_dir (Path): The path to the directory containing .pvtu and .vtu files.

    Returns:
        None
    """

    # Remove existing .pvd files
    for pvd_file in res_dir.glob("*.pvd"):
        pvd_file.unlink()

    # Collect all .pvtu files and sort them
    pvtu_files = sorted(res_dir.glob("*.pvtu"))

    # Create a new .pvd file content
    pvd_content = (
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
    )
    pvd_content += "  <Collection>\n"

    for timestep, pvtu_file in enumerate(pvtu_files):
        pvd_content += f'    <DataSet timestep="{timestep}" group="" part="0" file="{pvtu_file.name}"/>\n'

    pvd_content += "  </Collection>\n"
    pvd_content += "</VTKFile>"

    # Write the new .pvd file
    combined_pvd_path = res_dir / "displacement.pvd"
    with combined_pvd_path.open("w") as file:
        file.write(pvd_content)

    logging.info(f"Created displacement.pvd with {len(pvtu_files)} timesteps.")
