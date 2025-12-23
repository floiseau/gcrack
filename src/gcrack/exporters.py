"""
Module for exporting simulation results in CSV and VTK formats.

This module provides utility functions for exporting FEniCSx function data to VTK files for visualization, appending simulation results to CSV files, and cleaning up VTK output directories by gathering .pvtu files into a single .pvd file.

Functions:
    export_res_to_csv: Appends a dictionary of results to a CSV file. The keys of the dictionary become the column headers, and the values are appended as a row.
    export_function: Exports a FEniCS function to a VTK file. The filename is constructed using the function name and the provided time step `t`.
    clean_vtk_files: Cleans a directory by removing existing .pvd files and creating a new .pvd file that lists all .pvtu files with their corresponding timesteps.
"""

from pathlib import Path
import csv

from dolfinx import io, fem


def export_res_to_csv(res: dict, filename: str):
    """
    Append the res dictionary to a CSV file.

    This function appends the contents of a dictionary to a CSV file. The keys of the dictionary
    become the column headers in the CSV file, and the values are appended to the associated column.

    Args:
        res (dict): The dictionary containing row data to be appended.
                     The keys are column headers and the values are the row values.
        filename (str): The name of the CSV file to be created.
    """
    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(res.keys()))
        if res["t"] == 0:
            writer.writeheader()
        writer.writerow(res)


def export_function(u: fem.Function, t: int, dir_path: Path):
    """
    Export a FEniCS function to a VTK file.

    This function writes the given FEniCS function to a VTK file for visualization. The filename
    is constructed using the function name with the provided time step `t`, and the file is saved
    in the specified directory.

    Args:
        u (fem.Function): The FEniCS function to be exported.
        t (int): The time step used to construct the filename.
        dir_path (Path): The path to the directory where the VTK file will be saved.
    """
    # Get function info
    V = u.function_space
    vtkfile = io.VTKFile(V.mesh.comm, dir_path / f"{u.name}_{t:04d}_.pvd", "w")
    vtkfile.write_function(u, 0)
    vtkfile.close()


def clean_vtk_files(res_dir: Path):
    """
    Clean the specified directory by removing existing .pvd files and create a new .pvd file listing all .pvtu files.

    This function removes all existing .pvd files in the given directory and creates a new .pvd file that lists all .pvtu files
    with their corresponding timesteps. The new .pvd file is named 'displacement.pvd'.

    Args:
        res_dir (Path): The path to the directory containing .pvtu and .vtu files.
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

    print(f"Created displacement.pvd with {len(pvtu_files)} timesteps.")
