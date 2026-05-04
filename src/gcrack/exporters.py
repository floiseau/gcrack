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
import jax.numpy as jnp

from gcrack.lefm import G_star, G_star_coupled


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


def export_heterogeneous_parameters(model, ela_pars: dict, dir_path: Path):
    """
    Export the heterogeneous parameters into a VTK file.

    Args:
        model (gcrack.ElasticModel): ElasticModel in gcrack.
        ela_pars (dict): Input (raw) parameters of the elastic model.
        dir_path (Path): The path to the directory where the VTK file will be saved.
    """
    # At first load step, also export the heterogeneous parameters
    if isinstance(ela_pars["E"], str):
        V = model.E.function_space
        model.E.name = "Young Modulus"
        vtkfile = io.VTKFile(V.mesh.comm, dir_path / "YoungModulus.pvd", "w")
        vtkfile.write_function(model.E, 0)
        vtkfile.close()
    if isinstance(ela_pars["nu"], str):
        V = model.nu.function_space
        model.nu.name = "Poisson Ratio"
        vtkfile = io.VTKFile(V.mesh.comm, dir_path / "PoissonRatio.pvd", "w")
        vtkfile.write_function(model.nu, 0)
        vtkfile.close()


def clean_vtk_files(
    res_dir: Path, export_strain: bool = False, export_stress: bool = False
):
    """
    Clean the specified directory by removing existing .pvd files and create a new .pvd file listing all .pvtu files.

    This function removes all existing .pvd files in the given directory and creates a new .pvd file that lists all .pvtu files
    with their corresponding timesteps. The new .pvd file is named 'displacement.pvd'.

    Args:
        res_dir (Path): The path to the directory containing .pvtu and .vtu files.
        export_strain (Optional[bool]): Flag to enable strain export in VTK files.
        export_stress (Optional[bool]): Flag to enable stress export in VTK files.
    """
    # Set the exported field
    exported_fields = ["Displacement"]
    if export_strain:
        exported_fields += ["Strain"]
    if export_stress:
        exported_fields += ["Stress"]

    # Clean of the field types
    for field in exported_fields:
        # Remove existing .pvd files
        for pvd_file in res_dir.glob(f"{field}_*.pvd"):
            pvd_file.unlink()

        # Collect all .pvtu files and sort them
        pvtu_files = sorted(res_dir.glob(f"{field}_*.pvtu"))

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
        combined_pvd_path = res_dir / f"{field}.pvd"
        with combined_pvd_path.open("w") as file:
            file.write(pvd_content)

        print(f"Created {field}.pvd with {len(pvtu_files)} timesteps.")


def export_G_star_vs_phi(
    phi: float,
    load_factor: float,
    phi0: float,
    SIFs_controlled: dict,
    SIFs_prescribed: dict,
    s: float,
    t: int,
    dir_name: Path,
    Gc: callable,
    Ep: float,
):

    KIc, KIIc, Tc = (
        SIFs_controlled["KI"],
        SIFs_controlled["KII"],
        SIFs_controlled["T"],
    )
    KIp, KIIp, Tp = (
        SIFs_prescribed["KI"],
        SIFs_prescribed["KII"],
        SIFs_prescribed["T"],
    )

    phi_vals = jnp.linspace(phi0 - jnp.pi, phi0 + jnp.pi, 361)
    lam = load_factor

    G_total_vals, Gs_cc_vals, Gs_cp_vals, Gs_pp_vals, gc_vals = [], [], [], [], []

    for phi in phi_vals:
        Gs_cc = float(G_star(phi, phi0, KIc, KIIc, Tc, Ep, s))
        Gs_cp = float(G_star_coupled(phi, phi0, KIc, KIIc, Tc, KIp, KIIp, Tp, Ep, s))
        Gs_pp = float(G_star(phi, phi0, KIp, KIIp, Tp, Ep, s))
        gc = float(Gc(jnp.array([phi]))[0])

        G_total_vals.append(Gs_pp + 2 * lam * Gs_cp + lam**2 * Gs_cc)
        Gs_cc_vals.append(Gs_cc)
        Gs_cp_vals.append(Gs_cp)
        Gs_pp_vals.append(Gs_pp)
        gc_vals.append(gc)

    out_path = dir_name / f"wulff_diagram_{t:08d}.csv"
    fieldnames = ["phi", "G_star", "G_c", "G_star_cc", "G_star_cp", "G_star_pp"]

    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, phi in enumerate(phi_vals):
            writer.writerow(
                {
                    "phi": phi,
                    "G_star": G_total_vals[i],
                    "G_c": gc_vals[i],
                    "G_star_cc": Gs_cc_vals[i],
                    "G_star_cp": Gs_cp_vals[i],
                    "G_star_pp": Gs_pp_vals[i],
                }
            )
