# Exporting the mesh into a file

In some cases, you may want to export the GMSH geometry into a file (e.g., export a figure to visualize the geometry in a parametric study).
This can be achieved by using the following code in the `generate_mesh` function:

```python
# Export the geometry
gmsh.model.geo.synchronize()
gmsh.fltk.initialize()
export_dir = "results_" + self.name
gmsh.write(f"{export_dir}/geometry_{crack_points[-1][0]}_{crack_points[-1][1]}.svg")
gmsh.fltk.finalize()
```

The file extension in the `gmsh.write()` call can be modified to export using different field formats.

!!! tip

    Add this code **before** calling `gmsh.model.mesh.generate` when exporting an image. 
    Otherwise, the full mesh will be exported, which can result in longer export duration and much larger file size.
