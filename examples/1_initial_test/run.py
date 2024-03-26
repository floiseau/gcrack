import sys

sys.path.append("/home/flavien.loiseau/ownCloud/codes/gcrack/src/gcrack")

from typing import List, Tuple

import numpy as np

import gmsh
from dolfinx import fem

from gcrack import gcrack

### Constants
# Geometry
L: float = 1e-3
# Numerical
R_int: float = L / 64
h_min: float = R_int / 32
h: float = L / 64
da: float = 1e-5


def generate_mesh(crack_points: List[Tuple[float, float, float]]) -> gmsh.model:
    # Clear existing model
    gmsh.clear()
    # Points
    # Bot
    p1: int = gmsh.model.geo.addPoint(0, 0, 0, h)
    p2: int = gmsh.model.geo.addPoint(L, 0, 0, h)
    p3: int = gmsh.model.geo.addPoint(L, L / 2, 0, h)  # Mid right node
    pc_bot: List[int] = []
    pc_top: List[int] = []
    for i, p in enumerate(reversed(crack_points)):
        # The crack tip is shared
        if i == 0:
            pc_new: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
            pc_bot.append(pc_new)
            pc_top.append(pc_new)
        else:
            pc_new_bot: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
            pc_bot.append(pc_new_bot)
            pc_new_top: int = gmsh.model.geo.addPoint(p[0], p[1], p[2], h)
            pc_top.append(pc_new_top)
    p5: int = gmsh.model.geo.addPoint(0, L / 2, 0, h)  # Bot crack lip
    # Top
    p6: int = gmsh.model.geo.addPoint(0, L, 0, h)
    p7: int = gmsh.model.geo.addPoint(L, L, 0, h)
    # Point(13) // Mid right node
    # Point(14) // Crack tip
    p8: int = gmsh.model.geo.addPoint(0, L / 2, 0, h)  # Top crack lip

    # Lines
    # Bot
    l1: int = gmsh.model.geo.addLine(p1, p2)
    l2: int = gmsh.model.geo.addLine(p2, p3)
    l3: int = gmsh.model.geo.addLine(p3, pc_bot[0])
    crack_lines_bot: List[int] = []
    for i in range(len(pc_bot) - 1):
        lb: int = gmsh.model.geo.addLine(pc_bot[i], pc_bot[i + 1])
        crack_lines_bot.append(lb)
    crack_lines_bot.append(gmsh.model.geo.addLine(pc_bot[-1], p5))
    l5: int = gmsh.model.geo.addLine(p5, p1)
    # Top
    l6: int = gmsh.model.geo.addLine(p6, p7)
    l7: int = gmsh.model.geo.addLine(p7, p3)
    # Line(13)
    # Top  crack line
    crack_lines_top: List[int] = []
    for i in range(len(pc_bot) - 1):
        lt: int = gmsh.model.geo.addLine(pc_top[i], pc_top[i + 1])
        crack_lines_top.append(lt)
    crack_lines_top.append(gmsh.model.geo.addLine(pc_top[-1], p8))
    l9: int = gmsh.model.geo.addLine(p8, p6)

    # Surfaces
    # Bot
    cl1: int = gmsh.model.geo.addCurveLoop([l1, l2, l3] + crack_lines_bot + [l5])
    s1: int = gmsh.model.geo.addPlaneSurface([cl1])
    # Top
    cl2: int = gmsh.model.geo.addCurveLoop([l6, l7, l3] + crack_lines_top + [l9])
    s2: int = gmsh.model.geo.addPlaneSurface([cl2])

    # Physical groups
    # Domain
    domain: int = gmsh.model.addPhysicalGroup(2, [s1, s2], tag=21)
    gmsh.model.setPhysicalName(2, domain, "domain")
    # Boundaries
    bot: int = gmsh.model.addPhysicalGroup(1, [l1], tag=11)
    gmsh.model.setPhysicalName(1, bot, "bot")
    top: int = gmsh.model.addPhysicalGroup(1, [l6], tag=12)
    gmsh.model.setPhysicalName(1, top, "top")

    # Element size
    # Refine around the crack line
    field1: int = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(field1, "PointsList", [pc_bot[0]])
    gmsh.model.mesh.field.setNumber(field1, "Sampling", 100)
    field2: int = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(field2, "InField", field1)
    gmsh.model.mesh.field.setNumber(field2, "DistMin", 2 * R_int)
    gmsh.model.mesh.field.setNumber(field2, "DistMax", 4 * R_int)
    gmsh.model.mesh.field.setNumber(field2, "SizeMin", h_min)
    gmsh.model.mesh.field.setNumber(field2, "SizeMax", h)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.field.setAsBackgroundMesh(field2)
    gmsh.model.mesh.generate(2)

    # Return the model
    return gmsh.model()


def show_mesh() -> None:
    gmsh.fltk.run()


def define_dirichlet_bcs(V_u: fem.FunctionSpace) -> List[fem.DirichletBC]:
    def on_bot_boundary(x):
        return np.isclose(x[1], 0)

    comp = 1
    bot_dofs = fem.locate_dofs_geometrical(
        (V_u.sub(comp), V_u.sub(comp).collapse()[0]), on_bot_boundary
    )
    u0_func = fem.Function(V_u.sub(comp).collapse()[0])
    with u0_func.vector.localForm() as bc_local:
        bc_local.set(0.0)
    bot_bc = fem.dirichletbc(u0_func, bot_dofs, V_u)

    def on_locked_point(x):
        return np.logical_and(np.isclose(x[0], 0), np.isclose(x[1], 0))

    comp = 0
    locked_dofs = fem.locate_dofs_geometrical(V_u, on_locked_point)
    locked_bc = fem.dirichletbc(np.array([0.0, 0.0]), locked_dofs, V_u)

    def on_uimp_boundary(x):
        return np.isclose(x[1], L)

    comp = 1
    uimp_dofs = fem.locate_dofs_geometrical(
        (V_u.sub(comp), V_u.sub(comp).collapse()[0]), on_uimp_boundary
    )
    uimp_func = fem.Function(V_u.sub(comp).collapse()[0])
    with uimp_func.vector.localForm() as bc_local:
        bc_local.set(1.0)
    uimp_bc = fem.dirichletbc(uimp_func, uimp_dofs, V_u)

    return [bot_bc, locked_bc, uimp_bc]


if __name__ == "__main__":
    xc = [L / 2, L / 2, 0]
    gcrack(generate_mesh, define_dirichlet_bcs, R_int, xc, da)
