from typing import List

from domain import Domain
from models import ElasticModel

import ufl
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem


def solve_elastic_problem(
    domain: Domain,
    model: ElasticModel,
    V_u: fem.FunctionSpace,
    bcs: List[fem.DirichletBC],
) -> fem.Function:
    # Define the variational formulation
    ds = ufl.Measure("ds", domain=domain.mesh)
    T = fem.Constant(domain.mesh, default_scalar_type((0, 0)))
    u = ufl.TrialFunction(V_u)
    v = ufl.TestFunction(V_u)
    f = fem.Constant(domain.mesh, default_scalar_type((0, 0)))
    a = ufl.inner(model.sig(u), model.eps(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

    #  Define and solve the problem
    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={
            "ksp_type": "cg",
            "ksp_rtol": 1e-12,
            "ksp_atol": 1e-12,
            "ksp_max_it": 1000,
            "pc_type": "gamg",
            "pc_gamg_agg_nsmooths": 1,
            "pc_gamg_esteig_ksp_type": "cg",
        },
    )
    return problem.solve()
