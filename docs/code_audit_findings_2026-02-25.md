# gcrack Code Audit Findings

Date: 2026-02-25  
Scope: `src/gcrack` <>, packaging/CI context reviewed for cross-checks  

## Executive Summary

This audit identified:

- **2 high-confidence runtime bug risks**
- **4 medium-severity robustness/scientific caveats**
- **several maintainability and packaging quality improvements**

No syntax/compile diagnostics were raised by workspace tooling during this review, but several logic and API risks remain.

---

## Severity Key

- **Critical**: likely crash or invalid core output in normal usage
- **High**: strong bug risk or major correctness concern
- **Medium**: robustness/scientific caveat likely to surface in edge/common workflows
- **Low**: maintainability, clarity, or non-blocking quality issue

---

## Findings

## 1) Missing constructor argument in `ICrackBase` load-factor solver call

- **Severity**: High
- **Location**: `src/gcrack/icrack.py` (around `load_factor_solver = LoadFactorSolver(model, self.Gc)`)
- **Evidence**:
  - `LoadFactorSolver.__init__` expects `(model, Gc_func, xc)` in `src/gcrack/optimization_solvers.py`
  - `ICrackBase.run()` currently passes only `(model, self.Gc)`
- **Risk/Impact**:
  - Strong runtime failure risk (`TypeError`) when this code path is executed.
- **Recommendation**:
  - Pass crack-tip position consistently (same approach as `GCrackBase` / `FCrackBase`).

### Fix (explicit)

1. In `src/gcrack/icrack.py`, update solver creation to:
  - `load_factor_solver = LoadFactorSolver(model, self.Gc, crack_points[-1])`
2. Run one `ICrackBase`-derived example to confirm no constructor error.
3. Add/extend a test that executes this run path to prevent regressions.

---

## 2) `FCrackBase.__post_init__` may fail when `da` is intentionally unset

- **Severity**: High
- **Location**: `src/gcrack/fcrack.py` (`__post_init__`)
- **Evidence**:
  - `da` is optional in dataclass (`da: Optional[float] = None`)
  - `__post_init__` computes radii with `self.da` immediately
  - control mode (`da` vs `dN`) is validated later in `run()`
- **Risk/Impact**:
  - If user selects `dN` mode (`da=None`), radii computation can fail before useful validation.
- **Recommendation**:
  - Validate control mode before radius computation, or lazily derive radii once `da` is known.

### Fix (explicit)

1. In `src/gcrack/fcrack.py`, guard `__post_init__`:
  - if `self.da is None`, defer `R_int/R_ext` initialization.
2. In `run()`, after selecting control mode:
  - if `control_type == "dN"`, either require a reference `da` for SIF radii or compute radii from a dedicated parameter.
3. Raise a clear `ValueError` if radii cannot be defined.
4. Add a test for `dN` mode with `da=None` to ensure predictable behavior.

---

## 3) Duplicate CSV export behavior in `ICrackBase`

- **Severity**: Medium
- **Location**: `src/gcrack/icrack.py` (`for sif_name in SIFs` block)
- **Evidence**:
  - `export_res_to_csv(res, ...)` is inside loop over SIF keys.
- **Risk/Impact**:
  - Multiple rows can be written for one simulation step, causing duplicated/ambiguous logs.
- **Recommendation**:
  - Move CSV export outside SIF loop so one row is written per step.

### Fix (explicit)

1. In `src/gcrack/icrack.py`, keep only SIF assignments inside:
  - `for sif_name in SIFs: res[sif_name] = SIFs[sif_name]`
2. Move `export_res_to_csv(res, dir_name / "results.csv")` to immediately after the loop.
3. Validate CSV row count equals number of simulated steps (plus header/initial state row).

---

## 4) Unresolved scientific TODO in SIF scaling (T-stress)

- **Severity**: Medium (scientific correctness)
- **Location**: `src/gcrack/sif.py` (Williams interpolation branch)
- **Evidence**:
  - Explicit TODO comment indicates scaling may be wrong.
- **Risk/Impact**:
  - T-stress estimates may be biased, affecting optimization and crack path/load outcomes.
- **Recommendation**:
  - Add verification case(s) with analytical/benchmark references and confirm scaling constant.

### Fix (explicit)

1. In `src/gcrack/sif.py`, isolate current T-stress scaling in a named constant/function.
2. Build 1–2 benchmark cases with known `T` (or validated reference data).
3. Compare recovered `T` vs reference for mesh/radius sensitivity.
4. Update scaling factor and remove TODO once validated.
5. Add regression test around accepted tolerance.

---

## 5) Placeholder string value in fatigue post-processing

- **Severity**: Medium
- **Location**: `src/gcrack/fcrack.py` (`res["fracture_dissipation"] = "TODO"`)
- **Risk/Impact**:
  - Breaks numeric expectations in downstream CSV analysis/plotting.
- **Recommendation**:
  - Replace placeholder with numeric computation or explicitly nullable numeric field.

### Fix (explicit)

1. In `src/gcrack/fcrack.py`, replace:
  - `res["fracture_dissipation"] = "TODO"`
  with a numeric value (`float`) every step.
2. If physics model is pending, use `np.nan` (documented) rather than a string.
3. Ensure CSV consumers can parse the column as numeric.

---

## 6) Ambiguous DOF emptiness check in nodal displacement handling

- **Severity**: Medium
- **Location**: `src/gcrack/boundary_conditions.py` (`if not dof:`)
- **Risk/Impact**:
  - Depending on returned type, truth-value evaluation may be ambiguous or incorrect.
- **Recommendation**:
  - Use explicit cardinality checks (`len(...)`, `.size`, or API-specific equivalent).

### Fix (explicit)

1. In `src/gcrack/boundary_conditions.py`, replace `if not dof:` with explicit check:
  - e.g., `if len(dof) == 0:` (or API-appropriate `.size == 0`).
2. Keep current error message but include boundary/node context for debugging.
3. Add a small test case where nodal target is missing.

---

## 7) Probe evaluation without collision guard

- **Severity**: Medium
- **Location**: `src/gcrack/postprocess.py` (`cell = colliding_cells.array[0]`)
- **Risk/Impact**:
  - Fails when probe point is outside mesh or no containing cell is found.
- **Recommendation**:
  - Guard empty results and raise actionable error message.

### Fix (explicit)

1. In `src/gcrack/postprocess.py`, before `colliding_cells.array[0]`, check:
  - whether `colliding_cells.array` is non-empty.
2. If empty, raise `ValueError` with point coordinates and a hint (point outside mesh / tolerance issue).
3. Optionally add fallback nearest-cell probing only if physically acceptable.

---

## 8) Side-effect mutation of user-provided measurement point list

- **Severity**: Low
- **Location**: `src/gcrack/postprocess.py` (`x.append(0)`)
- **Risk/Impact**:
  - Mutates data returned by user callback, potentially causing subtle repeated-call behavior.
- **Recommendation**:
  - Work on a copy instead of mutating `x` in place.

### Fix (explicit)

1. In `src/gcrack/postprocess.py`, avoid `x.append(0)`.
2. Use non-mutating construction:
  - `x_local = list(x)` then append to `x_local`, or create a new array directly.
3. Keep callback output immutable from caller perspective.

---

## 9) Heavy imports at package initialization

- **Severity**: Low/Medium (ergonomics & failure surface)
- **Location**: `src/gcrack/__init__.py`
- **Evidence**:
  - Top-level imports pull major solver classes immediately.
- **Risk/Impact**:
  - `import gcrack` can fail early due to heavy optional stack, and import time increases.
- **Recommendation**:
  - Consider lazy exports or lightweight package init.

### Fix (explicit)

1. In `src/gcrack/__init__.py`, avoid eager imports of heavy modules.
2. Prefer either:
  - lightweight public symbols only, or
  - lazy attribute loading (`__getattr__`) for `GCrackBase`, `ICrackBase`, `FCrackBase`.
3. Validate `import gcrack` in a minimal environment to confirm improved resilience.

---

## 10) API/typing and consistency quality notes

- **Severity**: Low
- **Locations**:
  - `src/gcrack/postprocess.py`: force vector allocated with fixed length 3 for all assumptions
  - `src/gcrack/utils/expression_parsers.py`: symbolic parser currently based on a single symbol `x`
  - various docstrings/typos (non-blocking)
- **Risk/Impact**:
  - Confusion and brittle behavior in edge cases.
- **Recommendation**:
  - Align shapes with model assumptions; clarify parser expectations and supported expression forms.

### Fix (explicit)

1. In `src/gcrack/postprocess.py`, shape force arrays by active component count where practical.
2. In `src/gcrack/utils/expression_parsers.py`, document supported expression grammar and variable convention (`x` as coordinate vector).
3. Add examples for 2D expressions (e.g., `x[0]`, `x[1]`) and validate with unit tests.
4. Clean minor docstring typos during next maintenance pass.

---

## Cross-Cutting Caveats (Packaging/CI affecting runtime confidence)

These are not direct source bugs but influence reliability:

1. **Conda dependency naming and availability sensitivity** (notably `fenics-*` stack).
2. **Windows CI is non-blocking by configuration**, so publish may proceed even when Windows build fails.
3. **Multi-Python matrix now includes 3.14**, but scientific stack availability may lag by platform.

---

## Prioritized Fix Order

1. Fix `ICrackBase -> LoadFactorSolver` constructor call mismatch.
2. Fix `FCrackBase.__post_init__` handling for `da=None` paths.
3. Move `ICrackBase` CSV write outside SIF loop.
4. Replace `fracture_dissipation = "TODO"` with numeric value (or explicit null strategy).
5. Add robust guards in point probing and DOF checks.
6. Validate/lock SIF T-stress scaling with benchmarks.

---

## Suggested Validation After Fixes

- Run existing tests under `tests/`.
- Add/extend targeted tests for:
  - `ICrackBase` run path solver invocation
  - `FCrackBase` with `dN` control and `da=None`
  - one-row-per-step CSV invariant
  - postprocess probe behavior outside mesh
  - SIF benchmark(s) for Williams branch

---

## Notes

This report intentionally avoids applying changes and is meant to serve as an actionable backlog for incremental hardening.
