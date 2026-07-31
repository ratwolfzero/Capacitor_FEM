# capacitor_fem_graded.py — Delta README

This document describes **only** the differences between the original
`capacitor_fem.py` (uniform Cartesian mesh) and the refactored variant
`capacitor_fem_graded_refactored.py`.

It does **not** repeat the physics, weak-form derivation, assembly details,
validation tables, or usage examples already covered in the main README.md.

---

## 1. Summary of changes

| Feature                    | Original                          | This variant                                              |
| -------------------------- | --------------------------------- | --------------------------------------------------------- |
| **Mesh (major)**           | Uniform Cartesian only            | Uniform **or** piecewise-uniform (graded) Cartesian       |
| **Startup check (major)**  | None                              | Optional §10.4 floating-point boundary-tolerance stress test |
| **Convergence figure**     | Console tables only               | Optional combined two-panel plot (`plot_convergence_study`) |
| Structure / tunables       | Literals scattered through the file | Central §0 “Tunable Runtime Parameters” block           |
| Config inheritance         | Duplicated `__post_init__` logic  | Shared `_AutoSpacingConfig` base class                    |
| Example helpers            | Monolithic `_solve_*` bodies      | Split geometry / mesh / print helpers                     |
| Plot / I/O defaults        | Always save; no interactive show  | Opt-in save, opt-in `plt.show()`, quieter notes           |
| Dependencies               | numpy, scipy, matplotlib          | **unchanged** (still zero native deps)                    |

The **major** functional additions are the graded Cartesian mesh, the
boundary-tolerance stress test, and the optional combined convergence figure.
Everything else is structural refactoring or minor default/UI adjustments that
leave the uniform-mesh physics path bit-compatible with the original when the
new switches are left at their defaults (or turned off).

---

## 2. Major change: Graded Cartesian mesh (README §11 intermediate step)

### Motivation

A uniform mesh spends the same resolution everywhere. Near plate corners
(geometric singularities) and through the dielectric gap the field varies
rapidly; far away it is almost constant. A graded mesh concentrates nodes
where they matter and coarsens the far-field, reducing total degrees of
freedom for a given local accuracy — without adding any native dependency.

### New primitive: `build_graded_coords`

```python
xs = build_graded_coords([
    (start, end, n_points),   # linspace(start, end, n_points)
    (end,   end2, n_points2), # consecutive segments must abut
    ...
])
```

- Segments are contiguous; the shared junction point is kept only once.
- Result is a strictly increasing 1-D coordinate array.
- Handed to `Mesh(xs=xs, ys=ys)`.

### Extended `Mesh` constructor

```python
# Original (still works)
mesh = Mesh(x0, y0, Lx, Ly, nx=nx, ny=ny)

# Graded
mesh = Mesh(xs=xs, ys=ys)
# or Mesh(x0, y0, Lx, Ly, xs=xs, ys=ys)  # x0/y0/Lx/Ly ignored for coords
```

Downstream code (assembly, fields, energy, plotting) is agnostic to
spacing; it only consumes `mesh.points` and `mesh.triangles`.

### Activation in the parallel-plate example

```python
result = _solve_parallel_plate(config, h, use_graded=True)
```

When `use_graded=True` the helper builds:

- **x-direction**: coarse margins → fine edge bands (plate ends) → medium plate interior → fine edge bands → coarse margins
- **y-direction**: coarse outer margins → medium through plates → fine through the gap (split at the dielectric interface so the material boundary is a segment junction)

All geometry lines that were snapped with `snap_to_grid` remain exactly
on mesh lines, so axis-aligned `contains()` classification stays exact.

Tuning knobs live in the frozen dataclass `GradedMeshTuning` (and the
module-level `GRADED_MESH_DEFAULTS` instance) in §0 of the script:
edge-band width factors, per-region spacing multipliers, and minimum
point counts.

The production solve in `example_parallel_plate()` can run both a uniform
and a graded mesh at the finest `h` and report the capacitance delta
(controlled by `RUN_GRADED_COMPARISON`).

### Streamline plotting on a graded mesh

`matplotlib.streamplot` requires equally spaced coordinates. On a graded
mesh the field is therefore interpolated onto a temporary uniform grid
via `scipy.interpolate.RegularGridInterpolator` before streamlines are
drawn; the colour maps and energy-density panel still use the native
(non-uniform) mesh.

### Caveats (unchanged from main README)

- Still structured / non-conforming → curved boundaries remain staircased.
- Transition zones produce mildly stretched triangles; harmless for linear
  electrostatics but can raise the condition number of \(K\).
- Node-count heuristics are empirical; they keep total nodes comparable to
  the uniform mesh at the same nominal `h`, but are not error-driven.
- The coax example still uses a uniform mesh; grading circular features
  would need a different segmentation strategy.

### Using the graded mesh from your own code

```python
from capacitor_fem_graded_tol import (
    ParallelPlateConfig, _solve_parallel_plate, build_graded_coords, Mesh
)

cfg = ParallelPlateConfig()
# uniform
r_uni = _solve_parallel_plate(cfg, cfg.mesh_spacing, use_graded=False)
# graded
r_grd = _solve_parallel_plate(cfg, cfg.mesh_spacing, use_graded=True)

print(r_uni['C']*1e12, r_grd['C']*1e12)
```

Or build a graded mesh directly:

```python
xs = build_graded_coords([(0, 0.01, 20), (0.01, 0.05, 80), (0.05, 0.06, 15)])
ys = build_graded_coords([(0, 0.02, 30), (0.02, 0.04, 60), (0.04, 0.06, 30)])
mesh = Mesh(xs=xs, ys=ys)
```

---

## 3. Major change: §10.4 boundary-tolerance hardening

### Named tolerance

The original module-level `_BOUNDARY_TOL = 1e-9` is exposed as the
documented constant `BOUNDARY_TOLERANCE_M` in the central tunables block.
Its value and role are unchanged: it absorbs float64 ULP noise so that
arithmetically equivalent boundary coordinates classify the same way
under `Shape.contains()`.

### Optional startup stress test

`verify_boundary_tolerance()` reconstructs the same edge coordinate
several different ways (including the classic `1.5 * 0.1e-3 * …` pattern
and ±5e-15 nudges) and confirms that classification remains stable.
It is gated by `RUN_BOUNDARY_STRESS_TEST` (default `False` in the current
refactored defaults). A failure prints a clear diagnostic and returns
`False`.

---

## 4. Combined convergence figure (`plot_convergence_study`)

### Purpose

The console convergence tables (unchanged in content) are useful for
precise numbers, but a single figure makes the two examples’ very
different convergence behaviour visible at a glance: fringing-dominated
offset for the parallel plate versus pure staircase error for the coax.

### Activation

```python
PLOT_CONVERGENCE: bool = True   # Section 0 switch
```

When enabled, `main` calls `plot_convergence_study(...)` after both
examples have finished. The figure respects the same I/O switches as the
field plots (`SHOW_PLOTS`, `SAVE_FIGURES`). Saved name:
`{OUTPUT_DIR}/convergence_study.png`.

### Layout

One figure, **two side-by-side panels**. Horizontal axis is mesh spacing
\(h\) in mm, **inverted** (finer mesh to the right). Each panel uses a
**twin y-axis**: capacitance [pF/m] on the left, relative difference or
error [%] on the right.

#### Left panel — Parallel-plate (partial dielectric slab)

| Series | Colour | Marker / style | Axis | Meaning |
| ------ | ------ | -------------- | ---- | ------- |
| FEM (uniform) | blue | circles, solid | left (C) | Capacitance from the energy method on the uniform mesh at each \(h\) in `convergence_spacings` |
| Ideal (no fringing) | red | squares, dashed | left (C) | Series-dielectric formula \(C' = \varepsilon_0 w / (d_1/\varepsilon_{r1} + d_2/\varepsilon_{r2})\), evaluated with the **snapped** gap and dielectric thickness at that \(h\) |
| FEM (graded) | orange | star | left (C) | Single production-resolution solve on the graded mesh (only if `RUN_GRADED_COMPARISON`) |
| (FEM − ideal) / ideal | green | triangles | **right** (%) | Relative difference \(100\cdot(C_\text{FEM}-C_\text{ideal})/C_\text{ideal}\) |

**How to read the green curve.** It is **not** pure discretisation or
staircase error. The plates are finite, so the FEM result lies above the
ideal formula mainly because of **fringing fields** at the plate edges
(~+17 % at the shipped production resolution). Mesh error is only a
small part of that offset. The ideal reference itself can step slightly
between \(h\) values when `snap_to_grid` rounds a physically meaningful
dimension (visible as a non-flat red series on “rounded” rows — see main
README §8.2).

#### Right panel — Coaxial cable (polyethylene fill)

| Series | Colour | Marker / style | Axis | Meaning |
| ------ | ------ | -------------- | ---- | ------- |
| FEM | blue | circles, solid | left (C) | Capacitance from the energy method at each \(h\) |
| Analytical \(2\pi\varepsilon/\ln(b/a)\) | red | horizontal dashed line | left (C) | Exact coax formula; constant because radii are **not** snapped |
| Staircase / mesh error | green | triangles | **right** (%) | Relative error \(100\cdot(C_\text{FEM}-C_\text{analytical})/C_\text{analytical}\) |

**How to read the green curve.** Here the analytical formula *is* the
exact solution of the continuous problem for true circular boundaries.
The only mismatch is the Cartesian **staircase** approximation of those
circles on a structured mesh (main README §10.1). The green series is
therefore almost pure mesh/staircase error and should approach 0 % as
\(h\) shrinks (with the small non-monotonic wiggles discussed in §8.2).

### Dynamic reference values

All reference capacitances are taken from the result dicts produced by
the live solve, not from hard-coded numbers. Changing
`ParallelPlateConfig` or `CoaxConfig` (gap, \(\varepsilon_r\), radii, …)
automatically updates both the ideal/analytical lines and the relative
error curves.

### Implementation notes

- `example_parallel_plate` returns `(C, C_ideal, results, graded_or_None)`.
- `example_coax` returns `(C, C_ideal, results)`.
- `plot_convergence_study(pp_results, coax_results, graded_result=…)`
  builds the figure; it does not re-solve anything.
- Twin axes are intentional: absolute \(C\) (~80–100 pF/m) and relative
  error (a few percent to tens of percent) cannot share one scale
  usefully.

---

## 5. Refactoring subelements (non-physics)

These changes do not alter the governing PDE, weak form, assembly, or
uniform-mesh numerical results. They reorganise the code and expose a few
runtime switches.

### 5.1 Central tunables block (§0)

Every behavioural switch, path, tolerance, and heuristic multiplier is
declared at the top of the file:

| Switch / constant              | Default (refactored) | Role |
| ------------------------------ | -------------------- | ---- |
| `RUN_BOUNDARY_STRESS_TEST`     | `False`              | Run §10.4 stress test on startup |
| `RUN_EXACT_CHECK`              | `False`              | Run machine-precision exact-solution validation |
| `RUN_GRADED_COMPARISON`        | `True`               | Also solve parallel-plate on a graded mesh and report ΔC |
| `PLOT_CONVERGENCE`             | `True`               | Draw the combined two-panel convergence figure after both examples |
| `SAVE_FIGURES`                 | `False`              | Write PNGs to `OUTPUT_DIR` |
| `SHOW_PLOTS`                   | `True`               | Call `plt.show()` after each figure |
| `VERBOSE_CONVERGENCE_NOTES`    | `False`              | Print the long explanatory notes after convergence tables |
| `OUTPUT_DIR`                   | `"Development"`      | Directory for output figures |
| `BOUNDARY_TOLERANCE_M`         | `1e-9`               | Absolute tolerance [m] for `Shape.contains()` |
| Memory-estimate coefficients   | same as original     | Peak-RSS power-law heuristics |
| `GRADED_MESH_DEFAULTS`         | `GradedMeshTuning()` | All graded-mesh spacing / band knobs |

In the original script the corresponding values were either hard-coded
literals or a single `RUN_EXACT_CHECK = True` near the bottom.

### 5.2 Shared config base class

`ParallelPlateConfig` and `CoaxConfig` both inherit from
`_AutoSpacingConfig`, which implements the convergence-spacing
auto-derivation once. Behaviour is identical to the original duplicated
`__post_init__` logic.

### 5.3 Split example helpers

The parallel-plate path is broken into single-purpose helpers:

- `_build_parallel_plate_geometry` — snap sizes, build conductors & dielectric
- `_build_graded_parallel_plate_mesh` — graded `xs` / `ys` from those sizes
- `_solve_parallel_plate(..., use_graded=False)` — orchestration
- `_print_parallel_plate_convergence` / `_print_parallel_plate_convergence_notes`
- `print_summary` — final table

The coax and exact-check paths are structurally the same as the original
(still uniform-mesh only).

### 5.4 Plot / I/O defaults

| Item                    | Original      | Refactored                          |
| ----------------------- | ------------- | ----------------------------------- |
| `streamline_density`    | `1.5`         | `1.9`                               |
| Figure save             | Always        | Only if `SAVE_FIGURES`              |
| Interactive display     | Never         | If `SHOW_PLOTS`                     |
| Long convergence notes  | Always printed| If `VERBOSE_CONVERGENCE_NOTES`      |
| Convergence figure      | None          | If `PLOT_CONVERGENCE`               |
| Default `OUTPUT_DIR`    | `""` (cwd)    | `"Development"`                     |

On a graded mesh the streamline panel additionally interpolates onto a
temporary uniform grid (see §2); colour maps and energy density still use
the native mesh.

### 5.5 What remains bit-compatible

When graded comparison and the stress test are left off (or
`use_graded=False`), the uniform-mesh path produces the same capacitance
numbers, the same convergence tables, and the same exact-check residuals
as the original. The governing PDE, weak form, P1 formulas, sparse
assembly, energy method, `snap_to_grid`, and high-level
`ElectrostaticProblem` API are unchanged. The convergence figure is pure
post-processing of those same results.

---

## 6. What did *not* change (physics & validation)

- Governing PDE, weak form, P1 element formulas, sparse assembly.
- `snap_to_grid` behaviour and the grid-alignment notes in the convergence
  tables.
- Exact-solution validation (`example_exact_check`) — still machine-precision
  when enabled.
- Coax example (still uniform mesh).
- Memory-warning heuristics (same fitted power law).
- High-level `ElectrostaticProblem` API and four-panel visualisation layout.

---

## 7. Expected console / figure output (with defaults as shipped)

With the refactored defaults (`RUN_BOUNDARY_STRESS_TEST=False`,
`RUN_EXACT_CHECK=False`, `RUN_GRADED_COMPARISON=True`,
`PLOT_CONVERGENCE=True`, `VERBOSE_CONVERGENCE_NOTES=False`):

1. Parallel-plate convergence (uniform) + graded production solve and ΔC
2. Coax convergence
3. Combined convergence figure (two panels; interactive if `SHOW_PLOTS`)
4. Summary table + total runtime

Enable the stress test and/or exact check via the §0 switches to restore
the longer original startup sequence. Figures are written only when
`SAVE_FIGURES=True` (field plots plus `convergence_study.png` when
`PLOT_CONVERGENCE` is also on).

---

## 8. Relationship to the main README roadmap

| Main README item                         | Status in this file                         |
| ---------------------------------------- | ------------------------------------------- |
| §11 Graded structured mesh               | **Implemented**                             |
| §10.4 Boundary-tolerance hardening       | **Implemented** (named constant + optional stress test) |
| §10.3 Cut-cell / area-fraction weighting | Not present (still future work)             |
| Unstructured / conforming mesh           | Not present (would add a native dependency) |
| Symmetry-aware / iterative solvers       | Not present (still future work)             |
| Nonlinear / anisotropic / floating BC / 3-D | Not present (still future work)          |

This variant stays strictly inside the original “one file, zero native
dependencies” constraint.
