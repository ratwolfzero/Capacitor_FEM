# capacitor_fem_graded_tol.py — Delta README

This document describes **only** the differences between the original
`capacitor_fem.py` (uniform Cartesian mesh, strict boundary comparisons)
and the combined variant `capacitor_fem_graded_tol.py`.

It does **not** repeat the physics, weak-form derivation, assembly details,
validation tables, or usage examples already covered in the main README.md.

---

## 1. Summary of changes

| Feature                 | Original                 | This variant                                        |
| ----------------------- | ------------------------ | --------------------------------------------------- |
| Mesh                    | Uniform Cartesian only   | Uniform **or** piecewise-uniform (graded) Cartesian |
| Boundary classification | Strict `<=` / `>=`       | Tolerant (`_BOUNDARY_TOL = 1e-9` m)                 |
| Startup check           | None                     | §10.4 floating-point stress test                    |
| Dependencies            | numpy, scipy, matplotlib | **unchanged** (still zero native deps)              |

Everything else — configuration dataclasses, CSG shapes, material functions,
stiffness assembly, Dirichlet solve, energy/capacitance post-processing,
high-level `ElectrostaticProblem` API, and the two worked examples —
remains bit-compatible with the original when the new options are left at
their defaults.

---

## 2. Graded Cartesian mesh (README §11 intermediate step)

### Motivation Graded Cartesian mesh

A uniform mesh spends the same resolution everywhere. Near plate corners
(geometric singularities) and through the dielectric gap the field varies
rapidly; far away it is almost constant. A graded mesh concentrates nodes
where they matter and coarsens the far-field, reducing total degrees of
freedom for a given local accuracy.

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
- **y-direction**: coarse outer margins → medium through plates → fine through the gap (and dielectric interface)

All geometry lines that were snapped with `snap_to_grid` remain exactly
on mesh lines, so axis-aligned `contains()` classification stays exact.

The production solve in `example_parallel_plate()` automatically runs both
a uniform and a graded mesh at the finest `h` and reports the capacitance
delta.

### Caveats (unchanged from main README)

- Still structured / non-conforming → curved boundaries remain staircased.
- Transition zones produce mildly stretched triangles; harmless for linear
  electrostatics but can raise the condition number of \(K\).
- Node-count heuristics are empirical; they keep total nodes comparable to
  the uniform mesh at the same nominal `h`, but are not error-driven.

---

## 3. §10.4 boundary-tolerance hardening

### Motivation boundary-tolerance hardening

`snap_to_grid` and `np.linspace` can produce boundary coordinates that
differ by a few ULPs even when they are mathematically identical. A strict
`<=` / `>=` comparison then flips an entire row or column of nodes,
changing capacitance by several percent. This was observed in practice
while implementing the `mesh_spacing → convergence_spacings` auto-derivation.

### The fix

A single constant:

```python
_BOUNDARY_TOL = 1e-9  # meters
```

applied in every primitive `contains()`:

```python
# Rectangle
(x >= self.x0 - _BOUNDARY_TOL) & (x <= self.x0 + self.width + _BOUNDARY_TOL) & ...

# Circle
(x-cx)**2 + (y-cy)**2 <= (radius + _BOUNDARY_TOL)**2

# OutsideCircle (complement — tolerance applied inward)
(x-cx)**2 + (y-cy)**2 >= (radius - _BOUNDARY_TOL)**2
```

CSG shapes (`Union`, `Intersection`, `Difference`) simply combine the
masks returned by the primitives, so the tolerance propagates automatically.

The tolerance is:

- many orders of magnitude larger than observed float64 noise (~1e-18 … 1e-15 m),
- many orders of magnitude smaller than the finest grid spacing used in the
  project (75 µm),  
  therefore it can only rescue a node that arithmetic nudged off its
  intended exact position; it cannot reach a neighbouring grid point.

### Startup stress test

`verify_boundary_tolerance()` runs automatically when the script is
executed. It reconstructs the same edge coordinate several different ways
(including the classic `1.5 * 0.1e-3 * …` pattern) and confirms that
classification remains stable. A failure would print a clear diagnostic.

---

## 4. What did *not* change

- Governing PDE, weak form, P1 element formulas, sparse assembly.
- `snap_to_grid` behaviour and the grid-alignment notes in the convergence
  tables.
- Exact-solution validation (`example_exact_check`) — still machine-precision.
- Coax example (still uniform mesh; grading circular features would need a
  different segmentation strategy).
- Memory-warning heuristics, high-level `ElectrostaticProblem` API,
  four-panel visualisation.

---

## 5. Quick start

```bash
python3 capacitor_fem_graded_tol.py
```

Expected console output order:

1. §10.4 stress-test table (all points classified inside)
2. Exact-solution check (relative error ~1e-16)
3. Parallel-plate convergence (uniform) + graded production solve
4. Coax convergence
5. Summary table + total runtime

Figures are written to the working directory
(`example1_parallel_plate.png`, `example2_coax.png`).

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

## 6. Relationship to the main README roadmap

| Main README item                         | Status in this file                         |
| ---------------------------------------- | ------------------------------------------- |
| §11 Graded structured mesh               | **Implemented**                             |
| §10.4 Tolerance-based classification     | **Implemented**                             |
| §10.3 Cut-cell / area-fraction weighting | Not present (still future work)             |
| Unstructured / conforming mesh           | Not present (would add a native dependency) |

This variant stays strictly inside the original “one file, zero native
dependencies” constraint.
