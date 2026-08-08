# Delta: optional rounded plate edges (`edge_radius`)

Opt-in, backward-compatible. `edge_radius=0.0` (default) is bit-for-bit
identical to the pre-delta file — see *Verification*.

## Changed

| Section | Symbol | Change |
|---|---|---|
| §3 GEOMETRY | `RoundedRectangle(Shape)` | new class |
| §2 CONFIGURATION | `ParallelPlateConfig.edge_radius` | new field, `float = 0.0`, + `__post_init__` bound |
| §10 `_build_parallel_plate_geometry` | — | emits `RoundedRectangle` instead of `Rectangle` for both plates when radius > 0; re-clamps radius to snapped dims; adds `dims["edge_radius"]` |
| §10 `_build_graded_parallel_plate_mesh` | — | widens `edge_band` to cover the fillet |
| §10 `example_parallel_plate` | — | prints edge treatment; plot titles note radius when > 0 |
| §9 `plot_solution` | `emag_vmin`, `emag_vmax` | new optional params, default `None` (unchanged autoscale). **General plotting utility, not rounded-edges-specific** — added so two separate calls (e.g. sharp vs. rounded) can share one `|E|` color scale; see *Known limitations* and *Usage*. |

**Untouched:** `Mesh`, `assemble_stiffness`, `apply_conductors_and_solve`,
`compute_fields`, `plot_solution`, `example_coax`, `_solve_exact_check`,
`_grid_alignment_note`. All conductor-consuming code already goes through
`Shape.contains(x, y)` polymorphically, so none of it needed to know a new
shape exists.

## `RoundedRectangle`

Rectangle `[x0, x0+width] × [y0, y0+height]`, all four corners filleted to
radius `r`. Exact rounded-box signed-distance test (not a polygon
approximation). With `cx, cy` the center, `ex = width/2 − r`, `ey = height/2 − r`:

```
qx = |x − cx| − ex
qy = |y − cy| − ey
d  = hypot(max(qx,0), max(qy,0)) + min(max(qx,qy), 0) − r
inside  ⇔  d ≤ BOUNDARY_TOLERANCE_M
```

`r = 0` short-circuits to the plain `Rectangle` test (identical output, not
just "close").

**Validation** (`ValueError`, same style as `Rectangle`/`Circle`):
`width, height` finite & > 0; `radius` finite & ≥ 0; and
`radius ≤ 0.5·min(width, height) + BOUNDARY_TOLERANCE_M`, else the four
fillets would overlap. Stored radius is clamped to exactly
`0.5·min(width, height)` to absorb float overshoot at equality.

## `ParallelPlateConfig.edge_radius`

One radius, shared by both plates (mirrors `plate_thickness`, also shared).
`__post_init__` requires:

```
0 ≤ edge_radius ≤ 0.5 · min(plate_thickness, bottom_plate_width, top_plate_width)
```

**Grid-snap re-clamp.** `snap_to_grid()` can shrink `plate_thickness` at a
coarse `h` below what the nominal check above saw. `_build_parallel_plate_geometry`
therefore recomputes, per `h`, without raising:

```
edge_radius_used = min(config.edge_radius, 0.5·plate_t, 0.5·bottom_w, 0.5·top_w)
```

silent by design, matching how snap-to-grid already perturbs other dims
elsewhere in the file (`_grid_alignment_note`). Exposed as
`dims["edge_radius"]` / `result["edge_radius"]` for inspection per step of a
convergence sweep.

## Graded mesh

`edge_band` (fine-spacing zone at each plate end) widens when rounding is
active:

```
edge_band = max(edge_band, edge_radius + edge_band_width_factor · h)
```

so the curved region isn't left in coarse interior spacing. Feeds unchanged
into the existing narrow-plate fallback (`plate_w ≤ 2·edge_band`).

## Known limitations: reading the `|E|` panel

**General to `capacitor_fem` — not caused by this delta, applies with or
without `edge_radius`:**

- **Colorbar is per-plot, not absolute.** The `|E|` panel calls
  `ax.pcolormesh(X, Y, EmagG_masked, ...)` with no `vmin`/`vmax`, so each
  figure autoscales to *its own* peak. Any two runs with a different peak
  field — different `h`, `voltage`, `gap`, rounded or not — aren't visually
  comparable; an unchanged value can render a different color purely
  because the *other* run's scale moved. Compare `result["C"]` or sampled
  field values, never colorbar hue. **Now fixable:**
  `plot_solution(..., emag_vmin=, emag_vmax=)` (both default `None`,
  unchanged) lets two calls share one scale — see *Usage*.
- **A sharp 90° conductor corner is a true field singularity**
  (`E ~ r^-1/3`, the Motz-problem exponent for its 270° reentrant angle),
  present in every rectangular plate before `edge_radius` existed. Its
  FEM-reported peak grows without bound as `h` shrinks and never converges:
  default-size plate, `edge_radius=0`: 46.3 → 56.3 → 59.8 → 69.0 →
  85.7 kV/m for h=0.4/0.2/0.15/0.1/0.05mm, fitted exponent −0.31 vs.
  theoretical −1/3. Rounding doesn't cause this — it's the fix for it.

**Specific to `edge_radius > 0`:** the now-*finite* rounded-corner peak
still converges slowly and non-monotonically on this mesh — same plate,
`edge_radius=0.5mm`: 52.5 → 56.3 → 55.9 → 59.7 → 62.3 kV/m over the same
five spacings, ~5× less sensitive than the sharp case at the finest step
but not settled. This is *not* because rounding "reduces the singularity"
— it's the generic cost of representing any curved boundary via node
membership on a structured grid with no local 2-D refinement (the same
mechanism already applies to `Circle`/`OutsideCircle` in `example_coax`;
rounding just brings it into the parallel-plate example for the first
time). Concretely, in `_build_graded_parallel_plate_mesh`: both the
x-spacing near a plate edge (`edge_spacing_factor · h`) and the y-spacing
through the plate thickness (`plate_spacing_factor · h`) scale with `h`
alone, not with `radius` — so the number of mesh points spanning the arc
scales as `r/h` in both directions. At `r=0.5mm` (capped at half the 1mm
plate thickness) and `h=0.05mm`, that's `r/h=10`: not enough for a smooth
approximation of a curve. A larger radius (thicker plate) would likely
converge faster, since more points would land on a bigger circle at the
same `h` — not tested here.

Trust `C`, `C_ideal`, and field values away from plate edges (Verification
#6) regardless of `edge_radius`; treat any near-edge peak reading — sharp
or rounded — as order-of-magnitude only.

## Out of scope

- Dielectric slab stays a plain `Rectangle` (unrounded).
- `example_coax`, `_solve_exact_check`: untouched — not applicable.
- Independent bottom/top radii: not implemented (single shared field only).

## Verification

1. **Regression** (`edge_radius=0.0`): dims dict, both plates' `contains()`
   masks over a 400×300 sample grid, nodal `V`, and `C` compared against the
   unmodified file — bit-for-bit identical, uniform and graded mesh.
2. **Unit**: `radius=0 ≡ Rectangle`; sharp corner excluded; edge midpoints
   included; invalid radius (> bound, negative) raises.
3. **Coarse-`h` clamp**: nominal `plate_thickness=1mm`, `edge_radius=0.45mm`
   (valid). At `h=0.4mm`, `plate_thickness` snaps to `0.8mm`; used radius
   correctly reduces to `0.4mm`, no exception.
4. **Narrow-plate fallback**: `bottom_plate_width=top_plate_width=3mm`,
   `edge_radius=0.5mm` (max) — graded mesh's narrow-plate branch solves
   correctly.
5. **End-to-end**: 1mm plate, 0.45mm fillet — fillets confirmed in all four
   plot panels; FEM-vs-ideal excess (fringing + corner concentration) fell
   from +28.7% (sharp) to +26.7% (rounded) at identical `h`.
6. **Bulk-field invariance**: sampled `|E|` at plate center and 5–7mm in
   from either edge, in both the dielectric and air-gap layers,
   `edge_radius=0` vs `0.5mm` at `h=0.1mm` (default finest spacing) — agree
   to ≤0.015%, and both match the ideal 1D series-capacitor formula to
   ~0.001%. Rounding the edges doesn't perturb the field away from them.
7. **`emag_vmin`/`emag_vmax`**: isolated check that `pcolormesh(...,
   vmin=X, vmax=Y)` clips to exactly `(X, Y)` and reproduces plain
   autoscale when both are `None`; generated a shared-scale sharp-vs-rounded
   comparison and confirmed both colorbars now show the same range with the
   interior rendering identically in both (difference correctly localized
   to the corners); confirmed a full `example_parallel_plate()` call with
   no override reproduces the pre-addition output byte-for-byte.

## Usage

```python
config = ParallelPlateConfig(edge_radius=0.4e-3)  # 0.4 mm fillet, both plates
C, C_ideal, results, graded = example_parallel_plate(config)
```

**Comparable-scale sharp vs. rounded plots** (see *Known limitations*):

```python
import numpy as np

def emag_peak(result):
    mesh = result["mesh"]
    X, Y = np.meshgrid(mesh.xs, mesh.ys)
    dVdy, dVdx = np.gradient(result["V"].reshape(mesh.ny, mesh.nx), mesh.ys, mesh.xs)
    Emag = np.hypot(dVdx, dVdy)
    cond = np.zeros_like(X, dtype=bool)
    for c in result["conductors"]:
        cond |= c.contains(X, Y)
    return np.ma.masked_where(cond, Emag).max()

h = 0.1e-3
r_sharp   = _solve_parallel_plate(ParallelPlateConfig(edge_radius=0.0),    h, use_graded=True)
r_rounded = _solve_parallel_plate(ParallelPlateConfig(edge_radius=0.4e-3), h, use_graded=True)
shared_vmax = max(emag_peak(r_sharp), emag_peak(r_rounded))

for label, r in [("sharp", r_sharp), ("rounded", r_rounded)]:
    plot_solution(r["mesh"], r["V"], r["eps_r_of_xy"], r["energy_density"],
                 r["conductors"], r["is_fixed"], f"{label} edges", f"{label}.png",
                 Ex=r["Ex"], Ey=r["Ey"], emag_vmin=0.0, emag_vmax=shared_vmax)
```
