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

Two things worth knowing before comparing field plots or peak-field numbers
across sharp vs. rounded runs. Neither is new in this delta — both are
pre-existing characteristics of `plot_solution` / the underlying mesh — but
`edge_radius` is what makes cross-run comparison worth doing, so they're
easy to trip over now.

**Colorbar is per-plot, not absolute.** The `|E|` panel calls
`ax.pcolormesh(X, Y, EmagG_masked, ...)` with no `vmin`/`vmax`, so each
figure autoscales to *its own* peak. Two saved PNGs are not visually
comparable — an unchanged interior value can render a different color
purely because the *other* run's peak (and hence its scale) moved. Compare
`result["C"]` or explicit sampled field values, never colorbar hue, across
runs.

**Peak `|E|` at a plate edge — sharp or rounded — is not mesh-converged at
the spacings this file uses.** A sharp 90° conductor corner has a genuine
continuum-field singularity, `E ~ r^-1/3` (the classic Motz-problem exponent
for its 270° reentrant field angle); its FEM-reported peak grows without
bound as `h` shrinks and never settles — default-size plate,
`edge_radius=0`: 46.3 → 56.3 → 59.8 → 69.0 → 85.7 kV/m for
h = 0.4/0.2/0.15/0.1/0.05 mm, fitted exponent −0.31 vs. theoretical −1/3. A
rounded corner has a genuine finite limit, but this structured-grid mesh
represents the arc only through node membership (no local 2-D refinement
there), so it approaches that limit slowly and non-monotonically — same
plate, `edge_radius=0.5mm`: 52.5 → 56.3 → 55.9 → 59.7 → 62.3 kV/m over the
same five spacings, roughly 5× less sensitive than the sharp case at the
finest step but not settled. Trust `C`, `C_ideal`, and field values away
from plate edges (Verification #6); treat any near-edge peak reading as
order-of-magnitude only.

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

## Usage

```python
config = ParallelPlateConfig(edge_radius=0.4e-3)  # 0.4 mm fillet, both plates
C, C_ideal, results, graded = example_parallel_plate(config)
```
