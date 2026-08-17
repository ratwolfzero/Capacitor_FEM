"""
cfu_driver.py — Practical driver & usage guide for capacitor_fem_universal.py

Treat capacitor_fem_universal.py as a *frozen* solver template.
This file shows how to drive it from the outside without modifying the solver.

What you will see here
----------------------
1. Importing the solver as a module and controlling it externally.
2. Runtime toggles (SHOW_PLOTS, SAVE_FIGURES) that control UI and file output.
3. Quick run of the built-in parallel-plate example (uniform / graded mesh).
4. Single-parameter override + shared-|E| comparison (compare_parallel_plate_runs).
5. Full dual-configuration comparison (different geometry + materials).
6. Two-axis convergence sweep: mesh_spacing (h) × domain_margin (m).
7. Custom geometry built with CSG + the high-level ElectrostaticProblem API
   (symmetrical slit in the top plate).
8. Default parallel-plate geometry with a small air bubble inside the glass slab.

Run the whole suite:
    python3 cfu_driver.py

Or call individual demo_* functions from a notebook / interactive session.
"""

# ---------------------------------------------------------------------------
# Standard-library helpers used by *this* driver script itself.
# ---------------------------------------------------------------------------
from dataclasses import replace
import os

# ---------------------------------------------------------------------------
# Import the solver as a module.
#
# Keeping the solver behind the "cfu" namespace makes it explicit that this
# file is an external driver of capacitor_fem_universal.py.
# ---------------------------------------------------------------------------
import capacitor_fem_universal as cfu


# =============================================================================
# Helper: shared geometric quantities derived from ParallelPlateConfig
# =============================================================================
def make_parallel_plate_domain(config: cfu.ParallelPlateConfig | None = None) -> dict:
    """
    Return the basic geometric quantities that both the built-in
    parallel-plate path and custom CSG demos usually need.

    Starting from a ParallelPlateConfig (or the shipped defaults) keeps
    custom geometries in sync with the rest of the project when the
    default plate dimensions change.
    """
    cfg = config or cfu.ParallelPlateConfig()

    plate_w = max(cfg.bottom_plate_width, cfg.top_plate_width)
    plate_t = cfg.plate_thickness
    gap = cfg.gap
    margin = cfg.domain_margin

    Lx = plate_w + 2 * margin
    Ly = 2 * plate_t + gap + 2 * margin

    return {
        "config": cfg,
        "plate_w": plate_w,
        "plate_t": plate_t,
        "gap": gap,
        "margin": margin,
        "Lx": Lx,
        "Ly": Ly,
        "x0": -Lx / 2.0,
        "y0": -Ly / 2.0,
    }


# =============================================================================
# 1. Quick parallel-plate run with runtime toggles
# =============================================================================
def demo_quick_parallel_plate(show_ui: bool = False,
                              save_png: bool = True,
                              use_graded: bool = True):
    """
    Run the built-in parallel-plate example while controlling plotting and
    file output *without* editing the solver source.

    Parameters
    ----------
    show_ui : bool
        If True, open interactive plot windows (desktop only).
    save_png : bool
        If True, write PNG figures to the current directory / OUTPUT_DIR.
    use_graded : bool
        Forwarded to the module-level RUN_GRADED_COMPARISON switch, which
        example_parallel_plate() checks to decide whether to also run and
        report a graded-mesh solve alongside the uniform-mesh convergence
        sweep.
    """
    print("\n=== 1. Quick Parallel-Plate Run ===")
    print(f"  SHOW_PLOTS   = {show_ui}")
    print(f"  SAVE_FIGURES = {save_png}")

    # Toggle global switches on the imported module
    cfu.SHOW_PLOTS = show_ui
    cfu.SAVE_FIGURES = save_png
    cfu.RUN_GRADED_COMPARISON = use_graded

    # Optional: silence the long convergence notes
    cfu.VERBOSE_CONVERGENCE_NOTES = False

    # Run the official example
    # (returns C_uniform, C_ideal, results, graded)
    C_uniform, C_ideal, results, graded = cfu.example_parallel_plate()

    print(f"\n  Uniform-mesh C  = {C_uniform * 1e12:.4f} pF/m")
    print(f"  Ideal (no fringe) = {C_ideal * 1e12:.4f} pF/m")

    if graded is not None:
        print(f"  Graded-mesh C   = {graded['C'] * 1e12:.4f} pF/m")
        print(f"  Δ (graded vs uniform) = "
              f"{100 * (graded['C'] - C_uniform) / C_uniform:+.3f} %")


# =============================================================================
# 2. Single-parameter override + shared-|E| comparison
# =============================================================================
def demo_single_parameter_override():
    """
    Start from the default ParallelPlateConfig, change only one field
    (here: plate gap), and compare the two runs on a *shared* |E| colour
    scale so the plots are visually honest.
    """
    print("\n=== 2. Single-Parameter Override (gap) ===")

    cfu.SHOW_PLOTS = False
    cfu.SAVE_FIGURES = True

    base = cfu.ParallelPlateConfig()                  # all defaults
    wide = replace(base, gap=6e-3)                    # only gap changed

    print("  baseline gap = 4 mm")
    print("  modified gap = 6 mm")
    print("  Running compare_parallel_plate_runs …")

    cfu.compare_parallel_plate_runs(
        config_a=base,
        config_b=wide,
        label_a="gap_4mm",
        label_b="gap_6mm",
        fname_prefix="compare_gap",
        use_graded=True,
    )

    print("  → figures: compare_gap_gap_4mm.png  /  compare_gap_gap_6mm.png")


# =============================================================================
# 3. Full dual-configuration comparison
# =============================================================================
def demo_two_full_configs():
    """
    Compare two completely independent configurations that differ in
    several parameters at once (geometry + material + edge treatment).
    """
    print("\n=== 3. Dual Full-Configuration Comparison ===")

    cfu.SHOW_PLOTS = False
    cfu.SAVE_FIGURES = True

    # Config A – asymmetric plates, sharp corners, low-k dielectric
    config_a = cfu.ParallelPlateConfig(
        bottom_plate_width=24e-3,
        top_plate_width=18e-3,
        edge_radius=0.0,
        dielectric_eps_r=2.2,          # PTFE-like
        mesh_spacing=0.1e-3,
    )

    # Config B – symmetric plates, rounded edges, high-k dielectric
    config_b = cfu.ParallelPlateConfig(
        bottom_plate_width=24e-3,
        top_plate_width=24e-3,
        edge_radius=0.4e-3,            # 0.4 mm fillet
        dielectric_eps_r=9.8,          # alumina-like
        mesh_spacing=0.1e-3,
    )

    print("  A: asymmetric + sharp + PTFE (εr=2.2)")
    print("  B: symmetric  + fillet + alumina (εr=9.8)")
    print("  Running compare_parallel_plate_runs …")

    cfu.compare_parallel_plate_runs(
        config_a=config_a,
        config_b=config_b,
        label_a="ptfe_asymmetric_sharp",
        label_b="alumina_symmetric_fillet",
        fname_prefix="compare_materials_geom",
        use_graded=True,
    )

    print("  → figures: compare_materials_geom_*.png")


# =============================================================================
# 4. Two-axis convergence sweep (h × domain_margin)
# =============================================================================
def demo_two_axis_convergence_sweep():
    """
    Sweep the two independent convergence axes documented in the README:

      • mesh_spacing  (h)  – discretisation error
      • domain_margin (m)  – domain-truncation error

    A small grid is used here so the demo finishes quickly.
    Expand the lists for a production study.
    """
    print("\n=== 4. Two-Axis Convergence Sweep (h × margin) ===")

    cfu.SHOW_PLOTS = False
    cfu.SAVE_FIGURES = False

    h_list = [0.20e-3, 0.10e-3]                 # [m]
    margin_list = [10e-3, 15e-3, 25e-3]         # [m]

    header = (f"{'h [mm]':>8s} | {'margin [mm]':>11s} | {'nodes':>8s} | "
              f"{'C [pF/m]':>10s} | {'solve [s]':>9s}")

    print(header)
    print("-" * len(header))

    for h in h_list:
        for m in margin_list:
            cfg = cfu.ParallelPlateConfig(
                mesh_spacing=h,
                domain_margin=m,
                # single-level “sweep” so we solve exactly once per (h,m)
                convergence_spacings=(h,),
            )

            # Call the internal helper directly (still public enough for studies)
            res = cfu._solve_parallel_plate(cfg, h, use_graded=True)

            print(
                f"{h*1e3:8.2f} | {m*1e3:11.1f} | "
                f"{res['mesh'].n_nodes:8d} | "
                f"{res['C']*1e12:10.4f} | "
                f"{res['solve_time']:9.3f}"
            )

    print("\n  Tip: the default production margin (15 mm) is a compromise.")
    print("       Larger margins raise C a little (less truncation);")
    print("       finer h reduces staircase / singularity error.")


# =============================================================================
# 5. Custom geometry – symmetrical slit in the top plate
# =============================================================================
def demo_custom_split_plate(slit_width: float = 8e-3,
                            h: float = 0.1e-3,
                            show_ui: bool = False,
                            save_png: bool = True,
                            config: cfu.ParallelPlateConfig | None = None):
    """
    Build a *new* geometry that is not one of the built-in examples:

        bottom plate  – solid ground (0 V)
        top plate     – same outer dimensions but with a centred rectangular
                        slit removed by CSG difference (100 V)

    Uses the high-level ElectrostaticProblem façade so the solver pipeline
    stays completely untouched.

    Geometric defaults are taken from ParallelPlateConfig (or a user-supplied
    config) via make_parallel_plate_domain(), so the custom demo stays in
    sync with the rest of the project.
    """
    print("\n=== 5. Custom Geometry: Split Top-Plate Capacitor ===")
    print(f"  slit width = {slit_width*1e3:.1f} mm, "
          f"h = {h*1e3:.2f} mm")

    cfu.SHOW_PLOTS = show_ui
    cfu.SAVE_FIGURES = save_png

    # --- geometry from shared helper ----------------------------------------
    geo = make_parallel_plate_domain(config)
    plate_w = geo["plate_w"]
    plate_t = geo["plate_t"]
    gap = geo["gap"]
    margin = geo["margin"]
    Lx = geo["Lx"]
    Ly = geo["Ly"]
    x0 = geo["x0"]
    y0 = geo["y0"]

    # Cartesian mesh (uniform for simplicity)
    nx = int(round(Lx / h)) + 1
    ny = int(round(Ly / h)) + 1

    mesh = cfu.Mesh(
        x0=x0,
        y0=y0,
        Lx=Lx,
        Ly=Ly,
        nx=nx,
        ny=ny,
    )

    # Bottom plate – solid rectangle at 0 V
    bot = cfu.Rectangle(
        x0=-plate_w / 2.0,
        y0=-gap / 2.0 - plate_t,
        width=plate_w,
        height=plate_t,
        name="bottom_plate",
    )

    # Top plate – full rectangle minus a centred slit
    top_full = cfu.Rectangle(
        x0=-plate_w / 2.0,
        y0=gap / 2.0,
        width=plate_w,
        height=plate_t,
        name="top_full",
    )

    # Make the slit slightly taller than the plate so the CSG cut is clean
    slit = cfu.Rectangle(
        x0=-slit_width / 2.0,
        y0=gap / 2.0 - 0.5 * h,
        width=slit_width,
        height=plate_t + h,
        name="centre_slit",
    )

    top_split = cfu.Difference(
        top_full,
        slit,
        name="top_split_plate",
    )

    # --- high-level problem -------------------------------------------------
    problem = cfu.ElectrostaticProblem(
        mesh,
        background_eps_r=1.0,
    )

    problem.add_conductor(bot, voltage=0.0)
    problem.add_conductor(top_split, voltage=100.0)

    print("  Solving …")
    problem.solve()

    C = problem.capacitance(
        v_hi=100.0,
        v_lo=0.0,
    )

    print(f"  Capacitance = {C * 1e12:.4f} pF/m")

    # Optional plot (uses the same four-panel style as the built-in examples).
    out_name = os.path.join(
        cfu.OUTPUT_DIR,
        "custom_split_plate.png",
    )

    problem.plot(
        title=f"Split top-plate capacitor "
        f"(slit = {slit_width*1e3:.1f} mm)",
        fname=out_name,
        xlim=(x0 + margin * 0.3, x0 + Lx - margin * 0.3),
        ylim=(y0 + margin * 0.3, y0 + Ly - margin * 0.3),
    )


# =============================================================================
# 6. Default parallel-plate + small air bubble inside the glass slab
# =============================================================================
def demo_air_bubble_in_glass(bubble_radius: float = 0.6e-3,
                             bubble_center_x: float = 0.0,
                             bubble_center_y: float | None = None,
                             h: float = 0.1e-3,
                             show_ui: bool = False,
                             save_png: bool = True,
                             config: cfu.ParallelPlateConfig | None = None):
    """
    Exactly the default parallel-plate geometry (solid plates + partial
    glass slab in the lower half of the gap) with one extra feature:

        a small circular air bubble (ε_r = 1) punched out of the glass slab.

    The bubble is realised by CSG difference (or, equivalently, by adding
    an overriding dielectric region with ε_r = 1 after the glass slab).
    Everything else — plates, voltages, gap, slab thickness, background
    air — stays identical to ParallelPlateConfig defaults.

    Parameters
    ----------
    bubble_radius : float
        Radius of the air bubble [m].  Default 0.6 mm keeps it well inside
        a 2 mm thick slab while remaining larger than a few mesh cells.
    bubble_center_x : float
        Horizontal offset of the bubble centre relative to the plate
        mid-plane [m].  Default 0 (centred).
    bubble_center_y : float or None
        Vertical coordinate of the bubble centre [m].  When None the
        bubble is placed at the mid-height of the glass slab.
    h : float
        Mesh spacing [m].
    show_ui, save_png : bool
        Plotting / file-output toggles.
    config : ParallelPlateConfig or None
        Optional base configuration; defaults to the shipped ParallelPlateConfig.
    """
    print("\n=== 6. Default Parallel-Plate + Air Bubble in Glass Slab ===")
    print(f"  bubble radius = {bubble_radius*1e3:.2f} mm, h = {h*1e3:.2f} mm")

    cfu.SHOW_PLOTS = show_ui
    cfu.SAVE_FIGURES = save_png

    cfg = config or cfu.ParallelPlateConfig()
    geo = make_parallel_plate_domain(cfg)

    plate_w = geo["plate_w"]
    plate_t = geo["plate_t"]
    gap = geo["gap"]
    margin = geo["margin"]
    Lx = geo["Lx"]
    Ly = geo["Ly"]
    x0 = geo["x0"]
    y0 = geo["y0"]

    # Dielectric slab occupies the lower half of the gap (default 2 mm of 4 mm)
    dielectric_t = cfg.dielectric_thickness
    # y-coordinates of the gap (plates sit outside the gap)
    y_gap_lo = -gap / 2.0
    y_gap_hi = +gap / 2.0
    y_slab_lo = y_gap_lo
    y_slab_hi = y_gap_lo + dielectric_t

    # Default bubble placement: centre of the glass slab, horizontally centred
    if bubble_center_y is None:
        bubble_center_y = 0.5 * (y_slab_lo + y_slab_hi)

    # Sanity: bubble must fit inside the slab
    if (bubble_center_y - bubble_radius < y_slab_lo - 1e-9 or
            bubble_center_y + bubble_radius > y_slab_hi + 1e-9):
        raise ValueError(
            f"Bubble (r={bubble_radius*1e3:.2f} mm at y={bubble_center_y*1e3:.2f} mm) "
            f"does not fit inside the glass slab "
            f"[{y_slab_lo*1e3:.2f}, {y_slab_hi*1e3:.2f}] mm. "
            "Reduce radius or move the centre.")

    print(f"  glass slab y ∈ [{y_slab_lo*1e3:.2f}, {y_slab_hi*1e3:.2f}] mm")
    print(
        f"  bubble centre = ({bubble_center_x*1e3:.2f}, {bubble_center_y*1e3:.2f}) mm")

    # Cartesian mesh (uniform for simplicity; graded is also possible)
    nx = int(round(Lx / h)) + 1
    ny = int(round(Ly / h)) + 1

    mesh = cfu.Mesh(
        x0=x0,
        y0=y0,
        Lx=Lx,
        Ly=Ly,
        nx=nx,
        ny=ny,
    )

    # ---- conductors (identical to default parallel-plate) --------------------
    bot = cfu.Rectangle(
        x0=-plate_w / 2.0,
        y0=-gap / 2.0 - plate_t,
        width=plate_w,
        height=plate_t,
        name="bottom_plate",
    )
    top = cfu.Rectangle(
        x0=-plate_w / 2.0,
        y0=+gap / 2.0,
        width=plate_w,
        height=plate_t,
        name="top_plate",
    )

    # ---- dielectrics -------------------------------------------------------
    # 1. Full glass slab (ε_r = dielectric_eps_r)
    glass_slab = cfu.Rectangle(
        x0=-plate_w / 2.0,
        y0=y_slab_lo,
        width=plate_w,
        height=dielectric_t,
        eps_r=cfg.dielectric_eps_r,
        name="glass_slab",
    )

    # 2. Air bubble – a circle that overrides the glass with ε_r = 1
    #    (later regions win in make_eps_r_function)
    air_bubble = cfu.Circle(
        center=(bubble_center_x, bubble_center_y),
        radius=bubble_radius,
        eps_r=1.0,                     # air
        name="air_bubble",
    )

    # Optional: you can also write the glass as a Difference
    #     glass_with_hole = glass_slab - air_bubble
    # and then add only that one dielectric.  The override approach above
    # is simpler and produces identical ε_r maps.

    # ---- high-level problem ------------------------------------------------
    problem = cfu.ElectrostaticProblem(
        mesh,
        background_eps_r=cfg.background_eps_r,
    )

    problem.add_conductor(bot, voltage=0.0)
    problem.add_conductor(top, voltage=cfg.voltage)

    problem.add_dielectric(glass_slab)          # glass first
    problem.add_dielectric(air_bubble)          # air bubble overrides

    print("  Solving …")
    problem.solve()

    C = problem.capacitance(v_hi=cfg.voltage, v_lo=0.0)
    print(f"  Capacitance (with bubble) = {C * 1e12:.4f} pF/m")

    # Reference: same geometry without the bubble (quick analytical-style estimate
    # is not trivial because of fringing; we just report the FEM value).
    # For a pure comparison one could also call example_parallel_plate() and
    # look at the graded/uniform result.

    out_name = os.path.join(cfu.OUTPUT_DIR, "custom_air_bubble_in_glass.png")

    problem.plot(
        title=(f"Parallel-plate + air bubble in glass "
               f"(r = {bubble_radius*1e3:.2f} mm)"),
        fname=out_name,
        xlim=(x0 + margin * 0.3, x0 + Lx - margin * 0.3),
        ylim=(y0 + margin * 0.3, y0 + Ly - margin * 0.3),
    )

    print(f"  → figure: {out_name}")
    return problem, C


# =============================================================================
# Main – run the whole driver suite
# =============================================================================
if __name__ == "__main__":
    print("capacitor_fem_universal – driver suite")
    print("=" * 60)

    # 1. Fast demo of the built-in parallel-plate path
    demo_quick_parallel_plate(
        show_ui=False,
        save_png=True,
    )

    # 2. One-parameter override + shared colour scale
    demo_single_parameter_override()

    # 3. Completely different configs side-by-side
    demo_two_full_configs()

    # 4. Small two-axis study (expand the lists for real work)
    demo_two_axis_convergence_sweep()

    # 5. Custom geometry that is not part of the original examples
    demo_custom_split_plate(
        slit_width=8e-3,
        show_ui=False,
        save_png=True,
    )

    """
    # 5b. Custom geometry – full config from scratch (example)
    my_cfg = cfu.ParallelPlateConfig(
        plate_thickness=1.5e-3,
        gap=5e-3,
        bottom_plate_width=28e-3,
        top_plate_width=28e-3,
        domain_margin=25e-3,
        voltage=150.0,
        dielectric_eps_r=4.5,
        mesh_spacing=0.08e-3,
    )

    demo_custom_split_plate(
        slit_width=6e-3,
        h=0.08e-3,
        config=my_cfg,
        show_ui=False,
        save_png=True,
    )
    """

    # 6. Default parallel-plate geometry with a small air bubble in the glass
    demo_air_bubble_in_glass(
        bubble_radius=0.6e-3,          # 0.6 mm radius – comfortably inside 2 mm slab
        bubble_center_x=0.0,           # horizontally centred
        # bubble_center_y left at default → mid-height of the glass slab
        h=0.1e-3,
        show_ui=False,
        save_png=True,
    )

    print("\n" + "=" * 60)
    print("All driver demos finished.")
    print("Inspect the generated PNG files and the console tables above.")
    print("You can now copy any demo_* function into your own script")
    print("and adapt the parameters to your geometry.")
