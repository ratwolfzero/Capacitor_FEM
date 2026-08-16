"""
guidance.py — Practical driver & usage guide for capacitor_fem_universal.py

Treat capacitor_fem_universal.py as a *frozen* solver template.
This file shows how to drive it from the outside without modifying the solver.

What you will see here
----------------------
1. Two import styles (full module vs selective).
2. Runtime toggles (SHOW_PLOTS, SAVE_FIGURES) that control UI and file output.
3. Quick run of the built-in parallel-plate example (uniform / graded mesh).
4. Single-parameter override + shared-|E| comparison (compare_parallel_plate_runs).
5. Full dual-configuration comparison (different geometry + materials).
6. Two-axis convergence sweep: mesh_spacing (h) × domain_margin (m).
7. Custom geometry built with CSG + the high-level ElectrostaticProblem API
   (symmetrical slit in the top plate).

Run the whole suite:
    python guidance.py

Or call individual demo_* functions from a notebook / interactive session.
"""

# ---------------------------------------------------------------------------
# Standard-library helpers used by *this* driver script itself
# (not by the solver).  Needed regardless of which import style you pick
# below: the demos call replace(...) and os.path.join(...) by these
# top-level names.  The solver also imports both, so under Pattern 1 you
# could write cfu.replace / cfu.os instead, but the ordinary names are
# clearer.
# ---------------------------------------------------------------------------

from dataclasses import replace
import os

# ---------------------------------------------------------------------------
# Pattern 1 – full-module import
#   Useful when you need to change global switches or call internal helpers.
# ---------------------------------------------------------------------------
import capacitor_fem_universal as cfu

# ---------------------------------------------------------------------------
# Pattern 2 – selective imports
#   Cleaner when you only need the public configuration / geometry / API.
# ---------------------------------------------------------------------------
from capacitor_fem_universal import (
    ParallelPlateConfig,
    ElectrostaticProblem,
    Mesh,
    Rectangle,
    Difference,
    compare_parallel_plate_runs,
)


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

    # Run the official example (returns C_uniform, C_ideal, results, graded)
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

    base = ParallelPlateConfig()                     # all defaults
    wide = replace(base, gap=6e-3)                   # only gap changed

    print("  baseline gap = 4 mm")
    print("  modified gap = 6 mm")
    print("  Running compare_parallel_plate_runs …")

    compare_parallel_plate_runs(
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
    config_a = ParallelPlateConfig(
        bottom_plate_width=24e-3,
        top_plate_width=18e-3,
        edge_radius=0.0,
        dielectric_eps_r=2.2,          # PTFE-like
        mesh_spacing=0.1e-3,
    )

    # Config B – symmetric plates, rounded edges, high-k dielectric
    config_b = ParallelPlateConfig(
        bottom_plate_width=24e-3,
        top_plate_width=24e-3,
        edge_radius=0.4e-3,            # 0.4 mm fillet
        dielectric_eps_r=9.8,          # alumina-like
        mesh_spacing=0.1e-3,
    )

    print("  A: asymmetric + sharp + PTFE (εr=2.2)")
    print("  B: symmetric  + fillet + alumina (εr=9.8)")
    print("  Running compare_parallel_plate_runs …")

    compare_parallel_plate_runs(
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
            cfg = ParallelPlateConfig(
                mesh_spacing=h,
                domain_margin=m,
                # single-level “sweep” so we solve exactly once per (h,m)
                convergence_spacings=(h,),
            )
            # Call the internal helper directly (still public enough for studies)
            res = cfu._solve_parallel_plate(cfg, h, use_graded=True)

            print(f"{h*1e3:8.2f} | {m*1e3:11.1f} | {res['mesh'].n_nodes:8d} | "
                  f"{res['C']*1e12:10.4f} | {res['solve_time']:9.3f}")

    print("\n  Tip: the default production margin (15 mm) is a compromise.")
    print("       Larger margins raise C a little (less truncation);")
    print("       finer h reduces staircase / singularity error.")


# =============================================================================
# 5. Custom geometry – symmetrical slit in the top plate
# =============================================================================
def demo_custom_split_plate(slit_width: float = 8e-3,
                            h: float = 0.1e-3,
                            show_ui: bool = False,
                            save_png: bool = True):
    """
    Build a *new* geometry that is not one of the built-in examples:

        bottom plate  – solid ground (0 V)
        top plate     – same outer dimensions but with a centred rectangular
                        slit removed by CSG difference (100 V)

    Uses the high-level ElectrostaticProblem façade so the solver pipeline
    stays completely untouched.
    """
    print("\n=== 5. Custom Geometry: Split Top-Plate Capacitor ===")
    print(f"  slit width = {slit_width*1e3:.1f} mm,  h = {h*1e3:.2f} mm")

    cfu.SHOW_PLOTS = show_ui
    cfu.SAVE_FIGURES = save_png

    # --- geometry parameters -------------------------------------------------
    # plate_w / plate_t / gap /margin mirror ParallelPlateConfig's defaults; 
    plate_w = 24e-3
    plate_t = 1e-3
    gap     = 4e-3
    margin  = 15e-3

    Lx = plate_w + 2 * margin
    Ly = 2 * plate_t + gap + 2 * margin
    x0 = -Lx / 2.0
    y0 = -Ly / 2.0

    # Cartesian mesh (uniform for simplicity)
    nx = int(round(Lx / h)) + 1
    ny = int(round(Ly / h)) + 1
    mesh = Mesh(x0=x0, y0=y0, Lx=Lx, Ly=Ly, nx=nx, ny=ny)

    # Bottom plate – solid rectangle at 0 V
    bot = Rectangle(
        x0=-plate_w / 2.0,
        y0=-gap / 2.0 - plate_t,
        width=plate_w,
        height=plate_t,
        name="bottom_plate",
    )

    # Top plate – full rectangle minus a centred slit
    top_full = Rectangle(
        x0=-plate_w / 2.0,
        y0=gap / 2.0,
        width=plate_w,
        height=plate_t,
        name="top_full",
    )
    # Make the slit slightly taller than the plate so the CSG cut is clean
    slit = Rectangle(
        x0=-slit_width / 2.0,
        y0=gap / 2.0 - 0.5 * h,
        width=slit_width,
        height=plate_t + h,
        name="centre_slit",
    )
    top_split = Difference(top_full, slit, name="top_split_plate")

    # --- high-level problem -------------------------------------------------
    problem = ElectrostaticProblem(mesh, background_eps_r=1.0)
    problem.add_conductor(bot, voltage=0.0)
    problem.add_conductor(top_split, voltage=100.0)

    print("  Solving …")
    problem.solve()

    C = problem.capacitance(v_hi=100.0, v_lo=0.0)
    print(f"  Capacitance = {C * 1e12:.4f} pF/m")

    # Optional plot (uses the same four-panel style as the built-in examples).
    # Routed through OUTPUT_DIR, same as every other figure in the module —
    # problem.plot() already prints its own "Figure saved" line when
    # SAVE_FIGURES is True, so there's nothing left to print here.
    out_name = os.path.join(cfu.OUTPUT_DIR, "custom_split_plate.png")
    problem.plot(
        title=f"Split top-plate capacitor (slit = {slit_width*1e3:.1f} mm)",
        fname=out_name,
        xlim=(x0 + margin * 0.3, x0 + Lx - margin * 0.3),
        ylim=(y0 + margin * 0.3, y0 + Ly - margin * 0.3),
    )


# =============================================================================
# Main – run the whole guidance suite
# =============================================================================
if __name__ == "__main__":
    print("capacitor_fem_universal – guidance suite")
    print("=" * 60)

    # 1. Fast demo of the built-in parallel-plate path
    demo_quick_parallel_plate(show_ui=False, save_png=True)

    # 2. One-parameter override + shared colour scale
    demo_single_parameter_override()

    # 3. Completely different configs side-by-side
    demo_two_full_configs()

    # 4. Small two-axis study (expand the lists for real work)
    demo_two_axis_convergence_sweep()

    # 5. Custom geometry that is not part of the original examples
    demo_custom_split_plate(slit_width=8e-3, show_ui=False, save_png=True)

    print("\n" + "=" * 60)
    print("All guidance demos finished.")
    print("Inspect the generated PNG files and the console tables above.")
    print("You can now copy any demo_* function into your own script")
    print("and adapt the parameters to your geometry.")
