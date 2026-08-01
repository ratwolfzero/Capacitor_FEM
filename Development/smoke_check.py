import os
import sys

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: install numpy, scipy, and matplotlib in the original "
        "Python environment before running this smoke check."
    ) from exc

sys.path.insert(0, os.path.dirname(__file__))
try:
    import capacitor_fem as fem
except ImportError as exc:
    raise SystemExit(
        "Could not import the FEM module. Run this script from the repository root "
        "with the same interpreter that has the scientific packages installed."
    ) from exc

# Run this with the original system Python interpreter, not a project-local
# virtual environment.


def main():
    fem.RUN_GRADED_COMPARISON = False
    fem.PLOT_CONVERGENCE = False
    fem.SAVE_FIGURES = False
    fem.SHOW_PLOTS = False

    cfg = fem.ParallelPlateConfig(mesh_spacing=0.2e-3, convergence_spacings=(0.2e-3,))
    result = fem._solve_parallel_plate(cfg, cfg.mesh_spacing)

    assert np.isfinite(result["C"]), result
    assert result["C"] > 0.0, result
    assert result["mesh"].n_nodes > 0, result
    assert result["mesh"].n_tris > 0, result

    print("smoke ok")
    print(f"C = {result['C'] * 1e12:.6f} pF/m")


if __name__ == "__main__":
    main()
