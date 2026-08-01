import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, repo_root)

import capacitor_fem as fem
import numpy as np


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
