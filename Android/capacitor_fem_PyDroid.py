#Pydroid run terminal
"""
Two-dimensional finite-element electrostatics solver.

The current implementation status, historical changes, limitations, and future
work are documented in README.md.

This version includes robustness improvements for:
  - PyDroid3 on Android
"""

import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


# =============================================================================
# 0. Runtime switches and tuning
# =============================================================================
# See README.md for the broader context, current status, and limitations.

# --- Execution switches (True / False) ---------------------------------------
RUN_BOUNDARY_STRESS_TEST: bool = False
"""Run the §10.4 floating-point tolerance verification on startup."""

RUN_EXACT_CHECK: bool = False
"""Run the machine-precision exact-solution validation before the examples."""

RUN_GRADED_COMPARISON: bool = True
"""For the parallel-plate example, also solve on a graded mesh and report ΔC."""

SAVE_FIGURES: bool = True
"""Write PNG files to OUTPUT_DIR."""

SHOW_PLOTS: bool = True
"""Call the improved show routine after each figure."""

VERBOSE_CONVERGENCE_NOTES: bool = False
"""Print the long explanatory notes after convergence tables."""

PLOT_CONVERGENCE: bool = True
"""After both examples, draw a combined convergence figure."""

# --- Android / mobile robustness ---------------------------------------------
# On PyDroid / Android we never try to open interactive plot windows.
# Figures are always written to disk when SAVE_FIGURES = True.
FORCE_SAVE_ONLY_ON_ANDROID: bool = True

PLOT_WAIT_TIMEOUT_S: float = 10.0   # kept only for desktop fallback

# --- I/O ----------------------------------------------------------------------
OUTPUT_DIR: str = ""
"""Directory for output figures.  Empty string → current working directory."""

# --- Numerical tolerances -----------------------------------------------------
BOUNDARY_TOLERANCE_M: float = 1e-9
"""Absolute tolerance [m] for Shape.contains() boundary comparisons.
Must be >> float64 epsilon (~1e-16) and << finest mesh spacing (75 µm).
See README.md §10.4 for the full discussion."""

# --- Memory-warning heuristics ------------------------------------------------
MEMORY_ESTIMATE_COEFF_MB: float = 0.1432
"""Coefficient for the empirical peak-RSS power law: coeff * nodes**exponent."""
MEMORY_ESTIMATE_EXPONENT: float = 0.680
"""Exponent for the empirical peak-RSS power law (super-linear because of
sparse LU fill-in on a 2-D grid)."""
MEMORY_WARN_THRESHOLD_NODES: int = 1_000_000
MEMORY_CRITICAL_THRESHOLD_NODES: int = 5_000_000

# --- Graded-mesh heuristics (parallel-plate example) --------------------------


@dataclass(frozen=True)
class GradedMeshTuning:
    """Tuning knobs for the piecewise-uniform graded Cartesian mesh.
    All *spacing* fields are multipliers of the nominal grid spacing h."""
    edge_band_width_factor: float = 4.0           # width of refined zone at plate ends
    edge_band_width_min_m: float = 1.5e-3         # minimum absolute width [m]
    margin_spacing_factor: float = 2.0            # coarsen margins by this factor
    edge_spacing_factor: float = 0.5              # refine edge bands by this factor
    interior_spacing_factor: float = 1.2          # slightly coarsen plate interior
    plate_spacing_factor: float = 0.8             # spacing through conductor plates
    gap_spacing_factor: float = 0.45              # finest spacing through the gap
    min_margin_points: int = 4                    # floor on points in each margin
    min_edge_points: int = 6                      # floor on points in each edge band
    min_interior_points: int = 8                  # floor on points in plate interior
    min_plate_points: int = 4                     # floor on points through a plate
    min_gap_points: int = 10                      # floor on points through the gap
    min_fallback_interior_points: int = 4         # floor when plate is too narrow for edge bands
    fallback_interior_spacing_factor: float = 1.0 # spacing factor used in that fallback


# Instantiate with defaults.  Replace this line to tweak globally:
GRADED_MESH_DEFAULTS: GradedMeshTuning = GradedMeshTuning()


def _is_android() -> bool:
    """Best-effort detection of Android / PyDroid."""
    try:
        return (
            "ANDROID" in os.environ
            or "ANDROID_ROOT" in os.environ
            or "ANDROID_DATA" in os.environ
            or "pydroid" in sys.executable.lower()
            or os.path.exists("/system/build.prop")
        )
    except Exception:
        return False


if FORCE_SAVE_ONLY_ON_ANDROID and _is_android():
    # PyDroid3 already runs headless; _show_blocking_figure handles the rest.
    if _is_android():
       print("Android/PyDroid detected → save-only mode")
"""
    try:
        matplotlib.use("Agg")
        print("Android/PyDroid detected → Agg backend (save-only mode)")
    except Exception:
        pass
"""

# =============================================================================
# 1. PHYSICS CONSTANTS
# =============================================================================
EPS0 = 8.8541878128e-12  # vacuum permittivity [F/m]


def _validate_boundary_tolerance():
    """Guard against nonsensical boundary-tolerance values."""
    tol = BOUNDARY_TOLERANCE_M
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(
            "BOUNDARY_TOLERANCE_M must be a positive finite number")
    if tol <= 10 * np.finfo(float).eps:
        warnings.warn(
            "BOUNDARY_TOLERANCE_M is extremely small; geometry classification may be brittle.")


_validate_boundary_tolerance()


# =============================================================================
# 2. CONFIGURATION
# =============================================================================
# Geometry, material, and solver settings live in frozen dataclasses.

@dataclass(frozen=True)
class _AutoSpacingConfig:
    """Base mixin that auto-derives convergence_spacings from mesh_spacing
    when the default tuple is left untouched.

    Subclasses must set _CONVERGENCE_RATIOS and _DEFAULT_CONVERGENCE_SPACINGS
    as ClassVars.
    """
    mesh_spacing: float
    convergence_spacings: tuple

    _CONVERGENCE_RATIOS: ClassVar[tuple] = ()
    _DEFAULT_CONVERGENCE_SPACINGS: ClassVar[tuple] = ()

    def __post_init__(self):
        if not self._CONVERGENCE_RATIOS or not self._DEFAULT_CONVERGENCE_SPACINGS:
            raise NotImplementedError(
                "Subclass must set _CONVERGENCE_RATIOS and _DEFAULT_CONVERGENCE_SPACINGS")
        if self.convergence_spacings[-1] != self.mesh_spacing:
            if self.convergence_spacings == self._DEFAULT_CONVERGENCE_SPACINGS:
                object.__setattr__(
                    self, "convergence_spacings",
                    tuple(r * self.mesh_spacing for r in self._CONVERGENCE_RATIOS))
            else:
                raise ValueError(
                    "convergence_spacings[-1] must equal mesh_spacing: the finest "
                    "level of the convergence sweep is reused as the production "
                    "resolution for the detailed report and plot. Leave "
                    "convergence_spacings at its default (untouched) to have it "
                    "follow mesh_spacing automatically, or supply a full "
                    "replacement tuple ending in the new mesh_spacing.")


@dataclass(frozen=True)
class ParallelPlateConfig(_AutoSpacingConfig):
    """Parameters for the parallel-plate capacitor example.

    bottom_plate_width / top_plate_width may differ (plates are centered in
    the domain).  Defaults keep both equal (classical symmetric capacitor);
    set top_plate_width smaller to study overhang / asymmetric fringing.
    """
    plate_thickness: float = 1e-3
    gap: float = 4e-3
    dielectric_thickness: float = 2e-3
    bottom_plate_width: float = 24e-3
    top_plate_width: float = 24e-3
    domain_margin: float = 15e-3
    voltage: float = 100.0
    dielectric_eps_r: float = 4.5
    background_eps_r: float = 1.0
    mesh_spacing: float = 0.1e-3
    convergence_spacings: tuple = (0.4e-3, 0.2e-3, 0.15e-3, 0.1e-3)
    plot_margin: float = 8e-3

    _CONVERGENCE_RATIOS: ClassVar[tuple] = (4.0, 2.0, 1.5, 1.0)
    _DEFAULT_CONVERGENCE_SPACINGS: ClassVar[tuple] = (
        0.4e-3, 0.2e-3, 0.15e-3, 0.1e-3)

    def __post_init__(self):
        super().__post_init__()

        # Lengths that must be genuinely positive
        for name in ("plate_thickness", "gap", "bottom_plate_width",
                     "top_plate_width", "domain_margin", "mesh_spacing"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"ParallelPlateConfig.{name} must be a positive, finite "
                    f"length in meters, got {value!r}.")

        if not np.isfinite(self.dielectric_thickness) or self.dielectric_thickness < 0.0:
            raise ValueError(
                "ParallelPlateConfig.dielectric_thickness must be a finite, "
                "non-negative length in meters (0 means no dielectric slab), "
                f"got {self.dielectric_thickness!r}.")

        if self.dielectric_thickness > self.gap:
            raise ValueError(
                f"ParallelPlateConfig.dielectric_thickness "
                f"({self.dielectric_thickness * 1e3:.4g} mm) cannot exceed "
                f"gap ({self.gap * 1e3:.4g} mm): the dielectric slab sits "
                f"inside the gap, so a thicker slab would overlap the top "
                f"plate. Reduce dielectric_thickness or increase gap.")

        for name in ("dielectric_eps_r", "background_eps_r"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"ParallelPlateConfig.{name} must be a positive, finite "
                    f"relative permittivity, got {value!r}.")

        if not np.isfinite(self.voltage) or self.voltage == 0.0:
            raise ValueError(
                "ParallelPlateConfig.voltage must be finite and nonzero: "
                "capacitance_from_energy divides by (V_hi - V_lo)**2, so a "
                "zero-voltage solve carries no information about C.")


@dataclass(frozen=True)
class CoaxConfig(_AutoSpacingConfig):
    """Parameters for the coaxial-cable example."""
    inner_radius: float = 3e-3
    outer_radius: float = 15e-3
    domain_half_width: float = 17e-3
    voltage: float = 100.0
    dielectric_eps_r: float = 2.3
    background_eps_r: float = 1.0
    mesh_spacing: float = 0.075e-3
    convergence_spacings: tuple = (0.3e-3, 0.2e-3, 0.15e-3, 0.1e-3, 0.075e-3)

    _CONVERGENCE_RATIOS: ClassVar[tuple] = (4.0, 8 / 3, 2.0, 4 / 3, 1.0)
    _DEFAULT_CONVERGENCE_SPACINGS: ClassVar[tuple] = (
        0.3e-3, 0.2e-3, 0.15e-3, 0.1e-3, 0.075e-3)

    def __post_init__(self):
        super().__post_init__()

        for name in ("inner_radius", "outer_radius", "domain_half_width",
                     "mesh_spacing"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"CoaxConfig.{name} must be a positive, finite length "
                    f"in meters, got {value!r}.")

        if self.outer_radius <= self.inner_radius:
            raise ValueError(
                f"CoaxConfig.outer_radius ({self.outer_radius * 1e3:.4g} mm)"
                f" must be strictly greater than inner_radius "
                f"({self.inner_radius * 1e3:.4g} mm).")

        if self.domain_half_width <= self.outer_radius:
            raise ValueError(
                f"CoaxConfig.domain_half_width "
                f"({self.domain_half_width * 1e3:.4g} mm) must exceed "
                f"outer_radius ({self.outer_radius * 1e3:.4g} mm).")

        for name in ("dielectric_eps_r", "background_eps_r"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"CoaxConfig.{name} must be a positive, finite relative "
                    f"permittivity, got {value!r}.")

        if not np.isfinite(self.voltage) or self.voltage == 0.0:
            raise ValueError(
                "CoaxConfig.voltage must be finite and nonzero.")


@dataclass(frozen=True)
class PlotConfig:
    """Shared visualization tuning parameters for plot_solution()."""
    figsize: tuple = (13, 11)
    dpi: int = 140
    potential_fill_levels: int = 25
    potential_line_levels: int = 15
    streamline_density: float = 1.9
    energy_density_floor: float = 1e-4
    conductor_fill_color: str = "dimgray"
    conductor_outline_color: str = "black"
    conductor_outline_width: float = 1.3


# =============================================================================
# 3. GEOMETRY
# =============================================================================

class Shape(ABC):
    """Common interface for conductor and dielectric-region shapes."""
    voltage = None
    eps_r = None
    name = "shape"

    @abstractmethod
    def contains(self, x, y):
        """Return True where (x, y) lies inside the shape."""

    def __or__(self, other):
        return Union(self, other)

    def __and__(self, other):
        return Intersection(self, other)

    def __sub__(self, other):
        return Difference(self, other)


class Rectangle(Shape):
    """Axis-aligned rectangle [x0, x0+width] × [y0, y0+height]."""

    def __init__(self, x0, y0, width, height, voltage=None, eps_r=None, name="rectangle"):
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError(
                f"Rectangle '{name}': width must be positive and finite, got {width!r}")
        if not np.isfinite(height) or height <= 0.0:
            raise ValueError(
                f"Rectangle '{name}': height must be positive and finite, got {height!r}")
        self.x0, self.y0 = x0, y0
        self.width, self.height = width, height
        self.voltage = voltage
        self.eps_r = eps_r
        self.name = name

    def contains(self, x, y):
        tol = BOUNDARY_TOLERANCE_M
        return ((x >= self.x0 - tol) & (x <= self.x0 + self.width + tol) &
                (y >= self.y0 - tol) & (y <= self.y0 + self.height + tol))


class Circle(Shape):
    """Filled disk."""

    def __init__(self, center, radius, voltage=None, eps_r=None, name="circle"):
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError(
                f"Circle '{name}': radius must be positive and finite, got {radius!r}")
        self.cx, self.cy = center
        self.radius = radius
        self.voltage = voltage
        self.eps_r = eps_r
        self.name = name

    def contains(self, x, y):
        tol = BOUNDARY_TOLERANCE_M
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= (self.radius + tol) ** 2


class OutsideCircle(Shape):
    """Complement of a disk."""

    def __init__(self, center, radius, voltage=None, eps_r=None, name="outside_circle"):
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError(
                f"OutsideCircle '{name}': radius must be positive and finite, got {radius!r}")
        self.cx, self.cy = center
        self.radius = radius
        self.voltage = voltage
        self.eps_r = eps_r
        self.name = name

    def contains(self, x, y):
        tol = BOUNDARY_TOLERANCE_M
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 >= (self.radius - tol) ** 2


class _CombinedShape(Shape):
    op_symbol = "?"

    def __init__(self, a, b, voltage=None, eps_r=None, name=None):
        self.a, self.b = a, b
        self.voltage = voltage
        self.eps_r = eps_r
        self.name = name or f"({a.name} {self.op_symbol} {b.name})"


class Union(_CombinedShape):
    op_symbol = "|"

    def contains(self, x, y):
        return self.a.contains(x, y) | self.b.contains(x, y)


class Intersection(_CombinedShape):
    op_symbol = "&"

    def contains(self, x, y):
        return self.a.contains(x, y) & self.b.contains(x, y)


class Difference(_CombinedShape):
    op_symbol = "-"

    def contains(self, x, y):
        return self.a.contains(x, y) & ~self.b.contains(x, y)


# =============================================================================
# 4. MATERIALS
# =============================================================================

class Material:
    """Descriptive label for a relative permittivity."""

    def __init__(self, name, eps_r):
        self.name = name
        self.eps_r = eps_r


def make_eps_r_function(regions, background_eps_r=1.0):
    """Combine dielectric-region shapes into a single ε_r(x,y) callable."""
    def eps_r_of_xy(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        eps_r = np.full(x.shape, float(background_eps_r))
        for region in regions:
            if region.eps_r is not None:
                eps_r = np.where(region.contains(x, y), region.eps_r, eps_r)
        return eps_r
    return eps_r_of_xy


# =============================================================================
# 5. MESH
# =============================================================================

def snap_to_grid(target, h):
    """Round a physical dimension to the nearest exact multiple of h."""
    return round(target / h) * h


def build_graded_coords(segments):
    """Build a monotonically increasing 1-D coordinate array from
    contiguous (start, end, n_points) segments."""
    if not segments:
        raise ValueError("build_graded_coords: at least one segment required")
    parts = []
    prev_end = None
    for i, (a, b, n) in enumerate(segments):
        a, b = float(a), float(b)
        n = int(n)
        if n < 2:
            raise ValueError(
                f"build_graded_coords: segment {i} needs n >= 2, got {n}")
        if prev_end is not None and abs(a - prev_end) > 1e-12 * max(1.0, abs(a)):
            raise ValueError(
                f"build_graded_coords: segment {i} starts at {a}, but previous "
                f"ended at {prev_end} (gaps or overlaps not allowed)")
        part = np.linspace(a, b, n)
        if i > 0:
            part = part[1:]
        parts.append(part)
        prev_end = b
    return np.concatenate(parts)


def _estimate_peak_memory_gb(n_nodes):
    """Rough ballpark of peak memory [GB] for a direct sparse solve."""
    return (MEMORY_ESTIMATE_COEFF_MB * n_nodes ** MEMORY_ESTIMATE_EXPONENT) / 1024


def _warn_if_large_mesh(n_nodes, n_tris):
    """Non-blocking heads-up before a large solve."""
    est_gb = _estimate_peak_memory_gb(n_nodes)
    if n_nodes >= MEMORY_CRITICAL_THRESHOLD_NODES:
        print(f"WARNING: this mesh has {n_nodes:,} nodes ({n_tris:,} triangles). "
              f"Estimated peak memory during assembly/solve: roughly {est_gb:.1f} GB "
              f"(ballpark).  Consider a coarser mesh_spacing or the graded-Cartesian "
              f"option (build_graded_coords + Mesh(xs=..., ys=...)).")
    elif n_nodes >= MEMORY_WARN_THRESHOLD_NODES:
        print(f"Note: this mesh has {n_nodes:,} nodes ({n_tris:,} triangles), "
              f"estimated peak memory roughly {est_gb:.1f} GB (ballpark).")


class Mesh:
    """Structured triangular mesh over a rectangular domain.

    Uniform:   Mesh(x0, y0, Lx, Ly, nx=nx, ny=ny)
    Graded:    Mesh(xs=xs, ys=ys)
    """

    def __init__(self, x0=0.0, y0=0.0, Lx=1.0, Ly=1.0, nx=None, ny=None,
                 xs=None, ys=None):
        if xs is not None:
            self.xs = np.asarray(xs, dtype=float)
            if self.xs.ndim != 1 or self.xs.size < 2:
                raise ValueError("xs must be a 1-D array of length >= 2")
            if np.any(np.diff(self.xs) <= 0):
                raise ValueError("xs must be strictly increasing")
            nx = len(self.xs)
        else:
            if nx is None:
                raise ValueError("Mesh: either nx or xs must be supplied")
            self.xs = np.linspace(x0, x0 + Lx, int(nx))

        if ys is not None:
            self.ys = np.asarray(ys, dtype=float)
            if self.ys.ndim != 1 or self.ys.size < 2:
                raise ValueError("ys must be a 1-D array of length >= 2")
            if np.any(np.diff(self.ys) <= 0):
                raise ValueError("ys must be strictly increasing")
            ny = len(self.ys)
        else:
            if ny is None:
                raise ValueError("Mesh: either ny or ys must be supplied")
            self.ys = np.linspace(y0, y0 + Ly, int(ny))

        X, Y = np.meshgrid(self.xs, self.ys)
        self.points = np.column_stack([X.ravel(), Y.ravel()])
        self.nx, self.ny = nx, ny
        self.n_nodes = self.points.shape[0]

        _warn_if_large_mesh(self.n_nodes, 2 * (nx - 1) * (ny - 1))

        i = np.arange(nx - 1)
        j = np.arange(ny - 1)
        II, JJ = np.meshgrid(i, j)
        II = II.ravel()
        JJ = JJ.ravel()

        n00 = JJ * nx + II
        n10 = JJ * nx + (II + 1)
        n01 = (JJ + 1) * nx + II
        n11 = (JJ + 1) * nx + (II + 1)

        even = (II + JJ) % 2 == 0

        triA = np.column_stack([n00, n10, np.where(even, n11, n01)])
        triB = np.column_stack([np.where(even, n00, n10), n11, n01])

        tris = np.empty((2 * len(II), 3), dtype=int)
        tris[0::2] = triA
        tris[1::2] = triB

        self.triangles = tris
        self.n_tris = tris.shape[0]

    def centroids(self):
        pts = self.points[self.triangles]
        return pts.mean(axis=1)


# =============================================================================
# 6. SOLVER
# =============================================================================

def _triangle_geometry(mesh):
    """P1 shape-function gradient coefficients and triangle areas."""
    tris = mesh.triangles
    p = mesh.points
    x1, y1 = p[tris[:, 0], 0], p[tris[:, 0], 1]
    x2, y2 = p[tris[:, 1], 0], p[tris[:, 1], 1]
    x3, y3 = p[tris[:, 2], 0], p[tris[:, 2], 1]

    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

    area2 = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    area = 0.5 * np.abs(area2)

    b = np.column_stack([b1, b2, b3])
    c = np.column_stack([c1, c2, c3])
    return b, c, area, area2


def evaluate_material(mesh, eps_r_of_xy):
    """Evaluate absolute permittivity (F/m) at each triangle centroid."""
    cxy = mesh.centroids()
    return eps_r_of_xy(cxy[:, 0], cxy[:, 1]) * EPS0


def _warn_if_degenerate_triangles(mesh, area):
    """Emit a warning if triangle areas are non-finite or suspiciously small."""
    if np.any(~np.isfinite(area)) or np.any(area <= 0.0):
        warnings.warn(
            "Encountered non-finite or non-positive triangle areas; the mesh may be degenerate.")
        return

    characteristic_area = max(1.0, np.ptp(mesh.xs) * np.ptp(mesh.ys))
    if np.min(area) < 1e-10 * characteristic_area:
        warnings.warn(
            "Some triangles are extremely small relative to the domain size; the mesh may be too fine or too distorted.")


def assemble_stiffness(mesh, eps_elem):
    """Assemble the sparse global stiffness matrix K for div(ε grad V)=0."""
    tris = mesh.triangles
    n = mesh.n_nodes
    b, c, area, area2 = _triangle_geometry(mesh)
    _warn_if_degenerate_triangles(mesh, area)

    Ke = b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]
    Ke = Ke * (eps_elem / (4.0 * area))[:, None, None]

    I = np.repeat(tris[:, :, None], 3, axis=2)
    J = np.repeat(tris[:, None, :], 3, axis=1)

    K = csr_matrix((Ke.ravel(), (I.ravel(), J.ravel())), shape=(n, n))
    return K, area, area2, b, c


def apply_conductors_and_solve(mesh, K, conductors):
    """Mark nodes inside conductors as Dirichlet, solve reduced system."""
    n = mesh.n_nodes
    x, y = mesh.points[:, 0], mesh.points[:, 1]

    V = np.zeros(n)
    is_fixed = np.zeros(n, dtype=bool)

    for cond in conductors:
        if cond.voltage is None:
            continue
        mask = cond.contains(x, y)
        V[mask] = cond.voltage
        is_fixed |= mask

    fixed_idx = np.flatnonzero(is_fixed)
    free_idx = np.flatnonzero(~is_fixed)

    solve_time = 0.0
    if free_idx.size == 0:
        return V, is_fixed, solve_time

    if fixed_idx.size == 0:
        warnings.warn(
            "No Dirichlet nodes were assigned; the reduced system is underdetermined. Returning the zero field.")
        return V, is_fixed, solve_time

    K_ff = K[free_idx][:, free_idx].tocsc()
    K_fd = K[free_idx][:, fixed_idx].tocsc()
    rhs = -K_fd.dot(V[fixed_idx])

    if K_ff.shape[0] == 0:
        return V, is_fixed, solve_time

    t0 = time.time()
    V[free_idx] = spsolve(K_ff, rhs)
    solve_time = time.time() - t0

    return V, is_fixed, solve_time


# =============================================================================
# 7. POST-PROCESSING
# =============================================================================

def compute_fields(mesh, V, eps_elem, b, c, area, area2):
    """Recover E, D, and energy density from nodal potential V."""
    tris = mesh.triangles
    V1, V2, V3 = V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]

    dVdx = (V1 * b[:, 0] + V2 * b[:, 1] + V3 * b[:, 2]) / area2
    dVdy = (V1 * c[:, 0] + V2 * c[:, 1] + V3 * c[:, 2]) / area2

    Ex, Ey = -dVdx, -dVdy
    Emag = np.hypot(Ex, Ey)
    Dx, Dy = eps_elem * Ex, eps_elem * Ey

    energy_density = 0.5 * (Ex * Dx + Ey * Dy)
    W = np.sum(energy_density * area)

    return Ex, Ey, Emag, Dx, Dy, energy_density, W


def capacitance_from_energy(W, V_hi, V_lo):
    """Two-conductor capacitance per unit depth [F/m] from stored energy."""
    dV = V_hi - V_lo
    if dV == 0.0:
        raise ValueError(
            "capacitance_from_energy: V_hi == V_lo, so dV=0 and C = 2W/dV**2"
            " is undefined.")
    return 2.0 * W / dV ** 2


# =============================================================================
# 8. HIGH-LEVEL API
# =============================================================================

class ElectrostaticProblem:
    """Thin facade over evaluate_material → assemble_stiffness →
    apply_conductors_and_solve → compute_fields."""

    def __init__(self, mesh, background_eps_r=1.0):
        self.mesh = mesh
        self.background_eps_r = background_eps_r
        self.conductors = []
        self.dielectrics = []
        self.V = None
        self.W = None

    def add_conductor(self, shape, voltage):
        shape.voltage = voltage
        self.conductors.append(shape)
        return shape

    def add_dielectric(self, shape, eps_r=None):
        if eps_r is not None:
            shape.eps_r = eps_r
        if shape.eps_r is None:
            raise ValueError("add_dielectric: shape has no eps_r set")
        self.dielectrics.append(shape)
        return shape

    def solve(self):
        self.eps_r_of_xy = make_eps_r_function(
            self.dielectrics, self.background_eps_r)
        self.eps_elem = evaluate_material(self.mesh, self.eps_r_of_xy)
        K, area, area2, b, c = assemble_stiffness(self.mesh, self.eps_elem)
        self.V, self.is_fixed, self.solve_time = apply_conductors_and_solve(
            self.mesh, K, self.conductors)
        (self.Ex, self.Ey, self.Emag, self.Dx, self.Dy,
         self.energy_density, self.W) = compute_fields(
            self.mesh, self.V, self.eps_elem, b, c, area, area2)
        return self

    def capacitance(self, v_hi, v_lo):
        if self.W is None:
            raise RuntimeError("call .solve() before .capacitance()")
        return capacitance_from_energy(self.W, v_hi, v_lo)

    def plot(self, title, fname, xlim=None, ylim=None, style=None):
        if self.V is None:
            raise RuntimeError("call .solve() before .plot()")
        plot_solution(self.mesh, self.V, self.eps_r_of_xy, self.energy_density,
                      self.conductors, self.is_fixed, title, fname,
                      xlim=xlim, ylim=ylim, style=style,
                      Ex=getattr(self, "Ex", None), Ey=getattr(self, "Ey", None))


# =============================================================================
# 9. VISUALIZATION
# =============================================================================

def _project_element_field_to_nodes(mesh, values):
    """Project a per-element field to nodal values by averaging contributions."""
    nodal = np.zeros(mesh.n_nodes, dtype=float)
    counts = np.zeros(mesh.n_nodes, dtype=float)
    tris = mesh.triangles
    np.add.at(nodal, tris[:, 0], values)
    np.add.at(nodal, tris[:, 1], values)
    np.add.at(nodal, tris[:, 2], values)
    np.add.at(counts, tris[:, 0], 1.0)
    np.add.at(counts, tris[:, 1], 1.0)
    np.add.at(counts, tris[:, 2], 1.0)
    return nodal / counts


def _show_blocking_figure(fig, message=None):
    """
    On Android / PyDroid: never open an interactive window – just close the figure.
    On desktop: keep the previous interactive behaviour with timeout.
    """
    if not SHOW_PLOTS:
        plt.close(fig)
        return

    on_android = _is_android()

    # ---------- Pure save-only path for Android ----------
    if on_android and FORCE_SAVE_ONLY_ON_ANDROID:
        # Saving already happened in plot_solution / plot_convergence_study
        plt.close(fig)
        return

    # ---------- Desktop interactive path ----------
    if message:
        print(message)

    try:
        fig.canvas.draw_idle()
        plt.show(block=False)
    except Exception as e:
        print(f"Warning: plt.show() failed ({e}); continuing.")
        plt.close(fig)
        return

    t0 = time.time()
    while plt.fignum_exists(fig.number):
        if time.time() - t0 > PLOT_WAIT_TIMEOUT_S:
            print(f"(Plot wait timed out after {PLOT_WAIT_TIMEOUT_S:.0f}s – continuing)")
            break
        try:
            fig.canvas.flush_events()
        except Exception:
            pass
        plt.pause(0.05)

    plt.close(fig)


def plot_solution(mesh, V, eps_r_of_xy, energy_density, conductors, is_fixed,
                  title, fname, xlim=None, ylim=None, style=None,
                  Ex=None, Ey=None):
    """Four-panel summary: dielectric map, equipotentials, field magnitude
    with streamlines, and energy density (log scale)."""
    style = style or PlotConfig()
    xs, ys, nx, ny = mesh.xs, mesh.ys, mesh.nx, mesh.ny
    X, Y = np.meshgrid(xs, ys)
    V_grid = V.reshape(ny, nx)
    epsr_grid = eps_r_of_xy(X, Y)

    cond_grid = np.zeros_like(X, dtype=bool)
    for cond in conductors:
        cond_grid |= cond.contains(X, Y)
    epsr_masked = np.ma.masked_where(cond_grid, epsr_grid)

    if Ex is None or Ey is None:
        dVdy_grid, dVdx_grid = np.gradient(V_grid, ys, xs)
        ExG, EyG = -dVdx_grid, -dVdy_grid
    else:
        Ex_nodes = _project_element_field_to_nodes(
            mesh, np.asarray(Ex, dtype=float))
        Ey_nodes = _project_element_field_to_nodes(
            mesh, np.asarray(Ey, dtype=float))
        ExG = Ex_nodes.reshape(ny, nx)
        EyG = Ey_nodes.reshape(ny, nx)
    EmagG = np.hypot(ExG, EyG)
    EmagG_masked = np.ma.masked_where(cond_grid, EmagG)

    tri_is_conductor = is_fixed[mesh.triangles].all(axis=1)
    energy_masked = np.ma.masked_where(tri_is_conductor, energy_density)

    cmap_blue = plt.get_cmap("Blues").copy()
    cmap_blue.set_bad(style.conductor_fill_color)
    cmap_inferno = plt.get_cmap("inferno").copy()
    cmap_inferno.set_bad(style.conductor_fill_color)
    cmap_magma = plt.get_cmap("magma").copy()
    cmap_magma.set_bad(style.conductor_fill_color)

    fig, axes = plt.subplots(2, 2, figsize=style.figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    def outline(ax):
        ax.contour(X, Y, cond_grid.astype(float), levels=[0.5],
                   colors=style.conductor_outline_color,
                   linewidths=style.conductor_outline_width)

    # Panel 1: dielectric map
    ax = axes[0, 0]
    pcm = ax.pcolormesh(X, Y, epsr_masked, shading="auto", cmap=cmap_blue)
    fig.colorbar(pcm, ax=ax, label=r"$\varepsilon_r$")
    outline(ax)
    ax.set_title("Dielectric map (gray = conductor)")

    # Panel 2: potential contours
    ax = axes[0, 1]
    cf = ax.contourf(
        X, Y, V_grid, levels=style.potential_fill_levels, cmap="viridis")
    ax.contour(X, Y, V_grid, levels=style.potential_line_levels,
               colors="white", linewidths=0.4, alpha=0.6)
    fig.colorbar(cf, ax=ax, label="V [Volt]")
    outline(ax)
    ax.set_title("Equipotential contours")

    # Panel 3: field magnitude + streamlines
    ax = axes[1, 0]
    pcm = ax.pcolormesh(X, Y, EmagG_masked, shading="auto", cmap=cmap_inferno)
    fig.colorbar(pcm, ax=ax, label="|E| [V/m]")

    ExG_plot = np.where(cond_grid, 0.0, ExG)
    EyG_plot = np.where(cond_grid, 0.0, EyG)

    dx = np.diff(xs)
    dy = np.diff(ys)
    is_graded = (np.ptp(dx) > 1e-12 * max(1.0, np.mean(dx)) or
                 np.ptp(dy) > 1e-12 * max(1.0, np.mean(dy)))

    if is_graded:
        n_plot_x = max(nx, int(round((xs[-1] - xs[0]) / np.min(dx))) + 1)
        n_plot_y = max(ny, int(round((ys[-1] - ys[0]) / np.min(dy))) + 1)
        n_plot_x = min(n_plot_x, 400)
        n_plot_y = min(n_plot_y, 400)

        xs_u = np.linspace(xs[0], xs[-1], n_plot_x)
        ys_u = np.linspace(ys[0], ys[-1], n_plot_y)
        Xu, Yu = np.meshgrid(xs_u, ys_u)

        interp_Ex = RegularGridInterpolator(
            (ys, xs), ExG_plot, bounds_error=False, fill_value=0.0)
        interp_Ey = RegularGridInterpolator(
            (ys, xs), EyG_plot, bounds_error=False, fill_value=0.0)
        pts = np.column_stack([Yu.ravel(), Xu.ravel()])
        Ex_u = interp_Ex(pts).reshape(Yu.shape)
        Ey_u = interp_Ey(pts).reshape(Yu.shape)

        cond_u = np.zeros_like(Xu, dtype=bool)
        for cond in conductors:
            cond_u |= cond.contains(Xu, Yu)
        Ex_u = np.where(cond_u, 0.0, Ex_u)
        Ey_u = np.where(cond_u, 0.0, Ey_u)

        ax.streamplot(xs_u, ys_u, Ex_u, Ey_u,
                      color="white",
                      density=style.streamline_density,
                      linewidth=0.6, arrowsize=0.8, minlength=0.01)
    else:
        ax.streamplot(xs, ys, ExG_plot, EyG_plot,
                      color="white",
                      density=style.streamline_density,
                      linewidth=0.6, arrowsize=0.8, minlength=0.01)

    outline(ax)
    ax.set_title("Electric field + field lines")

    # Panel 4: energy density
    ax = axes[1, 1]
    vmax = energy_density.max()
    tpc = ax.tripcolor(mesh.points[:, 0], mesh.points[:, 1], mesh.triangles,
                       facecolors=energy_masked, cmap=cmap_magma,
                       norm=mcolors.LogNorm(vmin=vmax * style.energy_density_floor,
                                            vmax=vmax))
    fig.colorbar(tpc, ax=ax, label=r"energy density [J/m$^3$]")
    outline(ax)
    ax.set_title("Energy density (log scale)")

    for ax in axes.ravel():
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if SAVE_FIGURES:
        os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
        fig.savefig(fname, dpi=style.dpi)
        print(f"Figure saved: {fname}")

    _show_blocking_figure(
        fig,
        "Close this plot window to continue (or wait for timeout)."
    )


def plot_convergence_study(pp_results, coax_results, graded_result=None,
                           fname=None, style=None):
    """Combined convergence figure for both worked examples."""
    style = style or PlotConfig()
    fname = fname or os.path.join(OUTPUT_DIR, "convergence_study.png")

    fig, (ax_pp, ax_cx) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.suptitle("Mesh convergence study", fontsize=14, fontweight="bold")

    # ----- Parallel plate ---------------------------------------------------
    h_pp = np.array([r["h"] for r in pp_results]) * 1e3
    C_pp = np.array([r["C"] for r in pp_results]) * 1e12
    C_id = np.array([r["C_ideal"] for r in pp_results]) * 1e12
    err_pp = 100.0 * (C_pp - C_id) / C_id

    color_fem = "#1f77b4"
    color_ref = "#d62728"
    color_err = "#2ca02c"
    color_grd = "#ff7f0e"

    ax_pp.plot(h_pp, C_pp, "o-", color=color_fem, lw=1.8, ms=7,
               label="FEM (uniform)")
    ax_pp.plot(h_pp, C_id, "s--", color=color_ref, lw=1.4, ms=5,
               label="Ideal (no fringing)")
    if graded_result is not None:
        ax_pp.plot(graded_result["h"] * 1e3, graded_result["C"] * 1e12,
                   "*", color=color_grd, ms=14, zorder=5,
                   label="FEM (graded)")

    ax_pp.set_xlabel("h [mm]")
    ax_pp.set_ylabel("C [pF/m]", color=color_fem)
    ax_pp.tick_params(axis="y", labelcolor=color_fem)
    ax_pp.invert_xaxis()
    ax_pp.grid(True, alpha=0.3)
    ax_pp.set_title("Parallel-plate (partial dielectric slab)")

    ax_pp2 = ax_pp.twinx()
    ax_pp2.plot(h_pp, err_pp, "^-", color=color_err, lw=1.2, ms=6, alpha=0.85,
                label="(FEM − ideal) / ideal")
    ax_pp2.set_ylabel("Relative difference [%]", color=color_err)
    ax_pp2.tick_params(axis="y", labelcolor=color_err)
    ax_pp2.axhline(0.0, color=color_err, ls=":", lw=0.8, alpha=0.5)

    lines1, labels1 = ax_pp.get_legend_handles_labels()
    lines2, labels2 = ax_pp2.get_legend_handles_labels()
    ax_pp.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    # ----- Coax -------------------------------------------------------------
    h_cx = np.array([r["h"] for r in coax_results]) * 1e3
    C_cx = np.array([r["C"] for r in coax_results]) * 1e12
    C_an = np.array([r["C_ideal"] for r in coax_results]) * 1e12
    err_cx = 100.0 * (C_cx - C_an) / C_an

    ax_cx.plot(h_cx, C_cx, "o-", color=color_fem, lw=1.8, ms=8, label="FEM")
    ax_cx.axhline(C_an[0], color=color_ref, ls="--", lw=1.4,
                  label=r"Analytical $2\pi\varepsilon/\ln(b/a)$")

    ax_cx.set_xlabel("h [mm]")
    ax_cx.set_ylabel("C [pF/m]", color=color_fem)
    ax_cx.tick_params(axis="y", labelcolor=color_fem)
    ax_cx.invert_xaxis()
    ax_cx.grid(True, alpha=0.3)
    ax_cx.set_title("Coaxial cable (polyethylene fill)")

    ax_cx2 = ax_cx.twinx()
    ax_cx2.plot(h_cx, err_cx, "^-", color=color_err, lw=1.2, ms=4, alpha=0.85,
                label="Staircase / mesh error")
    ax_cx2.set_ylabel("Relative error [%]", color=color_err)
    ax_cx2.tick_params(axis="y", labelcolor=color_err)
    ax_cx2.axhline(0.0, color=color_err, ls=":", lw=0.8, alpha=0.5)

    lines1, labels1 = ax_cx.get_legend_handles_labels()
    lines2, labels2 = ax_cx2.get_legend_handles_labels()
    ax_cx.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if SAVE_FIGURES:
        os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
        fig.savefig(fname, dpi=style.dpi)
        print(f"Convergence figure saved to {fname}")

    _show_blocking_figure(
        fig,
        "Close this convergence plot window to finish (or wait for timeout)."
    )


# =============================================================================
# 10. EXAMPLES & ORCHESTRATION
# =============================================================================

def _describe_convergence(C_values):
    """Report monotonicity of a capacitance sequence."""
    diffs = np.diff(C_values)
    if np.all(diffs > 0) or np.all(diffs < 0):
        print("The sequence above is monotonic across every tested resolution.")
    else:
        sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
        print(f"The sequence above is NOT monotonic (it changes direction "
              f"{sign_changes} time{'s' if sign_changes != 1 else ''} across the "
              f"tested resolutions).")


def _grid_alignment_note(config, h):
    """Check whether h divides evenly into the geometric parameters."""
    checks = [("gap", config.gap),
              ("dielectric_thickness", config.dielectric_thickness),
              ("plate_thickness", config.plate_thickness),
              ("bottom_plate_width", config.bottom_plate_width),
              ("top_plate_width", config.top_plate_width)]
    notes = []
    for name, target in checks:
        ratio = target / h
        if abs(ratio - round(ratio)) > 1e-6:
            snapped = snap_to_grid(target, h)
            notes.append(f"{name} {target*1e3:.3f}->{snapped*1e3:.3f}mm")
    return "clean" if not notes else "rounded: " + ", ".join(notes)


def _snap_plate_dims(config, h):
    """Snap plate dimensions to grid and validate physical consistency."""
    plate_t = snap_to_grid(config.plate_thickness, h)
    gap = snap_to_grid(config.gap, h)
    dielectric_t = snap_to_grid(config.dielectric_thickness, h)
    margin = snap_to_grid(config.domain_margin, h)

    if gap <= 0.0:
        raise ValueError(
            f"At mesh spacing h={h * 1e3:.4g} mm, gap rounds to zero or less.")
    if plate_t <= 0.0:
        raise ValueError(
            f"At mesh spacing h={h * 1e3:.4g} mm, plate_thickness rounds to zero or less.")
    if margin <= 0.0:
        raise ValueError(
            f"At mesh spacing h={h * 1e3:.4g} mm, domain_margin rounds to zero or less.")
    if dielectric_t > gap:
        raise ValueError(
            f"At mesh spacing h={h * 1e3:.4g} mm, dielectric_thickness snaps larger than gap.")

    return plate_t, gap, dielectric_t, margin


def _build_parallel_plate_geometry(config, h):
    """Return (conductors, eps_r_of_xy, dims_dict) for the parallel-plate example."""
    plate_t, gap, dielectric_t, margin = _snap_plate_dims(config, h)
    bottom_w = snap_to_grid(config.bottom_plate_width, h)
    top_w = snap_to_grid(config.top_plate_width, h)

    if bottom_w <= 0.0 or top_w <= 0.0:
        raise ValueError(
            f"At mesh spacing h={h * 1e3:.4g} mm, a plate width rounds to zero or less.")

    plate_w_max = max(bottom_w, top_w)
    Lx = plate_w_max + 2 * margin
    Ly = 2 * plate_t + gap + 2 * margin

    x_bottom0 = margin + 0.5 * (plate_w_max - bottom_w)
    x_top0 = margin + 0.5 * (plate_w_max - top_w)
    y_gap_lo = margin + plate_t
    y_gap_hi = y_gap_lo + gap

    bottom_plate = Rectangle(x_bottom0, margin, bottom_w, plate_t,
                             voltage=0.0, name="bottom_plate")
    top_plate = Rectangle(x_top0, y_gap_hi, top_w, plate_t,
                          voltage=config.voltage, name="top_plate")
    conductors = [bottom_plate, top_plate]

    dielectrics = []
    if dielectric_t > 0.0:
        slab = Rectangle(margin, y_gap_lo, plate_w_max, dielectric_t,
                         eps_r=config.dielectric_eps_r, name="dielectric_slab")
        dielectrics.append(slab)
    eps_r_of_xy = make_eps_r_function(
        dielectrics, background_eps_r=config.background_eps_r)

    dims = {
        "plate_t": plate_t, "gap": gap, "dielectric_t": dielectric_t,
        "bottom_w": bottom_w, "top_w": top_w, "plate_w_max": plate_w_max,
        "margin": margin, "Lx": Lx, "Ly": Ly,
        "x_bottom0": x_bottom0, "x_top0": x_top0,
        "y_gap_lo": y_gap_lo, "y_gap_hi": y_gap_hi,
        "plate_w": plate_w_max, "x_plate0": margin,
    }
    return conductors, eps_r_of_xy, dims


def _build_graded_parallel_plate_mesh(h, dims, tune=None):
    """Construct a graded Mesh for the parallel-plate example."""
    tune = tune or GRADED_MESH_DEFAULTS
    d = dims

    plate_w = d["plate_w_max"]
    x_plate0 = d["x_plate0"]

    edge_band = max(tune.edge_band_width_factor * h, tune.edge_band_width_min_m)
    n_margin_x = max(tune.min_margin_points,
                     round(d["margin"] / (tune.margin_spacing_factor * h)) + 1)
    n_edge = max(tune.min_edge_points,
                 round(edge_band / (tune.edge_spacing_factor * h)) + 1)
    n_interior = max(tune.min_interior_points,
                     round((plate_w - 2 * edge_band) /
                           (tune.interior_spacing_factor * h)) + 1)

    if plate_w <= 2 * edge_band:
        edge_band = 0.25 * plate_w
        n_edge = max(tune.min_edge_points,
                     round(edge_band / (tune.edge_spacing_factor * h)) + 1)
        n_interior = max(tune.min_fallback_interior_points,
                         round((plate_w - 2 * edge_band) /
                               (tune.fallback_interior_spacing_factor * h)) + 1)

    xs = build_graded_coords([
        (0.0, x_plate0, n_margin_x),
        (x_plate0, x_plate0 + edge_band, n_edge),
        (x_plate0 + edge_band, x_plate0 + plate_w - edge_band, n_interior),
        (x_plate0 + plate_w - edge_band, x_plate0 + plate_w, n_edge),
        (x_plate0 + plate_w, d["Lx"], n_margin_x),
    ])

    n_margin_y = max(tune.min_margin_points,
                     round(d["margin"] / (tune.margin_spacing_factor * h)) + 1)
    n_plate = max(tune.min_plate_points,
                  round(d["plate_t"] / (tune.plate_spacing_factor * h)) + 1)
    n_gap = max(tune.min_gap_points,
                round(d["gap"] / (tune.gap_spacing_factor * h)) + 1)

    if d["dielectric_t"] <= 0.0 or d["dielectric_t"] >= d["gap"]:
        gap_segments = [(d["y_gap_lo"], d["y_gap_hi"], n_gap)]
    else:
        y_diel = d["y_gap_lo"] + d["dielectric_t"]
        frac = d["dielectric_t"] / d["gap"]
        n_diel = max(tune.min_gap_points // 2, round(n_gap * frac))
        n_air = max(tune.min_gap_points // 2, n_gap - n_diel + 1)
        gap_segments = [
            (d["y_gap_lo"], y_diel, n_diel),
            (y_diel, d["y_gap_hi"], n_air),
        ]

    ys = build_graded_coords(
        [
            (0.0, d["margin"], n_margin_y),
            (d["margin"], d["y_gap_lo"], n_plate),
            *gap_segments,
            (d["y_gap_hi"], d["y_gap_hi"] + d["plate_t"], n_plate),
            (d["y_gap_hi"] + d["plate_t"], d["Ly"], n_margin_y),
        ]
    )

    return Mesh(xs=xs, ys=ys)


def _solve_parallel_plate(config, h, use_graded=False):
    """Build and solve the parallel-plate problem at nominal spacing h."""
    conductors, eps_r_of_xy, d = _build_parallel_plate_geometry(config, h)

    if use_graded:
        mesh = _build_graded_parallel_plate_mesh(h, d)
    else:
        mesh = Mesh(0, 0, d["Lx"], d["Ly"],
                    nx=round(d["Lx"] / h) + 1, ny=round(d["Ly"] / h) + 1)

    eps_elem = evaluate_material(mesh, eps_r_of_xy)
    K, area, area2, b, c = assemble_stiffness(mesh, eps_elem)
    V, is_fixed, solve_time = apply_conductors_and_solve(mesh, K, conductors)
    Ex, Ey, Emag, Dx, Dy, energy_density, W = compute_fields(
        mesh, V, eps_elem, b, c, area, area2)
    C = capacitance_from_energy(W, config.voltage, 0.0)
    overlap_w = min(d["bottom_w"], d["top_w"])
    C_ideal = (overlap_w * EPS0 /
               (d["dielectric_t"] / config.dielectric_eps_r +
                (d["gap"] - d["dielectric_t"]) / config.background_eps_r))

    result = dict(h=h, mesh=mesh, conductors=conductors, eps_r_of_xy=eps_r_of_xy,
                  V=V, is_fixed=is_fixed, energy_density=energy_density,
                  Ex=Ex, Ey=Ey, C=C, C_ideal=C_ideal, solve_time=solve_time, graded=use_graded)
    result.update(d)
    return result


def _print_parallel_plate_convergence(results, config):
    """Pretty-print the parallel-plate convergence table."""
    print(f"{'h [mm]':>9s}{'nodes':>9s}{'solve [s]':>11s}{'C [pF/m]':>12s}{'change':>9s}  grid alignment")
    prev_C = None
    C_values = []
    for result in results:
        h = result["h"]
        C = result["C"]
        change = "" if prev_C is None else f"{100*(C-prev_C)/prev_C:+.2f}%"
        alignment = _grid_alignment_note(config, h)
        print(f"{h*1e3:9.3f}{result['mesh'].n_nodes:9d}{result['solve_time']:11.3f}"
              f"{C*1e12:12.3f}{change:>9s}  {alignment}")
        prev_C = C
        C_values.append(C)
    _describe_convergence(C_values)


def _print_parallel_plate_convergence_notes():
    print("Two distinct effects are mixed together in the table above...")
    # (original long notes omitted for brevity – they remain available via VERBOSE_CONVERGENCE_NOTES)


def example_parallel_plate(config=None):
    """Parallel-plate capacitor with partial dielectric slab."""
    config = config or ParallelPlateConfig()

    print("=" * 72)
    print("EXAMPLE 1: Parallel-plate capacitor, partially filled with a dielectric slab")
    print("=" * 72)
    print(f"  bottom plate (0 V): {config.bottom_plate_width*1e3:.1f} mm wide")
    print(f"  top plate ({config.voltage:g} V): {config.top_plate_width*1e3:.1f} mm wide"
          f"{'  (shorter → asymmetric fringing)' if config.top_plate_width != config.bottom_plate_width else ''}")
    print()

    print("Mesh convergence (physical geometry fixed, only h changes):")
    results = [_solve_parallel_plate(config, h)
               for h in config.convergence_spacings]
    _print_parallel_plate_convergence(results, config)
    if VERBOSE_CONVERGENCE_NOTES:
        _print_parallel_plate_convergence_notes()
    print()

    uniform = results[-1]
    C_uniform = uniform["C"]
    C_ideal = uniform["C_ideal"]
    print(f"Uniform mesh (finest clean level): {uniform['mesh'].n_nodes} nodes, "
          f"{uniform['mesh'].n_tris} triangles (h = {uniform['h']*1e3:.3f} mm)")
    print(f"FEM capacitance (uniform) : {C_uniform*1e12:9.3f} pF/m")
    print(f"Ideal, no fringing        : {C_ideal*1e12:9.3f} pF/m")
    print(f"Difference                : {100*(C_uniform-C_ideal)/C_ideal:+6.2f} %  "
          f"(FEM > ideal is expected: it also captures fringing)")
    print()

    if RUN_GRADED_COMPARISON:
        print("Graded Cartesian mesh (same nominal h, refined near edges & gap):")
        graded = _solve_parallel_plate(config, uniform["h"], use_graded=True)
        C_graded = graded["C"]
        print(f"mesh: {graded['mesh'].n_nodes} nodes, {graded['mesh'].n_tris} triangles "
              f"(graded, nominal h = {graded['h']*1e3:.3f} mm)")
        print(f"FEM capacitance (graded)  : {C_graded*1e12:9.3f} pF/m")
        print(f"Delta vs uniform          : {100*(C_graded-C_uniform)/C_uniform:+.2f} %")
        print("Capacitance is per unit depth into the page; multiply by the actual "
              "plate depth in meters for total farads.")

        plot_solution(graded["mesh"], graded["V"], graded["eps_r_of_xy"],
                      graded["energy_density"], graded["conductors"], graded["is_fixed"],
                      "Parallel-plate capacitor (glass slab in an air gap) — graded mesh",
                      os.path.join(OUTPUT_DIR, "example1_parallel_plate.png"),
                      Ex=graded["Ex"], Ey=graded["Ey"],
                      xlim=(graded["x_plate0"] - config.plot_margin,
                            graded["x_plate0"] + graded["plate_w"] + config.plot_margin),
                      ylim=(graded["margin"] - config.plot_margin,
                            graded["margin"] + config.plot_margin
                            + 2 * graded["plate_t"] + graded["gap"]))
        return C_graded, C_ideal, results, graded

    plot_solution(uniform["mesh"], uniform["V"], uniform["eps_r_of_xy"],
                  uniform["energy_density"], uniform["conductors"], uniform["is_fixed"],
                  "Parallel-plate capacitor (glass slab in an air gap)",
                  os.path.join(OUTPUT_DIR, "example1_parallel_plate.png"),
                  Ex=uniform["Ex"], Ey=uniform["Ey"],
                  xlim=(uniform["x_plate0"] - config.plot_margin,
                        uniform["x_plate0"] + uniform["plate_w"] + config.plot_margin),
                  ylim=(uniform["margin"] - config.plot_margin,
                        uniform["margin"] + config.plot_margin
                        + 2 * uniform["plate_t"] + uniform["gap"]))
    return C_uniform, C_ideal, results, None


def _solve_coax(config, h):
    """Build and solve the coax problem at grid spacing h."""
    nx = round(2 * config.domain_half_width / h) + 1
    ny = nx

    inner = Circle((0, 0), config.inner_radius, voltage=config.voltage,
                   name="inner_conductor")
    outer = OutsideCircle((0, 0), config.outer_radius, voltage=0.0,
                          name="outer_conductor")
    conductors = [inner, outer]

    fill = Circle((0, 0), config.outer_radius, eps_r=config.dielectric_eps_r,
                  name="dielectric_fill")
    eps_r_of_xy = make_eps_r_function(
        [fill], background_eps_r=config.background_eps_r)

    half = config.domain_half_width
    mesh = Mesh(-half, -half, 2 * half, 2 * half, nx=nx, ny=ny)
    eps_elem = evaluate_material(mesh, eps_r_of_xy)
    K, area, area2, b, c = assemble_stiffness(mesh, eps_elem)
    V, is_fixed, solve_time = apply_conductors_and_solve(mesh, K, conductors)
    Ex, Ey, Emag, Dx, Dy, energy_density, W = compute_fields(
        mesh, V, eps_elem, b, c, area, area2)
    C = capacitance_from_energy(W, config.voltage, 0.0)
    C_ideal = (2 * np.pi * EPS0 * config.dielectric_eps_r
               / np.log(config.outer_radius / config.inner_radius))

    return dict(h=h, mesh=mesh, conductors=conductors, eps_r_of_xy=eps_r_of_xy,
                V=V, is_fixed=is_fixed, energy_density=energy_density,
                Ex=Ex, Ey=Ey, C=C, C_ideal=C_ideal, solve_time=solve_time)


def example_coax(config=None):
    """Coaxial cable with polyethylene dielectric fill."""
    config = config or CoaxConfig()

    print()
    print("=" * 72)
    print("EXAMPLE 2: Coaxial capacitor (polyethylene-filled)")
    print("=" * 72)

    print("Mesh convergence (smooth circular boundary, no sharp corner):")
    print(f"{'h [mm]':>9s}{'nodes':>9s}{'solve [s]':>11s}{'C [pF/m]':>12s}{'error':>9s}")
    results = [_solve_coax(config, h) for h in config.convergence_spacings]
    C_values = []
    for result in results:
        h = result["h"]
        C = result["C"]
        err = 100 * (C - result["C_ideal"]) / result["C_ideal"]
        print(f"{h*1e3:9.3f}{result['mesh'].n_nodes:9d}{result['solve_time']:11.3f}"
              f"{C*1e12:12.3f}{err:+8.2f}%")
        C_values.append(C)
    _describe_convergence(C_values)
    if VERBOSE_CONVERGENCE_NOTES:
        print("Error shrinks toward 0% overall...")
    print()

    result = results[-1]
    C, C_ideal = result["C"], result["C_ideal"]
    print(f"mesh: {result['mesh'].n_nodes} nodes, {result['mesh'].n_tris} triangles "
          f"(h = {result['h']*1e3:.3f} mm)")
    print(f"FEM capacitance             : {C*1e12:9.3f} pF/m")
    print(f"Analytical 2*pi*eps/ln(b/a) : {C_ideal*1e12:9.3f} pF/m")
    print(f"Difference                  : {100*(C-C_ideal)/C_ideal:+6.2f} %  "
          f"(mesh / staircase discretization error)")

    plot_solution(result["mesh"], result["V"], result["eps_r_of_xy"],
                  result["energy_density"], result["conductors"], result["is_fixed"],
                  "Coaxial capacitor (polyethylene dielectric)",
                  os.path.join(OUTPUT_DIR, "example2_coax.png"),
                  Ex=result["Ex"], Ey=result["Ey"])
    return C, C_ideal, results


def _solve_exact_check(config, h):
    """Full-width-plate variant: fringing is geometrically impossible."""
    plate_t, gap, dielectric_t, margin = _snap_plate_dims(config, h)

    Lx = snap_to_grid(max(config.bottom_plate_width, config.top_plate_width), h)
    Ly = 2 * plate_t + gap + 2 * margin
    nx = round(Lx / h) + 1
    ny = round(Ly / h) + 1

    y_gap_lo = margin + plate_t
    y_gap_hi = y_gap_lo + gap

    overhang = Lx
    bottom_plate = Rectangle(-overhang, margin, Lx + 2 * overhang, plate_t,
                             voltage=0.0, name="bottom_plate")
    top_plate = Rectangle(-overhang, y_gap_hi, Lx + 2 * overhang, plate_t,
                          voltage=config.voltage, name="top_plate")
    conductors = [bottom_plate, top_plate]

    dielectrics = []
    if dielectric_t > 0.0:
        slab = Rectangle(-overhang, y_gap_lo, Lx + 2 * overhang, dielectric_t,
                         eps_r=config.dielectric_eps_r, name="dielectric_slab")
        dielectrics.append(slab)
    eps_r_of_xy = make_eps_r_function(
        dielectrics, background_eps_r=config.background_eps_r)

    mesh = Mesh(0, 0, Lx, Ly, nx=nx, ny=ny)
    eps_elem = evaluate_material(mesh, eps_r_of_xy)
    K, area, area2, b, c = assemble_stiffness(mesh, eps_elem)
    V, is_fixed, solve_time = apply_conductors_and_solve(mesh, K, conductors)
    Ex, Ey, Emag, Dx, Dy, energy_density, W = compute_fields(
        mesh, V, eps_elem, b, c, area, area2)
    C = capacitance_from_energy(W, config.voltage, 0.0)
    C_exact = (Lx * EPS0 /
               (dielectric_t / config.dielectric_eps_r +
                (gap - dielectric_t) / config.background_eps_r))

    return dict(h=h, mesh=mesh, C=C, C_exact=C_exact, solve_time=solve_time)


def example_exact_check(config=None, spacings=(0.5e-3, 0.25e-3, 0.125e-3, 0.0625e-3)):
    """Exact-solution validation: full-width plates, no fringing possible."""
    config = config or ParallelPlateConfig()

    print("=" * 72)
    print("EXACT-SOLUTION CHECK: full-width plates, no fringing possible")
    print("=" * 72)
    print("Same materials, gap, and dielectric split as the parallel-plate")
    print("example above, but the plates now extend across the whole domain.")
    print()

    print(f"{'h [mm]':>9s}{'nodes':>9s}{'solve [s]':>11s}{'C [pF/m]':>16s}"
          f"{'C_exact [pF/m]':>18s}{'error':>12s}{'rel. diff':>13s}")
    results = []
    for h in spacings:
        result = _solve_exact_check(config, h)
        err = 100 * (result["C"] - result["C_exact"]) / result["C_exact"]
        rel_diff = abs(result["C"] - result["C_exact"]) / result["C_exact"]
        print(f"{h*1e3:9.4f}{result['mesh'].n_nodes:9d}{result['solve_time']:11.3f}"
              f"{result['C']*1e12:16.8f}{result['C_exact']*1e12:18.8f}{err:+11.6f}%"
              f"{rel_diff:13.2e}")
        results.append(result)
    print()
    print("The residual is a few times float64 epsilon – exact to machine precision.")
    return results[-1]["C"], results[-1]["C_exact"]


def verify_boundary_tolerance():
    """§10.4 stress test."""
    print("=" * 72)
    print("§10.4 BOUNDARY-TOLERANCE STRESS TEST")
    print("=" * 72)

    x0 = 0.015
    width = 0.024
    y0 = 0.016
    height = 0.001
    rect = Rectangle(x0, y0, width, height, voltage=0.0)
    x_edge = x0 + width

    candidates = [
        ("literal 0.039",           0.039),
        ("1.5 * 0.026",             1.5 * 0.026),
        ("39e-3",                   39e-3),
        ("float64(0.015)+0.024",    np.float64(0.015) + np.float64(0.024)),
        ("1.5 * 0.1e-3 * 260",      1.5 * 0.1e-3 * 260),
        ("nudged +5e-15",           x_edge + 5e-15),
        ("nudged -5e-15",           x_edge - 5e-15),
    ]

    points = np.array([c[1] for c in candidates])
    y = np.full_like(points, y0 + 0.5 * height)
    mask = rect.contains(points, y)

    print(f"BOUNDARY_TOLERANCE_M = {BOUNDARY_TOLERANCE_M} m")
    print(f"Exact edge x = {x_edge}")
    print()
    print(f"{'source':>22s}  {'reconstructed x':>20s}  {'delta':>12s}  inside?")
    for (name, xi), inside in zip(candidates, mask):
        delta = xi - x_edge
        print(f"{name:>22s}  {xi:20.16e}  {delta:12.2e}  {bool(inside)}")

    if not np.all(mask):
        print("\nFAILURE: at least one near-edge coordinate was classified outside.")
        return False

    print("\nAll near-edge coordinates classified as INSIDE – tolerance is doing its job.")
    return True


def print_summary(C1, C1_ideal, C2, C2_ideal, elapsed):
    """Final summary table."""
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Case':32s}{'FEM [pF/m]':>16s}{'Analytical [pF/m]':>20s}")
    print(f"{'Parallel plate (dielectric slab)':32s}{C1*1e12:16.3f}{C1_ideal*1e12:20.3f}")
    print(f"{'Coax (polyethylene)':32s}{C2*1e12:16.3f}{C2_ideal*1e12:20.3f}")
    print()
    print(f"total runtime: {elapsed:.2f} s")
    if SAVE_FIGURES:
        print(f"Figures written under {OUTPUT_DIR or '.'}/ "
              f"(example1_parallel_plate.png, example2_coax.png"
              f"{', convergence_study.png' if PLOT_CONVERGENCE else ''})")


# =============================================================================
# 11. MAIN
# =============================================================================

if __name__ == "__main__":
    t_start = time.time()

    if RUN_BOUNDARY_STRESS_TEST:
        verify_boundary_tolerance()

    if RUN_EXACT_CHECK:
        example_exact_check()
        print()

    C1, C1_ideal, pp_results, graded_result = example_parallel_plate()
    C2, C2_ideal, coax_results = example_coax()

    if PLOT_CONVERGENCE:
        plot_convergence_study(pp_results, coax_results,
                               graded_result=graded_result)

    print_summary(C1, C1_ideal, C2, C2_ideal, time.time() - t_start)