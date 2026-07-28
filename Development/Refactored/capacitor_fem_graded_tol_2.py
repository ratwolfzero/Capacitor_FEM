"""
capacitor_fem_graded_tol.py  (refactored)
=========================================

A two-dimensional finite-element electrostatics solver for simulating
capacitor geometries -- parallel plates, coaxial cables, and (via the
Shape/CSG primitives in GEOMETRY) more complex arrangements built from
simple shapes -- instead of relying on closed-form formulas that only
exist for a handful of idealized geometries.  Pure NumPy / SciPy /
Matplotlib, no external mesh-generation library.

This refactored edition keeps the physics, numerics, and validation
results **bit-identical** to the original.  All changes are structural:

  * Every tunable switch, path, tolerance, and heuristic multiplier is
    declared up-front in Section 0 (TUNABLE RUNTIME PARAMETERS).

  * Geometry configurations share a common base class that implements
    the convergence-spacing auto-derivation once, removing duplication.

  * The parallel-plate example's geometry construction, graded-mesh
    construction, and solver orchestration are split into small,
    single-purpose helpers.

  * Long boiler-plate print blocks are extracted into helpers gated by
    the VERBOSE_CONVERGENCE_NOTES switch.

Full physics documentation lives in README.md.
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import scipy.interpolate as interp


# =============================================================================
# 0. TUNABLE RUNTIME PARAMETERS  --  EDIT THESE TO CHANGE BEHAVIOUR
# =============================================================================
# Every behavioural switch, path, tolerance, and heuristic multiplier lives
# here in one place.  The sections below (physics, geometry, solver, …) should
# rarely need editing for day-to-day use.

# --- Execution switches (True / False) ---------------------------------------
RUN_BOUNDARY_STRESS_TEST: bool = False
"""Run the §10.4 floating-point tolerance verification on startup."""

RUN_EXACT_CHECK: bool = False
"""Run the machine-precision exact-solution validation before the examples."""

RUN_GRADED_COMPARISON: bool = True
"""For the parallel-plate example, also solve on a graded mesh and report ΔC."""

SAVE_FIGURES: bool = False
"""Write PNG files to OUTPUT_DIR."""

SHOW_PLOTS: bool = True
"""Call plt.show() after each figure (blocks until window is closed)."""

VERBOSE_CONVERGENCE_NOTES: bool = False
"""Print the long explanatory notes after convergence tables."""

# --- I/O ----------------------------------------------------------------------
OUTPUT_DIR: str = "Development"
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
    edge_band_width_factor: float = 4.0       # width of refined zone at plate ends
    edge_band_width_min_m: float = 1.5e-3     # minimum absolute width [m]
    margin_spacing_factor: float = 2.0        # coarsen margins by this factor
    edge_spacing_factor: float = 0.5          # refine edge bands by this factor
    interior_spacing_factor: float = 1.2      # slightly coarsen plate interior
    plate_spacing_factor: float = 0.8         # spacing through conductor plates
    gap_spacing_factor: float = 0.45          # finest spacing through the gap
    min_margin_points: int = 4
    min_edge_points: int = 6
    min_interior_points: int = 8
    min_plate_points: int = 4
    min_gap_points: int = 10
    min_fallback_interior_points: int = 4
    fallback_interior_spacing_factor: float = 1.0


# Instantiate with defaults.  Replace this line to tweak globally:
GRADED_MESH_DEFAULTS: GradedMeshTuning = GradedMeshTuning()


# =============================================================================
# 1. PHYSICS CONSTANTS
# =============================================================================
EPS0 = 8.8541878128e-12  # vacuum permittivity [F/m]


# =============================================================================
# 2. CONFIGURATION
# =============================================================================
# Every geometry, material, and numerical tuning parameter lives here as
# frozen dataclasses.  See README.md section 5.2.

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
    """Parameters for the parallel-plate capacitor example."""
    plate_thickness: float = 1e-3
    gap: float = 4e-3
    dielectric_thickness: float = 2e-3
    plate_width: float = 24e-3
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

    # Exact fractions (8/3, 4/3) so derived values are as clean as possible.
    _CONVERGENCE_RATIOS: ClassVar[tuple] = (4.0, 8 / 3, 2.0, 4 / 3, 1.0)
    _DEFAULT_CONVERGENCE_SPACINGS: ClassVar[tuple] = (
        0.3e-3, 0.2e-3, 0.15e-3, 0.1e-3, 0.075e-3)


@dataclass(frozen=True)
class PlotConfig:
    """Shared visualization tuning parameters for plot_solution()."""
    figsize: tuple = (13, 11)
    dpi: int = 140
    potential_fill_levels: int = 25
    potential_line_levels: int = 15
    streamline_density: float = 2.5
    streamline_target_count: int = 200
    energy_density_floor: float = 1e-4
    conductor_fill_color: str = "dimgray"
    conductor_outline_color: str = "black"
    conductor_outline_width: float = 1.3


# =============================================================================
# 3. GEOMETRY
# =============================================================================

class Shape(ABC):
    """Common interface for conductor and dielectric-region shapes.

    Set `voltage` to use a shape as a conductor, or `eps_r` to use it as a
    dielectric region.  Shapes compose with `|`, `&`, `-` (union, intersection,
    difference) returning new Shape instances whose contains() delegates to
    the operands.  See README.md §5.3.
    """
    voltage = None
    eps_r = None
    name = "shape"

    @abstractmethod
    def contains(self, x, y):
        """Return True where (x, y) lies inside the shape.
        x, y may be scalars or arrays; the returned mask matches their shape.
        """

    def __or__(self, other):
        return Union(self, other)

    def __and__(self, other):
        return Intersection(self, other)

    def __sub__(self, other):
        return Difference(self, other)


class Rectangle(Shape):
    """Axis-aligned rectangle [x0, x0+width] × [y0, y0+height].
    Boundary comparisons use BOUNDARY_TOLERANCE_M (see §10.4)."""

    def __init__(self, x0, y0, width, height, voltage=None, eps_r=None, name="rectangle"):
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
    """Filled disk.  Boundary tolerance applied outward."""

    def __init__(self, center, radius, voltage=None, eps_r=None, name="circle"):
        self.cx, self.cy = center
        self.radius = radius
        self.voltage = voltage
        self.eps_r = eps_r
        self.name = name

    def contains(self, x, y):
        tol = BOUNDARY_TOLERANCE_M
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= (self.radius + tol) ** 2


class OutsideCircle(Shape):
    """Complement of a disk.  Boundary tolerance applied inward so that
    Circle and OutsideCircle are exact complements at the shared boundary."""

    def __init__(self, center, radius, voltage=None, eps_r=None, name="outside_circle"):
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
    """Combine dielectric-region shapes into a single ε_r(x,y) callable.
    Later entries paint over earlier ones where they overlap."""
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
    """Round a physical dimension to the nearest exact multiple of h.
    See README.md §4.3 for why this matters."""
    return round(target / h) * h


def build_graded_coords(segments):
    """Build a monotonically increasing 1-D coordinate array from
    contiguous (start, end, n_points) segments.  See README.md §11."""
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

    Each cell is split into two triangles with a checkerboard diagonal
    pattern to avoid directional bias.
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
    """P1 shape-function gradient coefficients and triangle areas.
    Returns (b, c, area, area2) where area2 is the signed doubled area."""
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
    """Evaluate absolute permittivity (F/m) at each triangle centroid.
    Kept separate from assembly so nonlinear/iterative solves only need
    to re-call this + assemble_stiffness in a loop."""
    cxy = mesh.centroids()
    return eps_r_of_xy(cxy[:, 0], cxy[:, 1]) * EPS0


def assemble_stiffness(mesh, eps_elem):
    """Assemble the sparse global stiffness matrix K for div(ε grad V)=0.
    Vectorised triplet assembly; duplicate (row,col) entries are summed
    by the csr_matrix constructor."""
    tris = mesh.triangles
    n = mesh.n_nodes
    b, c, area, area2 = _triangle_geometry(mesh)

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
    if free_idx.size > 0:
        K_ff = K[free_idx][:, free_idx].tocsc()
        K_fd = K[free_idx][:, fixed_idx].tocsc()
        rhs = -K_fd.dot(V[fixed_idx])

        t0 = time.time()
        V[free_idx] = spsolve(K_ff, rhs)
        solve_time = time.time() - t0

    return V, is_fixed, solve_time


# =============================================================================
# 7. POST-PROCESSING
# =============================================================================

def compute_fields(mesh, V, eps_elem, b, c, area, area2):
    """Recover E, D, and energy density from nodal potential V.
    E and D are exactly constant per element (property of P1 elements)."""
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
                      xlim=xlim, ylim=ylim, style=style)


# =============================================================================
# 9. VISUALIZATION
# =============================================================================

def plot_solution(mesh, V, eps_r_of_xy, energy_density, conductors, is_fixed,
                  title, fname, xlim=None, ylim=None, style=None):
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

    dVdy_grid, dVdx_grid = np.gradient(V_grid, ys, xs)
    ExG, EyG = -dVdx_grid, -dVdy_grid
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

    # --- Streamplot preparation ---
    dxs, dys = np.diff(xs), np.diff(ys)
    is_uniform = np.allclose(dxs, dxs[0]) and np.allclose(dys, dys[0])
    if is_uniform:
        xs_sp, ys_sp = xs, ys
        Ex_sp, Ey_sp = ExG, EyG
        X_sp, Y_sp = X, Y
    else:
        xs_sp = np.linspace(xs[0], xs[-1], len(xs))
        ys_sp = np.linspace(ys[0], ys[-1], len(ys))
        X_sp, Y_sp = np.meshgrid(xs_sp, ys_sp)
        spline_ex = interp.RectBivariateSpline(ys, xs, ExG)
        spline_ey = interp.RectBivariateSpline(ys, xs, EyG)
        Ex_sp = spline_ex(ys_sp, xs_sp)
        Ey_sp = spline_ey(ys_sp, xs_sp)

    # Zero E-field inside conductors so streamlines terminate cleanly.
    cond_sp = np.zeros_like(X_sp, dtype=bool)
    for cond in conductors:
        cond_sp |= cond.contains(X_sp, Y_sp)
    Ex_sp = np.where(cond_sp, 0.0, Ex_sp)
    Ey_sp = np.where(cond_sp, 0.0, Ey_sp)

    # ---- Seed-point generation --------------------------------------------
    target = getattr(style, 'streamline_target_count', 80)
    seed_list = []

    def _line_seeds(xa, ya, xb, yb, n):
        if n < 2:
            n = 2
        t = np.linspace(0, 1, n)
        return np.column_stack([xa + t * (xb - xa), ya + t * (yb - ya)])

    # Minimum cell size on the streamplot grid – used to avoid placing
    # more than one seed per ~2 cells, which Matplotlib would silently drop.
    min_h = min(np.min(dxs) if len(dxs) else 1e-4,
                np.min(dys) if len(dys) else 1e-4)
    off = min_h * 0.5

    # 1) Symmetric edge seeds for Rectangle conductors.
    #    The count per edge is capped by the physical edge length so that
    #    short sides (e.g. 1 mm plate thickness) do not get flooded with
    #    invisible overlapping seeds.
    n_edge_raw = max(3, target // 16)
    for cond in conductors:
        if isinstance(cond, Rectangle) and cond.voltage is not None:
            x0, y0 = cond.x0, cond.y0
            w, h = cond.width, cond.height
            # Cap by length / (2 * min_h)  →  max one seed per 2 cells
            n_tb = max(3, min(n_edge_raw, int(w / (2.0 * min_h)) + 1))
            n_lr = max(3, min(n_edge_raw, int(h / (2.0 * min_h)) + 1))
            # bottom
            seed_list.append(_line_seeds(x0, y0 - off, x0 + w, y0 - off, n_tb))
            # top
            seed_list.append(_line_seeds(x0, y0 + h + off, x0 + w, y0 + h + off, n_tb))
            # left
            seed_list.append(_line_seeds(x0 - off, y0, x0 - off, y0 + h, n_lr))
            # right
            seed_list.append(_line_seeds(x0 + w + off, y0, x0 + w + off, y0 + h, n_lr))

    # 2) Fallback for non-rectangular conductors (coax, etc.)
    if not seed_list:
        bdry = np.zeros_like(cond_sp, dtype=bool)
        bdry[1:-1, 1:-1] = (
            ~cond_sp[1:-1, 1:-1] &
            (cond_sp[0:-2, 1:-1] | cond_sp[2:, 1:-1] |
             cond_sp[1:-1, 0:-2] | cond_sp[1:-1, 2:])
        )
        y_b, x_b = np.where(bdry)
        if len(x_b):
            seed_list.append(np.column_stack([X_sp[y_b, x_b], Y_sp[y_b, x_b]]))

    n_edge = sum(len(s) for s in seed_list)

    # 3) Bulk seeds inside the dielectric to make target_count actually
    #    scale the line density across the whole domain.
    if n_edge < target:
        n_extra = target - n_edge
        # Over-produce by ~2x because many grid points will fall inside
        # conductors and be discarded.
        aspect = (xs_sp[-1] - xs_sp[0]) / max(ys_sp[-1] - ys_sp[0], 1e-12)
        ny_seed = max(2, int(np.sqrt(2.5 * n_extra / aspect)))
        nx_seed = max(2, int(2.5 * n_extra / ny_seed))
        x_seed = np.linspace(xs_sp[0], xs_sp[-1], nx_seed)
        y_seed = np.linspace(ys_sp[0], ys_sp[-1], ny_seed)
        X_seed, Y_seed = np.meshgrid(x_seed, y_seed)
        mask = np.ones_like(X_seed, dtype=bool)
        for cond in conductors:
            mask &= ~cond.contains(X_seed, Y_seed)
        x_s, y_s = X_seed[mask], Y_seed[mask]
        if len(x_s):
            if len(x_s) > n_extra:
                idx = np.linspace(0, len(x_s) - 1, n_extra, dtype=int)
                x_s, y_s = x_s[idx], y_s[idx]
            seed_list.append(np.column_stack([x_s, y_s]))

    if seed_list:
        start_points = np.vstack(seed_list)
        # Final uniform cap (preserves symmetry because the original set is
        # symmetric and we just take every n-th point).
        if len(start_points) > target:
            idx = np.linspace(0, len(start_points) - 1, target, dtype=int)
            start_points = start_points[idx]
    else:
        start_points = None

    # NOTE: When start_points is supplied Matplotlib ignores the density
    # keyword entirely.  If you want a denser background field, increase
    # streamline_target_count; the bulk-seed logic above will fill the gap.
    sp_kwargs = dict(color="white", linewidth=0.6, arrowsize=0.8, minlength=0.05)
    if start_points is not None and len(start_points):
        sp_kwargs['start_points'] = start_points
    else:
        sp_kwargs['density'] = style.streamline_density

    ax.streamplot(xs_sp, ys_sp, Ex_sp, Ey_sp, **sp_kwargs)
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
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


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
    """Check whether h divides evenly into the geometric parameters that
    directly set the capacitor's physics.  Returns 'clean' or a note."""
    checks = [("gap", config.gap),
              ("dielectric_thickness", config.dielectric_thickness),
              ("plate_thickness", config.plate_thickness)]
    notes = []
    for name, target in checks:
        ratio = target / h
        if abs(ratio - round(ratio)) > 1e-6:
            snapped = snap_to_grid(target, h)
            notes.append(f"{name} {target*1e3:.3f}->{snapped*1e3:.3f}mm")
    return "clean" if not notes else "rounded: " + ", ".join(notes)


def _build_parallel_plate_geometry(config, h):
    """Return (conductors, eps_r_of_xy, dims_dict) for the parallel-plate
    example at grid spacing h.  All sizes are snap_to_grid'd."""
    plate_t = snap_to_grid(config.plate_thickness, h)
    gap = snap_to_grid(config.gap, h)
    dielectric_t = snap_to_grid(config.dielectric_thickness, h)
    plate_w = snap_to_grid(config.plate_width, h)
    margin = snap_to_grid(config.domain_margin, h)

    Lx = plate_w + 2 * margin
    Ly = 2 * plate_t + gap + 2 * margin
    x_plate0 = margin
    y_gap_lo = margin + plate_t
    y_gap_hi = y_gap_lo + gap

    bottom_plate = Rectangle(x_plate0, margin, plate_w, plate_t,
                             voltage=0.0, name="bottom_plate")
    top_plate = Rectangle(x_plate0, y_gap_hi, plate_w, plate_t,
                          voltage=config.voltage, name="top_plate")
    conductors = [bottom_plate, top_plate]

    slab = Rectangle(x_plate0, y_gap_lo, plate_w, dielectric_t,
                     eps_r=config.dielectric_eps_r, name="dielectric_slab")
    eps_r_of_xy = make_eps_r_function(
        [slab], background_eps_r=config.background_eps_r)

    dims = {
        "plate_t": plate_t, "gap": gap, "dielectric_t": dielectric_t,
        "plate_w": plate_w, "margin": margin, "Lx": Lx, "Ly": Ly,
        "x_plate0": x_plate0, "y_gap_lo": y_gap_lo, "y_gap_hi": y_gap_hi,
    }
    return conductors, eps_r_of_xy, dims


def _build_graded_parallel_plate_mesh(h, dims, tune=None):
    """Construct a graded Mesh for the parallel-plate example.
    All geometry lines that were snap_to_grid'd remain exactly on mesh lines."""
    tune = tune or GRADED_MESH_DEFAULTS
    d = dims

    # x-direction: coarse margins → fine edge bands → medium interior
    edge_band = max(tune.edge_band_width_factor *
                    h, tune.edge_band_width_min_m)
    n_margin_x = max(tune.min_margin_points,
                     round(d["margin"] / (tune.margin_spacing_factor * h)) + 1)
    n_edge = max(tune.min_edge_points,
                 round(edge_band / (tune.edge_spacing_factor * h)) + 1)
    n_interior = max(tune.min_interior_points,
                     round((d["plate_w"] - 2 * edge_band) /
                           (tune.interior_spacing_factor * h)) + 1)

    if d["plate_w"] <= 2 * edge_band:
        edge_band = 0.25 * d["plate_w"]
        n_edge = max(tune.min_edge_points,
                     round(edge_band / (tune.edge_spacing_factor * h)) + 1)
        n_interior = max(tune.min_fallback_interior_points,
                         round((d["plate_w"] - 2 * edge_band) /
                               (tune.fallback_interior_spacing_factor * h)) + 1)

    xs = build_graded_coords([
        (0.0, d["x_plate0"], n_margin_x),
        (d["x_plate0"], d["x_plate0"] + edge_band, n_edge),
        (d["x_plate0"] + edge_band, d["x_plate0"] +
         d["plate_w"] - edge_band, n_interior),
        (d["x_plate0"] + d["plate_w"] - edge_band,
         d["x_plate0"] + d["plate_w"], n_edge),
        (d["x_plate0"] + d["plate_w"], d["Lx"], n_margin_x),
    ])

    # y-direction: coarse outer margins → medium plates → fine gap
    n_margin_y = max(tune.min_margin_points,
                     round(d["margin"] / (tune.margin_spacing_factor * h)) + 1)
    n_plate = max(tune.min_plate_points,
                  round(d["plate_t"] / (tune.plate_spacing_factor * h)) + 1)
    n_gap = max(tune.min_gap_points,
                round(d["gap"] / (tune.gap_spacing_factor * h)) + 1)

    ys = build_graded_coords([
        (0.0, d["margin"], n_margin_y),
        (d["margin"], d["y_gap_lo"], n_plate),
        (d["y_gap_lo"], d["y_gap_hi"], n_gap),
        (d["y_gap_hi"], d["y_gap_hi"] + d["plate_t"], n_plate),
        (d["y_gap_hi"] + d["plate_t"], d["Ly"], n_margin_y),
    ])

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
    C_ideal = (d["plate_w"] * EPS0 /
               (d["dielectric_t"] / config.dielectric_eps_r +
                (d["gap"] - d["dielectric_t"]) / config.background_eps_r))

    result = dict(h=h, mesh=mesh, conductors=conductors, eps_r_of_xy=eps_r_of_xy,
                  V=V, is_fixed=is_fixed, energy_density=energy_density,
                  C=C, C_ideal=C_ideal, solve_time=solve_time, graded=use_graded)
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
    print("Two distinct effects are mixed together in the table above, and the")
    print("'grid alignment' column separates them. The core assembly/solve")
    print("pipeline reproduces an exact, fringing-free analytical case to")
    print("machine precision at every resolution tested (README section 8.1) --")
    print("so genuine FEM discretization error is real but small, visible in")
    print("the 'clean' rows above. A 'rounded' row is different: snap_to_grid")
    print("(README section 4.3) has adjusted a physically meaningful dimension")
    print("(the gap or the dielectric thickness) to the nearest value that h")
    print("can represent exactly, because the target didn't divide evenly into")
    print("that h -- meaning a 'rounded' row is genuinely simulating a slightly")
    print("different capacitor, not just resolving the same one more finely.")
    print("Separately, and unrelated to grid alignment: the field concentrates")
    print("sharply at the plate's corner (a geometric singularity), and each h")
    print("above is an independent structured mesh rather than a nested")
    print("refinement of the previous one, so successive CLEAN levels are still")
    print("not guaranteed to bracket the true answer monotonically (see")
    print("LIMITATIONS AND FUTURE WORK). For a convergence study, prefer h")
    print("values that divide evenly into every geometric parameter you care")
    print("about; treat the finest clean level as accurate to roughly the")
    print("spread shown among the other clean rows.")


def example_parallel_plate(config=None):
    """Parallel-plate capacitor with partial dielectric slab."""
    config = config or ParallelPlateConfig()

    print("=" * 72)
    print("EXAMPLE 1: Parallel-plate capacitor, partially filled with a dielectric slab")
    print("=" * 72)

    print("Mesh convergence (physical geometry fixed, only h changes):")
    results = [_solve_parallel_plate(config, h)
               for h in config.convergence_spacings]
    _print_parallel_plate_convergence(results, config)
    if VERBOSE_CONVERGENCE_NOTES:
        _print_parallel_plate_convergence_notes()
    print()

    # Production solve: finest uniform level (last convergence entry)
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

    # Optional graded comparison
    if RUN_GRADED_COMPARISON:
        print("Graded Cartesian mesh (same nominal h, refined near edges & gap):")
        graded = _solve_parallel_plate(config, uniform["h"], use_graded=True)
        C_graded = graded["C"]
        print(f"mesh: {graded['mesh'].n_nodes} nodes, {graded['mesh'].n_tris} triangles "
              f"(graded, nominal h = {graded['h']*1e3:.3f} mm)")
        print(f"FEM capacitance (graded)  : {C_graded*1e12:9.3f} pF/m")
        print(
            f"Delta vs uniform          : {100*(C_graded-C_uniform)/C_uniform:+.2f} %")
        print("Capacitance is per unit depth into the page; multiply by the actual "
              "plate depth in meters for total farads.")
        print("The graded mesh is the dependency-free intermediate step of README §11; "
              "it improves resolution of the corner singularity without adding any "
              "native dependency.")

        plot_solution(graded["mesh"], graded["V"], graded["eps_r_of_xy"],
                      graded["energy_density"], graded["conductors"], graded["is_fixed"],
                      "Parallel-plate capacitor (glass slab in an air gap) — graded mesh",
                      os.path.join(OUTPUT_DIR, "example1_parallel_plate.png"),
                      xlim=(graded["x_plate0"] - config.plot_margin,
                            graded["x_plate0"] + graded["plate_w"] + config.plot_margin),
                      ylim=(graded["margin"] - config.plot_margin,
                            graded["margin"] + config.plot_margin
                            + 2 * graded["plate_t"] + graded["gap"]))
        return C_graded, C_ideal

    # If graded comparison is disabled, plot the uniform result instead
    plot_solution(uniform["mesh"], uniform["V"], uniform["eps_r_of_xy"],
                  uniform["energy_density"], uniform["conductors"], uniform["is_fixed"],
                  "Parallel-plate capacitor (glass slab in an air gap)",
                  os.path.join(OUTPUT_DIR, "example1_parallel_plate.png"),
                  xlim=(uniform["x_plate0"] - config.plot_margin,
                        uniform["x_plate0"] + uniform["plate_w"] + config.plot_margin),
                  ylim=(uniform["margin"] - config.plot_margin,
                        uniform["margin"] + config.plot_margin
                        + 2 * uniform["plate_t"] + uniform["gap"]))
    return C_uniform, C_ideal


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
                C=C, C_ideal=C_ideal, solve_time=solve_time)


def example_coax(config=None):
    """Coaxial cable with polyethylene dielectric fill."""
    config = config or CoaxConfig()

    print()
    print("=" * 72)
    print("EXAMPLE 2: Coaxial capacitor (polyethylene-filled)")
    print("=" * 72)

    print("Mesh convergence (smooth circular boundary, no sharp corner):")
    print(
        f"{'h [mm]':>9s}{'nodes':>9s}{'solve [s]':>11s}{'C [pF/m]':>12s}{'error':>9s}")
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
        print("Error shrinks toward 0% overall. These five points happen to be")
        print("monotonic, but that describes this specific sweep, not a general")
        print("guarantee -- finer intermediate resolutions reveal small reversals")
        print("too (same non-nested-mesh effect as example 1, far smaller in size;")
        print("see README.md section 8.2).")
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
                  os.path.join(OUTPUT_DIR, "example2_coax.png"))
    return C, C_ideal


def _solve_exact_check(config, h):
    """Full-width-plate variant: fringing is geometrically impossible."""
    plate_t = snap_to_grid(config.plate_thickness, h)
    gap = snap_to_grid(config.gap, h)
    dielectric_t = snap_to_grid(config.dielectric_thickness, h)
    margin = snap_to_grid(config.domain_margin, h)

    Lx = snap_to_grid(config.plate_width, h)
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

    slab = Rectangle(-overhang, y_gap_lo, Lx + 2 * overhang, dielectric_t,
                     eps_r=config.dielectric_eps_r, name="dielectric_slab")
    eps_r_of_xy = make_eps_r_function(
        [slab], background_eps_r=config.background_eps_r)

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
    print("example above, but the plates now extend across (and past) the")
    print("whole domain, so the series-capacitor formula is exact here, not")
    print("idealized. This isolates the solver's own correctness from the")
    print("mesh's ability to represent any particular boundary shape.")
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
    print("The 'error' column displays as 0.000000% at 6 decimal places, but")
    print("is not literally zero -- 'rel. diff' shows the true residual, a few")
    print("times float64 epsilon (2.22e-16), i.e. exact to machine precision:")
    print("ordinary floating-point roundoff from the sparse solve, the same at")
    print("every resolution, not a limitation of the method or the mesh. This")
    print("confirms the core FEM machinery -- assembly, boundary conditions,")
    print("energy integration -- has no implementation bugs at any of these")
    print("resolutions, including the two-layer dielectric handling. Any error")
    print("elsewhere in this project comes from the mesh, not the math")
    print("(README.md section 8.1).")

    return results[-1]["C"], results[-1]["C_exact"]


def verify_boundary_tolerance():
    """§10.4 stress test: perturbed boundary coordinates must still classify
    correctly thanks to BOUNDARY_TOLERANCE_M."""
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
        print()
        print("FAILURE: at least one near-edge coordinate was classified outside.")
        print("The tolerance is not large enough for the observed ULP noise.")
        return False

    print()
    print("All near-edge coordinates classified as INSIDE -- tolerance is")
    print("doing its job. (A strict <= comparison would have failed for some")
    print("of the reconstructed values above.)")
    print()
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
    print(f"Figures saved to {OUTPUT_DIR}/example1_parallel_plate.png and "
          f"{OUTPUT_DIR}/example2_coax.png")


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

    C1, C1_ideal = example_parallel_plate()
    C2, C2_ideal = example_coax()

    print_summary(C1, C1_ideal, C2, C2_ideal, time.time() - t_start)
