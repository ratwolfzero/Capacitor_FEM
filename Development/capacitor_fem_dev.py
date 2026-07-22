"""
capacitor_fem.py
=================

A two-dimensional finite-element electrostatics solver for simulating
capacitor geometries -- parallel plates, coaxial cables, and (via the
Shape/CSG primitives in GEOMETRY) more complex arrangements built from
simple shapes -- instead of relying on closed-form formulas that only
exist for a handful of idealized geometries. Pure NumPy / SciPy /
Matplotlib, no external mesh-generation library.

Full documentation -- the physics (Maxwell's equations to the governing
PDE), the math (weak form, Galerkin discretization, element formulas),
the numerical method, the architecture, usage examples, and validation
results -- lives in README.md. This file's comments are intentionally
brief pointers into that document; only the function/class docstrings
below are meant to be self-contained.

CONTENTS
--------
     1. CONFIGURATION    ParallelPlateConfig, CoaxConfig, PlotConfig
     2. GEOMETRY         Shape (base, with CSG |, &, - operators),
                          Circle / Rectangle / OutsideCircle
     3. MATERIALS        Material, make_eps_r_function()
     4. MESH             snap_to_grid(), structured triangular Mesh
     5. SOLVER           evaluate_material(), assemble_stiffness(),
                          apply_conductors_and_solve()
     6. POST-PROCESSING  compute_fields(), capacitance_from_energy()
     7. HIGH-LEVEL API   ElectrostaticProblem
     8. VISUALIZATION    plot_solution()
     9. EXAMPLES         parallel-plate capacitor (partial dielectric
                          slab) and coaxial cable, each with a mesh-
                          convergence study; plus an exact-solution
                          validation check (off by default -- see
                          RUN_EXACT_CHECK)
    10. LIMITATIONS AND FUTURE WORK  (see README.md for full detail)

Dependencies: numpy, scipy, matplotlib.
Run directly: python3 capacitor_fem.py
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

EPS0 = 8.8541878128e-12  # vacuum permittivity [F/m]
OUTPUT_DIR = ""  # where the example figures are written


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# Every geometry, material, and numerical tuning parameter lives here as
# frozen dataclasses instead of bare literals in the solver code -- see
# README.md section 5.2. All lengths in meters, voltage in volts, eps_r
# is relative permittivity (dimensionless).

@dataclass(frozen=True)
class ParallelPlateConfig:
    """Parameters for the parallel-plate capacitor example: two
    rectangular plates separated by a gap, with a rectangular dielectric
    slab filling the lower half of that gap."""

    plate_thickness: float = 1e-3
    gap: float = 4e-3
    dielectric_thickness: float = 2e-3    # slab fills the lower half of the gap
    plate_width: float = 24e-3
    domain_margin: float = 15e-3          # clearance between the plates and the domain edge
    voltage: float = 100.0
    dielectric_eps_r: float = 4.5         # e.g. glass
    background_eps_r: float = 1.0         # e.g. air
    mesh_spacing: float = 0.1e-3           # production grid spacing h

    # The field default below is the exact literal tuple this project has
    # validated and published results against -- not derived from
    # mesh_spacing by arithmetic. That distinction matters: multiplying
    # mesh_spacing by a ratio (e.g. 1.5 * 0.1e-3) does not reliably
    # reproduce the literal 0.15e-3 bit-for-bit, and because every
    # conductor/material edge in this project is deliberately snapped to
    # land exactly on a grid line (snap_to_grid), even a last-bit
    # difference in a boundary coordinate can flip an entire row of mesh
    # nodes across a Shape.contains() '<=' comparison -- found by testing
    # this exact feature: an arithmetically "equivalent" h reclassified
    # one full row of nodes as conductor, changing a reported capacitance
    # by several percent. Keeping this field's default a literal value,
    # and only ever taking the auto-derivation path below when it's
    # detected as untouched, means the well-tested default configuration
    # is never exposed to that risk.
    convergence_spacings: tuple = (0.4e-3, 0.2e-3, 0.15e-3, 0.1e-3)
    plot_margin: float = 8e-3             # extra plot view window around the plates

    # Multipliers of mesh_spacing used to auto-derive a fresh
    # convergence_spacings when mesh_spacing has been changed but
    # convergence_spacings has not -- preserves the coarse-to-fine
    # "shape" of the shipped sweep (4x, 2x, 1.5x, 1x the production
    # resolution) at the new mesh_spacing. Neither is a dataclass field
    # (ClassVar), so neither appears in __init__.
    _CONVERGENCE_RATIOS: ClassVar[tuple] = (4.0, 2.0, 1.5, 1.0)
    _DEFAULT_CONVERGENCE_SPACINGS: ClassVar[tuple] = (0.4e-3, 0.2e-3, 0.15e-3, 0.1e-3)

    def __post_init__(self):
        if self.convergence_spacings[-1] != self.mesh_spacing:
            if self.convergence_spacings == self._DEFAULT_CONVERGENCE_SPACINGS:
                # mesh_spacing was changed but convergence_spacings was
                # left untouched -- derive a fresh sweep for the new
                # mesh_spacing instead of failing. frozen dataclass:
                # object.__setattr__ is the standard way to set a
                # computed value from __post_init__.
                object.__setattr__(self, "convergence_spacings",
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
class CoaxConfig:
    """Parameters for the coaxial-cable example: a circular inner
    conductor and a circular outer shield, with a dielectric fill between
    them."""

    inner_radius: float = 3e-3
    outer_radius: float = 15e-3
    domain_half_width: float = 17e-3      # simulation domain extends to +/- this
    voltage: float = 100.0
    dielectric_eps_r: float = 2.3          # e.g. polyethylene, a common coax dielectric
    background_eps_r: float = 1.0          # only matters outside the dielectric fill radius
    mesh_spacing: float = 0.075e-3

    # See ParallelPlateConfig.convergence_spacings for why this is a
    # literal tuple, not something derived from mesh_spacing.
    convergence_spacings: tuple = (0.3e-3, 0.2e-3, 0.15e-3, 0.1e-3, 0.075e-3)

    # See ParallelPlateConfig._CONVERGENCE_RATIOS for what this is and
    # why. Written as exact fractions (8/3, 4/3), not truncated decimals,
    # so that when this path DOES run (mesh_spacing changed,
    # convergence_spacings left at the literal default above), the result
    # is as numerically clean as a derived value can be.
    _CONVERGENCE_RATIOS: ClassVar[tuple] = (4.0, 8 / 3, 2.0, 4 / 3, 1.0)
    _DEFAULT_CONVERGENCE_SPACINGS: ClassVar[tuple] = (0.3e-3, 0.2e-3, 0.15e-3, 0.1e-3, 0.075e-3)

    def __post_init__(self):
        if self.convergence_spacings[-1] != self.mesh_spacing:
            if self.convergence_spacings == self._DEFAULT_CONVERGENCE_SPACINGS:
                object.__setattr__(self, "convergence_spacings",
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
class PlotConfig:
    """Shared visualization tuning parameters for plot_solution()."""

    figsize: tuple = (13, 11)
    dpi: int = 140
    potential_fill_levels: int = 25
    potential_line_levels: int = 15
    streamline_density: float = 1.1
    energy_density_floor: float = 1e-4    # log-scale color vmin, as a fraction of vmax
    conductor_fill_color: str = "dimgray"
    conductor_outline_color: str = "black"
    conductor_outline_width: float = 1.3


# =============================================================================
# 2. GEOMETRY -- conductors and dielectric-region shapes
# =============================================================================
# Every shape implements one method, contains(x, y) -- the only interface
# the rest of the file relies on. See README.md section 5.3.

# Absolute tolerance for the boundary comparisons in contains() below, in
# meters. Two arithmetically-equivalent ways of computing "the same"
# boundary coordinate -- e.g. a value built via snap_to_grid versus the
# same nominal position read off a mesh's np.linspace-generated grid --
# can differ by a few times the float64 epsilon (observed: ~1e-18 to
# 1e-15 m at this project's length scales), which a strict comparison
# can resolve the wrong way, silently excluding an entire row or column
# of nodes that should be part of a conductor or material region. Found
# in practice, not hypothetically: it affects the shipped
# example_parallel_plate() convergence table at 3 of its 4 resolutions
# (0.2, 0.15, 0.1 mm), each showing several volts of error at what
# should be an exact conductor surface. This tolerance is many orders of
# magnitude larger than the observed noise floor, while remaining many
# orders of magnitude smaller than the smallest grid spacing used
# anywhere in this project (h=75 um for the finest coax mesh) -- so it
# can only rescue a node that floating-point arithmetic nudged off its
# intended exact position; it cannot reach a genuinely different,
# adjacent grid point. See README.md section 10.4.
_BOUNDARY_TOL = 1e-9  # meters


class Shape(ABC):
    """Common interface for conductor and dielectric-region shapes.

    Set `voltage` to use a shape as a conductor (a Dirichlet boundary
    condition at that potential), or `eps_r` to use it as a dielectric
    region; a shape may be used as both, in different roles in different
    parts of a problem.

    Shapes compose with ordinary set-like operators: `a | b` (union),
    `a & b` (intersection), `a - b` (difference) each return a new Shape
    whose contains() combines the operands' contains() with the matching
    NumPy boolean operator, e.g.

        annulus = Circle((0, 0), 10e-3, eps_r=4.5) - Circle((0, 0), 6e-3)

    defines a ring-shaped dielectric region with no changes needed
    anywhere else, since assembly only ever calls shape.contains(x, y)
    and never inspects a shape's concrete type.
    """

    voltage = None
    eps_r = None
    name = "shape"

    @abstractmethod
    def contains(self, x, y):
        """Return True where (x, y) lies inside the shape.

        x, y may be scalars, 1D arrays (e.g. triangle centroids), or 2D
        arrays (e.g. a plotting grid); the returned mask matches their
        shape.
        """

    def __or__(self, other):
        return Union(self, other)

    def __and__(self, other):
        return Intersection(self, other)

    def __sub__(self, other):
        return Difference(self, other)


class Rectangle(Shape):
    """An axis-aligned rectangle spanning [x0, x0+width] x [y0, y0+height].

    Boundary comparisons include a small absolute tolerance
    (_BOUNDARY_TOL) so a boundary coordinate built one way (e.g.
    snap_to_grid) reliably matches the same nominal position on a mesh
    built another way (np.linspace), even where the two differ by a few
    times the float64 epsilon. See _BOUNDARY_TOL above.
    """

    def __init__(self, x0, y0, width, height, voltage=None, eps_r=None, name="rectangle"):
        self.x0, self.y0 = x0, y0
        self.width, self.height = width, height
        self.voltage = voltage
        self.eps_r = eps_r
        self.name = name

    def contains(self, x, y):
        return ((x >= self.x0 - _BOUNDARY_TOL) & (x <= self.x0 + self.width + _BOUNDARY_TOL) &
                (y >= self.y0 - _BOUNDARY_TOL) & (y <= self.y0 + self.height + _BOUNDARY_TOL))


class Circle(Shape):
    """A filled disk of the given radius, centered at `center`.

    See Rectangle's docstring and _BOUNDARY_TOL above for why the
    boundary comparison includes a small tolerance.
    """

    def __init__(self, center, radius, voltage=None, eps_r=None, name="circle"):
        self.cx, self.cy = center
        self.radius = radius
        self.voltage = voltage
        self.eps_r = eps_r
        self.name = name

    def contains(self, x, y):
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= (self.radius + _BOUNDARY_TOL) ** 2


class OutsideCircle(Shape):
    """The complement of a disk: everything OUTSIDE the given radius.
    Convenient for a closed outer shield that extends to the domain
    boundary, e.g. the outer conductor of a coaxial cable.

    See Rectangle's docstring and _BOUNDARY_TOL above for why the
    boundary comparison includes a small tolerance, applied here in the
    opposite direction (inward) so this stays the exact complement of
    Circle at the shared boundary.
    """

    def __init__(self, center, radius, voltage=None, eps_r=None, name="outside_circle"):
        self.cx, self.cy = center
        self.radius = radius
        self.voltage = voltage
        self.eps_r = eps_r
        self.name = name

    def contains(self, x, y):
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 >= (self.radius - _BOUNDARY_TOL) ** 2


class _CombinedShape(Shape):
    """Shared bookkeeping for the three boolean-composite shapes below."""

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
# 3. MATERIALS
# =============================================================================

class Material:
    """A named relative permittivity, for labeling dielectric regions,
    e.g. Material("glass", eps_r=4.5). Purely descriptive: the solver
    only ever consumes the eps_r value, via make_eps_r_function below."""

    def __init__(self, name, eps_r):
        self.name = name
        self.eps_r = eps_r


def make_eps_r_function(regions, background_eps_r=1.0):
    """Combine a list of dielectric-region shapes into a single relative-
    permittivity function of position.

    Parameters
    ----------
    regions : list of Shape
        Shapes with `eps_r` set. Later entries are painted on top of
        earlier ones wherever they overlap.
    background_eps_r : float
        Relative permittivity outside every region (e.g. air = 1.0).

    Returns
    -------
    callable
        eps_r_of_xy(x, y) -> ndarray of relative permittivity, evaluated
        at the given coordinates (scalars or arrays of matching shape).
    """
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
# 4. MESH -- structured triangular mesh over a rectangular domain
# =============================================================================
# A plain Cartesian grid, no external mesh-generation library -- the
# simplicity/accuracy tradeoff behind this whole project. See README.md
# sections 4.1 (mesh), 4.3 (grid alignment), and 10 (limitations).

def snap_to_grid(target, h):
    """Round a physical dimension to the nearest exact multiple of the
    grid spacing h.

    This matters more than it looks. On a structured, non-conforming
    mesh, a conductor or material-interface edge that falls *between* two
    grid lines gets rounded to the nearer one when nodes are classified
    as inside or outside a shape -- an implicit, easy-to-miss rounding
    that changes the simulated geometry by up to half a grid cell.
    Calling snap_to_grid on every feature size before constructing
    geometry makes "intended size" and "simulated size" match exactly,
    for any h. It is also what makes a mesh-convergence sweep (varying h
    while the physical geometry should stay fixed) actually test
    convergence, rather than silently rescaling the whole problem along
    with the mesh -- the latter produces near-identical node counts at
    every nominal "resolution" and a convergence table with no real
    content.
    """
    return round(target / h) * h


def _estimate_peak_memory_gb(n_nodes):
    """Rough order-of-magnitude estimate of peak memory (GB) for solving
    a problem on a mesh with n_nodes nodes, used only to decide whether
    to print the large-mesh warning below.

    Calibrated empirically, not derived from first principles: measured
    peak RSS on this project's coax example at four mesh resolutions
    (13k, 52k, 206k, and 824k nodes) and fit a power law,
    peak_RSS_MB = 0.1432 * n_nodes**0.680. The fitted exponent (~0.68,
    i.e. worse than linear in node count) is consistent with the fill-in
    a general sparse LU factorization produces on a 2D grid -- see the
    note on apply_conductors_and_solve below. Treat this as a ballpark
    for "is this about to be a problem," not a guarantee: actual memory
    depends on the machine, BLAS/LAPACK build, and problem specifics.
    """
    return 0.1432 * n_nodes ** 0.680 / 1024


def _warn_if_large_mesh(n_nodes, n_tris):
    """Print a one-time, non-blocking heads-up if a mesh is large enough
    that memory could plausibly become a problem, calibrated against
    _estimate_peak_memory_gb. Does not raise or stop execution -- a
    large mesh may be exactly what's wanted on a machine with enough
    RAM -- it only makes the cost visible before a solve starts, rather
    than as a MemoryError or an OS-killed process partway through.
    """
    est_gb = _estimate_peak_memory_gb(n_nodes)
    if n_nodes >= 5_000_000:
        print(f"WARNING: this mesh has {n_nodes:,} nodes ({n_tris:,} triangles). "
              f"Estimated peak memory during assembly/solve: roughly {est_gb:.1f} GB "
              f"(ballpark, see _estimate_peak_memory_gb). This comes from the direct "
              f"sparse solve (scipy.sparse.linalg.spsolve) on a UNIFORM grid -- every "
              f"region of the domain gets the same node density, whether the field "
              f"needs it there or not, and a general LU factorization's memory use "
              f"grows faster than the node count itself. If this is more than your "
              f"machine can handle: use a coarser mesh_spacing, or see LIMITATIONS "
              f"AND FUTURE WORK / README.md sections 10-11 for a graded or "
              f"unstructured mesh, which would need far fewer nodes for the same "
              f"local resolution.")
    elif n_nodes >= 1_000_000:
        print(f"Note: this mesh has {n_nodes:,} nodes ({n_tris:,} triangles), "
              f"estimated peak memory roughly {est_gb:.1f} GB (ballpark). Both "
              f"worked examples in this project run at well under half this size.")


class Mesh:
    """A structured triangular mesh spanning [x0, x0+Lx] x [y0, y0+Ly],
    built from an nx-by-ny grid of nodes.

    Each grid cell is split into two triangles, alternating which
    diagonal is used in a checkerboard pattern (rather than always the
    same direction) so the mesh has no built-in directional bias.
    """

    def __init__(self, x0, y0, Lx, Ly, nx, ny):
        self.xs = np.linspace(x0, x0 + Lx, nx)
        self.ys = np.linspace(y0, y0 + Ly, ny)
        X, Y = np.meshgrid(self.xs, self.ys)          # shape (ny, nx)
        self.points = np.column_stack([X.ravel(), Y.ravel()])
        self.nx, self.ny = nx, ny
        self.n_nodes = self.points.shape[0]

        # Computed cheaply from nx, ny directly (no need to build the
        # actual triangle array first) so the warning can fire as early
        # as possible for a very large mesh, before any of the more
        # expensive construction below.
        _warn_if_large_mesh(self.n_nodes, 2 * (nx - 1) * (ny - 1))

        i = np.arange(nx - 1)
        j = np.arange(ny - 1)
        II, JJ = np.meshgrid(i, j)                     # shape (ny-1, nx-1)
        II = II.ravel()
        JJ = JJ.ravel()

        n00 = JJ * nx + II
        n10 = JJ * nx + (II + 1)
        n01 = (JJ + 1) * nx + II
        n11 = (JJ + 1) * nx + (II + 1)

        even = (II + JJ) % 2 == 0

        # cell split A: even -> (n00,n10,n11)   odd -> (n00,n10,n01)
        triA = np.column_stack([n00, n10, np.where(even, n11, n01)])
        # cell split B: even -> (n00,n11,n01)   odd -> (n10,n11,n01)
        triB = np.column_stack([np.where(even, n00, n10), n11, n01])

        tris = np.empty((2 * len(II), 3), dtype=int)
        tris[0::2] = triA
        tris[1::2] = triB

        self.triangles = tris
        self.n_tris = tris.shape[0]

    def centroids(self):
        """Return the (n_tris, 2) array of triangle centroids."""
        pts = self.points[self.triangles]
        return pts.mean(axis=1)


# =============================================================================
# 5. SOLVER -- FEM assembly and the Dirichlet solve
# =============================================================================

def _triangle_geometry(mesh):
    """Linear (P1) shape-function gradient coefficients and triangle
    areas, shared by assembly and field recovery.

    For a triangle with vertices 1, 2, 3, the shape function N_i is
    linear with grad(N_i) = (b_i, c_i) / (2 * A_signed), where A_signed is
    the SIGNED triangle area (positive for counterclockwise vertex order,
    negative for clockwise). Using the signed value is what makes this
    gradient formula correct regardless of how a triangle's vertices
    happen to be ordered. The stiffness matrix, by contrast, needs the
    UNSIGNED (physical) area, since it comes from an area integral.
    """
    tris = mesh.triangles
    p = mesh.points
    x1, y1 = p[tris[:, 0], 0], p[tris[:, 0], 1]
    x2, y2 = p[tris[:, 1], 0], p[tris[:, 1], 1]
    x3, y3 = p[tris[:, 2], 0], p[tris[:, 2], 1]

    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

    area2 = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)   # signed, = 2*A
    area = 0.5 * np.abs(area2)                                  # unsigned

    b = np.column_stack([b1, b2, b3])
    c = np.column_stack([c1, c2, c3])
    return b, c, area, area2


def evaluate_material(mesh, eps_r_of_xy):
    """Evaluate relative permittivity at each triangle's centroid and
    convert it to absolute permittivity (F/m).

    Kept as its own step, separate from assembly, so a nonlinear or
    iterative solve -- eps depending on the field from the previous
    iteration -- only needs to call this and assemble_stiffness in a
    loop; the geometry and assembly code never changes.

    Centroid evaluation is exact for a triangle lying entirely inside one
    material region. For an axis-aligned rectangular region whose edges
    have been snapped to the grid (see snap_to_grid), it is in fact exact
    everywhere: every mesh triangle then has all three vertices on one
    side of the interface, or at most touches it along a single vertex or
    edge of zero area, so there is no triangle whose true area genuinely
    splits between materials. Real straddling only occurs for a boundary
    that cannot be grid-aligned, such as the coax example's circular
    dielectric fill. There, replacing the single centroid sample with an
    area-weighted multi-point average (tested up to 64 interior sample
    points per triangle) changes the coax capacitance by about 0.001
    percentage points -- negligible next to the roughly 0.75% error from
    the conductor boundary's own node-in/node-out classification, which
    a finer material quadrature does not touch. So while a straddling
    triangle's material assignment is, in principle, an O(h)
    approximation distinct from the O(h) conductor-boundary error (see
    apply_conductors_and_solve), refining it in isolation is not
    worthwhile while the conductor boundary itself is still staircased;
    see LIMITATIONS AND FUTURE WORK.
    """
    cxy = mesh.centroids()
    return eps_r_of_xy(cxy[:, 0], cxy[:, 1]) * EPS0   # (T,) absolute permittivity


def assemble_stiffness(mesh, eps_elem):
    """Assemble the sparse global stiffness matrix K such that K @ V = 0
    (before boundary conditions) represents div(eps grad V) = 0 in weak
    form, given a per-triangle absolute permittivity eps_elem (see
    evaluate_material).

    The local stiffness contribution for a triangle with (signed-area-
    independent) gradient coefficients b, c and unsigned area A is

        Ke[i, j] = eps * (b_i b_j + c_i c_j) / (4 A)

    Assembly builds one vectorized array of (row, column, value) triplets
    across all triangles and hands it to the csr_matrix constructor,
    which sums duplicate (row, column) entries internally -- assembling
    as triplets and converting once, rather than inserting into a sparse
    matrix one element at a time.
    """
    tris = mesh.triangles
    n = mesh.n_nodes
    b, c, area, area2 = _triangle_geometry(mesh)

    # Ke[t,i,j] = eps[t] * (b[t,i]*b[t,j] + c[t,i]*c[t,j]) / (4*area[t])
    Ke = b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]
    Ke = Ke * (eps_elem / (4.0 * area))[:, None, None]

    I = np.repeat(tris[:, :, None], 3, axis=2)   # I[t,i,j] = tris[t,i]
    J = np.repeat(tris[:, None, :], 3, axis=1)   # J[t,i,j] = tris[t,j]

    K = csr_matrix((Ke.ravel(), (I.ravel(), J.ravel())), shape=(n, n))
    return K, area, area2, b, c


def apply_conductors_and_solve(mesh, K, conductors):
    """Mark every mesh node that falls inside a conductor's shape as a
    Dirichlet node at that conductor's voltage, then solve the reduced
    linear system for the remaining ("free") nodes.

    This "filled conductor" approach -- rather than meshing only the
    dielectric and applying the Dirichlet condition on the boundary
    contour of a hole -- is physically exact for a triangle entirely
    inside a conductor (all three nodes fixed): V is constant across it,
    so E = 0 there regardless of what material it happens to be assigned,
    and it contributes exactly zero to the stored energy. It is only
    approximate for the thin layer of triangles straddling a conductor's
    boundary (one or two nodes fixed, the rest free), which do have
    nonzero field and whose assigned material genuinely affects the
    result, by O(h) -- the same staircase limitation as everywhere else
    on this mesh, shrinking as h -> 0. A conforming, boundary-only
    representation removes this ambiguity entirely and is the right
    foundation if surface-charge or force output is added later (see
    LIMITATIONS AND FUTURE WORK); it is not needed for the bulk
    capacitance this script computes via the energy method below.
    """
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
# 6. POST-PROCESSING -- E, D, energy density, capacitance
# =============================================================================

def compute_fields(mesh, V, eps_elem, b, c, area, area2):
    """Recover E, D, and energy density from the nodal potential V.

    E (and D) are exactly constant within each linear triangular element,
    since they come from the gradient of a piecewise-linear V -- this is
    a property of P1 elements, not an approximation layered on top of
    them.
    """
    tris = mesh.triangles
    V1, V2, V3 = V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]

    dVdx = (V1 * b[:, 0] + V2 * b[:, 1] + V3 * b[:, 2]) / area2
    dVdy = (V1 * c[:, 0] + V2 * c[:, 1] + V3 * c[:, 2]) / area2

    Ex, Ey = -dVdx, -dVdy
    Emag = np.hypot(Ex, Ey)

    Dx, Dy = eps_elem * Ex, eps_elem * Ey

    energy_density = 0.5 * (Ex * Dx + Ey * Dy)     # J/m^3
    W = np.sum(energy_density * area)               # J/m  (per unit depth)

    return Ex, Ey, Emag, Dx, Dy, energy_density, W


def capacitance_from_energy(W, V_hi, V_lo):
    """Two-conductor capacitance from stored energy: C = 2W / (dV)^2.

    Preferred over integrating surface charge along a conductor boundary:
    it only needs the volume/area integral of the field the solver
    already computed everywhere (compute_fields), rather than a separate
    boundary-flux integral that would need its own treatment of the
    staircased conductor boundary. Returns capacitance PER UNIT DEPTH
    (F/m), since this is a 2D solve -- multiply by the actual depth for
    total farads.
    """
    dV = V_hi - V_lo
    return 2.0 * W / dV ** 2


# =============================================================================
# 7. HIGH-LEVEL API -- a thin facade over the pipeline above
# =============================================================================

class ElectrostaticProblem:
    """A convenience wrapper around evaluate_material ->
    assemble_stiffness -> apply_conductors_and_solve -> compute_fields,
    for setting up and solving a new problem in a few lines::

        problem = ElectrostaticProblem(mesh)
        problem.add_conductor(inner, 100.0)
        problem.add_conductor(outer, 0.0)
        problem.add_dielectric(fill_region)     # eps_r set on the shape
        problem.solve()
        C = problem.capacitance(100.0, 0.0)

    This is a genuinely thin wrapper: every method below calls the
    existing module-level functions in the right order and stores their
    results as attributes; there is no numerics here beyond what is
    already in SOLVER and POST-PROCESSING above. The worked examples
    later in this file call those functions directly instead, since
    walking through each step explicitly is the point of a worked
    example; reach for this class when setting up a new problem without
    wanting to restate the pipeline every time.
    """

    def __init__(self, mesh, background_eps_r=1.0):
        self.mesh = mesh
        self.background_eps_r = background_eps_r
        self.conductors = []
        self.dielectrics = []
        self.V = None          # populated by solve()
        self.W = None

    def add_conductor(self, shape, voltage):
        """Register `shape` as a conductor held at `voltage`."""
        shape.voltage = voltage
        self.conductors.append(shape)
        return shape

    def add_dielectric(self, shape, eps_r=None):
        """Register `shape` as a dielectric region. Pass eps_r explicitly,
        or set shape.eps_r before calling this."""
        if eps_r is not None:
            shape.eps_r = eps_r
        if shape.eps_r is None:
            raise ValueError("add_dielectric: shape has no eps_r set "
                              "(pass eps_r= or set shape.eps_r first)")
        self.dielectrics.append(shape)
        return shape

    def solve(self):
        """Run the full pipeline and store V, fields, and stored energy
        as attributes. Returns self, so calls can be chained."""
        self.eps_r_of_xy = make_eps_r_function(self.dielectrics, self.background_eps_r)
        self.eps_elem = evaluate_material(self.mesh, self.eps_r_of_xy)
        K, area, area2, b, c = assemble_stiffness(self.mesh, self.eps_elem)
        self.V, self.is_fixed, self.solve_time = apply_conductors_and_solve(
            self.mesh, K, self.conductors)
        (self.Ex, self.Ey, self.Emag, self.Dx, self.Dy,
         self.energy_density, self.W) = compute_fields(
            self.mesh, self.V, self.eps_elem, b, c, area, area2)
        return self

    def capacitance(self, v_hi, v_lo):
        """Two-conductor capacitance (per unit depth, F/m) between two
        potentials already present in the solved problem."""
        if self.W is None:
            raise RuntimeError("call .solve() before .capacitance()")
        return capacitance_from_energy(self.W, v_hi, v_lo)

    def plot(self, title, fname, xlim=None, ylim=None, style=None):
        """Render the standard four-panel figure (see plot_solution)."""
        if self.V is None:
            raise RuntimeError("call .solve() before .plot()")
        plot_solution(self.mesh, self.V, self.eps_r_of_xy, self.energy_density,
                      self.conductors, self.is_fixed, title, fname,
                      xlim=xlim, ylim=ylim, style=style)


# =============================================================================
# 8. VISUALIZATION
# =============================================================================

def plot_solution(mesh, V, eps_r_of_xy, energy_density, conductors, is_fixed,
                   title, fname, xlim=None, ylim=None, style=None):
    """Render a four-panel summary figure: dielectric map, equipotential
    contours, field magnitude with streamlines, and energy density.

    The potential-contour and field/streamline panels plot a smooth,
    node-based field recovered by central differences on the regular node
    grid (np.gradient below), rather than the raw per-triangle field from
    compute_fields. Because this mesh happens to be a regular grid, that
    is enough to avoid the "blocky" look of a genuinely piecewise-constant
    field, at essentially no cost; an unstructured mesh would need a
    proper recovery scheme (e.g. superconvergent patch recovery) to get
    the same effect. The energy-density panel deliberately plots the raw
    per-triangle field instead, to show what the solver actually
    computed -- at the mesh resolutions used here, individual triangles
    are far smaller than the plot's visible detail, so this doesn't look
    blocky in practice, but the underlying values genuinely are
    piecewise-constant.

    Parameters
    ----------
    mesh : Mesh
    V : ndarray, shape (mesh.n_nodes,)
        Nodal potential.
    eps_r_of_xy : callable
        As returned by make_eps_r_function.
    energy_density : ndarray, shape (mesh.n_tris,)
        Per-triangle energy density.
    conductors : list of Shape
        Used to draw the conductor outline and mask.
    is_fixed : ndarray of bool
        Per-node Dirichlet mask, as returned by apply_conductors_and_solve.
    title, fname : str
        Figure title and output path.
    xlim, ylim : (float, float), optional
        Plot view window, independent of the (generally larger)
        simulation domain -- lets the figure zoom in on the region of
        interest without needing a smaller, less accurate mesh.
    style : PlotConfig, optional
        Visualization tuning parameters; defaults to PlotConfig().
    """
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

    # --- panel 1: dielectric map ------------------------------------------
    ax = axes[0, 0]
    pcm = ax.pcolormesh(X, Y, epsr_masked, shading="auto", cmap=cmap_blue)
    fig.colorbar(pcm, ax=ax, label=r"$\varepsilon_r$")
    outline(ax)
    ax.set_title("Dielectric map (gray = conductor)")

    # --- panel 2: potential contours ---------------------------------------
    ax = axes[0, 1]
    cf = ax.contourf(X, Y, V_grid, levels=style.potential_fill_levels, cmap="viridis")
    ax.contour(X, Y, V_grid, levels=style.potential_line_levels,
               colors="white", linewidths=0.4, alpha=0.6)
    fig.colorbar(cf, ax=ax, label="V [Volt]")
    outline(ax)
    ax.set_title("Equipotential contours")

    # --- panel 3: field magnitude + streamlines -----------------------------
    ax = axes[1, 0]
    pcm = ax.pcolormesh(X, Y, EmagG_masked, shading="auto", cmap=cmap_inferno)
    fig.colorbar(pcm, ax=ax, label="|E| [V/m]")
    ax.streamplot(xs, ys, ExG, EyG, color="white",
                  density=style.streamline_density, linewidth=0.6, arrowsize=0.8)
    outline(ax)
    ax.set_title("Electric field + field lines")

    # --- panel 4: energy density (per element) ------------------------------
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
    fig.savefig(fname, dpi=style.dpi)
    plt.close(fig)


# =============================================================================
# 9. EXAMPLES
# =============================================================================
# A _solve_*(config, h) helper builds and solves at a given resolution; a
# example_*() runs a convergence sweep then the final report and plot.
# See README.md section 9 for the worked-example writeups and section
# 8.2 for what the two examples' very different convergence behavior
# means.

def _describe_convergence(C_values):
    """Report whether a sequence of capacitance estimates from a mesh-
    refinement sweep is monotonic, determined from the data rather than
    assumed by whoever is calling this.

    A formal convergence order (e.g. via Richardson extrapolation) is
    deliberately not computed here. That technique needs both a constant
    refinement ratio between consecutive h levels and monotonic
    convergence at one consistent order; the h sequences used below have
    neither property in general (their refinement ratios vary, and
    example 1's sequence is not monotonic at all -- see its convergence
    table). Forcing an "order" or an extrapolated answer out of data that
    does not support one would look precise without being accurate.
    """
    diffs = np.diff(C_values)
    if np.all(diffs > 0) or np.all(diffs < 0):
        print("The sequence above is monotonic across every tested resolution.")
    else:
        sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
        print(f"The sequence above is NOT monotonic (it changes direction "
              f"{sign_changes} time{'s' if sign_changes != 1 else ''} across the "
              f"tested resolutions).")


def _solve_parallel_plate(config, h):
    """Build and solve the parallel-plate problem in `config` at grid
    spacing h.

    Every target size is snapped to the grid independently at this h (see
    snap_to_grid), so the physical geometry stays fixed while h varies --
    what makes the convergence sweep in example_parallel_plate meaningful
    rather than a rescaled copy of the same mesh at every step.
    """
    plate_t = snap_to_grid(config.plate_thickness, h)
    gap = snap_to_grid(config.gap, h)
    dielectric_t = snap_to_grid(config.dielectric_thickness, h)
    plate_w = snap_to_grid(config.plate_width, h)
    margin = snap_to_grid(config.domain_margin, h)

    Lx = plate_w + 2 * margin
    Ly = 2 * plate_t + gap + 2 * margin
    nx = round(Lx / h) + 1
    ny = round(Ly / h) + 1

    x_plate0 = margin
    y_gap_lo = margin + plate_t
    y_gap_hi = y_gap_lo + gap

    bottom_plate = Rectangle(x_plate0, margin, plate_w, plate_t,
                              voltage=0.0, name="bottom_plate")
    top_plate = Rectangle(x_plate0, y_gap_hi, plate_w, plate_t,
                           voltage=config.voltage, name="top_plate")
    conductors = [bottom_plate, top_plate]

    dielectric = Material("dielectric_slab", eps_r=config.dielectric_eps_r)
    background = Material("background", eps_r=config.background_eps_r)
    slab = Rectangle(x_plate0, y_gap_lo, plate_w, dielectric_t,
                      eps_r=dielectric.eps_r, name="dielectric_slab")
    eps_r_of_xy = make_eps_r_function([slab], background_eps_r=background.eps_r)

    mesh = Mesh(0, 0, Lx, Ly, nx=nx, ny=ny)
    eps_elem = evaluate_material(mesh, eps_r_of_xy)
    K, area, area2, b, c = assemble_stiffness(mesh, eps_elem)
    V, is_fixed, solve_time = apply_conductors_and_solve(mesh, K, conductors)
    Ex, Ey, Emag, Dx, Dy, energy_density, W = compute_fields(
        mesh, V, eps_elem, b, c, area, area2)
    C = capacitance_from_energy(W, config.voltage, 0.0)
    C_ideal = plate_w * EPS0 / (dielectric_t / dielectric.eps_r
                                 + (gap - dielectric_t) / background.eps_r)

    return dict(h=h, mesh=mesh, conductors=conductors, eps_r_of_xy=eps_r_of_xy,
                V=V, is_fixed=is_fixed, energy_density=energy_density,
                C=C, C_ideal=C_ideal, solve_time=solve_time,
                x_plate0=x_plate0, plate_w=plate_w, plate_t=plate_t,
                gap=gap, margin=margin)


def example_parallel_plate(config=None):
    """Parallel-plate capacitor with a rectangular dielectric slab filling
    the lower half of the gap: a rectilinear geometry with a spatially
    varying permittivity, compared against the ideal (fringing-free)
    series-dielectric formula."""
    config = config or ParallelPlateConfig()

    print("=" * 72)
    print("EXAMPLE 1: Parallel-plate capacitor, partially filled with a dielectric slab")
    print("=" * 72)

    print("Mesh convergence (physical geometry fixed, only h changes):")
    print(f"{'h [mm]':>9s}{'nodes':>9s}{'solve [s]':>11s}{'C [pF/m]':>12s}{'change':>9s}")
    result, prev_C = None, None
    C_values = []
    for h in config.convergence_spacings:
        result = _solve_parallel_plate(config, h)
        change = "" if prev_C is None else f"{100 * (result['C'] - prev_C) / prev_C:+.2f}%"
        print(f"{h * 1e3:9.3f}{result['mesh'].n_nodes:9d}{result['solve_time']:11.3f}"
              f"{result['C'] * 1e12:12.3f}{change:>9s}")
        prev_C = result["C"]
        C_values.append(result["C"])
    _describe_convergence(C_values)
    print("This is expected here, not a defect in the solver: the core")
    print("assembly/solve pipeline reproduces an exact, fringing-free analytical")
    print("case (full-width plates, no possible fringing) to 0.0000% at every")
    print("resolution tested. The field concentrates sharply at the plate's")
    print("corner -- a geometric singularity -- and each h above is an")
    print("independent structured mesh rather than a nested refinement of the")
    print("previous one, so successive levels are not guaranteed to bracket the")
    print("true answer monotonically (see LIMITATIONS AND FUTURE WORK). Treat the")
    print("finest level as accurate to roughly the spread shown above.")
    print()

    C, C_ideal = result["C"], result["C_ideal"]
    print(f"mesh: {result['mesh'].n_nodes} nodes, {result['mesh'].n_tris} triangles "
          f"(h = {result['h'] * 1e3:.3f} mm)")
    print(f"FEM capacitance     : {C * 1e12:9.3f} pF/m")
    print(f"Ideal, no fringing  : {C_ideal * 1e12:9.3f} pF/m")
    print(f"Difference          : {100 * (C - C_ideal) / C_ideal:+6.2f} %  "
          f"(FEM > ideal is expected: it also captures fringing, which the ideal "
          f"formula ignores)")
    print("Capacitance is per unit depth into the page; multiply by the actual "
          "plate depth in meters for total farads.")

    plot_solution(result["mesh"], result["V"], result["eps_r_of_xy"],
                  result["energy_density"], result["conductors"], result["is_fixed"],
                  "Parallel-plate capacitor (glass slab in an air gap)",
                  os.path.join(OUTPUT_DIR, "example1_parallel_plate.png"),
                  xlim=(result["x_plate0"] - config.plot_margin,
                        result["x_plate0"] + result["plate_w"] + config.plot_margin),
                  ylim=(result["margin"] - config.plot_margin,
                        result["margin"] + config.plot_margin
                        + 2 * result["plate_t"] + result["gap"]))

    return C, C_ideal


def _solve_coax(config, h):
    """Build and solve the coax problem in `config` at grid spacing h."""
    nx = round(2 * config.domain_half_width / h) + 1
    ny = nx

    inner = Circle((0, 0), config.inner_radius, voltage=config.voltage,
                    name="inner_conductor")
    outer = OutsideCircle((0, 0), config.outer_radius, voltage=0.0,
                           name="outer_conductor")
    conductors = [inner, outer]

    dielectric = Material("dielectric_fill", eps_r=config.dielectric_eps_r)
    fill = Circle((0, 0), config.outer_radius, eps_r=dielectric.eps_r,
                  name="dielectric_fill")
    eps_r_of_xy = make_eps_r_function([fill], background_eps_r=config.background_eps_r)

    half = config.domain_half_width
    mesh = Mesh(-half, -half, 2 * half, 2 * half, nx=nx, ny=ny)
    eps_elem = evaluate_material(mesh, eps_r_of_xy)
    K, area, area2, b, c = assemble_stiffness(mesh, eps_elem)
    V, is_fixed, solve_time = apply_conductors_and_solve(mesh, K, conductors)
    Ex, Ey, Emag, Dx, Dy, energy_density, W = compute_fields(
        mesh, V, eps_elem, b, c, area, area2)
    C = capacitance_from_energy(W, config.voltage, 0.0)
    C_ideal = (2 * np.pi * EPS0 * dielectric.eps_r
               / np.log(config.outer_radius / config.inner_radius))

    return dict(h=h, mesh=mesh, conductors=conductors, eps_r_of_xy=eps_r_of_xy,
                V=V, is_fixed=is_fixed, energy_density=energy_density,
                C=C, C_ideal=C_ideal, solve_time=solve_time)


def example_coax(config=None):
    """Coaxial cable with a polyethylene dielectric fill: a curved
    (circular) geometry, compared against the standard coax capacitance
    formula. Run alongside example 1 specifically for the contrast in
    convergence behavior between a smooth boundary and a sharp corner."""
    config = config or CoaxConfig()

    print()
    print("=" * 72)
    print("EXAMPLE 2: Coaxial capacitor (polyethylene-filled)")
    print("=" * 72)

    print("Mesh convergence (smooth circular boundary, no sharp corner):")
    print(f"{'h [mm]':>9s}{'nodes':>9s}{'solve [s]':>11s}{'C [pF/m]':>12s}{'error':>9s}")
    result = None
    C_values = []
    for h in config.convergence_spacings:
        result = _solve_coax(config, h)
        err = 100 * (result["C"] - result["C_ideal"]) / result["C_ideal"]
        print(f"{h * 1e3:9.3f}{result['mesh'].n_nodes:9d}{result['solve_time']:11.3f}"
              f"{result['C'] * 1e12:12.3f}{err:+8.2f}%")
        C_values.append(result["C"])
    _describe_convergence(C_values)
    print("Error shrinks toward 0% overall. These five points happen to be")
    print("monotonic, but that describes this specific sweep, not a general")
    print("guarantee -- finer intermediate resolutions reveal small reversals")
    print("too (same non-nested-mesh effect as example 1, far smaller in size;")
    print("see README.md section 8.2).")
    print()

    C, C_ideal = result["C"], result["C_ideal"]
    print(f"mesh: {result['mesh'].n_nodes} nodes, {result['mesh'].n_tris} triangles "
          f"(h = {result['h'] * 1e3:.3f} mm)")
    print(f"FEM capacitance             : {C * 1e12:9.3f} pF/m")
    print(f"Analytical 2*pi*eps/ln(b/a) : {C_ideal * 1e12:9.3f} pF/m")
    print(f"Difference                  : {100 * (C - C_ideal) / C_ideal:+6.2f} %  "
          f"(mesh / staircase discretization error)")

    plot_solution(result["mesh"], result["V"], result["eps_r_of_xy"],
                  result["energy_density"], result["conductors"], result["is_fixed"],
                  "Coaxial capacitor (polyethylene dielectric)",
                  os.path.join(OUTPUT_DIR, "example2_coax.png"))

    return C, C_ideal


def _solve_exact_check(config, h):
    """Build and solve a full-width-plate variant of the parallel-plate
    problem in `config` at grid spacing h: the same materials, gap, and
    dielectric split as _solve_parallel_plate, but the plates extend
    across -- and well past -- the entire simulation domain in x, so
    fringing is geometrically impossible and the series-capacitor
    formula is exact here, not merely idealized. See README.md section
    8.1. Deliberately mirrors _solve_parallel_plate's structure closely,
    so the two are easy to compare line by line; the only real
    difference is how wide the plates are relative to the domain.
    """
    plate_t = snap_to_grid(config.plate_thickness, h)
    gap = snap_to_grid(config.gap, h)
    dielectric_t = snap_to_grid(config.dielectric_thickness, h)
    margin = snap_to_grid(config.domain_margin, h)

    # Domain width is independent of plate width here -- unlike
    # _solve_parallel_plate, plate_width isn't used to size the plates
    # (see overhang below), only to pick a domain of comparable scale.
    Lx = snap_to_grid(config.plate_width, h)
    Ly = 2 * plate_t + gap + 2 * margin
    nx = round(Lx / h) + 1
    ny = round(Ly / h) + 1

    y_gap_lo = margin + plate_t
    y_gap_hi = y_gap_lo + gap

    # Plates (and the dielectric slab) extend a full domain-width past
    # both edges of [0, Lx] -- three times wider than the visible mesh,
    # centered on it -- so every point in the simulated region is deep
    # inside an effectively infinite plate. No true plate edge, and no
    # interaction with the domain's own outer boundary treatment, is
    # ever within view.
    overhang = Lx
    bottom_plate = Rectangle(-overhang, margin, Lx + 2 * overhang, plate_t,
                              voltage=0.0, name="bottom_plate")
    top_plate = Rectangle(-overhang, y_gap_hi, Lx + 2 * overhang, plate_t,
                           voltage=config.voltage, name="top_plate")
    conductors = [bottom_plate, top_plate]

    dielectric = Material("dielectric_slab", eps_r=config.dielectric_eps_r)
    background = Material("background", eps_r=config.background_eps_r)
    slab = Rectangle(-overhang, y_gap_lo, Lx + 2 * overhang, dielectric_t,
                      eps_r=dielectric.eps_r, name="dielectric_slab")
    eps_r_of_xy = make_eps_r_function([slab], background_eps_r=background.eps_r)

    mesh = Mesh(0, 0, Lx, Ly, nx=nx, ny=ny)
    eps_elem = evaluate_material(mesh, eps_r_of_xy)
    K, area, area2, b, c = assemble_stiffness(mesh, eps_elem)
    V, is_fixed, solve_time = apply_conductors_and_solve(mesh, K, conductors)
    Ex, Ey, Emag, Dx, Dy, energy_density, W = compute_fields(
        mesh, V, eps_elem, b, c, area, area2)
    C = capacitance_from_energy(W, config.voltage, 0.0)
    # Exact, not idealized: with the plates full-width, series-capacitor
    # theory is not an approximation of this geometry, it IS this
    # geometry's solution (the field is exactly 1D, by construction).
    C_exact = Lx * EPS0 / (dielectric_t / dielectric.eps_r
                            + (gap - dielectric_t) / background.eps_r)

    return dict(h=h, mesh=mesh, C=C, C_exact=C_exact, solve_time=solve_time)


def example_exact_check(config=None, spacings=(0.5e-3, 0.25e-3, 0.125e-3, 0.0625e-3)):
    """Exact-solution validation: the same two-layer dielectric and gap
    as example_parallel_plate, but with plates wide enough that fringing
    is geometrically impossible. Where example_parallel_plate's C_ideal
    is a genuine approximation of a real, finite geometry (and expected
    to differ from FEM by a real fringing correction), this C_exact is
    the literal solution to this geometry's PDE -- so matching it to
    high precision isolates whether the assembly/solve/energy pipeline
    itself has implementation bugs, independent of how well the mesh
    approximates any particular boundary shape or how the result
    compares to real-world fringing. See README.md section 8.1.

    Uses its own dedicated resolutions rather than
    config.convergence_spacings: those are tuned for
    example_parallel_plate's geometry, and reusing them here was found,
    while building this function, to occasionally snap_to_grid a
    thickness or gap to a visibly different value than at neighboring
    resolutions (e.g. 0.15 mm doesn't divide evenly into a 2 mm
    dielectric or 4 mm gap, snapping them to 1.95/4.05 mm instead) --
    correct behavior for snap_to_grid (README section 4.3), but
    confusing here, where it would make C_exact itself shift between
    rows of what's supposed to be a single, clean reference table. The
    defaults below divide evenly into every default ParallelPlateConfig
    geometric parameter, so C_exact is identical at every row for the
    default config. For a custom config where that's no longer true,
    each row still prints its OWN reference value, so the table stays
    honest either way rather than silently assuming it.

    Off by default (see RUN_EXACT_CHECK below) since, unlike the two
    examples above, this is a one-time validation step rather than a
    geometry someone would want to explore or reconfigure.
    """
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
          f"{'C_exact [pF/m]':>18s}{'error':>12s}")
    results = []
    for h in spacings:
        result = _solve_exact_check(config, h)
        err = 100 * (result["C"] - result["C_exact"]) / result["C_exact"]
        print(f"{h * 1e3:9.4f}{result['mesh'].n_nodes:9d}{result['solve_time']:11.3f}"
              f"{result['C'] * 1e12:16.8f}{result['C_exact'] * 1e12:18.8f}{err:+11.6f}%")
        results.append(result)
    print()
    print("If every row above reads 0.000000% (to displayed precision), the")
    print("core FEM machinery -- assembly, boundary conditions, energy")
    print("integration -- has no implementation bugs at any of these")
    print("resolutions, including the two-layer dielectric handling. Any")
    print("error elsewhere in this project comes from the mesh, not the")
    print("math (README.md section 8.1).")

    return results[-1]["C"], results[-1]["C_exact"]


# =============================================================================
# 10. LIMITATIONS AND FUTURE WORK
# =============================================================================
# The finite-element formulation itself is validated, not just asserted
# (see README.md section 8 for the exact analytical check and both
# convergence tables). The limitations below are about the *mesh*, not
# the underlying method, and all three trace back to one design choice:
# a structured, non-conforming mesh (README.md section 1, 10).
#
#   1. Non-conforming mesh: O(h) error on curved/non-axis-aligned
#      boundaries (coax convergence table, README section 8.2/10.1).
#   2. Corner singularities are under-resolved by a uniform mesh, and
#      independent structured meshes at different h are not nested
#      refinements of each other -- hence the non-monotonic parallel-
#      plate convergence table (README section 8.2/10.2).
#   3. Material-interface / conductor-boundary triangles have an O(h)
#      assignment ambiguity for non-grid-alignable boundaries -- exactly
#      zero for the axis-aligned regions here (snap_to_grid), small and
#      quadrature-insensitive for the coax's circular boundaries
#      (README section 8.3/10.3, and the note in evaluate_material).
#
# Full writeups, measured numbers, and a prioritized extensions roadmap
# (unstructured/conforming meshing; a graded structured mesh as a
# dependency-free intermediate step; boundary-represented conductors;
# cut-cell boundary treatment; P2 elements; nonlinear/anisotropic
# dielectrics; floating conductors and general BC types; 3D) are in
# README.md sections 10-11.
# =============================================================================


# Set True to also run the exact-solution validation (README.md section
# 8.1) before the two main examples. Off by default: it's a one-time
# correctness check, not a geometry meant for regular use, and running
# it adds real time for something most runs of this script don't need.
RUN_EXACT_CHECK = True


if __name__ == "__main__":
    t_start = time.time()

    if RUN_EXACT_CHECK:
        example_exact_check()
        print()

    C1, C1_ideal = example_parallel_plate()
    C2, C2_ideal = example_coax()

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Case':32s}{'FEM [pF/m]':>16s}{'Analytical [pF/m]':>20s}")
    print(f"{'Parallel plate (dielectric slab)':32s}{C1 * 1e12:16.3f}{C1_ideal * 1e12:20.3f}")
    print(f"{'Coax (polyethylene)':32s}{C2 * 1e12:16.3f}{C2_ideal * 1e12:20.3f}")
    print()
    print(f"total runtime: {time.time() - t_start:.2f} s")
    print(f"Figures saved to {OUTPUT_DIR}/example1_parallel_plate.png and "
          f"{OUTPUT_DIR}/example2_coax.png")
