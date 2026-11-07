"""Rotating-reference-frame LBM (Coriolis + centrifugal forcing) and its
classical verification anchors.

Context
-------
Lattice Boltzmann solvers written for ship / marine hydrodynamics almost always
live in an inertial frame.  Turbo-machinery (impellers, propellers in a
rotating duct), geophysical rotating flows (Ekman layers, Taylor columns) and
rotating-tank experiments, however, are most naturally formulated in a frame
rotating with angular velocity :math:`\\boldsymbol{\\Omega}`.  There the
momentum equation carries two pseudo-forces

.. math::

   \\frac{D\\mathbf{u}}{Dt} = -\\frac{\\nabla p}{\\rho}
       - 2\\boldsymbol{\\Omega}\\times\\mathbf{u}
       - \\boldsymbol{\\Omega}\\times(\\boldsymbol{\\Omega}\\times\\mathbf{r})
       + \\nu\\nabla^2\\mathbf{u},

the **Coriolis** and **centrifugal** accelerations.  This module supplies the
force terms, a body-force-ready D3Q19 stepper, and the analytic reference
solutions / post-processing used by the three verified benchmarks
``benchmarks/verified/rotating_ekman``, ``rotating_taylor_column`` and
``rotating_source_sink``.

Numerical scheme
----------------
* D3Q19 single-relaxation-time collision on the library equilibrium
  (:func:`tensorlbm.d3q19.equilibrium3d`).
* Forces enter through the second-order **Guo (2002)** scheme::

      F_i = (1 - 1/(2 tau)) w_i [ (c_i - u)/cs^2 + (c_i.u) c_i / cs^4 ] . F

  with the half-step ("Guo") velocity ``u* = sum_i c_i f_i + F/(2 rho)`` used
  both in the equilibrium and in the force term.  Because the Coriolis
  acceleration itself depends on the velocity, ``F`` is evaluated at ``u*``
  by a short fixed-point iteration (``n_guo_iter``, default 2); the exact
  solution of the linear system ``u = u_raw - (Omega x u)/rho`` is recovered in
  the limit of more iterations, and the residual at 2 iterations is
  ``O((Omega/rho)^3)`` — irrelevant for the regimes verified here.
* Walls: half-way bounce-back for no-slip solids, specular (mirror in the
  wall-normal direction) reflection for free-slip solids.  Both are applied
  through pre-computed *arrival masks*, so the whole step stays vectorised and
  device agnostic.
* Optional volumetric mass source/sink ``S(x)`` (mass per step per cell),
  added as ``S * w_i`` to the post-collision populations.  Since
  ``sum_i c_i w_i = 0`` this injects mass without momentum: continuity becomes
  ``d rho/dt + div(rho u) = S`` and the momentum equation is untouched.  This
  is the mechanism used by the rotating-tank source-sink benchmark.

Centrifugal force
-----------------
``Omega x (Omega x r)`` is a *gradient* field
(``-grad(Omega^2 r_perp^2 / 2)``), so for a constant-density flow it can be
absorbed exactly into a reduced pressure and has no effect on the velocity
field.  :func:`centrifugal_accel` is therefore provided for completeness (and
its curl is verified to vanish in the test-suite), while the stepper keeps the
term switchable (``centrifugal=True``) for genuine free-surface / stratified
problems where the reduced pressure is not a valid substitute.

References
----------
* Guo, Zheng & Shi, *Phys. Rev. E* **65**, 046308 (2002) — forcing scheme.
* Greenspan, *The Theory of Rotating Fluids* (1968) — Ekman spin-up, Taylor
  columns.
* Pedlosky, *Geophysical Fluid Dynamics* (1987) — Ekman spiral, eq. (3.5.x).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .d3q19 import C, W, equilibrium3d, macroscopic3d

__all__ = [
    "CS2",
    "RotatingFrameLBM",
    "axial_uniformity",
    "centrifugal_accel",
    "coriolis_accel",
    "ekman_delta",
    "ekman_profile",
    "fit_ekman_thickness",
    "fit_power_law",
    "guo_forcing_delta",
    "lattice_scaling",
    "radial_average",
    "spiral_phase_depth",
]

CS2 = 1.0 / 3.0


# --------------------------------------------------------------------------- #
# 1. Pseudo-forces                                                             #
# --------------------------------------------------------------------------- #
def _as_vec3(vec: Sequence[float] | torch.Tensor) -> tuple[float, float, float]:
    vals = [float(v) for v in vec]
    if len(vals) != 3:
        raise ValueError(f"expected a 3-component vector, got {vals!r}")
    return vals[0], vals[1], vals[2]


def coriolis_accel(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    omega: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coriolis acceleration ``a_c = -2 Omega x u`` (lattice units).

    Args:
        ux, uy, uz: velocity components, arbitrary common shape.
        omega: angular velocity of the rotating frame, ``(Ox, Oy, Oz)``.

    Returns:
        ``(ax, ay, az)``, each of the same shape as the inputs.
    """
    ox, oy, oz = _as_vec3(omega)
    # Omega x u
    cx = oy * uz - oz * uy
    cy = oz * ux - ox * uz
    cz = ox * uy - oy * ux
    return -2.0 * cx, -2.0 * cy, -2.0 * cz


def centrifugal_accel(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor | None = None,
    omega: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Centrifugal acceleration ``a_cf = -Omega x (Omega x r) = +Omega^2 r_perp``.

    ``r`` is measured from the axis of rotation (the origin).  For
    ``Omega = (0, 0, Oz)`` the result is ``(Oz^2 x, Oz^2 y, 0)``.

    The field is curl-free, hence representable as ``grad(Omega^2 r_perp^2/2)``
    and absorbable into a reduced pressure (see module docstring).
    """
    ox, oy, oz = _as_vec3(omega)
    if z is None:
        z = torch.zeros_like(x)
    # Omega x (Omega x r) = Omega (Omega.r) - r (Omega.Omega)
    om_dot_r = ox * x + oy * y + oz * z
    om_sq = ox * ox + oy * oy + oz * oz
    return (
        -(ox * om_dot_r - x * om_sq),
        -(oy * om_dot_r - y * om_sq),
        -(oz * om_dot_r - z * om_sq),
    )


def guo_forcing_delta(
    Fx: torch.Tensor,
    Fy: torch.Tensor,
    Fz: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    tau: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Guo (2002) second-order force term for D3Q19, shape ``(19, *grid)``.

    ::

        dF_i = (1 - 1/(2 tau)) w_i [ (c_i - u)/cs^2 + (c_i.u) c_i / cs^4 ] . F

    Bit-compatible with :func:`tensorlbm.free_surface_common.guo_force_delta_3d`
    (cross-checked in ``tests/test_rotating_frame.py``).
    """
    if device is None:
        device = ux.device
    c = C.to(device=device, dtype=dtype)  # (19, 3)
    w = W.to(device=device, dtype=dtype).view(19, 1, 1, 1)
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)

    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0) + cz * uz.unsqueeze(0)
    cF = cx * Fx.unsqueeze(0) + cy * Fy.unsqueeze(0) + cz * Fz.unsqueeze(0)
    uF = (ux * Fx + uy * Fy + uz * Fz).unsqueeze(0)

    inv_cs2 = 1.0 / CS2
    inv_cs4 = inv_cs2 * inv_cs2
    delta = cF * inv_cs2 - uF * inv_cs2 + cu * cF * inv_cs4
    return (1.0 - 1.0 / (2.0 * tau)) * w * delta


# --------------------------------------------------------------------------- #
# 2. Analytic reference solutions                                              #
# --------------------------------------------------------------------------- #
def ekman_delta(nu: float, f: float) -> float:
    """Ekman-layer thickness ``delta_E = sqrt(2 nu / f)``.

    ``f = 2 Omega`` is the Coriolis parameter.  Written in terms of the frame
    rotation, ``delta_E = sqrt(nu / Omega)``.  The campaign formula
    ``sqrt(2 nu / Omega_campaign)`` is recovered identically when
    ``Omega_campaign`` denotes the Coriolis parameter ``f`` (see
    ``benchmarks/verified/rotating_ekman/README.md``).
    """
    if nu <= 0 or f <= 0:
        raise ValueError("nu and f must be positive")
    return math.sqrt(2.0 * nu / f)


def ekman_profile(
    zeta,
    U_g: float = 1.0,
    nu: float | None = None,
    f: float | None = None,
):
    """Classic Ekman spiral in the frame rotating with the fluid.

    Steady, horizontally homogeneous, linearised boundary layer above a
    no-slip plate that is at rest in the rotating frame.  The far field is a
    geostrophic flow ``U_g`` along ``x``.  With ``zeta = z/delta_E``:

    .. math::

       u_\\parallel(\\zeta) &= U_g\\,[1 - e^{-\\zeta}\\cos\\zeta],\\\\
       u_\\perp(\\zeta)     &= U_g\\,e^{-\\zeta}\\sin\\zeta .

    ``u_parallel`` is the component along the geostrophic (in rotating-disk
    language: azimuthal) direction, ``u_perp`` the cross-isobar (radial)
    component.  Both are returned in lattice/physical units of ``U_g``.

    Args:
        zeta: normalised height(s) above the plate, array-like or float.
        U_g: geostrophic velocity.
        nu, f: optional; if both given, ``zeta`` is interpreted as *height*
            ``z`` and normalised internally.
    """
    if nu is not None and f is not None:
        zeta = _to_tensor(zeta) / ekman_delta(nu, f)
    z = _to_tensor(zeta)
    decay = torch.exp(-z)
    return U_g * (1.0 - decay * torch.cos(z)), U_g * decay * torch.sin(z)


def spiral_phase_depth(nu: float, f: float, phase: float = math.pi) -> float:
    """Depth at which the Ekman velocity vector has turned by ``phase``.

    The spiral turns by ``zeta`` radians, so the depth where the velocity is
    first *parallel* to the geostrophic vector (one full turn of the hodograph)
    is ``z = pi delta_E``.
    """
    return phase * ekman_delta(nu, f)


def _to_tensor(val) -> torch.Tensor:
    if isinstance(val, torch.Tensor):
        return val.to(torch.float64)
    return torch.as_tensor(val, dtype=torch.float64)


def fit_ekman_thickness(
    z: torch.Tensor | Sequence[float],
    u_parallel: torch.Tensor | Sequence[float],
    u_perp: torch.Tensor | Sequence[float],
    *,
    delta_bounds: tuple[float, float] | None = None,
    n_scan: int = 400,
) -> dict:
    """Least-squares fit of ``delta_E`` and ``U_g`` to the Ekman spiral.

    For every trial ``delta`` the model is *linear* in ``U_g``
    (``u_par = U_g A(z), u_perp = U_g B(z)`` with ``A = 1-e^{-zeta}cos zeta``,
    ``B = e^{-zeta}sin zeta``), so the optimal ``U_g`` is closed-form and only
    a 1-D scan + golden-section refinement over ``delta`` is needed.

    Returns a dict with ``delta``, ``U_g``, ``residual_rms``,
    ``delta_bounds`` and the sampled scan curve (for diagnostics/figures).
    """
    z = _to_tensor(z).flatten()
    up = _to_tensor(u_parallel).flatten()
    uq = _to_tensor(u_perp).flatten()
    if z.numel() != up.numel() or z.numel() != uq.numel():
        raise ValueError("z, u_parallel and u_perp must have the same length")
    if delta_bounds is None:
        zmax = float(z.max())
        delta_bounds = (zmax / 40.0, zmax / 0.4)

    def cost(delta: float) -> tuple[float, float, torch.Tensor]:
        zeta = z / delta
        a = 1.0 - torch.exp(-zeta) * torch.cos(zeta)
        b = torch.exp(-zeta) * torch.sin(zeta)
        denom = torch.sum(a * a + b * b)
        u_g = torch.sum(a * up + b * uq) / denom
        resid = (up - u_g * a) ** 2 + (uq - u_g * b) ** 2
        return float(torch.sqrt(torch.mean(resid))), float(u_g), resid

    lo, hi = delta_bounds
    grid = torch.linspace(lo, hi, n_scan, dtype=torch.float64)
    costs = [cost(float(d))[0] for d in grid]
    k = int(min(range(len(costs)), key=lambda i: costs[i]))

    # golden-section refinement around the best grid point
    step = float(grid[1] - grid[0])
    a_lo = float(grid[max(k - 1, 0)])
    a_hi = float(grid[min(k + 1, n_scan - 1)])
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c1 = a_hi - gr * (a_hi - a_lo)
    c2 = a_lo + gr * (a_hi - a_lo)
    f1, f2 = cost(c1)[0], cost(c2)[0]
    for _ in range(80):
        if abs(a_hi - a_lo) < 1e-6 * max(1.0, step):
            break
        if f1 < f2:
            a_hi, c2, f2 = c2, c1, f1
            c1 = a_hi - gr * (a_hi - a_lo)
            f1 = cost(c1)[0]
        else:
            a_lo, c1, f1 = c1, c2, f2
            c2 = a_lo + gr * (a_hi - a_lo)
            f2 = cost(c2)[0]
    delta = 0.5 * (a_lo + a_hi)
    rms, u_g, _ = cost(delta)
    return {
        "delta": delta,
        "U_g": u_g,
        "residual_rms": rms,
        "delta_bounds": [lo, hi],
        "scan_delta": [float(v) for v in grid],
        "scan_cost": costs,
    }


# --------------------------------------------------------------------------- #
# 3. Post-processing helpers                                                   #
# --------------------------------------------------------------------------- #
def fit_power_law(r: torch.Tensor | Sequence[float], v: torch.Tensor | Sequence[float]) -> dict:
    """Fit ``v = a r^n`` by linear least squares on ``(log r, log v)``.

    Returns ``{"exponent": n, "coefficient": a, "r2": ...}``.  ``r2`` is the
    coefficient of determination of the log-log fit.
    """
    r = _to_tensor(r).flatten()
    v = _to_tensor(v).flatten()
    if r.numel() < 3:
        raise ValueError("need at least 3 points for a power-law fit")
    if bool(torch.any(r <= 0)) or bool(torch.any(v <= 0)):
        raise ValueError("power-law fit requires strictly positive values")
    x = torch.log(r)
    y = torch.log(v)
    n = x.numel()
    sx, sy = x.sum(), y.sum()
    sxx, sxy = (x * x).sum(), (x * y).sum()
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    pred = slope * x + intercept
    ss_res = float(torch.sum((y - pred) ** 2))
    ss_tot = float(torch.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "exponent": float(slope),
        "coefficient": float(torch.exp(intercept)),
        "r2": r2,
        "n_points": int(n),
    }


def radial_average(
    field: torch.Tensor,
    r: torch.Tensor,
    mask: torch.Tensor,
    r_edges: Sequence[float],
) -> dict:
    """Shell-average a scalar/vector field over annuli of constant ``r``.

    Args:
        field: ``(nz, ny, nx)`` or ``(k, nz, ny, nx)`` sample values.
        r: ``(nz, ny, nx)`` radius of each cell from the axis.
        mask: ``(nz, ny, nx)`` boolean, cells to include.
        r_edges: monotonically increasing shell edges.

    Returns:
        ``{"r": [...], "mean": [[...]]/[...], "count": [...]}`` where ``mean``
        keeps the leading axis when ``field`` is stacked.
    """
    stacked = field.ndim == r.ndim + 1
    f = field if stacked else field.unsqueeze(0)
    edges = torch.as_tensor(list(r_edges), dtype=r.dtype)
    centres, means, counts = [], [], []
    for i in range(len(r_edges) - 1):
        sel = mask & (r >= edges[i]) & (r < edges[i + 1])
        n = int(sel.sum())
        counts.append(n)
        centres.append(float(0.5 * (edges[i] + edges[i + 1])))
        if n == 0:
            means.append([float("nan")] * f.shape[0])
        else:
            means.append([float(f[k][sel].mean()) for k in range(f.shape[0])])
    transposed = [list(col) for col in zip(*means, strict=False)]
    return {
        "r": centres,
        "mean": transposed if stacked else transposed[0],
        "count": counts,
    }


def axial_uniformity(field: torch.Tensor, mask: torch.Tensor, *, eps: float = 1e-12) -> dict:
    """Max relative axial (``z``) variation of a field inside ``mask``.

    ``field`` may be ``(nz, ny, nx)``; the mean over ``z`` is taken per column
    ``(y, x)`` and the deviation ``|f - fbar| / max|fbar|`` is maximised over
    the masked cells.  For a Taylor-Proudman column this is the quantity that
    must stay small.
    """
    f = field.to(torch.float64)
    m = mask.bool()
    if int(m.sum()) == 0:
        raise ValueError("mask selects no cells")
    fz = torch.where(m, f, torch.zeros_like(f))
    cnt = m.sum(dim=0).clamp(min=1)
    fbar = fz.sum(dim=0) / cnt  # (ny, nx)
    scale = float(fbar[m.any(dim=0)].abs().max()) if bool(m.any(dim=0).any()) else 0.0
    scale = max(scale, eps)
    dev = torch.where(m, (f - fbar.unsqueeze(0)).abs(), torch.zeros_like(f))
    return {
        "max_rel_dev": float(dev.max()) / scale,
        "rms_rel_dev": float(torch.sqrt((dev * dev).sum() / cnt.sum())) / scale,
        "columns": int(m.any(dim=0).sum()),
    }


def lattice_scaling(nu_phys: float, f_phys: float, delta_nodes: float, nu_lat: float) -> dict:
    """Lattice <-> physical mapping for a rotating-flow case.

    Fixes the Ekman thickness to ``delta_nodes`` lattice nodes:

    * ``dx = delta_phys / delta_nodes`` with ``delta_phys = sqrt(2 nu/f)``
    * ``dt = nu_lat dx^2 / nu_phys``
    * ``f_lat = 2 nu_lat / delta_nodes^2``  and ``Omega_lat = f_lat / 2``
    * consistency check: ``f_phys * dt`` must equal ``f_lat``.

    Returns a dict with all of ``dx, dt, nu_lat, f_lat, omega_lat,
    delta_phys, delta_nodes, f_phys, omega_phys, U_from_lat_scale``.
    """
    delta_phys = ekman_delta(nu_phys, f_phys)
    dx = delta_phys / delta_nodes
    dt = nu_lat * dx * dx / nu_phys
    f_lat = 2.0 * nu_lat / (delta_nodes * delta_nodes)
    return {
        "dx": dx,
        "dt": dt,
        "nu_lat": nu_lat,
        "f_lat": f_lat,
        "omega_lat": 0.5 * f_lat,
        "delta_phys": delta_phys,
        "delta_nodes": delta_nodes,
        "f_phys": f_phys,
        "omega_phys": 0.5 * f_phys,
        "f_lat_check": f_phys * dt,
        "length_per_node": dx,
        "time_per_step": dt,
        "velocity_scale": dx / dt,
    }


# --------------------------------------------------------------------------- #
# 4. The stepper                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class _Reflection:
    """Pre-computed boundary bookkeeping (masks + reflection tables)."""

    reflect: torch.Tensor  # (19,) index map applied on slip solids
    opposite: torch.Tensor  # (19,) index map applied on no-slip solids
    mask_solid: torch.Tensor  # (19, nz, ny, nx) arrival came from a solid cell
    mask_slip: torch.Tensor  # (19, nz, ny, nx) ... from a free-slip solid cell
    solid: torch.Tensor  # (nz, ny, nx) bool


def _mirror_z_table() -> torch.Tensor:
    """Permutation that flips ``c_z`` and keeps ``c_x, c_y``."""
    table = torch.empty(len(C), dtype=torch.int64)
    key = {(int(cx), int(cy), int(cz)): i for i, (cx, cy, cz) in enumerate(C.tolist())}
    for i, (cx, cy, cz) in enumerate(C.tolist()):
        table[i] = key[(int(cx), int(cy), int(-cz))]
    return table


def _opposite_table() -> torch.Tensor:
    table = torch.empty(len(C), dtype=torch.int64)
    key = {(int(cx), int(cy), int(cz)): i for i, (cx, cy, cz) in enumerate(C.tolist())}
    for i, (cx, cy, cz) in enumerate(C.tolist()):
        table[i] = key[(-int(cx), -int(cy), -int(cz))]
    return table


class RotatingFrameLBM:
    """D3Q19 BGK LBM in a frame rotating with ``omega``.

    The grid layout is the library convention ``f.shape == (19, nz, ny, nx)``
    with the first array axis being ``z``.

    Args:
        shape: ``(nz, ny, nx)``.
        tau: BGK relaxation time (``nu = (tau - 1/2)/3``).
        omega: frame angular velocity ``(Ox, Oy, Oz)`` in lattice units.
        body_force: constant specific body force ``(Fx, Fy, Fz)`` (per unit
            mass; equivalent to an imposed pressure gradient), e.g. the
            geostrophic forcing of the Ekman benchmark.
        solid: ``(nz, ny, nx)`` boolean, ``True`` where cells are solid.
        slip_solid: optional mask of solid cells that are free-slip
            (specular reflection); all other solid cells are no-slip.
        mass_source: optional ``(nz, ny, nx)`` mass source term ``S`` (mass
            per step per cell, may be negative for a sink).
        centrifugal: include the centrifugal pseudo-force explicitly (needs
            ``centrifugal_origin``); default ``False`` because it is a gradient
            field and is absorbed into the reduced pressure.
        centrifugal_origin: grid origin ``(x0, y0)`` in cells used as the
            rotation axis for the centrifugal term.
        n_guo_iter: fixed-point iterations for the Guo velocity.
        dtype, device: torch placement.
        rho0: initial density (uniform equilibrium at rest).
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        tau: float,
        omega: Sequence[float] = (0.0, 0.0, 0.0),
        body_force: Sequence[float] = (0.0, 0.0, 0.0),
        solid: torch.Tensor | None = None,
        slip_solid: torch.Tensor | None = None,
        mass_source: torch.Tensor | None = None,
        centrifugal: bool = False,
        centrifugal_origin: tuple[float, float] = (0.0, 0.0),
        n_guo_iter: int = 2,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        rho0: float = 1.0,
        u0: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        if len(shape) != 3:
            raise ValueError("shape must be (nz, ny, nx)")
        if tau <= 0.5:
            raise ValueError("tau must exceed 1/2")
        self.shape = tuple(int(s) for s in shape)
        self.tau = float(tau)
        self.nu = (self.tau - 0.5) / 3.0
        self.omega = _as_vec3(omega)
        self.body_force = _as_vec3(body_force)
        self.n_guo_iter = int(n_guo_iter)
        self.dtype = dtype
        self.device = device if device is not None else torch.device("cpu")
        self.centrifugal = bool(centrifugal)
        self.rho0 = float(rho0)
        self.steps = 0

        nz, ny, nx = self.shape
        zero = torch.zeros(self.shape, dtype=dtype, device=self.device)
        self._f = equilibrium3d(
            torch.full_like(zero, self.rho0),
            torch.full_like(zero, float(u0[0])),
            torch.full_like(zero, float(u0[1])),
            torch.full_like(zero, float(u0[2])),
        )

        if solid is None:
            solid = torch.zeros(self.shape, dtype=torch.bool, device=self.device)
        self.solid = solid.to(device=self.device, dtype=torch.bool)
        if slip_solid is None:
            slip_solid = torch.zeros_like(self.solid)
        slip = slip_solid.to(device=self.device, dtype=torch.bool) & self.solid

        self.mass_source = None
        if mass_source is not None:
            ms = mass_source.to(device=self.device, dtype=dtype)
            ms = torch.where(self.solid, torch.zeros_like(ms), ms)
            self.mass_source = ms

        self._boundary = self._build_reflection(slip)

        self._cx = self._centrifugal_fields(centrifugal_origin) if self.centrifugal else None

    def initialize_from_velocity(
        self, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor, rho: float | None = None
    ) -> None:
        """Reset the populations to the local equilibrium of a given field.

        Used by the benchmarks to start from an analytic profile (fixed-point
        test) as well as from rest.
        """
        r = self.rho0 if rho is None else rho
        rho_t = torch.full_like(ux, float(r))
        self._f = equilibrium3d(rho_t, ux.to(self.dtype), uy.to(self.dtype), uz.to(self.dtype))

    # -- pre-computation --------------------------------------------------- #
    def _build_reflection(self, slip: torch.Tensor) -> _Reflection:
        cz = C[:, 2].to(torch.int64)
        cy = C[:, 1].to(torch.int64)
        cx = C[:, 0].to(torch.int64)
        masks, slips = [], []
        for i in range(len(C)):
            shift = (int(cz[i]), int(cy[i]), int(cx[i]))
            m = torch.roll(self.solid, shifts=shift, dims=(0, 1, 2))
            masks.append(m)
            slips.append(torch.roll(slip, shifts=shift, dims=(0, 1, 2)))
        return _Reflection(
            reflect=_mirror_z_table(),
            opposite=_opposite_table(),
            mask_solid=torch.stack(masks),
            mask_slip=torch.stack(slips),
            solid=self.solid,
        )

    def _centrifugal_fields(self, origin: tuple[float, float]):
        nz, ny, nx = self.shape
        y = torch.arange(ny, dtype=self.dtype, device=self.device).view(1, ny, 1) - origin[1]
        x = torch.arange(nx, dtype=self.dtype, device=self.device).view(1, 1, nx) - origin[0]
        xx = x.expand(nz, ny, nx).contiguous()
        yy = y.expand(nz, ny, nx).contiguous()
        zz = torch.zeros_like(xx)
        return centrifugal_accel(xx, yy, zz, self.omega)

    # -- streaming --------------------------------------------------------- #
    def _stream_with_walls(self, f_post: torch.Tensor) -> torch.Tensor:
        arrive = torch.stack(
            [
                torch.roll(
                    f_post[i],
                    shifts=(int(C[i, 2]), int(C[i, 1]), int(C[i, 0])),
                    dims=(0, 1, 2),
                )
                for i in range(len(C))
            ]
        )
        b = self._boundary
        reflected = f_post[b.reflect]  # specular (free-slip) branch
        bounced = f_post[b.opposite]  # no-slip branch
        return torch.where(
            b.mask_solid,
            torch.where(b.mask_slip, reflected, bounced),
            arrive,
        )

    # -- one time step ------------------------------------------------------ #
    def _specific_force(
        self,
        ux: torch.Tensor,
        uy: torch.Tensor,
        uz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Total specific force at a given velocity: body + centrifugal + Coriolis."""
        bx, by, bz = self.body_force
        Fx = torch.full_like(ux, bx)
        Fy = torch.full_like(ux, by)
        Fz = torch.full_like(ux, bz)
        if self._cx is not None:
            Fx = Fx + self._cx[0]
            Fy = Fy + self._cx[1]
            Fz = Fz + self._cx[2]
        if self.omega != (0.0, 0.0, 0.0):
            ax, ay, az = coriolis_accel(ux, uy, uz, self.omega)
            Fx = Fx + ax
            Fy = Fy + ay
            Fz = Fz + az
        return Fx, Fy, Fz

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            rho, ux_raw, uy_raw, uz_raw = macroscopic3d(self._f)

            # Guo (2002) half-step velocity: u* = u_raw + F(u*)/(2 rho).
            # The fixed point is reached by iterating from u_raw; the -1/2 tau
            # splitting of the force is what makes the scheme second order in
            # dt, so getting this right matters (a doubled correction inflates
            # the force by 3/8 and shrinks the Ekman thickness by ~15 %).
            inv_rho = 1.0 / rho
            Fx, Fy, Fz = self._specific_force(ux_raw, uy_raw, uz_raw)
            for _it in range(max(0, self.n_guo_iter - 1)):
                ux = ux_raw + 0.5 * Fx * inv_rho
                uy = uy_raw + 0.5 * Fy * inv_rho
                uz = uz_raw + 0.5 * Fz * inv_rho
                Fx, Fy, Fz = self._specific_force(ux, uy, uz)
            ux = ux_raw + 0.5 * Fx * inv_rho
            uy = uy_raw + 0.5 * Fy * inv_rho
            uz = uz_raw + 0.5 * Fz * inv_rho

            feq = equilibrium3d(rho, ux, uy, uz)
            f_post = self._f + (feq - self._f) * (1.0 / self.tau)
            f_post = f_post + guo_forcing_delta(
                Fx, Fy, Fz, ux, uy, uz, self.tau, dtype=self.dtype, device=self.device
            )
            if self.mass_source is not None:
                f_post = f_post + self.mass_source.unsqueeze(0) * W.to(
                    device=self.device, dtype=self.dtype
                ).view(-1, 1, 1, 1)
            # no dynamics inside solids
            f_post = torch.where(self.solid.unsqueeze(0), self._f, f_post)
            self._f = self._stream_with_walls(f_post)
            self.steps += 1

    # -- observables -------------------------------------------------------- #
    def fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(rho, ux, uy, uz)`` (garbage inside solids — mask it)."""
        return macroscopic3d(self._f)

    def velocity(self) -> torch.Tensor:
        """Stacked velocity ``(3, nz, ny, nx)``."""
        _, ux, uy, uz = macroscopic3d(self._f)
        return torch.stack([ux, uy, uz])

    def run_until_steady(
        self,
        tol: float = 1e-7,
        max_steps: int = 200_000,
        check_every: int = 200,
        min_steps: int = 0,
        period: int | None = None,
    ) -> dict:
        """Integrate until the (period-averaged) velocity stops changing.

        With ``period=None`` the residual is ``max|u(t) - u(t-check_every)|``.
        Rotating flows, however, are dominated by *inertial oscillations* at
        ``f = 2 Omega``: the instantaneous residual then has a floor of
        ``|u'| f dt`` and a strict tolerance would force meaningless
        over-integration.  Passing ``period`` (steps of one inertial period,
        ``2*pi/Omega``) compares the state with the one a full period earlier,
        which is the physically meaningful "steady state" criterion for a
        rotating flow (nett spin-up balance).

        Returns a convergence record (steps, residual history) so every
        benchmark can state the residual it converged to instead of claiming
        "converged" by assertion.
        """
        fluid = ~self.solid
        history: list[dict] = []
        residual = float("inf")
        if period is None:
            prev = self.velocity().clone()
            while self.steps < max_steps:
                self.step(check_every)
                cur = self.velocity()
                residual = float((cur - prev).abs()[:, fluid].max())
                history.append({"steps": self.steps, "max_du": residual})
                prev = cur
                if residual < tol and self.steps >= min_steps:
                    break
        else:
            period = max(int(period), 2)
            check_every = max(1, int(round(period / 8)))
            buf: list[tuple[int, torch.Tensor]] = [(self.steps, self.velocity().clone())]
            while self.steps < max_steps:
                self.step(check_every)
                cur = self.velocity()
                buf.append((self.steps, cur.clone()))
                target = self.steps - period
                # drop snapshots that are older than one period
                while len(buf) > 1 and buf[1][0] <= target:
                    buf.pop(0)
                old_steps, old = buf[0]
                residual = float((cur - old).abs()[:, fluid].max())
                history.append({"steps": self.steps, "compare_to": old_steps, "max_du": residual})
                if residual < tol and self.steps >= max(min_steps, period):
                    break
        return {
            "steps": self.steps,
            "residual": residual,
            "tol": tol,
            "period": period,
            "converged": residual < tol,
            "history": history,
        }
