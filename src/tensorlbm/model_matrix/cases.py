"""Analytic verification cases for the model matrix.

Every case is *lattice-generic* and *model-agnostic*: it needs only the
repo-native ``equilibrium``/``macroscopic``/``stream`` trio of the model's
lattice, so one implementation can be applied to every registry entry.  Each
case carries its own reference, its own tolerance and its own enforcement
level; none of them inspects the model internals.

Cases (all periodic, no boundary condition -- so a failure is attributable to
the collision operator alone):

``feq_fixed_point``  (:class:`FeqFixedPoint`)
    The discrete Maxwell-Boltzmann equilibrium is a fixed point of *every*
    collision operator and the collision conserves mass and momentum exactly.
    Reference = exact algebraic identity (no numerical tolerance on physics):
    ``||C(f_eq) - f_eq||_inf`` and ``|rho' - rho|``, ``|u' - u|`` must be at
    round-off level.  Catches operators that mis-handle the conserved moments
    or that secretly change the equation of state.

``shear_decay``  (:class:`ShearDecay`)
    Transverse shear wave ``u_x = U0 cos(k y)`` in a periodic box decays as
    ``U0 exp(-nu k^2 t)`` with ``nu = (tau - 1/2)/3`` -- a closed-form solution
    of the incompressible Navier-Stokes equations and the classical
    "viscosity of a collision operator" check (used e.g. by the Palabos/TCLB
    regression suites).  Metric = fitted decay rate vs the analytic value.
    This is the gate: the molecular viscosity of a model is a property of its
    definition, so a deviation larger than the tolerance is a defect, not a
    numerical accident.

``tg_decay``  (:func:`run_taylor_green_decay`)
    Taylor-Green vortex at ``U0 = 0.1`` (Ma ~ 0.17): ``u = U0 (sin kx cos ky,
    -cos kx sin ky)``, kinetic energy decays as ``E0 exp(-4 nu k^2 t)`` in the
    incompressible limit (the field carries modes (k,k), so nu*lap u = -2 nu k^2 u).  Unlike the shear wave -- where ``(u.grad)u = 0``
    identically, so *no* Mach dependence can be measured -- the TG vortex has a
    genuine ``O(Ma^2)`` compressibility error, which is why it is the
    *advisory* case: the deviation is recorded and reported but does not fail
    the build.  (This replaces the originally planned finite-Mach shear wave,
    whose two Mach numbers returned identical viscosities to 6 digits -- see
    `docs/MODEL_MATRIX_CASE_SELECTION.md`.)

Nothing in this module hard-codes a benchmark result: references are formulas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .registry import (
    ModelSpec,
    build_collider,
    bulk_viscosity,
    lattice_ops,
    nu_from_tau,
)

CS = 1.0 / math.sqrt(3.0)
"""Lattice speed of sound (isothermal LBM), units of c = dx/dt."""


@dataclass(frozen=True)
class CaseResult:
    case: str
    model: str
    status: str
    """PASS / FAIL / GUARD / NOT_APPLICABLE / MISSING / ERROR."""
    metrics: dict[str, float]
    ref: dict[str, Any]
    tolerance: dict[str, Any]
    notes: str = ""
    error: str = ""
    curves: dict[str, list[float]] | None = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "case": self.case,
            "model": self.model,
            "status": self.status,
            "metrics": self.metrics,
            "ref": self.ref,
            "tolerance": self.tolerance,
            "notes": self.notes,
        }
        if self.error:
            d["error"] = self.error
        if self.curves is not None:
            d["curves"] = self.curves
        return d


# --------------------------------------------------------------------------- #
#  Geometry / common machinery
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Grid:
    """Periodic box; ``nz = 1`` keeps the 3-D lattices cheap."""

    nx: int = 32
    ny: int = 32
    nz: int = 1

    def shape(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    @property
    def k(self) -> float:
        """Fundamental wavenumber of the shear wave along y."""
        return 2.0 * math.pi / self.ny


def _shear_field(spec: ModelSpec, grid: Grid, u0: float, dtype: torch.dtype) -> torch.Tensor:
    ops = lattice_ops(spec.lattice)
    shape = grid.shape()
    y = torch.arange(grid.ny, dtype=dtype).view(1, grid.ny, 1)
    cos = torch.cos(grid.k * y)
    ux = u0 * cos.expand(shape).contiguous()
    zero = torch.zeros(shape, dtype=dtype)
    rho = torch.ones(shape, dtype=dtype)
    return ops["equilibrium"](rho, ux, zero, zero).to(dtype)


def _amplitude(f: torch.Tensor, grid: Grid, spec: ModelSpec) -> float:
    """Project the velocity field onto the cos(k y) mode (mode amplitude)."""
    ops = lattice_ops(spec.lattice)
    _rho, ux, _uy, _uz = ops["macroscopic"](f)
    y = torch.arange(grid.ny, dtype=ux.dtype).view(1, grid.ny, 1)
    cos = torch.cos(grid.k * y)
    return float(2.0 * (ux * cos).mean())


def _mass(f: torch.Tensor, spec: ModelSpec) -> float:
    rho, _ux, _uy, _uz = lattice_ops(spec.lattice)["macroscopic"](f)
    return float(rho.mean())


def _fit_decay_rate(times: list[float], amps: list[float], floor_ratio: float) -> tuple[float, float]:
    """Least-squares slope of ``ln|A|`` vs ``t`` over the useful window.

    Returns ``(rate, r2)`` where ``rate`` is the *decay* rate (positive).  The
    window starts at the first sample (amplitude ``U0``) and stops when the
    amplitude has decayed to ``floor_ratio * U0``, so the fit never leans on
    round-off noise of an almost-dead wave.
    """
    if not times or not amps:
        raise ValueError("empty decay series")
    a0 = abs(amps[0])
    keep = [(t, a) for t, a in zip(times, amps) if abs(a) > floor_ratio * a0]
    if len(keep) < 3:
        raise ValueError(f"decay window too short ({len(keep)} samples)")
    ts = torch.tensor([t for t, _ in keep], dtype=torch.float64)
    ls = torch.log(torch.tensor([abs(a) for _, a in keep], dtype=torch.float64))
    tm, lm = ts.mean(), ls.mean()
    slope = float(((ts - tm) * (ls - lm)).sum() / ((ts - tm) ** 2).sum())
    ss_tot = float(((ls - lm) ** 2).sum())
    ss_res = float(((ls - (lm + slope * (ts - tm))) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return -slope, r2


# --------------------------------------------------------------------------- #
#  Case: equilibrium fixed point
# --------------------------------------------------------------------------- #


def run_feq_fixed_point(spec: ModelSpec, grid: Grid, tau: float, *, dtype=torch.float64) -> dict:
    ops = lattice_ops(spec.lattice)
    shape = grid.shape()
    rho = torch.full(shape, 1.0 + 0.1, dtype=dtype)
    ux = torch.full(shape, 0.02, dtype=dtype)
    uy = torch.full(shape, -0.01, dtype=dtype)
    uz = torch.full(shape, 0.03, dtype=dtype)
    feq = ops["equilibrium"](rho, ux, uy, uz).to(dtype)
    collider = build_collider(spec.name, tau)
    f_post = collider(feq)
    scale = float(feq.abs().max())
    d_inf = float((f_post - feq).abs().max()) / scale
    rho2, ux2, uy2, uz2 = ops["macroscopic"](f_post)
    d_rho = float((rho2 - rho).abs().max())
    d_u = max(
        float((ux2 - ux).abs().max()),
        float((uy2 - uy).abs().max()),
        float((uz2 - uz).abs().max()),
    )
    tol = 1e-6 if dtype == torch.float64 else 1e-5
    ok = max(d_inf, d_rho, d_u) <= tol
    return {
        "status": "PASS" if ok else "FAIL",
        "metrics": {
            "rel_feq_fixed_point_error": d_inf,
            "max_abs_rho_error": d_rho,
            "max_abs_velocity_error": d_u,
        },
        "ref": {
            "statement": "f_eq is a fixed point of the collision operator; rho and u are conserved",
            "expected_rel_feq_fixed_point_error": 0.0,
            "expected_max_abs_rho_error": 0.0,
            "expected_max_abs_velocity_error": 0.0,
        },
        "tolerance": {"max_rel_error": tol, "dtype": str(dtype)},
        "notes": "exact algebraic identity; tolerance is round-off of the working dtype",
    }


# --------------------------------------------------------------------------- #
#  Case: shear-wave decay (analytic viscosity)
# --------------------------------------------------------------------------- #


def run_shear_decay(
    spec: ModelSpec,
    grid: Grid,
    tau: float,
    *,
    u0: float = 0.01,
    steps: int | None = None,
    dtype=torch.float64,
    tol_pct: float = 3.0,
    enforcement: str = "gate",
    floor_ratio: float = 0.25,
) -> dict:
    k = grid.k
    nu_ref = nu_from_tau(tau)
    omega = nu_ref * k * k
    if steps is None:
        # ~1.6 e-folds of decay: long enough to fit a slope, short enough to
        # stay far from the round-off floor.
        steps = max(40, int(round(1.6 / omega)))

    collider = build_collider(spec.name, tau)
    stream = lattice_ops(spec.lattice)["stream"]

    f = _shear_field(spec, grid, u0, dtype)
    m0 = _mass(f, spec)
    times: list[float] = [0.0]
    amps: list[float] = [_amplitude(f, grid, spec)]
    mass_drift = 0.0
    finite = True
    # Diagnostics (amplitude projection + mass) are sampled, not per step: the
    # invariants are smooth in time and every macroscopic call costs about as
    # much as a time step on these small boxes.
    sample_every = max(1, steps // 16)
    for n in range(1, steps + 1):
        f = stream(collider(f))
        if n % sample_every == 0 or n == steps:
            times.append(float(n))
            amps.append(_amplitude(f, grid, spec))
            mass_drift = max(mass_drift, abs(_mass(f, spec) - m0) / m0)
            if not torch.isfinite(f).all():
                finite = False
                break

    if not finite:
        return {
            "status": "FAIL",
            "metrics": {"nonfinite": 1.0},
            "ref": {"nu": nu_ref, "decay_rate": omega, "u0": u0},
            "tolerance": {"nu_rel_error_pct": tol_pct},
            "notes": "distribution went non-finite before the fit window closed",
        }

    rate, r2 = _fit_decay_rate(times, amps, floor_ratio)
    nu_fit = rate / (k * k)
    # analytic reference series for the same samples (for the figure)
    ref_amps = [u0 * math.exp(-omega * t) for t in times]
    err_pct = 100.0 * (nu_fit - nu_ref) / nu_ref
    field_err = 0.0
    if amps and ref_amps:
        num = sum((a - r) ** 2 for a, r in zip(amps, ref_amps))
        den = sum(r * r for r in ref_amps)
        field_err = 100.0 * math.sqrt(num / den) if den > 0 else 0.0

    # criterion: fitted viscosity within tol of the analytic value AND mass
    # conserved at round-off level AND the fit is a genuine exponential.
    ok = abs(err_pct) <= tol_pct and mass_drift <= 1e-8 and r2 >= 0.999
    return {
        "status": "PASS" if ok else "FAIL",
        "metrics": {
            "nu_fit": nu_fit,
            "nu_ref": nu_ref,
            "nu_rel_error_pct": err_pct,
            "decay_rate_fit": rate,
            "decay_rate_ref": omega,
            "amplitude_series_rel_l2_pct": field_err,
            "fit_r2": r2,
            "mass_drift_rel": mass_drift,
            "u0": u0,
            "mach": u0 / math.sqrt(1.0 / 3.0),
            "steps": float(steps),
            "k": k,
            "tau": tau,
        },
        "ref": {
            "solution": "u_x(y,t) = u0 cos(k y) exp(-nu k^2 t), periodic",
            "nu": nu_ref,
            "decay_rate": omega,
            "nu_formula": "(tau - 0.5) / 3",
        },
        "tolerance": {
            "nu_rel_error_pct": tol_pct,
            "mass_drift_rel": 1e-8,
            "fit_r2_min": 0.999,
            "enforcement": enforcement,
        },
        "notes": "gate case: molecular viscosity is part of the model definition",
        "curves": {"t": times, "amplitude": amps, "amplitude_analytic": ref_amps},
    }



# --------------------------------------------------------------------------- #
#  Case: wall-bounded Poiseuille channel (boundary-condition axis)
# --------------------------------------------------------------------------- #


def _wall_ops(lattice: str):
    """Production lattice tables + production half-way bounce-back for *lattice*.

    The wall case must not carry its own boundary numerics: the reflection is
    the shipped ``bounce_back_cells_3d`` / ``bounce_back_cells_27`` swap, the
    streaming is the shipped periodic roll, and the body force is injected with
    the lattice's own ``C``/``W`` tables.
    """
    key = lattice.upper()
    if key == "D3Q19":
        from ..boundary.boundaries3d import bounce_back_cells_3d  # noqa: PLC0415
        from ..d3q19 import C, W  # noqa: PLC0415

        return C, W, bounce_back_cells_3d
    if key == "D3Q27":
        from ..boundary.boundaries_d3q27 import bounce_back_cells_27  # noqa: PLC0415
        from ..d3q27 import C, W  # noqa: PLC0415

        return C, W, bounce_back_cells_27
    raise ValueError(f"lattice {lattice!r} has no wall operator in this repository")


def wall_poiseuille_exact(y: torch.Tensor, ny: int, g: float, nu: float) -> torch.Tensor:
    """Analytic plane-Poiseuille profile for a half-way-bounce-back channel.

    The solid rows are ``y = 0`` and ``y = ny - 1``; half-way bounce-back places
    the *numerical* walls half a lattice unit outside the first *fluid* node,
    i.e. at ``Y = 1/2`` and ``Y = ny - 3/2`` (fluid nodes sit at ``Y = 1 ..
    ny-2``, channel width ``H = ny - 2``).  The steady solution of
    ``nu d2u/dY2 = -g`` with no-slip at those walls is

    .. math:: u(Y) = \\frac{g}{2\\nu} (Y - 1/2)(ny - 3/2 - Y)

    This is the *a-priori* reference: ``g`` and ``nu = (tau-1/2)/3`` are inputs,
    nothing is fitted to the simulation.
    """
    return (g / (2.0 * nu)) * (y - 0.5) * ((ny - 1.5) - y)


def _parabolic_wall_offset(y: torch.Tensor, u: torch.Tensor) -> tuple[float, float]:
    """Wall positions implied by a least-squares parabola through the profile.

    Two independent choices make the number checkable rather than fitted: the
    parabola is anchored on the *interior* fluid nodes only (one wall node of
    each side is dropped), and the two roots are taken separately so the low
    and high wall are not assumed symmetric.
    """
    yin = y[1:-1]
    uin = u[1:-1]
    a = torch.stack([yin * yin, yin, torch.ones_like(yin)], dim=1)
    coef = torch.linalg.lstsq(a, uin.unsqueeze(1)).solution.flatten()
    c2, c1, c0 = (float(v) for v in coef)
    disc = c1 * c1 - 4.0 * c2 * c0
    if c2 == 0.0 or disc < 0.0:
        return float("nan"), float("nan")
    sq = math.sqrt(disc)
    r1 = (-c1 - sq) / (2.0 * c2)
    r2 = (-c1 + sq) / (2.0 * c2)
    return min(r1, r2), max(r1, r2)


def wall_poiseuille_step(
    spec: ModelSpec,
    f: torch.Tensor,
    solid: torch.Tensor,
    source: torch.Tensor,
    collider: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """One wall-bounded step: collide -> body-force kick -> NoDynamics -> BB -> stream.

    This is the exact order documented by :func:`tensorlbm.lbm_step_correct.
    lbm_step_correct` in ``wall_treatment='bb'`` mode (collision, solid restore,
    half-way bounce-back *before* streaming), with the constant body force
    added between collision and the solid restore.  Every piece is a shipped
    operator; the composition itself is cross-checked against
    :class:`tensorlbm.rotating_frame.RotatingFrameLBM` (a validated Guo-forced
    stepper) in ``tests/test_model_matrix_wall.py``.
    """
    stream = lattice_ops(spec.lattice)["stream"]
    f_pre = f
    f_post = collider(f) + source
    f_post = torch.where(solid.unsqueeze(0), f_pre, f_post)
    f_post = _wall_ops(spec.lattice)[2](f_post, solid)
    return stream(f_post)


def run_wall_poiseuille(
    spec: ModelSpec,
    grid: Grid,
    tau: float,
    *,
    g: float = 1e-5,
    dtype=torch.float64,
    tol_pct: float = 3.0,
    enforcement: str = "gate",
    max_offset: float = 0.15,
    max_steps: int = 40000,
) -> dict:
    """Steady plane Poiseuille flow between two half-way bounce-back walls.

    Why this case exists: every other case in this module is *periodic*, so the
    whole model matrix is blind to the boundary operator -- an inverted
    reflection table, a one-cell mask slip or a missing NoDynamics restore is
    invisible to it.  This case puts the production wall operator into the
    matrix and measures three things that a broken wall cannot fake:

    ``rel_l2_pct``
        velocity profile vs the analytic parabola (walls at ``y = +-1/2``)
    ``wall_offset_lattice``
        the wall position implied by a parabola fitted through the interior
        nodes.  Ideal half-way bounce-back gives ``0.0 +- O(1/H)``; a mask or
        reflection that lands the wall one cell away gives ``~1``.
    ``symmetry_rel``
        ``max_j |u(j) - u(ny-1-j)| / u_max`` -- the channel is symmetric, so an
        asymmetric streaming/reflection convention shows up here.

    plus the mass drift (a wall that leaks mass) and the centreline velocity.

    The inverse of this case is not claimed: it verifies the *discrete*
    steady state of a straight, grid-aligned wall.  Curved/interpolated walls
    and unsteady wall forces are out of scope (see the round-3 doc).
    """
    ops = lattice_ops(spec.lattice)
    C, W, _bb = _wall_ops(spec.lattice)
    ny = int(grid.ny)
    if ny < 6:
        return {
            "status": "FAIL",
            "metrics": {"ny": float(ny)},
            "ref": {},
            "tolerance": {},
            "notes": "channel needs at least 4 fluid rows",
        }
    nx = 2  # the flow is x- and z-invariant: two cells keep the case cheap
    nz = 1
    shape = (nz, ny, nx)
    nu = nu_from_tau(tau)
    H = ny - 2

    solid = torch.zeros(shape, dtype=torch.bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    c = C.to(dtype=dtype)
    w = W.to(dtype=dtype)
    # constant body force: sum_i c_i (3 w_i c_i.F) = F exactly, sum_i 3 w_i c_i.F = 0
    source = (3.0 * w).view(-1, 1, 1, 1) * (c[:, 0].view(-1, 1, 1, 1) * g)

    y = torch.arange(ny, dtype=dtype)
    u_exact = wall_poiseuille_exact(y, ny, g, nu)
    u0 = torch.full(shape, 0.0, dtype=dtype)
    u0 = u_exact.view(1, ny, 1).expand(shape).clone()
    rho0 = torch.ones(shape, dtype=dtype)
    f = ops["equilibrium"](rho0, u0, torch.zeros_like(u0), torch.zeros_like(u0)).to(dtype)
    collider = build_collider(spec.name, tau)

    m0 = float(rho0.mean()) * ny * nx * nz
    # The transient that the analytic initial state leaves behind is the wall
    # boundary layer; 4 diffusive times is enough to push it below round-off
    # level for every tau used by the matrix.
    steps_min = int(max(800, round(4.0 * H * H / (math.pi**2 * nu))))
    mass_drift = 0.0
    u_prev = u_exact.clone()  # 1-D profile, same shape as the sampled slice
    steps = 0
    converged = False
    for n in range(1, max_steps + 1):
        f = wall_poiseuille_step(spec, f, solid, source, collider)
        steps = n
        if n < steps_min:
            continue
        if n % 100 == 0 or n == steps_min:
            rho, ux, _uy, _uz = ops["macroscopic"](f)
            mass_drift = max(mass_drift, abs(float(rho.mean()) * ny * nx * nz - m0) / m0)
            if not torch.isfinite(f).all():
                break
            u_now = ux[0, :, 0]
            scale = max(float(u_now.abs().max()), 1e-30)
            if float((u_now - u_prev).abs().max()) / scale < 1e-10:
                converged = True
                break
            u_prev = u_now

    rho, ux, _uy, _uz = ops["macroscopic"](f)
    finite = bool(torch.isfinite(f).all())
    profile = ux[0, :, 0].clone()
    fl = slice(1, ny - 1)
    uf = profile[fl]
    ue = u_exact[fl]
    den = float((ue * ue).sum())
    rel_l2 = 100.0 * math.sqrt(float(((uf - ue) ** 2).sum()) / den) if den > 0 else float("inf")
    u_center = float(profile[ny // 2])
    u_center_exact = float(u_exact[ny // 2])
    u_max = float(ue.abs().max())
    sym = float((profile[fl] - profile.flip(0)[fl]).abs().max()) / max(u_max, 1e-30)
    y_lo, y_hi = _parabolic_wall_offset(y, profile)
    off_lo = y_lo - 0.5
    off_hi = (ny - 1.5) - y_hi
    offset = 0.5 * (off_lo + off_hi)

    ok = (
        finite
        and rel_l2 <= tol_pct
        and abs(offset) <= max_offset
        and sym <= 5e-3
        and mass_drift <= 1e-10
        and u_center > 0.0
    )
    notes = (
        "gate case: boundary operator axis (half-way bounce-back + NoDynamics restore); "
        "reference is the a-priori parabola with walls at Y=+-1/2"
    )
    if not finite:
        notes = "distribution went non-finite (wall + forcing composition unstable)"
    elif not converged:
        notes += f"; steady-state detector did not fire within {steps} steps"
    return {
        "status": "PASS" if ok else "FAIL",
        "metrics": {
            "rel_l2_pct": rel_l2,
            "wall_offset_lattice": offset,
            "wall_offset_low": off_lo,
            "wall_offset_high": off_hi,
            "u_center": u_center,
            "u_center_exact": u_center_exact,
            "u_center_rel_err_pct": 100.0 * (u_center - u_center_exact) / u_center_exact,
            "u_max_exact": u_max,
            "mach": u_max / CS,
            "symmetry_rel": sym,
            "mass_drift_rel": mass_drift,
            "steps": float(steps),
            "converged": 1.0 if converged else 0.0,
            "tau": tau,
            "nu": nu,
            "H": float(H),
            "g": g,
        },
        "ref": {
            "solution": "u_x(Y) = g/(2 nu) (Y - 1/2)(ny - 3/2 - Y), no-slip at Y = 1/2 and Y = ny - 3/2",
            "nu": nu,
            "g": g,
            "wall_position": "half-way bounce-back => Y = 1/2 and Y = ny - 3/2",
        },
        "tolerance": {
            "rel_l2_pct": tol_pct,
            "wall_offset_lattice_max": max_offset,
            "symmetry_rel_max": 5e-3,
            "mass_drift_rel": 1e-10,
            "enforcement": enforcement,
        },
        "notes": notes,
        "curves": {
            "y": [float(v) for v in y.tolist()],
            "u": [float(v) for v in profile.tolist()],
            "u_exact": [float(v) for v in u_exact.tolist()],
        },
    }


# --------------------------------------------------------------------------- #
#  Case: acoustic-wave decay (analytic bulk viscosity)
# --------------------------------------------------------------------------- #


def _sound_field(spec: ModelSpec, grid: Grid, amp: float, dtype: torch.dtype) -> torch.Tensor:
    """Right-going linear acoustic wave ``rho = 1 + a cos(kx)``, ``u_x = c_s a cos(kx)``.

    ``(dq, du) ~ (1, c_s)`` is the acoustic eigenvector, so the initial state
    excites one sound mode and nothing else: no vorticity mode (``u`` is
    curl-free) and no entropy mode (``u = c_s dq`` closes the impedance
    relation).  The wave is periodic because ``k = 2 pi / nx``.
    """
    ops = lattice_ops(spec.lattice)
    shape = grid.shape()
    k = 2.0 * math.pi / grid.nx
    x = torch.arange(grid.nx, dtype=dtype).view(1, 1, grid.nx)
    cos = torch.cos(k * x).expand(shape).contiguous()
    rho = 1.0 + amp * cos
    ux = CS * amp * cos
    zero = torch.zeros(shape, dtype=dtype)
    return ops["equilibrium"](rho, ux, zero, zero).to(dtype)


def _acoustic_amplitude(f: torch.Tensor, grid: Grid, spec: ModelSpec) -> float:
    """Envelope of the running sound wave: ``|A_c + i A_s|``.

    A *travelling* wave rotates between the ``cos(kx)`` and ``sin(kx)``
    quadratures at frequency ``c_s k``; projecting onto ``cos`` alone would
    therefore measure an oscillation, not a decay.  Both quadratures are
    sampled and the magnitude is used.  (This was the bug in the first
    attempt -- see the R2 report, "注意点".)
    """
    ops = lattice_ops(spec.lattice)
    rho, _ux, _uy, _uz = ops["macroscopic"](f)
    k = 2.0 * math.pi / grid.nx
    x = torch.arange(grid.nx, dtype=rho.dtype).view(1, 1, grid.nx)
    dr = rho - 1.0
    ac = 2.0 * float((dr * torch.cos(k * x)).mean())
    as_ = 2.0 * float((dr * torch.sin(k * x)).mean())
    return math.hypot(ac, as_)


def run_acoustic_decay(
    spec: ModelSpec,
    grid: Grid,
    tau: float,
    *,
    amp: float = 1e-4,
    steps: int | None = None,
    dtype=torch.float64,
    tol_pct: float = 3.0,
    enforcement: str = "gate",
    floor_ratio: float = 0.20,
    sample_every: int = 2,
) -> dict:
    """Sound attenuation: the *bulk* channel of the same collision operator.

    Linear acoustics of the Navier-Stokes equations gives, for the amplitude
    attenuation rate ``gamma`` of a plane wave,

    .. math::
        \gamma = \tfrac{1}{2} k^2 \left( \tfrac{4}{3}\nu + \nu_b \right)
        \quad\Longrightarrow\quad
        S \equiv \tfrac{2\gamma}{k^2} = \tfrac{4}{3}\nu + \nu_b ,
        \qquad \nu_b = \tfrac{2}{9}\left(\tfrac{1}{s_b} - \tfrac{1}{2}\right).

    ``shear_decay`` cannot see ``nu_b`` at all (a shear wave is solenoidal, so
    the trace moment is never excited); this case is what makes the bulk
    channel of every registry entry verifiable, and it is the reason a
    deliberately corrupted ``s_b`` no longer hides behind a green gate
    (``model_matrix.mutation``, class ``bulk_only``).
    """
    k = 2.0 * math.pi / grid.nx
    nu = nu_from_tau(tau)
    nu_b, s_b = bulk_viscosity(spec, tau)
    S_ref = 4.0 * nu / 3.0 + nu_b
    gamma = 0.5 * S_ref * k * k
    if not gamma > 0.0:
        return {
            "status": "NOT_APPLICABLE",
            "metrics": {"S_ref": S_ref, "s_b": s_b},
            "ref": {},
            "tolerance": {},
            "notes": f"non-dissipative acoustic channel (s_b={s_b:.4f}, nu_b={nu_b:.6g})",
        }
    if steps is None:
        # ~2 e-folds of envelope decay: long enough for a slope, short enough
        # to stay away from the round-off floor.
        steps = int(round(2.0 / gamma))
        steps = max(200, min(steps, 4000))

    collider = build_collider(spec.name, tau)
    stream = lattice_ops(spec.lattice)["stream"]
    f = _sound_field(spec, grid, amp, dtype)
    mass0 = float(lattice_ops(spec.lattice)["macroscopic"](f)[0].mean())

    times: list[float] = []
    amps: list[float] = []
    mass_drift = 0.0
    for n in range(0, steps + 1):
        if n % sample_every == 0:
            times.append(float(n))
            amps.append(_acoustic_amplitude(f, grid, spec))
            rho, _ux, _uy, _uz = lattice_ops(spec.lattice)["macroscopic"](f)
            mass_drift = max(mass_drift, abs(float(rho.mean()) - mass0) / mass0)
            if not torch.isfinite(f).all():
                return {
                    "status": "FAIL",
                    "metrics": {"nonfinite": 1.0, "tau": tau},
                    "ref": {"S": S_ref},
                    "tolerance": {"acoustic_rel_error_pct": tol_pct},
                    "notes": "distribution went non-finite before the fit window closed",
                }
        if n < steps:
            f = stream(collider(f))

    rate, r2 = _fit_decay_rate(times, amps, floor_ratio)
    S_meas = 2.0 * rate / (k * k)
    err_pct = 100.0 * (S_meas - S_ref) / S_ref
    # invert the measurement for the eigenvalue the operator actually uses
    num = 9.0 * (S_meas - 4.0 * nu / 3.0) / 2.0 + 0.5
    s_b_implied = 1.0 / num if num != 0.0 else float("inf")

    # The envelope of a *travelling* wave carries a small residual from the
    # lattice dispersion of the acoustic branch (it shrinks with resolution:
    # r2 = 0.9983/0.9993/0.9996 at nx = 16/24/32 for the cascaded operator), so
    # the fit-linearity guard here is a sanity bound, not the discriminator --
    # the discriminator is the attenuation error itself (<=0.4% measured).
    ok = abs(err_pct) <= tol_pct and mass_drift <= 1e-10 and r2 >= 0.995
    return {
        "status": "PASS" if ok else "FAIL",
        "metrics": {
            "S_meas": S_meas,
            "S_ref": S_ref,
            "acoustic_rel_error_pct": err_pct,
            "nu_b_ref": nu_b,
            "nu_b_implied": S_meas - 4.0 * nu / 3.0,
            "s_b": s_b,
            "s_b_implied": s_b_implied,
            "nu_ref": nu,
            "decay_rate_fit": rate,
            "decay_rate_ref": gamma,
            "fit_r2": r2,
            "mass_drift_rel": mass_drift,
            "k": k,
            "amp": amp,
            "mach": CS * amp,
            "steps": float(steps),
            "tau": tau,
        },
        "ref": {
            "solution": "rho-1 = amp cos(kx - c_s k t) exp(-gamma t), u_x = c_s (rho-1)",
            "S_formula": "2 gamma / k^2 = 4 nu/3 + nu_b",
            "nu_b_formula": "(2/9)(1/s_b - 1/2)",
            "S": S_ref,
            "nu_b": nu_b,
            "s_b": s_b,
            "nu": nu,
        },
        "tolerance": {
            "acoustic_rel_error_pct": tol_pct,
            "mass_drift_rel": 1e-10,
            "fit_r2_min": 0.995,
            "enforcement": enforcement,
        },
        "notes": "gate case: sound attenuation measures 4nu/3+nu_b (the bulk channel)",
        "curves": {"t": times, "amplitude": amps},
    }


# --------------------------------------------------------------------------- #
#  Case: Taylor-Green vortex decay (advisory, finite-Mach probe)
# --------------------------------------------------------------------------- #


def _tg_field(spec: ModelSpec, grid: Grid, u0: float, dtype: torch.dtype) -> torch.Tensor:
    ops = lattice_ops(spec.lattice)
    shape = grid.shape()
    x = torch.arange(grid.nx, dtype=dtype).view(1, 1, grid.nx)
    y = torch.arange(grid.ny, dtype=dtype).view(1, grid.ny, 1)
    k = grid.k
    ux = (u0 * torch.sin(k * x) * torch.cos(k * y)).expand(shape).contiguous()
    uy = (-u0 * torch.cos(k * x) * torch.sin(k * y)).expand(shape).contiguous()
    zero = torch.zeros(shape, dtype=dtype)
    rho = torch.ones(shape, dtype=dtype)
    return ops["equilibrium"](rho, ux, uy, zero).to(dtype)


def _kinetic_energy(f: torch.Tensor, spec: ModelSpec) -> float:
    _rho, ux, uy, uz = lattice_ops(spec.lattice)["macroscopic"](f)
    return 0.5 * float((ux * ux + uy * uy + uz * uz).mean())


def run_taylor_green_decay(
    spec: ModelSpec,
    grid: Grid,
    tau: float,
    *,
    u0: float = 0.1,
    steps: int | None = None,
    dtype=torch.float64,
    tol_pct: float = 5.0,
    enforcement: str = "advisory",
    floor_ratio: float = 0.25,
) -> dict:
    k = grid.k
    nu_ref = nu_from_tau(tau)
    # The TG field above carries modes (k, k), so laplacian(u) = -2 k^2 u and the
    # velocity decays at 2 nu k^2; the kinetic energy E ~ u^2 therefore decays
    # twice as fast: dE/dt = -4 nu k^2 E.
    rate_ref = 4.0 * nu_ref * k * k
    if steps is None:
        steps = max(40, int(round(1.4 / rate_ref)))
    collider = build_collider(spec.name, tau)
    stream = lattice_ops(spec.lattice)["stream"]

    f = _tg_field(spec, grid, u0, dtype)
    times = [0.0]
    energies = [_kinetic_energy(f, spec)]
    finite = True
    sample_every = max(1, steps // 16)
    for n in range(1, steps + 1):
        f = stream(collider(f))
        if n % sample_every == 0 or n == steps:
            times.append(float(n))
            energies.append(_kinetic_energy(f, spec))
            if not torch.isfinite(f).all():
                finite = False
                break
    if not finite:
        return {
            "status": "FAIL",
            "metrics": {"nonfinite": 1.0},
            "ref": {"nu": nu_ref, "decay_rate": rate_ref, "u0": u0},
            "tolerance": {"nu_rel_error_pct": tol_pct},
            "notes": "distribution went non-finite before the fit window closed",
        }
    rate, r2 = _fit_decay_rate(times, energies, floor_ratio)
    nu_fit = rate / (4.0 * k * k)
    err_pct = 100.0 * (nu_fit - nu_ref) / nu_ref
    ref_energies = [energies[0] * math.exp(-rate_ref * t) for t in times]
    num = sum((e - r) ** 2 for e, r in zip(energies, ref_energies))
    den = sum(r * r for r in ref_energies)
    energy_err = 100.0 * math.sqrt(num / den) if den > 0 else 0.0
    ok = abs(err_pct) <= tol_pct
    return {
        "status": "PASS" if ok else "FAIL",
        "metrics": {
            "nu_fit": nu_fit,
            "nu_ref": nu_ref,
            "nu_rel_error_pct": err_pct,
            "decay_rate_fit": rate,
            "decay_rate_ref": rate_ref,
            "energy_series_rel_l2_pct": energy_err,
            "fit_r2": r2,
            "u0": u0,
            "mach": u0 / math.sqrt(1.0 / 3.0),
            "steps": float(steps),
            "k": k,
            "tau": tau,
        },
        "ref": {
            "solution": "Taylor-Green (modes k,k): E(t) = E0 exp(-4 nu k^2 t) (incompressible limit)",
            "nu": nu_ref,
            "decay_rate": rate_ref,
            "nu_formula": "(tau - 0.5) / 3",
        },
        "tolerance": {"nu_rel_error_pct": tol_pct, "enforcement": enforcement},
        "notes": "advisory: deviations carry the O(Ma^2) compressibility error",
        "curves": {"t": times, "energy": energies, "energy_analytic": ref_energies},
    }


# --------------------------------------------------------------------------- #
#  Case registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CaseSpec:
    name: str
    description: str
    enforcement: str
    runner: Callable[..., dict]
    tol_pct: float
    metrics_doc: str


CASES: dict[str, CaseSpec] = {
    "feq_fixed_point": CaseSpec(
        name="feq_fixed_point",
        description="equilibrium fixed point + exact conservation of rho and u (algebraic identity)",
        enforcement="gate",
        runner=run_feq_fixed_point,
        tol_pct=0.0,
        metrics_doc="max relative |C(f_eq)-f_eq|, max |rho'-rho|, max |u'-u|",
    ),
    "shear_decay": CaseSpec(
        name="shear_decay",
        description="periodic shear wave decay: fitted nu vs analytic nu=(tau-1/2)/3, u0=0.01",
        enforcement="gate",
        runner=run_shear_decay,
        tol_pct=3.0,
        metrics_doc="nu_fit, nu_ref, nu_rel_error_pct, fit_r2, mass_drift_rel",
    ),
    "acoustic_decay": CaseSpec(
        name="acoustic_decay",
        description="periodic acoustic wave: measured 4nu/3+nu_b vs analytic 4nu/3+(2/9)(1/s_b-1/2), amp=1e-4",
        enforcement="gate",
        runner=run_acoustic_decay,
        tol_pct=3.0,
        metrics_doc="S_meas, S_ref, acoustic_rel_error_pct, nu_b_implied, fit_r2",
    ),
    "wall_poiseuille": CaseSpec(
        name="wall_poiseuille",
        description=(
            "steady plane Poiseuille between two half-way bounce-back walls: velocity profile "
            "vs the analytic parabola (walls at Y=+-1/2), implied wall offset, channel symmetry"
        ),
        enforcement="gate",
        runner=run_wall_poiseuille,
        tol_pct=3.0,
        metrics_doc="rel_l2_pct, wall_offset_lattice, symmetry_rel, mass_drift_rel, u_center_rel_err_pct",
    ),
    "tg_decay": CaseSpec(
        name="tg_decay",
        description="Taylor-Green vortex at u0=0.1: fitted nu vs analytic, advisory (O(Ma^2) probe)",
        enforcement="advisory",
        runner=run_taylor_green_decay,
        tol_pct=5.0,
        metrics_doc="nu_fit, nu_ref, nu_rel_error_pct (advisory), energy_series_rel_l2_pct",
    ),
}


def case_names() -> list[str]:
    return sorted(CASES)


def run_case(
    case_name: str,
    spec: ModelSpec,
    grid: Grid,
    tau: float,
    *,
    dtype=torch.float64,
) -> CaseResult:
    cs = CASES[case_name]
    if case_name in spec.inapplicable_reason:
        return CaseResult(
            case=case_name,
            model=spec.name,
            status="NOT_APPLICABLE",
            metrics={},
            ref={},
            tolerance={"note": spec.inapplicable_reason[case_name]},
            notes=spec.inapplicable_reason[case_name],
        )
    if case_name not in spec.cases:
        return CaseResult(
            case=case_name,
            model=spec.name,
            status="MISSING",
            metrics={},
            ref={},
            tolerance={},
            notes="case is not declared for this model (coverage hole)",
        )
    try:
        if case_name == "tg_decay":
            out = cs.runner(spec, grid, tau, dtype=dtype, tol_pct=cs.tol_pct, enforcement="advisory")
        elif case_name == "shear_decay":
            out = cs.runner(spec, grid, tau, dtype=dtype, tol_pct=cs.tol_pct, enforcement="gate")
        elif case_name == "acoustic_decay":
            out = cs.runner(spec, grid, tau, dtype=dtype, tol_pct=cs.tol_pct, enforcement="gate")
        else:
            out = cs.runner(spec, grid, tau, dtype=dtype)
    except Exception as exc:  # noqa: BLE001 - a crashing model is a matrix result
        return CaseResult(
            case=case_name,
            model=spec.name,
            status="ERROR",
            metrics={},
            ref={},
            tolerance={},
            error=f"{type(exc).__name__}: {exc}",
        )
    if spec.enforcement == "guard" and out["status"] == "PASS" and case_name != "feq_fixed_point":
        out["status"] = "GUARD"
    return CaseResult(
        case=case_name,
        model=spec.name,
        status=out["status"],
        metrics=out["metrics"],
        ref=out["ref"],
        tolerance=out["tolerance"],
        notes=out["notes"],
        curves=out.get("curves"),
    )