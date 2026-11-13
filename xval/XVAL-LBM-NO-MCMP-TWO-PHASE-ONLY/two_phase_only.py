"""Two-phase-only ("no component index") layered Poiseuille cross-validation.

Anchor
------
``XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY`` — inventory row ``oss_solutions`` id 274
(``lssfau/walberla`` / feature ``GAP-LBM-MCMP-POISEUILLE`` / tag
``no-mcmp-two-phase-only``).  OSS scope, verbatim: *the OSS repository has no
MCMP (no component index, no per-component transport equation); the
two-component layered Poiseuille benchmark cannot be mapped onto it directly.
What can be borrowed is only the phase-field layered initialisation geometry
plus the variable-density equilibrium assembly — i.e. "two-component" is
downgraded to "two-phase layered".*

Question asked here
-------------------
Is that downgrade accurate enough to stand in for the two-component layered
Poiseuille analytic benchmark inside the owner 15 % band, and over which
stratification range?

Model implemented (M2P, "two-phase only")
----------------------------------------
* **One** distribution function per node — no component index ``p``, no
  per-component density, hence no component transport equation
  (the structural fact that makes the OSS route "no-mcmp").
* Phase indicator ``C(y) = 0.5 * (1 - tanh((y - y_int) / w))``, 1 in the lower
  (viscous) layer and 0 in the upper layer, **frozen**: the layering is given
  only by the initial value of ``C`` (the inventory's own wording).  A frozen
  indicator is legitimate here only because the target flow is strictly
  parallel, so the exact normal velocity at the interface is zero; the measured
  spurious cross-stream velocity is reported as the check of that assumption.
* Local kinematic viscosity from ``C`` by one of three node-local mixing rules
  (``linear`` / ``geometric`` / ``harmonic``) and ``tau = 1/2 + 3 nu``.
* Variable-coefficient BGK with Guo's second-order forcing, force density
  ``F = rho * G`` (``G`` = per-unit-mass body acceleration), i.e. the same
  driving convention as the analytic reference below.
* Full-way bounce-back on the two wall rows (no-slip planes at ``y = 0`` and
  ``y = ny - 1``, lattice units).

Analytic reference
------------------
Steady Stokes flow of two superposed layers in a plane channel, each layer
driven by the body-force density ``rho_sigma * G`` (no imposed pressure
gradient).  With ``u''_sigma = -G / nu_sigma`` the solution in each layer is
parabolic; the four coefficients follow from

    u_1(y0) = 0,  u_2(y1) = 0,            (no slip on both walls)
    u_1(yi) = u_2(yi),                    (velocity continuity)
    mu_1 u'_1(yi) = mu_2 u'_2(yi),        (shear-stress continuity)

which is solved here as a 4x4 linear system (no hand-derived closed form, so
the reference cannot inherit a transcription slip).  Note that when the body
force is imposed per unit mass the velocity solution is independent of the
density ratio — only the viscosity ratio and the interface position enter.

Library reuse (no physics is re-implemented where the library already has it):
``tensorlbm.d2q9.equilibrium`` / ``macroscopic``, ``tensorlbm.solver.stream``,
``tensorlbm.boundary.boundaries.bounce_back_cells``,
``tensorlbm.multiphase.guo_force_term`` (called with a *tensor* tau for the
variable-coefficient case) and ``tensorlbm.multiphase.collide_sc_two_component``
for the MCMP comparison arm.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from tensorlbm.boundary.boundaries import bounce_back_cells
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.multiphase import collide_sc_two_component, guo_force_term
from tensorlbm.solver import stream

CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Analytic two-layer Poiseuille (reference)
# ---------------------------------------------------------------------------


@dataclass
class TwoLayerSolution:
    """Piecewise-parabolic two-layer channel solution plus its own checks."""

    y0: float
    y1: float
    y_int: float
    G: float
    nu_lo: float
    nu_hi: float
    coef_lo: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))
    coef_hi: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))

    def u(self, y):
        """Velocity at ``y`` (array or scalar), evaluated on the right layer."""
        y = np.asarray(y, dtype=np.float64)
        a_lo, b_lo, c_lo = self.coef_lo
        a_hi, b_hi, c_hi = self.coef_hi
        lo = a_lo * y * y + b_lo * y + c_lo
        hi = a_hi * y * y + b_hi * y + c_hi
        return np.where(y <= self.y_int, lo, hi)

    def du(self, y):
        y = np.asarray(y, dtype=np.float64)
        _, b_lo, _ = self.coef_lo
        _, b_hi, _ = self.coef_hi
        a_lo, _, _ = self.coef_lo
        a_hi, _, _ = self.coef_hi
        return np.where(y <= self.y_int, 2.0 * a_lo * y + b_lo, 2.0 * a_hi * y + b_hi)

    def residuals(self) -> dict[str, float]:
        """Machine-checkable invariants of the solved coefficients."""
        _, b_lo, c_lo = self.coef_lo
        _, b_hi, c_hi = self.coef_hi
        a_lo = -self.G / (2.0 * self.nu_lo)
        a_hi = -self.G / (2.0 * self.nu_hi)
        u_lo = float(a_lo * self.y_int**2 + b_lo * self.y_int + c_lo)
        u_hi = float(a_hi * self.y_int**2 + b_hi * self.y_int + c_hi)
        # rho = 1 in each layer, so mu_sigma = nu_sigma.
        tau_lo = self.nu_lo * float(2.0 * a_lo * self.y_int + b_lo)
        tau_hi = self.nu_hi * float(2.0 * a_hi * self.y_int + b_hi)
        return {
            "wall_lo": abs(float(a_lo * self.y0**2 + b_lo * self.y0 + c_lo)),
            "wall_hi": abs(float(a_hi * self.y1**2 + b_hi * self.y1 + c_hi)),
            "u_jump": abs(u_lo - u_hi),
            "stress_jump": abs(tau_lo - tau_hi),
        }


def two_layer_analytic(
    ny: int, y_int: float, G: float, nu_lo: float, nu_hi: float
) -> TwoLayerSolution:
    """Solve the two-layer Stokes problem on ``[0, ny-1]`` with the interface at ``y_int``.

    ``nu_lo`` acts in ``0 <= y <= y_int`` and ``nu_hi`` in ``y_int <= y <= ny-1``.
    Units are lattice units; densities are 1 in both layers so that
    ``mu_sigma == nu_sigma``.
    """
    y0, y1 = 0.0, float(ny - 1)
    if not (y0 < y_int < y1):
        raise ValueError("interface must lie strictly inside the channel")
    a_lo = -G / (2.0 * nu_lo)
    a_hi = -G / (2.0 * nu_hi)
    # Unknowns: b_lo, c_lo, b_hi, c_hi.
    A = np.array(
        [
            [y0, 1.0, 0.0, 0.0],
            [0.0, 0.0, y1, 1.0],
            [y_int, 1.0, -y_int, -1.0],
            [nu_lo, 0.0, -nu_hi, 0.0],
        ],
        dtype=np.float64,
    )
    rhs = np.array(
        [
            -a_lo * y0 * y0,
            -a_hi * y1 * y1,
            -(a_lo - a_hi) * y_int * y_int,
            -2.0 * (nu_lo * a_lo - nu_hi * a_hi) * y_int,
        ],
        dtype=np.float64,
    )
    b_lo, c_lo, b_hi, c_hi = np.linalg.solve(A, rhs)
    return TwoLayerSolution(
        y0=y0,
        y1=y1,
        y_int=float(y_int),
        G=G,
        nu_lo=nu_lo,
        nu_hi=nu_hi,
        coef_lo=(a_lo, float(b_lo), float(c_lo)),
        coef_hi=(a_hi, float(b_hi), float(c_hi)),
    )


def analytic_profile(sol: TwoLayerSolution, ny: int) -> np.ndarray:
    """Node values ``u(j)`` for ``j = 0 .. ny-1``."""
    return sol.u(np.arange(ny, dtype=np.float64))


# ---------------------------------------------------------------------------
# Viscosity mixing rules on the frozen phase indicator
# ---------------------------------------------------------------------------


def phase_indicator(ny: int, nx: int, y_int: float, width: float, device=None) -> torch.Tensor:
    """Frozen ``C`` field: 1 below the interface, 0 above, tanh of ``width`` cells."""
    ys = torch.arange(ny, dtype=torch.float32, device=device).view(ny, 1).expand(ny, nx)
    return 0.5 * (1.0 - torch.tanh((ys - y_int) / width))


def mix_viscosity(C: torch.Tensor, nu_lo: float, nu_hi: float, rule: str) -> torch.Tensor:
    """Node-local viscosity from the phase indicator.

    ``linear``    nu = C nu_lo + (1-C) nu_hi          (arithmetic average)
    ``geometric`` nu = nu_lo**C nu_hi**(1-C)           (log-linear average)
    ``harmonic``  nu = 1 / (C/nu_lo + (1-C)/nu_hi)     (series average)

    The three rules bracket the diffuse-interface averaging choices; they are
    not interchangeable (see the report's error table).
    """
    if rule == "linear":
        return C * nu_lo + (1.0 - C) * nu_hi
    if rule == "geometric":
        return torch.exp(C * math.log(nu_lo) + (1.0 - C) * math.log(nu_hi))
    if rule == "harmonic":
        return 1.0 / (C / nu_lo + (1.0 - C) / nu_hi)
    raise ValueError(f"unknown viscosity rule {rule!r}")


# ---------------------------------------------------------------------------
# Two-phase-only (single distribution) solver
# ---------------------------------------------------------------------------


def _wall_mask(ny: int, nx: int, device) -> torch.Tensor:
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[0, :] = True
    mask[-1, :] = True
    return mask


def two_phase_only_step(
    f: torch.Tensor,
    tau: torch.Tensor,
    force_x: torch.Tensor,
    wall: torch.Tensor,
) -> torch.Tensor:
    """One collide+force+stream+bounce-back step of the single-distribution model."""
    rho, ux, uy = macroscopic(f)
    rho_s = torch.clamp(rho, min=1e-12)
    # Guo: the equilibrium is built on the force-corrected velocity u* = u + F/(2 rho).
    ux = ux + 0.5 * force_x / rho_s
    feq = equilibrium(rho, ux, uy)
    f = f - (f - feq) / tau
    f = f + guo_force_term(rho, ux, uy, force_x, torch.zeros_like(force_x), tau)
    f = stream(f)
    return bounce_back_cells(f, wall)


def run_two_phase_only(
    ny: int,
    nx: int = 8,
    *,
    G: float = 4e-6,
    nu_lo: float = 0.5,
    nu_hi: float = 0.25,
    width: float = 2.0,
    rule: str = "harmonic",
    rho_lo: float = 1.0,
    rho_hi: float | None = None,
    max_steps: int = 120_000,
    stop_tol: float = 1e-7,
    check_every: int = 200,
    device: str | torch.device = "cpu",
) -> dict:
    """Steady two-layer Poiseuille in the two-phase-only (no-MCMP) model.

    ``rho_hi`` (default: same as ``rho_lo``) sets a density contrast across the
    frozen interface through the initial density field.  When the two densities
    differ the density field is free to evolve (the model has no interface
    sharpening mechanism) — that drift is measured and reported, it is one of
    the structural costs of the no-MCMP downgrade.
    """
    device = torch.device(device)
    rho_hi = rho_lo if rho_hi is None else rho_hi
    y_int = 0.5 * (ny - 1)
    C = phase_indicator(ny, nx, y_int, width, device=device)
    nu = mix_viscosity(C, nu_lo, nu_hi, rule)
    tau = 0.5 + 3.0 * nu

    rho0 = rho_lo * C + rho_hi * (1.0 - C)
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    force_x = rho0 * G

    wall = _wall_mask(ny, nx, device)
    # The walls are solid: keep their initial equilibrium as the bounce-back buffer.
    mass0 = float(rho0[1:-1, :].sum())

    sol = two_layer_analytic(ny, y_int, G, nu_lo, nu_hi)
    y = np.arange(ny, dtype=np.float64)
    u_ana = analytic_profile(sol, ny)

    history: list[tuple[int, float]] = []
    prev = None
    steps = 0
    converged = False
    t0 = time.time()
    for step in range(1, max_steps + 1):
        f = two_phase_only_step(f, tau, force_x, wall)
        steps = step
        if step % check_every == 0:
            rho, ux, uy = macroscopic(f)
            u_peak = float(ux[1:-1, :].abs().max())
            history.append((step, u_peak))
            if prev is not None and abs(u_peak - prev) <= stop_tol * max(abs(prev), 1e-12):
                converged = True
                break
            prev = u_peak
        if not torch.isfinite(f).all():
            break
    wall_s = time.time() - t0

    rho, ux, uy = macroscopic(f)
    # Steady profile: average over the periodic x direction (the flow is x-invariant).
    prof = ux[:, :].mean(dim=1).double().cpu().numpy()
    uy_max = float(uy.abs().max())
    u_peak_sim = float(np.abs(prof[1:-1]).max())
    rho_prof = rho.mean(dim=1).double().cpu().numpy()

    fluid = slice(1, ny - 1)
    sim = prof[fluid]
    ana = u_ana[fluid]
    denom = float(np.linalg.norm(ana))
    l2_rel = float(np.linalg.norm(sim - ana) / denom) if denom > 0 else float("nan")
    max_abs = float(np.max(np.abs(sim - ana)))
    u_peak_ana = float(np.abs(ana).max())
    max_rel_peak = max_abs / u_peak_ana if u_peak_ana > 0 else float("nan")

    # Per-layer volumetric flux (density-weighted, so a density contrast shows up).
    y_nodes = y[fluid]
    lo_sel = y_nodes <= y_int
    hi_sel = ~lo_sel
    # Layer fluxes are density-weighted so that a density contrast shows up; the
    # same weight is applied to the analytic profile (like compared with like).
    Q_lo = float(np.sum(rho_prof[fluid][lo_sel] * sim[lo_sel]))
    Q_hi = float(np.sum(rho_prof[fluid][hi_sel] * sim[hi_sel]))
    Q_lo_ana = float(np.sum(rho_prof[fluid][lo_sel] * ana[lo_sel]))
    Q_hi_ana = float(np.sum(rho_prof[fluid][hi_sel] * ana[hi_sel]))

    def _rel(a: float, b: float) -> float:
        return float(abs(a - b) / abs(b)) if b != 0 else float("nan")

    # Density-field drift (interface smearing of the initial stratification).
    width_10_90 = _interface_width_10_90(rho_prof)
    centre = _interface_centre(rho_prof, y)

    return {
        "config": {
            "ny": ny,
            "nx": nx,
            "G": G,
            "nu_lo": nu_lo,
            "nu_hi": nu_hi,
            "nu_ratio": nu_lo / nu_hi,
            "width": width,
            "rule": rule,
            "rho_lo": rho_lo,
            "rho_hi": rho_hi,
            "max_steps": max_steps,
            "stop_tol": stop_tol,
            "check_every": check_every,
        },
        "steps": steps,
        "converged": converged,
        "wall_s": wall_s,
        "u_peak_sim": u_peak_sim,
        "u_peak_ana": u_peak_ana,
        "u_peak_ratio": (u_peak_sim / u_peak_ana) if u_peak_ana else float("nan"),
        "ma_max": float(3.0 * u_peak_sim),
        "l2_rel": l2_rel,
        "max_rel_peak": max_rel_peak,
        "max_abs_err": max_abs,
        "Q_lo_rel": _rel(Q_lo, Q_lo_ana),
        "Q_hi_rel": _rel(Q_hi, Q_hi_ana),
        "uy_spurious_rel": (uy_max / u_peak_sim) if u_peak_sim else float("nan"),
        "mass_drift_rel": abs(float(rho[1:-1, :].sum()) - mass0) / mass0,
        "interface_width_10_90": width_10_90,
        "interface_centre": centre,
        "interface_centre_shift": abs(centre - y_int) if centre is not None else None,
        "profile": [float(v) for v in prof],
        "profile_ana": [float(v) for v in u_ana],
        "rho_profile": [float(v) for v in rho_prof],
        "tau_profile": [float(v) for v in tau.mean(dim=1).cpu().numpy()],
        "analytic_checks": sol.residuals(),
        "history": history,
    }


def _interface_width_10_90(profile: np.ndarray) -> float | None:
    """tanh 10-90 width of a monotone density profile (None if not resolvable)."""
    p = np.asarray(profile, dtype=np.float64)
    lo_plateau = float(np.median(p[: max(2, len(p) // 10)]))
    hi_plateau = float(np.median(p[-max(2, len(p) // 10) :]))
    if abs(hi_plateau - lo_plateau) < 1e-3:
        return None
    up = (p - lo_plateau) / (hi_plateau - lo_plateau)
    y = np.arange(len(p), dtype=np.float64)
    y10 = _crossing(y, up, 0.1)
    y90 = _crossing(y, up, 0.9)
    if y10 is None or y90 is None:
        return None
    return float(abs(y90 - y10))


def _crossing(y: np.ndarray, v: np.ndarray, level: float) -> float | None:
    d = v - level
    s = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if s.size == 0:
        return None
    i = int(s[0])
    dv = d[i + 1] - d[i]
    if dv == 0:
        return float(y[i])
    return float(y[i] - d[i] * (y[i + 1] - y[i]) / dv)


def _interface_centre(profile: np.ndarray, y: np.ndarray) -> float | None:
    p = np.asarray(profile, dtype=np.float64)
    lo_plateau = float(np.median(p[: max(2, len(p) // 10)]))
    hi_plateau = float(np.median(p[-max(2, len(p) // 10) :]))
    if abs(hi_plateau - lo_plateau) < 1e-3:
        return None
    up = (p - lo_plateau) / (hi_plateau - lo_plateau)
    return _crossing(y, up, 0.5)


# ---------------------------------------------------------------------------
# MCMP comparison arm (two components, per-component distributions)
# ---------------------------------------------------------------------------


def run_mcmp_two_component(
    ny: int,
    nx: int = 8,
    *,
    G: float = 4e-6,
    tau_lo: float = 2.0,
    tau_hi: float = 1.25,
    G_12: float = -1.0,
    rho_init: tuple[float, float] = (0.7, 0.3),
    width: float = 2.0,
    max_steps: int = 120_000,
    stop_tol: float = 1e-7,
    check_every: int = 200,
    device: str | torch.device = "cpu",
) -> dict:
    """Same layered Poiseuille driven through the two-component Shan-Chen path.

    This is the MCMP reference *inside this library* (two distributions, one per
    component, plus the inter-component coupling ``G_12``).  It is run so that
    the two-phase-only downgrade can be compared against a real component-
    resolved model at the same viscosity ratio, not only against the analytic
    solution.
    """
    device = torch.device(device)
    y_int = 0.5 * (ny - 1)
    C = phase_indicator(ny, nx, y_int, width, device=device)
    # Component 1 is dense in the lower layer, component 2 in the upper one.
    r1 = rho_init[0] * C + rho_init[1] * (1.0 - C)
    r2 = rho_init[1] * C + rho_init[0] * (1.0 - C)
    z = torch.zeros_like(r1)
    f1 = equilibrium(r1, z, z)
    f2 = equilibrium(r2, z, z)
    wall = _wall_mask(ny, nx, device)

    nu_lo = (tau_lo - 0.5) / 3.0
    nu_hi = (tau_hi - 0.5) / 3.0
    sol = two_layer_analytic(ny, y_int, G, nu_lo, nu_hi)
    u_ana = analytic_profile(sol, ny)

    history: list[tuple[int, float]] = []
    prev = None
    steps, converged = 0, False
    t0 = time.time()
    for step in range(1, max_steps + 1):
        f1, f2 = collide_sc_two_component(
            f1, f2, G_12=G_12, tau1=tau_lo, tau2=tau_hi, gx=G, gy=0.0,
            solid_mask=wall, u_eq="mixture", force_scheme="guo",
        )
        f1 = stream(f1)
        f2 = stream(f2)
        f1 = bounce_back_cells(f1, wall)
        f2 = bounce_back_cells(f2, wall)
        steps = step
        if step % check_every == 0:
            rho1, ux1, uy1 = macroscopic(f1)
            rho2, ux2, uy2 = macroscopic(f2)
            tot = torch.clamp(rho1 + rho2, min=1e-12)
            ux = (rho1 * ux1 + rho2 * ux2) / tot
            u_peak = float(ux[1:-1, :].abs().max())
            history.append((step, u_peak))
            if prev is not None and abs(u_peak - prev) <= stop_tol * max(abs(prev), 1e-12):
                converged = True
                break
            prev = u_peak
        if not (torch.isfinite(f1).all() and torch.isfinite(f2).all()):
            break
    wall_clock = time.time() - t0

    rho1, ux1, uy1 = macroscopic(f1)
    rho2, ux2, uy2 = macroscopic(f2)
    tot = torch.clamp(rho1 + rho2, min=1e-12)
    ux = (rho1 * ux1 + rho2 * ux2) / tot
    uy = (rho1 * uy1 + rho2 * uy2) / tot
    prof = ux.mean(dim=1).double().cpu().numpy()
    fluid = slice(1, ny - 1)
    sim, ana = prof[fluid], u_ana[fluid]
    denom = float(np.linalg.norm(ana))
    u_peak_sim = float(np.abs(sim).max())
    u_peak_ana = float(np.abs(ana).max())
    finite = bool(torch.isfinite(ux).all() and torch.isfinite(uy).all())
    return {
        "config": {
            "ny": ny, "nx": nx, "G": G, "tau_lo": tau_lo, "tau_hi": tau_hi,
            "nu_lo": nu_lo, "nu_hi": nu_hi, "G_12": G_12,
            "rho_init": list(rho_init), "width": width, "u_eq": "mixture",
            "force_scheme": "guo", "max_steps": max_steps,
        },
        "steps": steps,
        "converged": converged,
        "finite": finite,
        "wall_s": wall_clock,
        "u_peak_sim": u_peak_sim,
        "u_peak_ana": u_peak_ana,
        "l2_rel": float(np.linalg.norm(sim - ana) / denom) if denom else float("nan"),
        "max_rel_peak": float(np.max(np.abs(sim - ana))) / u_peak_ana if u_peak_ana else float("nan"),
        "profile": [float(v) for v in prof],
        "profile_ana": [float(v) for v in u_ana],
        "analytic_checks": sol.residuals(),
        "history": history,
    }