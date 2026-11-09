"""Static contact-angle benchmark harness for the SCMP wetting wall force.

Question the harness answers
----------------------------
The pseudopotential (Shan-Chen) wall force exposes one knob -- the wall
pseudopotential ``psi_w``, given as a fraction ``f_w = psi_w/psi(rho_l)`` of the
bulk liquid pseudopotential.  A downstream simulation (e.g. the layered MCMP
Poiseuille node ``GAP-LBM-MCMP-POISEUILLE``) needs to *set* a contact angle with
that knob.  This harness measures what angle the knob actually produces, on two
resolutions, and evaluates a pre-registered acceptance criterion

    |theta_measured - theta_prescribed| < 5 deg

for **two competing prescription laws** (this is the experiment: which law, if
any, is the knob calibrated against):

``young``
    The thermodynamic (Young) relation that the library's own docstrings cite,

        cos(theta) = (psi_w - psi_v) / (psi_l - psi_v)
        => f_w(theta) = [psi_v + cos(theta) (psi_l - psi_v)] / psi_l.

    ``psi_l`` and ``psi_v`` are taken from the library's own pseudo-potential
    and coexistence pair -- no fitted parameter enters.  If the wall force
    realised the Young relation, this law would be exact.

``calibrated``
    The null-model alternative: the knob is ``f_w``-monotone but not Young; then
    the angle must be prescribed through the *measured* curve ``theta(f_w)``.
    The curve is fitted on the **coarse** grid and the prescription is then
    verified on the **fine** grid, so the verification is a hold-out (a
    calibration that only reproduced itself would not pass).

Both laws are evaluated on the same three prescribed angles; the report states
which one passes, and the deviation of the other is the quantitative finding.

Geometry and conventions
------------------------
* Drop initialised as the *exact* circular segment of ``theta_prescribed``
  (:func:`~tensorlbm.contact_line.engine.sessile_drop_state` with the centre at
  ``y_wall - R cos(theta)``), so relaxation only removes the discretisation
  error of the level set.  Starting from a 90 deg blob instead leaves the
  measured angle drifting for >10^4 steps and is not used.
* ``theta`` is the physical angle inside the liquid (Young's convention).
  The library's contour estimators return its supplement -- see
  :mod:`contact_angle`, which converts.
* Primary estimator: ``cap`` (liquid-content area + apex height, validated to
  <2 deg on analytic caps of 60-130 deg in the pytest module).  The whole-cap
  circle fit and the near-wall band fit are recorded alongside as cross-checks
  (``spread`` = their max-min, the instrument's uncertainty floor).
* A case is only judged if it has *relaxed*: the angle drift over the last 25 %
  of the run must be < 1 deg, otherwise the case is flagged ``UNCONVERGED``
  and excluded from the verdict rather than silently averaged.

Everything numeric in the report/figures is injected from the JSON artifact
written by ``assemble.py``; nothing is transcribed by hand.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from tensorlbm.contact_line.engine import (
    RHO_L_DEFAULT,
    RHO_V_DEFAULT,
    Channel,
    WallStepper,
    sessile_drop_state,
)

from contact_angle import measure_contact_angle

__all__ = [
    "ACCEPT_DEG",
    "CALIB_FRACTIONS",
    "PRESCRIBED_DEG",
    "CaseResult",
    "Grid",
    "GRIDS",
    "cos_from_wall_fraction",
    "measure_contact_angle",
    "reachable_angle_limit_deg",
    "run_case",
    "theta_from_wall_fraction",
    "wall_fraction_for_angle",
]

#: Pre-registered acceptance tolerance on ``|theta_measured - theta_prescribed|``.
ACCEPT_DEG = 5.0

#: Pre-registered angle ladder (degrees, physical convention).
PRESCRIBED_DEG: tuple[float, ...] = (70.0, 90.0, 110.0)

#: Wall fractions scanned on the coarse grid to build the measured calibration.
CALIB_FRACTIONS: tuple[float, ...] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80)

#: Pre-registered convergence criterion (deg of drift over the last 25 % of the run).
CONVERGED_DRIFT_DEG = 1.0


@dataclass(frozen=True)
class Grid:
    """Wall-bounded channel + droplet size for one resolution level."""

    name: str
    nx: int
    ny: int
    radius: float
    steps: int
    wall_rows: int = 2
    samples: int = 8

    @property
    def channel(self) -> Channel:
        return Channel(nx=self.nx, ny=self.ny, wall_rows=self.wall_rows)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GRIDS: tuple[Grid, ...] = (
    Grid(name="coarse", nx=96, ny=64, radius=13.0, steps=8_000),
    Grid(name="fine", nx=144, ny=96, radius=20.0, steps=12_000),
)


# ---------------------------------------------------------------------------
# Young relation for the pseudopotential wall
# ---------------------------------------------------------------------------


def cos_from_wall_fraction(frac: float, psi_l: float, psi_v: float) -> float:
    """``cos(theta)`` predicted by the Shan-Chen wall (Young) relation."""
    if psi_l == psi_v:
        raise ValueError("degenerate coexistence: psi_l == psi_v")
    return (float(frac) * float(psi_l) - float(psi_v)) / (float(psi_l) - float(psi_v))


def theta_from_wall_fraction(frac: float, psi_l: float, psi_v: float) -> float:
    """Contact angle (degrees, inside the liquid) of a wall fraction ``f_w``."""
    c = min(1.0, max(-1.0, cos_from_wall_fraction(frac, psi_l, psi_v)))
    return math.degrees(math.acos(c))


def wall_fraction_for_angle(theta_eq_deg: float, psi_l: float, psi_v: float) -> float:
    """Inverse of :func:`theta_from_wall_fraction` (the ``young`` prescription).

    Raises:
        ValueError: if the requested angle needs a *negative* wall
            pseudopotential, which the non-negative ``psi`` cannot represent.
    """
    frac = (float(psi_v) + math.cos(math.radians(float(theta_eq_deg))) * (float(psi_l) - float(psi_v))) / float(psi_l)
    if frac < 0.0:
        raise ValueError(
            f"theta={theta_eq_deg} deg needs a negative wall pseudopotential (f_w={frac:.4f}); "
            f"reachable range with psi_w>=0 is 0 .. {reachable_angle_limit_deg(psi_l, psi_v):.2f} deg"
        )
    return frac


def reachable_angle_limit_deg(psi_l: float, psi_v: float) -> float:
    """Largest angle representable with ``psi_w >= 0`` (i.e. ``psi_w = 0``)."""
    return theta_from_wall_fraction(0.0, psi_l, psi_v)


# ---------------------------------------------------------------------------
# One relaxed sessile drop
# ---------------------------------------------------------------------------


def cap_init(grid: Grid, theta_deg: float) -> tuple[torch.Tensor, Channel]:
    """Exact circular segment of ``theta_deg`` (physical convention) on the wall."""
    channel = grid.channel
    f, _ = sessile_drop_state(
        channel,
        radius=grid.radius,
        height=-grid.radius * math.cos(math.radians(theta_deg)),
    )
    return f, channel


def _psi_pair() -> tuple[float, float]:
    """``(psi_l, psi_v)`` of the library's coexistence pair, from the library itself."""
    stepper = WallStepper(Channel(nx=16, ny=16), wall_fraction=1.0)
    return float(stepper.psi_l), float(stepper.psi_fn(torch.tensor(stepper.rho_v)))


def run_case(
    theta_eq_deg: float,
    grid: Grid,
    *,
    wall_fraction: float | None = None,
    law: str = "young",
    init_deg: float | None = None,
    tol_deg: float = ACCEPT_DEG,
    threads: int = 4,
    keep_points: bool = True,
) -> dict[str, Any]:
    """Relax one sessile drop and return the prescribed-vs-measured record.

    Args:
        theta_eq_deg:  Prescribed angle (degrees, physical convention).
        grid:          Resolution level.
        wall_fraction: ``f_w = psi_w/psi_l``; when ``None`` it comes from the
                       ``law`` (``"young"``) or, for a calibration scan, from the
                       caller (``theta_eq_deg`` is then only the initial shape).
        law:           ``"young"`` (analytic prescription) or ``"scan"`` (no
                       prescription; the case only contributes a point to the
                       measured ``theta(f_w)`` curve).
        init_deg:      Angle of the initial cap (defaults to ``theta_eq_deg``).
        tol_deg:       Acceptance tolerance.
        threads:       Torch threads for this process.
        keep_points:   Store the down-sampled interface contour in the record.
    """
    torch.set_num_threads(int(threads))
    t0 = time.time()
    psi_l, psi_v = _psi_pair()
    record: dict[str, Any] = {
        "kind": "SCMP",
        "law": law,
        "theta_eq": float(theta_eq_deg),
        "grid": grid.name,
        "grid_meta": grid.to_dict(),
        "psi_l": psi_l,
        "psi_v": psi_v,
        "theta_max_psi_ge_0": reachable_angle_limit_deg(psi_l, psi_v),
        "theta_young": None,
    }

    if wall_fraction is None:
        try:
            wall_fraction = wall_fraction_for_angle(theta_eq_deg, psi_l, psi_v)
        except ValueError as exc:
            record.update(status="OUT_OF_RANGE", reason=str(exc), wall_fraction=None,
                          theta_measured=None, delta=None, trace=[], points=[],
                          runtime_s=round(time.time() - t0, 2))
            return record
    record["wall_fraction"] = float(wall_fraction)
    record["theta_young"] = theta_from_wall_fraction(wall_fraction, psi_l, psi_v)

    f, channel = cap_init(grid, theta_eq_deg if init_deg is None else init_deg)
    stepper = WallStepper(channel, wall_fraction=float(wall_fraction))
    solid = channel.solid_mask().numpy()

    m0 = float(stepper.density(f).sum())
    every = max(1, grid.steps // max(1, grid.samples))
    trace: list[dict[str, float]] = []
    step = 0
    for _ in range(grid.samples):
        f = stepper.run(f, every, every=0)
        step += every
        m = measure_contact_angle(stepper.density(f).numpy(), rho_l=stepper.rho_l,
                                 rho_v=stepper.rho_v, y_wall=channel.y_wall(),
                                 solid=solid, radius=grid.radius)
        trace.append({"step": float(step), "theta": float(m["theta"]),
                      "theta_circle": float(m["theta_circle"]),
                      "theta_band": float(m["theta_band"])})
    final = measure_contact_angle(stepper.density(f).numpy(), rho_l=stepper.rho_l,
                                 rho_v=stepper.rho_v, y_wall=channel.y_wall(),
                                 solid=solid, radius=grid.radius, keep_points=keep_points)
    rho = stepper.density(f)
    spurious = stepper.spurious_current(f)
    theta_meas = float(final["theta"])
    delta = theta_meas - theta_eq_deg
    finite_trace = [t["theta"] for t in trace if math.isfinite(t["theta"])]
    n_tail = max(1, len(finite_trace) // 4)
    drift = (abs(finite_trace[-1] - finite_trace[-1 - n_tail])
             if len(finite_trace) > n_tail else float("nan"))
    converged = math.isfinite(drift) and drift < CONVERGED_DRIFT_DEG
    if not math.isfinite(theta_meas):
        status = "NO_INTERFACE"
    elif not converged:
        status = "UNCONVERGED"
    elif abs(delta) < tol_deg:
        status = "PASS"
    else:
        status = "FAIL"
    record.update(
        theta_measured=theta_meas,
        theta_measured_circle=float(final["theta_circle"]),
        theta_measured_band=float(final["theta_band"]),
        estimator_spread=float(final["spread"]),
        rms_fit=float(final["rms_fit"]),
        n_points=int(final["n_points"]),
        apex_height=float(final["apex_height"]),
        area=float(final["area"]),
        delta=float(delta),
        abs_delta=abs(delta),
        tol_deg=float(tol_deg),
        status=status,
        converged=bool(converged),
        drift_deg=float(drift),
        mass_start=m0,
        mass_end=float(rho.sum()),
        mass_drift=float(rho.sum() - m0),
        spurious_max=float(spurious["max"]),
        trace=trace,
        points=final.get("points", []),
    )
    record["runtime_s"] = round(time.time() - t0, 2)
    return record


# ---------------------------------------------------------------------------
# calibration curve (measured theta(f_w)) and its inversion
# ---------------------------------------------------------------------------


def calibration_curve(scan: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Monotone ``theta(f_w)`` from scan cases (sorted by measured angle).

    Only converged, finite cases enter.  The curve is returned both as raw
    points and as the arrays needed by :func:`invert_calibration`.
    """
    pts = sorted(
        ((c["theta_measured"], c["wall_fraction"], c["grid"], c["theta_eq"])
         for c in scan
         if math.isfinite(c.get("theta_measured", float("nan"))) and c.get("converged")),
        key=lambda t: t[0],
    )
    return {
        "theta_deg": [p[0] for p in pts],
        "wall_fraction": [p[1] for p in pts],
        "source_grid": [p[2] for p in pts],
        "source_theta_init": [p[3] for p in pts],
        "monotone_decreasing_fw": bool(all(
            pts[i][1] >= pts[i + 1][1] - 1e-12 for i in range(len(pts) - 1))),
    }


def invert_calibration(curve: dict[str, Any], theta_deg: float) -> float:
    """Wall fraction predicted to give ``theta_deg`` (piecewise-linear inverse).

    Piecewise linear in ``theta`` on the *measured* points -- no functional form
    is assumed, so the inversion cannot smuggle in the Young relation it is
    supposed to be tested against.
    """
    th = np.asarray(curve["theta_deg"], dtype=float)
    fw = np.asarray(curve["wall_fraction"], dtype=float)
    if th.size < 2:
        raise ValueError("calibration needs at least two measured points")
    if not (th[0] <= theta_deg <= th[-1]):
        raise ValueError(
            f"theta={theta_deg:.2f} deg outside the calibrated range "
            f"[{th[0]:.2f}, {th[-1]:.2f}] deg")
    return float(np.interp(theta_deg, th, fw))


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def _git_rev(repo_root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover - only when git is unavailable
        return None


def sweep_meta(repo_root: Path, *, threads: int | None = None, workers: int | None = None) -> dict[str, Any]:
    """Provenance block written next to the numbers (environment of record)."""
    psi_l, psi_v = _psi_pair()
    return {
        "benchmark": "wetting_contact_angle",
        "anchors": "GAP-LBM-MCMP-POISEUILLE (wetting criterion)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_rev": _git_rev(repo_root),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "threads_per_worker": threads,
        "workers": workers,
        "coexistence": {"rho_l": RHO_L_DEFAULT, "rho_v": RHO_V_DEFAULT,
                        "psi_l": psi_l, "psi_v": psi_v},
        "accept_deg": ACCEPT_DEG,
        "converged_drift_deg": CONVERGED_DRIFT_DEG,
        "prescribed_deg": list(PRESCRIBED_DEG),
        "calib_fractions": list(CALIB_FRACTIONS),
        "grids": [g.to_dict() for g in GRIDS],
        "estimator": "cap (liquid-content area + apex height); circle/band cross-checks",
        "library_convention": "theta_lib = 180 - theta_physical (converted in contact_angle.py)",
    }


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path