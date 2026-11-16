"""Transient (start-up) verification of the force-corrected NEBB body-force driver.

Why this module
---------------
``docs/xval_nebb_bodyforce_anchor.md`` §5 records three explicit holes in the
anchor of ``XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S``; the first one
is:

    | transient / unsteady channel start-up | — | out of scope (steady-state
      driver only) |

The steady benchmark cannot see several failure modes of the three ingredients
the driver assembles (half-force-step wall velocity, shear-reconstruction NEBB,
``+g/(2 rho)`` velocity correction): a wrong half-step *sign*, a doubled
forcing term or an inconsistent wall closure all still converge to *a* parabola
-- just to the wrong one, or with the wrong relaxation time.  The transient is
the sharper referee: it fixes both the steady amplitude **and** the decay rate
of the slowest diffusion mode, i.e. it measures the effective viscosity and the
effective gap height from a time series instead of from a single profile.

Primary referee (closed form, independent of our code)
-----------------------------------------------------
Impulsively started plane Poiseuille flow: fluid at rest for ``t < 0``, no-slip
walls at ``y = 0`` and ``y = H``, constant body force ``g_x`` switched on at
``t = 0``.  Exact solution of the incompressible Navier-Stokes equations
(Batchelor 1967 §4.2; Schlichting, *Boundary-Layer Theory*, ch. 5 "starting
flow"; the same series is quoted e.g. in White, *Viscous Fluid Flow* §3-2)::

    u_s(y)     = g_x / (2 nu) * y * (H - y)
    u(y, t)    = u_s(y) - (4 g_x H^2 / (nu pi^3))
                          * sum_{k odd} sin(k pi y / H) / k^3
                            * exp(-k^2 pi^2 nu t / H^2)

At ``t -> 0`` the series reproduces ``u_s`` exactly
(``y(H-y) = (8H^2/pi^3) sum_{k odd} sin(k pi y / H)/k^3``), so ``u = 0``;
at ``t -> inf`` only ``u_s`` survives.  The *slowest* mode decays as
``exp(-pi^2 nu t / H^2)`` with amplitude ``4 g_x H^2 / (nu pi^3)`` at mid-gap,
which is the number the decay-rate check below measures.

Wall convention (same trap as the steady anchor)
------------------------------------------------
``H`` is measured between the **no-slip planes**.  For the node-wall closure of
:mod:`tensorlbm.nebb_bodyforce` the walls sit *on* rows ``0`` and ``ny-1``, so
``H = ny-1`` and ``y`` runs over ``arange(ny)``.  The relative error of using
the half-way-bounce-back convention (``H = ny-2``, ``y = arange(ny)-0.5``)
instead is reported as a negative control -- it is a convention error, not a
numerical error, and it must be visible.

What this benchmark measures
----------------------------
* profile error vs the series at several times (lattice units, ``dt = dx = 1``);
* sine-mode amplitude ``a_1(t)`` and the fitted decay rate vs ``nu pi^2 / H^2``
  (an independent, time-domain measurement of the effective viscosity);
* mass drift, and the consistency with the steady run of
  :class:`tensorlbm.nebb_bodyforce.BodyForceChannel`;
* both forcing arms (``edm`` and ``guo``) and the two wall-density policies.

Entry point::

    python -m tensorlbm.nebb_bodyforce_startup --ny 21 --nu 0.1 --out runs/startup

The module is NumPy/float64 only (it reuses the NumPy driver); it takes no
arguments from the platform, so nothing here is domain-specific.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .nebb_bodyforce import (
    BodyForceChannel,
    ChannelConfig,
    macroscopic,
    velocity_correction,
)

__all__ = [
    "StartupSample",
    "startup_errors",
    "time_offset_scan",
    "fit_decay_rate",
    "modal_amplitude",
    "plane_poiseuille_startup",
    "run_startup",
    "steady_poiseuille",
    "startup_amplitude_envelope",
]


# --------------------------------------------------------------------------- #
# closed-form referee
# --------------------------------------------------------------------------- #
def _odd_modes(n_terms: int) -> np.ndarray:
    if n_terms < 1:
        raise ValueError("n_terms must be >= 1")
    return np.arange(1, 2 * int(n_terms), 2, dtype=np.float64)


def steady_poiseuille(y, gx: float, nu: float, H: float):
    """``u_s(y) = gx/(2 nu) y (H - y)`` (no-slip at ``y = 0`` and ``y = H``)."""
    y = np.asarray(y, dtype=np.float64)
    return gx / (2.0 * nu) * y * (H - y)


def plane_poiseuille_startup(y, t, gx: float, nu: float, H: float, n_terms: int = 512):
    """Exact start-up solution (Batchelor 1967 §4.2) at time ``t``.

    ``t`` is in the same units as ``nu`` (lattice steps for the LBM run).
    A negative ``t`` is treated as ``0``.  Terms whose exponential has decayed
    below ``1e-18`` relative are dropped, so the cost stays O(1) for late
    times; ``n_terms`` only matters near ``t = 0``.
    """
    y = np.asarray(y, dtype=np.float64)
    t = max(float(t), 0.0)
    y = np.clip(y, 0.0, H)
    us = steady_poiseuille(y, gx, nu, H)
    k = _odd_modes(n_terms)
    decay = np.exp(-(k**2) * math.pi**2 * nu * t / (H * H))
    keep = decay > 1e-18
    if not np.any(keep):
        return us
    k = k[keep]
    decay = decay[keep]
    series = np.sin(np.pi * np.outer(y, k) / H) @ ((decay / k**3))
    return us - (4.0 * gx * H * H / (nu * math.pi**3)) * series


def startup_amplitude_envelope(t, gx: float, nu: float, H: float) -> float:
    """Mid-gap amplitude of the deviation from the steady state (leading mode)."""
    t = max(float(t), 0.0)
    return float(
        4.0 * gx * H * H / (nu * math.pi**3) * math.exp(-math.pi**2 * nu * t / (H * H))
    )


def modal_amplitude(u, us, H: float) -> float:
    """Sine-projection of ``us - u`` onto ``sin(pi y / H)`` (rows on ``y=0..``)."""
    u = np.asarray(u, dtype=np.float64)
    us = np.asarray(us, dtype=np.float64)
    y = np.linspace(0.0, H, u.size)
    mode = np.sin(math.pi * y / H)
    dev = us - u
    # trapezoid over the open interior: the two wall rows carry no information
    # (they are imposed), and leaving them in biases the projection.
    return float(
        np.trapezoid(dev * mode, y) * (2.0 / H)
    )


def fit_decay_rate(t, amplitude) -> dict[str, float]:
    """Least-squares ``ln a = ln a0 - rate t``; returns rate and goodness.

    Returns ``rate`` in inverse time units (per lattice step for the LBM run),
    plus ``r2`` and the number of points used.  Values with ``a <= 0`` or a
    non-finite log are dropped; fewer than 3 usable points raise.
    """
    t = np.asarray(t, dtype=np.float64)
    a = np.asarray(amplitude, dtype=np.float64)
    ok = np.isfinite(a) & (a > 0.0) & np.isfinite(t)
    if ok.sum() < 3:
        raise ValueError(f"need >= 3 usable samples for the fit, got {int(ok.sum())}")
    slope, intercept = np.polyfit(t[ok], np.log(a[ok]), 1)
    pred = slope * t[ok] + intercept
    resid = np.log(a[ok]) - pred
    ss_tot = float(((np.log(a[ok]) - np.log(a[ok]).mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return {
        "rate": float(-slope),
        "amplitude0": float(math.exp(intercept)),
        "r2": r2,
        "n_points": int(ok.sum()),
    }


def startup_errors(got, ref, scale: float | None = None) -> dict[str, float]:
    """Error metrics of one transient snapshot against the series solution.

    Unlike :func:`tensorlbm.nebb_bodyforce.profile_errors` the relative metrics
    are normalised by the **steady-state** amplitude ``u_max = g H^2/(8 nu)``
    (the natural scale of the problem), not by ``max |ref|``: at ``t -> 0`` the
    reference profile vanishes and a self-normalising metric divides by
    rounding noise (measured: a meaningless ``2.4e6`` at ``step = 0``).
    """
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if scale is None:
        scale = max(float(np.abs(ref).max()), 1e-300)
    scale = max(float(scale), 1e-300)
    d = np.abs(got - ref)
    interior = slice(1, -1)
    floor = 1e-3 * scale
    out = {
        "max_abs": float(d.max()),
        "max_rel_to_scale": float(d.max() / scale),
        "l2_rel_to_scale": float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-300)),
        "umax_got": float(got.max()),
        "umax_ref": float(ref.max()),
        "umax_rel_err": float(abs(got.max() - ref.max()) / scale),
        "max_abs_interior": float(d[interior].max()),
        "max_rel_interior_to_scale": float(d[interior].max() / scale),
        "max_rel_interior_pointwise": float(
            (d[interior] / np.maximum(np.abs(ref[interior]), floor)).max()
        ),
    }
    return out


def time_offset_scan(
    profile,
    y,
    t: float,
    gx: float,
    nu: float,
    H: float,
    offsets: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
    n_terms: int = 512,
) -> dict[str, float]:
    """Error of one snapshot against the series evaluated at ``t + offset``.

    The force-corrected driver reports ``u + g/(2 rho)`` from the *pre-collision*
    moment, so its output can legitimately sit at a half time step away from the
    nominal integer step.  Scanning the offset turns that convention question
    into a measurement instead of an assumption: the best-fitting offset is the
    time level the driver's velocity actually lives on, and the residual at that
    offset is the physical (compressibility / start-up) error, not a
    bookkeeping artefact.
    """
    scale = float(steady_poiseuille(y, gx, nu, H).max())
    out = {}
    for o in offsets:
        ref = plane_poiseuille_startup(y, t + o, gx, nu, H, n_terms=n_terms)
        out[f"{o:+.1f}"] = float(np.abs(np.asarray(profile) - ref).max() / scale)
    return out


# --------------------------------------------------------------------------- #
# LBM transient driver
# --------------------------------------------------------------------------- #
@dataclass
class StartupSample:
    """One transient snapshot compared against the closed form."""

    step: int
    t: float
    profile: np.ndarray
    analytic: np.ndarray
    errors: dict[str, float]
    mode_amplitude: float
    offset_scan: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "t": self.t,
            "mode_amplitude": self.mode_amplitude,
            "umax": float(self.profile.max()),
            "umax_analytic": float(self.analytic.max()),
            "errors": self.errors,
            "time_offset_scan_max_rel": self.offset_scan,
        }


@dataclass
class StartupResult:
    """Full transient record of one run."""

    config: dict[str, Any]
    samples: list[StartupSample] = field(default_factory=list)
    mass_initial: float = 0.0
    mass_final: float = 0.0
    steady_reference: np.ndarray | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "samples": [s.as_dict() for s in self.samples],
            "mass_initial": self.mass_initial,
            "mass_final": self.mass_final,
            "mass_drift_rel": self.mass_drift_rel,
        }

    @property
    def mass_drift_rel(self) -> float:
        if self.mass_initial == 0.0:
            return float("nan")
        return float(abs(self.mass_final - self.mass_initial) / abs(self.mass_initial))


def _mid_column_profile(f, gx: float, gy: float = 0.0) -> np.ndarray:
    """Force-corrected physical velocity ``u + g/(2 rho)``, averaged over ``x``."""
    rho, ux, uy = macroscopic(f)
    uxc = ux + velocity_correction(gx, rho)
    return uxc.mean(axis=0)


def run_startup(
    ny: int = 21,
    nx: int = 2,
    nu: float = 0.1,
    gx: float | None = None,
    scheme: str = "edm",
    wall: str = "force_corrected_nebb",
    sample_steps: Sequence[int] | None = None,
    n_terms: int = 512,
) -> StartupResult:
    """Run the start-from-rest channel and compare against the closed form.

    The flow is initialised at rest (``init="rest"``, i.e. ``u = 0`` everywhere),
    which is precisely the initial condition of the analytic problem -- the
    OSS example's ``force_shifted`` start would inject a ``-g/2`` profile that
    is *not* part of the reference solution.

    All quantities are in lattice units (``dx = dt = 1``); ``H = ny - 1`` is the
    node-wall gap.  ``gx`` defaults to the value giving ``u_max = 0.01``
    (Mach 0.017, comfortably incompressible).
    """
    H = float(ny - 1)
    if gx is None:
        gx = 8.0 * nu * 0.01 / (H * H)
    if sample_steps is None:
        # 0 at the start, then logarithmically-ish up to ~6 diffusion times
        t_diff = H * H / (math.pi**2 * nu)
        sample_steps = sorted(
            {0}
            | {
                max(1, int(round(f * t_diff)))
                for f in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
            }
        )
    cfg = ChannelConfig(
        nx=nx,
        ny=ny,
        nu=nu,
        gx=gx,
        scheme=scheme,
        wall=wall,
        init="rest",
        steps=int(max(sample_steps)),
        steady_tol=0.0,
        check_every=0,
    )
    sim = BodyForceChannel(cfg)
    y = np.arange(ny, dtype=np.float64)
    res = StartupResult(
        config={
            "nx": nx,
            "ny": ny,
            "nu": nu,
            "gx": gx,
            "scheme": scheme,
            "wall": wall,
            "init": "rest",
            "H": H,
            "tau": cfg.tau,
            "umax_steady_analytic": float(steady_poiseuille(y, gx, nu, H).max()),
            "diffusion_time": H * H / (math.pi**2 * nu),
            "sample_steps": list(sample_steps),
        },
        mass_initial=float(sim.f.sum()),
    )
    us = steady_poiseuille(y, gx, nu, H)
    scale = float(us.max())
    step_set = sorted(set(int(s) for s in sample_steps))
    cur = 0
    for target in step_set:
        while cur < target:
            sim.step()
            cur += 1
        prof = _mid_column_profile(sim.f, gx)
        ana = plane_poiseuille_startup(y, cur, gx, nu, H, n_terms=n_terms)
        res.samples.append(
            StartupSample(
                step=cur,
                t=float(cur),
                profile=prof,
                analytic=ana,
                errors=startup_errors(prof, ana, scale=scale),
                mode_amplitude=modal_amplitude(prof, us, H),
                offset_scan=time_offset_scan(prof, y, cur, gx, nu, H),
            )
        )
    res.mass_final = float(sim.f.sum())
    res.steady_reference = us
    return res


def decay_rate_report(
    result: StartupResult, fit_from_factor: float = 0.5
) -> dict[str, Any]:
    """Fit the slowest-mode decay rate from the samples and compare to theory.

    Only samples with ``t >= fit_from_factor * H^2 / (pi^2 nu)`` are used: by
    then the ``k = 3, 5, ...`` modes have decayed by ``e^{-8}`` and ``e^{-24}``
    respectively, so the projection is a clean single-mode measurement.
    """
    cfg = result.config
    nu, H = float(cfg["nu"]), float(cfg["H"])
    t_fit_min = fit_from_factor * H * H / (math.pi**2 * nu)
    pts = [(s.t, s.mode_amplitude) for s in result.samples if s.t >= t_fit_min]
    theory = math.pi**2 * nu / (H * H)
    out: dict[str, Any] = {
        "rate_theory": theory,
        "fit_from_t": t_fit_min,
        "n_points": len(pts),
    }
    if len(pts) < 3:
        out["rate_measured"] = float("nan")
        out["rate_rel_err"] = float("nan")
        return out
    fit = fit_decay_rate([p[0] for p in pts], [p[1] for p in pts])
    out["rate_measured"] = fit["rate"]
    out["amplitude0"] = fit["amplitude0"]
    out["r2"] = fit["r2"]
    out["n_points_used"] = fit["n_points"]
    out["rate_rel_err"] = abs(fit["rate"] - theory) / theory
    out["nu_effective"] = fit["rate"] * H * H / math.pi**2
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _table(rows: Sequence[dict[str, Any]]) -> str:
    head = (
        f"{'scheme':<6}{'wall':<26}{'ny':>3}{'nu':>6}{'step':>7}"
        f"{'t/tD':>7}{'umax':>11}{'ana':>11}{'err/scale':>11}{'a1':>11}{'best_off':>10}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['scheme']:<6}{r['wall']:<26}{r['ny']:>3}{r['nu']:>6.3f}"
            f"{r['step']:>7}{r['t_over_tD']:>7.2f}{r['umax']:>11.3e}"
            f"{r['umax_analytic']:>11.3e}{r['max_rel_to_scale']:>11.3e}"
            f"{r['mode_amplitude']:>11.3e}{r['best_offset']:>10}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ny", type=int, default=21)
    ap.add_argument("--nx", type=int, default=2)
    ap.add_argument("--nu", type=float, default=0.1)
    ap.add_argument("--gx", type=float, default=None)
    ap.add_argument("--scheme", default="edm", choices=("edm", "guo"))
    ap.add_argument("--wall", default="force_corrected_nebb")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    res = run_startup(
        ny=args.ny, nx=args.nx, nu=args.nu, gx=args.gx, scheme=args.scheme, wall=args.wall
    )
    rows = []
    t_diff = res.config["diffusion_time"]
    for s in res.samples:
        rows.append(
            {
                "scheme": res.config["scheme"],
                "wall": res.config["wall"],
                "ny": args.ny,
                "nu": args.nu,
                "step": s.step,
                "t_over_tD": s.t / t_diff,
                "umax": s.errors["umax_got"],
                "umax_analytic": s.errors["umax_ref"],
                "max_rel_to_scale": s.errors["max_rel_to_scale"],
                "mode_amplitude": s.mode_amplitude,
                "best_offset": min(s.offset_scan, key=s.offset_scan.get) if s.offset_scan else "-",
            }
        )
    print(_table(rows))
    fit = decay_rate_report(res)
    print(
        f"decay rate: measured={fit['rate_measured']:.6e} theory={fit['rate_theory']:.6e} "
        f"rel_err={fit['rate_rel_err']:.3e} r2={fit.get('r2', float('nan')):.6f} "
        f"nu_eff={fit.get('nu_effective', float('nan')):.6f}"
    )
    print(f"mass_drift_rel={res.mass_drift_rel:.3e}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = res.summary()
        payload["decay"] = fit
        payload["profiles_final"] = {
            "y": list(range(args.ny)),
            "lbm": res.samples[-1].profile.tolist(),
            "analytic": res.samples[-1].analytic.tolist(),
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())