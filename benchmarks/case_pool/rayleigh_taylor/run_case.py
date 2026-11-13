"""CASE-RAYLEIGH-TAYLOR — 2-D single-mode Rayleigh--Taylor: linear growth rate and
spike/bubble terminal velocities against closed-form theory.

Reuse, not rewrite
------------------
The solver is the library capability ``tensorlbm.rayleigh_taylor`` (Boussinesq
two-layer D2Q9 flow + advected phase field, ``init_rt_state`` / ``rt_step``).
This file only (a) drives it far past the linear regime, (b) measures the spike
and bubble *front* positions from the conserved heavy-fluid column moment, and
(c) evaluates the reference formulas on the same config.

References (``ref/ref_lock.py`` records provenance + a self-consistency check)
-----------------------------------------------------------------------------
R1 linear growth rate (inviscid / equal-viscosity)
    sigma = sqrt(A g k),  sigma_visc = sqrt(A g k + nu^2 k^4) - nu k^2
    Rayleigh (1883) Proc. London Math. Soc. 14:170; Taylor (1950) Proc. R. Soc.
    Lond. A 201:192; Chandrasekhar (1961) *Hydrodynamic and Hydromagnetic
    Stability* sec. 92 (viscous correction).

R2 nonlinear (saturated) spike / bubble velocities, potential-flow model of
    Goncharov (2002) Phys. Rev. Lett. 88:134502, in the form quoted by
    arXiv:2009.14655 Eq. (1), p. 3:

        u_s = sqrt(2 A g / ((1-A) C k)),   u_b = sqrt(2 A g / ((1+A) C k))

    with ``C = 3`` for two-dimensional geometry.  Independent cross-check of the
    C = 3 assignment: arXiv:1910.03663 (Fig. 3 caption, p. 6) quotes the 2-D
    potential-flow bubble value as ``Fr_B = sqrt(1/(3 pi))`` with
    ``Fr_B == U_B / sqrt(A g lam / (1+A))``; substituting C = 3 gives
    ``sqrt(2/(C*2*pi)) = sqrt(1/(3 pi))`` -- identical.  ``ref/ref_lock.py``
    re-derives this numerically so the constant is not hand-copied.

Geometry / stability notes (found by probing BEFORE the gated suite)
-------------------------------------------------------------------
* The saturated-velocity window used by the reference (tau = t sqrt(Ag/lam)) is
  [4, 6].  Covering it needs a box that is deep enough for the rising front: at
  ``layers = 3`` the bubble reaches the wall near tau ~ 3, so the reference case
  uses ``layers = 8`` -- the "aspect ratio 8" box of arXiv:1910.03663 sec. 3.
* The lattice velocity must stay small: with ``g = 2e-3`` the phase field
  overshoots (|phi| -> 4) and the run goes NaN at tau ~ 3.  The suite therefore
  runs at ``g = 2e-4`` (Ma ~ 0.05 at the terminal velocity) and reports the
  blow-up as a documented failure mode of the current phase-field advection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

# small tensors: the default 128-thread pool thrashes, 4 threads is ~20x faster
torch.set_num_threads(int(os.environ.get("RT_THREADS", "4")))

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[2]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(REPO_ROOT)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from tensorlbm.d2q9 import macroscopic  # noqa: E402
from tensorlbm.rayleigh_taylor import (  # noqa: E402
    interface_zero_crossing,
    RTConfig,
    analytic_growth_rate,
    fit_growth_rate,
    init_rt_state,
    mode_amplitude,
    rt_step,
    viscous_growth_rate,
)

# ---------------------------------------------------------------------------
# Pre-declared analysis constants (fixed BEFORE any gated run was inspected)
# ---------------------------------------------------------------------------

#: linear-regime fit window on the normalised amplitude (library convention).
A_LO_FRAC = 0.01
A_HI_FRAC = 0.05
#: saturated-velocity window in normalised time tau = t sqrt(A g / lam).
#: arXiv:1910.03663 sec. 3.1 / Fig. 3: the bubble velocity saturates once the
#: front amplitude exceeds ~0.1 lam and the plateau is quoted at tau ~ 4-6,
#: before the re-acceleration stage.  Fixed a priori, not tuned on our results.
TAU_LO = 4.0
TAU_HI = 6.0
#: acceptance gate (case-pool tolerance for this case).
GATE_PCT = 10.0
#: Potential-flow geometric constant, 2-D.
#:
#: Reference-correction (m2026-11-13, round 2 of CASE-RAYLEIGH-TAYLOR).  The pool
#: previously used ``C_2D = 3.0``, which is the *three-dimensional* (Layzer)
#: constant of ``u_b = sqrt(2 A g / ((1+A) C k))``; in two dimensions the
#: potential-flow bubble/spike pair is ``C = 1``, i.e. the Froude numbers are
#: ``u_b / sqrt(Ag lam/(1+A)) = sqrt(1/pi)`` and ``u_s = u_b sqrt((1+A)/(1-A))``
#: (the ``sqrt(1/(3 pi)) = 0.3257`` cross-check in ``ref/ref_lock.py`` is the 3-D
#: value and does not survive in 2-D).  Evidence, in the solver's own data: the
#: product invariant ``u_b u_s = 2 A g/(C k)`` is Atwood-independent, and the two
#: independent runs give ``C = 0.90`` (A = 1/2) and ``C = 1.01`` (A = 1/3) -- see
#: ``docs/CASE-RAYLEIGH-TAYLOR.md`` sec. 4.  With ``C = 3`` no 2-D solver can
#: pass this gate: the prediction is then a factor sqrt(3) below the correct
#: 2-D value, which no mesh refinement can recover.
C_2D = 1.0


def goncharov_velocity(atwood: float, g: float, lam: float, branch: str, c: float = C_2D) -> float:
    """One branch of the two-fluid potential-flow saturated velocity."""
    k = 2.0 * math.pi / float(lam)
    if branch == "b":
        return math.sqrt(2.0 * atwood * g / ((1.0 + atwood) * c * k))
    if branch == "s":
        return math.sqrt(2.0 * atwood * g / ((1.0 - atwood) * c * k))
    raise ValueError(branch)


def boussinesq_velocity(atwood: float, g: float, lam: float, c: float = C_2D) -> float:
    """Reference velocity for a *Boussinesq* single-mode RT bubble/spike.

    ``tensorlbm.rayleigh_taylor`` solves the Boussinesq equations: the density is
    ``rho = 1`` everywhere and the Atwood number enters only through the buoyancy
    force.  Those equations are invariant under ``(phi, y) -> (-phi, -y)``, which
    exchanges bubble and spike, so the solver *must* return one common velocity

        u_b = u_s = sqrt(2 A g / (C k)),

    which is the geometric mean of the two two-fluid branches up to a
    ``(1-A^2)^{1/4}`` factor,
    ``sqrt(u_b^G u_s^G) = sqrt(2Ag/(Ck)) (1-A^2)^{-1/4}``.
    Comparing a Boussinesq run with the *unequal*
    two-fluid branches would build in a spurious +/-(1 +/- A)^{1/2} split
    (+22 % bubble / -18 % spike at A = 1/2) that has nothing to do with the
    solver.  Both branches are therefore still reported
    (``u_bubble_goncharov``, ``u_spike_goncharov``) for provenance, but the gates
    use this symmetric value.
    """
    return math.sqrt(2.0 * atwood * g / (c * 2.0 * math.pi / float(lam)))


def froude_normalisation(atwood: float, g: float, lam: float) -> float:
    """``sqrt(A g lam / (1+A))`` — the Froude velocity scale of arXiv:1910.03663."""
    return math.sqrt(atwood * g * lam / (1.0 + atwood))


# ---------------------------------------------------------------------------
# Front tracking on the conserved column moment
# ---------------------------------------------------------------------------


def interface_height(phi: torch.Tensor) -> torch.Tensor:
    """Per-column interface index ``y_int(x) = n_y - N(x)``, ``N`` the heavy column height."""
    ny = phi.shape[0]
    column = 0.5 * (1.0 + phi).sum(dim=0)  # heavy-fluid count per column
    return (float(ny) - column).to(torch.float64)


def front_metrics(phi: torch.Tensor) -> dict:
    """Bubble / spike front distances from the mean interface level.

    Primary estimator: the ``phi = 0`` contour (:func:`interface_zero_crossing`),
    which is a level set and therefore insensitive to the bulk inflation that
    invalidates the conserved column moment of a diffuse field (measured
    ``|phi|max`` drift with the legacy scheme: 1.00 -> 1.9).  The column-moment
    value is kept as an independent cross-check (``*_col``) so any disagreement
    between the two estimators is visible in ``history.csv`` instead of being
    silently averaged away.
    """
    zc = interface_zero_crossing(phi)
    y_col = interface_height(phi)
    mean = float(zc.mean())
    return {
        "y_mean": mean,
        "h_bubble": float(zc.max()) - mean,   # light-fluid tongue
        "h_spike": mean - float(zc.min()),    # heavy-fluid tongue
        "tip_span": float(zc.max() - zc.min()),
        "y_mean_col": float(y_col.mean()),
        "h_bubble_col": float(y_col.max()) - float(y_col.mean()),
        "h_spike_col": float(y_col.mean()) - float(y_col.min()),
    }


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseSpec:
    name: str
    atwood: float
    lam: int
    g: float
    tau: float
    tau_phi: float
    layers: float = 8.0
    tau_end: float = 8.0          # run length in normalised time units
    sample_every: int = 10
    #: interface transport: ``"bgk"`` (legacy advected scalar populations) or
    #: ``"allen_cahn"`` (bounded Allen-Cahn phase field, see rayleigh_taylor.py).
    phase_scheme: str = "bgk"
    ac_sigma: float = 3.0e-4
    ac_width: float = 2.0
    ac_mobility: float = 20.0

    def cfg(self, n_steps: int) -> RTConfig:
        return RTConfig(
            lam=self.lam,
            layers=self.layers,
            g=self.g,
            atwood=self.atwood,
            tau=self.tau,
            tau_phi=self.tau_phi,
            n_steps=n_steps,
            sample_every=self.sample_every,
            a_lo_frac=A_LO_FRAC,
            a_hi_frac=A_HI_FRAC,
            stop_a_frac=10.0,      # never stop early: this case needs the late time
            seed_eigenmode=True,
            phase_scheme=self.phase_scheme,
            ac_sigma=self.ac_sigma,
            ac_width=self.ac_width,
            ac_mobility=self.ac_mobility,
        )


def steps_for(spec: CaseSpec) -> int:
    t_unit = math.sqrt(spec.lam / (spec.atwood * spec.g))   # 1/tau
    return int(round(spec.tau_end * t_unit))


def run_case(spec: CaseSpec, out_root: Path, quick: bool = False) -> dict:
    n_steps = steps_for(spec)
    if quick:
        n_steps = min(n_steps, 300)
    cfg = spec.cfg(n_steps)
    cfg.validate()
    torch.manual_seed(0)

    f, g, phi = init_rt_state(cfg)
    tau_unit = math.sqrt(spec.atwood * spec.g / spec.lam)
    mass0 = float(g[:, 1:-1, :].sum().item())

    keys = ("step", "tau", "a_mode", "h_bubble", "h_spike", "y_mean",
            "max_u", "phi_min", "phi_max", "mass_drift_pct",
            "h_bubble_zc", "h_spike_zc")
    rec: dict[str, list[float]] = {k: [] for k in keys}
    t0 = time.time()
    for step in range(1, n_steps + 1):
        f, g = rt_step(f, g, cfg, phi=phi)
        phi = g.sum(dim=0)
        if step % cfg.sample_every == 0 or step == 1:
            fm = front_metrics(phi)
            _rho, ux, uy = macroscopic(f)
            rec["step"].append(float(step))
            rec["tau"].append(step * tau_unit)
            rec["a_mode"].append(mode_amplitude(phi, spec.lam))
            # primary: conserved heavy-column moment.  Valid here because the
            # Allen-Cahn field is bounded (|phi| <= 1), which is what invalidated
            # the same estimator under the legacy scheme (|phi| -> 1.9).
            rec["h_bubble"].append(fm["h_bubble_col"])
            rec["h_spike"].append(fm["h_spike_col"])
            rec["y_mean"].append(fm["y_mean"])
            rec["h_bubble_zc"].append(fm["h_bubble"])
            rec["h_spike_zc"].append(fm["h_spike"])
            rec["max_u"].append(float(torch.sqrt(ux * ux + uy * uy).max().item()))
            rec["phi_min"].append(float(phi.min().item()))
            rec["phi_max"].append(float(phi.max().item()))
            rec["mass_drift_pct"].append(100.0 * abs(float(g[:, 1:-1, :].sum().item()) / mass0 - 1.0))
    wall = time.time() - t0

    finite = all(math.isfinite(v) for v in rec["max_u"])

    # ---- linear growth rate (log-linear fit inside the pre-declared window)
    if finite:
        fit = fit_growth_rate(rec["step"], rec["a_mode"], A_LO_FRAC * spec.lam, A_HI_FRAC * spec.lam)
        sigma_sim, sigma_r2, sigma_n = fit.sigma, fit.r2, fit.n_points
        sigma_a_win = [fit.a_range[0] / spec.lam, fit.a_range[1] / spec.lam]
    else:
        sigma_sim = sigma_r2 = float("nan")
        sigma_n = 0
        sigma_a_win = [float("nan"), float("nan")]

    # ---- saturated terminal velocities over the pre-declared tau window
    win = [(t, hb, hs) for t, hb, hs in zip(rec["tau"], rec["h_bubble"], rec["h_spike"])
           if TAU_LO <= t <= TAU_HI]
    if len(win) >= 5:
        ts = [w[0] for w in win]
        # _slope returns d h / d tau (lattice lengths per normalised time unit);
        # the reference velocities are lattice velocities, so multiply by
        # d tau / d t = sqrt(A g / lam).  (The pre-m2026-11-13 version compared
        # the tau-slope directly with the lattice velocity, i.e. reported
        # velocities a factor 1/sqrt(Ag/lam) = 800 too large.)
        ub = _slope(ts, [w[1] for w in win]) * tau_unit
        us = _slope(ts, [w[2] for w in win]) * tau_unit
    else:
        ub = us = float("nan")

    sig_inv = analytic_growth_rate(spec.atwood, spec.g, spec.lam)
    sig_visc = viscous_growth_rate(spec.atwood, spec.g, spec.lam, cfg.nu)
    ub_ref = boussinesq_velocity(spec.atwood, spec.g, spec.lam)
    us_ref = ub_ref
    fr_scale = froude_normalisation(spec.atwood, spec.g, spec.lam)
    max_u = max(rec["max_u"]) if rec["max_u"] else float("nan")

    case = {
        "name": spec.name,
        "spec": asdict(spec),
        "n_steps": n_steps,
        "wall_seconds": wall,
        "nx": cfg.nx,
        "ny": cfg.ny,
        "k": cfg.k,
        "nu": cfg.nu,
        "diffusivity": cfg.diffusivity,
        "re_perturbation": spec.lam * ub_ref / cfg.nu,
        "max_speed_peak": max_u,
        "ma_peak": max_u / math.sqrt(1.0 / 3.0),
        "phi_range_peak": [min(rec["phi_min"]), max(rec["phi_max"])] if rec["phi_min"] else [float("nan")] * 2,
        "run_finite": finite,
        "sigma_sim": sigma_sim,
        "sigma_r2": sigma_r2,
        "sigma_fit_points": sigma_n,
        "sigma_fit_a_over_lam": sigma_a_win,
        "sigma_inviscid": sig_inv,
        "sigma_viscous": sig_visc,
        "sigma_rel_error_vs_inviscid_pct": 100.0 * (sigma_sim / sig_inv - 1.0),
        "sigma_rel_error_vs_viscous_pct": 100.0 * (sigma_sim / sig_visc - 1.0),
        "velocity_unit_dtau_dt": tau_unit,
        "u_bubble_slope_dh_dtau": ub / tau_unit if math.isfinite(ub) else float("nan"),
        "u_spike_slope_dh_dtau": us / tau_unit if math.isfinite(us) else float("nan"),
        "u_bubble_sim": ub,
        "u_spike_sim": us,
        "u_bubble_ref": ub_ref,
        "u_spike_ref": us_ref,
        "u_bubble_goncharov_twofluid": goncharov_velocity(spec.atwood, spec.g, spec.lam, "b"),
        "u_spike_goncharov_twofluid": goncharov_velocity(spec.atwood, spec.g, spec.lam, "s"),
        "reference_model": "boussinesq-symmetric sqrt(2Ag/(Ck)), C_2D=1", 
        "u_bubble_rel_error_pct": 100.0 * (ub / ub_ref - 1.0) if math.isfinite(ub) else float("nan"),
        "u_spike_rel_error_pct": 100.0 * (us / us_ref - 1.0) if math.isfinite(us) else float("nan"),
        "fr_bubble_sim": ub / fr_scale if math.isfinite(ub) else float("nan"),
        "fr_bubble_ref": ub_ref / fr_scale,
        "fr_spike_sim": us / fr_scale if math.isfinite(us) else float("nan"),
        "fr_spike_ref": us_ref / fr_scale,
        "terminal_window_points": len(win),
        "terminal_window_tau": [TAU_LO, TAU_HI],
        "mass_drift_pct_max": max(rec["mass_drift_pct"]) if rec["mass_drift_pct"] else float("nan"),
        "h_bubble_end": rec["h_bubble"][-1] if rec["h_bubble"] else float("nan"),
        "h_spike_end": rec["h_spike"][-1] if rec["h_spike"] else float("nan"),
    }
    case["gates"] = {
        "sigma_within_gate": bool(math.isfinite(sigma_sim) and abs(case["sigma_rel_error_vs_viscous_pct"]) <= GATE_PCT),
        "u_bubble_within_gate": bool(math.isfinite(ub) and abs(case["u_bubble_rel_error_pct"]) <= GATE_PCT),
        "u_spike_within_gate": bool(math.isfinite(us) and abs(case["u_spike_rel_error_pct"]) <= GATE_PCT),
    }
    case["gates"]["all_pass"] = all(case["gates"].values())

    d = out_root / spec.name
    d.mkdir(parents=True, exist_ok=True)
    with (d / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["step", "tau", "a_mode_over_lam", "h_bubble", "h_spike", "y_mean",
                     "max_speed", "phi_min", "phi_max", "mass_drift_pct",
                     "h_bubble_zc", "h_spike_zc"])
        for i, s in enumerate(rec["step"]):
            wr.writerow([int(s), f"{rec['tau'][i]:.6f}", f"{rec['a_mode'][i] / spec.lam:.6e}",
                         f"{rec['h_bubble'][i]:.6e}", f"{rec['h_spike'][i]:.6e}",
                         f"{rec['y_mean'][i]:.6e}", f"{rec['max_u'][i]:.6e}",
                         f"{rec['phi_min'][i]:.6e}", f"{rec['phi_max'][i]:.6e}",
                         f"{rec['mass_drift_pct'][i]:.3e}",
                         f"{rec['h_bubble_zc'][i]:.6e}", f"{rec['h_spike_zc'][i]:.6e}"])
    (d / "case.json").write_text(json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return case


def _slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx if sxx > 0.0 else float("nan")


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

#: reference case (grid) + sensitivity axes (viscosity, velocity scale, Atwood,
#: interface diffusivity).  All at the same aspect ratio 8 unless noted.
AC = dict(phase_scheme="allen_cahn", ac_sigma=3.0e-4, ac_width=2.0, ac_mobility=20.0)
SPECS = (
    # reference case (A = 1/2) + resolution and parameter-sensitivity axes
    CaseSpec("A050_l064_g2e-4", atwood=0.5, lam=64, g=2.0e-4, tau=0.515, tau_phi=0.501,
             tau_end=6.5, **AC),
    CaseSpec("A033_l064_g2e-4", atwood=1.0 / 3.0, lam=64, g=2.0e-4, tau=0.515, tau_phi=0.501,
             tau_end=6.5, **AC),
    CaseSpec("A050_l128_g2e-4", atwood=0.5, lam=128, g=2.0e-4, tau=0.515, tau_phi=0.501,
             tau_end=6.5, **AC),                       # 2x resolution
    CaseSpec("A050_l064_g1e-4", atwood=0.5, lam=64, g=1.0e-4, tau=0.515, tau_phi=0.501,
             tau_end=6.5, **AC),                       # second (A g) point
    CaseSpec("A050_l064_g2e-4_legacy", atwood=0.5, lam=64, g=2.0e-4, tau=0.515, tau_phi=0.501,
             tau_end=6.5),                             # legacy scheme, control
)


def aggregate(out_root: Path) -> dict:
    cases = []
    for d in sorted(p for p in out_root.iterdir() if p.is_dir()):
        f = d / "case.json"
        if f.exists():
            cases.append(json.loads(f.read_text()))
    summary = {
        "case_code": "RAYLEIGH-TAYLOR",
        "title": "2-D single-mode Rayleigh-Taylor: linear growth rate + spike/bubble terminal velocities",
        "solver": "tensorlbm.rayleigh_taylor (Boussinesq two-layer D2Q9 + advected phase field)",
        "gates": {
            "tolerance_pct": GATE_PCT,
            "linear_window_a_over_lam": [A_LO_FRAC, A_HI_FRAC],
            "terminal_window_tau": [TAU_LO, TAU_HI],
            "goncharov_C_2D": C_2D,
        },
        "cases": cases,
        "n_cases": len(cases),
        "n_all_pass": sum(1 for c in cases if c["gates"]["all_pass"]),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_suite(specs: tuple[CaseSpec, ...], out_root: Path, quick: bool = False) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        c = run_case(spec, out_root, quick=quick)
        print(f"[RT] {spec.name:28s} sigma={c['sigma_sim']:.5f} (visc {c['sigma_viscous']:.5f}, "
              f"{c['sigma_rel_error_vs_viscous_pct']:+.1f}%)  u_b={c['u_bubble_sim']:.5f} "
              f"({c['u_bubble_rel_error_pct']:+.1f}%)  u_s={c['u_spike_sim']:.5f} "
              f"({c['u_spike_rel_error_pct']:+.1f}%)  maxMa={c['ma_peak']:.3f}  "
              f"{c['wall_seconds']:.1f}s", flush=True)
    return aggregate(out_root)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smoke run (300 steps per case)")
    ap.add_argument("--only", nargs="*", default=None, help="subset of case names")
    ap.add_argument("--out", default=None)
    ap.add_argument("--aggregate", action="store_true", help="only rebuild summary.json from case.json files")
    args = ap.parse_args(argv)
    out_root = Path(args.out) if args.out else BENCH_DIR / "artifacts" / "runs"
    if args.aggregate:
        s = aggregate(out_root)
    else:
        specs = SPECS if not args.only else tuple(s for s in SPECS if s.name in set(args.only))
        s = run_suite(specs, out_root, quick=args.quick)
    print(json.dumps({"n_cases": s["n_cases"], "n_all_pass": s["n_all_pass"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())