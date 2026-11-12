"""Post-process CASE-RAYLEIGH-TAYLOR artifacts: salvage the gates from runs that
blow up late in time.

Why this exists
---------------
``run_case.py`` marks a case "not finite" if *any* sample of the run is NaN, and
then reports ``sigma = NaN``.  That is too coarse: several configurations deliver
a perfectly good linear-window growth-rate fit and only become unstable in the
nonlinear stage.  This script recomputes, from the already written
``history.csv`` (no re-simulation), (a) the growth rate over the samples that are
finite and inside the pre-declared linear window, and (b) the normalised time at
which the first non-finite sample appears (``blowup_tau``), so the report can
say *where* each configuration stops being usable instead of just "nan".

Nothing here changes a threshold: the window and the 10 % gate are imported from
``run_case.py``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import run_case as rc

RUNS = Path(__file__).resolve().parent / "artifacts" / "runs"


def recompute(case_dir: Path) -> dict:
    case = json.loads((case_dir / "case.json").read_text())
    lam = case["spec"]["lam"]
    atwood = case["spec"]["atwood"]
    g = case["spec"]["g"]
    tau = case["spec"]["tau"]
    rows = list(csv.DictReader((case_dir / "history.csv").open()))

    # first non-finite sample
    blowup_tau = float("nan")
    good: list[dict] = []
    for r in rows:
        vals = [float(v) for v in r.values()]
        if not all(math.isfinite(v) for v in vals):
            blowup_tau = float(rows[max(0, rows.index(r) - 1)]["tau"])
            break
        good.append(r)
    else:
        good = rows

    steps = [float(r["step"]) for r in good]
    amps = [float(r["a_mode_over_lam"]) * lam for r in good]
    a_lo, a_hi = rc.A_LO_FRAC * lam, rc.A_HI_FRAC * lam
    try:
        fit = rc.fit_growth_rate(steps, amps, a_lo, a_hi)
        sigma, r2, npts = fit.sigma, fit.r2, fit.n_points
    except Exception:                      # not enough samples in the window
        sigma, r2, npts = float("nan"), float("nan"), 0

    nu = (tau - 0.5) / 3.0
    sig_inv = rc.analytic_growth_rate(atwood, g, lam)
    sig_visc = rc.viscous_growth_rate(atwood, g, lam, nu)

    # terminal velocities: only usable if the whole pre-declared window survived
    win = [(float(r["tau"]), float(r["h_bubble"]), float(r["h_spike"])) for r in good
           if rc.TAU_LO <= float(r["tau"]) <= rc.TAU_HI]
    if len(win) >= 5:
        ts = [w[0] for w in win]
        ub = rc._slope(ts, [w[1] for w in win])
        us = rc._slope(ts, [w[2] for w in win])
    else:
        ub = us = float("nan")

    case.update({
        "run_finite": bool(math.isnan(blowup_tau)),
        "blowup_tau": blowup_tau,
        "sigma_sim": sigma,
        "sigma_r2": r2,
        "sigma_fit_points": npts,
        "sigma_rel_error_vs_inviscid_pct": 100.0 * (sigma / sig_inv - 1.0) if math.isfinite(sigma) else float("nan"),
        "sigma_rel_error_vs_viscous_pct": 100.0 * (sigma / sig_visc - 1.0) if math.isfinite(sigma) else float("nan"),
        "u_bubble_sim": ub,
        "u_spike_sim": us,
        "u_bubble_rel_error_pct": 100.0 * (ub / case["u_bubble_ref"] - 1.0) if math.isfinite(ub) else float("nan"),
        "u_spike_rel_error_pct": 100.0 * (us / case["u_spike_ref"] - 1.0) if math.isfinite(us) else float("nan"),
        "terminal_window_points": len(win),
    })
    case["gates"] = {
        "sigma_within_gate": bool(math.isfinite(sigma)
                                  and abs(case["sigma_rel_error_vs_viscous_pct"]) <= rc.GATE_PCT),
        "u_bubble_within_gate": bool(math.isfinite(ub)
                                     and abs(case["u_bubble_rel_error_pct"]) <= rc.GATE_PCT),
        "u_spike_within_gate": bool(math.isfinite(us)
                                    and abs(case["u_spike_rel_error_pct"]) <= rc.GATE_PCT),
    }
    case["gates"]["all_pass"] = all(case["gates"].values())
    (case_dir / "case.json").write_text(json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return case


def main() -> int:
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        if not (d / "case.json").exists():
            continue
        c = recompute(d)
        print(f"{c['name']:30s} sigma={c['sigma_sim']:.5f} ({c['sigma_rel_error_vs_viscous_pct']:+.1f}%) "
              f"blowup_tau={c['blowup_tau']:.2f} ub={c['u_bubble_sim']:.2f} gates={c['gates']}")
    s = rc.aggregate(RUNS)
    print(json.dumps({"n_cases": s["n_cases"], "n_all_pass": s["n_all_pass"],
                      "n_sigma_pass": sum(1 for c in s["cases"] if c["gates"]["sigma_within_gate"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())