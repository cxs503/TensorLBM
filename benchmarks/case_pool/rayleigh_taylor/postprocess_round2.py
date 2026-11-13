"""Round-2 (m18081) acceptance table for CASE-RAYLEIGH-TAYLOR.

Recomputes, from the written ``history.csv`` of every run (no re-simulation),
the terminal bubble/spike velocities over the pre-declared window
``tau in [TAU_LO, TAU_HI]`` and compares them with the Boussinesq-consistent
potential-flow reference ``sqrt(2 A g / (C k))`` with ``C_2D = 1``.

Two estimator columns are handled: ``h_bubble`` (current runs: conserved
heavy-column moment) and ``h_bubble_col`` (runs written before the round-2
swap, where the column moment lived under that name).  The ``tau``-slope is
converted to a lattice velocity by ``sqrt(A g / lam)`` (the round-2 unit fix).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import run_case as rc

ROOT = Path(__file__).resolve().parent


def _slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def evaluate(case_dir: Path) -> dict:
    case = json.loads((case_dir / "case.json").read_text())
    spec = case["spec"]
    rows = list(csv.DictReader((case_dir / "history.csv").open()))
    hb_key = "h_bubble" if "h_bubble" in rows[0] else "h_bubble_col"
    hs_key = "h_spike" if "h_spike" in rows[0] else "h_spike_col"
    win = [r for r in rows if rc.TAU_LO <= float(r["tau"]) <= rc.TAU_HI]
    out = {"case": case["name"], "phase_scheme": spec.get("phase_scheme", "bgk"),
           "lam": spec["lam"], "atwood": spec["atwood"], "g": spec["g"],
           "n_window": len(win)}
    if len(win) < 5:
        out.update(u_b=float("nan"), u_s=float("nan"))
        return out
    ts = [float(r["tau"]) for r in win]
    unit = math.sqrt(spec["atwood"] * spec["g"] / spec["lam"])
    ub = _slope(ts, [float(r[hb_key]) for r in win]) * unit
    us = _slope(ts, [float(r[hs_key]) for r in win]) * unit
    ref = rc.boussinesq_velocity(spec["atwood"], spec["g"], spec["lam"])
    out.update(u_b=ub, u_s=us, u_ref=ref,
               err_b_pct=100.0 * (ub / ref - 1.0), err_s_pct=100.0 * (us / ref - 1.0),
               phi_max_peak=case.get("phi_range_peak", [None, None])[1],
               ma_peak=case.get("ma_peak"), pass_b=abs(ub / ref - 1.0) <= 0.10,
               pass_s=abs(us / ref - 1.0) <= 0.10)
    return out


def main() -> int:
    rows = []
    for root in sorted((ROOT.parents[3] / "runs" / "m18081").glob("suite*")):
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if (d / "case.json").exists():
                rows.append(evaluate(d))
    out_dir = ROOT / "artifacts" / "round2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "acceptance.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    for r in rows:
        print(f"{r['case']:24s} {r['phase_scheme']:10s} lam={r['lam']:3d} "
              f"u_b={r.get('u_b', float('nan')):.5f} u_s={r.get('u_s', float('nan')):.5f} "
              f"ref={r.get('u_ref', float('nan')):.5f} "
              f"err=({r.get('err_b_pct', float('nan')):+.1f}%,{r.get('err_s_pct', float('nan')):+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
