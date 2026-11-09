"""Assemble per-case records into the benchmark artifact (and plan the hold-out).

    python tests/benchmarks/wetting_contact_angle/assemble.py --plan     # write b2_jobs.txt
    bash   tests/benchmarks/wetting_contact_angle/run_all.sh b2         # run the hold-out
    python tests/benchmarks/wetting_contact_angle/assemble.py            # write the artifact

The calibration curve is built from the coarse scan only; the hold-out
prescription for the fine grid is derived from that curve by
:func:`wetting_bench.invert_calibration`, so the verification cannot reuse
fine-grid information.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wetting_bench as wb  # noqa: E402


def load_cases(cases_dir: Path, prefix: str) -> list[dict]:
    out = []
    for p in sorted(cases_dir.glob(f"{prefix}*.json")):
        out.append(json.loads(p.read_text()))
    return out


def build(cases_dir: Path) -> dict:
    young = load_cases(cases_dir, "a_")
    scan = load_cases(cases_dir, "b1_")
    verify = load_cases(cases_dir, "b2_")
    metastable = load_cases(cases_dir, "c_")
    curve = wb.calibration_curve(scan)

    def verdict(cases):
        ok = [c for c in cases if c["status"] == "PASS"]
        return {
            "n_cases": len(cases),
            "n_pass": len(ok),
            "n_fail": len(cases) - len(ok),
            "max_abs_delta": max((c.get("abs_delta") or 0.0) for c in cases) if cases else None,
            "status": "PASS" if cases and len(ok) == len(cases) else "FAIL",
        }

    payload = {
        "meta": wb.sweep_meta(Path(__file__).resolve().parents[3], threads=None, workers=None),
        "estimator_control": estimator_control(),
        "stage_a_young": {"cases": young},
        "stage_b_calibration": {"cases": scan, "curve": curve},
        "stage_b_verify": {"cases": verify, "skipped": []},
        "stage_c_metastability": {
            "cases": metastable,
            "split_deg": (max(c["theta_measured"] for c in metastable)
                          - min(c["theta_measured"] for c in metastable))
            if len(metastable) > 1 else None,
        },
        "grid_sensitivity": grid_sensitivity(young),
        "verdicts": {"young": verdict(young), "calibrated": verdict(verify)},
    }
    return payload


def grid_sensitivity(cases: list[dict]) -> list[dict]:
    """Same wall fraction, two resolutions -> the measured angle's grid dependence."""
    by_fw: dict[float, dict[str, dict]] = {}
    for c in cases:
        fw = c.get("wall_fraction")
        if fw is None or c.get("theta_measured") is None:
            continue
        by_fw.setdefault(round(float(fw), 6), {})[c["grid"]] = c
    out = []
    for fw, per_grid in sorted(by_fw.items()):
        if len(per_grid) < 2:
            continue
        grids = sorted(per_grid)
        lo, hi = per_grid[grids[0]], per_grid[grids[1]]
        out.append({"wall_fraction": fw, "theta_eq": lo["theta_eq"],
                    "grid_a": grids[0], "theta_a": lo["theta_measured"],
                    "grid_b": grids[1], "theta_b": hi["theta_measured"],
                    "diff_deg": abs(lo["theta_measured"] - hi["theta_measured"])})
    return out


def estimator_control() -> list[dict]:
    """Analytic-cap control: the estimator's bias at both benchmark radii.

    If the measured angle shifts between the coarse and fine grid while the
    estimator recovers analytic caps of the same angles at *both* radii to <1.5
    deg, the shift is a property of the simulation, not of the measurement.
    """
    import contact_angle as ca

    out = []
    for grid in wb.GRIDS:
        for theta in wb.PRESCRIBED_DEG:
            rho, solid, y_wall = ca.synthetic_cap_density(
                theta_deg=theta, radius=grid.radius, nx=grid.nx, ny=grid.ny,
                rho_l=wb.RHO_L_DEFAULT, rho_v=wb.RHO_V_DEFAULT)
            m = ca.measure_contact_angle(rho, rho_l=wb.RHO_L_DEFAULT, rho_v=wb.RHO_V_DEFAULT,
                                        y_wall=y_wall, solid=solid, radius=grid.radius)
            out.append({"grid": grid.name, "radius": grid.radius, "theta_true": theta,
                        "theta_cap": m["theta_cap"], "theta_circle": m["theta_circle"],
                        "theta_band": m["theta_band"],
                        "bias_cap": m["theta_cap"] - theta})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default=str(HERE / "results" / "cases"))
    ap.add_argument("--out", default=str(HERE / "results" / "contact_angle_sweep.json"))
    ap.add_argument("--plan", action="store_true", help="only write results/cases/b2_jobs.txt")
    args = ap.parse_args(argv)
    cases_dir = Path(args.cases)
    if args.plan:
        curve = wb.calibration_curve(load_cases(cases_dir, "b1_"))
        lines = []
        for th in wb.PRESCRIBED_DEG:
            try:
                fw = wb.invert_calibration(curve, th)
            except ValueError as exc:
                print(f"skip {th}: {exc}", file=sys.stderr)
                continue
            lines.append(f"--theta {th} --grid fine --law calibrated --fw {fw:.6f} "
                         f"--out {cases_dir}/b2_fine_{int(th)}.json")
        plans = cases_dir / "b2_jobs.txt"
        plans.write_text("\n".join(lines) + "\n")
        print(plans.read_text())
        return 0
    payload = build(cases_dir)
    out = wb.write_json(payload, args.out)
    print(f"artifact: {out}")
    for name, v in payload["verdicts"].items():
        print(f"  {name:>10s}: {v['status']}  {v['n_pass']}/{v['n_cases']} "
              f"(max |delta| = {v['max_abs_delta']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())