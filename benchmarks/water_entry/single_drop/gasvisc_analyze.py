"""DE-X-GAS1-GASVISC: score ctrl vs arm and emit the triple comparison.

Usage:
    python3 gasvisc_analyze.py <ctrl.json> <arm.json> [--out triple.json]

Uses the repo's own score.score_run so the triple is computed with the exact
same protocol/gates as the prior cards (CONV12/CAVSWEEP/ABBS).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks.water_entry.single_drop.score import reference, score_run


def triple(run_path, case):
    run = json.loads(Path(run_path).read_text())
    t_ref, h_ref = reference(case)
    r = score_run(run, t_ref, h_ref, case)
    return {
        "stop_reason": r["stop_reason"],
        "contact_step": r["contact_step"],
        "peak_depth_D": r["full_window_peak_depth_over_D"],
        "peak_time_ms": r["full_window_peak_time_ms"],
        "peak_error_pct": (
            100.0 * r["peak_relative_error"] if r["peak_relative_error"] is not None else None
        ),
        "postcontact_healthy_ms": r["postcontact_healthy_ms"],
        "max_mach": None,
        "max_geometric_volume_drift": r["max_geometric_volume_relative_drift"],
        "numeric_pass": r["numeric_pass"],
        "gas_nu_ratio": run.get("gas_nu_ratio"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ctrl")
    ap.add_argument("arm")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    case = json.loads((Path(__file__).parent / "protocol.json").read_text())
    c = triple(args.ctrl, case)
    a = triple(args.arm, case)
    out = {
        "ctrl": c,
        "arm": a,
        "delta_peak_depth_D": (
            a["peak_depth_D"] - c["peak_depth_D"]
            if (a["peak_depth_D"] is not None and c["peak_depth_D"] is not None)
            else None
        ),
        "delta_healthy_ms": (
            a["postcontact_healthy_ms"] - c["postcontact_healthy_ms"]
            if (a["postcontact_healthy_ms"] is not None and c["postcontact_healthy_ms"] is not None)
            else None
        ),
    }
    text = json.dumps(out, indent=2, allow_nan=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
