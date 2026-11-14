#!/usr/bin/env python3
"""Round-2 extended-ladder fragment: Taylor-Green arms at the predeclared ladder resolution.

The full ladder in ``run_xval.py`` (TG + the 30000-step channel arm at Ny in {16,24}) does not fit the
120 s per-call cap of the control container, and the compute-cluster job submitted for it (run_compute
cj2867) was still queued when the mission wall budget ran out. This script therefore executes the
Taylor-Green half of the ladder only -- the part that carries the resolution/Reynolds sensitivity --
and writes a fragment that is merged into the round-2 report next to the ``--quick`` result.json.

It re-uses the driver's own case runner and estimators (``run_xval.run_tg``), so nothing is
re-implemented: only the loop over configurations is here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_xval as R  # noqa: E402

CASES = [
    {"label": "n64_Re100", "n": 64, "re": 100.0, "steps": 2400},
    {"label": "n32_Re100", "n": 32, "re": 100.0, "steps": 2400},
    {"label": "n64_Re50", "n": 64, "re": 50.0, "steps": 1200},
]


def main() -> int:
    out = HERE.parent.parent / "runs" / "xval_conv_as_streaming_r2"
    out.mkdir(parents=True, exist_ok=True)
    oss = R.load_oss("D2Q9")
    t0 = time.time()
    cases = []
    for cfg in CASES:
        spec = R.TG2DSpec(
            n=cfg["n"], re=cfg["re"], steps=cfg["steps"], sample_every=max(10, cfg["steps"] // 40)
        )
        for arm in ("A_native_gather", "B_conv_fixed_kernel"):
            c = R.run_tg(spec, arm, oss, threads=4)
            c["label"] = cfg["label"]
            c["gamma_K_rel_err_pct"] = (
                100.0 * abs(c["gamma_K_fit"] - c["gamma_K_theory"]) / c["gamma_K_theory"]
            )
            c["gamma_vel_rel_err_pct"] = (
                100.0 * abs(c["gamma_vel_fit"] - c["gamma_vel_theory"]) / c["gamma_vel_theory"]
            )
            c.pop("_ux", None)
            c.pop("_uy", None)
            cases.append(c)
    pairs = []
    for cfg in CASES:
        a = next(c for c in cases if c["label"] == cfg["label"] and c["arm"] == "A_native_gather")
        b = next(c for c in cases if c["label"] == cfg["label"] and c["arm"].startswith("B"))
        pairs.append(
            {
                "config": cfg["label"],
                "identical_populations": a["f_sha256"] == b["f_sha256"],
                "gamma_K_abs_diff": abs(a["gamma_K_fit"] - b["gamma_K_fit"]),
                "umax_abs_diff": abs(a["umax_final"] - b["umax_final"]),
            }
        )
    frag = {
        "anchor": R.ANCHOR,
        "mission": 20684,
        "fragment": "tg_extended_ladder",
        "cases": CASES,
        "wall_s": round(time.time() - t0, 1),
        "tg_ladder": cases,
        "tg_ladder_pair": pairs,
    }
    (out / "tg_ladder_fragment.json").write_text(json.dumps(frag, indent=1) + "\n")
    worst = max(max(c["gamma_K_rel_err_pct"], c["gamma_vel_rel_err_pct"], c["field_l2_pct"]) for c in cases)
    print(json.dumps({"wall_s": frag["wall_s"], "worst_kpi_pct": round(worst, 4),
                      "identical": all(p["identical_populations"] for p in pairs)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())