#!/usr/bin/env python3
"""Derive every number the m17849 audit quotes, directly from the audited artifacts.

Inputs (all inside ``refs/``, i.e. byte copies of the m17713 branch at 24b4af6a):
  refs/xval/benchmark_method_selection/runs/<arm>_D10/result.json
  refs/xval/benchmark_method_selection/{summary.json,oss_crosscheck.json,table.csv}
  refs/xval/benchmark_method_selection/logs/*.log        (D=14 ladder evidence)
Output: audit_numbers.json (single source of truth for AUDIT.md and the report).

No number is typed by hand: rerun this script to re-derive the audit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE / "refs" / "xval" / "benchmark_method_selection"
TOL = 15.0
KPIS = ("cd_mean", "cl_amp", "st_fft")


def pct(new: float, base: float) -> float:
    return 100.0 * (new - base) / abs(base)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    res = {}
    for arm in ("A1", "A2", "A4"):
        p = REF / "runs" / f"{arm}_D10" / "result.json"
        d = json.loads(p.read_text())
        res[arm] = {k: d[k] for k in KPIS} | {
            "wall_s": d.get("wall_s"), "ms_per_step": d.get("ms_per_step"),
            "total_steps": d.get("total_steps"),
            "src": str(p.relative_to(HERE)), "sha256": sha256(p)}
    fails = {}
    for arm in ("A3", "A5"):
        p = REF / "runs" / f"{arm}_D10" / "fail.json"
        d = json.loads(p.read_text())
        fails[arm] = {k: d.get(k) for k in
                      ("status", "fail_stage", "step", "total_steps_planned",
                       "lambda_trt", "tau_minus", "cl_max_abs")} | {"src": str(p.relative_to(HERE))}

    dev = {arm: {k: pct(res[arm][k], res["A1"][k]) for k in KPIS} for arm in ("A2", "A4")}
    dev_av = {k: pct(res["A2"][k], res["A4"][k]) for k in KPIS}   # collision knob only
    dev_sa = dev["A4"]                                            # sponge knob only

    summary = json.loads((REF / "summary.json").read_text())
    cross = json.loads((REF / "oss_crosscheck.json").read_text())
    band = summary["thresholds"]["external_band"]
    cd_band_top = band["cd_mean"][1]
    st_band = band["st_fft"]

    logs = {p.name: p.stat().st_size for p in sorted((REF / "logs").glob("*.log"))}

    out = {
        "anchor": "XVAL-LBM-BENCHMARK-METHOD-SELECTION",
        "audited_mission": 17713,
        "audited_commit": "24b4af6a",
        "tolerance_pct": TOL,
        "arms_d10": res,
        "fails_d10": fails,
        "deviation_vs_A1_pct": dev,
        "sponge_knob_only_pct": dev_sa,
        "collision_knob_only_A2_vs_A4_pct": dev_av,
        "arms_exceeding_15pct": {a: sorted(k for k, v in dev[a].items() if abs(v) >= TOL)
                                 for a in dev},
        "deliverable_verdict_block": summary["verdict"],
        "convergence_block_empty": summary["convergence"] == {},
        "d14_log_bytes": logs,
        "external_band": band,
        "cd_mean_vs_band_top_pct": pct(res["A1"]["cd_mean"], cd_band_top),
        "cd_mean_vs_band_centre_pct": pct(res["A1"]["cd_mean"], sum(band["cd_mean"]) / 2),
        "st_in_band": st_band[0] <= res["A1"]["st_fft"] <= st_band[1],
        "oss_crosscheck": {
            "repo": cross["oss_repo"], "commit": cross["oss_commit"],
            "k1_rel_l2_collision": cross["k1_collision"]["rel_l2_collision"],
            "k2_rel_l2_streaming": cross["k2_streaming"]["rel_l2_streaming_interior"],
            "k0_all_directions_agree": cross["k0_direction_map"]["all_agree"],
            "source_sha256": cross["oss_source_sha256_unmodified"],
            "env_shim": cross["env_shim"],
            "throughput_ratio": cross["throughput"]["s_per_step_oss_100x100"] /
                                cross["throughput"]["s_per_step_tensorlbm_measured"],
        },
        "n_kpis": len(KPIS),
    }
    (HERE / "audit_numbers.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in
                      ("deviation_vs_A1_pct", "sponge_knob_only_pct",
                       "collision_knob_only_A2_vs_A4_pct", "arms_exceeding_15pct",
                       "convergence_block_empty", "d14_log_bytes",
                       "cd_mean_vs_band_top_pct", "cd_mean_vs_band_centre_pct",
                       "st_in_band")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()