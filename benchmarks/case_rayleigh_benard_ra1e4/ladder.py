#!/usr/bin/env python
"""CASE-RAYLEIGH-BENARD-RA1E4 网格/参数阶梯驱动（集群分片入口）。

用法：python benchmarks/case_rayleigh_benard_ra1e4/ladder.py <shard_i> <n_shards>
每个分片跑下表中若干档，输出 runs/case_rayleigh_benard_ra1e4/<tag>.json。

阶梯设计（预注册，见 docs/CASE_RAYLEIGH_BENARD_RA1E4.md §⑤）：
- 网格收敛：nx = 32 / 48 / 64 @ tau=0.8（固定 Ma 口径）
- 参数敏感性：nx=48 上 tau ∈ {0.65, 0.8, 0.9}（Ma 与离散误差对照）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "case_rayleigh_benard_ra1e4"))

from tensorlbm.thermal import simulate_natural_convection  # noqa: E402

REF = {"nu": 2.243, "u_max": 16.178, "v_max": 19.617}
OUTDIR = REPO / "runs" / "case_rayleigh_benard_ra1e4"

# (tag, nx, tau, max_steps, report_every)
CASES = [
    ("nx32_tau0.80", 32, 0.80, 80000, 2000),
    ("nx48_tau0.80", 48, 0.80, 160000, 2000),
    ("nx64_tau0.80", 64, 0.80, 260000, 2000),
    ("nx48_tau0.65", 48, 0.65, 300000, 2000),
    ("nx48_tau0.90", 48, 0.90, 100000, 2000),
]


def run_one(tag: str, nx: int, tau: float, max_steps: int, every: int) -> dict:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    res = simulate_natural_convection(
        nx=nx, ra=1e4, pr=0.71, tau=tau, steps=max_steps, device="cpu",
        seed_t="linear", report_every=every, stop_tol_nu=1e-5, stop_patience=4,
    )
    H, alpha = res["H"], res["alpha"]
    hist = res["nu_history"]
    nu = [h["nu"] for h in hist]
    payload = {
        "case": "CASE-RAYLEIGH-BENARD-RA1E4",
        "tag": tag,
        "config": {"nx": nx, "ny": nx, "ra": 1e4, "pr": 0.71, "tau": tau,
                   "max_steps": max_steps, "report_every": every,
                   "tau_T": res["tau_T"], "dtype": "float64", "device": "cpu",
                   "torch": torch.__version__, "prereg_stop_tol_nu": 1e-5,
                   "stop_patience": 4},
        "lattice": {"nu": res["nu"], "alpha": alpha, "H": H,
                    "g_beta": res["g_beta"], "u_max_lat": res["u_max"],
                    "ma_at_u_max_ref": 16.178 * alpha / H / 0.5773502691896258},
        "ref": REF,
        "result": {
            "steps_done": res["steps_done"], "stopped_early": res["stopped_early"],
            "elapsed_s": res["elapsed_s"],
            "nu_grad2": res["nu_grad2"], "nu_grad1": res["nu_grad1"],
            "nu_halfway": res["nu_halfway"],
            "nu_left_grad2": res["nu_left_grad2"],
            "nu_right_grad2": res["nu_right_grad2"],
            "nu_err_pct": 100.0 * (res["nu_grad2"] / REF["nu"] - 1.0),
            "u_max_nd": res["u_max_x"] * H / alpha,
            "v_max_nd": res["v_max_y"] * H / alpha,
            "u_max_err_pct": 100.0 * (res["u_max_x"] * H / alpha / REF["u_max"] - 1.0),
            "v_max_err_pct": 100.0 * (res["v_max_y"] * H / alpha / REF["v_max"] - 1.0),
            "y_at_u_max": res["y_at_u_max_x"], "x_at_v_max": res["x_at_v_max_y"],
            "resid_u": res["last_resid_u"], "resid_T": res["last_resid_T"],
            "T_min": res["T_min"], "T_max": res["T_max"],
        },
        "nu_history": hist,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{tag}.json"
    out.write_text(json.dumps(payload, indent=1))
    print(tag, "->", out, json.dumps(payload["result"]), flush=True)
    return payload


def main() -> int:
    i = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    for k, case in enumerate(CASES):
        if k % n != (i - 1):
            continue
        run_one(*case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())