#!/usr/bin/env python
"""CASE-RAYLEIGH-BENARD-RA1E4 —— 自然对流基准算例（de Vahl Davis 1983 锚）。

复用库内既有能力 ``tensorlbm.thermal.simulate_natural_convection``
（D2Q9 + D2Q5 双分布 BGK，Boussinesq 浮力 + Guo 力，half-way bounce-back
无滑移壁 + 上下绝热、左右等温），按网格阶梯 {32,48,64,96} 跑到稳态，
对照 de Vahl Davis (1983) 的 Ra=1e4, Pr=0.71 基准值：
    Nu(ref)     = 2.243        （热壁面平均 Nusselt 数）
    u_max(ref)  = 16.178 α/L   （水平速度极大值）
    v_max(ref)  = 19.617 α/L   （竖直速度极大值）

用法（单档）：
    python benchmarks/case_rayleigh_benard_ra1e4/run.py --nx 64 --steps 200000 \
        --out runs/case_rayleigh_benard_ra1e4/nx64.json

输出为机器可读 JSON（含逐采样 Nu 历史 → 收敛曲线/残差图数据）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.thermal import simulate_natural_convection  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ra", type=float, default=1.0e4)
    ap.add_argument("--pr", type=float, default=0.71)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--report-every", type=int, default=2000)
    ap.add_argument("--tol-nu", type=float, default=1e-6,
                    help="Nu 平台判据：连续 3 个采样 |ΔNu| < tol-nu 即认为稳态")
    ap.add_argument("--tol-resid", type=float, default=1e-9)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed-t", type=str, default="linear")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    t0 = time.time()
    res = simulate_natural_convection(
        nx=args.nx, ra=args.ra, pr=args.pr, tau=args.tau,
        steps=args.steps, device="cpu", seed_t=args.seed_t,
        report_every=args.report_every,
    )
    wall = time.time() - t0

    H = res["H"]
    alpha = res["alpha"]
    hist = res["nu_history"]
    # Nu 平台检测：末尾 3 个采样窗口
    nu_vals = [h["nu"] for h in hist]
    nu_drift = abs(nu_vals[-1] - nu_vals[-2]) if len(nu_vals) >= 2 else float("nan")
    steady = bool(len(nu_vals) >= 3 and nu_drift < args.tol_nu)

    payload = {
        "case": "CASE-RAYLEIGH-BENARD-RA1E4",
        "reference": "de Vahl Davis 1983 (Int. J. Numer. Methods Fluids 3:249-264), "
                     "Ra=1e4, Pr=0.71, 差分加热方腔",
        "config": {
            "nx": args.nx, "ny": args.nx, "ra": args.ra, "pr": args.pr,
            "tau": args.tau, "tau_T": res["tau_T"], "steps": args.steps,
            "report_every": args.report_every, "seed_t": args.seed_t,
            "tol_nu": args.tol_nu, "threads": args.threads,
            "dtype": "float64", "device": "cpu",
            "py": sys.version.split()[0], "torch": torch.__version__,
        },
        "lattice": {
            "nu": res["nu"], "alpha": alpha, "H": H, "g_beta": res["g_beta"],
            "u_max_ref_nd": 16.178, "v_max_ref_nd": 19.617, "nu_ref": 2.243,
        },
        "result": {
            "elapsed_s": round(wall, 1),
            "nu_grad2": res["nu_grad2"],
            "nu_grad1": res["nu_grad1"],
            "nu_halfway": res["nu_halfway"],
            "nu_left_grad2": res["nu_left_grad2"],
            "nu_right_grad2": res["nu_right_grad2"],
            "u_max_lat": res["u_max"],
            "u_max_nd": res["u_max_x"] * H / alpha,
            "u_max_nd_neg": res["u_max_x_neg"] * H / alpha,
            "v_max_nd": res["v_max_y"] * H / alpha,
            "v_max_nd_neg": res["v_max_y_neg"] * H / alpha,
            "u_max_ref_nd": 16.178, "v_max_ref_nd": 19.617, "nu_ref": 2.243,
            "y_at_u_max": res["y_at_u_max_x"],
            "x_at_v_max": res["x_at_v_max_y"],
            "resid_u": res["last_resid_u"], "resid_T": res["last_resid_T"],
            "steady": steady, "nu_drift_last": nu_drift,
            "T_min": res["T_min"], "T_max": res["T_max"],
        },
        "nu_history": hist,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps({
        "nx": args.nx, "elapsed_s": payload["result"]["elapsed_s"],
        "nu_grad2": res["nu_grad2"], "nu_halfway": res["nu_halfway"],
        "u_max_nd": payload["result"]["u_max_nd"],
        "v_max_nd": payload["result"]["v_max_nd"],
        "resid_u": res["last_resid_u"], "steady": steady,
        "out": str(out),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())