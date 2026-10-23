#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""自然对流方腔（左热右冷、上下绝热）— de Vahl Davis 1983 验证。

模型：双分布函数热 LBM（src/tensorlbm/thermal.py）
- 速度场 D2Q9 BGK + Guo 力格式注入 Boussinesq 浮力 F_y = ρ·g·β·(T − T_ref)
- 温度场 D2Q5 BGK advection-diffusion，α = (τ_T − 1/2)/3
- 边界：四壁 no-slip（pre-streaming 半程反弹）；左壁 T=T_hot、右壁 T=T_cold
  （anti-bounce-back，half-way 二阶 Dirichlet）；上下绝热（bounce-back 零通量）

无量纲：Ra = g·β·ΔT·L³/(ν·α)，Pr = ν/α；方腔物理长度 L = nx（格子单位，
与 verified/cavity_re100 的 H=nx 约定一致）。取 ΔT = 1（T_hot=1, T_cold=0），
τ = 0.6（ν = 1/30），Pr = 0.71 → α = ν/Pr，τ_T = 3α + 1/2，g·β = Ra·ν·α/L³。

参考：de Vahl Davis (1983) Int. J. Numer. Methods Fluids 3:249-264，
Ra=1e4, Pr=0.71 平均 Nusselt 数 Nu = 2.243。

Nu 口径：壁面热通量积分 Nu = −∂T/∂x·L/ΔT 沿整壁平均——
- grad2：节点二阶单侧差分（壁面近似在节点上）
- grad1：节点一阶单侧差分
- halfway：一阶差分取在 half-way 壁面位置（与 ABB 几何一致）
判定：真实模拟（无外推），Nu 相对 2.243 误差 ≤3%，且 ≥2 档网格收敛。
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

from tensorlbm.thermal import simulate_natural_convection  # noqa: E402

NU_DE_VAHL_DAVIS_1E4 = 2.243  # de Vahl Davis 1983, Ra=1e4, Pr=0.71


def run_case(nx, ra, pr, tau, steps, device):
    res = simulate_natural_convection(
        nx=nx,
        ra=ra,
        pr=pr,
        tau=tau,
        steps=steps,
        device=device,
        report_every=10000,
    )
    nu = res["nu_grad2"]
    err_pct = 100.0 * abs(nu - NU_DE_VAHL_DAVIS_1E4) / NU_DE_VAHL_DAVIS_1E4
    print(
        f"[cavity_nc nx={nx}] steps={steps} t={res['elapsed_s']}s "
        f"resid_u={res['last_resid_u']:.2e} resid_T={res['last_resid_T']:.2e} "
        f"Nu_grad2={nu:.4f} Nu_grad1={res['nu_grad1']:.4f} "
        f"Nu_halfway={res['nu_halfway']:.4f} "
        f"Nu_left={res['nu_left_grad2']:.4f} Nu_right={res['nu_right_grad2']:.4f} "
        f"err={err_pct:.2f}% u_max={res['u_max']:.4f}",
        flush=True,
    )
    return {
        "nx": nx,
        "ra": ra,
        "pr": pr,
        "tau": tau,
        "tau_T": round(res["tau_T"], 6),
        "steps": steps,
        "elapsed_s": res["elapsed_s"],
        "last_resid_u": res["last_resid_u"],
        "last_resid_T": res["last_resid_T"],
        "nu": nu,
        "nu_grad1": res["nu_grad1"],
        "nu_halfway": res["nu_halfway"],
        "nu_left": res["nu_left_grad2"],
        "nu_right": res["nu_right_grad2"],
        "nu_ref": NU_DE_VAHL_DAVIS_1E4,
        "err_pct": round(err_pct, 3),
        "u_max": res["u_max"],
        "T_min": res["T_min"],
        "T_max": res["T_max"],
        "nu_history": res["nu_history"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", nargs="+", type=int, default=[64, 128])
    ap.add_argument("--ra", type=float, default=1e4)
    ap.add_argument("--pr", type=float, default=0.71)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="/tmp/cavity_natural_convection_verified.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(32)

    results = {"case": "cavity_natural_convection_ra1e4_dvd"}
    grids = {}
    for nx in args.nx:
        grids[str(nx)] = run_case(nx, args.ra, args.pr, args.tau, args.steps, device)
    results["grids"] = grids

    devs = [grids[str(nx)]["err_pct"] for nx in args.nx]
    # 收敛判定：误差 ≤3% 且随网格细化单调下降
    err_decreased = devs[-1] < devs[0]
    # 网格收敛（Richardson 意义上的稳定）：两档 Nu 相对差 < 5%
    grid_stable = (
        100.0
        * abs(grids[str(args.nx[-1])]["nu"] - grids[str(args.nx[0])]["nu"])
        / grids[str(args.nx[0])]["nu"]
        < 5.0
    )
    verdict = {
        "err_pct": devs,
        "err_decreased": err_decreased,
        "grid_stable_pct": round(
            100.0
            * abs(grids[str(args.nx[-1])]["nu"] - grids[str(args.nx[0])]["nu"])
            / grids[str(args.nx[0])]["nu"],
            2,
        ),
        "pass": all(d <= 3.0 for d in devs) and err_decreased and grid_stable,
    }
    results["convergence"] = verdict
    results["verified"] = bool(verdict["pass"])
    results["verified_date"] = "2026-08-19"
    results["description"] = (
        "自然对流方腔 Ra=1e4 Pr=0.71，双分布函数热 LBM（D2Q9 速度 + D2Q5 温度），"
        "de Vahl Davis 1983 验证"
    )
    results["lattice"] = "D2Q9+D2Q5"
    results["collision"] = "bgk (Guo force)"
    results["boundary"] = (
        "速度: pre-streaming 半程反弹(四壁 no-slip)；温度: 左壁 T=1 右壁 T=0 "
        "anti-bounce-back，上下绝热 bounce-back"
    )
    results["extrap"] = "none"
    results["reference"] = "de Vahl Davis 1983, Ra=1e4, Pr=0.71: Nu=2.243"

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"verdict: {json.dumps(verdict)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
