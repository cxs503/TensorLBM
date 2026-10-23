#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B25: 平板边界层 Blasius 相似解验证 (D2Q9 BGK, 外部绕流).

物理：均匀流 U 掠过半无限平板（前缘 x=le，无滑移），层流边界层。
Blasius 相似解：u/U = f'(eta), eta = y*sqrt(U/(nu*x'))，x' 为距前缘距离；
局部摩擦系数 C_f = 2*tau_w/(rho*U^2) = 0.664/sqrt(Re_x), Re_x = U*x'/nu。

真实模拟，无外推 (extrap: none)：
  - 库函数原语：d2q9.equilibrium/macroscopic/OPPOSITE,
    solver.collide_bgk/stream, boundaries.zou_he_inlet_velocity
  - 半程反弹（pre-streaming 变体，与已验证 Poiseuille B13 一致）在平板上
  - 对称（镜面反射 specular）顶边界 = 顶对称；前缘上游/尾缘下游底边界亦为对称
  - 出口零梯度（f[:,:,-1] = f[:,:,-2]）；入口均匀流（Zou-He 速度边界）
  - 无修正因子、无结果调参

参考解：自编 RK4 打靶求解 Blasius ODE f''' + f*f'' = 0
(f(0)=f'(0)=0, f'(inf)=1)，f''(0) 打靶收敛到 ~0.33206（标准值）。

两档网格（板长 200 / 400 格）证明收敛：
  - plate200: nx=400 ny=100 le=20 板长 200 (x in [20,220)) 测 x=200 (x'=180)
  - plate400: nx=600 ny=100 le=20 板长 400 (x in [20,420)) 测 x=400 (x'=380)
U=0.05, nu=0.01 (tau=0.53), Ma~0.087。

用法:
    run.py single <plate200|plate400> <out.json> [--U U] [--nu nu]
        [--min-steps N] [--max-steps N] [--seed 0] [--device cpu]
    run.py scan <out_dir> [--min-steps N] [--max-steps N]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

import numpy as np
import torch

from tensorlbm.boundaries import zou_he_inlet_velocity
from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream
from tensorlbm.utils import get_reproducibility_metadata

CS2 = 1.0 / 3.0

GRIDS = {
    "plate200": dict(name="plate200", nx=400, ny=100, le=20, plate_len=200, meas_x=200),
    "plate400": dict(name="plate400", nx=600, ny=100, le=20, plate_len=400, meas_x=400),
}


# ---------------------------------------------------------------------------
# Blasius 参考解：f''' + f*f'' = 0 的 RK4 打靶解（自洽标准解，非外推）
# ---------------------------------------------------------------------------
def blasius_ode_solution(
    eta_max: float = 15.0, n: int = 30001
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve Blasius ODE by RK4 shooting on f''(0); return (eta, f', f''(0))."""
    h = eta_max / n

    def rhs(y):
        f, fp, fpp = y
        return np.array([fp, fpp, -f * fpp])

    def integrate(s):
        y = np.array([0.0, 0.0, s])
        vals = np.empty(n + 1)
        vals[0] = 0.0
        for i in range(n):
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * h * k1)
            k3 = rhs(y + 0.5 * h * k2)
            k4 = rhs(y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            vals[i + 1] = y[1]
        return vals, float(y[1])

    # bisection on f''(0) so that f'(eta_max) -> 1
    lo, hi = 0.20, 0.60
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        _, fp_end = integrate(mid)
        if fp_end < 1.0:
            lo = mid
        else:
            hi = mid
    s0 = 0.5 * (lo + hi)
    vals, fp_end = integrate(s0)
    eta = np.linspace(0.0, eta_max, n + 1)
    return eta, vals, s0


# ---------------------------------------------------------------------------
# 边界条件（库函数 + 标准镜面反射/零梯度，均为教材公式）
# ---------------------------------------------------------------------------
def apply_bc(
    f: torch.Tensor, u_in: float, le: int, plate_end: int, ny: int, nx: int
) -> torch.Tensor:
    """Post-stream boundary treatment.

    - inlet  (x=0):      Zou-He 均匀速度入口 (库函数)
    - outlet (x=nx-1):   零梯度
    - top    (y=ny-1):   对称（镜面反射, cy<0 未知方向 <- cy>0 已知方向）
    - bottom (y=0):      对称（镜面反射）在前缘上游与尾缘下游；
                         平板区域为 pre-streaming 半程反弹（在主循环中处理）
    """
    f = zou_he_inlet_velocity(f, u_in)
    f = f.clone()
    f[:, :, -1] = f[:, :, -2]  # outlet zero-gradient

    # top symmetry: unknown cy=-1 dirs (4,7,8) <- reflected cy=+1 dirs (2,6,5)
    f[4, ny - 1, :] = f[2, ny - 1, :]
    f[7, ny - 1, :] = f[6, ny - 1, :]
    f[8, ny - 1, :] = f[5, ny - 1, :]

    # bottom symmetry upstream of LE and downstream of TE:
    # unknown cy=+1 dirs (2,5,6) <- reflected cy=-1 dirs (4,8,7)
    f[2, 0, :le] = f[4, 0, :le]
    f[5, 0, :le] = f[7, 0, :le]
    f[6, 0, :le] = f[8, 0, :le]
    if plate_end < nx:
        f[2, 0, plate_end:] = f[4, 0, plate_end:]
        f[5, 0, plate_end:] = f[7, 0, plate_end:]
        f[6, 0, plate_end:] = f[8, 0, plate_end:]
    return f


# ---------------------------------------------------------------------------
# 单档网格模拟
# ---------------------------------------------------------------------------
def run_case(
    grid: dict,
    u_in: float,
    nu: float,
    min_steps: int,
    max_steps: int,
    out_path: str,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(seed)
    dev = torch.device(device)
    nx, ny = grid["nx"], grid["ny"]
    le, plate_len, meas_x = grid["le"], grid["plate_len"], grid["meas_x"]
    plate_end = le + plate_len
    tau = 3.0 * nu + 0.5
    x_prime = meas_x - le  # 距前缘距离
    re_x = u_in * x_prime / nu
    cf_blasius = 0.664 / math.sqrt(re_x)

    # 平板（底行 x in [le, plate_end)）— pre-streaming 半程反弹
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=dev)
    wall[0, le:plate_end] = True

    # 初始：全场均匀流
    rho0 = torch.ones((ny, nx), device=dev)
    ux0 = torch.full((ny, nx), u_in, device=dev)
    uy0 = torch.zeros((ny, nx), device=dev)
    f = equilibrium(rho0, ux0, uy0)
    initial_mass = float(f.sum().item())

    # 收敛监视：测量列剖面 u(y), y=1..40
    prof_rows = list(range(1, 41))
    prev_prof: list[float] | None = None
    steady = False
    t0 = time.time()
    step = 0
    for step in range(1, max_steps + 1):
        f_pre = f.clone()
        f = collide_bgk(f, tau)
        f = torch.where(wall.unsqueeze(0), f_pre[OPPOSITE.to(dev)], f)  # pre-stream BB
        f = stream(f)
        f = apply_bc(f, u_in, le, plate_end, ny, nx)

        if step % 2000 == 0:
            _, ux, _ = macroscopic(f)
            prof = [float(ux[y, meas_x]) for y in prof_rows]
            if prev_prof is not None:
                rel = max(abs(a - b) for a, b in zip(prof, prev_prof, strict=True)) / max(
                    u_in, 1e-12
                )
                if step >= min_steps and rel < 1e-5:
                    steady = True
                    break
            prev_prof = prof
    elapsed = time.time() - t0

    # 末 1000 步时间平均剖面（稳态，去噪声）
    prof_acc = torch.zeros(ny, device=dev)
    for _ in range(1000):
        f_pre = f.clone()
        f = collide_bgk(f, tau)
        f = torch.where(wall.unsqueeze(0), f_pre[OPPOSITE.to(dev)], f)
        f = stream(f)
        f = apply_bc(f, u_in, le, plate_end, ny, nx)
        _, ux, _ = macroscopic(f)
        prof_acc += ux[:, meas_x]
    prof_acc /= 1000.0

    rho, ux, uy = macroscopic(f)
    u_num = prof_acc.cpu().numpy().astype(np.float64)  # (ny,)
    mass_drift_pct = (float(f.sum().item()) - initial_mass) / initial_mass * 100.0

    # ---- Blasius 对比：eta = (y - y_w)*sqrt(U/(nu x')), y_w = 0.5 (半程反弹壁面)
    y_wall = 0.5
    y_phys = np.arange(ny, dtype=np.float64) - y_wall
    eta_phys = y_phys * math.sqrt(u_in / (nu * x_prime))

    eta_ref, fp_ref, s0 = blasius_ode_solution()
    fp_interp = np.interp(eta_phys, eta_ref, fp_ref)  # Blasius u/U at each row

    # 剖面误差：eta in [0, eta_cut]（完整覆盖边界层，超出部分 u/U~1 无信息量）
    eta_cut = 5.5
    sel = (eta_phys > 0.0) & (eta_phys <= eta_cut)
    eta_s = eta_phys[sel]
    u_s = u_num[sel] / u_in
    blas_s = fp_interp[sel]
    l2_rel = float(np.linalg.norm(u_s - blas_s) / np.linalg.norm(blas_s))
    max_abs = float(np.max(np.abs(u_s - blas_s)))
    mask_strong = blas_s > 0.05
    max_rel_strong = float(
        np.max(np.abs(u_s[mask_strong] - blas_s[mask_strong]) / blas_s[mask_strong]) * 100.0
    )

    # ---- 壁面摩擦 C_f
    tau_w1 = 2.0 * nu * float(prof_acc[1].item())  # u(1)/(0.5), rho=1
    cf_num = 2.0 * tau_w1 / (u_in * u_in)
    cf_err_pct = (cf_num - cf_blasius) / cf_blasius * 100.0
    # 交叉验证：前 4 个流体点的线性拟合斜率（距壁 0.5..3.5）
    d = np.arange(1, 5, dtype=np.float64) - 0.5
    slope_fit = float(np.polyfit(d, u_num[1:5], 1)[0])
    cf_fit = 2.0 * nu * slope_fit / (u_in * u_in)
    cf_fit_err_pct = (cf_fit - cf_blasius) / cf_blasius * 100.0

    u_freestream = float(u_num[-1] / u_in)  # 顶行 u/U（应~1）

    result = {
        "case": "B25_blasius_flat_plate",
        "grid": grid["name"],
        "lattice": "D2Q9",
        "collision": "bgk",
        "boundary": "zou_he_velocity_inlet + zero-gradient outlet + top symmetry (specular) + pre-streaming half-way bounce-back plate",
        "extrap": "none",
        "nx": nx,
        "ny": ny,
        "le": le,
        "plate_len": plate_len,
        "plate_end": plate_end,
        "meas_x": meas_x,
        "x_prime": x_prime,
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Ma": u_in / math.sqrt(CS2),
        "re_x": re_x,
        "n_steps": step,
        "steady": steady,
        "min_steps": min_steps,
        "elapsed_s": round(elapsed, 1),
        "mass_drift_pct": mass_drift_pct,
        "finite": bool(torch.isfinite(f).all().item()),
        "blasius_fpp0": float(s0),
        "profile": {
            "eta_cut": eta_cut,
            "l2_rel_err": l2_rel,
            "max_abs_err": max_abs,
            "max_rel_err_strong_pct": max_rel_strong,
            "n_points": int(sel.sum()),
            "eta": [round(float(v), 6) for v in eta_s],
            "u_over_U": [round(float(v), 8) for v in u_s],
            "blasius_fp": [round(float(v), 8) for v in blas_s],
        },
        "cf": {
            "cf_num_tauw1": cf_num,
            "cf_blasius": cf_blasius,
            "cf_err_pct": cf_err_pct,
            "cf_num_fit": cf_fit,
            "cf_fit_err_pct": cf_fit_err_pct,
            "u_wall1": float(prof_acc[1].item()),
        },
        "freestream": {"u_over_U_top": u_freestream},
        "reproducibility": get_reproducibility_metadata(),
    }
    Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="B25 平板边界层 Blasius (D2Q9 BGK)")
    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("single")
    p1.add_argument("grid", choices=list(GRIDS))
    p1.add_argument("out_json", type=str)
    p1.add_argument("--U", type=float, default=0.05)
    p1.add_argument("--nu", type=float, default=0.01)
    p1.add_argument("--min-steps", type=int, default=20000)
    p1.add_argument("--max-steps", type=int, default=60000)
    p1.add_argument("--seed", type=int, default=0)
    p1.add_argument("--device", type=str, default="cpu")

    p2 = sub.add_parser("scan")
    p2.add_argument("out_dir", type=str)
    p2.add_argument("--U", type=float, default=0.05)
    p2.add_argument("--nu", type=float, default=0.01)
    p2.add_argument("--min-steps", type=int, default=20000)
    p2.add_argument("--max-steps", type=int, default=60000)

    args = ap.parse_args()
    if args.mode == "single":
        r = run_case(
            GRIDS[args.grid],
            args.U,
            args.nu,
            args.min_steps,
            args.max_steps,
            args.out_json,
            args.seed,
            args.device,
        )
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in [
                        "grid",
                        "nx",
                        "ny",
                        "u_in",
                        "nu",
                        "tau",
                        "re_x",
                        "n_steps",
                        "steady",
                        "mass_drift_pct",
                        "finite",
                        "elapsed_s",
                    ]
                }
                | {
                    "profile_l2": r["profile"]["l2_rel_err"],
                    "profile_max_abs": r["profile"]["max_abs_err"],
                    "profile_max_rel_pct": r["profile"]["max_rel_err_strong_pct"],
                    "cf_err_pct": r["cf"]["cf_err_pct"],
                    "cf_fit_err_pct": r["cf"]["cf_fit_err_pct"],
                    "u_top": r["freestream"]["u_over_U_top"],
                },
                indent=2,
            )
        )
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cases = []
        for gname in ("plate200", "plate400"):
            p = out_dir / f"{gname}.json"
            r = run_case(GRIDS[gname], args.U, args.nu, args.min_steps, args.max_steps, str(p))
            cases.append(r)
            print(
                f"{r['grid']}: Re_x={r['re_x']:.0f} steps={r['n_steps']} steady={r['steady']} "
                f"profile_L2={r['profile']['l2_rel_err']:.5f} max_rel={r['profile']['max_rel_err_strong_pct']:.3f}% "
                f"Cf_err={r['cf']['cf_err_pct']:+.3f}% u_top={r['freestream']['u_over_U_top']:.5f}",
                flush=True,
            )
        g1, g2 = cases[0], cases[1]
        pass_profile = g1["profile"]["l2_rel_err"] <= 0.03 and g2["profile"]["l2_rel_err"] <= 0.03
        pass_cf = abs(g1["cf"]["cf_err_pct"]) <= 3.0 and abs(g2["cf"]["cf_err_pct"]) <= 3.0
        converged = g2["profile"]["l2_rel_err"] <= g1["profile"]["l2_rel_err"]
        summary = {
            "case": "B25_blasius_flat_plate_convergence",
            "lattice": "D2Q9",
            "collision": "bgk",
            "boundary": "zou_he_velocity_inlet + zero-gradient outlet + top symmetry + pre-streaming BB plate",
            "extrap": "none",
            "U": args.U,
            "nu": args.nu,
            "min_steps": args.min_steps,
            "max_steps": args.max_steps,
            "criteria": {
                "profile_l2_err_le_3pct": True,
                "cf_err_le_3pct": True,
                "grid_convergence": converged,
            },
            "per_grid": [
                {
                    "grid": r["grid"],
                    "nx": r["nx"],
                    "ny": r["ny"],
                    "plate_len": r["plate_len"],
                    "meas_x": r["meas_x"],
                    "x_prime": r["x_prime"],
                    "re_x": r["re_x"],
                    "n_steps": r["n_steps"],
                    "steady": r["steady"],
                    "profile_l2_rel_err": r["profile"]["l2_rel_err"],
                    "profile_max_rel_pct": r["profile"]["max_rel_err_strong_pct"],
                    "cf_err_pct": r["cf"]["cf_err_pct"],
                    "cf_fit_err_pct": r["cf"]["cf_fit_err_pct"],
                    "u_top": r["freestream"]["u_over_U_top"],
                    "mass_drift_pct": r["mass_drift_pct"],
                }
                for r in cases
            ],
            "verdict": {
                "pass": pass_profile and pass_cf and converged,
                "profile_pass": pass_profile,
                "cf_pass": pass_cf,
                "converged": converged,
            },
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary["verdict"], indent=2))


if __name__ == "__main__":
    main()
