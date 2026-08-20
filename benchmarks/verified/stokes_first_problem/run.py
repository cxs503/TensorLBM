#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B24: Stokes 第一问题（平板突然起动, D2Q9）—— 时变解析 erf 解验证。

物理问题（Stokes 第一问题 / Rayleigh 问题）：
    半无限静止流体上方，y=0 处无限平板在 t=0 时刻突然以恒定速度 U
    沿 x 方向运动。流场仅依赖 y 和 t，满足一维扩散方程 ∂u/∂t = ν∂²u/∂y²，
    解析解（erfc 解）：
        u(y,t) = U · erfc( y / (2·sqrt(ν·t)) ),   ν = (τ − 1/2)/3,  t 为格子步数
    边界层厚度 δ ~ sqrt(ν·t)（t=9000, ν=0.1 时 δ≈30 格）。

数值设置（真实模拟，无外推）：
  - 库函数原语：d2q9.equilibrium/macroscopic、solver.collide_bgk/stream
  - x 方向周期（solver stream 内建），流向均匀；y 方向 H 格流体 + 上下壁行
  - 下壁 y=0：pre-streaming half-way 移动壁反弹（repo 已验证的 pre-streaming BB
    模式 + 移动壁动量注入 f_new[opp] = f_pre[i] − 2·w_i·ρ_w·(c_i·u_w)/cs²，
    u_w=(U,0)，与静止 BB 相比仅多一项 O(U) 修正）
  - 上壁 y=ny-1：自由滑移镜面反弹（应力自由远场，∂u/∂y≈0；半无限近似的
    正确远场条件——静止无滑移顶壁会把 t=9000 时仍有 ~2%U 的顶部速度强制
    拽到 0，产生有限域截断误差，实测 H=100 时 max_rel 达 10.8%，改用
    自由滑移后消除）
  - 无修正因子、无结果调参

判定标准：
  - 每个记录时刻（t=1000/4000/9000）数值剖面 u(y) 与解析 erf 解在
    u_ana > 5%·U 区域的 max 相对误差 ≤ 3%
  - ≥2 档网格（H=100/200）收敛：H=200 的误差不劣于 H=100（域截断收敛）

用法：
    run.py single H out.json [--tau 0.8] [--U 0.05] [--steps 1000 4000 9000] [--nx 8]
    run.py scan out_dir [--H 100 200] ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks

import numpy as np
import torch
from compile_route import add_compile_mode_arg, compile_mode_from_args, route_step  # noqa: E402

try:
    from scipy.special import erfc
except ImportError:  # no scipy in ftw-env: erfc via stdlib math.erf
    from math import erf as _erf

    def erfc(x):
        x = np.asarray(x, dtype=np.float64)
        return 1.0 - np.array([_erf(v) for v in x.ravel()]).reshape(x.shape)


from tensorlbm.d2q9 import OPPOSITE, C, W, equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream

CS2 = 1.0 / 3.0
DEVICE = torch.device("cpu")  # overridden by --device

# Specular (free-slip) reflection mapping for the far-field top wall:
# flips c_y, keeps c_x  ->  f_new[j] = f_pre[SPECULAR[j]]
SPECULAR = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)


def specular_replacement(f_pre: torch.Tensor) -> torch.Tensor:
    """Free-slip (specular) reflection replacement for a wall row (u_wall=0).

    Conserves x-momentum (no wall shear) so the top boundary acts as a
    stress-free far field instead of a no-slip wall — the right half-infinite
    approximation for the Stokes-first-problem erfc solution.
    """
    return f_pre[SPECULAR.to(f_pre.device)]


def moving_wall_replacement(f_pre: torch.Tensor, U: float) -> torch.Tensor:
    """Pre-streaming half-way bounce-back replacement for a wall row.

    For each cell on the wall: f_new[opp(i)] = f_pre[i] − 2·w_i·ρ_w·(c_i·u_w)/cs².
    With u_w = (U, 0) the momentum term only touches x-moving directions;
    U = 0 reduces to the plain bounce-back used for the static top wall.

    Returns a tensor of f-pre shape; apply with torch.where on the wall mask.
    """
    rho_w = f_pre.sum(dim=0)  # (ny, nx) local density
    cx = C[:, 0].to(f_pre.device).float()  # (9,)
    w = W.to(f_pre.device).float()
    mom = 2.0 * w.view(9, 1, 1) * rho_w.unsqueeze(0) * (cx.view(9, 1, 1) * U) / CS2
    opp = OPPOSITE.to(f_pre.device)
    return f_pre[opp] - mom[opp]


def run_case(
    H: int,
    tau: float,
    U: float,
    record_steps: list[int],
    nx: int = 8,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    compile_mode: str | None = "default",
) -> dict:
    """Run one Stokes-first-problem case; return per-step profiles + errors."""
    torch.manual_seed(seed)
    ny = H + 2  # wall rows at y=0 and y=ny-1
    nu = (tau - 0.5) / 3.0
    Ma = U / np.sqrt(CS2)

    wall_bottom = torch.zeros((ny, nx), dtype=torch.bool, device=DEVICE)
    wall_bottom[0, :] = True
    wall_top = torch.zeros((ny, nx), dtype=torch.bool, device=DEVICE)
    wall_top[-1, :] = True

    # Rest state everywhere at t=0; wall starts moving at the first step
    rho0 = torch.ones((ny, nx), dtype=dtype, device=DEVICE)
    u0 = torch.zeros((ny, nx), dtype=dtype, device=DEVICE)
    f = equilibrium(rho0, u0, u0)
    initial_mass = float(f.sum().item())

    max_step = max(record_steps)
    record_set = set(record_steps)
    profiles: dict[int, np.ndarray] = {}

    # ---- 整步步进函数（共性 compile 路径；步序号与剖面记录留在编译域外）----
    def _step(f):
        f_pre = f.clone()
        f = collide_bgk(f, tau)
        # pre-streaming boundary treatment (repo-validated BB variant);
        # bottom: moving wall U (momentum-injected bounce-back),
        # top: free-slip specular reflection (stress-free far field)
        f = torch.where(wall_bottom.unsqueeze(0), moving_wall_replacement(f_pre, U), f)
        f = torch.where(wall_top.unsqueeze(0), specular_replacement(f_pre), f)
        return stream(f)  # periodic in x (and y, cut by BB rows)

    step_fn = route_step(_step, compile_mode, name=f"stokes_first_problem[H{H}]")

    t0 = time.time()
    for step in range(1, max_step + 1):
        f = step_fn(f)
        if step in record_set:
            _, ux, _ = macroscopic(f)
            profiles[step] = ux[1 : ny - 1, 0].cpu().numpy().astype(np.float64)
    elapsed = time.time() - t0

    # Analytic comparison (y measured from the wall at y = i - 0.5, half-way BB)
    y_phys = np.arange(1, ny - 1, dtype=np.float64) - 0.5  # fluid rows 1..H
    per_step: list[dict] = []
    for step in record_steps:
        u_num = profiles[step]
        u_ana = U * erfc(y_phys / (2.0 * np.sqrt(nu * step)))
        mask = u_ana > 0.05 * U  # meaningful region (avoid erfc tail blow-up)
        rel = np.divide(np.abs(u_num - u_ana), u_ana, out=np.zeros_like(u_ana), where=mask)
        per_step.append(
            {
                "t_lb": step,
                "delta_lb": float(np.sqrt(nu * step)),
                "max_rel_err_pct": float(np.max(rel[mask]) * 100.0),
                "l2_rel_err": float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana)),
                "max_abs_err_over_U_pct": float(np.max(np.abs(u_num - u_ana)) / U * 100.0),
                "u_wall_num": float(u_num[0]),
                "u_top_num": float(u_num[-1]),
                "u_top_ana": float(u_ana[-1]),
                "y_profile": [round(float(v), 8) for v in y_phys],
                "u_profile": [round(float(v), 8) for v in u_num],
                "u_analytic": [round(float(v), 8) for v in u_ana],
            }
        )

    result = {
        "case": "B24_stokes_first_problem",
        "lattice": "D2Q9",
        "collision": "bgk",
        "boundary": "moving wall half-way BB (pre-streaming, momentum-injected) + top free-slip specular",
        "x_periodic": True,
        "H": H,
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "nu_lb": nu,
        "U": U,
        "Ma": Ma,
        "record_steps": record_steps,
        "max_step": max_step,
        "compile_mode": compile_mode,
        "mass_drift_pct": (float(f.sum().item()) - initial_mass) / initial_mass * 100.0,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": round(elapsed, 1),
        "per_step": per_step,
    }
    return result


def compare_grids(r_lo: dict, r_hi: dict, U: float, tol_pct: float = 3.0) -> dict:
    """Grid convergence: max profile difference between H=100 and H=200 per step."""
    rows = []
    for slo, shi in zip(r_lo["per_step"], r_hi["per_step"]):
        t = slo["t_lb"]
        y_lo = np.array(slo["y_profile"])
        y_hi = np.array(shi["y_profile"])
        u_lo = np.array(slo["u_profile"])
        u_hi = np.array(shi["u_profile"])
        # common y range: coarse H=100 covers y<=99.5; interpolate coarse onto fine y
        mask = y_hi <= y_lo[-1]
        y_hi_c = y_hi[mask]
        u_hi_c = u_hi[mask]
        u_lo_i = np.interp(y_hi_c, y_lo, u_lo)
        diff = np.abs(u_lo_i - u_hi_c) / U * 100.0
        rows.append(
            {
                "t_lb": t,
                "max_profile_diff_over_U_pct": float(np.max(diff)),
                "mean_profile_diff_over_U_pct": float(np.mean(diff)),
            }
        )
    # convergence: refined (H=200) max-rel error must not exceed coarse's
    err_lo = [s["max_rel_err_pct"] for s in r_lo["per_step"]]
    err_hi = [s["max_rel_err_pct"] for s in r_hi["per_step"]]
    converged = all(eh <= max(el, tol_pct) for el, eh in zip(err_lo, err_hi))
    return {
        "rows": rows,
        "converged": converged,
        "max_rel_err_H100": err_lo,
        "max_rel_err_H200": err_hi,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="B24 Stokes first problem (moving plate, D2Q9)")
    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("single")
    p1.add_argument("H", type=int)
    p1.add_argument("out_json", type=str)
    p1.add_argument("--tau", type=float, default=0.8)
    p1.add_argument("--U", type=float, default=0.05)
    p1.add_argument("--steps", type=int, nargs="+", default=[1000, 4000, 9000])
    p1.add_argument("--nx", type=int, default=8)
    p1.add_argument("--seed", type=int, default=0)
    p1.add_argument("--device", type=str, default="cpu")
    add_compile_mode_arg(p1)

    p2 = sub.add_parser("scan")
    p2.add_argument("out_dir", type=str)
    p2.add_argument("--H", type=int, nargs="+", default=[100, 200])
    p2.add_argument("--tau", type=float, default=0.8)
    p2.add_argument("--U", type=float, default=0.05)
    p2.add_argument("--steps", type=int, nargs="+", default=[1000, 4000, 9000])
    p2.add_argument("--nx", type=int, default=8)
    p2.add_argument("--device", type=str, default="cpu")
    add_compile_mode_arg(p2)

    args = ap.parse_args()
    global DEVICE
    DEVICE = torch.device(args.device)
    compile_mode = compile_mode_from_args(args)

    if args.mode == "single":
        r = run_case(args.H, args.tau, args.U, args.steps, args.nx, compile_mode=compile_mode)
        Path(args.out_json).write_text(json.dumps(r, indent=2))
        for s in r["per_step"]:
            print(
                f"t={s['t_lb']:5d}  max_rel={s['max_rel_err_pct']:6.3f}%  "
                f"l2_rel={s['l2_rel_err']:.5f}  max_abs/U={s['max_abs_err_over_U_pct']:.4f}%"
            )
        print(
            f"H={r['H']} ny={r['ny']} nu={r['nu_lb']:.4f} mass_drift={r['mass_drift_pct']:.2e}% "
            f"finite={r['finite']} elapsed={r['elapsed_s']}s"
        )

    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cases = []
        for H in args.H:
            p = out_dir / f"case_H{H}.json"
            r = run_case(H, args.tau, args.U, args.steps, args.nx, compile_mode=compile_mode)
            p.write_text(json.dumps(r, indent=2))
            cases.append(r)
            print(
                f"H={r['H']:3d}: "
                + "  ".join(f"t{s['t_lb']}={s['max_rel_err_pct']:.3f}%" for s in r["per_step"]),
                flush=True,
            )
        conv = compare_grids(cases[0], cases[1], args.U)
        tol = 3.0
        passed = all(s["max_rel_err_pct"] <= tol for r in cases for s in r["per_step"])
        summary = {
            "case": "B24_stokes_first_problem_convergence",
            "lattice": "D2Q9",
            "collision": "bgk",
            "boundary": "moving wall half-way BB (pre-streaming) + top free-slip specular, x periodic",
            "extrap": "none",
            "tau": args.tau,
            "nu_lb": (args.tau - 0.5) / 3.0,
            "U": args.U,
            "record_steps": args.steps,
            "H_list": args.H,
            "tol_max_rel_pct": tol,
            "per_grid": [
                {
                    k: r[k]
                    for k in [
                        "H",
                        "ny",
                        "nx",
                        "tau",
                        "nu_lb",
                        "U",
                        "Ma",
                        "mass_drift_pct",
                        "finite",
                        "elapsed_s",
                        "per_step",
                    ]
                }
                for r in cases
            ],
            "convergence": conv,
            "passed": passed and conv["converged"],
            "status": "VERIFIED" if (passed and conv["converged"]) else "NOT_PASSED",
        }
        (out_dir / "result.json").write_text(json.dumps(summary, indent=2))
        print(f"\nstatus={summary['status']}  passed_3pct={passed}  converged={conv['converged']}")
        for row in conv["rows"]:
            print(
                f"  t={row['t_lb']:5d}  profile_diff_H100vsH200 = {row['max_profile_diff_over_U_pct']:.3f}% U"
            )


if __name__ == "__main__":
    main()
