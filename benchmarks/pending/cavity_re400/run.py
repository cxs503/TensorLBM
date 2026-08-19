#!/usr/bin/env python3
"""方腔流（lid-driven cavity）Re=400 benchmark — Ghia (1982) 验证.

真实模拟（无外推）：全部使用 TensorLBM 库函数
  - 碰撞: solver.collide_mrt (D2Q9 MRT)
  - 流场演化: solver.stream (周期 gather)
  - 壁面: boundaries.bounce_back_cells (静止壁全反弹) + lid_driven_cavity.zou_he_moving_lid (顶盖 Zou/He 动壁)
  - 平衡态/宏观量: d2q9.equilibrium / d2q9.macroscopic

物理设置
  - 方腔 nx×nx，顶盖 u=U0=0.06（格点单位），其余壁静止（无滑移）
  - 顶盖角点：先全反弹（速度 0），再 Zou/He 只覆盖内部格点（x=1..nx-2）——角点保持反弹
  - Re = U0·H/ν = 400，H=nx，ν=(τ−0.5)/3 ⇒ τ = 3·U0·nx/Re + 0.5
    → 128²: ν=0.0192/τ=0.5576，192²: ν=0.0288/τ=0.5864（均 <0.6 → MRT，与模块 run_lid_driven_cavity 的碰撞选择一致）
  - 网格: 128×128 与 192×192（≥2 档网格证明收敛）
  - 稳态: ≥100000 步或残差<1e-8；每 5000 步记录残差（内部格点 max|Δu|）

对比 Ghia et al. (1982) 表值（库内 GHIA_RE400 数据，129×129 多网格解）：
  - 中线 u(x=0.5, y) 与 v(x, y=0.5) 全剖面（17 点 × 2 条中线）
  - 关键点: u@y=0.25/0.5/0.75、v@x=0.25/0.5/0.75
  - 主涡涡心 (0.5547, 0.6055)

判定: 中线速度对 Ghia 最大偏差 ≤3% 且 128/192 收敛 → 写入 benchmarks/verified/cavity_re400/。
用法: PYTHONPATH=src python benchmarks/cavity_re400/run.py [--grids 128,192] [--max-steps 100000]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tensorlbm.boundaries import bounce_back_cells
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.lid_driven_cavity import (
    GHIA_RE400,
    make_cavity_wall_mask,
    zou_he_moving_lid,
)
from tensorlbm.solver import collide_mrt, stream

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

U_LID = 0.06          # 顶盖速度（格点单位，0.05–0.1 范围内）
RE = 400.0            # Re = U0·H/ν
MAX_STEPS = 100_000   # 稳态步数上限
RESID_TOL = 1e-8      # 残差阈值（内部格点 max|Δu| 每步）
CHECK_INTERVAL = 5000 # 残差记录间隔
GHIA_VORTEX = (0.5547, 0.6055)  # Ghia (1982) Re=400 主涡涡心
KEY_POINTS = {
    "u_y": [0.25, 0.5, 0.75],   # u(x=0.5, y) 关键 y
    "v_x": [0.25, 0.5, 0.75],   # v(x, y=0.5) 关键 x
}


def tau_from_re(nx: int, u_lid: float = U_LID, re: float = RE) -> float:
    """Re = u_lid·nx/ν, ν=(τ−0.5)/3 ⇒ τ = 3·u_lid·nx/Re + 0.5."""
    nu = u_lid * nx / re
    return 3.0 * nu + 0.5


def run_grid(
    nx: int,
    max_steps: int = MAX_STEPS,
    device: torch.device | None = None,
) -> dict:
    """运行单个网格的方腔流，返回全部诊断与剖面。"""
    device = device or torch.device("cpu")
    ny = nx
    tau = tau_from_re(nx)
    u_lid = U_LID
    torch.manual_seed(0)

    t_start = time.time()
    rho0 = torch.ones((ny, nx), device=device)
    u0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, u0, u0)

    # 静止壁（底/左/右）+ 顶盖行；顶盖内部格点随后由 Zou/He 覆盖，角点保持反弹
    wall_mask = make_cavity_wall_mask(ny, nx, device, include_top=False)
    interior = ~wall_mask
    # 残差统计用内部格点（去掉顶盖行，避免固定 u=U0 的格点干扰；顶盖下方第一行包含）
    interior_resid = interior.clone()
    interior_resid[-1, :] = False  # 顶盖行不参与残差

    residual_hist: list[dict] = []
    conv_ok = False
    stop_step = max_steps

    # 每个 CHECK_INTERVAL 窗口开头的速度场（用于窗口内最大变化量）
    _, ux_prev, uy_prev = macroscopic(f)
    ux_prev = ux_prev.detach().clone()
    uy_prev = uy_prev.detach().clone()

    for step in range(1, max_steps + 1):
        f = collide_mrt(f, tau=tau)
        f = stream(f)
        f = bounce_back_cells(f, wall_mask)
        f = zou_he_moving_lid(f, u_lid)

        if step % CHECK_INTERVAL == 0:
            _, ux, uy = macroscopic(f)
            du = torch.max(
                torch.abs(ux[interior_resid] - ux_prev[interior_resid]),
                torch.abs(uy[interior_resid] - uy_prev[interior_resid]),
            ).max().item()
            residual_hist.append({"step": step, "residual": float(du)})
            ux_prev = ux.detach().clone()
            uy_prev = uy.detach().clone()
            if du < RESID_TOL:
                conv_ok = True
                stop_step = step
                break

    elapsed = time.time() - t_start

    # 最终场
    rho, ux, uy = macroscopic(f)
    ux_w = ux.masked_fill(wall_mask, 0.0)
    uy_w = uy.masked_fill(wall_mask, 0.0)
    ux_np = ux_w.detach().cpu().numpy() / u_lid
    uy_np = uy_w.detach().cpu().numpy() / u_lid

    # ── 中线剖面 ──────────────────────────────────────────────
    x_mid, y_mid = nx // 2, ny // 2
    y_pos = np.linspace(0.0, 1.0, ny)   # 格点位置 i/(ny-1)
    x_pos = np.linspace(0.0, 1.0, nx)
    u_centerline = ux_np[:, x_mid]      # u(x=0.5, y)/U0
    v_centerline = uy_np[y_mid, :]      # v(x, y=0.5)/U0

    ghia = GHIA_RE400
    u_ghia_interp = np.interp(ghia["y"], y_pos, u_centerline)
    v_ghia_interp = np.interp(ghia["x"], x_pos, v_centerline)

    # ── 误差指标 ──────────────────────────────────────────────
    rmse_u = float(np.sqrt(np.mean((u_ghia_interp - np.array(ghia["u"])) ** 2)))
    rmse_v = float(np.sqrt(np.mean((v_ghia_interp - np.array(ghia["v"])) ** 2)))
    # 全 34 个 Ghia 表点上的最大绝对偏差（归一化到 U0）
    dev_u = np.abs(u_ghia_interp - np.array(ghia["u"]))
    dev_v = np.abs(v_ghia_interp - np.array(ghia["v"]))
    max_abs_dev = float(max(dev_u.max(), dev_v.max()))
    max_abs_dev_pct = 100.0 * max_abs_dev
    # 内部点（剔除近顶盖边界层 y≥0.9531 与对应 x 侧）的最大偏差
    inner_u_mask = np.array(ghia["y"]) < 0.9531
    inner_v_mask = np.array(ghia["x"]) < 0.9531
    max_abs_dev_inner_pct = 100.0 * float(
        max(dev_u[inner_u_mask].max(), dev_v[inner_v_mask].max())
    )

    # 关键点（u@y=0.25/0.5/0.75，v@x=0.25/0.5/0.75）
    # 注意：GHIA_RE400["y"] 是降序（1.0→0.0），np.interp 需要升序 xp
    ghia_y_asc = np.array(ghia["y"])[::-1]
    ghia_u_asc = np.array(ghia["u"])[::-1]
    key_points = []
    for yq in KEY_POINTS["u_y"]:
        lbm = float(np.interp(yq, y_pos, u_centerline))
        ref = float(np.interp(yq, ghia_y_asc, ghia_u_asc))
        key_points.append(
            {
                "profile": "u(x=0.5,y)",
                "at": yq,
                "lbm": lbm,
                "ghia": ref,
                "abs_dev": abs(lbm - ref),
                "rel_dev_pct": (abs(lbm - ref) / abs(ref) * 100.0) if abs(ref) >= 0.05 else None,
            }
        )
    for xq in KEY_POINTS["v_x"]:
        lbm = float(np.interp(xq, x_pos, v_centerline))
        ref = float(np.interp(xq, ghia["x"], ghia["v"]))
        key_points.append(
            {
                "profile": "v(x,y=0.5)",
                "at": xq,
                "lbm": lbm,
                "ghia": ref,
                "abs_dev": abs(lbm - ref),
                "rel_dev_pct": (abs(lbm - ref) / abs(ref) * 100.0) if abs(ref) >= 0.05 else None,
            }
        )
    key_rel = [k["rel_dev_pct"] for k in key_points if k["rel_dev_pct"] is not None]
    max_key_rel_pct = max(key_rel) if key_rel else None

    # ── 主涡涡心（内部最小速度点 + 二次抛物面细化）──────────
    # 搜索区收缩到 [2, nx-3]×[2, ny-3]，保证 3 点抛物模板不越界/不触碰 inf 边界行
    speed2 = ux_np**2 + uy_np**2
    speed2[0, :] = speed2[-1, :] = np.inf
    speed2[:, 0] = speed2[:, -1] = np.inf
    inner_slice = speed2[2 : ny - 2, 2 : nx - 2]
    iy1, ix1 = np.unravel_index(np.argmin(inner_slice), inner_slice.shape)
    iy0, ix0 = iy1 + 2, ix1 + 2

    def _parab_min(vals: np.ndarray) -> float:
        """3 点抛物插值最小值位置偏移（相对中心，单位：格距）。"""
        a, b, c = vals
        denom = a - 2.0 * b + c
        if abs(denom) < 1e-12:
            return 0.0
        return 0.5 * (a - c) / denom

    dx = _parab_min(speed2[iy0, ix0 - 1 : ix0 + 2])
    dy = _parab_min(speed2[iy0 - 1 : iy0 + 2, ix0])
    vx = (ix0 + dx) / (nx - 1)
    vy = (iy0 + dy) / (ny - 1)
    vortex = {"cell": [int(iy0), int(ix0)], "x": float(vx), "y": float(vy),
              "ghia": list(GHIA_VORTEX),
              "dist_to_ghia": float(np.hypot(vx - GHIA_VORTEX[0], vy - GHIA_VORTEX[1]))}

    # 辅助诊断：速度场前 5 个最小速度内部格点（判断最小点是否落在角涡而非主涡）
    flat_idx = np.argsort(speed2, axis=None)[:5]
    min_pts = []
    for fi in flat_idx:
        iy, ix = np.unravel_index(fi, speed2.shape)
        if np.isinf(speed2[iy, ix]):
            break
        min_pts.append({"cell": [int(iy), int(ix)], "x": round(ix / (nx - 1), 4),
                        "y": round(iy / (ny - 1), 4),
                        "speed": round(float(np.sqrt(speed2[iy, ix])), 6)})

    return {
        "nx": nx,
        "ny": ny,
        "tau": tau,
        "nu": (tau - 0.5) / 3.0,
        "re": RE,
        "u_lid": u_lid,
        "collision": "mrt",
        "lattice": "D2Q9",
        "boundary": "bounce_back_cells(静止壁) + zou_he_moving_lid(顶盖, 角点反弹)",
        "max_steps": max_steps,
        "steps_run": stop_step,
        "converged_resid": conv_ok,
        "final_residual": residual_hist[-1]["residual"] if residual_hist else None,
        "residual_hist": residual_hist,
        "elapsed_s": round(elapsed, 1),
        "rmse_u": rmse_u,
        "rmse_v": rmse_v,
        "max_abs_dev": max_abs_dev,
        "max_abs_dev_pct": max_abs_dev_pct,
        "max_abs_dev_inner_pct": max_abs_dev_inner_pct,
        "max_key_rel_pct": max_key_rel_pct,
        "key_points": key_points,
        "vortex": vortex,
        "min_speed_points": min_pts,
        "finite": bool(np.isfinite(ux_np).all() and np.isfinite(uy_np).all()),
        "mass_drift": float(rho.sum().item() - rho0.sum().item()),
        "u_centerline": [round(float(v), 8) for v in u_centerline],
        "v_centerline": [round(float(v), 8) for v in v_centerline],
        "u_ghia": ghia["u"],
        "v_ghia": ghia["v"],
        "ghia_y": ghia["y"],
        "ghia_x": ghia["x"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="128,192", help="逗号分隔的网格尺寸")
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=MAX_STEPS)
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    grids = [int(g) for g in args.grids.split(",") if g.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    results = {}
    for nx in grids:
        print(f"── nx={nx} 开始 ──", flush=True)
        res = run_grid(nx, max_steps=args.max_steps, device=device)
        results[str(nx)] = res
        print(
            f"nx={nx}: steps={res['steps_run']} 收敛={res['converged_resid']} "
            f"末残差={res['final_residual']:.2e} RMSE_u={res['rmse_u']:.5f} "
            f"RMSE_v={res['rmse_v']:.5f} max_abs_dev={res['max_abs_dev_pct']:.2f}% "
            f"内域max_abs_dev={res['max_abs_dev_inner_pct']:.2f}% "
            f"关键点max_rel={res['max_key_rel_pct']:.2f}% "
            f"涡心=({res['vortex']['x']:.4f},{res['vortex']['y']:.4f}) "
            f"距Ghia={res['vortex']['dist_to_ghia']:.4f} ({res['elapsed_s']}s)",
            flush=True,
        )

    # ── 收敛判定（≥2 档网格）──────────────────────────────────
    gs = list(results.keys())
    conv_report = {"grids": gs}
    if len(gs) >= 2:
        r1, r2 = results[gs[0]], results[gs[1]]
        # 关键点值在两档网格间的最大相对变化
        deltas = []
        for k1, k2 in zip(r1["key_points"], r2["key_points"]):
            v1, v2 = k1["lbm"], k2["lbm"]
            denom = max(abs(v1), abs(v2), 1e-9)
            deltas.append(abs(v2 - v1) / denom * 100.0)
        conv_report["max_key_change_pct"] = max(deltas)
        conv_report["err_decreased"] = (
            r2["max_abs_dev_pct"] <= r1["max_abs_dev_pct"] * 1.02
            and r2["max_key_rel_pct"] <= r1["max_key_rel_pct"] * 1.02
        )
        conv_report["vortex_dist_change"] = abs(
            r2["vortex"]["dist_to_ghia"] - r1["vortex"]["dist_to_ghia"]
        )
    # 网格收敛：关键点随细化变化小（<2%）且误差不增大
    conv_ok_grids = (
        conv_report.get("max_key_change_pct", 1e9) < 2.0
        and conv_report.get("err_decreased", False)
    )

    # ── 判定 ──────────────────────────────────────────────────
    all_finite = all(r["finite"] for r in results.values())
    # 主判据: 内域+全剖面最大偏差 ≤3%，关键点相对误差 ≤3%（|ref|≥0.05 处）
    worst_full = max(r["max_abs_dev_pct"] for r in results.values())
    worst_inner = max(r["max_abs_dev_inner_pct"] for r in results.values())
    worst_key = max(r["max_key_rel_pct"] or 0.0 for r in results.values())
    verdict = {
        "pass": bool(
            all_finite and worst_full <= 3.0 and worst_key <= 3.0 and conv_ok_grids
        ),
        "max_abs_dev_pct_all": worst_full,
        "max_abs_dev_pct_inner": worst_inner,
        "max_key_rel_pct": worst_key,
        "grid_convergence": conv_ok_grids,
        "all_finite": all_finite,
        "criteria": {
            "max_abs_dev_pct <= 3": worst_full <= 3.0,
            "max_key_rel_pct <= 3": worst_key <= 3.0,
            "grid_convergence": bool(conv_ok_grids),
            "finite": bool(all_finite),
        },
    }

    summary = {
        "case": "cavity_re400",
        "description": "方腔流 Re=400，D2Q9 MRT，Ghia(1982) 验证",
        "u_lid": U_LID,
        "re": RE,
        "collision": "mrt",
        "lattice": "D2Q9",
        "boundary": "bounce_back + zou_he_moving_lid（角点反弹速度0）",
        "steady_state": {"max_steps": args.max_steps, "resid_tol": RESID_TOL,
                         "check_interval": CHECK_INTERVAL},
        "grids": results,
        "convergence": conv_report,
        "verdict": verdict,
        "ghia_vortex_center": list(GHIA_VORTEX),
    }

    out_json = outdir / "result.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n结果写入: {out_json}")
    print(f"判定: {'PASS ✅' if verdict['pass'] else 'FAIL ❌'}")
    print(f"  max_abs_dev(全剖面)={worst_full:.2f}%  内域={worst_inner:.2f}%  "
          f"关键点max_rel={worst_key:.2f}%  网格收敛={conv_ok_grids}")
    return verdict["pass"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
