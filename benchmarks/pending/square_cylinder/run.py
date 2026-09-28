#!/usr/bin/env python3
"""方柱 Re=100 自由流涡脱落 — W8-B 基准复活（双测力 A/B）。

工程链与旧档（benchmarks/pending/square_cylinder/run.py @ 36224617f19）
完全同参同链，唯一新增 = 第二测力列：

  主列  wet：wet-node 表面链接 MEM，F = 2·Σ_{fluid→solid links} c_q·f_q(x_solid)
             （库 momentum_exchange_wet_node 的 2D 同式约简，库无 2D 实现；
             含对角 corner 链接；采样点 post-stream、pre-bounce-back）
  对照列 mem：旧档全固体 Ladd MEM（compute_obstacle_forces，含固体内部
             solid→solid 链接 = ghost 贡献）

两列同一 f、同一步、同一采样点；主列恰为对照列的"跨界链接"子集，
故 A/B 差值 = 纯测力口径差（ghost 链接贡献）。

参考（prereg.md 锁定，sha256 b9928c15…，2026-09-28T11:24:03Z 冻结）：
  主门参考 = 2D 低堵塞数值簇中位数 Cd_ref=1.4785 / St_ref=0.1456
  敏感性旁列 = Okajima 实验 (1.59/0.141)、Norberg (0.143)、Przulj@B=2.5% 内插
             (1.43/0.142)、旧档 Okajima 口径 (1.6/0.14)

库入口（零手写 collide/stream/equilibrium/bounce/zou_he/far_field）：
- tensorlbm.solver.collide_mrt（tau_field 逐格松弛 = sponge）+ stream
- tensorlbm.boundaries.far_field_bc_2d / make_sponge_strength /
  compute_obstacle_forces（对照列）
- tensorlbm.d2q9.equilibrium

用法：
    run.py --D 32 48 64 --device cuda:0 [--compile-mode default] --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


def _bootstrap(repo: str) -> None:
    benchmarks = str(Path(repo) / "benchmarks")
    if benchmarks not in sys.path:
        sys.path.insert(0, benchmarks)


import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO_DEFAULT = "/nfs/wangxi/worktrees/bm_w8"
_bootstrap(_REPO_DEFAULT)

from compile_route import add_compile_mode_arg, compile_mode_from_args, route_step  # noqa: E402

from tensorlbm.boundaries import (  # noqa: E402
    compute_obstacle_forces,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import OPPOSITE, C, equilibrium  # noqa: E402
from tensorlbm.solver import collide_mrt, stream  # noqa: E402

# ---- prereg 锁定参考（prereg.md §1.1/§1.2，冻结于任何新仿真输出之前）----
REF_CD = 1.4785  # 2D 低堵塞数值簇中位数（8 样本，prereg §1.1）
REF_ST = 0.1456  # 同簇 St 中位数（10 样本）
SENS = {  # 敏感性旁列（prereg §1.2，不判 PASS/FAIL）
    "okajima_exp": {"cd": 1.59, "st": 0.141},
    "norberg_exp": {"st": 0.143},
    "przulj_interp_b2.5pct": {"cd": 1.43, "st": 0.142},
    "old_archive_okajima": {"cd": 1.6, "st": 0.14},
}
OLD_ARCH = {32: (1.4816, 0.1482), 48: (1.4844, 0.1480)}  # 旧档 MEM 值（复现门）

# 域/播种参数（与旧档逐字同值）
DOMAIN_D = 40.0
SQUARE_X_D = 10.0
SPONGE_D = 10.0
SPONGE_ALPHA = 10.0
ST_SEED = 0.14
SEED_AMPL = 0.10
SEED_PERIODS = 2.0

N_BLOCKS = 12  # Cd 分块均值块数（≥10，prereg §2）


def square_mask(
    nx: int, ny: int, cx: float, cy: float, side: float, device: torch.device
) -> torch.Tensor:
    """正置方块掩码（与旧档逐字同实现），half-way 壁面有效边长精确 side。"""
    x0 = int(round(cx - side / 2.0))
    x1 = x0 + int(round(side)) - 1
    y0 = int(round(cy - side / 2.0))
    y1 = y0 + int(round(side)) - 1
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        indexing="ij",
    )
    return (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)


def surface_link_maps(solid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """wet-node MEM 的双侧链接权重图：(solid 侧 s2, fluid 侧 lf)。

    s2[q] 在"x 固体且 x−c_q 流体"处为 1（solid 侧）；lf[q] 在"x 流体且
    x+c_q 固体"处为 1（fluid 侧）。两者是同一组 fluid→solid 链接的两种
    索引（含对角 corner 链接），与库 momentum_exchange_wet_node /
    momentum_exchange_pair（links="full"）的 3D 约定同构。

    三列力（f 一律 post-stream、pre-bounce-back）：
      wet  主列：F = 2 · Σ_q c_q · Σ_x f[q,x]·s2[q,x]
            （刚流上固体节点、即将被 BB 反转的布居；库 docstring 明示
            勿用 fluid 侧单侧值——两步错位、自由流背景不抵消）
      pair 闭合列：F = Σ_q c_q · Σ_x ( f[q,x]·s2[q,x] + f[opp_q,x]·lf[q,x] )
            （进出配对 = 完成步的离散动量预算；稳态下应等于 wet 列，
            库在 3D 球上数值验证到 4 位小数。若 pair≈mem 则测力口径
            结论反转，故作判决性诊断）
      mem  对照列：compute_obstacle_forces 全固体和（= wet + 固体内部
            solid→solid 链接贡献，逐链接精确分解）
    """
    ny, nx = solid.shape
    fluid = ~solid
    s2 = torch.zeros((9, ny, nx), dtype=torch.float32, device=solid.device)
    lf = torch.zeros((9, ny, nx), dtype=torch.float32, device=solid.device)
    c = C.to(solid.device)
    for q in range(9):
        dx, dy = int(c[q, 0].item()), int(c[q, 1].item())
        if dx == 0 and dy == 0:
            continue  # 静止方向：c_q=0，无力贡献
        fluid_at_prev = torch.roll(fluid, shifts=(dy, dx), dims=(0, 1))
        s2[q] = (solid & fluid_at_prev).to(torch.float32)
        solid_at_next = torch.roll(solid, shifts=(-dy, -dx), dims=(0, 1))
        lf[q] = (fluid & solid_at_next).to(torch.float32)
    return s2, lf


def window_stats(
    cd_series: np.ndarray, cl_series: np.ndarray, w0: int, D: int, u_in: float
) -> dict:
    """分析窗统计（口径与旧档逐字一致 + 分块均值扩展）。

    - cd_mean：窗内总均值；另报 N_BLOCKS 分块均值的 std/min/max（≥10 块）
    - St：Cl 谱峰（Hann 窗 FFT + 抛物线内插）；滞回过零为交叉核对
    - cl_amp：窗内 max|Cl − mean(Cl)|（旧档同式）
    """
    cd_w = cd_series[w0:]
    cl_full = cl_series[w0:]
    cd_mean = float(cd_w.mean())
    blocks = [float(b.mean()) for b in np.array_split(cd_w, N_BLOCKS)]
    cl_w = cl_full - cl_full.mean()

    st_val = float("nan")
    st_cross = float("nan")
    if cl_w.size >= 256:
        spec = np.abs(np.fft.rfft(cl_w * np.hanning(cl_w.size)))
        k = int(np.argmax(spec[1:])) + 1
        delta = 0.0
        if 0 < k < spec.size - 1:
            a, b, c = spec[k - 1], spec[k], spec[k + 1]
            den = a - 2 * b + c
            if abs(den) > 1e-30:
                delta = float(0.5 * (a - c) / den)
        st_val = float((k + delta) / cl_w.size * D / u_in)
        thr = 0.25 * float(np.abs(cl_w).max())
        sig = np.where(cl_w >= thr, 1, np.where(cl_w <= -thr, -1, 0))
        idx = np.where(sig != 0, np.arange(sig.size), -1)
        last = np.maximum.accumulate(idx)
        state = np.where(last >= 0, sig[np.maximum(last, 0)], 0)
        crossings = np.where((state[:-1] < 0) & (state[1:] > 0))[0] + 1
        if len(crossings) >= 3:
            t_shed = float(np.median(np.diff(crossings)))
            st_cross = D / (u_in * t_shed)

    return {
        "cd": round(cd_mean, 4),
        "cd_block_std": round(float(np.std(blocks)), 6),
        "cd_block_min": round(min(blocks), 4),
        "cd_block_max": round(max(blocks), 4),
        "cd_blocks": [round(b, 4) for b in blocks],
        "st": round(st_val, 4) if math.isfinite(st_val) else None,
        "st_crossing": round(st_cross, 4) if math.isfinite(st_cross) else None,
        "cl_amp": float(np.abs(cl_w).max()),
    }


def run_case(
    D: int,
    re: float,
    u_in: float,
    steps: int,
    device: torch.device,
    compile_mode: str | None = "default",
    warmup_frac: float = 0.5,
    out_path: str | None = None,
) -> dict:
    nx = int(DOMAIN_D * D)
    ny = nx
    nu = u_in * D / re
    tau = 0.5 + 3.0 * nu

    mask = square_mask(nx, ny, SQUARE_X_D * D, ny / 2.0, D, device)
    s2, lf = surface_link_maps(mask)  # 2D wet-node 双侧链接图（solid/fluid 侧）
    cx_v = C.to(device)[:, 0].view(9, 1, 1).float()
    cy_v = C.to(device)[:, 1].view(9, 1, 1).float()
    opp_idx = OPPOSITE.to(device).long()

    sigma = make_sponge_strength(
        ny, nx, int(nx - SPONGE_D * D), int(SPONGE_D * D), power=2.0, device=device
    )
    tau_field = tau * (1.0 + SPONGE_ALPHA * sigma)

    rho0 = torch.ones((ny, nx), device=device)
    f = equilibrium(rho0, torch.full_like(rho0, u_in), torch.zeros_like(rho0))

    n_links = int(s2.sum().item())

    # ---- 三测力探针（wet 主列 / pair 闭合列 / mem 对照列）+ 库远场 BC ----
    def _triple_probe(f):
        fx_mem, fy_mem = compute_obstacle_forces(f, mask)  # 对照列（旧档全固体）
        links_solid = f * s2  # 主列：solid 侧跨界链接布居（post-stream）
        fx_wet = 2.0 * (cx_v * links_solid).sum()
        fy_wet = 2.0 * (cy_v * links_solid).sum()
        f_opp = torch.index_select(f, 0, opp_idx)
        paired = f * s2 + f_opp * lf  # 闭合列：进出配对（无因子 2）
        fx_pair = (cx_v * paired).sum()
        fy_pair = (cy_v * paired).sum()
        return (
            far_field_bc_2d(f, u_in, mask),
            fx_mem,
            fy_mem,
            fx_wet,
            fy_wet,
            fx_pair,
            fy_pair,
        )

    def _step_plain(f):
        return _triple_probe(stream(collide_mrt(f, tau, tau_field=tau_field)))

    def _step_seeded(f, uy_col):
        f, fxm, fym, fxw, fyw, fxp, fyp = _step_plain(f)
        f[:, :, 0] = equilibrium(
            torch.ones((ny, 1), device=f.device, dtype=f.dtype),
            torch.full((ny, 1), u_in, device=f.device, dtype=f.dtype),
            uy_col.view(ny, 1),
        )[:, :, 0]
        return f, fxm, fym, fxw, fyw, fxp, fyp

    step_plain = route_step(_step_plain, compile_mode, name=f"square_re100_w8b[D{D}]plain")
    step_seeded = route_step(
        _step_seeded, compile_mode, name=f"square_re100_w8b[D{D}]seed", quiet=True
    )

    omega_seed = 2.0 * math.pi * ST_SEED * u_in / D
    seed_steps = int(round(SEED_PERIODS / (ST_SEED * u_in / D)))
    uy_amp = SEED_AMPL * u_in

    fx_mem_h: list[torch.Tensor] = []
    fy_mem_h: list[torch.Tensor] = []
    fx_wet_h: list[torch.Tensor] = []
    fy_wet_h: list[torch.Tensor] = []
    fx_pair_h: list[torch.Tensor] = []
    fy_pair_h: list[torch.Tensor] = []
    t0 = time.time()
    for step in range(1, steps + 1):
        if step <= seed_steps:
            uy_val = uy_amp * math.sin(omega_seed * step)
            uy_col = torch.full((ny,), uy_val, device=device, dtype=f.dtype)
            f, fxm, fym, fxw, fyw, fxp, fyp = step_seeded(f, uy_col)
        else:
            f, fxm, fym, fxw, fyw, fxp, fyp = step_plain(f)
        fx_mem_h.append(fxm)
        fy_mem_h.append(fym)
        fx_wet_h.append(fxw)
        fy_wet_h.append(fyw)
        fx_pair_h.append(fxp)
        fy_pair_h.append(fyp)
    elapsed = time.time() - t0
    if not bool(torch.isfinite(f).all().item()):
        raise RuntimeError(f"D={D}: non-finite populations after {steps} steps")

    def _hist(hx, hy):
        return (
            torch.stack(hx).detach().cpu().numpy().astype(np.float64),
            torch.stack(hy).detach().cpu().numpy().astype(np.float64),
        )

    fxm_np, fym_np = _hist(fx_mem_h, fy_mem_h)
    fxw_np, fyw_np = _hist(fx_wet_h, fy_wet_h)
    fxp_np, fyp_np = _hist(fx_pair_h, fy_pair_h)

    q_dyn = 0.5 * 1.0 * u_in * u_in * D
    w0 = int(warmup_frac * steps)
    cd_mem, cl_mem = fxm_np / q_dyn, fym_np / q_dyn
    cd_wet, cl_wet = fxw_np / q_dyn, fyw_np / q_dyn
    cd_pair, cl_pair = fxp_np / q_dyn, fyp_np / q_dyn

    stats_mem = window_stats(cd_mem, cl_mem, w0, D, u_in)
    stats_wet = window_stats(cd_wet, cl_wet, w0, D, u_in)
    stats_pair = window_stats(cd_pair, cl_pair, w0, D, u_in)

    def _errs(s):
        ecd = (s["cd"] - REF_CD) / REF_CD * 100.0
        est = (s["st"] - REF_ST) / REF_ST * 100.0 if s["st"] is not None else None
        return round(ecd, 2), (round(est, 2) if est is not None else None)

    err_wet = _errs(stats_wet)
    err_mem = _errs(stats_mem)
    err_pair = _errs(stats_pair)
    ab_cd_pct = (stats_wet["cd"] - stats_mem["cd"]) / stats_mem["cd"] * 100.0
    ab_st_pct = (
        (stats_wet["st"] - stats_mem["st"]) / stats_mem["st"] * 100.0
        if None not in (stats_wet["st"], stats_mem["st"])
        else None
    )
    closure_wet_pct = (stats_pair["cd"] - stats_wet["cd"]) / stats_wet["cd"] * 100.0
    closure_mem_pct = (stats_pair["cd"] - stats_mem["cd"]) / stats_mem["cd"] * 100.0

    repro = None
    if D in OLD_ARCH:
        old_cd, old_st = OLD_ARCH[D]
        repro = {
            "old_cd": old_cd,
            "old_st": old_st,
            "mem_cd_delta_pct": round((stats_mem["cd"] - old_cd) / old_cd * 100.0, 3),
            "mem_st_delta_pct": round((stats_mem["st"] - old_st) / old_st * 100.0, 3),
            "gate_abs_0.5pct": bool(
                abs((stats_mem["cd"] - old_cd) / old_cd) <= 0.005
                and abs((stats_mem["st"] - old_st) / old_st) <= 0.005
            ),
        }

    def _round6(arr):
        return [round(float(v), 6) for v in arr]

    result = {
        "case": "square_cylinder_re100_w8b_dual_force",
        "baseline_commit": "36224617f19 (bm_w8 worktree, read-only)",
        "prereg": "sha256 b9928c156538aae1b56fc7c902f7d68950fd6a999ae9e345a687e7d04f65ea45, frozen 2026-09-28T11:24:03Z",
        "D": D,
        "nx": nx,
        "ny": ny,
        "re": re,
        "u_in": u_in,
        "nu_lb": nu,
        "tau": round(tau, 6),
        "sponge": {
            "x0": int(nx - SPONGE_D * D),
            "width": int(SPONGE_D * D),
            "alpha": SPONGE_ALPHA,
            "power": 2.0,
        },
        "seed": {
            "st_seed": ST_SEED,
            "ampl_frac": SEED_AMPL,
            "periods": SEED_PERIODS,
            "steps": seed_steps,
        },
        "force_columns": {
            "wet": "surface-link MEM (fluid->solid links incl. corners), PRIMARY",
            "pair": "paired two-population MEM (momentum-budget closure test)",
            "mem": "full-solid Ladd MEM (compute_obstacle_forces), diagnostic",
            "n_links_wet": int(s2.sum().item()),
            "n_solid_cells": int(mask.sum().item()),
        },
        "steps": steps,
        "warmup_frac": warmup_frac,
        "analyze_from": w0,
        "n_blocks": N_BLOCKS,
        "compile_mode": compile_mode,
        "wet": {**stats_wet, "err_cd_pct": err_wet[0], "err_st_pct": err_wet[1]},
        "mem": {**stats_mem, "err_cd_pct": err_mem[0], "err_st_pct": err_mem[1]},
        "pair": {**stats_pair, "err_cd_pct": err_pair[0], "err_st_pct": err_pair[1]},
        "ab_diff": {
            "cd_wet_minus_mem": round(stats_wet["cd"] - stats_mem["cd"], 4),
            "cd_pct_of_mem": round(ab_cd_pct, 2),
            "st_pct_of_mem": round(ab_st_pct, 2) if ab_st_pct is not None else None,
        },
        "closure_pair_vs_wet_cd_pct": round(closure_wet_pct, 2),
        "closure_pair_vs_mem_cd_pct": round(closure_mem_pct, 2),
        "repro_vs_old_archive": repro,
        "finite": True,
        "elapsed_s": round(elapsed, 1),
        "traces": {
            "cd_mem": _round6(cd_mem),
            "cl_mem": _round6(cl_mem),
            "cd_wet": _round6(cd_wet),
            "cl_wet": _round6(cl_wet),
            "cd_pair": _round6(cd_pair),
            "cl_pair": _round6(cl_pair),
        },
    }
    print(
        f"[w8b D={D}] steps={steps} t={elapsed:.0f}s links={n_links} | "
        f"WET Cd={stats_wet['cd']:.4f} ({err_wet[0]:+.2f}%) St={stats_wet['st']:.4f} "
        f"({err_wet[1]:+.2f}%) cl_amp={stats_wet['cl_amp']:.4f} | "
        f"MEM Cd={stats_mem['cd']:.4f} ({err_mem[0]:+.2f}%) St={stats_mem['st']:.4f} "
        f"({err_mem[1]:+.2f}%) | A/B cd {ab_cd_pct:+.2f}% | "
        f"PAIR Cd={stats_pair['cd']:.4f} (vs wet {closure_wet_pct:+.2f}%, "
        f"vs mem {closure_mem_pct:+.2f}%)",
        flush=True,
    )
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="square cylinder Re=100 W8-B revival: dual-force (wet-node vs full-solid MEM)"
    )
    ap.add_argument("--D", type=int, nargs="+", default=[32, 48, 64])
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--u-in", type=float, default=0.05)
    ap.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="per-D step counts; default 60000/90000/120000 for D=32/48/64",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warmup-frac", type=float, default=0.5)
    ap.add_argument("--out", default="")
    ap.add_argument("--repo", default=_REPO_DEFAULT)
    add_compile_mode_arg(ap)
    args = ap.parse_args()

    _bootstrap(args.repo)

    device = torch.device(args.device)
    compile_mode = compile_mode_from_args(args)
    default_steps = {32: 60000, 48: 90000, 64: 120000}
    steps_list = args.steps or [default_steps.get(D, 120000) for D in args.D]

    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    grids = {}
    for D, steps in zip(args.D, steps_list):
        grids[str(D)] = run_case(
            D,
            args.re,
            args.u_in,
            steps,
            device,
            compile_mode=compile_mode,
            warmup_frac=args.warmup_frac,
            out_path=str(out / f"case_D{D}.json") if out else None,
        )

    def _series(col, key):
        return [grids[str(D)][col][key] for D in args.D if grids[str(D)][col][key] is not None]

    cds_pair = _series("pair", "cd")
    sts_pair = _series("pair", "st")
    cds_wet = _series("wet", "cd")
    sts_wet = _series("wet", "st")
    cds_mem = _series("mem", "cd")
    sts_mem = _series("mem", "st")

    def _monotone(vals, ref):
        errs = [abs(v - ref) / ref for v in vals]
        if len(errs) < 2:
            return None
        flat_floor = all(e <= 0.01 for e in errs)  # 各档 |err|<=1% 视为地板（prereg §2）
        return bool(all(errs[i + 1] <= errs[i] + 1e-12 for i in range(len(errs) - 1)) or flat_floor)

    finest = grids[str(args.D[-1])]
    summary = {
        "case": "square_cylinder_re100_w8b_dual_force",
        "reference_locked": {
            "main": f"2D low-blockage numerical cluster median: Cd_ref={REF_CD}, St_ref={REF_ST}",
            "prereg_sha256_initial": "b9928c156538aae1b56fc7c902f7d68950fd6a999ae9e345a687e7d04f65ea45",
            "prereg_sha256_amended": "9fbc6d4ac59396a14fb55784c10caf8cba9ac771f7ad0820da417b6c50a31aad",
            "amendment": "primary force column wet->pair (momentum-budget closure, prereg §5)",
            "sensitivity": SENS,
        },
        "grids": {k: {kk: vv for kk, vv in v.items() if kk != "traces"} for k, v in grids.items()},
        "verdict_prereg": {
            "primary_column": "pair (prereg §5 修订：动量预算闭合列；wet 为诊断列)",
            "cd_within_3pct": bool(abs(finest["pair"]["err_cd_pct"]) <= 3.0),
            "st_within_3pct": bool(abs(finest["pair"]["err_st_pct"]) <= 3.0),
            "cd_monotone": _monotone(cds_pair, REF_CD),
            "st_monotone": _monotone(sts_pair, REF_ST),
            "cds_pair": cds_pair,
            "sts_pair": sts_pair,
            "cds_wet_diagnostic": cds_wet,
            "sts_wet_diagnostic": sts_wet,
            "cds_mem": cds_mem,
            "sts_mem": sts_mem,
        },
        "sensitivity_err_pct_finest_pair": {
            k: {
                f"{kk}_err": round((finest["pair"][kk] - vv) / vv * 100.0, 2)
                for kk, vv in v.items()
                if kk in ("cd", "st")
            }
            for k, v in SENS.items()
        },
        "ab_summary": {str(D): grids[str(D)]["ab_diff"] for D in args.D},
        "repro_gate": {
            str(D): grids[str(D)]["repro_vs_old_archive"]
            for D in args.D
            if grids[str(D)]["repro_vs_old_archive"] is not None
        },
    }
    if out:
        (out / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["verdict_prereg"], indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
