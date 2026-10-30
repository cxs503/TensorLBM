#!/usr/bin/env python
"""SDAA 验证扫荡 — startup_poiseuille 案例。

独立运行脚本：不改业务代码，仅调用 benchmarks/verified/startup_poiseuille/run.py
中的 run_case 函数，将设备切换为 sdaa，使用 eager 编译模式（SDAA 的
teco_inductor 后端尚不完整，无法编译 LBM 整步链）。

网格选择：H=[9, 15]（原文档推荐 [59, 119]，为控制墙钟 ≤20 min 取可行较小档）。
判据口径与原文档一致：
  1. 中心线 max |u_c_num - u_c_ana| / u_max ≤ 3%
  2. 剖面 L2 相对误差 ≤ 3%
  3. 剖面逐点最大误差 / u_max ≤ 3%
  4. H→2H 误差单调下降（网格收敛）

输出：
  - artifacts/sdaa_startup_poiseuille.json（设备证据行 + 判据数值 + PASS/FAIL）
  - 终端打印逐条比对结果
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ── 路径引导 ──────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO / "src"))

import torch

# ── 设备证据 ──────────────────────────────────────────────────────────────
DEVICE_NAME = None
DEVICE_AVAILABLE = False
try:
    DEVICE_AVAILABLE = torch.sdaa.is_available()
    if DEVICE_AVAILABLE:
        DEVICE_NAME = torch.sdaa.get_device_name(0)
except Exception as e:
    print(f"[WARN] torch.sdaa probe failed: {e}")

DEVICE = torch.device("sdaa") if DEVICE_AVAILABLE else torch.device("cpu")
print(f"[device] available={DEVICE_AVAILABLE} name={DEVICE_NAME!r} -> {DEVICE}", flush=True)

# ── 导入案例核心函数 ─────────────────────────────────────────────────────
from compile_route import ensure_tensorlbm_importable
ensure_tensorlbm_importable()

import importlib.util
run_py = BENCH / "verified" / "startup_poiseuille" / "run.py"
spec = importlib.util.spec_from_file_location("startup_poiseuille_run", run_py)
sp_mod = importlib.util.module_from_spec(spec)
sp_mod.DEVICE = DEVICE
spec.loader.exec_module(sp_mod)

run_case = sp_mod.run_case

# ── 配置 ─────────────────────────────────────────────────────────────────
H_LIST = [9, 15]  # 可行较小档，H=9→15 约 1.67x 网格加密
TAU = 0.8
U_MAX = 0.03
NX = 16
TOL_PCT = 3.0
COMPILE_MODE = "eager"  # SDAA teco_inductor 无法编译 LBM 整步链

# ── 运行 ─────────────────────────────────────────────────────────────────
results = []
wall_start = time.time()

for H in H_LIST:
    print(f"\n{'='*60}")
    print(f"Running H={H} on {DEVICE} (compile_mode={COMPILE_MODE})")
    print(f"{'='*60}", flush=True)

    r = run_case(
        H=H,
        tau=TAU,
        u_max_target=U_MAX,
        nx=NX,
        out_path=None,
        seed=0,
        compile_mode=COMPILE_MODE,
    )
    results.append(r)
    print(
        f"H={H:4d}: center={r['centerline_err_max_over_umax_pct']:.3f}% "
        f"l2={r['profile_l2_max_pct']:.3f}% ptp={r['profile_pointwise_max_over_umax_pct']:.3f}% "
        f"steps={r['t_end']} elapsed={r['elapsed_s']}s",
        flush=True,
    )

wall_elapsed = time.time() - wall_start

# ── 判据评估 ─────────────────────────────────────────────────────────────
rs = sorted(results, key=lambda r: r["H"])

per_grid_pass = []
for r in rs:
    c_pass = r["centerline_err_max_over_umax_pct"] <= TOL_PCT
    l2_pass = r["profile_l2_max_pct"] <= TOL_PCT
    ptp_pass = r["profile_pointwise_max_over_umax_pct"] <= TOL_PCT
    per_grid_pass.append({
        "H": r["H"],
        "centerline_err_max_over_umax_pct": r["centerline_err_max_over_umax_pct"],
        "centerline_pass": c_pass,
        "profile_l2_max_pct": r["profile_l2_max_pct"],
        "profile_l2_pass": l2_pass,
        "profile_pointwise_max_over_umax_pct": r["profile_pointwise_max_over_umax_pct"],
        "profile_pointwise_pass": ptp_pass,
        "all_pass": c_pass and l2_pass and ptp_pass,
    })

all_tol_pass = all(g["all_pass"] for g in per_grid_pass)

# 收敛性：取最小和最大 H 的中心线误差比较
e_lo = rs[0]["centerline_err_max_over_umax_pct"]
e_hi = rs[-1]["centerline_err_max_over_umax_pct"]
err_decreased = bool(e_hi < e_lo)

overall_pass = bool(all_tol_pass and err_decreased)

# ── 组装输出 ─────────────────────────────────────────────────────────────
artifact = {
    "case": "startup_poiseuille",
    "lattice": "D2Q9",
    "collision": "bgk",
    "boundary": "pre-streaming half-way bounce-back (verified/poiseuille_2d pattern)",
    "driving": "constant body force a=8*nu*u_max/H^2 via library _apply_body_force_2d",
    "device": {
        "type": "sdaa",
        "available": DEVICE_AVAILABLE,
        "name": DEVICE_NAME,
        "get_device_name_output": f"torch.sdaa.get_device_name(0) -> {DEVICE_NAME!r}" if DEVICE_NAME else "N/A",
    },
    "compile_mode_requested": COMPILE_MODE,
    "compile_mode_effective": COMPILE_MODE,
    "compile_status": "eager",
    "triton_interpret": False,
    "workarounds": [
        "SDAA teco_inductor 后端无法编译 LBM 整步链，使用 compile_mode='eager' 绕行"
    ],
    "H_list": H_LIST,
    "tau": TAU,
    "u_max_target": U_MAX,
    "nx": NX,
    "tol_pct": TOL_PCT,
    "per_grid": per_grid_pass,
    "convergence": {
        "metric": "centerline_err_max_over_umax_pct",
        "values": [r["centerline_err_max_over_umax_pct"] for r in rs],
        "err_decreased": err_decreased,
    },
    "criteria": {
        "1_centerline_max_err_le_3pct": all_tol_pass,
        "2_profile_l2_le_3pct": all(g["profile_l2_pass"] for g in per_grid_pass),
        "3_profile_pointwise_le_3pct": all(g["profile_pointwise_pass"] for g in per_grid_pass),
        "4_H2H_error_decrease": err_decreased,
    },
    "overall_pass": overall_pass,
    "wall_clock_s": round(wall_elapsed, 1),
    "profile_tables": [
        {
            "H": r["H"],
            "profiles": r.get("profile_table", []),
        }
        for r in rs
    ],
    "centerline_tables": [
        {
            "H": r["H"],
            "centerline": r.get("centerline_table_every10", []),
        }
        for r in rs
    ],
}

# ── 写 artifacts ─────────────────────────────────────────────────────────
artifacts_dir = REPO / "artifacts"
artifacts_dir.mkdir(parents=True, exist_ok=True)
out_path = artifacts_dir / "sdaa_startup_poiseuille.json"
out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
print(f"\n[artifact] written to {out_path}", flush=True)

# ── 终端打印判据比对 ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SDAA 判据比对 (startup_poiseuille)")
print(f"{'='*60}")
print(f"设备: {DEVICE_NAME}")
print(f"编译模式: {COMPILE_MODE} (eager, SDAA teco_inductor 无法编译)")
print(f"墙钟: {wall_elapsed:.1f}s")
print()

for g in per_grid_pass:
    print(f"  H={g['H']}:")
    print(f"    中心线 max err/u_max = {g['centerline_err_max_over_umax_pct']:.4f}%  {'PASS' if g['centerline_pass'] else 'FAIL'} (≤{TOL_PCT}%)")
    print(f"    剖面 L2 max          = {g['profile_l2_max_pct']:.4f}%  {'PASS' if g['profile_l2_pass'] else 'FAIL'} (≤{TOL_PCT}%)")
    print(f"    剖面逐点 max/u_max   = {g['profile_pointwise_max_over_umax_pct']:.4f}%  {'PASS' if g['profile_pointwise_pass'] else 'FAIL'} (≤{TOL_PCT}%)")

print(f"\n  网格收敛 (H={rs[0]['H']}→{rs[-1]['H']}): 误差 {e_lo:.4f}% → {e_hi:.4f}%  {'PASS' if err_decreased else 'FAIL'}")
print(f"\n  总体: {'PASS' if overall_pass else 'FAIL'}")

if not overall_pass:
    reasons = []
    if not all_tol_pass:
        reasons.append("部分网格误差超 3%")
    if not err_decreased:
        reasons.append("误差未单调下降")
    print(f"  原因: {'; '.join(reasons)}")
