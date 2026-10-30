#!/usr/bin/env python3
"""SDAA 验证扫荡：方腔流 cavity（lid-driven cavity, Ghia 1982）。

独立运行脚本——不改业务代码，仅调用 benchmarks/verified/cavity/re100/run.py
的入口逻辑，将张量放到 'sdaa' 设备上运行。

判据（与 NVIDIA/原口径一致）：
  1. max_abs_dev_pct <= 3.0%（两档网格均满足）
  2. err_decreased（大网格误差 < 小网格误差，单调收敛）
  3. 两者同时满足 → PASS，否则 FAIL

网格/步数：为适配 SDAA 墙钟 ≤20 分钟，采用 32²/48²，10k 步。
若入口依赖 triton，设置 TRITON_INTERPRET=1 并如实记录绕行。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 0. 环境准备：repo src 入 sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
_BENCH_DIR = _REPO_ROOT / "benchmarks"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

# ---------------------------------------------------------------------------
# 1. 设备证据
# ---------------------------------------------------------------------------
device_name = "unknown"
device_available = False
try:
    if hasattr(torch, "sdaa") and torch.sdaa.device_count() > 0:
        device_name = torch.sdaa.get_device_name(0)
        device_available = True
except Exception as e:
    device_name = f"error: {e}"

# ---------------------------------------------------------------------------
# 2. Triton 绕行记录
# ---------------------------------------------------------------------------
triton_interpret_orig = os.environ.get("TRITON_INTERPRET", "0")
triton_workaround = False
try:
    import triton  # noqa: F401
    if triton_interpret_orig != "1":
        os.environ["TRITON_INTERPRET"] = "1"
        triton_workaround = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 3. 导入 cavity 运行逻辑
# ---------------------------------------------------------------------------
from compile_route import route_step  # noqa: E402
from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic  # noqa: E402
from tensorlbm.lid_driven_cavity import GHIA_RE100, zou_he_moving_lid  # noqa: E402
from tensorlbm.solver import collide_mrt, stream  # noqa: E402


def stationary_pre_bounce(f_pre, f, wall):
    """pre-streaming 半程反弹（三静止壁，V3 关键修复）。"""
    opp = OPPOSITE.to(f.device)
    return torch.where(wall.unsqueeze(0), f_pre[opp], f)


def run_case(nx, re, u_lid, steps, device, compile_mode="eager"):
    """运行单档网格的 cavity 模拟，返回结果字典。"""
    ny = nx
    tau = 3.0 * u_lid * nx / re + 0.5
    rho0 = torch.ones((ny, nx), device=device)
    u0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, u0, u0)

    wall_mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall_mask[0, :] = True
    wall_mask[:, 0] = True
    wall_mask[:, -1] = True

    interior = ~wall_mask
    resid_mask = interior.clone()
    resid_mask[-1, :] = False

    def _step(f):
        f_pre = f
        f = collide_mrt(f, tau=tau)
        f = stationary_pre_bounce(f_pre, f, wall_mask)
        f = stream(f)
        return zou_he_moving_lid(f, u_lid)

    step_fn = route_step(_step, compile_mode, name=f"cavity_sdaa[{nx}]")

    _, ux_prev, uy_prev = macroscopic(f)
    ux_prev = ux_prev.detach().clone()
    uy_prev = uy_prev.detach().clone()
    last_resid = None

    t0 = time.time()
    for step in range(1, steps + 1):
        f = step_fn(f)
        if step % 5000 == 0 or step == steps:
            _, ux, uy = macroscopic(f)
            du = (
                torch.max(
                    torch.abs(ux[resid_mask] - ux_prev[resid_mask]),
                    torch.abs(uy[resid_mask] - uy_prev[resid_mask]),
                )
                .max()
                .item()
            )
            last_resid = du
            ux_prev = ux.detach().clone()
            uy_prev = uy_prev.detach().clone()
            print(f"  step {step}/{steps}, resid={du:.2e}", flush=True)
    elapsed = time.time() - t0

    # ---- 度量 ----
    rho, ux, uy = macroscopic(f)
    ux_w = ux.masked_fill(wall_mask, 0.0)
    uy_w = uy.masked_fill(wall_mask, 0.0)
    ux_np = ux_w.detach().cpu().numpy() / u_lid
    uy_np = uy_w.detach().cpu().numpy() / u_lid

    x_mid, y_mid = nx // 2, ny // 2
    y_pos = np.linspace(0.0, 1.0, ny)
    x_pos = np.linspace(0.0, 1.0, nx)
    u_cl = ux_np[:, x_mid]
    v_cl = uy_np[y_mid, :]
    u_gi = np.interp(GHIA_RE100["y"], y_pos, u_cl)
    v_gi = np.interp(GHIA_RE100["x"], x_pos, v_cl)
    rmse_u = float(np.sqrt(np.mean((u_gi - np.array(GHIA_RE100["u"])) ** 2)))
    rmse_v = float(np.sqrt(np.mean((v_gi - np.array(GHIA_RE100["v"])) ** 2)))
    dev = np.concatenate(
        [np.abs(u_gi - np.array(GHIA_RE100["u"])), np.abs(v_gi - np.array(GHIA_RE100["v"]))]
    )
    max_abs_dev_pct = 100.0 * float(dev.max())

    u_mid = float(np.interp(0.5, y_pos, u_cl))
    u_bot = float(np.interp(0.0625, y_pos, u_cl))
    v_mid = float(np.interp(0.5, x_pos, v_cl))

    # 主涡心
    speed2 = ux_np**2 + uy_np**2
    speed2[: int(0.30 * ny), :] = np.inf
    speed2[int(0.90 * ny) :, :] = np.inf
    speed2[:, : int(0.30 * nx)] = np.inf
    speed2[:, int(0.90 * nx) :] = np.inf
    iy0, ix0 = np.unravel_index(np.argmin(speed2), speed2.shape)

    def refine(axis, i):
        a = speed2[i - 1, ix0] if axis == 0 else speed2[iy0, i - 1]
        b = speed2[i, ix0] if axis == 0 else speed2[iy0, i]
        c = speed2[i + 1, ix0] if axis == 0 else speed2[iy0, i + 1]
        denom = a - 2 * b + c
        return i + 0.5 * (a - c) / denom if abs(denom) > 1e-30 else i

    vx = refine(1, ix0) / (nx - 1)
    vy = refine(0, iy0) / (ny - 1)

    print(
        f"[cavity_sdaa nx={nx}] steps={steps} t={elapsed:.0f}s resid={last_resid:.2e} "
        f"u(0.5,0.5)={u_mid:+.4f} u_bot={u_bot:+.4f} v(0.5,0.5)={v_mid:+.4f} "
        f"vortex=({vx:.3f},{vy:.3f}) rmse_u={rmse_u:.4f} rmse_v={rmse_v:.4f} "
        f"max_abs_dev={max_abs_dev_pct:.2f}%",
        flush=True,
    )

    return {
        "nx": nx,
        "re": re,
        "u_lid": u_lid,
        "tau": round(tau, 4),
        "steps": steps,
        "compile_mode": compile_mode,
        "elapsed_s": round(elapsed, 1),
        "last_resid": last_resid,
        "u_mid": u_mid,
        "u_mid_ghia": -0.20581,
        "u_bot": u_bot,
        "v_mid": v_mid,
        "vortex": [round(vx, 4), round(vy, 4)],
        "vortex_ghia": [0.6172, 0.7344],
        "rmse_u": rmse_u,
        "rmse_v": rmse_v,
        "max_abs_dev_pct": max_abs_dev_pct,
    }


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------
def main():
    if not device_available:
        print("ERROR: SDAA device not available", flush=True)
        artifact = {
            "case": "cavity_re100_ghia_sdaa",
            "device_evidence": device_name,
            "device_available": False,
            "error": "SDAA device not available",
            "verdict": {"pass": False, "reason": "SDAA device not available"},
        }
        out_path = _REPO_ROOT / "artifacts" / "sdaa_cavity.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Artifact written to {out_path}", flush=True)
        sys.exit(1)

    # 配置：32²/48²，10k 步（适配 SDAA 墙钟 ≤20 分钟）
    re = 100.0
    u_lid = 0.06
    steps = 10000
    grids = [32, 48]
    compile_mode = "eager"

    device = torch.device("sdaa:0")
    print(f"Running on device: {device}", flush=True)
    print(f"Device name: {device_name}", flush=True)
    print(f"Grids: {grids}, steps: {steps}, re: {re}, u_lid: {u_lid}", flush=True)
    print(f"Compile mode: {compile_mode}", flush=True)
    if triton_workaround:
        print(f"TRITON_INTERPRET workaround applied (was {triton_interpret_orig})", flush=True)

    results = {}
    total_start = time.time()
    for nx in grids:
        print(f"\n--- Running nx={nx} ---", flush=True)
        results[str(nx)] = run_case(nx, re, u_lid, steps, device, compile_mode=compile_mode)

    total_elapsed = time.time() - total_start

    # ---- 判据 ----
    devs = [results[str(nx)]["max_abs_dev_pct"] for nx in grids]
    err_decreased = devs[-1] < devs[0]
    all_under_3 = all(d <= 3.0 for d in devs)
    verdict_pass = all_under_3 and err_decreased

    verdict = {
        "max_abs_dev_pct": devs,
        "err_decreased": err_decreased,
        "all_under_3pct": all_under_3,
        "pass": verdict_pass,
    }

    # ---- 逐条判据比对 ----
    criteria = []
    criteria.append({
        "name": "max_abs_dev_pct (32²) <= 3%",
        "value": devs[0],
        "threshold": 3.0,
        "pass": devs[0] <= 3.0,
    })
    criteria.append({
        "name": "max_abs_dev_pct (48²) <= 3%",
        "value": devs[1],
        "threshold": 3.0,
        "pass": devs[1] <= 3.0,
    })
    criteria.append({
        "name": "err_decreased (48² < 32²)",
        "value": err_decreased,
        "threshold": True,
        "pass": err_decreased,
    })

    # ---- 组装 artifact ----
    artifact = {
        "case": "cavity_re100_ghia_sdaa",
        "description": "方腔流 Re=100，D2Q9 MRT，Ghia(1982) 验证，SDAA 设备运行",
        "lattice": "D2Q9",
        "collision": "mrt",
        "boundary": "V3: pre-streaming 半程反弹(三静止壁) + zou_he_moving_lid(顶盖)",
        "reference": "Ghia 1982 Re=100 中线表值 + 涡心 (0.6172, 0.7344)",
        "device_evidence": device_name,
        "device_available": device_available,
        "triton_interpret_workaround": triton_workaround,
        "triton_interpret_env": os.environ.get("TRITON_INTERPRET", "0"),
        "compile_mode": compile_mode,
        "grids": results,
        "convergence": verdict,
        "criteria": criteria,
        "total_elapsed_s": round(total_elapsed, 1),
        "verified": bool(verdict_pass),
    }

    out_path = _REPO_ROOT / "artifacts" / "sdaa_cavity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nArtifact written to {out_path}", flush=True)
    print(f"Verdict: {json.dumps(verdict, ensure_ascii=False)}", flush=True)
    print(f"Total elapsed: {total_elapsed:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
