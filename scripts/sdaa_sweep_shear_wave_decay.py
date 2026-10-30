#!/usr/bin/env python
"""SDAA 验证扫荡：shear_wave_decay (B30) — 2D 衰减剪切波基准。

独立运行脚本，不改业务代码。计算张量放 'sdaa' 设备。
若入口依赖 triton，用 TRITON_INTERPRET=1 并如实记录绕行。
网格/步数按案例文档推荐值（H=64/128 两档）。

判据（来自 benchmarks/verified/shear_wave_decay/README.md）：
  1. |γ_vel_sim/γ_vel_theory − 1| ≤ 3%
  2. |γ_E_sim/γ_E_theory − 1| ≤ 3%
  3. 网格收敛：|err| 随 H 增大单调下降
  4. 指数衰减质量：R² ≥ 0.999
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── 确保 tensorlbm 可导入 ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

# ── Triton 绕行记录 ────────────────────────────────────────────────────
TRITON_INTERPRET = os.environ.get("TRITON_INTERPRET", "0")
triton_workaround_note = ""
if TRITON_INTERPRET != "1":
    try:
        import triton  # noqa: F401
    except ImportError:
        os.environ["TRITON_INTERPRET"] = "1"
        TRITON_INTERPRET = "1"
        triton_workaround_note = "TRITON_INTERPRET=1 (triton import failed, set interpret mode)"

# ── 设备检测 ───────────────────────────────────────────────────────────
DEVICE = "sdaa"
device_name = ""
device_evidence = ""

try:
    if hasattr(torch, "sdaa"):
        cnt = torch.sdaa.device_count()
        device_name = torch.sdaa.get_device_name(0) if cnt > 0 else "no sdaa device"
        device_evidence = f"torch.sdaa.get_device_name(0) = '{device_name}' (device_count={cnt})"
    else:
        device_evidence = "torch.sdaa not available"
except Exception as e:
    device_evidence = f"torch.sdaa error: {e}"

print(f"[sdaa_sweep] Device evidence: {device_evidence}", flush=True)

# ── 案例参数（与 README 一致）──────────────────────────────────────────
CANONICAL = dict(tau=0.8, u0=0.05, dtype="float32")
GRIDS = [64, 128]
RECORD_EVERY = 100


def to_native(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def auto_steps(n: int, tau: float, record_every: int, cap: int = 10000, min_steps: int = 2000) -> int:
    """按速度振幅目标衰减 u_max/U0 → e^-3 自动确定步数。"""
    nu = (tau - 0.5) / 3.0
    k = 2.0 * math.pi / n
    gamma = nu * k * k
    steps = max(min_steps, int(math.ceil(3.0 / gamma / record_every) * record_every))
    return min(steps, cap)


def run_case(
    n: int,
    tau: float,
    u0: float,
    steps: int,
    record_every: int = 100,
    device: str = "sdaa",
    dtype: str = "float32",
) -> dict:
    """运行单档网格的剪切波衰减模拟。"""
    dt = torch.float64 if dtype == "float64" else torch.float32

    L = float(n)
    k = 2.0 * math.pi / L
    nu = (tau - 0.5) / 3.0
    gamma_vel_theory = nu * k * k
    gamma_e_theory = 2.0 * nu * k * k

    # 初始场
    y, x = torch.meshgrid(
        torch.arange(n, device=device, dtype=dt),
        torch.arange(n, device=device, dtype=dt),
        indexing="ij",
    )
    ux0 = u0 * torch.sin(k * y)
    uy0 = torch.zeros_like(ux0)
    rho = torch.ones_like(ux0)
    f = equilibrium(rho, ux0, uy0)

    mass0 = float(f.sum().item())

    # 主循环：eager 模式（SDAA 上 torch.compile 已知会 fallback）
    def _step(f):
        return collide_bgk(stream(f), tau)

    times: list[int] = []
    energies: list[float] = []
    umaxs: list[float] = []
    wall0 = time.time()
    for step in range(1, steps + 1):
        f = _step(f)
        if step % record_every == 0:
            _, uxm, uym = macroscopic(f)
            e = float((0.5 * (uxm * uxm + uym * uym)).mean().item())
            um = float((uxm * uxm + uym * uym).sqrt().max().item())
            times.append(step)
            energies.append(e)
            umaxs.append(um)
    wall = time.time() - wall0

    # 拟合
    t = np.asarray(times, dtype=np.float64)
    lnE = np.log(np.asarray(energies, dtype=np.float64))
    a_e, b_e = np.polyfit(t, lnE, 1)
    gamma_e_sim = -a_e
    resid_e = lnE - (b_e + a_e * t)
    r2_e = 1.0 - float(np.sum(resid_e**2) / np.sum((lnE - lnE.mean()) ** 2))

    lnU = np.log(np.asarray(umaxs, dtype=np.float64))
    a_u, b_u = np.polyfit(t, lnU, 1)
    gamma_vel_sim = -a_u
    resid_u = lnU - (b_u + a_u * t)
    r2_u = 1.0 - float(np.sum(resid_u**2) / np.sum((lnU - lnU.mean()) ** 2))

    # 半窗口一致性
    half = len(t) // 2
    gv_h1 = -float(np.polyfit(t[:half], lnU[:half], 1)[0])
    gv_h2 = -float(np.polyfit(t[half:], lnU[half:], 1)[0])

    err_vel_pct = (gamma_vel_sim - gamma_vel_theory) / gamma_vel_theory * 100.0
    err_e_pct = (gamma_e_sim - gamma_e_theory) / gamma_e_theory * 100.0
    mass_end = float(f.sum().item())

    return {
        "H": n,
        "tau": tau,
        "u0": u0,
        "nu": nu,
        "k": k,
        "steps": steps,
        "record_every": record_every,
        "dtype": dtype,
        "device": device,
        "gamma_vel_theory": gamma_vel_theory,
        "gamma_vel_sim": gamma_vel_sim,
        "err_vel_pct": err_vel_pct,
        "r2_vel": r2_u,
        "gamma_vel_half1": gv_h1,
        "gamma_vel_half2": gv_h2,
        "gamma_e_theory": gamma_e_theory,
        "gamma_e_sim": gamma_e_sim,
        "err_e_pct": err_e_pct,
        "r2_e": r2_e,
        "mass_drift_rel": (mass_end - mass0) / mass0,
        "n_samples": len(times),
        "wall_sec": wall,
    }


def judge(cases: list[dict], tol_pct: float = 3.0) -> dict:
    """两档网格收敛判定。"""
    by_h = {c["H"]: c for c in cases}
    hs = sorted(by_h)
    errs = [by_h[h]["err_vel_pct"] for h in hs]
    errs_e = [by_h[h]["err_e_pct"] for h in hs]
    r2s = [by_h[h]["r2_vel"] for h in hs]
    converged = all(abs(errs[i + 1]) < abs(errs[i]) for i in range(len(errs) - 1))
    within = all(abs(e) <= tol_pct for e in errs) and all(abs(e) <= tol_pct for e in errs_e)
    expo = all(r2 >= 0.999 for r2 in r2s)
    verified = bool(within and converged and expo and len(hs) >= 2)
    return {
        "grids": hs,
        "err_vel_pct_per_grid": {str(h): errs[i] for i, h in enumerate(hs)},
        "err_e_pct_per_grid": {str(h): errs_e[i] for i, h in enumerate(hs)},
        "r2_vel_per_grid": {str(h): r2s[i] for i, h in enumerate(hs)},
        "converged_monotone": converged,
        "within_tol": within,
        "exponential_r2_ok": expo,
        "tol_pct": tol_pct,
        "verified": verified,
    }


def main() -> None:
    print(f"[sdaa_sweep] shear_wave_decay SDAA sweep starting", flush=True)
    print(f"[sdaa_sweep] TRITON_INTERPRET={TRITON_INTERPRET}", flush=True)
    if triton_workaround_note:
        print(f"[sdaa_sweep] Triton workaround: {triton_workaround_note}", flush=True)

    results = []
    total_wall = 0.0
    for n in GRIDS:
        steps = auto_steps(n, CANONICAL["tau"], RECORD_EVERY)
        print(f"\n=== H={n}  tau={CANONICAL['tau']}  U0={CANONICAL['u0']}  steps={steps} ===", flush=True)
        res = run_case(
            n,
            CANONICAL["tau"],
            CANONICAL["u0"],
            steps,
            record_every=RECORD_EVERY,
            device=DEVICE,
            dtype=CANONICAL["dtype"],
        )
        results.append(res)
        total_wall += res["wall_sec"]
        print(
            f"  gamma_vel_sim={res['gamma_vel_sim']:.6e} theory={res['gamma_vel_theory']:.6e} "
            f"err_vel={res['err_vel_pct']:+.4f}%  err_E={res['err_e_pct']:+.4f}%  "
            f"R2={res['r2_vel']:.6f}  wall={res['wall_sec']:.1f}s",
            flush=True,
        )

    verdict = judge(results)

    # ── 逐条判据 PASS/FAIL ─────────────────────────────────────────────
    criteria = []

    # 判据 1: 速度衰减率偏差 ≤ 3%
    for h in verdict["grids"]:
        err = verdict["err_vel_pct_per_grid"][str(h)]
        passed = bool(abs(err) <= verdict["tol_pct"])
        criteria.append({
            "name": f"速度衰减率偏差 H={h}",
            "value": f"{err:+.4f}%",
            "threshold": f"≤ {verdict['tol_pct']}%",
            "pass": passed,
        })

    # 判据 2: 动能衰减率偏差 ≤ 3%
    for h in verdict["grids"]:
        err = verdict["err_e_pct_per_grid"][str(h)]
        passed = bool(abs(err) <= verdict["tol_pct"])
        criteria.append({
            "name": f"动能衰减率偏差 H={h}",
            "value": f"{err:+.4f}%",
            "threshold": f"≤ {verdict['tol_pct']}%",
            "pass": passed,
        })

    # 判据 3: 网格收敛
    criteria.append({
        "name": "网格收敛（|err| 单调下降）",
        "value": str(bool(verdict["converged_monotone"])),
        "threshold": "True",
        "pass": bool(verdict["converged_monotone"]),
    })

    # 判据 4: 指数衰减质量 R² ≥ 0.999
    for h in verdict["grids"]:
        r2 = verdict["r2_vel_per_grid"][str(h)]
        passed = bool(r2 >= 0.999)
        criteria.append({
            "name": f"指数衰减 R² H={h}",
            "value": f"{r2:.6f}",
            "threshold": "≥ 0.999",
            "pass": passed,
        })

    all_pass = bool(all(c["pass"] for c in criteria))

    # ── 构建输出 JSON ──────────────────────────────────────────────────
    output = {
        "benchmark": "shear_wave_decay",
        "description": "2D decaying shear wave: u=U0·sin(2πy/H)·e^{-νk²t}, viscous dissipation check",
        "device": DEVICE,
        "device_evidence": device_evidence,
        "device_name": device_name,
        "triton_interpret": TRITON_INTERPRET,
        "triton_workaround": triton_workaround_note if triton_workaround_note else "none",
        "compile_mode": "eager (SDAA torch.compile known to fallback)",
        "lattice": "D2Q9",
        "collision": "bgk",
        "boundary": "periodic",
        "formula": "gamma_vel_theory = nu·k², nu=(tau-1/2)/3, k=2π/H; gamma_e_theory = 2·gamma_vel_theory",
        "cases": results,
        "convergence": verdict,
        "criteria": criteria,
        "all_criteria_pass": all_pass,
        "total_wall_sec": total_wall,
        "verdict": "PASS" if all_pass else "FAIL",
    }

    # 写入 artifacts
    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "sdaa_shear_wave_decay.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False, default=to_native)

    print(f"\n{'='*60}", flush=True)
    print(f"[sdaa_sweep] Results written to {out_path}", flush=True)
    print(f"[sdaa_sweep] Verdict: {output['verdict']}", flush=True)
    print(f"[sdaa_sweep] Total wall time: {total_wall:.1f}s", flush=True)
    for c in criteria:
        status = "✅ PASS" if c["pass"] else "❌ FAIL"
        print(f"  {status} {c['name']}: {c['value']} (threshold: {c['threshold']})", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
