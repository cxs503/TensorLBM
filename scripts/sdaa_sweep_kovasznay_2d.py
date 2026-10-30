#!/usr/bin/env python
"""SDAA 验证案例扫荡 — kovasznay_2d（B18）。

独立运行脚本：不改业务代码，仅调用 benchmarks/verified/kovasznay_2d/run.py 的 run_case。
- 设备：sdaa（torch.sdaa）
- 编译模式：eager（SDAA 上 torch.compile/teco_inductor 已知不完整，compile_route 有自动回退）
- 网格：ny=64, 128（验收档）
- 判据：同原案例（max err <= 3%，64→128 单调收敛，finite，min_steps_ok）
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

# 确保 tensorlbm 可导入
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 确保 benchmarks 目录在 path 中（用于导入 compile_route 和 run）
BENCH_DIR = REPO_ROOT / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import numpy as np
import torch

# ---- 设备证据 ----
def get_device_evidence() -> dict:
    """获取 SDAA 设备证据。"""
    evidence = {
        "device_type": "sdaa",
        "sdaa_available": False,
        "device_name": None,
        "device_count": 0,
        "torch_version": torch.__version__,
    }
    if hasattr(torch, "sdaa"):
        evidence["sdaa_available"] = torch.sdaa.is_available()
        if evidence["sdaa_available"]:
            evidence["device_count"] = torch.sdaa.device_count()
            try:
                evidence["device_name"] = torch.sdaa.get_device_name(0)
            except Exception as e:
                evidence["device_name"] = f"error: {e}"
    return evidence


# ---- 从 run.py 复制核心函数（不改业务代码，仅复制必要部分） ----
# 注意：我们直接 import run 模块来复用 run_case，而不是复制代码。
# 但 run.py 的 scan 函数会写文件到磁盘，我们直接调用 run_case 更干净。

CS2 = 1.0 / 3.0
TWOPI = 2.0 * math.pi


def kov_lambda(re: float) -> float:
    """Kovasznay 衰减率 λ（负根，精确公式）。"""
    return re / 2.0 - math.sqrt(re * re / 4.0 + 4.0 * math.pi * math.pi)


def analytic_field(
    nx: int,
    ny: int,
    u0: float,
    lam: float,
    xmax: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """解析 u/v 场。"""
    from tensorlbm.d2q9 import equilibrium

    y = torch.arange(ny, device=device, dtype=dtype)
    x = torch.arange(nx, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xp = xx / nx * xmax
    yp = yy / ny
    e = torch.exp(lam * xp)
    u = u0 * (1.0 - e * torch.cos(TWOPI * yp))
    v = u0 * (lam / TWOPI) * e * torch.sin(TWOPI * yp)
    return u, v


def outlet_zero_gradient(f: torch.Tensor) -> torch.Tensor:
    """零梯度出口。"""
    f = f.clone()
    f[:, :, -1] = f[:, :, -2]
    return f


def outlet_analytic_dirichlet(
    f: torch.Tensor, u_ana_col: torch.Tensor, v_ana_col: torch.Tensor
) -> torch.Tensor:
    """解析 Dirichlet 出口。"""
    from tensorlbm.d2q9 import equilibrium

    ny = u_ana_col.shape[0]
    rho_c = torch.ones((ny, 1), device=f.device, dtype=f.dtype)
    f = f.clone()
    f[:, :, -1] = equilibrium(rho_c, u_ana_col.view(ny, 1), v_ana_col.view(ny, 1))[:, :, 0]
    return f


def run_case_sdaa(
    ny: int,
    re: float = 40.0,
    u0: float = 0.03,
    xmax: float = 3.0,
    outlet: str = "zerograd",
    min_steps: int = 20000,
    max_steps: int = 60000,
    device: str = "sdaa",
    compile_mode: str = "eager",
) -> dict:
    """在 SDAA 设备上运行单个 Kovasznay 2D 案例。"""
    from tensorlbm.boundaries import zou_he_inlet_velocity
    from tensorlbm.d2q9 import equilibrium, macroscopic
    from tensorlbm.solver import collide_bgk, stream

    dev = torch.device(device)
    dt = torch.float32
    nx = int(round(ny * xmax))

    lam = kov_lambda(re)
    nu = u0 * ny / re
    tau = 0.5 + 3.0 * nu
    ma_max = 2.0 * u0 / math.sqrt(CS2)

    u_ana, v_ana = analytic_field(nx, ny, u0, lam, xmax, dev, dt)
    f = equilibrium(torch.ones((ny, nx), device=dev, dtype=dt), u_ana, v_ana)
    initial_mass = float(f.sum().item())

    u_in = u_ana[:, 0].contiguous()
    v_in = v_ana[:, 0].contiguous()

    def _step(f):
        f = collide_bgk(f, tau)
        f = stream(f)
        if outlet == "zerograd":
            f = outlet_zero_gradient(f)
        else:
            f = outlet_analytic_dirichlet(f, u_ana[:, -1], v_ana[:, -1])
        return zou_he_inlet_velocity(f, u_in, v_in)

    # SDAA 上 torch.compile 不完整，用 eager 模式
    if compile_mode == "eager":
        step_fn = _step
    else:
        # 尝试用 compile_route 的路由（会自动回退到 eager）
        from compile_route import route_step
        step_fn = route_step(_step, compile_mode, name=f"kovasznay_2d_sdaa[ny{ny}]")

    t0 = time.time()
    l2_hist: list[tuple[int, torch.Tensor]] = []
    step = 0
    steady = False
    for step in range(1, max_steps + 1):
        f = step_fn(f)
        if step % 500 == 0:
            _, ux, _ = macroscopic(f)
            l2_hist.append((step, ux.double()))
            if step >= min_steps and len(l2_hist) >= 5:
                u_prev = l2_hist[-5][1]
                num = float(((ux.double() - u_prev) ** 2).sum().sqrt().item())
                den = float((ux.double() ** 2).sum().sqrt().item())
                if num / max(den, 1e-12) < 5e-5:
                    steady = True
                    break
    elapsed = time.time() - t0

    final_drift = 0.0
    if len(l2_hist) >= 5:
        u_prev = l2_hist[-5][1]
        num = float(((l2_hist[-1][1] - u_prev) ** 2).sum().sqrt().item())
        den = float((l2_hist[-1][1] ** 2).sum().sqrt().item())
        final_drift = num / max(den, 1e-12)

    # 末 100 步时间平均
    acc_u = torch.zeros((ny, nx), device=dev, dtype=torch.float64)
    acc_v = torch.zeros((ny, nx), device=dev, dtype=torch.float64)
    for _ in range(100):
        f = step_fn(f)
        _, ux, uy = macroscopic(f)
        acc_u += ux.to(torch.float64)
        acc_v += uy.to(torch.float64)
    acc_u /= 100.0
    acc_v /= 100.0

    u_num = acc_u.cpu().numpy()
    v_num = acc_v.cpu().numpy()
    u_a = u_ana.cpu().numpy().astype(np.float64)
    v_a = v_ana.cpu().numpy().astype(np.float64)

    def l2_rel(num: np.ndarray, ana: np.ndarray) -> float:
        return float(np.linalg.norm(num - ana) / np.linalg.norm(ana))

    def max_rel_masked(num: np.ndarray, ana: np.ndarray, thresh: float) -> float:
        mask = np.abs(ana) > thresh
        if mask.sum() == 0:
            return float("nan")
        return float(np.max(np.abs(num[mask] - ana[mask]) / np.abs(ana[mask])) * 100.0)

    u_l2 = l2_rel(u_num, u_a)
    v_l2 = l2_rel(v_num, v_a)
    u_max_rel = max_rel_masked(u_num, u_a, 0.1 * u0)
    v_max_rel = max_rel_masked(v_num, v_a, 0.1 * float(np.max(np.abs(v_a))))
    u_l2_int = l2_rel(u_num[:, 1:-1], u_a[:, 1:-1])
    v_l2_int = l2_rel(v_num[:, 1:-1], v_a[:, 1:-1])
    max_abs_u = float(np.max(np.abs(u_num - u_a)) / u0)
    max_abs_v = float(np.max(np.abs(v_num - v_a)) / u0)
    mass_drift_pct = (float(f.sum().item()) - initial_mass) / initial_mass * 100.0

    return {
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "nu_lb": nu,
        "re": re,
        "lambda": lam,
        "u0": u0,
        "ma_max": ma_max,
        "xmax": xmax,
        "outlet": outlet,
        "compile_mode": compile_mode,
        "n_steps": step,
        "steady": steady,
        "final_drift_rel": final_drift,
        "u_l2_rel": u_l2,
        "v_l2_rel": v_l2,
        "u_l2_rel_interior": u_l2_int,
        "v_l2_rel_interior": v_l2_int,
        "u_max_rel_pct": u_max_rel,
        "v_max_rel_pct": v_max_rel,
        "max_abs_err_u_over_u0": max_abs_u,
        "max_abs_err_v_over_u0": max_abs_v,
        "mass_drift_pct": mass_drift_pct,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    print("=" * 60)
    print("SDAA Sweep: Kovasznay 2D (B18)")
    print("=" * 60)

    # 设备证据
    device_evidence = get_device_evidence()
    print(f"\n设备证据:")
    for k, v in device_evidence.items():
        print(f"  {k}: {v}")

    if not device_evidence["sdaa_available"]:
        print("\n[ERROR] SDAA 不可用，无法继续。")
        # 仍然输出 JSON 记录失败
        result = {
            "case": "B18_kovasznay_2d_sdaa_sweep",
            "device_evidence": device_evidence,
            "error": "SDAA not available",
            "verdict": "FAIL",
        }
        Path("artifacts/sdaa_kovasznay_2d.json").write_text(json.dumps(result, indent=2))
        return

    # 配置
    grids = [64, 128]  # 验收网格
    re = 40.0
    u0 = 0.03
    xmax = 3.0
    outlet = "zerograd"
    min_steps = 20000
    max_steps = 60000
    compile_mode = "eager"  # SDAA 上 torch.compile 不完整

    print(f"\n配置: grids={grids}, re={re}, u0={u0}, xmax={xmax}, outlet={outlet}")
    print(f"编译模式: {compile_mode}（SDAA 绕行：teco_inductor 不完整）")
    print()

    # 运行每个网格
    cases = []
    wall_start = time.time()
    for ny in grids:
        print(f"\n--- 运行 ny={ny} ---")
        t_case = time.time()
        try:
            r = run_case_sdaa(
                ny=ny,
                re=re,
                u0=u0,
                xmax=xmax,
                outlet=outlet,
                min_steps=min_steps,
                max_steps=max_steps,
                device="sdaa",
                compile_mode=compile_mode,
            )
            cases.append(r)
            t_case_elapsed = time.time() - t_case
            print(
                f"  ny={r['ny']:3d} nx={r['nx']:3d} tau={r['tau']:.3f} "
                f"steps={r['n_steps']:6d} steady={r['steady']} "
                f"u_l2={r['u_l2_rel']:.5f} v_l2={r['v_l2_rel']:.5f} "
                f"u_mr={r['u_max_rel_pct']:.2f}% v_mr={r['v_max_rel_pct']:.2f}% "
                f"elapsed={r['elapsed_s']}s (wall={t_case_elapsed:.1f}s)"
            )
        except Exception as e:
            print(f"  [ERROR] ny={ny} 运行失败: {e}")
            cases.append({
                "ny": ny,
                "error": str(e),
                "finite": False,
                "n_steps": 0,
                "steady": False,
                "u_l2_rel": float("nan"),
                "v_l2_rel": float("nan"),
                "u_max_rel_pct": float("nan"),
                "v_max_rel_pct": float("nan"),
                "elapsed_s": round(time.time() - t_case, 1),
            })

    wall_elapsed = time.time() - wall_start
    print(f"\n总墙钟时间: {wall_elapsed:.1f}s")

    # 判据评估
    by_n = {r.get("ny"): r for r in cases if "ny" in r and "error" not in r}
    conv = {}
    verdict = "FAIL"
    verdict_reasons = []

    # 检查 finite
    all_finite = all(r.get("finite", False) for r in cases if "error" not in r)
    if not all_finite:
        verdict_reasons.append("not all finite")

    # 检查 min_steps
    min_steps_ok = all(r.get("n_steps", 0) >= min_steps for r in cases if "error" not in r)
    if not min_steps_ok:
        verdict_reasons.append("min_steps not met")

    # 收敛性检查（64→128）
    if 64 in by_n and 128 in by_n:
        c64, c128 = by_n[64], by_n[128]
        conv = {
            "u_l2_decreases": c128["u_l2_rel"] < c64["u_l2_rel"],
            "v_l2_decreases": c128["v_l2_rel"] < c64["v_l2_rel"],
            "u_max_rel_decreases": c128["u_max_rel_pct"] < c64["u_max_rel_pct"],
            "v_max_rel_decreases": c128["v_max_rel_pct"] < c64["v_max_rel_pct"],
        }
        errs_64 = max(
            c64["u_l2_rel"],
            c64["v_l2_rel"],
            c64["u_max_rel_pct"] / 100.0,
            c64["v_max_rel_pct"] / 100.0,
        )
        errs_128 = max(
            c128["u_l2_rel"],
            c128["v_l2_rel"],
            c128["u_max_rel_pct"] / 100.0,
            c128["v_max_rel_pct"] / 100.0,
        )
        max_err = max(errs_64, errs_128) * 100.0
        conv["max_err_pct"] = round(max_err, 4)
        conv["max_err_pct_64"] = round(errs_64 * 100.0, 4)
        conv["max_err_pct_128"] = round(errs_128 * 100.0, 4)

        conv_ok = all(conv.get(k, False) for k in ["u_l2_decreases", "v_l2_decreases",
                                                     "u_max_rel_decreases", "v_max_rel_decreases"])
        if not conv_ok:
            verdict_reasons.append("not convergent (64->128)")
        if max_err > 3.0:
            verdict_reasons.append(f"max_err {max_err:.4f}% > 3%")

        if all_finite and min_steps_ok and max_err <= 3.0 and conv_ok:
            verdict = "PASS"
    else:
        verdict_reasons.append("missing grids for convergence check")

    if verdict_reasons:
        verdict = "FAIL"

    # 逐条判据
    criteria = {
        "all_finite": {"value": all_finite, "pass": all_finite},
        "min_steps_ok": {"value": min_steps_ok, "pass": min_steps_ok},
        "max_err_pct": {"value": conv.get("max_err_pct", None), "pass": conv.get("max_err_pct", 999) <= 3.0},
        "u_l2_decreases": {"value": conv.get("u_l2_decreases"), "pass": conv.get("u_l2_decreases", False)},
        "v_l2_decreases": {"value": conv.get("v_l2_decreases"), "pass": conv.get("v_l2_decreases", False)},
        "u_max_rel_decreases": {"value": conv.get("u_max_rel_decreases"), "pass": conv.get("u_max_rel_decreases", False)},
        "v_max_rel_decreases": {"value": conv.get("v_max_rel_decreases"), "pass": conv.get("v_max_rel_decreases", False)},
    }

    # 构建结果 JSON
    result = {
        "case": "B18_kovasznay_2d_sdaa_sweep",
        "device_evidence": device_evidence,
        "config": {
            "grids": grids,
            "re": re,
            "u0": u0,
            "xmax": xmax,
            "outlet": outlet,
            "min_steps": min_steps,
            "max_steps": max_steps,
            "compile_mode": compile_mode,
        },
        "wall_clock_s": round(wall_elapsed, 1),
        "workaround": {
            "compile_mode_eager": True,
            "reason": "SDAA teco_inductor (torch.compile) 不完整，compile_route 有自动回退机制；本脚本直接用 eager 模式避免编译开销和潜在失败。",
        },
        "per_grid": cases,
        "convergence": conv,
        "criteria": criteria,
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
    }

    # 写入 artifacts
    out_path = Path("artifacts/sdaa_kovasznay_2d.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n结果已写入: {out_path}")
    print(f"Verdict: {verdict}")
    if verdict_reasons:
        print(f"原因: {verdict_reasons}")


if __name__ == "__main__":
    main()
