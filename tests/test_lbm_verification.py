#!/usr/bin/env python3
"""LBM 解析解验证梯 Step A–D（v2, 2026-11-06 verifier 席重制）

四个案例都有闭式解析解，误差阈值 1%（报告另给 3% 判据）：

  Step A: 剪切波衰减  u_y(x,t)=A0·cos(kx)·exp(-νk²t)  → 反演 ν
  Step B: Couette 流  u(y)=u_top·(y-0.5)/(ny-1-0.5)   → 反弹无滑移
  Step C: Poiseuille  u(y)=G/(2ν)·(y-0.5)·(H+0.5-y)   → 体力 + 剖面
  Step D: 壁面摩擦     Cf=2ν/((H-0.5)·U)              → 壁面剪切应力

v2 相对 v1 修好的三处"验证欠账"（v1 在 CI 里被 quarantine, 见
``tests/ci_quarantine.txt`` 历史行）:

1. **设备**：v1 的测试函数签名默认 ``device="sdaa:0"``，pytest 收集到的正是
   这个默认值，无 SDAA 硬件的机器上直接挂死/报错。v2 统一走
   :func:`resolve_device`：``TENSORLBM_TEST_DEVICE`` 环境变量优先，缺省
   ``cpu``；请求 sdaa 但不可用时 ``pytest.skip``。
2. **断言**：v1 的测试函数 ``return passed``（bool），pytest 里没有 assert，
   于是**物理错误的三步也永远绿**——验证梯形同虚设。v2 每个 Step 都断言
   ``err_pct < TOL_PCT``，并把实测数字写进断言消息。
3. **耗时**：v1 每个步内用 ``for q in range(19): f[q] = torch.where(...)``
   和 19 次逐方向加力的 Python 循环，80×12×4 的格子要 ~12 ms/步。v2 用单次
   广播 ``torch.where(sm, f_pre, f)`` 与一次向量化 Guo 强迫，物理逐位等价
   （同一 elementwise 表达式），步耗时降到 ~0.4 ms（另配合
   ``TENSORLBM_TEST_THREADS`` 抑制 128 核机器上的线程超订——超订时实测慢 29×）。

CLI 用法::

    PYTHONPATH=src python tests/test_lbm_verification.py          # 打印表格
    PYTHONPATH=src python tests/test_lbm_verification.py --json artifacts/lbm_verification_ladder/metrics.json

精度说明：默认 ``float64``（验证梯查的是数值方法本身，不是 float32 舍入）；
``TENSORLBM_TEST_DTYPE=float32`` 可对照，差异记在
``docs/lbm_verification_ladder_r2.md``。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, "src")

from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.d3q19 import C, W, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d

TOL_PCT = 1.0  # v1 的原始判据：每个 Step 误差 < 1%
DEFAULT_STEPS = {"A": 100, "B": 2000, "C": 3000, "D": 3000}


# ─────────────────────────── 运行环境辅助 ───────────────────────────
def resolve_device(explicit: str | None = None) -> torch.device:
    """设备解析：explicit > ``TENSORLBM_TEST_DEVICE`` > ``cpu``。"""
    name = explicit or os.environ.get("TENSORLBM_TEST_DEVICE", "cpu")
    if name.startswith("sdaa"):
        sdaa = getattr(torch, "sdaa", None)
        if sdaa is None or not sdaa.is_available():
            import pytest

            pytest.skip(f"requested device {name!r} but torch.sdaa is unavailable")
    return torch.device(name)


def resolve_dtype(explicit: torch.dtype | None = None) -> torch.dtype:
    """dtype 解析：explicit > ``TENSORLBM_TEST_DTYPE`` > ``float64``。"""
    if explicit is not None:
        return explicit
    name = os.environ.get("TENSORLBM_TEST_DTYPE", "float64").lower()
    if name not in {"float32", "float64"}:
        raise ValueError(f"TENSORLBM_TEST_DTYPE must be float32/float64, got {name!r}")
    return getattr(torch, name)


def _limit_threads() -> None:
    """抑制 CPU 线程超订（只有显式设置时才动全局状态）。"""
    n = os.environ.get("TENSORLBM_TEST_THREADS")
    if n:
        torch.set_num_threads(int(n))


# ─────────────────────────── 通道流公共内核 ───────────────────────────
def _zeros(shape, device, dtype):
    return torch.zeros(*shape, device=device, dtype=dtype)


def _channel_run(
    steps: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    nx: int = 80,
    ny: int = 12,
    nz: int = 4,
    tau: float = 1.0,
    u_top: float | None = None,
    gravity: float | None = None,
) -> dict:
    """带底壁无滑移、顶壁为运动平衡态(或闭壁)的二维通道，流向周期。

    ``u_top`` 给定时顶壁是指定速度的运动壁（Couette / 摩擦）；不给定时顶壁
    也是固体（Poiseuille 压力驱动）。``gravity`` 为沿流向的 Guo 强迫体力。
    """
    solid = _zeros((nz, ny, nx), device, torch.bool)
    solid[:, 0, :] = True
    if u_top is None:
        solid[:, -1, :] = True
    mask = solid.unsqueeze(0)  # (1,nz,ny,nx) 广播到 19 方向

    shape = (nz, ny, nx)
    f = equilibrium3d(
        torch.ones(shape, device=device, dtype=dtype),
        _zeros(shape, device, dtype),
        _zeros(shape, device, dtype),
        _zeros(shape, device, dtype),
    )

    # 向量化常数：Guo 强迫 f_q += 3·w_q·c_qx·G
    if gravity is not None:
        force = (W.to(device=device, dtype=dtype).view(19, 1, 1, 1)
                 * 3.0 * C.to(device=device, dtype=dtype)[:, 0].view(19, 1, 1, 1)) * gravity

    for _ in range(steps):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        f = torch.where(mask, f_pre, f)          # NoDynamics：固体格点回滚
        f = bounce_back_cells_3d(f, solid)       # 半路反弹
        if u_top is not None:                    # 顶壁运动平衡态
            rho1 = torch.ones(shape, device=device, dtype=dtype)
            feq = equilibrium3d(
                rho1,
                torch.full_like(rho1, u_top),
                torch.zeros_like(rho1),
                torch.zeros_like(rho1),
            )
            f[:, :, -1, :] = feq[:, :, -1, :]
        if gravity is not None:
            f = f + force                        # 向量化 Guo 强迫
        f = stream3d(f)
        f[:, :, :, 0] = f[:, :, :, -2]           # 流向周期
        f[:, :, :, -1] = f[:, :, :, -2]

    rho, ux, uy, uz = macroscopic3d(f)
    return {"rho": rho, "ux": ux, "uy": uy, "uz": uz}


# ─────────────────────────── Step A ───────────────────────────
def step_a_shear_wave(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    steps: int | None = None,
) -> dict:
    """剪切波衰减：u_y(x,t)=A0·cos(kx)·exp(-νk²t) ⇒ ν = -ln(A_t/A_0)/(k²t)。"""
    device = device or resolve_device()
    dtype = dtype or resolve_dtype()
    steps = steps or DEFAULT_STEPS["A"]

    nx, ny, nz = 64, 4, 4
    tau = 1.0
    nu_exact = (tau - 0.5) / 3.0
    k = 2 * math.pi / nx
    A0 = 0.01

    x = torch.arange(nx, device=device, dtype=dtype)
    uy = (A0 * torch.cos(k * x)).view(1, 1, nx).expand(nz, ny, nx).clone()
    zero = _zeros((nz, ny, nx), device, dtype)
    f = equilibrium3d(torch.ones(nz, ny, nx, device=device, dtype=dtype), zero, uy, zero)
    A_init = float(macroscopic3d(f)[2][0, 0, 0])

    history = [A_init]
    for _ in range(steps):
        f = collide_bgk3d(f, tau=tau)
        f = stream3d(f)  # 全周期域：迁移即全部
        history.append(float(macroscopic3d(f)[2][0, 0, 0]))

    A_end = history[-1]
    nu_num = -math.log(A_end / A_init) / (k * k * steps)
    err_pct = abs(nu_num - nu_exact) / nu_exact * 100.0
    return {
        "step": "A",
        "case": "shear_wave",
        "nu_exact": nu_exact,
        "nu_num": nu_num,
        "A_init": A_init,
        "A_end": A_end,
        "err_pct": err_pct,
        "tolerance_pct": TOL_PCT,
        "passed": err_pct < TOL_PCT,
        "steps": steps,
        "history": history,
    }


# ─────────────────────────── Step B ───────────────────────────
def step_b_couette(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    steps: int | None = None,
) -> dict:
    """Couette 流：线性剖面 u(y)=u_top·(y-0.5)/(ny-1-0.5)，查反弹无滑移。"""
    device = device or resolve_device()
    dtype = dtype or resolve_dtype()
    steps = steps or DEFAULT_STEPS["B"]

    ny, nx, tau, u_top = 12, 80, 1.0, 0.05
    out = _channel_run(steps, device=device, dtype=dtype, nx=nx, ny=ny,
                       tau=tau, u_top=u_top)
    u = out["ux"][0, 1:-1, nx // 2].cpu().numpy()
    y = np.arange(1, ny - 1)
    u_exact = u_top * (y - 0.5) / (ny - 1 - 0.5)
    err_pct = float(np.max(np.abs(u - u_exact) / u_top) * 100.0)
    return {
        "step": "B",
        "case": "couette",
        "u_top": u_top,
        "err_pct": err_pct,
        "tolerance_pct": TOL_PCT,
        "passed": err_pct < TOL_PCT,
        "steps": steps,
        "y": y.tolist(),
        "u_num": u.tolist(),
        "u_exact": u_exact.tolist(),
    }


# ─────────────────────────── Step C ───────────────────────────
def step_c_poiseuille(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    steps: int | None = None,
) -> dict:
    """Poiseuille 流：u(y)=G/(2ν)·(y-0.5)·(H+0.5-y)，查体力实现。"""
    device = device or resolve_device()
    dtype = dtype or resolve_dtype()
    steps = steps or DEFAULT_STEPS["C"]

    ny, nx, nz, tau, u_max = 12, 80, 4, 1.0, 0.05
    nu = (tau - 0.5) / 3.0
    H = ny - 2
    G = 2 * nu * u_max / H**2

    out = _channel_run(steps, device=device, dtype=dtype, nx=nx, ny=ny, nz=nz,
                       tau=tau, gravity=G)
    u = out["ux"][0, 1:-1, nx // 2].cpu().numpy()
    y = np.arange(1, ny - 1)
    u_exact = G / (2 * nu) * (y - 0.5) * (H + 0.5 - y)
    err_pct = float(np.max(np.abs(u - u_exact) / max(u_exact.max(), 1e-30)) * 100.0)
    return {
        "step": "C",
        "case": "poiseuille",
        "nu": nu,
        "gravity": G,
        "u_max_num": float(u.max()),
        "u_max_exact": float(u_exact.max()),
        "err_pct": err_pct,
        "tolerance_pct": TOL_PCT,
        "passed": err_pct < TOL_PCT,
        "steps": steps,
        "y": y.tolist(),
        "u_num": u.tolist(),
        "u_exact": u_exact.tolist(),
    }


# ─────────────────────────── Step D ───────────────────────────
def step_d_friction(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    steps: int | None = None,
) -> dict:
    """壁面摩擦：τ_w=ν·du/dy（半路反弹壁面在 y=0.5），Cf=2τ_w/U² = 2ν/((H-0.5)U)。"""
    device = device or resolve_device()
    dtype = dtype or resolve_dtype()
    steps = steps or DEFAULT_STEPS["D"]

    ny, nx, tau, u_top = 12, 80, 1.0, 0.05
    nu = (tau - 0.5) / 3.0
    gap = 10.5  # 底壁 y=0.5 → 顶壁 y=ny-1=11
    cf_exact = 2 * nu / (gap * u_top)

    out = _channel_run(steps, device=device, dtype=dtype, nx=nx, ny=ny,
                       tau=tau, u_top=u_top)
    u_wall = float(out["ux"][0, 1, nx // 2])
    du_dy = u_wall / 0.5
    tau_w = nu * du_dy
    cf_num = 2 * tau_w / u_top**2
    err_pct = abs(cf_num - cf_exact) / cf_exact * 100.0
    return {
        "step": "D",
        "case": "wall_friction",
        "nu": nu,
        "u_wall": u_wall,
        "tau_w": tau_w,
        "cf_num": cf_num,
        "cf_exact": cf_exact,
        "err_pct": err_pct,
        "tolerance_pct": TOL_PCT,
        "passed": err_pct < TOL_PCT,
        "steps": steps,
    }


# ─────────────────────────── pytest 门（真断言） ───────────────────────────
@pytest.fixture(autouse=True)
def _thread_budget():
    """把 BLAS/OpenMP 线程限到 1（可用 TENSORLBM_TEST_THREADS 覆盖）。

    格子只有 ~4k 个（80×12×4），线程超订纯属损耗：128 核机器上实测
    12 ms/步 → 0.4 ms/步（29×）。退出时恢复原值，避免污染同进程其它测试。
    """
    want = int(os.environ.get("TENSORLBM_TEST_THREADS", "1"))
    prev = torch.get_num_threads()
    torch.set_num_threads(want)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


def test_step_a_shear_wave():
    r = step_a_shear_wave()
    assert r["err_pct"] < TOL_PCT, (
        f"Step A shear-wave viscosity err {r['err_pct']:.4g}% >= {TOL_PCT}% "
        f"(nu_num={r['nu_num']:.6f}, nu_exact={r['nu_exact']:.6f})"
    )


def test_step_b_couette():
    r = step_b_couette()
    assert r["err_pct"] < TOL_PCT, (
        f"Step B Couette profile err {r['err_pct']:.4g}% >= {TOL_PCT}%"
    )


def test_step_c_poiseuille():
    r = step_c_poiseuille()
    assert r["err_pct"] < TOL_PCT, (
        f"Step C Poiseuille profile err {r['err_pct']:.4g}% >= {TOL_PCT}% "
        f"(u_max_num={r['u_max_num']:.6g}, u_max_exact={r['u_max_exact']:.6g})"
    )


def test_step_d_friction():
    r = step_d_friction()
    assert r["err_pct"] < TOL_PCT, (
        f"Step D friction Cf err {r['err_pct']:.4g}% >= {TOL_PCT}% "
        f"(Cf_num={r['cf_num']:.6g}, Cf_exact={r['cf_exact']:.6g})"
    )


# ─────────────────────────── CLI / 报告产物 ───────────────────────────
STEPS = {
    "A": ("Step A: 剪切波衰减 (ν)", step_a_shear_wave),
    "B": ("Step B: Couette 流 (无滑移)", step_b_couette),
    "C": ("Step C: Poiseuille 流 (体力)", step_c_poiseuille),
    "D": ("Step D: 壁面摩擦 (Cf)", step_d_friction),
}


def run_ladder(only: list[str] | None = None, **kwargs) -> dict:
    _limit_threads()
    results = {}
    for key, (label, fn) in STEPS.items():
        if only and key not in only:
            continue
        results[key] = {"label": label, **fn(**kwargs)}
    return results


def render_figure(results: dict, path: Path) -> bool:
    """剖面/衰减对照图；matplotlib 缺失时返回 False（文档里注明无图原因）。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - 可选依赖
        return False

    panels = [k for k in ("B", "C") if k in results]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.2 * (len(panels) + 1), 3.6))
    axes = np.atleast_1d(axes)

    if "A" in results:
        r = results["A"]
        ax = axes[0]
        hist = np.asarray(r["history"])
        t = np.arange(hist.size)
        k = 2 * math.pi / 64
        analytic = r["A_init"] * np.exp(-r["nu_exact"] * k * k * t)
        ax.semilogy(t, hist / r["A_init"], "o", ms=2, label="LBM")
        ax.semilogy(t, analytic / r["A_init"], "-", lw=1, label="exp(-νk²t)")
        ax.set_title(f"A shear wave  err={r['err_pct']:.2e}%")
        ax.set_xlabel("step")
        ax.set_ylabel("A(t)/A(0)")
        ax.legend(fontsize=7)
    else:
        axes[0].axis("off")

    for ax, key in zip(axes[1:], panels):
        r = results[key]
        ax.plot(r["u_num"], r["y"], "o", ms=3, label="LBM")
        ax.plot(r["u_exact"], r["y"], "-", lw=1, label="analytic")
        ax.set_title(f"{r['step']} {r['case']}  err={r['err_pct']:.3f}%")
        ax.set_xlabel("u_x")
        ax.set_ylabel("y")
        ax.legend(fontsize=7)

    fig.suptitle("TensorLBM analytic verification ladder (Step A-D)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def merge_shards(directory: Path) -> dict:
    """合并 ``metrics_<steps>.json`` 分块结果（大机器上整梯一次跑不完时用）。"""
    merged: dict = {}
    for path in sorted(Path(directory).glob("metrics_*.json")):
        if path.name == "metrics.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, res in data.get("steps", {}).items():
            merged[key] = res
    return {k: merged[k] for k in STEPS if k in merged}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TensorLBM analytic verification ladder")
    ap.add_argument("--json", type=Path, default=None, help="metrics JSON 输出路径")
    ap.add_argument("--figure", type=Path, default=None, help="对照图 PNG 输出路径")
    ap.add_argument("--only", default=None, help="只跑指定 Step，如 ABCD 的子集 'AC'")
    ap.add_argument("--merge", type=Path, default=None,
                    help="分块跑完后合并：读 <dir>/metrics_*.json 里各 Step 的结果")
    args = ap.parse_args(argv)

    if args.merge:
        results = merge_shards(args.merge)
        if len(results) < 4:
            print(f"  [warn] 只合并到 {len(results)}/{len(STEPS)} 个 Step: {sorted(results)}")
    else:
        only = list(args.only) if args.only else None
        results = run_ladder(only=only)

    print("\n" + "=" * 74)
    print("  LBM 解析解验证梯 Step A-D（判据：误差 < 1%）")
    print("=" * 74)
    print(f"  {'Step':6} {'case':16} {'err %':>12} {'tol %':>7} {'verdict':>8}")
    for key, r in results.items():
        verdict = "PASS" if r["passed"] else "FAIL"
        print(f"  {key:6} {r['case']:16} {r['err_pct']:12.4g} {r['tolerance_pct']:7.1f} {verdict:>8}")

    n_pass = sum(1 for r in results.values() if r["passed"])
    print(f"\n  通过: {n_pass}/{len(results)}")

    payload = {
        "device": str(resolve_device()),
        "dtype": str(resolve_dtype()).replace("torch.", ""),
        "tol_pct": TOL_PCT,
        "n_pass": n_pass,
        "n_total": len(results),
        "steps": results,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  metrics → {args.json}")
    if args.figure:
        ok = render_figure(results, args.figure)
        print(f"  figure  → {args.figure}{'' if ok else ' (matplotlib 不可用，已跳过)'}")

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())