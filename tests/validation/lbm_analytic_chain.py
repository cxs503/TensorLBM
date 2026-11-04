#!/usr/bin/env python3
"""LBM 解析验证链（Step A–D）的单一可执行真源。

本模块把 ``docs/LBM_VERIFICATION_TEACHING.md`` 里描述的四步解析验证
（剪切波衰减 / Couette / Poiseuille / 壁面摩擦）实现成**可被测试与文档
注册表共用**的一次计算：

* :func:`run_chain` 返回 :class:`Measurement` 列表（实测值/解析值/相对误差）；
* ``python tests/validation/lbm_analytic_chain.py --emit docs/....json``
  把结果写成文档数字注册表，**数字由程序注入，不手抄**。

为什么需要这个模块（验证欠账背景，m3579）：
原 ``tests/test_lbm_verification.py`` 的四个用例只 ``print`` 误差并
``return passed``，pytest 从不校验返回值，因此误差 459% 也会"通过"；
四个用例又整体被 ``tests/ci_quarantine.txt`` 摘出 CI 闸门。两者叠加
= 验证面失效。现在：计算结果集中于此，测试做 assert，注册表测试做
"文档声称 vs 实测"漂移判定。

设备：默认 ``auto``（有 sdaa 用 sdaa，否则 cpu），不再硬编码 ``sdaa:0``，
保证 CPU-only CI 可运行。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

try:  # 常规情况：tensorlbm 已在 import 路径上
    from tensorlbm.boundaries3d import bounce_back_cells_3d
    from tensorlbm.d3q19 import C, W, equilibrium3d, macroscopic3d
    from tensorlbm.solver3d import collide_bgk3d, stream3d
except ImportError:  # 直接从源码树运行时兜底
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from tensorlbm.boundaries3d import bounce_back_cells_3d
    from tensorlbm.d3q19 import C, W, equilibrium3d, macroscopic3d
    from tensorlbm.solver3d import collide_bgk3d, stream3d

__all__ = [
    "Measurement",
    "resolve_device",
    "run_shear_wave",
    "run_couette",
    "run_poiseuille",
    "run_wall_friction",
    "run_chain",
    "render_markdown_table",
    "WITHHELD_CLAIMS",
    "bounce_back_after_stream_err",
    "wall_convention_offset_pct",
    "negative_control_tau_perturbation",
]


#: 仓内**无**可执行证据的文档声称（必须显式登记，禁止沉默地当作已验证）。
WITHHELD_CLAIMS = [
    {
        "id": "step_e_cylinder_cd_grid_convergence",
        "claim": "圆柱 Re=200 压力积分 Cd 网格收敛：D=24 1.48(14.1%) / D=48 1.42(9.5%) / "
                 "D=96 1.39(7.2%) / D=200 1.29(0.8%)，参考 Cd=1.30",
        "status": "WITHHELD",
        "reason": "全仓检索不到任何可执行的圆柱 Re=200 Cd 收敛脚本或测试（Step E 从未落成代码，"
                  "只有文档表格），所以这五个数字没有任何可复算证据。另外基准归属存疑："
                  "Schäfer–Turek(1996) DFG 二维基准是 Re=100、Cd≈3.22–3.24；Re=200 二维圆柱"
                  "常用文献值 Cd≈1.34。文档把 1.30 归给 Schäfer–Turek 并据此算 0.8%，"
                  "Re 与基准归属都需要重新核实后再定判据。",
        "cost": "D=200 需约 4.4M 格点 × 数万步，CPU 单机数十分钟至数小时量级；"
                "本机 load15≈122/128，本轮不启动该重算。",
        "required_to_close": "① 提供圆柱 Re=200 算例脚本（几何/边界/收敛判据可复现）；"
                             "② 先跑 D=24/48 证明误差随加密单调下降；③ 引用可核对的参考值区间"
                             "并说明选值理由；④ 再谈 D=96/200 的成本与结论。",
    },
    {
        "id": "bug6_bounce_back_timing_4_76pct",
        "claim": "Bug 6：反弹放在迁移之后（全路反弹）导致 4.76% 剖面误差",
        "status": "WITHHELD",
        "reason": "m3579 实测不复现：定常 Couette 通道里反弹放在流前/流后收敛到同一解"
                  "（实测差 <1e-3 个百分点）；4.76% 精确等于 0.5/(ny-1.5)=0.5/10.5，"
                  "是壁面全路 vs 半路约定的解析偏移，不是该错法的实测误差。",
        "cost": "已完成复算（tests/test_lbm_verification.py::"
                "test_step_b_bounce_back_order_only_affects_transient）。",
        "required_to_close": "若确有非定常算例会产生 4.76% 量级误差，需给出该算例（含启动瞬态"
                             "取数时刻）；否则文档应把该条改写为「壁面约定」说明。",
    },
]


TAU_DEFAULT = 1.0  # ν = (τ-0.5)/3 = 1/6，精确分数
THRESHOLD_PCT = 1.0  # 教学模块声明：误差 <1% 才算通过


@dataclass
class Measurement:
    """单条解析验证的实测记录。"""

    key: str
    quantity: str
    measured: float
    expected: float
    rel_err_pct: float
    threshold_pct: float = THRESHOLD_PCT
    settings: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return abs(self.rel_err_pct) < self.threshold_pct

    def as_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def resolve_device(pref: str = "auto") -> torch.device:
    """设备解析。

    ``auto``（默认）= 环境变量 ``TENSORLBM_VERIFY_DEVICE`` 或 ``cpu``。
    历史上这里硬编码 ``sdaa:0``，导致 CPU-only CI 直接 collect 失败、
    四个用例被整体 quarantine。在本机实测（80x12x4 通道，100 步）
    cpu=0.54s vs sdaa:0=2.12s，本链全部网格都在万级格点、走 CPU 更快，
    且结果与设备无关（float32 逐点算子），故默认 cpu；
    有真实加速卡时显式传 ``--device sdaa:0`` 即可。
    """
    if pref and pref != "auto":
        return torch.device(pref)
    env = os.environ.get("TENSORLBM_VERIFY_DEVICE", "").strip()
    if env:
        return torch.device(env)
    return torch.device("cpu")


def _rel_err_pct(measured: float, expected: float) -> float:
    return abs(measured - expected) / abs(expected) * 100.0


def _device_tag(d: torch.device) -> str:
    return str(d)


# ---------------------------------------------------------------------------
# Step A: 剪切波衰减 → 验证 ν
# ---------------------------------------------------------------------------
def run_shear_wave(device: str = "auto", tau: float = TAU_DEFAULT, steps: int = 100,
                   nx: int = 64) -> Measurement:
    """u_y(x,t)=A·cos(kx)·exp(-νk²t) → ν_measured = -ln(A_t/A_0)/(k²t)。"""
    d = resolve_device(device)
    ny = nz = 4
    nu_expected = (tau - 0.5) / 3.0
    k = 2 * math.pi / nx
    A0 = 0.01

    uy = torch.zeros(nz, ny, nx, device=d)
    for i in range(nx):
        uy[:, :, i] = A0 * math.cos(k * i)
    f = equilibrium3d(
        torch.ones(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
        uy.clone(),
        torch.zeros(nz, ny, nx, device=d),
    )

    _, _, uy0, _ = macroscopic3d(f)
    a_init = float(uy0[0, 0, 0].item())

    for _ in range(steps):
        f = collide_bgk3d(f, tau=tau)
        f = stream3d(f)

    _, _, uy_f, _ = macroscopic3d(f)
    a_final = float(uy_f[0, 0, 0].item())
    nu_measured = -math.log(abs(a_final / a_init)) / (k ** 2 * steps)

    return Measurement(
        key="shear_wave",
        quantity="ν (kinematic viscosity)",
        measured=nu_measured,
        expected=nu_expected,
        rel_err_pct=_rel_err_pct(nu_measured, nu_expected),
        settings={
            "grid": [nx, ny, nz], "tau": tau, "steps": steps,
            "A_init": a_init, "A_final": a_final,
            "nu_expected": nu_expected, "device": _device_tag(d),
        },
    )


# ---------------------------------------------------------------------------
# Step B: Couette 流 → 验证反弹位置（半路 0.5）+ 无滑移
# ---------------------------------------------------------------------------
def _channel_setup(d, nx, ny, nz, sm_what="bottom"):
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=d)
    solid[:, 0, :] = True
    if sm_what == "both":
        solid[:, -1, :] = True
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
    return solid, sm


def _zero_field(d, nz, ny, nx):
    z = torch.zeros(nz, ny, nx, device=d)
    return equilibrium3d(torch.ones_like(z), z.clone(), z.clone(), z.clone())


def run_couette(device: str = "auto", tau: float = TAU_DEFAULT, steps: int = 2000,
                nx: int = 80, ny: int = 12, nz: int = 4, u_top: float = 0.05) -> Measurement:
    """u(y) = U·(y-0.5)/(ny-1-0.5)：半路反弹时壁面在 y=0.5。"""
    d = resolve_device(device)
    solid, sm = _channel_setup(d, nx, ny, nz, "bottom")
    f = _zero_field(d, nz, ny, nx)
    rho1 = torch.ones(nz, ny, nx, device=d)
    feq_top = equilibrium3d(
        rho1, torch.full_like(rho1, u_top), torch.zeros_like(rho1), torch.zeros_like(rho1)
    )

    for _ in range(steps):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)                     # 碰撞
        for q in range(19):                                # NoDynamics
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)                 # 半路反弹（流之前）
        f[:, :, -1, :] = feq_top[:, :, -1, :]              # 顶壁运动
        f = stream3d(f)                                    # 迁移
        f[:, :, :, 0] = f[:, :, :, -2]                     # 流向周期
        f[:, :, :, -1] = f[:, :, :, -2]

    _, ux, _, _ = macroscopic3d(f)
    u = ux[0, 1:-1, nx // 2].cpu().numpy()
    y = np.arange(1, ny - 1)
    u_exact = u_top * (y - 0.5) / (ny - 1 - 0.5)
    max_err = float(np.max(np.abs(u - u_exact) / u_top) * 100)

    return Measurement(
        key="couette",
        quantity="max abs(u-u_exact)/U (wall-normal profile)",
        measured=max_err,
        expected=0.0,
        rel_err_pct=max_err,
        settings={
            "grid": [nx, ny, nz], "tau": tau, "steps": steps, "u_top": u_top,
            "u_numeric": [round(float(v), 6) for v in u],
            "u_exact": [round(float(v), 6) for v in u_exact],
            "device": _device_tag(d),
        },
    )


@lru_cache(maxsize=4)
def bounce_back_after_stream_err(device: str = "auto", tau: float = TAU_DEFAULT,
                                 steps: int = 2000, nx: int = 80, ny: int = 12,
                                 nz: int = 4, u_top: float = 0.05) -> float:
    """把反弹放在迁移之后（文档所称"Bug 6：全路反弹/时机错误"）时的剖面误差 [%]。

    文档声称该错法带来 4.76% 误差。实测（m3579）：Couette 是**定常**流，
    反弹时机只影响瞬态，两种次序到头来收敛到同一个定常解 —— 实测差
    0.001% 量级，4.76% 复现不出来（4.76% = 0.5/10.5，是"壁面按格点
    而非半格"时解析剖面的相对偏移，属解析上界而非本链的实测误差）。
    """
    d = resolve_device(device)
    solid, sm = _channel_setup(d, nx, ny, nz, "bottom")
    f = _zero_field(d, nz, ny, nx)
    rho1 = torch.ones(nz, ny, nx, device=d)
    feq_top = equilibrium3d(
        rho1, torch.full_like(rho1, u_top), torch.zeros_like(rho1), torch.zeros_like(rho1)
    )

    for _ in range(steps):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f[:, :, -1, :] = feq_top[:, :, -1, :]
        f = stream3d(f)
        f = bounce_back_cells_3d(f, solid)   # ← 故意放在迁移之后
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    _, ux, _, _ = macroscopic3d(f)
    u = ux[0, 1:-1, nx // 2].cpu().numpy()
    y = np.arange(1, ny - 1)
    u_exact = u_top * (y - 0.5) / (ny - 1 - 0.5)
    return float(np.max(np.abs(u - u_exact) / u_top) * 100)


def wall_convention_offset_pct(ny: int = 12) -> float:
    """解析值：若把壁面当格点（全路）而非半格，Couette 剖面相对偏移 = 0.5/(ny-1.5) [%]。"""
    return 0.5 / (ny - 1 - 0.5) * 100.0


@lru_cache(maxsize=4)
def negative_control_tau_perturbation(device: str = "auto", dx_tau: float = 0.02,
                                      steps: int = 100, nx: int = 64) -> dict:
    """负控（判据有牙齿）：把 τ 从 1.0 扰动到 1.02（ν +4%），1% 阈值必须判 FAIL。

    这条负控检验的是**判据的灵敏度**，而不是被验代码：它证明本链的
    1% 阈值真的能把 4% 量级的粘度误差挡住（旧用例 return bool、从不
    assert，任何误差都"通过"）。
    """
    base = run_shear_wave(device=device, tau=TAU_DEFAULT, steps=steps, nx=nx)
    pert = run_shear_wave(device=device, tau=TAU_DEFAULT + dx_tau, steps=steps, nx=nx)
    # 被扰动运行仍用扰动后的 τ 作为"期望"时误差应回到 0；把它与 τ=1.0 的
    # 标称 ν 相比，才是"若代码粘度偏 4% 会不会被拦下"。
    nu_nominal = base.expected
    dev_pct = _rel_err_pct(pert.measured, nu_nominal)
    return {
        "what": "shear-wave viscosity with tau perturbed by +%.2f (nu +%.1f%%)"
                % (dx_tau, dx_tau / 0.5 * 100),
        "nu_nominal": nu_nominal,
        "nu_perturbed": pert.measured,
        "deviation_pct": dev_pct,
        "threshold_pct": THRESHOLD_PCT,
        "judged_fail": dev_pct >= THRESHOLD_PCT,
    }


# ---------------------------------------------------------------------------
# Step C: Poiseuille 流 → 验证体力项 + 速度剖面
# ---------------------------------------------------------------------------
def run_poiseuille(device: str = "auto", tau: float = TAU_DEFAULT, steps: int = 3000,
                   nx: int = 80, ny: int = 12, nz: int = 4, u_max: float = 0.05) -> Measurement:
    """u(y) = G/(2ν)·(y-0.5)(H+0.5-y)，G = 2ν·u_max/H²。"""
    d = resolve_device(device)
    nu = (tau - 0.5) / 3.0
    h = ny - 2
    g = 2 * nu * u_max / h ** 2

    solid, sm = _channel_setup(d, nx, ny, nz, "both")
    c = C.to(d).float()
    w = W.to(d).float()
    f = _zero_field(d, nz, ny, nx)

    for _ in range(steps):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        for q in range(19):                       # Guo 体力（一阶近似）
            f[q] = f[q] + w[q] * 3 * c[q, 0] * g
        f = stream3d(f)
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    _, ux, _, _ = macroscopic3d(f)
    u = ux[0, 1:-1, nx // 2].cpu().numpy()
    y = np.arange(1, ny - 1)
    u_exact = g / (2 * nu) * (y - 0.5) * (h + 0.5 - y)
    max_err = float(np.max(np.abs(u - u_exact) / max(u_exact.max(), 1e-10)) * 100)

    return Measurement(
        key="poiseuille",
        quantity="max abs(u-u_exact)/u_max (parabolic profile)",
        measured=max_err,
        expected=0.0,
        rel_err_pct=max_err,
        settings={
            "grid": [nx, ny, nz], "tau": tau, "steps": steps, "nu": nu, "G": g, "H": h,
            "u_max_numeric": round(float(u.max()), 6),
            "u_max_exact": round(float(u_exact.max()), 6),
            "device": _device_tag(d),
        },
    )


# ---------------------------------------------------------------------------
# Step D: 壁面摩擦 → 验证壁面剪切应力
# ---------------------------------------------------------------------------
def run_wall_friction(device: str = "auto", tau: float = TAU_DEFAULT, steps: int = 3000,
                      nx: int = 80, ny: int = 12, nz: int = 4, u_top: float = 0.05) -> Measurement:
    """Cf = 2τ_w/U²，τ_w = ν·du/dy|_wall，半路反弹下壁面距首个流点 0.5。"""
    d = resolve_device(device)
    nu = (tau - 0.5) / 3.0
    gap = ny - 1 - 0.5                       # 底壁 0.5 → 顶壁 ny-1
    cf_exact = 2 * nu / (gap * u_top)

    solid, sm = _channel_setup(d, nx, ny, nz, "bottom")
    f = _zero_field(d, nz, ny, nx)
    rho1 = torch.ones(nz, ny, nx, device=d)
    feq_top = equilibrium3d(
        rho1, torch.full_like(rho1, u_top), torch.zeros_like(rho1), torch.zeros_like(rho1)
    )

    for _ in range(steps):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        f[:, :, -1, :] = feq_top[:, :, -1, :]
        f = stream3d(f)
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    _, ux, _, _ = macroscopic3d(f)
    u_wall = float(ux[0, 1, nx // 2].item())
    du_dy = u_wall / 0.5
    tau_w = nu * du_dy
    cf_num = 2 * tau_w / u_top ** 2

    return Measurement(
        key="wall_friction",
        quantity="Cf (skin-friction coefficient)",
        measured=cf_num,
        expected=cf_exact,
        rel_err_pct=_rel_err_pct(cf_num, cf_exact),
        settings={
            "grid": [nx, ny, nz], "tau": tau, "steps": steps, "u_top": u_top,
            "u_wall": u_wall, "gap": gap, "device": _device_tag(d),
        },
    )


@lru_cache(maxsize=4)
def _run_chain_cached(device: str) -> tuple:
    return (
        run_shear_wave(device),
        run_couette(device),
        run_poiseuille(device),
        run_wall_friction(device),
    )


def run_chain(device: str = "auto") -> list:
    """运行 Step A–D，返回 Measurement 列表（进程内缓存，避免重复重算）。"""
    return list(_run_chain_cached(device))


STEP_LABELS = {
    "shear_wave": "A 剪切波",
    "couette": "B Couette",
    "poiseuille": "C Poiseuille",
    "wall_friction": "D 壁面摩擦",
}


def render_markdown_table(rows, nc_tau=None, bb_before=None, bb_after=None) -> str:
    """把实测结果渲染成 markdown 表（文档里由生成器注入，测试再逐字比对）。"""
    lines = [
        "| Step | 验证量 | 实测 | 解析值 | 误差 | 判定 |",
        "|---|---|---|---|---|---|",
    ]
    for m in rows:
        lines.append(
            f"| {STEP_LABELS.get(m.key, m.key)} | {m.quantity} | {m.measured:.8f} | "
            f"{m.expected:.8f} | {m.rel_err_pct:.3f}% | {'PASS' if m.passed else 'FAIL'} |"
        )
    if nc_tau is not None:
        lines.append("")
        lines.append(
            f"负控（τ 扰动 {nc_tau['deviation_pct']:.2f}% 粘度偏差）："
            f"{'FAIL → 判据有牙齿' if nc_tau['judged_fail'] else 'PASS → 判据失效！'}"
        )
    if bb_before is not None and bb_after is not None:
        lines.append(
            f"反弹次序：流前 {bb_before:.3f}% / 流后 {bb_after:.3f}%"
            f"（定常 Couette 下互差可忽略；4.76% 系解析壁面约定偏移 "
            f"{wall_convention_offset_pct():.2f}%）"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LBM Step A–D analytic verification chain")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--emit", default=None, help="写出 JSON 注册表路径")
    ap.add_argument("--markdown", default=None, help="写出 markdown 表路径")
    args = ap.parse_args(argv)

    rows = run_chain(args.device)
    print("\nLBM analytic verification chain (Step A–D)")
    print(f"{'step':<16}{'quantity':<46}{'measured':>14}{'expected':>14}{'err[%]':>9}  verdict")
    for m in rows:
        verdict = "PASS" if m.passed else "FAIL"
        print(f"{m.key:<16}{m.quantity:<46}{m.measured:>14.8f}{m.expected:>14.8f}"
              f"{m.rel_err_pct:>9.3f}  {verdict}")

    nc_tau = negative_control_tau_perturbation(args.device)
    bb_after = bounce_back_after_stream_err(args.device)
    bb_before = next(m for m in rows if m.key == "couette").rel_err_pct
    print(f"\nnegative control (teeth): nu perturbed +{nc_tau['deviation_pct']:.2f}% -> "
          f"{'correctly judged FAIL' if nc_tau['judged_fail'] else 'GATE HAS NO TEETH'}")
    print(f"bounce-back order: before-stream {bb_before:.3f}% vs after-stream {bb_after:.3f}% "
          f"(steady Couette -> order paints only the transient)")
    print(f"analytic wall-convention offset (full-way vs half-way) = "
          f"{wall_convention_offset_pct():.2f}%  <- doc's 4.76% is this number, not a measured error")

    if args.emit:
        payload = {
            "schema": "tensorlbm.verification.registry.v1",
            "generated_by": "tests/validation/lbm_analytic_chain.py --emit",
            "device": str(resolve_device(args.device)),
            "torch_version": torch.__version__,
            "threshold_pct": THRESHOLD_PCT,
            "measurements": [m.as_dict() for m in rows],
            "withheld_claims": WITHHELD_CLAIMS,
            "negative_controls": {
                "tau_perturbation": {**nc_tau, "deviation_pct": round(nc_tau["deviation_pct"], 4)},
                "bounce_back_order": {
                    "before_stream_err_pct": round(bb_before, 4),
                    "after_stream_err_pct": round(bb_after, 4),
                    "analytic_wall_convention_offset_pct": round(wall_convention_offset_pct(), 4),
                    "finding": "文档称反弹时机错导致 4.76% 误差；实测两种次序收敛到同一定常解"
                               "（Couette 定常，次序只影响瞬态）。4.76% = 0.5/10.5 是壁面"
                               "全路/半路约定的解析偏移，非本链实测误差。",
                },
            },
        }
        Path(args.emit).parent.mkdir(parents=True, exist_ok=True)
        Path(args.emit).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {args.emit}")

    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(
            render_markdown_table(rows, nc_tau, bb_before, bb_after)
        )
        print(f"wrote {args.markdown}")

    return 0 if all(m.passed for m in rows) else 1


if __name__ == "__main__":
    sys.exit(main())