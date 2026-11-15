#!/usr/bin/env python3
"""R2 复核执行器 — XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION。

首轮（m20921）与 R1（m20984）把 2D 圆柱液滴的 Laplace 标定做到了
（τ=1, velocity-shift/EDM）σ=0.0555，并用两条非循环腿（平面界面 Maxwell 共存、
平面力学平衡 p_l=p_v）支撑它。R1 的「未做的检查」里留有两条与本锚点直接相关的欠账：

* **3D 球形液滴（Δp = 2σ/R）的独立复核**；
* **格点截断（模板四阶矩 κ3）对 σ 的支配性**（R1 只作为「讨论」提及，未作为判据）。

本轮不新写求解器、不改物理内核，只做两件事：

1. **模板算术（无需模拟）**：由库里的速度集/权重算各向同性模板的
   ``a1 = Σ w c_x²``、``a3 = (1/6) Σ w c_x⁴``、``κ3 = a3/a1``。若界面张力由
   Korteweg 截断应力支配（``σ_K = |G_eff κ3| ∫(∂_n ψ)² dn``，见
   ``tensorlbm.sc_surface_tension`` 式 (6)），则
   ``κ3(D2Q9) == κ3(D3Q19)`` ⇒ **预言 σ_3D = σ_2D**（跨维不变）。这是可被否证的预言。
2. **两条数值腿**（同一测量协议、同一后处理代码，维度只作参数）：
   * 平面液膜（2D 与 3D）：到收敛，读两侧体相密度与 EOS 机械压力，
     力学平衡 ``p_l=p_v`` 是格式自洽的闸门；同时由法向剖面算 ``∫(∂_n ψ)² dn``
     给出**独立于曲率**的 σ_K。
   * 液滴半径扫描（2D 与 3D，R=10/14/18）：``Δp = (dims-1) σ / R``，
     ``σ_i = Δp R_eq/(dims-1)``；逐半径一致性 + 跨维比值 + σ_K 对标。

判据（**运行前固定**，见 ``predeclared.json``；运行后未做任何调整）：

* P1 跨维预言：``σ_3D/σ_2D`` 对 1 的相对偏差 < 15%（owner 15% 标准），并显式回答 < 3%。
* P2 平面力学平衡：``|p_l - p_v| / p_v < 1e-3``（2D 与 3D）。
* P3 逐半径一致性：``max_i |σ_i - σ_fit| / σ_fit < 15%``。
* P4 独立锚：``|σ_K(flat) - σ_Lap(droplet)| / σ_K < 15%``（同维比较）。
* P5 离散共存 vs 解析 Maxwell：``|ρ - ρ_Maxwell|/ρ < 15%``。

用法::

    python run_r2.py --stage all --jobs 5       # 全量（集群）
    python run_r2.py --stage all --quick        # 冒烟（小盒/短步）
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.d2q9 import W as W2  # noqa: E402
from tensorlbm.d2q9 import C as C2  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.d3q19 import W as W3  # noqa: E402
from tensorlbm.d3q19 import C as C3  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.multiphase import collide_sc_single_component, psi_exp  # noqa: E402
from tensorlbm.multiphase3d import collide_sc_single_component_3d  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402
from tensorlbm.solver3d import stream3d  # noqa: E402

# ---------------------------------------------------------------------------
# 固定参数（与首轮 / R1 完全一致，未调参）
# ---------------------------------------------------------------------------

G_LIB = 5.0  # 库调用约定（后向 gather）：G_LIB>0 = 吸引，= 教科书 G_eff=-5
G_EFF = -5.0  # 物理约定（压力闭包用）
TAU = 1.0
CS2 = 1.0 / 3.0
W_INIT = 4.0  # 初始 tanh 界面宽

RADII = [10.0, 14.0, 18.0]
L_FACTOR = 4  # L = 4R（与首轮一致）

FLAT = {
    "2": {"ny": 192, "nx": 8, "half": 48.0},
    "3": {"nz": 192, "ny": 12, "nx": 12, "half": 48.0},
}

DROPLET = {
    "steps": 12000,
    "min_steps": 4000,
    "sample_every": 500,
    "conv_dp": 2e-4,
    "conv_r": 2e-3,
}

FLAT_RUN = {"steps": 24000, "sample_every": 500, "conv_tol": 2e-6, "min_steps": 4000}

QUICK = {
    "radii": [10.0, 14.0],
    "flat_steps": 3000,
    "droplet_steps": 2500,
    "flat_half": 24.0,
    "flat_n": 96,
}

THRESHOLDS = {"tolerance": 0.15, "tight": 0.03, "mech_balance": 1e-3}


# ---------------------------------------------------------------------------
# 物理量
# ---------------------------------------------------------------------------


def psi(rho: torch.Tensor) -> torch.Tensor:
    return psi_exp(rho)


def p_eos(rho: torch.Tensor) -> torch.Tensor:
    """机械（EOS 闭合）压力，物理约定 G_eff = -5。"""
    ps = psi(rho)
    return rho * CS2 + 0.5 * G_EFF * CS2 * ps * ps


def kappa3(dim: int) -> dict[str, float]:
    """各向同性模板的四阶矩比 κ3 = a3/a1（式 (4)），纯算术、无模拟。"""
    C = torch.as_tensor(C2 if dim == 2 else C3, dtype=torch.float64)
    W = torch.as_tensor(W2 if dim == 2 else W3, dtype=torch.float64)
    a1 = float((W * C[:, 0] ** 2).sum())
    a3 = float((W * C[:, 0] ** 4).sum()) / 6.0
    return {"a1": a1, "a3": a3, "kappa3": a3 / a1, "n_vel": int(C.shape[0])}


def maxwell_coexistence(n: int = 20001) -> dict[str, float]:
    """SC EOS ``p(ρ)=ρc_s²+(G_eff c_s²/2)ψ²`` 上的 Maxwell 等面积构造（解析基准）。

    在两条 spinodal 压力之间对 ``p_c`` 二分，使 ``∫ dp/ρ = 0``（等化学势）在
    ``ρ_g(p_c)…ρ_l(p_c)`` 上成立。三根搜索按 spinodal 分段，避免跳到高密度支。
    """
    rho = torch.linspace(1e-4, 4.0, n, dtype=torch.float64)
    p = p_eos(rho)
    dpdrho = torch.gradient(p, spacing=(rho,))[0]
    neg = torch.nonzero(dpdrho[1:-1] < 0).flatten() + 1
    if neg.numel() == 0:
        raise RuntimeError("no spinodal: G_eff too weak for phase separation")
    i_max = int(neg[0]) - 1  # 局部极大（第一个负斜率之前）
    i_min = int(neg[-1]) + 1  # 局部极小
    p_hi = float(p[i_max].item())
    p_lo = float(p[i_min].item())

    def root_in(lo_i: int, hi_i: int, pc: float) -> float:
        seg_p, seg_r = p[lo_i : hi_i + 1], rho[lo_i : hi_i + 1]
        for k in range(len(seg_p) - 1):
            if (seg_p[k] - pc) * (seg_p[k + 1] - pc) <= 0.0:
                return float(
                    seg_r[k]
                    + (pc - seg_p[k]) * (seg_r[k + 1] - seg_r[k]) / (seg_p[k + 1] - seg_p[k])
                )
        return float("nan")

    def chem_pot(pc: float) -> float:
        rg = root_in(0, i_max, pc)
        rl = root_in(i_min, n - 1, pc)
        if not (rg > 0 and rl > rg):
            return float("nan")
        m = (rho >= rg) & (rho <= rl)
        r, pp = rho[m], p[m]
        integrand = 1.0 / r
        return float(torch.trapezoid(integrand, pp).item())

    lo, hi = p_lo, p_hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        mu = chem_pot(mid)
        if not math.isfinite(mu):
            lo = mid
            continue
        if mu > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    pc = 0.5 * (lo + hi)
    rg = root_in(0, i_max, pc)
    rl = root_in(i_min, n - 1, pc)
    return {"rho_v": rg, "rho_l": rl, "p_c": pc, "mu_residual": chem_pot(pc),
            "p_spinodal": [p_lo, p_hi]}


# ---------------------------------------------------------------------------
# 求解器封装（只用库内核：collide + stream）
# ---------------------------------------------------------------------------


def _init_droplet(dim: int, L: int, R: float, rho_l: float, rho_v: float, device):
    a = torch.arange(L, dtype=torch.float32, device=device)
    if dim == 2:
        yy, xx = torch.meshgrid(a, a, indexing="ij")
        r = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    else:
        zz, yy, xx = torch.meshgrid(a, a, a, indexing="ij")
        r = torch.sqrt(
            (xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2 + (zz - L / 2.0) ** 2
        )
    s = 0.5 * (1.0 - torch.tanh((r - R) / W_INIT))
    return rho_v + (rho_l - rho_v) * s, r


def _radius_field(dim: int, L: int, device):
    a = torch.arange(L, dtype=torch.float32, device=device)
    if dim == 2:
        yy, xx = torch.meshgrid(a, a, indexing="ij")
        return torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    zz, yy, xx = torch.meshgrid(a, a, a, indexing="ij")
    return torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2 + (zz - L / 2.0) ** 2)


def _step(dim: int, f: torch.Tensor) -> torch.Tensor:
    if dim == 2:
        return stream(
            collide_sc_single_component(f, G=G_LIB, tau=TAU, psi_fn=psi_exp)
        )
    return stream3d(
        collide_sc_single_component_3d(f, G=G_LIB, tau=TAU, psi_fn=psi_exp)
    )


def _macro(dim: int, f: torch.Tensor):
    if dim == 2:
        rho, ux, uy = macroscopic(f)
        return rho, torch.sqrt(ux**2 + uy**2)
    rho, ux, uy, uz = macroscopic3d(f)
    return rho, torch.sqrt(ux**2 + uy**2 + uz**2)


def measure_droplet(dim: int, f: torch.Tensor, rr: torch.Tensor) -> dict[str, float]:
    """体带 → 体相密度 → 中间密度等值面 → R_eq → Δp（同一代码，维度只作参数）。"""
    rho, umag = _macro(dim, f)
    p = p_eos(rho)
    r_eq = float("nan")
    rho_in = rho_out = float("nan")
    # 由液体体积反演等效半径（自洽迭代 3 次）
    mid = 0.5 * (float(rho.max().item()) + float(rho.min().item()))
    for _ in range(3):
        inside = rr <= mid_band(r_eq, 0.5) if math.isfinite(r_eq) else rr <= rr.max() * 0.2
        outside = rr >= mid_band(r_eq, 1.5) if math.isfinite(r_eq) else rr >= rr.max() * 0.5
        rho_in = float(rho[inside].mean().item())
        rho_out = float(rho[outside].mean().item())
        mid = 0.5 * (rho_in + rho_out)
        n_liq = int((rho > mid).sum().item())
        new = (
            math.sqrt(n_liq / math.pi)
            if dim == 2
            else (3.0 * n_liq / (4.0 * math.pi)) ** (1.0 / 3.0)
        )
        if abs(new - r_eq) < 1e-4:
            r_eq = new
            break
        r_eq = new
    inside = rr <= mid_band(r_eq, 0.5)
    outside = rr >= mid_band(r_eq, 1.5)
    p_in = float(p[inside].mean().item())
    p_out = float(p[outside].mean().item())
    rho_in = float(rho[inside].mean().item())
    rho_out = float(rho[outside].mean().item())
    return {
        "R_eq": r_eq,
        "p_in": p_in,
        "p_out": p_out,
        "dp": p_in - p_out,
        "rho_in": rho_in,
        "rho_out": rho_out,
        "max_u": float(umag.max().item()),
        "mass": float(rho.sum().item()),
    }


def mid_band(r: float, f: float) -> float:
    return r * f


# ---------------------------------------------------------------------------
# 腿 A：液滴半径扫描（2D / 3D）
# ---------------------------------------------------------------------------


def run_droplet_case(args: tuple) -> dict[str, Any]:
    dim, R, steps, min_steps, device_s = args
    device = torch.device(device_s)
    L = int(L_FACTOR * R)
    rho_l, rho_v = 1.957, 0.1596  # 首轮 / 库锁定的 2D 离散共存（初始猜测，模型自行弛豫）
    rho0, _ = _init_droplet(dim, L, R, rho_l, rho_v, device)
    zero = torch.zeros_like(rho0)
    f = (
        equilibrium(rho0, zero, zero)
        if dim == 2
        else equilibrium3d(rho0, zero, zero, zero)
    )
    rr = _radius_field(dim, L, device)
    mass0 = float(rho0.sum().item())
    hist: list[dict[str, float]] = []
    conv = False
    step = 0
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        f = _step(dim, f)
        if step % DROPLET["sample_every"] == 0:
            rho_cur = f.sum(dim=0)
            if not torch.isfinite(rho_cur).all() or float(rho_cur.min().item()) < 0.0:
                raise RuntimeError(f"dim={dim} R={R}: NaN/negative rho at step {step}")
            m = measure_droplet(dim, f, rr)
            m["step"] = float(step)
            hist.append(m)
            if len(hist) >= 2 and step >= min_steps:
                d_dp = abs(hist[-1]["dp"] - hist[-2]["dp"]) / abs(hist[-1]["dp"])
                d_r = abs(hist[-1]["R_eq"] - hist[-2]["R_eq"]) / hist[-1]["R_eq"]
                if d_dp < DROPLET["conv_dp"] and d_r < DROPLET["conv_r"]:
                    conv = True
                    break
    tail = hist[-3:]
    mean = lambda k: float(sum(s[k] for s in tail) / len(tail))  # noqa: E731
    R_eq = mean("R_eq")
    dp = mean("dp")
    sigma = dp * R_eq / (dim - 1)
    return {
        "dim": dim,
        "R_init": R,
        "L": L,
        "steps": int(step),
        "converged": conv,
        "R_eq": R_eq,
        "dp": dp,
        "sigma": sigma,
        "rho_in": mean("rho_in"),
        "rho_out": mean("rho_out"),
        "max_u": mean("max_u"),
        "mass_drift": abs(float(hist[-1]["mass"]) - mass0) / mass0,
        "dt_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# 腿 B：平面液膜（2D / 3D）— 共存 + 力学平衡 + Korteweg 锚
# ---------------------------------------------------------------------------


def _init_slab(dim: int, n: dict[str, int], half: float, device):
    if dim == 2:
        ny, nx = n["ny"], n["nx"]
        ys = torch.arange(ny, dtype=torch.float32, device=device)
        s = 0.5 * (
            torch.tanh((ys - (ny / 2.0 - half)) / W_INIT)
            - torch.tanh((ys - (ny / 2.0 + half)) / W_INIT)
        )
        s = s.view(ny, 1).expand(ny, nx).contiguous()
    else:
        nz, ny, nx = n["nz"], n["ny"], n["nx"]
        zs = torch.arange(nz, dtype=torch.float32, device=device)
        s1 = 0.5 * (
            torch.tanh((zs - (nz / 2.0 - half)) / W_INIT)
            - torch.tanh((zs - (nz / 2.0 + half)) / W_INIT)
        )
        s = s1.view(nz, 1, 1).expand(nz, ny, nx).contiguous()
    return 0.1596 + (1.957 - 0.1596) * s


def _profile(rho: torch.Tensor) -> torch.Tensor:
    """横向平均的一维法向剖面。"""
    if rho.dim() == 2:
        return rho.mean(dim=1)
    return rho.mean(dim=(1, 2))


def _bulk(prof: torch.Tensor, half: float) -> dict[str, float]:
    n = prof.shape[0]
    c = n // 2
    # 液膜中心带 / 气相端带（周期：两端都是气相）
    liq = prof[c - int(half * 0.5) : c + int(half * 0.5) + 1]
    gas = torch.cat([prof[: int(half * 0.5)], prof[-int(half * 0.5) :]])
    rho_l = float(liq.mean().item())
    rho_v = float(gas.mean().item())
    p_l = float(p_eos(torch.tensor([rho_l])).item())
    p_v = float(p_eos(torch.tensor([rho_v])).item())
    return {"rho_l": rho_l, "rho_v": rho_v, "p_l": p_l, "p_v": p_v}


def sigma_korteweg_1d(prof: torch.Tensor, k3: float, half: float) -> dict[str, float]:
    """式 (6) 的 1D 版本：σ_K = |G_eff κ3| ∫ (∂_n ψ)² dn（法向，横向已平均）。"""
    ps = psi_exp(prof.double())
    dpsi = torch.gradient(ps)[0]
    integral = float((dpsi**2).sum().item())
    sigma = abs(G_EFF * k3) * integral
    # tanh 拟合的同一积分（抗噪版本）
    fit = _fit_tanh_slab(prof, half)
    ps_fit = psi_exp(fit["profile"].double())
    dpsi_fit = torch.gradient(ps_fit)[0]
    integral_fit = float((dpsi_fit**2).sum().item())
    return {
        "int_dpsi_sq": integral,
        "int_dpsi_sq_tanh": integral_fit,
        "sigma_K": sigma,
        "sigma_K_tanh": abs(G_EFF * k3) * integral_fit,
        "width_fit": fit["w"],
        "drho_fit": fit["drho"],
    }


def _fit_tanh_slab(prof: torch.Tensor, half: float) -> dict[str, Any]:
    """把剖面拟合成两侧 tanh 液膜（宽度 w 网格搜索），返回拟合宽度与拟合剖面。"""
    n = prof.shape[0]
    x = torch.arange(n, dtype=torch.float64)
    rho_l = float(prof.max().item())
    rho_v = float(prof.min().item())
    best_err, w_best = float("inf"), 2.0
    for w in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
        shape = 0.5 * (
            torch.tanh((x - (n / 2.0 - half)) / w)
            - torch.tanh((x - (n / 2.0 + half)) / w)
        )
        err = float(((rho_v + (rho_l - rho_v) * shape - prof.double()) ** 2).mean().item())
        if err < best_err:
            best_err, w_best = err, w
    shape = 0.5 * (
        torch.tanh((x - (n / 2.0 - half)) / w_best)
        - torch.tanh((x - (n / 2.0 + half)) / w_best)
    )
    return {
        "w": w_best,
        "drho": rho_l - rho_v,
        "rms": best_err**0.5,
        "profile": (rho_v + (rho_l - rho_v) * shape).float(),
    }


def run_flat_case(args: tuple) -> dict[str, Any]:
    dim, steps, device_s, half, nscale = args
    device = torch.device(device_s)
    n = dict(FLAT[str(dim)])
    n = {k: int(v * nscale) for k, v in n.items()}
    half = half * nscale if dim == 2 else half * nscale
    rho0 = _init_slab(dim, n, half, device)
    zero = torch.zeros_like(rho0)
    f = (
        equilibrium(rho0, zero, zero)
        if dim == 2
        else equilibrium3d(rho0, zero, zero, zero)
    )
    prof_prev = _profile(rho0)
    conv = False
    step = 0
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        f = _step(dim, f)
        if step % FLAT_RUN["sample_every"] == 0:
            prof = _profile(f.sum(dim=0))
            dec = float((prof - prof_prev).abs().max().item())
            prof_prev = prof
            if dec < FLAT_RUN["conv_tol"] and step >= FLAT_RUN["min_steps"]:
                conv = True
                break
    prof = _profile(f.sum(dim=0))
    k3 = kappa3(dim)["kappa3"]
    out = {
        "dim": dim,
        "steps": int(step),
        "converged": conv,
        "n": n,
        "half": half,
        "dt_s": time.perf_counter() - t0,
        **_bulk(prof, half),
        **sigma_korteweg_1d(prof, k3, half),
    }
    out["mech_gap"] = abs(out["p_l"] - out["p_v"]) / abs(out["p_v"])
    out["profile"] = [float(v) for v in prof.tolist()]
    return out


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def _fit_sigma(rows: list[dict[str, Any]], dim: int) -> dict[str, Any]:
    inv_r = [1.0 / r["R_eq"] for r in rows]
    y = [r["dp"] / (dim - 1) for r in rows]  # σ_i = Δp R/(dims-1) 形式
    num = sum(a * b for a, b in zip(inv_r, y, strict=True))
    den = sum(a * a for a in inv_r)
    sigma_o = num / den
    s = [r["sigma"] for r in rows]
    sigma_mean = sum(s) / len(s)
    dev = [abs(v - sigma_mean) / sigma_mean for v in s]
    # 自由截距最小二乘（诊断用）
    n = len(inv_r)
    sx = sum(inv_r)
    sy = sum(y)
    sxx = sum(a * a for a in inv_r)
    sxy = sum(a * b for a, b in zip(inv_r, y, strict=True))
    denom = n * sxx - sx * sx
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return {
        "sigma_through_origin": sigma_o,
        "sigma_mean_of_radii": sigma_mean,
        "max_dev": max(dev),
        "per_radius_dev": dev,
        "slope_free": a,
        "intercept_free": b,
        "intercept_frac": abs(b) * max(inv_r) / abs(a),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "flat", "droplet", "theory"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    radii = QUICK["radii"] if args.quick else RADII
    flat_steps = QUICK["flat_steps"] if args.quick else FLAT_RUN["steps"]
    drop_steps = QUICK["droplet_steps"] if args.quick else DROPLET["steps"]
    nscale = 0.5 if args.quick else 1.0
    half = QUICK["flat_half"] if args.quick else FLAT["2"]["half"]

    out: dict[str, Any] = {
        "protocol": {
            "anchor": "XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION",
            "round": "R2",
            "model": "SCMP SC94, psi=1-exp(-rho), D2Q9/D3Q19",
            "G_lib": G_LIB,
            "G_eff": G_EFF,
            "tau": TAU,
            "radii": radii,
            "L_factor": L_FACTOR,
            "flat": FLAT,
            "flat_steps": flat_steps,
            "droplet_steps": drop_steps,
            "dims": [2, 3],
            "law": "dp = (dims-1) sigma / R",
            "quick": args.quick,
        },
        "thresholds": THRESHOLDS,
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "threads": args.threads,
        },
    }

    if args.stage in ("all", "theory"):
        out["kappa3"] = {"2": kappa3(2), "3": kappa3(3)}
        out["kappa3_ratio"] = out["kappa3"]["3"]["kappa3"] / out["kappa3"]["2"]["kappa3"]
        out["maxwell"] = maxwell_coexistence()

    if args.stage in ("flat", "all"):
        jobs = [(dim, flat_steps, args.device, half, nscale) for dim in (2, 3)]
        with Pool(min(args.jobs, len(jobs))) as pool:
            flats = pool.map(run_flat_case, jobs)
        out["flat"] = {"2": None, "3": None}
        for r in flats:
            prof = r.pop("profile")
            out["flat"][str(r["dim"])] = r
            out[f"profile_{r['dim']}d"] = prof

    if args.stage in ("droplet", "all"):
        jobs = [(dim, R, drop_steps, DROPLET["min_steps"], args.device) for dim in (2, 3) for R in radii]
        with Pool(min(args.jobs, len(jobs))) as pool:
            rows = pool.map(run_droplet_case, jobs)
        out["droplets"] = rows
        out["fit"] = {}
        for dim in (2, 3):
            sub = [r for r in rows if r["dim"] == dim]
            out["fit"][str(dim)] = _fit_sigma(sub, dim)

    if args.stage == "all":
        s2 = out["fit"]["2"]["sigma_mean_of_radii"]
        s3 = out["fit"]["3"]["sigma_mean_of_radii"]
        out["cross_dim"] = {
            "sigma_2d": s2,
            "sigma_3d": s3,
            "ratio": s3 / s2,
            "dev_from_prediction_1": abs(s3 / s2 - 1.0),
            "predicted_ratio_from_kappa3": out["kappa3_ratio"],
        }
        out["verdicts"] = {
            "P1_cross_dim": out["cross_dim"]["dev_from_prediction_1"] < THRESHOLDS["tolerance"],
            "P1_cross_dim_tight3pct": out["cross_dim"]["dev_from_prediction_1"] < THRESHOLDS["tight"],
            "P2_mech_balance_2d": out["flat"]["2"]["mech_gap"] < THRESHOLDS["mech_balance"],
            "P2_mech_balance_3d": out["flat"]["3"]["mech_gap"] < THRESHOLDS["mech_balance"],
            "P3_radius_consistency_2d": out["fit"]["2"]["max_dev"] < THRESHOLDS["tolerance"],
            "P3_radius_consistency_3d": out["fit"]["3"]["max_dev"] < THRESHOLDS["tolerance"],
            "P4_sigmaK_vs_lap_2d": abs(out["flat"]["2"]["sigma_K_tanh"] - s2)
            / out["flat"]["2"]["sigma_K_tanh"]
            < THRESHOLDS["tolerance"],
            "P4_sigmaK_vs_lap_3d": abs(out["flat"]["3"]["sigma_K_tanh"] - s3)
            / out["flat"]["3"]["sigma_K_tanh"]
            < THRESHOLDS["tolerance"],
        }
        mw = out["maxwell"]
        out["verdicts"]["P5_coexist_vs_maxwell_2d"] = (
            abs(out["flat"]["2"]["rho_l"] - mw["rho_l"]) / mw["rho_l"] < THRESHOLDS["tolerance"]
            and abs(out["flat"]["2"]["rho_v"] - mw["rho_v"]) / mw["rho_v"] < THRESHOLDS["tolerance"]
        )
        out["verdicts"]["P5_coexist_vs_maxwell_3d"] = (
            abs(out["flat"]["3"]["rho_l"] - mw["rho_l"]) / mw["rho_l"] < THRESHOLDS["tolerance"]
            and abs(out["flat"]["3"]["rho_v"] - mw["rho_v"]) / mw["rho_v"] < THRESHOLDS["tolerance"]
        )

    outp = Path(args.out) if args.out else HERE / "artifacts" / ("r2_quick.json" if args.quick else "r2_summary.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k in ("kappa3", "maxwell", "verdicts", "cross_dim")}, indent=2))
    print(f"\nSaved → {outp}")


if __name__ == "__main__":
    main()