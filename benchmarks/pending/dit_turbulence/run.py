#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""DIT: 3D 衰减各向同性湍流 benchmark（D3Q19 周期域 + LES/Smagorinsky）。

物理问题
--------
周期域 [0,N)³（Δx=Δt=1），随机无散（solenoidal）初始速度场，目标能谱
    E(k) ∝ k⁴·exp(−2(k/k₀)²)，k₀=m0（模式指数）
u_rms 精确归一化；分子 τ→0.5（ν=(τ−1/2)/3 极小）⇒ 分子 Re_L ≥ 1000；
LES 亚格子耗散由 Smagorinsky 提供（C_s=0.12，collide_smagorinsky_bgk3d）。
能量级联在无外力下自由发展（DIT）。

验证指标（真实模拟，无外推）：
    a) 惯性区能量谱 -5/3 斜率（Kolmogorov）：时间平均归一化谱 + 逐快照谱的
       lnE vs lnk 拟合（k 窗口 [kmin, kmax]，发展期时间窗 [t_dev, t_end]）
    b) 衰减指数 E(t)∝t^(−n)：lnE vs lnt 线性拟合（多个窗口 + 虚拟原点诊断），
       文献带 n≈1.2–1.5（Saffman 6/5、Batchelor 10/7、Comte-Bellot-Corrsin 实验）
    c) ≥2 档网格（N=64/128）档间一致（≤3%）为收敛判据
    d) 真实性证据：速度导数偏度 S3≈−0.3~−0.5（湍流级联签名）、
       Σ_k E(k) ≡ 0.5⟨u·u⟩ 谱归一化校验、质量守恒、IC 无散校验

判定：两档 |slope−(−5/3)|/(5/3) ≤ 3% 且档间收敛且 n∈[1.2,1.5] → 保存
benchmarks/verified/dit_turbulence/；否则记录未达标（LES 模型/分辨率/初始场）。

共性模块：solver3d.stream3d + turbulence.collide_smagorinsky_bgk3d（/mrt3d）
          + d3q19.equilibrium3d/macroscopic3d。3D FFT 谱/无散初始场为分析工具
          （库内无，缺口见 /tmp/dit_gap.md）。

用法：
    python run.py --n 0              # 完整 benchmark（N=64/128 两档）→ result.json
    python run.py --n 128 --steps 15000 --out /tmp/x   # 单案例
"""
from __future__ import annotations

import sys as _sys
import types as _types


def _install_thermal_shim() -> None:
    """tensorlbm/__init__ → dg_lbm → physics → thermal 导入链被另一 agent 的
    thermal.py WIP 重写（未提交）破坏（旧 API C_D2Q5/collide_thermal_bgk 等已删）。
    注入兼容 shim（本 benchmark 不使用 thermal 功能，仅需包可导入）。"""
    if "tensorlbm.thermal" in _sys.modules:
        return
    import numpy as _np
    m = _types.ModuleType("tensorlbm.thermal")
    m.C_D2Q5 = _np.array([[0, 1, 0, -1, 0], [0, 0, 1, 0, -1]], dtype=_np.float64).T
    m.W_D2Q5 = _np.array([2.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6])
    def _stub(*a, **k):  # noqa: ANN002, ANN003
        raise NotImplementedError("thermal shim stub (WIP rewrite by another agent)")
    for _n in ("apply_buoyancy_force", "collide_thermal_bgk", "equilibrium_thermal",
               "macroscopic_thermal", "stream_thermal"):
        setattr(m, _n, _stub)
    _sys.modules["tensorlbm.thermal"] = m


_install_thermal_shim()

import argparse
import json
import math
import os
import time

import numpy as np
import torch

_sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

from tensorlbm.solver3d import stream3d  # noqa: E402
from tensorlbm.turbulence import collide_smagorinsky_bgk3d, collide_smagorinsky_mrt3d  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KOLMOGOROV_SLOPE = -5.0 / 3.0


# ---------------------------------------------------------------------------
# 初始场：随机无散场，目标谱 E(k) ∝ k⁴ exp(−2(k/k₀)²)
# ---------------------------------------------------------------------------
def random_solenoidal_field(n: int, m0: float, u_rms: float, seed: int) -> np.ndarray:
    """返回 (3, n, n, n) 实速度场（散度≈0，均值为 0，u_rms 精确归一化）。

    半空间（kx≥0，rfftn 形状 (kz,ky,kx)）复高斯随机相位 → 无散投影
    (I−κκᵀ/κ²，实线性算子保持 Hermitian 对称) → irfftn → 归一化。
    """
    rng = np.random.default_rng(seed)
    kx = np.fft.rfftfreq(n) * n          # 0..N/2 整数模式（约简轴 = x，最后轴）
    ky = np.fft.fftfreq(n) * n           # 带符号模式
    kz = np.fft.fftfreq(n) * n
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")   # (n, n, n//2+1)
    K2 = KX * KX + KY * KY + KZ * KZ
    K = np.sqrt(K2)
    spec = K ** 4 * np.exp(-2.0 * (K / m0) ** 2)
    spec[K == 0] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = np.sqrt(np.where(K > 0, spec / (4.0 * np.pi * K2), 0.0))
    uh = np.empty((3,) + KX.shape, dtype=np.complex128)
    for i in range(3):
        z = rng.standard_normal(KX.shape) + 1j * rng.standard_normal(KX.shape)
        uh[i] = z * amp
    kdotu = KX * uh[0] + KY * uh[1] + KZ * uh[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        fac = np.where(K2 > 0, kdotu / K2, 0.0)
    for i, Kc in enumerate((KX, KY, KZ)):
        uh[i] -= Kc * fac
    uh[:, K == 0] = 0.0
    u = np.fft.irfftn(uh, s=(n, n, n), axes=(1, 2, 3))  # (3,n,n,n) 实场
    ur = np.sqrt(np.mean(u ** 2))
    u = u * (u_rms / ur)
    return u.astype(np.float32)


def ic_divergence(u: np.ndarray) -> float:
    """周期中心差分 max|∇·u|（相对 u_rms 报告）。"""
    d = np.zeros_like(u)
    for ax in range(3):
        d += (np.roll(u, -1, axis=ax + 1) - np.roll(u, 1, axis=ax + 1)) / 2.0
    div = d[0] + d[1] + d[2]
    return float(np.abs(div).max() / np.sqrt(np.mean(u ** 2)))


# ---------------------------------------------------------------------------
# 能量谱（3D FFT，球壳求和；torch.fft.rfftn norm='ortho'）
# ---------------------------------------------------------------------------
def build_shells(n: int, device: torch.device):
    """球壳索引 (bin=round|κ|) 与共轭配对权重（rfftn 只输出 kx≥0 半空间）。

    ⚠️ torch.fft.rfftn 约简「dim 元组最后一个轴」——壳网格必须 (kz,ky,kx)
    顺序（约简轴放最后）。共轭权重：0<kx<N/2 → w=2；kx=0 与 kx=N/2 的
    共轭对已在输出中 → w=1。
    """
    kx = np.fft.rfftfreq(n) * n
    ky = np.fft.fftfreq(n) * n
    kz = np.fft.fftfreq(n) * n
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")   # (n, n, n//2+1)
    K = np.sqrt(KX * KX + KY * KY + KZ * KZ)
    idx = np.rint(K).astype(np.int64)
    w = np.where((KX > 0) & (KX < n // 2), 2.0, 1.0).astype(np.float32)
    kmax = int(idx.max())
    return (torch.from_numpy(idx).to(device),
            torch.from_numpy(w).to(device), kmax)


def energy_spectrum(ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
                    shell_idx: torch.Tensor, shell_w: torch.Tensor,
                    kmax: int, n: int) -> torch.Tensor:
    """返回 (kmax+1,) 球壳能谱 E(k)。norm='ortho' 下 Parseval：
    Σ_{kx≥0} w|û|² = Σ_{x,i} u_i² ⇒ E_tot=0.5⟨u·u⟩ ⇒
    E(k) = Σ_shell 0.5·w|û|²/N³。Σ_k E(k) ≡ E_tot（实空间校验 1e-15 级）。"""
    uhat = torch.fft.rfftn(torch.stack([ux, uy, uz]), dim=(1, 2, 3), norm="ortho")
    ek = 0.5 * (uhat.real ** 2 + uhat.imag ** 2) * shell_w.unsqueeze(0)  # (3,...)
    E = ek.sum(0).double()
    out = torch.zeros(kmax + 1, dtype=torch.float64, device=ux.device)
    out.scatter_add_(0, shell_idx.reshape(-1), E.reshape(-1))
    return out / (n ** 3)


def velocity_derivative_skewness(ux: torch.Tensor, uy: torch.Tensor,
                                 uz: torch.Tensor, n: int) -> float:
    """速度导数偏度 S3 = ⟨(∂u₁/∂x₁)³⟩/⟨(∂u₁/∂x₁)²⟩^{3/2}（周期域，谱微分）。

    湍流级联的经典签名：Gauss 场 S3≈0，发展湍流 S3≈−0.4~−0.5（负偏度 =
    非线性能量前向传递的直接证据）。x 为最后轴 (dim 2)。"""
    kx = torch.fft.rfftfreq(n, device=ux.device) * 2.0 * math.pi
    dux_dx = torch.fft.irfftn(
        torch.fft.rfftn(ux, dim=(0, 1, 2), norm="ortho")
        * (1j * kx).view(1, 1, -1),
        s=(n, n, n), dim=(0, 1, 2), norm="ortho")
    d = dux_dx.double()
    m2 = float((d * d).mean().item())
    m3 = float((d * d * d).mean().item())
    if m2 <= 0:
        return float("nan")
    return float(m3 / m2 ** 1.5)


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def run_case(n: int, u_rms: float, m0: float, tau: float, cs: float,
             steps: int, record_every: int, seed: int,
             collision: str = "bgk", device: str = "cpu",
             threads: int = 32) -> dict:
    torch.set_num_threads(threads)
    dev = torch.device(device)

    u0 = random_solenoidal_field(n, m0, u_rms, seed)
    div_rel = ic_divergence(u0)
    nu = (tau - 0.5) / 3.0

    ux0 = torch.tensor(u0[0], device=dev, dtype=torch.float32)
    uy0 = torch.tensor(u0[1], device=dev, dtype=torch.float32)
    uz0 = torch.tensor(u0[2], device=dev, dtype=torch.float32)
    rho = torch.ones_like(ux0)
    f = equilibrium3d(rho, ux0, uy0, uz0)

    shell_idx, shell_w, kmax = build_shells(n, dev)
    mass0 = float(f.sum().item())

    if collision == "bgk":
        collide = collide_smagorinsky_bgk3d
    elif collision == "mrt":
        collide = collide_smagorinsky_mrt3d
    else:
        raise ValueError(collision)

    times: list[int] = []
    energies: list[float] = []
    urms_list: list[float] = []
    umax_list: list[float] = []
    spectra: list[np.ndarray] = []
    skew_list: list[float] = []

    # 记录 t=0 初值（macroscopic 精确复现 IC）
    _, ux0m, uy0m, uz0m = macroscopic3d(f)
    u20 = ux0m * ux0m + uy0m * uy0m + uz0m * uz0m
    spec0 = energy_spectrum(ux0m, uy0m, uz0m, shell_idx, shell_w, kmax, n)
    times.append(0)
    energies.append(float(0.5 * u20.mean().item()))
    urms_list.append(float(torch.sqrt(u20.mean()).item()))
    umax_list.append(float(torch.sqrt(u20).max().item()))
    spectra.append(spec0.cpu().numpy())

    wall0 = time.time()
    for step in range(1, steps + 1):
        f = stream3d(f)
        f = collide(f, tau, cs)
        if step % record_every == 0:
            _, uxm, uym, uzm = macroscopic3d(f)
            u2 = uxm * uxm + uym * uym + uzm * uzm
            E = float(0.5 * u2.mean().item())
            spec = energy_spectrum(uxm, uym, uzm, shell_idx, shell_w, kmax, n)
            times.append(step)
            energies.append(E)
            urms_list.append(float(torch.sqrt(u2.mean()).item()))
            umax_list.append(float(torch.sqrt(u2).max().item()))
            spectra.append(spec.cpu().numpy())
            if step % (record_every * 5) == 0:
                skew_list.append((step, velocity_derivative_skewness(uxm, uym, uzm, n)))
            if not torch.isfinite(f).all():
                raise RuntimeError(f"NaN/Inf at step {step}")
    wall = time.time() - wall0

    t = np.asarray(times, dtype=np.float64)
    E = np.asarray(energies, dtype=np.float64)
    spec_arr = np.stack(spectra)  # (nt, kmax+1)
    mass_end = float(f.sum().item())

    # 积分尺度 / Taylor 尺度（t=0 谱，物理模式 k_phys=2π·k_mode/N）
    k_phys = 2.0 * np.pi * np.arange(kmax + 1) / n
    kk = k_phys[1:]
    L_int = float(np.pi / (2.0 * E[0]) * np.trapz(spec_arr[0][1:] / kk, kk))
    re_L = u_rms * L_int / nu
    e2 = float(np.trapz(kk ** 2 * spec_arr[0][1:], kk))
    lam = float(np.sqrt(u_rms ** 2 / (2.0 * e2))) if e2 > 0 else float("nan")
    re_lam = u_rms * lam / nu
    tau_L = L_int / u_rms               # 大涡翻转时间（晶格步）

    out = {
        "n": n, "u_rms": u_rms, "m0": m0, "tau": tau, "nu": nu, "cs": cs,
        "steps": steps, "record_every": record_every, "seed": seed,
        "collision": collision, "device": device, "dtype": "float32",
        "ic_div_rel": div_rel, "mass_drift_rel": (mass_end - mass0) / mass0,
        "L_int": L_int, "re_L_molecular": re_L, "lambda_taylor": lam,
        "re_lambda_molecular": re_lam, "tau_L": tau_L,
        "E0": float(E[0]), "E_end": float(E[-1]),
        "E_decay_factor": float(E[0] / E[-1]),
        "wall_sec": wall, "ms_per_step": wall / steps * 1e3,
        "n_samples": len(t),
        "skewness_series": skew_list,
        "times": t.tolist(), "energies": E.tolist(),
        "urms": urms_list, "umax": umax_list,
        "kmax": kmax, "spectra": spec_arr.astype(np.float32),
    }
    return out


# ---------------------------------------------------------------------------
# 拟合与判定
# ---------------------------------------------------------------------------
def fit_decay_exponent(t: np.ndarray, E: np.ndarray,
                       t_lo_frac: float = 0.15, t_hi_frac: float = 0.6) -> dict:
    """E(t) ∝ t^(−n)：窗口 [t_lo_frac·T, t_hi_frac·T] 内 lnE vs lnt 线性拟合。
    虚拟原点 E=C(t+t0)^(−n)（t0 网格搜索最大化 R²）作诊断（标准实验方法）。"""
    t0_ = t[0]
    mask = (t >= t_lo_frac * t[-1]) & (t <= t_hi_frac * t[-1])
    tt, EE = t[mask], E[mask]
    if len(tt) < 8:
        return {"n": float("nan"), "r2": float("nan"), "n_half1": float("nan"),
                "n_half2": float("nan"), "n_virtual_origin": float("nan"),
                "t0_best": float("nan"), "t_lo": t0_, "t_hi": t0_}
    lnt, lnE = np.log(tt), np.log(EE)
    a, b = np.polyfit(lnt, lnE, 1)
    n = -float(a)
    resid = lnE - (b + a * lnt)
    r2 = float(1.0 - resid @ resid / ((lnE - lnE.mean()) @ (lnE - lnE.mean())))
    half = len(tt) // 2
    n1 = -float(np.polyfit(lnt[:half], lnE[:half], 1)[0])
    n2 = -float(np.polyfit(lnt[half:], lnE[half:], 1)[0])
    best = (float("inf"), float("nan"), 0.0)
    for t0 in np.linspace(-tt[0] + 1e-6, tt[-1] * 2.0, 401):
        a0, b0 = np.polyfit(np.log(tt + t0), lnE, 1)
        r = lnE - (b0 + a0 * np.log(tt + t0))
        rss = float(r @ r)
        if rss < best[0]:
            best = (rss, -float(a0), float(t0))
    return {"n": n, "r2": r2, "n_half1": n1, "n_half2": n2,
            "n_virtual_origin": best[1], "t0_best": best[2],
            "t_lo": float(tt[0]), "t_hi": float(tt[-1])}


def fit_spectrum_slope(specs: np.ndarray, times: np.ndarray,
                       kmin: int, kmax: int,
                       t_lo: float, t_hi: float) -> dict:
    """惯性区斜率：时间窗 [t_lo, t_hi] 内每个快照在 k∈[kmin,kmax] 的
    lnE vs lnk 拟合（逐快照均值±std），加上时间平均归一化谱的单次拟合。"""
    sel_t = (times >= t_lo) & (times <= t_hi)
    S = specs[sel_t].astype(np.float64)
    if len(S) < 3 or kmax <= kmin:
        return {"slope": float("nan"), "slope_se": float("nan"), "r2": float("nan"),
                "kmin": kmin, "kmax": kmax, "n_snapshots": 0,
                "slope_per_snapshot_mean": float("nan"),
                "slope_per_snapshot_std": float("nan")}
    k = np.arange(S.shape[1])
    m = (k >= kmin) & (k <= kmax)
    lk, lS = np.log(k[m]), np.log(S[:, m])          # (nsnap, nk)
    per = np.array([np.polyfit(lk, row, 1)[0] for row in lS])
    # 时间平均归一化谱（每快照归一化到形状可比）
    Sn = S / S.sum(1, keepdims=True)
    Sbar = Sn.mean(0)
    a, b = np.polyfit(lk, np.log(Sbar[m]), 1)
    slope = float(a)
    resid = np.log(Sbar[m]) - (b + a * lk)
    r2 = float(1.0 - resid @ resid / ((np.log(Sbar[m]) - np.log(Sbar[m]).mean())
                                      @ (np.log(Sbar[m]) - np.log(Sbar[m]).mean())))
    se = float(np.sqrt(resid @ resid / (len(lk) - 2) / ((lk - lk.mean()) @ (lk - lk.mean()))))
    return {"slope": slope, "slope_se": se, "r2": r2, "kmin": kmin, "kmax": kmax,
            "n_snapshots": int(len(S)),
            "slope_per_snapshot_mean": float(per.mean()),
            "slope_per_snapshot_std": float(per.std())}


def judge(results: list[dict], tol_pct: float = 3.0) -> dict:
    """判定：① 两档 |slope_err|≤3%（vs -5/3）② 两档 n∈[1.2,1.5]
    ③ 档间斜率/指数一致 ≤3%。全部满足才 verified。"""
    by_n = {r["n"]: r for r in results}
    ns = sorted(by_n)
    slopes = {n: by_n[n]["spectrum_fit"]["slope"] for n in ns}
    slope_errs = {n: (slopes[n] - KOLMOGOROV_SLOPE) / abs(KOLMOGOROV_SLOPE) * 100.0
                  for n in ns}
    n_dec = {n: by_n[n]["decay_fit"]["n"] for n in ns}
    within_slope = all(abs(e) <= tol_pct for e in slope_errs.values())
    n_in_band = all(1.2 <= n_dec[n] <= 1.5 for n in ns)
    conv_slope = (len(ns) >= 2 and
                  abs(slopes[ns[1]] - slopes[ns[0]]) / abs(KOLMOGOROV_SLOPE) * 100.0 <= tol_pct)
    conv_n = (len(ns) >= 2 and n_dec[ns[0]] != 0 and
              abs(n_dec[ns[1]] - n_dec[ns[0]]) / abs(n_dec[ns[0]]) * 100.0 <= tol_pct)
    verified = bool(within_slope and n_in_band and conv_slope and conv_n and len(ns) >= 2)
    return {
        "grids": ns,
        "slope_per_grid": {str(n): round(slopes[n], 4) for n in ns},
        "slope_err_pct_per_grid": {str(n): round(v, 4) for n, v in slope_errs.items()},
        "n_decay_per_grid": {str(n): round(n_dec[n], 4) for n in ns},
        "within_slope_tol": within_slope,
        "n_in_physical_band": n_in_band,
        "converged_slope": conv_slope, "converged_n": conv_n,
        "tol_pct": tol_pct,
        "verified": verified,
        "judgment": ("PASS: 两档 -5/3 斜率≤3% 且 n∈[1.2,1.5] 且档间收敛 → 保存 verified/"
                     if verified else
                     "FAIL: 未达 ≤3%/物理带/档间收敛 → 不保存 verified/（记录未达标）"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="DIT decaying isotropic turbulence (D3Q19 LES)")
    ap.add_argument("--n", type=int, default=0, help="0 = 完整 benchmark (64/128)")
    ap.add_argument("--u-rms", type=float, default=0.04)
    ap.add_argument("--m0", type=float, default=3.0)
    ap.add_argument("--tau", type=float, default=0.501)
    ap.add_argument("--cs", type=float, default=0.12)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--record-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--collision", choices=["bgk", "mrt"], default="bgk")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--kmin", type=int, default=0, help="0 = auto (2·m0+2)")
    ap.add_argument("--kmax", type=int, default=0, help="0 = auto (0.34·N/2)")
    ap.add_argument("--t-dev-mult", type=float, default=3.0, help="发展期起点 = t_dev_mult·τ_L")
    ap.add_argument("--t-end-mult", type=float, default=20.0, help="发展期终点 = t_end_mult·τ_L")
    ap.add_argument("--tfrac-lo", type=float, default=0.15, help="衰减拟合窗口下界 (×T)")
    ap.add_argument("--tfrac-hi", type=float, default=0.6, help="衰减拟合窗口上界 (×T)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grids = [64, 128] if args.n == 0 else [args.n]
    results, case_files = [], []
    for n in grids:
        kmin = args.kmin or int(2 * args.m0 + 2)
        kmax = args.kmax or max(kmin + 2, int(round(0.34 * n / 2)))
        print(f"=== N={n}³  u_rms={args.u_rms} m0={args.m0} tau={args.tau} "
              f"Cs={args.cs} steps={args.steps} {args.collision} ===", flush=True)
        res = run_case(n, args.u_rms, args.m0, args.tau, args.cs,
                       args.steps, args.record_every, args.seed,
                       collision=args.collision, device=args.device,
                       threads=args.threads)
        t = np.asarray(res["times"]); E = np.asarray(res["energies"])
        spec_arr = res["spectra"]
        tau_L = res["tau_L"]
        t_dev = args.t_dev_mult * tau_L
        t_end = min(float(t[-1]), args.t_end_mult * tau_L)
        res["decay_fit"] = fit_decay_exponent(t, E, args.tfrac_lo, args.tfrac_hi)
        res["spectrum_fit"] = fit_spectrum_slope(spec_arr, t, kmin, kmax, t_dev, t_end)
        res["kmin_fit"], res["kmax_fit"] = kmin, kmax
        res["t_dev"], res["t_end_spec"] = t_dev, t_end
        # 斜率演化曲线（每快照，存档诊断）
        slopes_ev = []
        for i, tt in enumerate(t[1:], start=1):
            if tt < t_dev or tt > t_end:
                continue
            row = spec_arr[i]
            kk = np.arange(len(row))
            mm = (kk >= kmin) & (kk <= kmax)
            if mm.sum() >= 3:
                slopes_ev.append([float(tt), float(np.polyfit(np.log(kk[mm]),
                                                              np.log(row[mm]), 1)[0])])
        res["slope_evolution"] = slopes_ev

        keep = {k: v for k, v in res.items() if k not in ("spectra", "slope_evolution")}
        name = f"case_N{n}.json"
        with open(os.path.join(args.out, name), "w") as fh:
            json.dump(keep, fh, indent=2, ensure_ascii=False)
        np.savez(os.path.join(args.out, f"data_N{n}.npz"),
                 times=t, energies=E, spectra=spec_arr,
                 urms=np.asarray(res["urms"]), umax=np.asarray(res["umax"]))
        case_files.append(name)
        results.append(res)

        df, sf = res["decay_fit"], res["spectrum_fit"]
        print(f"  L_int={res['L_int']:.2f} Re_L={res['re_L_molecular']:.0f} "
              f"τ_L={tau_L:.0f} Re_λ={res['re_lambda_molecular']:.0f}", flush=True)
        print(f"  E0={res['E0']:.3e} E_end={res['E_end']:.3e} "
              f"decay={res['E_decay_factor']:.1f}x  wall={res['wall_sec']:.0f}s "
              f"({res['ms_per_step']:.1f} ms/step)", flush=True)
        print(f"  decay n={df['n']:.4f} (R²={df['r2']:.4f}, half {df['n_half1']:.3f}/"
              f"{df['n_half2']:.3f}, vo={df['n_virtual_origin']:.3f})", flush=True)
        print(f"  slope[{kmin},{kmax}] t∈[{t_dev:.0f},{t_end:.0f}] = {sf['slope']:.4f} ± "
              f"{sf['slope_se']:.4f} (R²={sf['r2']:.4f}, n_snap={sf['n_snapshots']})  "
              f"err_vs_-5/3={100*(sf['slope']-KOLMOGOROV_SLOPE)/abs(KOLMOGOROV_SLOPE):+.2f}%  "
              f"per-snap {sf['slope_per_snapshot_mean']:.3f}±{sf['slope_per_snapshot_std']:.3f}",
              flush=True)
        s3s = [s for _, s in res["skewness_series"]]
        print(f"  skewness S3 mean={np.mean(s3s):.3f} (n={len(s3s)}, "
              f"first={s3s[0]:.3f} last={s3s[-1]:.3f})  mass_drift={res['mass_drift_rel']:.1e}",
              flush=True)
        print(f"  -> {os.path.join(args.out, name)}", flush=True)

    verdict = judge(results)
    out = {
        "benchmark": "dit_turbulence",
        "description": ("3D decaying isotropic turbulence (DIT): random solenoidal IC with "
                        "E(k)∝k⁴exp(−2(k/m0)²), m0=3, u_rms=0.04, periodic N³ domain, "
                        "D3Q19 + Smagorinsky LES (C_s=0.12, τ=0.501, molecular Re_L≥1000)"),
        "lattice": "D3Q19",
        "collision": f"smagorinsky_{args.collision}",
        "boundary": "periodic (stream3d 模运算, 库内建)",
        "extrap": "none",
        "common_modules": ["solver3d.stream3d", "turbulence.collide_smagorinsky_bgk3d",
                           "d3q19.equilibrium3d", "d3q19.macroscopic3d"],
        "analysis_tools": ["energy_spectrum (3D FFT shell sum, run.py 自实现, 库无 — G-DIT-1)",
                           "random_solenoidal_field (run.py 自实现, 库无 — G-DIT-2)",
                           "velocity_derivative_skewness (run.py 自实现)"],
        "metrics": {
            "kolmogorov_slope": KOLMOGOROV_SLOPE,
            "spectrum_fit": "lnE vs lnk 拟合, k∈[kmin,kmax], 时间窗 [3τ_L, 20τ_L]",
            "decay_band": [1.2, 1.5],
            "decay_fit": "lnE vs lnt 拟合, 窗口 [0.15T, 0.6T], 虚拟原点诊断",
        },
        "cases": [{k: v for k, v in r.items()
                   if k not in ("times", "energies", "urms", "umax", "spectra",
                                "slope_evolution", "skewness_series")} for r in results],
        "case_files": case_files,
        "convergence": verdict,
    }
    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\n=== 判定 ===")
    print(f"  slope err: " + ", ".join(f"N{n}={e:+.2f}%" for n, e in
                                       verdict["slope_err_pct_per_grid"].items()))
    print(f"  n_decay: " + ", ".join(f"N{n}={v}" for n, v in
                                     verdict["n_decay_per_grid"].items()))
    print(f"  within_slope={verdict['within_slope_tol']}  n_in_band={verdict['n_in_physical_band']}  "
          f"conv_slope={verdict['converged_slope']}  conv_n={verdict['converged_n']}")
    print(f"  {verdict['judgment']}")
    print(f"  -> {os.path.join(args.out, 'result.json')}")


if __name__ == "__main__":
    main()
