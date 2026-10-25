#!/nfs/wangxi/venvs/tensorlbm/bin/python
"""DIT benchmark — Dynamic Smagorinsky (dsmag) formal run.

晋级路径 ①：换耗散自适应模型（dynamic Smagorinsky, Germano-Lilly 1991/1992）。

三档网格 N=64/128/256，双判据（Kolmogorov -5/3 + 衰减指数 n∈[1.2,1.5]）。
库入口：tensorlbm.turbulence.collide_dynamic_smagorinsky_bgk3d（无需调 C_s）。

用法：
    CUDA_VISIBLE_DEVICES=5 python run_formal_dsmag.py          # 三档全跑
    CUDA_VISIBLE_DEVICES=5 python run_formal_dsmag.py --n 128  # 单档
"""

from __future__ import annotations

import sys as _sys
import types as _types


def _install_thermal_shim() -> None:
    """tensorlbm.thermal WIP 兼容 shim（本 benchmark 不使用 thermal）。"""
    if "tensorlbm.thermal" in _sys.modules:
        return
    import numpy as _np

    m = _types.ModuleType("tensorlbm.thermal")
    m.C_D2Q5 = _np.array([[0, 1, 0, -1, 0], [0, 0, 1, 0, -1]], dtype=_np.float64).T
    m.W_D2Q5 = _np.array([2.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6])

    def _stub(*a, **k):
        raise NotImplementedError("thermal shim stub")

    for _n in (
        "apply_buoyancy_force",
        "collide_thermal_bgk",
        "equilibrium_thermal",
        "macroscopic_thermal",
        "stream_thermal",
    ):
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

# Use the worktree src path
WORKTREE_SRC = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src")
)
if not os.path.isdir(WORKTREE_SRC):
    WORKTREE_SRC = "/nfs/wangxi/tensor_eco/agent_work/m167/TensorLBM/src"
_sys.path.insert(0, WORKTREE_SRC)

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import stream3d
from tensorlbm.turbulence import collide_dynamic_smagorinsky_bgk3d

HERE = os.path.dirname(os.path.abspath(__file__))
KOLMOGOROV_SLOPE = -5.0 / 3.0


# ---------------------------------------------------------------------------
# 初始场：随机无散场
# ---------------------------------------------------------------------------
def random_solenoidal_field(n, m0, u_rms, seed):
    rng = np.random.default_rng(seed)
    kx = np.fft.rfftfreq(n) * n
    ky = np.fft.fftfreq(n) * n
    kz = np.fft.fftfreq(n) * n
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    K2 = KX * KX + KY * KY + KZ * KZ
    K = np.sqrt(K2)
    spec = K**4 * np.exp(-2.0 * (K / m0) ** 2)
    spec[0, 0, 0] = 0.0
    amp = np.sqrt(spec / (4.0 * np.pi * K2 + 1e-30))
    amp[0, 0, 0] = 0.0
    phase_r = rng.standard_normal(spec.shape)
    phase_i = rng.standard_normal(spec.shape)
    phase = phase_r + 1j * phase_i
    ux_h = amp * phase
    uy_h = amp * (rng.standard_normal(spec.shape) + 1j * rng.standard_normal(spec.shape))
    uz_h = amp * (rng.standard_normal(spec.shape) + 1j * rng.standard_normal(spec.shape))
    K2_safe = K2 + (K2 == 0)
    kdotu = KX * ux_h + KY * uy_h + KZ * uz_h
    ux_h -= KX * kdotu / K2_safe
    uy_h -= KY * kdotu / K2_safe
    uz_h -= KZ * kdotu / K2_safe
    ux_h[0, 0, 0] = 0
    uy_h[0, 0, 0] = 0
    uz_h[0, 0, 0] = 0
    ux = np.fft.irfftn(ux_h, s=(n, n, n)).real
    uy = np.fft.irfftn(uy_h, s=(n, n, n)).real
    uz = np.fft.irfftn(uz_h, s=(n, n, n)).real
    u2_mean = (ux**2 + uy**2 + uz**2).mean()
    scale = u_rms / np.sqrt(u2_mean) if u2_mean > 0 else 1.0
    return np.stack([ux * scale, uy * scale, uz * scale])


def ic_divergence(u):
    kx = np.fft.rfftfreq(u.shape[1]) * u.shape[1]
    ky = np.fft.fftfreq(u.shape[1]) * u.shape[1]
    kz = np.fft.fftfreq(u.shape[1]) * u.shape[1]
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    ux_h = np.fft.rfftn(u[0])
    uy_h = np.fft.rfftn(u[1])
    uz_h = np.fft.rfftn(u[2])
    div_h = 1j * KX * ux_h + 1j * KY * uy_h + 1j * KZ * uz_h
    return float(
        np.sqrt(np.mean(np.abs(div_h) ** 2))
        / (np.sqrt(np.mean(np.abs(ux_h) ** 2 + np.abs(uy_h) ** 2 + np.abs(uz_h) ** 2)) + 1e-30)
    )


def build_shells(n, device):
    """球壳索引 + 共轭配对权重（rfftn 约简最后轴）。"""
    kx = np.fft.rfftfreq(n) * n
    ky = np.fft.fftfreq(n) * n
    kz = np.fft.fftfreq(n) * n
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    K = np.sqrt(KX * KX + KY * KY + KZ * KZ)
    idx = np.rint(K).astype(np.int64)
    w = np.where((KX > 0) & (KX < n // 2), 2.0, 1.0).astype(np.float32)
    kmax = int(idx.max())
    return (torch.from_numpy(idx).to(device), torch.from_numpy(w).to(device), kmax)


def energy_spectrum(ux, uy, uz, shell_idx, shell_w, kmax, n):
    """Parseval-orthonormal 球壳能谱 E(k)。"""
    uhat = torch.fft.rfftn(torch.stack([ux, uy, uz]), dim=(1, 2, 3), norm="ortho")
    ek = 0.5 * (uhat.real**2 + uhat.imag**2) * shell_w.unsqueeze(0)
    E = ek.sum(0).double()
    out = torch.zeros(kmax + 1, dtype=torch.float64, device=ux.device)
    out.scatter_add_(0, shell_idx.reshape(-1), E.reshape(-1))
    return out / (n**3)


def velocity_derivative_skewness(ux, uy, uz, n):
    """S3 = ⟨(∂u₁/∂x₁)³⟩/⟨(∂u₁/∂x₁)²⟩^{3/2}（谱微分，周期域）。"""
    kx = torch.fft.rfftfreq(n, device=ux.device) * 2.0 * math.pi
    dux_dx = torch.fft.irfftn(
        torch.fft.rfftn(ux, dim=(0, 1, 2), norm="ortho") * (1j * kx).view(1, 1, -1),
        s=(n, n, n),
        dim=(0, 1, 2),
        norm="ortho",
    )
    d = dux_dx.double()
    m2 = float((d * d).mean().item())
    m3 = float((d * d * d).mean().item())
    if m2 <= 0:
        return 0.0
    return m3 / (m2**1.5)


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def run_case(
    n, u_rms, m0, tau, steps, record_every, seed, device, threads, filter_width=2, lambda_clip=0.0
):
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

    times = []
    energies = []
    urms_list = []
    umax_list = []
    spectra = []
    skew_list = []

    _, ux0m, uy0m, uz0m = macroscopic3d(f)
    u20 = ux0m**2 + uy0m**2 + uz0m**2
    spec0 = energy_spectrum(ux0m, uy0m, uz0m, shell_idx, shell_w, kmax, n)
    times.append(0)
    energies.append(float(0.5 * u20.mean().item()))
    urms_list.append(float(torch.sqrt(u20.mean()).item()))
    umax_list.append(float(torch.sqrt(u20).max().item()))
    spectra.append(spec0.cpu().numpy())

    wall0 = time.time()
    for step in range(1, steps + 1):
        f = stream3d(f)
        f = collide_dynamic_smagorinsky_bgk3d(
            f, tau, filter_width=filter_width, lambda_clip=lambda_clip
        )
        if step % record_every == 0:
            _, uxm, uym, uzm = macroscopic3d(f)
            u2 = uxm**2 + uym**2 + uzm**2
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
    spec_arr = np.stack(spectra)
    mass_end = float(f.sum().item())
    k_phys = 2.0 * np.pi * np.arange(kmax + 1) / n
    kk = k_phys[1:]
    L_int = float(np.pi / (2.0 * E[0]) * np.trapezoid(spec_arr[0][1:] / kk, kk))
    re_L = u_rms * L_int / nu
    tau_L = L_int / u_rms if u_rms > 0 else float("inf")

    return {
        "n": n,
        "times": times,
        "energies": energies,
        "u_rms_history": urms_list,
        "u_max_history": umax_list,
        "spectra": spec_arr,
        "skewness": skew_list,
        "tau_L": tau_L,
        "Re_L": re_L,
        "L_int": L_int,
        "mass_conservation": abs(mass_end - mass0) / mass0,
        "ic_div_rel": div_rel,
        "wall_s": wall,
        "kmax": kmax,
        "collision": "dsmag",
        "filter_width": filter_width,
        "lambda_clip": lambda_clip,
    }


def fit_spectrum(S, kmin, kmax):
    if S.shape[0] == 0:
        return {
            "slope": float("nan"),
            "r2": float("nan"),
            "kmin": kmin,
            "kmax": kmax,
            "n_snapshots": 0,
            "slope_per_snapshot_mean": float("nan"),
            "slope_per_snapshot_std": float("nan"),
        }
    k = np.arange(S.shape[1])
    m = (k >= kmin) & (k <= kmax)
    lk, lS = np.log(k[m] + 1e-30), np.log(S[:, m] + 1e-30)
    per = np.array([np.polyfit(lk, row, 1)[0] for row in lS])
    Sn = S / (S.sum(1, keepdims=True) + 1e-30)
    Sbar = Sn.mean(0)
    a, b = np.polyfit(lk, np.log(Sbar[m] + 1e-30), 1)
    slope = float(a)
    resid = np.log(Sbar[m] + 1e-30) - (b + a * lk)
    ss_tot = (np.log(Sbar[m] + 1e-30) - np.log(Sbar[m] + 1e-30).mean()) @ (
        np.log(Sbar[m] + 1e-30) - np.log(Sbar[m] + 1e-30).mean()
    )
    r2 = float(1.0 - resid @ resid / (ss_tot + 1e-30))
    return {
        "slope": slope,
        "r2": r2,
        "kmin": kmin,
        "kmax": kmax,
        "n_snapshots": int(len(S)),
        "slope_per_snapshot_mean": float(per.mean()),
        "slope_per_snapshot_std": float(per.std()),
    }


def fit_decay(t, E, frac_lo, frac_hi):
    mask = (t >= frac_lo * t[-1]) & (t <= frac_hi * t[-1]) & (t > 0) & (E > 0)
    if mask.sum() < 3:
        return {"n": float("nan"), "r2": float("nan"), "frac_lo": frac_lo, "frac_hi": frac_hi}
    a, b = np.polyfit(np.log(t[mask]), np.log(E[mask]), 1)
    n_decay = float(-a)
    pred = np.exp(b) * t[mask] ** a
    r2 = float(
        1.0 - np.sum((E[mask] - pred) ** 2) / (np.sum((E[mask] - E[mask].mean()) ** 2) + 1e-30)
    )
    return {"n": n_decay, "r2": r2, "frac_lo": frac_lo, "frac_hi": frac_hi}


def judge(results, tol_pct=3.0):
    by_n = {r["n"]: r for r in results}
    ns = sorted(by_n)
    slopes = {n: by_n[n]["spectrum_fit"]["slope"] for n in ns}
    slope_errs = {n: (slopes[n] - KOLMOGOROV_SLOPE) / abs(KOLMOGOROV_SLOPE) * 100.0 for n in ns}
    n_dec = {n: by_n[n]["decay_fit"]["n"] for n in ns}
    within_slope = all(abs(e) <= tol_pct for e in slope_errs.values())
    n_in_band = all(1.2 <= n_dec[n] <= 1.5 for n in ns)
    conv_slope = (
        len(ns) >= 2
        and abs(slopes[ns[1]] - slopes[ns[0]]) / abs(KOLMOGOROV_SLOPE) * 100.0 <= tol_pct
    )
    conv_n = (
        len(ns) >= 2
        and n_dec[ns[0]] != 0
        and abs(n_dec[ns[1]] - n_dec[ns[0]]) / abs(n_dec[ns[0]]) * 100.0 <= tol_pct
    )
    verified = bool(within_slope and n_in_band and conv_slope and conv_n and len(ns) >= 2)
    return {
        "grids": ns,
        "slope_per_grid": {str(n): round(slopes[n], 6) for n in ns},
        "slope_err_pct_per_grid": {str(n): round(v, 4) for n, v in slope_errs.items()},
        "n_decay_per_grid": {str(n): round(n_dec[n], 4) for n in ns},
        "within_slope_tol": within_slope,
        "n_in_physical_band": n_in_band,
        "converged_slope": conv_slope,
        "converged_n": conv_n,
        "tol_pct": tol_pct,
        "verified": verified,
        "judgment": "PASS" if verified else "FAIL",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="0 = 三档 64/128/256")
    ap.add_argument("--u-rms", type=float, default=0.04)
    ap.add_argument("--m0", type=float, default=3.0)
    ap.add_argument("--tau", type=float, default=0.51)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--record-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--kmin", type=int, default=0)
    ap.add_argument("--kmax", type=int, default=0)
    ap.add_argument("--t-dev-mult", type=float, default=3.0)
    ap.add_argument("--t-end-mult", type=float, default=20.0)
    ap.add_argument("--tfrac-lo", type=float, default=0.15)
    ap.add_argument("--tfrac-hi", type=float, default=0.6)
    ap.add_argument("--filter-width", type=int, default=2)
    ap.add_argument("--lambda-clip", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    grids = [64, 128, 256] if args.n == 0 else [args.n]

    results = []
    case_files = []
    for n in grids:
        kmin = args.kmin or int(2 * args.m0 + 2)
        kmax = args.kmax or max(kmin + 2, int(round(0.34 * n / 2)))
        print(
            f"=== N={n}³ dsmag  tau={args.tau}  steps={args.steps}  "
            f"filter_width={args.filter_width} ===",
            flush=True,
        )
        res = run_case(
            n,
            args.u_rms,
            args.m0,
            args.tau,
            args.steps,
            args.record_every,
            args.seed,
            args.device,
            args.threads,
            filter_width=args.filter_width,
            lambda_clip=args.lambda_clip,
        )
        t = np.asarray(res["times"])
        E = np.asarray(res["energies"])
        spec_arr = res["spectra"]
        tau_L = res["tau_L"]
        t_dev = args.t_dev_mult * tau_L
        t_end = args.t_end_mult * tau_L
        t_mask = (t >= t_dev) & (t <= t_end)
        if t_mask.sum() < 2:
            t_mask = np.zeros_like(t, dtype=bool)
            t_mask[len(t) // 3 : 2 * len(t) // 3] = True
        spec_fit = fit_spectrum(spec_arr[t_mask], kmin, kmax)
        decay_fit = fit_decay(t, E, args.tfrac_lo, args.tfrac_hi)
        res["spectrum_fit"] = spec_fit
        res["decay_fit"] = decay_fit
        res["kmin_used"] = kmin
        res["kmax_used"] = kmax
        results.append(res)

        case_json = {
            "n": n,
            "collision": "dsmag",
            "tau": args.tau,
            "filter_width": args.filter_width,
            "lambda_clip": args.lambda_clip,
            "steps": args.steps,
            "record_every": args.record_every,
            "seed": args.seed,
            "u_rms": args.u_rms,
            "m0": args.m0,
            "kmin": kmin,
            "kmax": kmax,
            "t_dev_mult": args.t_dev_mult,
            "t_end_mult": args.t_end_mult,
            "tfrac_lo": args.tfrac_lo,
            "tfrac_hi": args.tfrac_hi,
            "spectrum_fit": spec_fit,
            "decay_fit": decay_fit,
            "tau_L": tau_L,
            "Re_L": res["Re_L"],
            "L_int": res["L_int"],
            "mass_conservation": res["mass_conservation"],
            "ic_div_rel": res["ic_div_rel"],
            "wall_s": res["wall_s"],
            "skewness_summary": [(s, float(v)) for s, v in res["skewness"]],
        }
        case_fn = f"case_N{n}_dsmag.json"
        with open(os.path.join(args.out, case_fn), "w") as fp:
            json.dump(case_json, fp, indent=2, default=str)

        data_fn = f"data_N{n}_dsmag.npz"
        np.savez_compressed(
            os.path.join(args.out, data_fn),
            times=t,
            energies=E,
            u_rms=np.array(res["u_rms_history"]),
            u_max=np.array(res["u_max_history"]),
            spectra=spec_arr,
        )
        case_files.append({"json": case_fn, "npz": data_fn})
        print(
            f"  slope={spec_fit['slope']:.4f}  err={spec_fit['slope'] - KOLMOGOROV_SLOPE:+.4f}  "
            f"n_decay={decay_fit['n']:.4f}  wall={res['wall_s']:.1f}s",
            flush=True,
        )

    verdict = judge(results)
    summary = {
        "benchmark": "dit_turbulence",
        "variant": "dynamic_smagorinsky",
        "description": (
            f"3D DIT, dsmag (Germano-Lilly), N={grids}, tau={args.tau}, "
            f"filter_width={args.filter_width}, lambda_clip={args.lambda_clip}, "
            f"{args.steps} steps, GPU {args.device}, seed={args.seed}"
        ),
        "collision": "dynamic_smagorinsky_bgk3d",
        "common_modules": [
            "solver3d.stream3d",
            "turbulence.collide_dynamic_smagorinsky_bgk3d",
            "d3q19.equilibrium3d",
            "d3q19.macroscopic3d",
        ],
        "case_files": case_files,
        "verdict": verdict,
    }
    with open(os.path.join(args.out, "result_dsmag.json"), "w") as fp:
        json.dump(summary, fp, indent=2, default=str)
    print("\n=== VERDICT ===", flush=True)
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
