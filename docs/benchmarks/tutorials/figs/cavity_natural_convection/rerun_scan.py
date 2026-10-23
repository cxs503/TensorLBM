#!/usr/bin/env python3
"""cavity_natural_convection 教程重算（m219）— 三档网格 + 稳态场快照。

与入库档完全同源：直接复用 benchmarks/verified/cavity_natural_convection/driver.py
（纯胶水编排，物理全部来自 tensorlbm.d2q9 / solver / thermal 库入口），
参数、网格、稳态触发协议与 run_rerun.py 逐项一致：

  - Ra=1e4、Pr=0.71、τ=0.6（ν=(τ−1/2)/3）
  - H = nx−1（等温壁位于节点 x=0 / x=nx−1），gβ = Ra·ν·α/H³
  - 初场线性导热态 T(x)=1−x/H，ρ=1，u=0
  - 稳态触发：|ΔNu| < 1e-4（含左右壁）连续 3 个 1000 步窗，触发下限
    MIN_STEPS 防止早期假触发；触发后追加 max(10000, 10%·t_stop) 验证段

相对入库 run_rerun.py 的增量（教程需要，不改物理）：
  - 稳态触发时把 T 场快照存 npz（供图件零手抄生成）
  - 额外记录剖面量：中线 u(y)、v(x)、T(x) 与热/冷壁局部 Nu(y)
  - 额外算 Richardson 观测阶 p 与 Nu_∞ 外推（纯诊断，不作判据）

用法：
  CUDA_VISIBLE_DEVICES=6 python rerun_scan.py --out-dir data --grids 64 128 256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
CASE_DIR = REPO_ROOT / "benchmarks" / "verified" / "cavity_natural_convection"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(CASE_DIR))
import driver  # noqa: E402
from tensorlbm.d2q9 import macroscopic  # noqa: E402

# 预注册参考（de Vahl Davis 1983，与入库 prereg 同源）
NU_REF = 2.243
U_MAX_REF = 16.178
V_MAX_REF = 19.617

MAX_STEPS = {64: 200_000, 128: 450_000, 256: 1_000_000}
MIN_STEPS = {64: 0, 128: 103_000, 256: 414_000}


def profiles(f, g, nx: int) -> dict:
    """中线剖面与壁面局部 Nu（速度用扩散尺度无量纲化，DVD 约定）。"""
    rho, ux, uy = macroscopic(f)
    T = g.sum(dim=0)
    H = float(nx - 1)
    scale = H / driver.lattice_params(nx, 1e4, 0.71, 0.6)["alpha"]
    xc = H / 2.0
    i0 = int(math.floor(xc))
    frac = xc - i0
    i1 = min(i0 + 1, nx - 1)
    u_line = ((1.0 - frac) * ux[:, i0] + frac * ux[:, i1]).cpu().numpy() * scale
    v_line = ((1.0 - frac) * uy[i0, :] + frac * uy[i1, :]).cpu().numpy() * scale
    T_line = ((1.0 - frac) * T[:, i0] + frac * T[:, i1]).cpu().numpy()
    T_hmid = T[i0, :].cpu().numpy()
    nu_hot = ((T[:, 0] - T[:, 1]) * H).cpu().numpy()  # ΔT=1
    nu_cold = ((T[:, -2] - T[:, -1]) * H).cpu().numpy()
    y = np.arange(T.shape[0], dtype=np.float64) / H
    return {
        "y_norm": y,
        "x_norm": np.arange(T.shape[1], dtype=np.float64) / H,
        "u_midwidth": u_line,
        "v_midheight": v_line,
        "T_midwidth": T_line,
        "T_midheight": T_hmid,
        "nu_hot_wall": nu_hot,
        "nu_cold_wall": nu_cold,
    }


def run_case(nx: int, device: torch.device, out_dir: Path, save_fields: bool = True) -> dict:
    ra, pr, tau = 1e4, 0.71, 0.6
    t_hot, t_cold = 1.0, 0.0
    f, g, wall, p = driver.make_fields(nx, ra, pr, tau, device)
    tau_T, g_beta = p["tau_T"], p["g_beta"]

    import tensorlbm.thermal as th_mod

    th_sha = hashlib.sha256(open(th_mod.__file__, "rb").read()).hexdigest()[:12]

    every = 1000
    target = MAX_STEPS[nx]
    min_steps = MIN_STEPS[nx]
    prev, quiet, t_stop, step = None, 0, None, 0
    hist = []
    t0 = time.time()
    while step < target:
        step += 1
        f, g = driver.advance(f, g, wall, tau, tau_T, g_beta, t_hot, t_cold)
        if step % every == 0:
            obs = driver.observables(f, g, p, t_hot, t_cold)
            drift = 0.0 if prev is None else max(
                abs(obs[k] - prev[k]) for k in ("nu", "nu_left", "nu_right")
            )
            hist.append(
                {
                    "step": step,
                    "nu": obs["nu"],
                    "nu_left": obs["nu_left"],
                    "nu_right": obs["nu_right"],
                    "u_max_nd": obs["u_max_nd"],
                    "v_max_nd": obs["v_max_nd"],
                    "drift": drift,
                }
            )
            prev = obs
            if t_stop is None:
                quiet = quiet + 1 if (step > every and step >= min_steps and drift < 1e-4) else 0
                if quiet >= 3:
                    t_stop = step
                    target = min(MAX_STEPS[nx], step + max(10_000, step // 10))
    elapsed = time.time() - t0

    obs = driver.observables(f, g, p, t_hot, t_cold)
    prof = profiles(f, g, nx)
    npz_path = out_dir / f"fields_n{nx}.npz"
    if save_fields:
        rho_f, ux_f, uy_f = macroscopic(f)
        H = float(nx - 1)
        scale_lu = H / p["alpha"]
        np.savez_compressed(
            npz_path,
            T=g.sum(dim=0).cpu().numpy().astype(np.float32),
            umag=(torch.sqrt(ux_f**2 + uy_f**2) * scale_lu).cpu().numpy().astype(np.float32),
            ux=(ux_f * scale_lu).cpu().numpy().astype(np.float32),
            uy=(uy_f * scale_lu).cpu().numpy().astype(np.float32),
            **{k: v for k, v in prof.items()},
        )

    res = {
        "tag": f"ra1e4_n{nx}",
        "ra": ra,
        "pr": pr,
        "tau": tau,
        "nx": nx,
        "ny": nx,
        "steps_run": step,
        "t_stop": t_stop,
        "steady_triggered": t_stop is not None,
        "verification_steps": target - (t_stop or 0),
        "max_steps": MAX_STEPS[nx],
        "min_steps": min_steps,
        "elapsed_s": round(elapsed, 1),
        "thermal_sha12": th_sha,
        "nu": obs["nu"],
        "nu_left": obs["nu_left"],
        "nu_right": obs["nu_right"],
        "nu_sym": obs["nu_sym"],
        "u_max_nd": obs["u_max_nd"],
        "v_max_nd": obs["v_max_nd"],
        "u_max_lu": obs["u_max_lu"],
        "v_max_lu": obs["v_max_lu"],
        "u_max_y": obs["u_max_y"],
        "v_max_x": obs["v_max_x"],
        "nu_mid": obs["nu_mid"],
        "nu_wall_max": obs["nu_wall_max"],
        "nu_wall_max_y": obs["nu_wall_max_y"],
        "nu_wall_min": obs["nu_wall_min"],
        "nu_wall_min_y": obs["nu_wall_min_y"],
        "T_min": obs["T_min"],
        "T_max": obs["T_max"],
        "u_absmax_lu": obs["u_absmax_lu"],
        "mach": obs["mach"],
        "nu_err_pct": 100.0 * (obs["nu"] - NU_REF) / NU_REF,
        "u_max_err_pct": 100.0 * (obs["u_max_nd"] - U_MAX_REF) / U_MAX_REF,
        "v_max_err_pct": 100.0 * (obs["v_max_nd"] - V_MAX_REF) / V_MAX_REF,
        "grid": {
            "nu": p["nu"],
            "alpha": p["alpha"],
            "tau_T": p["tau_T"],
            "H": p["H"],
            "g_beta": p["g_beta"],
        },
        "fields_npz": str(npz_path),
        "nu_history": hist,
    }
    with open(out_dir / f"result_n{nx}.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(
        f"[n{nx}] Nu={res['nu']:.6f} ({res['nu_err_pct']:+.3f}%) "
        f"u_max={res['u_max_nd']:.4f} ({res['u_max_err_pct']:+.3f}%) "
        f"v_max={res['v_max_nd']:.4f} ({res['v_max_err_pct']:+.3f}%) "
        f"steps={step} t_stop={t_stop} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return res


def richardson(grids: list[dict]) -> dict:
    """三档 Nu(N) 的观测阶 p 与 Nu(N→∞) 外推（纯诊断，不作判据）。"""
    n = np.array([g["nx"] for g in grids], dtype=np.float64)
    nu = np.array([g["nu"] for g in grids], dtype=np.float64)
    out: dict = {}
    if len(grids) == 3:
        d21 = nu[1] - nu[0]
        d32 = nu[2] - nu[1]
        r = n[2] / n[1]
        p_obs = float("nan")
        if abs(d21) > 0 and abs(d32) > 0 and d21 * d32 > 0:
            p_obs = float(math.log(abs(d21 / d32)) / math.log(r))
        out["p_obs"] = p_obs
        if not math.isnan(p_obs) and p_obs > 0:
            out["nu_inf"] = float(nu[-1] + (nu[-1] - nu[-2]) / (r**p_obs - 1.0))
            out["nu_inf_err_pct"] = float(100.0 * (out["nu_inf"] - NU_REF) / NU_REF)
        a, b = np.polyfit(1.0 / n, nu, 1)
        out["nu_inf_linear"] = float(b)
        out["nu_inf_linear_err_pct"] = float(100.0 * (b - NU_REF) / NU_REF)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="cavity_natural_convection 三档网格重算")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--grids", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "-"
    print(f"device={device} gpu={gpu}", flush=True)

    grids = []
    for nx in args.grids:
        grids.append(run_case(nx, device, out_dir))

    nu_errs = [abs(g["nu_err_pct"]) for g in grids]
    monotone = all(nu_errs[i] > nu_errs[i + 1] for i in range(len(nu_errs) - 1))
    scan = {
        "case": "cavity_natural_convection_ra1e4_tutorial_rerun",
        "title": "de Vahl Davis 方腔自然对流 Ra=1e4 Pr=0.71 — m219 教程重算（三档网格 + 场快照）",
        "lattice": "D2Q9 (velocity) + D2Q5 (temperature)",
        "collision": "bgk + Guo buoyancy",
        "boundary": "速度: pre-streaming 半程反弹(四壁); 温度: 左壁 T=1、右壁 T=0 (v3b), 上下绝热",
        "driver_scheme": "mask_wall_u=True（温度碰撞前壁环宏观 u 置零）",
        "extrap": "none",
        "reference": {
            "source": "de Vahl Davis 1983, Int. J. Numer. Methods Fluids 3, 249-264",
            "Nu": NU_REF,
            "u_max": U_MAX_REF,
            "v_max": V_MAX_REF,
        },
        "device": str(device),
        "gpu": gpu,
        "torch": torch.__version__,
        "grids": grids,
        "analysis": {
            "nu_err_abs_pct": nu_errs,
            "nu_err_monotone": monotone,
            "n_pass_3pct": sum(1 for e in nu_errs if e <= 3.0),
            "richardson": richardson(grids),
        },
    }
    with open(out_dir / "scan_result.json", "w", encoding="utf-8") as fh:
        json.dump(scan, fh, indent=2, ensure_ascii=False)
    print("scan_result.json written", flush=True)
    (out_dir / "DONE.json").write_text(
        json.dumps({"done": True, "grids": args.grids, "nu": [g["nu"] for g in grids]}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()