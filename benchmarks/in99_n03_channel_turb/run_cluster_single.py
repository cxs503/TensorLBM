#!/usr/bin/env python3
"""Run a single IN99-N03 case based on shard index."""
import sys, os, json, time, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import torch
from tensorlbm.turbulence.turbulent_channel3d import TurbulentChannel3D, TurbulentChannel3DConfig

DNS_DIR = Path(__file__).parent / "dns_ref"
OUT_DIR = Path(__file__).parent / "runs"
OUT_DIR.mkdir(exist_ok=True)

CASES = [
    # (re_tau, u_tau, nx, nz, ny, steps, warmup, cs, label)
    (180, 0.01, 256, 128, 64, 50000, 15000, 0.1, "retau180"),
    (395, 0.005, 384, 192, 96, 60000, 20000, 0.1, "retau395"),
    (590, 0.003, 512, 256, 128, 70000, 25000, 0.1, "retau590"),
]


def load_dns(re_tau):
    fname = {180: "mkm180.csv", 395: "mkm395.csv", 590: "hj590.csv"}[re_tau]
    data = {}
    with open(DNS_DIR / fname) as f:
        header = f.readline().strip().split(",")
        for k in header: data[k] = []
        for line in f:
            vals = line.strip().split(",")
            for k, v in zip(header, vals):
                data[k].append(float(v))
    return {k: np.array(v) for k, v in data.items()}


def run_case(idx):
    re_tau, u_tau, nx, nz, ny, steps, warmup, cs, label = CASES[idx]
    cfg = TurbulentChannel3DConfig(
        nx=nx, nz=nz, ny=ny, re_tau=re_tau, u_tau=u_tau,
        wall_origin=0.5, steps=steps, warmup_steps=warmup,
        sample_interval=5, perturbation_amplitude=0.35,
        perturbation_modes=8, seed=20261109, device="cpu",
        dtype="float32", smagorinsky_cs=cs,
        wall_scheme="interpolated", force_scheme="trt_split", label=label)

    print(f"[{label}] {nx}x{ny}x{nz} Re={re_tau} Cs={cs} steps={steps}")
    print(f"  nu={cfg.nu:.6f} tau+={cfg.tau_plus:.4f} dy+={cfg.dy_plus:.2f}")

    t0 = time.time()
    sim = TurbulentChannel3D(cfg)
    result = sim.run()
    elapsed = time.time() - t0

    y_plus = np.array(result["wall_units"]["y_plus"])
    u_plus = np.array(result["wall_units"]["u_plus"])
    uu_plus = np.array(result["wall_units"]["uu_plus"])
    vv_plus = np.array(result["wall_units"]["vv_plus"])
    ww_plus = np.array(result["wall_units"]["ww_plus"])
    uv_plus = np.array(result["wall_units"]["uv_plus"])

    dns = load_dns(re_tau)
    from scipy.interpolate import interp1d
    f_up = interp1d(dns["y_plus"], dns["u_plus"], kind="linear", fill_value="extrapolate")
    f_uu = interp1d(dns["y_plus"], dns["uu_plus"], kind="linear", fill_value="extrapolate")
    f_vv = interp1d(dns["y_plus"], dns["vv_plus"], kind="linear", fill_value="extrapolate")
    f_ww = interp1d(dns["y_plus"], dns["ww_plus"], kind="linear", fill_value="extrapolate")
    f_uv = interp1d(dns["y_plus"], dns["uv_plus"], kind="linear", fill_value="extrapolate")

    dns_up_i = f_up(y_plus)
    buf = (y_plus > 5) & (y_plus < 30)
    buf_err = float(np.abs(u_plus[buf] - dns_up_i[buf]).mean() / dns_up_i[buf].mean() * 100) if buf.sum() > 0 else float("nan")
    log = (y_plus > 30) & (y_plus < 0.5 * re_tau)
    log_err = float(np.abs(u_plus[log] - dns_up_i[log]).mean() / dns_up_i[log].mean() * 100) if log.sum() > 0 else float("nan")
    rms = (y_plus > 30) & (y_plus < 100)
    rms_err = float(np.sqrt(((u_plus[rms] - dns_up_i[rms])**2).mean()) / dns_up_i[rms].mean() * 100) if rms.sum() > 0 else float("nan")

    u_tau_bf = result["u_tau_routes"]["body_force"]
    u_tau_fit = result["u_tau_routes"]["stress_fit"]
    u_tau_exch = result["u_tau_routes"]["wall_exchange"]
    u_tau_sc = float(abs(u_tau_bf - u_tau_fit) / u_tau_bf * 100)

    uv_minus = -uv_plus
    pk = int(np.argmax(uv_minus))
    pk_yp = float(y_plus[pk])
    pk_mag = float(uv_minus[pk])
    dns_uv_m = -dns["uv_plus"]
    dns_pk = int(np.argmax(dns_uv_m))
    dns_pk_yp = float(dns["y_plus"][dns_pk])
    dns_pk_mag = float(dns_uv_m[dns_pk])

    analysis = {
        "label": label, "re_tau": re_tau, "elapsed_s": elapsed,
        "steps_per_s": steps / elapsed,
        "buffer_error_pct": buf_err, "log_error_pct": log_err, "rms_error_pct": rms_err,
        "u_tau_bf": u_tau_bf, "u_tau_fit": u_tau_fit, "u_tau_exch": u_tau_exch,
        "u_tau_consistency_pct": u_tau_sc,
        "rs_peak_yplus": pk_yp, "rs_peak_mag": pk_mag,
        "dns_peak_yplus": dns_pk_yp, "dns_peak_mag": dns_pk_mag,
        "peak_pos_err": abs(pk_yp - dns_pk_yp),
        "peak_mag_err_pct": abs(pk_mag - dns_pk_mag) / dns_pk_mag * 100,
        "anisotropy": bool(float(uu_plus[pk]) > float(ww_plus[pk]) > float(vv_plus[pk])),
        "uu_peak": float(uu_plus[pk]), "vv_peak": float(vv_plus[pk]), "ww_peak": float(ww_plus[pk]),
        "samples": result["config"]["samples"],
        "eddy_turnovers": result["config"]["eddy_turnovers_averaged"],
        "dy_plus": cfg.dy_plus, "Lx_over_h": nx / cfg.half_height, "Lz_over_h": nz / cfg.half_height,
        "y_plus": y_plus.tolist(), "u_plus": u_plus.tolist(),
        "uu_plus": uu_plus.tolist(), "vv_plus": vv_plus.tolist(),
        "ww_plus": ww_plus.tolist(), "uv_plus": uv_plus.tolist(),
        "dns_y_plus": dns["y_plus"].tolist(), "dns_u_plus": dns["u_plus"].tolist(),
        "dns_uu_plus": dns["uu_plus"].tolist(), "dns_vv_plus": dns["vv_plus"].tolist(),
        "dns_ww_plus": dns["ww_plus"].tolist(), "dns_uv_plus": dns["uv_plus"].tolist(),
    }

    out = OUT_DIR / f"{label}_analysis.json"
    with open(out, "w") as f:
        json.dump({k: v for k, v in analysis.items()
                  if k not in ("y_plus", "u_plus", "uu_plus", "vv_plus", "ww_plus", "uv_plus",
                               "dns_y_plus", "dns_u_plus", "dns_uu_plus", "dns_vv_plus",
                               "dns_ww_plus", "dns_uv_plus")},
                  f, indent=2)
    # Save full profiles separately for plotting
    with open(OUT_DIR / f"{label}_profiles.json", "w") as f:
        json.dump({"y_plus": y_plus.tolist(), "u_plus": u_plus.tolist(),
                   "uu_plus": uu_plus.tolist(), "vv_plus": vv_plus.tolist(),
                   "ww_plus": ww_plus.tolist(), "uv_plus": uv_plus.tolist(),
                   "dns_y_plus": dns["y_plus"].tolist(), "dns_u_plus": dns["u_plus"].tolist(),
                   "dns_uu_plus": dns["uu_plus"].tolist(), "dns_vv_plus": dns["vv_plus"].tolist(),
                   "dns_ww_plus": dns["ww_plus"].tolist(), "dns_uv_plus": dns["uv_plus"].tolist()},
                  f, indent=2)

    print(f"  Buffer: {buf_err:.2f}%  Log: {log_err:.2f}%  RMS: {rms_err:.2f}%")
    print(f"  u_tau cons: {u_tau_sc:.2f}%  RS peak y+: {pk_yp:.1f} (DNS: {dns_pk_yp:.1f})")
    print(f"  Anisotropy: {analysis['anisotropy']}")
    print(f"  Saved: {out}")
    return analysis


if __name__ == "__main__":
    idx = int(sys.argv[1])
    print(f"Running case {idx}: {CASES[idx][-1]}")
    run_case(idx)
