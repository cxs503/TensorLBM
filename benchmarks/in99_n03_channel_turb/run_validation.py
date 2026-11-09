#!/usr/bin/env python3
"""IN99-N03: LBM turbulent channel flow validation at Re_tau=180/395/590."""
from __future__ import annotations
import csv, json, math, sys, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from tensorlbm.turbulence.turbulent_channel3d import TurbulentChannel3D, TurbulentChannel3DConfig

DNS_DIR = Path(__file__).parent / "dns_ref"
RUNS_DIR = Path(__file__).parent / "runs"
FIGURES_DIR = Path(__file__).parent / "figures"
for d in [RUNS_DIR, FIGURES_DIR]: d.mkdir(exist_ok=True)

def load_dns(re_tau):
    path = DNS_DIR / {180: "mkm180.csv", 395: "mkm395.csv", 590: "hj590.csv"}[re_tau]
    data = {"y_over_h": [], "y_plus": [], "u_plus": [], "uu_plus": [], "vv_plus": [], "ww_plus": [], "uv_plus": []}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k in data: data[k].append(float(row[k]))
    return {k: np.array(v) for k, v in data.items()}

@dataclass
class GridConfig:
    re_tau: float; u_tau: float; nx: int; nz: int; ny: int
    steps: int; warmup: int; sample_interval: int; smagorinsky_cs: float; label: str; device: str = "cpu"

def get_grid_configs():
    return [
        GridConfig(180, 0.01, 256, 128, 64, 50000, 15000, 5, 0.1, "retau180"),
        GridConfig(395, 0.005, 384, 192, 96, 60000, 20000, 5, 0.1, "retau395"),
        GridConfig(590, 0.003, 512, 256, 128, 70000, 25000, 5, 0.1, "retau590"),
    ]

def run_case(cfg, sgs_on=True):
    cs = cfg.smagorinsky_cs if sgs_on else 0.0
    label = cfg.label if sgs_on else f"{cfg.label}_nosgs"
    config = TurbulentChannel3DConfig(
        nx=cfg.nx, nz=cfg.nz, ny=cfg.ny, re_tau=cfg.re_tau, u_tau=cfg.u_tau,
        wall_origin=0.5, steps=cfg.steps, warmup_steps=cfg.warmup,
        sample_interval=cfg.sample_interval, perturbation_amplitude=0.35,
        perturbation_modes=8, seed=20261109, device=cfg.device, dtype="float32",
        smagorinsky_cs=cs, wall_scheme="interpolated", force_scheme="trt_split", label=label)
    print(f"[{label}] nx={cfg.nx} nz={cfg.nz} ny={cfg.ny} Re={cfg.re_tau} Cs={cs}")
    t0 = time.time()
    sim = TurbulentChannel3D(config)
    result = sim.run()
    result["elapsed_seconds"] = time.time() - t0
    result["sgs_on"] = sgs_on
    return result

def analyze_result(result, dns):
    from scipy.interpolate import interp1d
    y_plus = np.array(result["wall_units"]["y_plus"])
    u_plus = np.array(result["wall_units"]["u_plus"])
    uu_plus = np.array(result["wall_units"]["uu_plus"])
    vv_plus = np.array(result["wall_units"]["vv_plus"])
    ww_plus = np.array(result["wall_units"]["ww_plus"])
    uv_plus = np.array(result["wall_units"]["uv_plus"])
    f_up = interp1d(dns["y_plus"], dns["u_plus"], kind="linear", fill_value="extrapolate")
    f_uu = interp1d(dns["y_plus"], dns["uu_plus"], kind="linear", fill_value="extrapolate")
    f_vv = interp1d(dns["y_plus"], dns["vv_plus"], kind="linear", fill_value="extrapolate")
    f_ww = interp1d(dns["y_plus"], dns["ww_plus"], kind="linear", fill_value="extrapolate")
    f_uv = interp1d(dns["y_plus"], dns["uv_plus"], kind="linear", fill_value="extrapolate")
    dns_up_i = f_up(y_plus); dns_uu_i = f_uu(y_plus); dns_vv_i = f_vv(y_plus)
    dns_ww_i = f_ww(y_plus); dns_uv_i = f_uv(y_plus)
    buf = (y_plus > 5) & (y_plus < 30)
    buf_err = float(np.abs(u_plus[buf] - dns_up_i[buf]).mean() / dns_up_i[buf].mean() * 100) if buf.sum() > 0 else float("nan")
    log = (y_plus > 30) & (y_plus < 0.5 * result["config"]["re_tau_target"])
    log_err = float(np.abs(u_plus[log] - dns_up_i[log]).mean() / dns_up_i[log].mean() * 100) if log.sum() > 0 else float("nan")
    rms = (y_plus > 30) & (y_plus < 100)
    rms_err = float(np.sqrt(((u_plus[rms] - dns_up_i[rms])**2).mean()) / dns_up_i[rms].mean() * 100) if rms.sum() > 0 else float("nan")
    u_tau_bf = result["u_tau_routes"]["body_force"]
    u_tau_fit = result["u_tau_routes"]["stress_fit"]
    u_tau_exch = result["u_tau_routes"]["wall_exchange"]
    u_tau_sc = float(abs(u_tau_bf - u_tau_fit) / u_tau_bf * 100)
    uv_minus = -np.array(result["wall_units"]["uv_plus"])
    pk = int(np.argmax(uv_minus))
    pk_yp = float(y_plus[pk]); pk_mag = float(uv_minus[pk])
    dns_uv_m = -dns["uv_plus"]; dns_pk = int(np.argmax(dns_uv_m))
    dns_pk_yp = float(dns["y_plus"][dns_pk]); dns_pk_mag = float(dns_uv_m[dns_pk])
    return {
        "buffer_layer_error_pct": buf_err, "log_law_error_pct": log_err,
        "rms_error_y30_100_pct": rms_err, "u_tau_self_consistency_pct": u_tau_sc,
        "u_tau_body_force": u_tau_bf, "u_tau_stress_fit": u_tau_fit, "u_tau_wall_exchange": u_tau_exch,
        "reynolds_stress_peak_yplus": pk_yp, "reynolds_stress_peak_magnitude": pk_mag,
        "dns_peak_yplus": dns_pk_yp, "dns_peak_magnitude": dns_pk_mag,
        "peak_position_error": float(abs(pk_yp - dns_pk_yp)),
        "peak_magnitude_error_pct": float(abs(pk_mag - dns_pk_mag) / dns_pk_mag * 100),
        "anisotropy_correct": bool(float(uu_plus[pk]) > float(ww_plus[pk]) > float(vv_plus[pk])),
        "uu_at_peak": float(uu_plus[pk]), "vv_at_peak": float(vv_plus[pk]), "ww_at_peak": float(ww_plus[pk]),
        "y_plus": y_plus.tolist(), "u_plus": u_plus.tolist(),
        "uu_plus": uu_plus.tolist(), "vv_plus": vv_plus.tolist(),
        "ww_plus": ww_plus.tolist(), "uv_plus": uv_plus.tolist(),
        "dns_y_plus": dns["y_plus"].tolist(), "dns_u_plus": dns["u_plus"].tolist(),
        "dns_uu_plus": dns["uu_plus"].tolist(), "dns_vv_plus": dns["vv_plus"].tolist(),
        "dns_ww_plus": dns["ww_plus"].tolist(), "dns_uv_plus": dns["uv_plus"].tolist(),
    }

def main():
    print("="*60 + "
IN99-N03: LBM Turbulent Channel Validation
" + "="*60)
    configs = get_grid_configs()
    all_results, all_analyses, all_criteria = {}, {}, {}
    for cfg in configs:
        print(f"
Running Re_tau={cfg.re_tau} ({cfg.label})")
        result = run_case(cfg, sgs_on=True)
        dns = load_dns(int(cfg.re_tau))
        analysis = analyze_result(result, dns)
        all_results[cfg.label] = result; all_analyses[cfg.label] = analysis
        out = RUNS_DIR / f"{cfg.label}_result.json"
        with open(out, "w") as f:
            json.dump({"result": {k: v for k, v in result.items() if k not in ("config", "wall_units")},
                       "analysis": {k: v for k, v in analysis.items() if "dns" not in k and k not in ("y_plus","u_plus","uu_plus","vv_plus","ww_plus","uv_plus")}}, f, indent=2, default=str)
        print(f"  Buffer err: {analysis["buffer_layer_error_pct"]:.2f}%  Log err: {analysis["log_law_error_pct"]:.2f}%  RMS: {analysis["rms_error_y30_100_pct"]:.2f}%")
        print(f"  u_tau consistency: {analysis["u_tau_self_consistency_pct"]:.2f}%  RS peak y+: {analysis["reynolds_stress_peak_yplus"]:.1f}")
    # Negative control
    cfg180 = configs[0]
    result_nosgs = run_case(cfg180, sgs_on=False)
    dns180 = load_dns(180)
    analysis_nosgs = analyze_result(result_nosgs, dns180)
    with open(RUNS_DIR / f"{cfg180.label}_nosgs_result.json", "w") as f:
        json.dump({"result": {k: v for k, v in result_nosgs.items() if k not in ("config", "wall_units")},
                   "analysis": {k: v for k, v in analysis_nosgs.items() if "dns" not in k}}, f, indent=2, default=str)
    print(f"
Negative control RMS error: {analysis_nosgs["rms_error_y30_100_pct"]:.2f}%")
    summary = {"mission": "IN99-N03", "analyses": {k: {kk: vv for kk, vv in v.items() if "dns" not in kk and kk not in ("y_plus","u_plus","uu_plus","vv_plus","ww_plus","uv_plus")} for k, v in all_analyses.items()},
               "negative_control": {k: v for k, v in analysis_nosgs.items() if "dns" not in k}}
    with open(RUNS_DIR / "summary.json", "w") as f: json.dump(summary, f, indent=2, default=str)
    print(f"
Summary saved to {RUNS_DIR / "summary.json"}")
    return summary

if __name__ == "__main__": main()
