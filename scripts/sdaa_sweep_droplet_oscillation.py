#!/usr/bin/env python
"""SDAA sweep for B24 droplet_oscillation — device verification + grid convergence + criterion.

Runs the droplet_oscillation benchmark on SDAA hardware, compares against the
pre-registered CUDA baseline (result.json), and produces an evidence file.

SDAA-specific fix: atan2(0,0) returns inf on SDAA (vs 0 on CPU/CUDA), so we
guard the center point in the elliptic initialization.

Usage:
    python scripts/sdaa_sweep_droplet_oscillation.py [--radii 64,128] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure tensorlbm is on the path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.multiphase import collide_sc_single_component, psi_exp
from tensorlbm.solver import stream

# ---- pre-registered constants (identical to benchmarks/verified/droplet_oscillation/run.py) ----
TAU = 1.0
G_LIB = 5.0
RHO_L_INIT, RHO_V_INIT = 1.957, 0.1596
W_INT = 4.0
EPS = 0.05
SAMPLE = 50
SKIP_STEPS = 500
PRIMARY_PERIODS = 2.0
OSC_PERIODS = 4.3
SIGMA_A, RHO_L_A, RHO_V_A = 0.056112, 1.957, 0.1596
MIN_STEPS_STATIC_FRAC = {"min": 3.5, "max": 7.0}

# ---- baseline from existing result.json ----
BASELINE_PATH = REPO_ROOT / "benchmarks" / "verified" / "droplet_oscillation" / "result.json"


# --------------------------------------------------------------------------
# initial fields (SDAA-safe: guard atan2(0,0))
# --------------------------------------------------------------------------
def init_ellipse_rho(L, R0, eps, rho_l, rho_v, device):
    ys = torch.arange(L, dtype=torch.float32, device=device)
    xs = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dx = xx - L / 2.0
    dy = yy - L / 2.0
    # SDAA fix: atan2(0,0) returns inf on SDAA; guard the center point
    r = torch.sqrt(dx * dx + dy * dy)
    theta = torch.where(r > 0, torch.atan2(dy, dx), torch.zeros_like(r))
    r_surf = R0 * (1.0 + eps * torch.cos(2.0 * theta))
    rho = rho_v + 0.5 * (rho_l - rho_v) * (1.0 + torch.tanh((r_surf - r) / W_INT))
    return rho.clamp(min=1e-3)


def init_droplet_rho(L, R, rho_l, rho_v, device):
    ys = torch.arange(L, dtype=torch.float32, device=device)
    xs = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    rho = rho_v + 0.5 * (rho_l - rho_v) * (1.0 + torch.tanh((R - r) / W_INT))
    return rho.clamp(min=1e-3)


def eos_pressure(rho, g_eff=-5.0):
    psi = 1.0 - torch.exp(-rho)
    return rho / 3.0 + g_eff * psi * psi / 6.0


# --------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------
def _interface_extent(rho, mid, L, axis):
    line = rho[L // 2, :] if axis == 0 else rho[:, L // 2]
    cross = (line[:-1] - mid) * (line[1:] - mid) < 0
    idx = torch.nonzero(cross).flatten().tolist()
    if len(idx) < 2:
        return float("nan")
    xint = []
    for i in idx:
        r0, r1 = float(line[i]), float(line[i + 1])
        xint.append(float(i) if abs(r1 - r0) < 1e-12 else i + (mid - r0) / (r1 - r0))
    return float(max(xint) - min(xint)) / 2.0


def measure_osc(f, R_guess, xx, yy, rr, L):
    rho, ux, uy = macroscopic(f)
    umag = torch.sqrt(ux * ux + uy * uy)
    r_eq = R_guess
    mid = 0.5 * (RHO_L_INIT + RHO_V_INIT)
    rho_in = rho_out = float("nan")
    for _ in range(3):
        inside, outside = rr <= r_eq * 0.5, rr >= r_eq * 1.5
        rho_in = float(rho[inside].mean().item()) if inside.any() else float("nan")
        rho_out = float(rho[outside].mean().item()) if outside.any() else float("nan")
        mid = 0.5 * (rho_in + rho_out)
        n_liq = int((rho > mid).sum().item())
        r_eq_new = math.sqrt(n_liq / math.pi)
        if abs(r_eq_new - r_eq) < 1e-4:
            r_eq = r_eq_new
            break
        r_eq = r_eq_new
    mask = rho > mid
    rho_liq = rho[mask]
    m_liq = float(rho_liq.sum().item())
    q = float((rho_liq * (xx[mask] ** 2 - yy[mask] ** 2)).sum().item()) / m_liq
    return {
        "Q": q,
        "R_eq": r_eq,
        "R_x": _interface_extent(rho, mid, L, 0),
        "R_y": _interface_extent(rho, mid, L, 1),
        "rho_in": rho_in,
        "rho_out": rho_out,
        "max_u": float(umag.max().item()),
        "mass": float(rho.sum().item()),
    }


# --------------------------------------------------------------------------
# damped-sine LSQ fitter
# --------------------------------------------------------------------------
def _rss_linear(t, yc, w, g):
    e = np.exp(-g * t)
    A = np.column_stack([e * np.cos(w * t), e * np.sin(w * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, yc, rcond=None)
    r = A @ coef - yc
    return float(r @ r), coef


def _zoom_argmin(t, yc, w_lo, w_hi, g_lo, g_hi, n):
    ws = np.linspace(w_lo, w_hi, n)
    gs = np.linspace(g_lo, g_hi, n)
    best = (math.inf, 0.0, 0.0)
    for w in ws:
        cw, sw = np.cos(w * t), np.sin(w * t)
        for g in gs:
            e = np.exp(-g * t)
            A = np.column_stack([e * cw, e * sw, np.ones_like(t)])
            coef, *_ = np.linalg.lstsq(A, yc, rcond=None)
            r = A @ coef - yc
            rss = float(r @ r)
            if rss < best[0]:
                best = (rss, w, g)
    return best


def damped_sine_fit(t, y, w_center, g_hi, n_grid=41, n_zoom=4):
    t = np.asarray(t, float)
    t = t - t[0]
    yc = np.asarray(y, float) - float(np.mean(y))
    N = len(t)
    w_lo, w_hi = 0.6 * w_center, 1.45 * w_center
    g_lo = 0.0
    if g_hi <= 0:
        g_hi = 1e-3
    rss_b, w_b, g_b = math.inf, w_center, 0.0
    for _ in range(n_zoom):
        r, w, g = _zoom_argmin(t, yc, w_lo, w_hi, g_lo, g_hi, n_grid)[:3]
        if r < rss_b:
            rss_b, w_b, g_b = r, w, g
        dw = (w_hi - w_lo) / (n_grid - 1)
        dg = (g_hi - g_lo) / (n_grid - 1)
        w_lo, w_hi = max(0.6 * w_center, w_b - 2 * dw), min(1.45 * w_center, w_b + 2 * dw)
        g_lo, g_hi = max(0.0, g_b - 2 * dg), g_b + 2 * dg
    rss_min, _ = _rss_linear(t, yc, w_b, g_b)
    thr = rss_min * (1.0 + math.sqrt(2.0 / N))

    def prof_w(w):
        return _zoom_argmin(
            t, yc, w, w, max(0.0, g_b - 0.25 * g_b - 1e-6), g_b + 0.25 * g_b + 1e-6, 9
        )[0]

    def prof_g(g):
        return _zoom_argmin(
            t, yc,
            max(0.6 * w_center, w_b - 0.05 * w_b),
            min(1.45 * w_center, w_b + 0.05 * w_b),
            g, g, 9,
        )[0]

    def _sigma(fprof, x0, dx_max):
        sig = float("nan")
        for sgn in (-1, 1):
            hi = x0
            step = 1e-4 * x0 + 1e-12
            n_it = 0
            while fprof(hi) < thr and n_it < 60 and abs(hi - x0) < dx_max:
                step *= 1.6
                hi = x0 + sgn * step
                n_it += 1
            if fprof(hi) >= thr:
                a, b = x0, hi
                for _ in range(50):
                    m = 0.5 * (a + b)
                    a, b = (m, b) if fprof(m) < thr else (a, m)
                d = abs(0.5 * (a + b) - x0)
                sig = d if math.isnan(sig) else max(sig, d)
        return sig

    sigma_w = _sigma(prof_w, w_b, 0.3 * w_center)
    sigma_g = _sigma(prof_g, g_b, g_hi)
    e = np.exp(-g_b * t)
    A = np.column_stack([e * np.cos(w_b * t), e * np.sin(w_b * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, yc, rcond=None)
    rss_f = float(np.sum((A @ coef - yc) ** 2))
    return {
        "omega_fit": float(w_b),
        "gamma_fit": float(g_b),
        "sigma_omega": float(sigma_w),
        "sigma_gamma": float(sigma_g),
        "amp_fit": float(math.hypot(coef[0], coef[1])),
        "offset_fit": float(coef[2] + float(np.mean(y))),
        "rss": rss_f,
        "r2": 1.0 - rss_f / float(np.sum(yc**2)),
        "n_points": int(N),
    }


# --------------------------------------------------------------------------
# static Laplace run
# --------------------------------------------------------------------------
def run_static(R, L, device, min_steps, max_steps, sample_interval=2000, conv_dp=3e-5, conv_r=2e-3):
    t0 = time.perf_counter()
    rho0 = init_droplet_rho(L, R, RHO_L_INIT, RHO_V_INIT, device)
    mass0 = float(rho0.sum().item())
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    ys = torch.arange(L, dtype=torch.float32, device=device)
    xs = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    rr = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)

    hist = []
    step = 0
    converged = False
    guard_fired = False
    for step in range(1, max_steps + 1):
        f = stream(collide_sc_single_component(f, G=G_LIB, tau=TAU, psi_fn=psi_exp))
        if step % sample_interval == 0:
            rho, ux, uy = macroscopic(f)
            if float(rho.min().item()) < 0.0 or not torch.isfinite(rho).all():
                raise RuntimeError(f"static R={R}: NaN/negative rho at step {step}")
            umag = torch.sqrt(ux * ux + uy * uy)
            p = eos_pressure(rho)
            r_eq, mid = R, 0.5 * (RHO_L_INIT + RHO_V_INIT)
            rho_in = rho_out = float("nan")
            for _ in range(3):
                inside, outside = rr <= r_eq * 0.5, rr >= r_eq * 1.5
                rho_in = float(rho[inside].mean().item()) if inside.any() else float("nan")
                rho_out = float(rho[outside].mean().item()) if outside.any() else float("nan")
                mid = 0.5 * (rho_in + rho_out)
                n_liq = int((rho > mid).sum().item())
                r_eq_new = math.sqrt(n_liq / math.pi)
                if abs(r_eq_new - r_eq) < 1e-4:
                    r_eq = r_eq_new
                    break
                r_eq = r_eq_new
            inside, outside = rr <= r_eq * 0.5, rr >= r_eq * 1.5
            p_in = float(p[inside].mean().item()) if inside.any() else float("nan")
            p_out = float(p[outside].mean().item()) if outside.any() else float("nan")
            hist.append({
                "step": step, "dp": p_in - p_out, "R_eq": r_eq,
                "rho_in": rho_in, "rho_out": rho_out,
                "max_u": float(umag.max().item()), "mass": float(rho.sum().item()),
            })
            if len(hist) >= 2 and step >= min_steps:
                d_dp = abs(hist[-1]["dp"] - hist[-2]["dp"]) / max(abs(hist[-1]["dp"]), 1e-12)
                d_r = abs(hist[-1]["R_eq"] - hist[-2]["R_eq"]) / max(hist[-1]["R_eq"], 1e-12)
                if d_dp < conv_dp and d_r < conv_r:
                    converged = True
                    break
            if step >= min_steps and len(hist) >= 10:
                med = float(np.median([s["dp"] for s in hist[-10:]]))
                if med > 0 and abs(hist[-1]["dp"] - med) / med > 0.05:
                    hist = hist[:-1]
                    guard_fired = True
                    break
    tail = hist[-8:]
    sigma_i = float(np.mean([s["dp"] * s["R_eq"] for s in tail]))
    return {
        "R_init": R, "L": L, "step": int(step), "converged": converged,
        "nucleation_guard_fired": bool(guard_fired), "sigma_i": sigma_i,
        "sigma_i_tail_std": float(np.std([s["dp"] * s["R_eq"] for s in tail])),
        "dp_tail_mean": float(np.mean([s["dp"] for s in tail])),
        "R_eq_static": float(np.mean([s["R_eq"] for s in tail])),
        "rho_in_static": float(np.mean([s["rho_in"] for s in tail])),
        "rho_out_static": float(np.mean([s["rho_out"] for s in tail])),
        "max_u_static": float(np.mean([s["max_u"] for s in tail])),
        "mass_drift_static": abs(float(tail[-1]["mass"]) - mass0) / mass0,
        "dt_s": time.perf_counter() - t0,
    }, hist


# --------------------------------------------------------------------------
# oscillation run
# --------------------------------------------------------------------------
def run_osc(R0, L, device, max_steps, sample_interval=SAMPLE):
    t0 = time.perf_counter()
    rho0 = init_ellipse_rho(L, R0, EPS, RHO_L_INIT, RHO_V_INIT, device)
    mass0 = float(rho0.sum().item())
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    ys = torch.arange(L, dtype=torch.float32, device=device)
    xs = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx, yy = xx - L / 2.0, yy - L / 2.0
    rr = torch.sqrt(xx * xx + yy * yy)
    hist = []
    for step in range(1, max_steps + 1):
        f = stream(collide_sc_single_component(f, G=G_LIB, tau=TAU, psi_fn=psi_exp))
        if step % sample_interval == 0:
            rho_cur = f.sum(dim=0)
            if float(rho_cur.min().item()) < 0.0 or not torch.isfinite(rho_cur).all():
                raise RuntimeError(f"osc R={R0}: NaN/negative rho at step {step}")
            m = measure_osc(f, R0, xx, yy, rr, L)
            m["step"] = step
            hist.append(m)
    meta = {
        "R_init": R0, "L": L, "eps": EPS,
        "max_steps": int(max_steps), "sample_interval": int(sample_interval),
        "n_samples": len(hist),
        "mass_drift": abs(float(hist[-1]["mass"]) - mass0) / mass0,
        "dt_s": time.perf_counter() - t0,
    }
    return meta, hist


# --------------------------------------------------------------------------
# analysis (simplified: primary window only)
# --------------------------------------------------------------------------
def analyse_level(R0, static, hist, sample=SAMPLE):
    steps = np.array([h["step"] for h in hist], float)
    R_eq_arr = np.array([h["R_eq"] for h in hist], float)
    Rxy = np.array([h["R_x"] for h in hist], float) - np.array([h["R_y"] for h in hist], float)
    ok = np.isfinite(Rxy)
    start = int(SKIP_STEPS // sample)

    sigma_i = static["sigma_i"]
    rho_lv = static["rho_in_static"] + static["rho_out_static"]

    def theory_at(end):
        r_eq = float(R_eq_arr[start:end].mean())
        w_C = math.sqrt(6.0 * sigma_i / (rho_lv * r_eq**3))
        w_A = math.sqrt(6.0 * SIGMA_A / ((RHO_L_A + RHO_V_A) * r_eq**3))
        w_v = math.sqrt(6.0 * sigma_i / (static["rho_in_static"] * r_eq**3))
        return r_eq, w_C, w_A, w_v

    end2 = min(len(hist), start + int(round(PRIMARY_PERIODS * (2.0 * math.pi / theory_at(len(hist))[1]) / sample)))
    r_eq_mean, w_th_C, w_th_A, w_th_C_vac = theory_at(end2)
    T_C = 2.0 * math.pi / w_th_C
    g_hi = 4.0 * 0.9 / r_eq_mean**2

    end = min(len(hist), start + int(round(PRIMARY_PERIODS * T_C / sample)))
    m = ok[start:end]
    prim = damped_sine_fit(steps[start:end][m], Rxy[start:end][m], w_th_C, g_hi)
    prim["window_steps"] = [int(steps[start]), int(steps[end - 1])]

    w_d, g = prim["omega_fit"], prim["gamma_fit"]
    w0 = math.sqrt(w_d**2 + g**2)
    return {
        "R_init": R0,
        "R_eq_mean": r_eq_mean,
        "sigma_i": sigma_i,
        "rho_l_C": static["rho_in_static"],
        "rho_v_C": static["rho_out_static"],
        "omega_theory_C": w_th_C,
        "omega_theory_A": w_th_A,
        "omega_theory_C_vacuum": w_th_C_vac,
        "T_theory_C": T_C,
        "omega_d": w_d,
        "gamma": g,
        "sigma_omega": prim["sigma_omega"],
        "sigma_omega_rel_pct": prim["sigma_omega"] / w_d * 100.0,
        "fit_r2": prim["r2"],
        "fit_window_steps": prim["window_steps"],
        "err_pct_primary": (w_d - w_th_C) / w_th_C * 100.0,
        "err_pct_basisA": (w_d - w_th_A) / w_th_A * 100.0,
        "err_pct_vacuum": (w_d - w_th_C_vac) / w_th_C_vac * 100.0,
        "gamma_over_omega_d": g / w_d,
        "pull_down_pct": (1.0 - w_d / w0) * 100.0,
        "omega0_diagnostic": w0,
        "err_pct_omega0_diagnostic": (w0 - w_th_C) / w_th_C * 100.0,
        "rho_in_mean_osc": float(np.mean([h["rho_in"] for h in hist[start:]])),
        "rho_out_mean_osc": float(np.mean([h["rho_out"] for h in hist[start:]])),
        "max_u_mean": float(np.mean([h["max_u"] for h in hist[start:]])),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", default="64")
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts"))
    args = ap.parse_args()
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Device verification ----
    print("=" * 60)
    print("STEP 1: SDAA Device Verification")
    print("=" * 60)
    sdaa_available = torch.sdaa.is_available()
    print(f"  torch.sdaa.is_available(): {sdaa_available}")
    if not sdaa_available:
        print("  FATAL: SDAA not available — aborting.")
        sys.exit(1)

    sdaa_count = torch.sdaa.device_count() if hasattr(torch.sdaa, "device_count") else 0
    print(f"  SDAA device count: {sdaa_count}")

    sdaa_name = "unknown"
    sdaa_memory = "unknown"
    if hasattr(torch.sdaa, "get_device_name"):
        sdaa_name = torch.sdaa.get_device_name(0)
        print(f"  SDAA device 0 name: {sdaa_name}")
    if hasattr(torch.sdaa, "get_device_properties"):
        props = torch.sdaa.get_device_properties(0)
        sdaa_memory = str(props.total_memory) if hasattr(props, "total_memory") else "unknown"
        print(f"  SDAA device 0 memory: {sdaa_memory}")

    # Quick tensor test on SDAA
    x = torch.randn(100, 100, device=device)
    y = torch.mm(x, x)
    print(f"  SDAA tensor test (100x100 matmul): OK, result dtype={y.dtype}")
    print()

    # ---- Load baseline ----
    baseline = {}
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
        print(f"  Baseline loaded from {BASELINE_PATH}")
        print(f"  Baseline cases: {list(baseline.get('cases', {}).keys())}")
    else:
        print(f"  WARNING: Baseline not found at {BASELINE_PATH}")
    print()

    # ---- Step 2: Run on SDAA ----
    radii = [float(x) for x in args.radii.split(",")]
    results = {
        "benchmark": "droplet_oscillation",
        "campaign": "SDAA sweep — droplet_oscillation verification",
        "device": {
            "type": "sdaa",
            "available": sdaa_available,
            "device_count": sdaa_count,
            "device_name": sdaa_name,
            "device_memory": sdaa_memory,
            "device_str": str(device),
        },
        "model": (
            "SCMP SC94 psi_exp, physical G_eff=-5.0 (lib G=+5.0), tau=1.0, f32; "
            "L=4R periodic; elliptic tanh init W=4 eps=0.05 m=2"
        ),
        "sdaa_fix": "atan2(0,0) returns inf on SDAA; guarded with torch.where(r>0, atan2(dy,dx), 0)",
        "cases": {},
    }

    print("=" * 60)
    print("STEP 2: SDAA Grid Convergence Run")
    print("=" * 60)

    for R0 in radii:
        L = int(4 * R0)
        R_eq_guess = 1.01 * R0
        T_stat = 2.0 * math.pi / math.sqrt(6.0 * 0.057 / (2.094 * R_eq_guess**3))
        min_steps = int(MIN_STEPS_STATIC_FRAC["min"] * T_stat)
        max_steps = int(MIN_STEPS_STATIC_FRAC["max"] * T_stat)
        min_steps = max(min_steps, 6000)

        print(f"\n--- R={R0:.0f}: static L={L} ---", flush=True)
        static, hist_s = run_static(R0, L, device, min_steps, max_steps)
        print(
            f"  static: steps={static['step']} conv={static['converged']} "
            f"sigma_i={static['sigma_i']:.6f} "
            f"rho_in={static['rho_in_static']:.4f} rho_out={static['rho_out_static']:.4f} "
            f"({static['dt_s']:.0f}s)",
            flush=True,
        )

        R_eq_est = 1.01 * static["R_eq_static"]
        w_est = math.sqrt(
            6.0 * static["sigma_i"]
            / ((static["rho_in_static"] + static["rho_out_static"]) * (R0 * 1.01) ** 3)
        )
        T_est = 2.0 * math.pi / w_est
        max_steps_osc = int(math.ceil((SKIP_STEPS + OSC_PERIODS * T_est) / 1000.0) * 1000)
        print(f"  osc: steps={max_steps_osc} (T_est={T_est:.0f})", flush=True)
        meta, hist = run_osc(R0, L, device, max_steps_osc)
        res = analyse_level(R0, static, hist)
        res.update(meta)
        results["cases"][f"R{int(R0)}"] = res

        # Save time series
        np.savez(
            out / f"sdaa_hist_R{int(R0)}.npz",
            step=np.array([h["step"] for h in hist]),
            Q=np.array([h["Q"] for h in hist]),
            R_eq=np.array([h["R_eq"] for h in hist]),
            R_x=np.array([h["R_x"] for h in hist]),
            R_y=np.array([h["R_y"] for h in hist]),
            rho_in=np.array([h["rho_in"] for h in hist]),
            rho_out=np.array([h["rho_out"] for h in hist]),
            max_u=np.array([h["max_u"] for h in hist]),
            mass=np.array([h["mass"] for h in hist]),
        )

        print(
            f"  osc: omega_d={res['omega_d']:.6e} +- {res['sigma_omega']:.1e} "
            f"gamma={res['gamma']:.3e} gamma/omega_d={res['gamma_over_omega_d']:.3f}\n"
            f"  err PRIMARY(basis C)={res['err_pct_primary']:+.3f}%  "
            f"basisA={res['err_pct_basisA']:+.3f}%  vacuum={res['err_pct_vacuum']:+.3f}%\n"
            f"  mass_drift={res['mass_drift']:.2e}  ({res['dt_s']:.0f}s)",
            flush=True,
        )

    # ---- Step 3: Criterion verification ----
    print()
    print("=" * 60)
    print("STEP 3: Criterion Verification (|err| < 3%)")
    print("=" * 60)

    errs = [results["cases"][f"R{int(r)}"]["err_pct_primary"] for r in radii]
    abs_errs = [abs(e) for e in errs]
    all_pass = all(e <= 3.0 for e in abs_errs)
    monotone = all(abs_errs[i + 1] < abs_errs[i] for i in range(len(abs_errs) - 1)) if len(abs_errs) > 1 else True
    fit_ok = all(
        results["cases"][f"R{int(r)}"]["sigma_omega_rel_pct"] < 1.0 for r in radii
    ) and all(results["cases"][f"R{int(r)}"]["mass_drift"] < 2e-3 for r in radii)

    results["pass"] = bool(all_pass and monotone and fit_ok)
    results["summary"] = {
        "err_pct_primary_by_R": {str(int(r)): e for r, e in zip(radii, errs)},
        "all_within_3pct": bool(all_pass),
        "monotone_convergence": bool(monotone),
        "fit_quality_ok": bool(fit_ok),
    }

    for r, e in zip(radii, errs):
        status = "PASS" if abs(e) <= 3.0 else "FAIL"
        print(f"  R={int(r):4d}: err = {e:+.4f}%  |err| = {abs(e):.4f}%  [{status}]")
    print(f"\n  All within 3%: {all_pass}")
    print(f"  Monotone convergence: {monotone}")
    print(f"  Fit quality OK: {fit_ok}")
    print(f"  OVERALL: {'PASS' if results['pass'] else 'FAIL'}")

    # ---- Baseline comparison ----
    if baseline.get("cases"):
        print("\n  Baseline comparison:")
        for r in radii:
            key = f"R{int(r)}"
            sdaa_err = results["cases"].get(key, {}).get("err_pct_primary", None)
            bl_err = baseline.get("cases", {}).get(key, {}).get("err_pct_primary", None)
            if sdaa_err is not None and bl_err is not None:
                delta = abs(sdaa_err - bl_err)
                print(f"    {key}: SDAA err={sdaa_err:+.4f}%  baseline err={bl_err:+.4f}%  delta={delta:.4f}%")
            elif sdaa_err is not None:
                print(f"    {key}: SDAA err={sdaa_err:+.4f}%  (no baseline for this R)")

    # ---- Save evidence ----
    evidence_path = out / "sdaa_droplet_oscillation_evidence.json"
    evidence_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"\nEvidence saved -> {evidence_path}")

    return results


if __name__ == "__main__":
    main()
