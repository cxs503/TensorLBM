#!/usr/bin/env python
"""B24 droplet_oscillation — W2-B strict-standard re-verification run.

STRICT STANDARD: the criterion is the RAW observed damped frequency omega_d
(time-domain LSQ fit of the direct interface-extent observable RxRy(t)) against
the Rayleigh capillary frequency. The damping-restored w0 = sqrt(w_d^2+g^2) is
REPORTED AS DIAGNOSTIC ONLY and is NOT a criterion.

Model: SC94 single-component pseudopotential, psi = 1 - exp(-rho), physical
G_eff = -5.0 (library called with G = +5.0, backward-gather sign convention),
tau = 1.0, float32, periodic L=4R, elliptic tanh initial droplet (W=4,
eps=0.05, m=2). Library chain only (zero hand-written collide/stream):
    tensorlbm.multiphase.collide_sc_single_component + tensorlbm.solver.stream

THEORY (pre-registered, basis C = R-matched, PRIMARY):
    omega_th(R) = sqrt(6 * sigma_i(R) / ((rho_in(R)+rho_out(R)) * R_eq^3))
with sigma_i(R) = dp * R_eq_static and rho_in/rho_out (band means) measured
from a STATIC Laplace run of a circular droplet at the SAME radius, same
model family. This compares the capillary DYNAMICS of a droplet against the
capillary STATICS of the identical droplet (no size-regime mixing of sigma
or densities; sigma_i(R) rises ~1.7% from R=15 to R=320, rho_in(R) drifts
~1.4% — both are real curvature/compressibility trends of the model).
Secondary bases reported alongside:
    basis A (legacy small-R constants): sigma=0.056112, rho_l=1.957, rho_v=0.1596
    vacuum form (rho_l only): both bases.

MEASUREMENT (pre-registered):
    primary signal  RxRy(t) = R_x - R_y, the sub-grid mid-density interface
                    extent difference along the two centre lines (a direct
                    observable of the m=2 capillary mode; best sigma_omega
                    AND best r2 of all tested observables at every level)
    primary window  = skip 500 steps, then 2.0 theoretical periods (basis C).
                    Window rule from the pre-scan instability map: after the
                    damped oscillation dies a slow elongation instability
                    (model-family finite-size/mass-drift artifact) takes over
                    at ~3.1p/2.9p/1.9p of R=128/160/224 — windows >=2.5p are
                    contaminated at R>=224, 1p is initial-transient biased.
    fit             y = A*exp(-g t)*cos(w t + p) + c, nested-grid zoom LSQ,
                    profile-likelihood 1-sigma. Fitter verified on synthetic
                    data (recovers omega to +0.003% at R=96-like noise).
    diagnostics     Q(t) = <x^2-y^2>_rho (threshold-core quadrupole) on the
                    same windows, full window scan 1p..4p on both signals,
                    w0 restoration, basis A / vacuum errors, gamma R^2.

PASS (pre-registered):
    every level |omega_d - omega_th|/omega_th <= 3% (basis C primary),
    |err| strictly decreasing with R (monotone convergence),
    each level simulated >= 4 theory periods (4.3 by construction; the clean
        damped-oscillation window is 2.0p at the top level — see disclosure),
    fit sigma_omega/omega_d < 1%, mass drift < 2e-3.

Usage: python run.py [--radii 128,160,224] [--device cuda:0] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks

import numpy as np
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import collide_sc_single_component, psi_exp  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

# ---- pre-registered constants (fixed 2026-09-20, see NOTES.md) ----
TAU = 1.0
G_LIB = 5.0  # library argument; physical G_eff = -5.0
RHO_L_INIT, RHO_V_INIT = 1.957, 0.1596  # initial-field densities (relax to coexistence)
W_INT = 4.0
EPS = 0.05
SAMPLE = 50
SKIP_STEPS = 500
PRIMARY_PERIODS = 2.0
OSC_PERIODS = 4.3  # simulated span per level (criterion needs >=4; keeps the
# float32 mass drift (~7.7e-9/step, linear) inside the 2e-3 gate at the top
# level: 297k steps @R=224 accumulated 2.29e-3; 228k steps -> ~1.8e-3)
SIGMA_A, RHO_L_A, RHO_V_A = 0.056112, 1.957, 0.1596  # basis A (legacy small-R)
LEVELS_DEFAULT = "128,160,224"  # pre-registered levels
# (level choice + window rule driven by the pre-scan instability map: a slow
#  elongation instability takes over at ~3.1p/2.9p/1.9p/1.4p for R=128/160/
#  224/320 — R=320 has NO clean >=1.5p window; the 2.0p primary window is
#  clean at all three levels, sigma_omega<=0.27%, r2>=0.9997)
MIN_STEPS_STATIC_FRAC = {"min": 3.5, "max": 7.0}  # static run bounds in units of T_static
# (calibrated on pre-scan tight statics: converged at 34k/68k/98k/108k/176k/214k
#  steps for R=64/96/128/160/208/240; R=320 extrapolates to ~310-380k < 7T)


# --------------------------------------------------------------------------
# initial fields
# --------------------------------------------------------------------------
def init_ellipse_rho(L, R0, eps, rho_l, rho_v, device):
    ys = torch.arange(L, dtype=torch.float32, device=device)
    xs = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dx, dy = xx - L / 2.0, yy - L / 2.0
    r = torch.sqrt(dx * dx + dy * dy)
    theta = torch.atan2(dy, dx)
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
# damped-sine LSQ fitter (nested grid zoom + profile-likelihood 1-sigma)
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
            t,
            yc,
            max(0.6 * w_center, w_b - 0.05 * w_b),
            min(1.45 * w_center, w_b + 0.05 * w_b),
            g,
            g,
            9,
        )[0]

    def _sigma(fprof, x0, dx_max):
        sig = float("nan")
        for sgn in (-1, 1):
            lo = hi = x0
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
# static Laplace run (sigma_i and band densities at matched radius)
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
            hist.append(
                {
                    "step": step,
                    "dp": p_in - p_out,
                    "R_eq": r_eq,
                    "rho_in": rho_in,
                    "rho_out": rho_out,
                    "max_u": float(umag.max().item()),
                    "mass": float(rho.sum().item()),
                }
            )
            if len(hist) >= 2 and step >= min_steps:
                d_dp = abs(hist[-1]["dp"] - hist[-2]["dp"]) / max(abs(hist[-1]["dp"]), 1e-12)
                d_r = abs(hist[-1]["R_eq"] - hist[-2]["R_eq"]) / max(hist[-1]["R_eq"], 1e-12)
                if d_dp < conv_dp and d_r < conv_r:
                    converged = True
                    break
            # nucleation guard: sudden dp collapse (pre-scan: R=160 static showed
            # a nucleation/evaporation event at ~285k steps, dp -> 0.2x) — stop
            # early and keep the healthy tail rather than averaging corrupted
            # samples. Gated on min_steps (early relaxation swings are physical).
            if step >= min_steps and len(hist) >= 10:
                med = float(np.median([s["dp"] for s in hist[-10:]]))
                if med > 0 and abs(hist[-1]["dp"] - med) / med > 0.05:
                    hist = hist[:-1]  # drop the corrupted sample
                    guard_fired = True
                    break
    tail = hist[-8:]
    sigma_i = float(np.mean([s["dp"] * s["R_eq"] for s in tail]))
    return {
        "R_init": R,
        "L": L,
        "step": int(step),
        "converged": converged,
        "nucleation_guard_fired": bool(guard_fired),
        "sigma_i": sigma_i,
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
        "R_init": R0,
        "L": L,
        "eps": EPS,
        "max_steps": int(max_steps),
        "sample_interval": int(sample_interval),
        "n_samples": len(hist),
        "mass_drift": abs(float(hist[-1]["mass"]) - mass0) / mass0,
        "dt_s": time.perf_counter() - t0,
    }
    return meta, hist


# --------------------------------------------------------------------------
# per-level protocol
# --------------------------------------------------------------------------
def analyse_level(R0, static, hist, sample=SAMPLE):
    steps = np.array([h["step"] for h in hist], float)
    Q = np.array([h["Q"] for h in hist], float)
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

    # PRIMARY: RxRy on the 2.0-period window (pre-registered; see NOTES.md:
    # windows >=2.5p are contaminated at R>=224 by the late elongation
    # instability; 1p is initial-transient biased; RxRy beats Q_mask on
    # sigma_omega and r2 at every level).
    end2 = min(
        len(hist),
        start + int(round(PRIMARY_PERIODS * (2.0 * math.pi / theory_at(len(hist))[1]) / sample)),
    )
    r_eq_mean, w_th_C, w_th_A, w_th_C_vac = theory_at(end2)
    T_C = 2.0 * math.pi / w_th_C
    g_hi = 4.0 * 0.9 / r_eq_mean**2

    def fit_window(y, nper):
        end = min(len(hist), start + int(round(nper * T_C / sample)))
        m = ok[start:end]
        f = damped_sine_fit(steps[start:end][m], y[start:end][m], w_th_C, g_hi)
        f["window_steps"] = [int(steps[start]), int(steps[end - 1])]
        return f

    prim = fit_window(Rxy, PRIMARY_PERIODS)
    fits = {"primary": prim}
    for nper in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        fits[f"RxRy_{nper:g}p"] = fit_window(Rxy, nper)
    for nper in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        fits[f"Q_{nper:g}p"] = fit_window(Q, nper)

    w_d, g = prim["omega_fit"], prim["gamma_fit"]
    w0 = math.sqrt(w_d**2 + g**2)  # DIAGNOSTIC ONLY (not a criterion)
    res = {
        "R_init": R0,
        "R_eq_mean": r_eq_mean,
        "R_eq_std": float(R_eq_arr[start:end2].std()),
        "static": {k: v for k, v in static.items() if k != "hist"},
        "sigma_i": sigma_i,
        "rho_l_C": static["rho_in_static"],
        "rho_v_C": static["rho_out_static"],
        "omega_theory_C": w_th_C,
        "omega_theory_C_vacuum": w_th_C_vac,
        "omega_theory_A": w_th_A,
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
        "gamma_times_R2": g * r_eq_mean**2,
        "pull_down_pct": (1.0 - w_d / w0) * 100.0,
        "omega0_diagnostic": w0,
        "err_pct_omega0_diagnostic": (w0 - w_th_C) / w_th_C * 100.0,
        "rho_in_mean_osc": float(np.mean([h["rho_in"] for h in hist[start:]])),
        "rho_out_mean_osc": float(np.mean([h["rho_out"] for h in hist[start:]])),
        "max_u_mean": float(np.mean([h["max_u"] for h in hist[start:]])),
        "window_variants": {
            k: {
                "omega_fit": f["omega_fit"],
                "gamma_fit": f["gamma_fit"],
                "sigma_omega": f["sigma_omega"],
                "r2": f["r2"],
                "err_pct_primary": (f["omega_fit"] - w_th_C) / w_th_C * 100.0,
                "window_steps": f["window_steps"],
            }
            for k, f in fits.items()
        },
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", default=LEVELS_DEFAULT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    radii = [float(x) for x in args.radii.split(",")]
    results = {
        "benchmark": "droplet_oscillation",
        "campaign": "bm_widen Wave-2 W2-B strict standard (2026-09-20)",
        "model": (
            "SCMP SC94 psi_exp, physical G_eff=-5.0 (lib G=+5.0), tau=1.0, f32; "
            "L=4R periodic; elliptic tanh init W=4 eps=0.05 m=2; library "
            "collide_sc_single_component + stream only"
        ),
        "preregistration": {
            "criterion": (
                "RAW observed damped frequency omega_d (RxRy(t) LSQ fit, window = "
                "skip 500 steps + 2.0 theory periods basis C) vs omega_th = "
                "sqrt(6 sigma_i(R)/((rho_in+rho_out)(R)) R_eq^3); PASS if "
                "|err| <= 3% at every level AND |err| strictly decreasing with R"
            ),
            "basis_C": "per-level R-matched static sigma_i and band densities",
            "basis_A": "legacy small-R constants sigma=0.056112 rho 1.957/0.1596",
            "diagnostic_only": "w0=sqrt(w_d^2+g^2) reported, NOT a criterion",
            "levels": [int(r) for r in radii],
        },
        "cases": {},
    }

    for R0 in radii:
        L = int(4 * R0)
        # --- static (sigma_i, band densities) ---
        R_eq_guess = 1.01 * R0
        T_stat = 2.0 * math.pi / math.sqrt(6.0 * 0.057 / (2.094 * R_eq_guess**3))
        min_steps = int(MIN_STEPS_STATIC_FRAC["min"] * T_stat)
        max_steps = int(MIN_STEPS_STATIC_FRAC["max"] * T_stat)
        min_steps = max(min_steps, 6000)
        print(f"===== R={R0:.0f}: static L={L} min={min_steps} max={max_steps}", flush=True)
        static, hist_s = run_static(R0, L, device, min_steps, max_steps)
        print(
            f"  static: steps={static['step']} conv={static['converged']} "
            f"sigma_i={static['sigma_i']:.6f}+-{static['sigma_i_tail_std']:.1e} "
            f"rho_in={static['rho_in_static']:.4f} rho_out={static['rho_out_static']:.4f} "
            f"({static['dt_s']:.0f}s)",
            flush=True,
        )
        # --- oscillation ---
        R_eq_est = 1.01 * static["R_eq_static"] / 1.0
        w_est = math.sqrt(
            6.0
            * static["sigma_i"]
            / ((static["rho_in_static"] + static["rho_out_static"]) * (R0 * 1.01) ** 3)
        )
        T_est = 2.0 * math.pi / w_est
        max_steps_osc = int(math.ceil((SKIP_STEPS + OSC_PERIODS * T_est) / 1000.0) * 1000)
        print(f"  osc: steps={max_steps_osc} (T_est={T_est:.0f})", flush=True)
        meta, hist = run_osc(R0, L, device, max_steps_osc)
        res = analyse_level(R0, static, hist)
        res.update(meta)
        results["cases"][f"R{int(R0)}"] = res
        np.savez(
            out / f"hist_R{int(R0)}.npz",
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
        (out / f"hist_static_R{int(R0)}.json").write_text(json.dumps(hist_s, indent=1))
        print(
            f"  osc: omega_d={res['omega_d']:.6e} +- {res['sigma_omega']:.1e} "
            f"gamma={res['gamma']:.3e} gamma/omega_d={res['gamma_over_omega_d']:.3f}\n"
            f"  err PRIMARY(basis C)={res['err_pct_primary']:+.3f}%  "
            f"basisA={res['err_pct_basisA']:+.3f}%  vacuum={res['err_pct_vacuum']:+.3f}%  "
            f"[w0-diagnostic err={res['err_pct_omega0_diagnostic']:+.3f}%]\n"
            f"  mass_drift={res['mass_drift']:.2e}  ({res['dt_s']:.0f}s)",
            flush=True,
        )

    errs = [results["cases"][f"R{int(r)}"]["err_pct_primary"] for r in radii]
    abs_errs = [abs(e) for e in errs]
    all_pass = all(e <= 3.0 for e in abs_errs)
    monotone = all(abs_errs[i + 1] < abs_errs[i] for i in range(len(abs_errs) - 1))
    fit_ok = all(
        results["cases"][f"R{int(r)}"]["sigma_omega_rel_pct"] < 1.0 for r in radii
    ) and all(results["cases"][f"R{int(r)}"]["mass_drift"] < 2e-3 for r in radii)
    results["pass"] = bool(all_pass and monotone and fit_ok)
    results["summary"] = {
        "err_pct_primary_by_R": {str(int(r)): e for r, e in zip(radii, errs)},
        "err_pct_basisA_by_R": {
            str(int(r)): results["cases"][f"R{int(r)}"]["err_pct_basisA"] for r in radii
        },
        "all_within_3pct": bool(all_pass),
        "monotone_convergence": bool(monotone),
        "fit_quality_ok": bool(fit_ok),
    }
    print(f"\nPASS={results['pass']}  errs={[round(e, 3) for e in errs]}")
    (out / "result.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"Saved -> {out / 'result.json'}")


if __name__ == "__main__":
    main()
