#!/usr/bin/env python
"""Taylor-Aris shear dispersion benchmark - plane slit, passive scalar (W3-A).

Physics
-------
Fully-developed plane Poiseuille flow between parallel plates (slit height H)
driven by a constant streamwise body force in an x-periodic domain, plus a
passive scalar (tracer) line cloud uniform across the gap.  Taylor (1953) /
Aris (1956) exact slit result:

    Deff / D = 1 + Pe^2 / 210,       Pe = u_mean * H / D

Measurement (preregistered in NOTES.md before the scan):
cross-section averaged concentration Cbar(x,t) (mean over the fluid rows),
longitudinal variance sigma_x^2(t) via wrapped/circular moments, linear fit
over t in [2, 3] H^2/D (skipping the initial transient), Deff_sim = slope/2.
No extrapolation, no correction factors.

Library entries (no handwritten kernels):
    tensorlbm.solver.collide_bgk / stream               (D2Q9, periodic)
    tensorlbm.thermal.pre_streaming_bounce_back          (velocity walls)
    tensorlbm.turbulent_channel._apply_body_force_2d     (Guo/Luo force)
    tensorlbm.thermal.temperature_equilibrium / temperature_collision /
               temperature_stream                         (D2Q5 passive scalar,
                                                         buoyancy NOT applied)
    tensorlbm.d2q9.equilibrium / macroscopic

Inline boundary recipe (preregistered + disclosed in NOTES.md A.8):
zero-flux scalar walls via unwrapped post-stream reflection at the first/last
FLUID rows, with the two wall rows kept as zero "dead" wrap pads.  The source
slices are this wall's own outgoing populations retrieved from their periodic
wrap landing positions - NOT the same-live-row post-stream neighbour that
constitutes the W2-A fake-sink defect.

Usage
-----
    run.py synthetic                                   (V3 estimator check)
    run.py profile  --pe 30 --H 64 [--device cuda]     (V1 steady profile)
    run.py diffusion [--device cuda]                   (V4 pure diffusion)
    run.py case  --pe 20 --H 64 --out case.json        (single dispersion run)
    run.py scan   --out-dir .                          (formal scan, 6 cases)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))  # <repo>/src

import numpy as np
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream
from tensorlbm.thermal import (
    pre_streaming_bounce_back,
    temperature_collision,
    temperature_equilibrium,
    temperature_stream,
)
from tensorlbm.turbulent_channel import _apply_body_force_2d

# ---- preregistered constants (NOTES.md section A; do not touch after) -----
TAU = 0.9  # momentum BGK relaxation
TAU_T = 0.8  # scalar BGK relaxation
NU = (TAU - 0.5) / 3.0  # 0.133333... kinematic viscosity
ALPHA = (TAU_T - 0.5) / 3.0  # 0.1 scalar diffusivity D
K1 = 2.0  # fit window start [H^2/alpha]
K2 = 3.0  # fit window end   [H^2/alpha]
K_SAMPLE_FROM = 1.0  # first sample     [H^2/alpha]
SAMPLE_DT = 0.01  # sample interval  [H^2/alpha]
SIGMA_FRAC = 0.25  # require sigma(t2) <= L/4
MARGIN = 1.15  # domain-length safety factor
SETTLE = 0.5  # flow-only settle [H^2/nu]
A_AMP = 0.05  # scalar cloud amplitude
SIGMA0 = 3.0  # scalar cloud initial sigma [nodes]
PE_LIST = (10, 20, 30)
H_LIST = (64, 128)
TOL_PCT = 3.0
# validation gates (preregistered)
V1_L2_MAX = 0.01
V1_PTW_MAX = 0.02
V1_UMEAN_MAX = 0.02
V2_MASS_MAX = 0.025  # revised pre-scan (NOTES.md B): raw fp32 global ratchet of
# the library collision (~3.3e-8/step, uniform rescale, cancels in normalized
# moments).  Wall-leak detection is carried by the stage-exactness audit
# (val_mass_probe.txt: stream & wall-BC stages contribute exactly 0.0, dead
# rows exactly 0) + the V4 slope gate.
V3_BIAS_MAX = 0.003
V4_DEFF_MAX = 0.01
V4_UMAX_ABS = 1e-6


def domain_multiple(pe: float) -> int:
    """nx = m*H with sigma(t2) <= SIGMA_FRAC*L and MARGIN headroom."""
    sigma_over_h = math.sqrt(2.0 * K2 * (1.0 + pe * pe / 210.0))
    return math.ceil(MARGIN * sigma_over_h / SIGMA_FRAC)


def wrapped_sigma2(cbar: np.ndarray) -> tuple[float, float, float]:
    """Wrapped/circular longitudinal variance, centre, R (all fp64)."""
    nx = cbar.size
    k = 2.0 * math.pi / nx
    x = np.arange(nx, dtype=np.float64)
    z = cbar.sum()
    s = (cbar * np.exp(1j * k * x)).sum()
    r = float(abs(s) / z)
    sigma2 = float(-2.0 * math.log(r) / (k * k))
    xc = float((np.angle(s) / k) % float(nx))
    return sigma2, xc, r


def naive_sigma2(cbar: np.ndarray, xc: float) -> float:
    """Naive central second moment about the circular centre (cross-check)."""
    nx = cbar.size
    x = np.arange(nx, dtype=np.float64)
    d = (x - xc + nx / 2.0) % nx - nx / 2.0
    return float((cbar * d * d).sum() / cbar.sum())


def scalar_wall_reflect(g: torch.Tensor) -> torch.Tensor:
    """Zero-flux scalar walls at y = 0.5 / ny-1.5 (unwrapped post-stream form).

    Rows 0 / ny-1 are zero 'dead' rows (pure periodic wrap pads; the scalar
    collision keeps them zero since g_eq(T=0) = 0).  After temperature_stream,
    this wall's own outgoing populations sit on the dead rows:

        g[4, 0, :]    = post-collision g[4, 1, :]  that left fluid row 1
                        downward and wrapped around onto dead row 0
        g[3, ny-1, :] = post-collision g[3, ny-2, :] that left fluid row ny-2
                        upward and wrapped around onto dead row ny-1

    Retrieve them, zero the dead rows, reflect back into the first/last fluid
    row (half-way bounce, zero-flux plane coincident with the momentum no-slip
    planes at y = 0.5 / ny-1.5).  The overwritten slots g[3,1,:] / g[4,ny-2,:]
    streamed in from the zero dead rows, so this is an exact relocation:
    sum(g) is conserved bit-for-bit up to fp32 placement.  The forbidden W2-A
    fake-sink form (same live-row post-stream neighbour, e.g. g[3,1,:]=g[4,1,:])
    is NOT used; the superficially similar index pair here reads a dead pad row
    whose only content is this wall's own wrapped outgoing population.
    """
    esc_b = g[4, 0, :].clone()
    esc_t = g[3, -1, :].clone()
    g[:, 0, :] = 0.0
    g[:, -1, :] = 0.0
    g[3, 1, :] = esc_b
    g[4, -2, :] = esc_t
    return g


def run_case(
    pe: float,
    H: int,
    device: torch.device,
    mode: str = "dispersion",
    out_json: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """One case.  mode: 'dispersion' (formal), 'diffusion' (V4), 'profile' (V1).

    dtype is a POST-SCAN diagnostic knob (fp64 root-cause control); the formal
    scan used the preregistered fp32.
    """
    torch.manual_seed(0)
    t_begin = time.time()
    ny = H + 2
    m = domain_multiple(pe)
    nx = m * H
    u_mean_target = pe * ALPHA / H
    a_force = 12.0 * NU * u_mean_target / (H * H)
    u_max = 1.5 * u_mean_target
    ma = u_max / math.sqrt(1.0 / 3.0)

    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True

    # ---- flow initial condition: analytic parabola on fluid rows ------------
    ux0 = torch.zeros((ny, nx), dtype=dtype, device=device)
    if mode != "diffusion":
        yhat = (torch.arange(1, ny - 1, dtype=torch.float64, device=device) - 0.5) / H
        prof0 = (6.0 * u_mean_target * yhat * (1.0 - yhat)).to(dtype)
        ux0[1 : ny - 1, :] = prof0.unsqueeze(1)
    f = equilibrium(torch.ones((ny, nx), dtype=dtype, device=device), ux0, torch.zeros_like(ux0))
    del ux0

    settle_steps = int(round(SETTLE * H * H / NU)) if mode != "diffusion" else 0
    t1 = int(round(K1 * H * H / ALPHA))
    t2 = int(round(K2 * H * H / ALPHA))
    dt_sample = max(1, int(round(SAMPLE_DT * H * H / ALPHA)))
    t_sample0 = int(round(K_SAMPLE_FROM * H * H / ALPHA))
    if mode == "diffusion":  # preregistered V4 window [1, 2] H^2/alpha
        a_force = 0.0
        t1 = int(round(1.0 * H * H / ALPHA))
        t2 = int(round(2.0 * H * H / ALPHA))

    # ---- flow-only settle ---------------------------------------------------
    t0 = time.time()
    for _ in range(settle_steps):
        f_pre = f
        f = collide_bgk(f, TAU)
        f = pre_streaming_bounce_back(f_pre, f, wall)
        f = _apply_body_force_2d(f, a_force)
        f = stream(f)
    settle_s = time.time() - t0

    # ---- V1 steady profile check (fluid rows vs analytic parabola) ----------
    _, ux, _ = macroscopic(f)
    u_prof = ux.mean(dim=1)[1 : ny - 1].double().cpu().numpy()
    yy = (np.arange(1, ny - 1, dtype=np.float64) - 0.5) / H
    u_ana = 6.0 * u_mean_target * yy * (1.0 - yy)
    v1_l2 = float(np.linalg.norm(u_prof - u_ana) / np.linalg.norm(u_ana))
    v1_ptw = float(np.max(np.abs(u_prof - u_ana)) / u_max)
    u_mean_settle = float(ux[1 : ny - 1, :].mean())
    if mode == "profile":
        res = {
            "mode": "profile",
            "pe_target": pe,
            "H": H,
            "ny": ny,
            "nx": nx,
            "tau": TAU,
            "tau_T": TAU_T,
            "nu": NU,
            "alpha": ALPHA,
            "u_mean_target": u_mean_target,
            "u_mean_measured": u_mean_settle,
            "u_mean_rel_diff": abs(u_mean_settle / u_mean_target - 1.0),
            "a_force": 12.0 * NU * u_mean_target / (H * H),
            "Ma": ma,
            "settle_steps": settle_steps,
            "v1_l2": v1_l2,
            "v1_ptw": v1_ptw,
            "v1_pass": bool(
                v1_l2 <= V1_L2_MAX
                and v1_ptw <= V1_PTW_MAX
                and abs(u_mean_settle / u_mean_target - 1.0) <= V1_UMEAN_MAX
            ),
            "u_profile_num": u_prof.tolist(),
            "u_profile_ana": u_ana.tolist(),
            "elapsed_s": round(time.time() - t_begin, 1),
        }
        if out_json:
            Path(out_json).write_text(json.dumps(res, indent=2))
        print(
            f"[profile pe={pe} H={H}] l2={v1_l2 * 100:.3f}% ptw={v1_ptw * 100:.3f}% "
            f"u_mean {u_mean_settle:.6f} vs {u_mean_target:.6f} "
            f"({abs(u_mean_settle / u_mean_target - 1.0) * 100:.3f}%) pass={res['v1_pass']}",
            flush=True,
        )
        return res

    # ---- scalar initial condition -------------------------------------------
    x = np.arange(nx, dtype=np.float64)
    blob_np = A_AMP * np.exp(-((x - nx / 4.0) ** 2) / (2.0 * SIGMA0**2))
    T0 = torch.zeros((ny, nx), dtype=dtype, device=device)
    T0[1 : ny - 1, :] = (
        torch.from_numpy(blob_np.astype(np.float32 if dtype == torch.float32 else np.float64))
        .to(device)
        .unsqueeze(0)
    )
    g = temperature_equilibrium(T0, ux, torch.zeros_like(ux))
    del T0
    mass0 = float(g.double().sum().item())
    f_mass0 = float(f.double().sum().item())

    # ---- coupled transport loop ----------------------------------------------
    samples: list[dict] = []
    t0 = time.time()
    n_steps = t2 if mode != "profile" else 0
    for step in range(1, n_steps + 1):
        _, ux, uy = macroscopic(f)
        g = temperature_collision(g, TAU_T, ux, uy)
        g = temperature_stream(g)
        g = scalar_wall_reflect(g)
        f_pre = f
        f = collide_bgk(f, TAU)
        f = pre_streaming_bounce_back(f_pre, f, wall)
        f = _apply_body_force_2d(f, a_force)
        f = stream(f)
        if step >= t_sample0 and (step - t_sample0) % dt_sample == 0:
            T_field = g.sum(dim=0)
            cbar = T_field[1 : ny - 1, :].mean(dim=0).double().cpu().numpy()
            sigma2, xc, r = wrapped_sigma2(cbar)
            um = float(ux[1 : ny - 1, :].mean())
            uu = float(((ux[1 : ny - 1, :] - um) ** 2).mean())
            samples.append(
                {
                    "step": step,
                    "sigma2": sigma2,
                    "sigma2_naive": naive_sigma2(cbar, xc),
                    "xc": xc,
                    "R": r,
                    "u_mean": um,
                    "u_abs_max": float(ux[1 : ny - 1, :].abs().max()),
                    "uu_fluct": uu,
                    "mass": float(g.double().sum().item()),
                    "f_mass": float(f.double().sum().item()),
                }
            )
    elapsed = time.time() - t0

    # ---- analysis (fp64) ------------------------------------------------------
    keys = (
        "step",
        "sigma2",
        "sigma2_naive",
        "xc",
        "R",
        "u_mean",
        "u_abs_max",
        "uu_fluct",
        "mass",
        "f_mass",
    )
    sarr = {k: np.array([s[k] for s in samples], dtype=np.float64) for k in keys}
    tst = sarr["step"]

    def _fit(lo: int, hi: int) -> tuple[float, int]:
        sel = (tst >= lo) & (tst <= hi)
        slope = float(np.polyfit(tst[sel], sarr["sigma2"][sel], 1)[0])
        return slope, int(sel.sum())

    slope, npts = _fit(t1, t2)
    deff_sim = slope / 2.0
    win = (tst >= t1) & (tst <= t2)
    u_mean_win = float(sarr["u_mean"][win].mean())
    pe_sim = u_mean_win * H / ALPHA
    deff_theory = ALPHA * (1.0 + pe_sim**2 / 210.0)
    err = abs(deff_sim / deff_theory - 1.0)
    sens: dict[str, dict] = {}
    for lo, hi, tag in ((1.5, 3.0, "w1p5_3"), (2.0, 2.5, "w2_2p5"), (2.5, 3.0, "w2p5_3")):
        hi_step = int(round(hi * H * H / ALPHA))
        if hi_step > t2:  # e.g. diffusion mode has a shorter simulated range
            continue
        sl, n_ = _fit(int(round(lo * H * H / ALPHA)), hi_step)
        sens[tag] = {
            "deff": sl / 2.0,
            "err_vs_prereg_theory": abs(sl / 2.0 / deff_theory - 1.0),
            "n": n_,
        }
    mass_drift = float(np.max(np.abs(sarr["mass"] - mass0)) / mass0)
    f_mass_drift = float(np.max(np.abs(sarr["f_mass"] - f_mass0)) / f_mass0)
    # drift-based u_mean cross-check (unwrap xc first)
    xc_unw = np.unwrap(sarr["xc"] * 2.0 * math.pi / nx) * nx / (2.0 * math.pi)
    drift_u = float(np.polyfit(tst[win], xc_unw[win], 1)[0])
    # finite check
    finite = bool(torch.isfinite(f).all().item() and torch.isfinite(g).all().item())

    res = {
        "mode": mode,
        "case": "taylor_aris",
        "dtype": str(dtype).replace("torch.", ""),
        "pe_target": pe,
        "H": H,
        "ny": ny,
        "nx": nx,
        "domain_multiple": m,
        "tau": TAU,
        "tau_T": TAU_T,
        "nu": NU,
        "alpha": ALPHA,
        "u_mean_target": u_mean_target,
        "a_force": a_force,
        "Ma": ma,
        "settle_steps": settle_steps,
        "t1": t1,
        "t2": t2,
        "dt_sample": dt_sample,
        "window_samples": npts,
        "v1_l2": v1_l2,
        "v1_ptw": v1_ptw,
        "u_mean_measured_window": u_mean_win,
        "u_mean_rel_diff": abs(u_mean_win / u_mean_target - 1.0),
        "u_mean_from_drift": drift_u,
        "uu_fluct_over_umean2": float((sarr["uu_fluct"][win] / (sarr["u_mean"][win] ** 2)).mean()),
        "pe_sim": pe_sim,
        "deff_sim": deff_sim,
        "deff_theory": deff_theory,
        "err_pct": err * 100.0,
        "pass_tol": bool(err * 100.0 <= TOL_PCT),
        "sensitivity": sens,
        "mass0": mass0,
        "mass_drift_max": mass_drift,
        "f_mass_drift_max": f_mass_drift,
        "v2_pass": bool(mass_drift <= V2_MASS_MAX),
        "finite": finite,
        "R_at_t2": float(sarr["R"][-1]),
        "sigma2_at_t2": float(sarr["sigma2"][-1]),
        "sigma_over_L_at_t2": float(math.sqrt(sarr["sigma2"][-1]) / nx),
        "naive_over_wrapped_at_t2": float(sarr["sigma2_naive"][-1] / sarr["sigma2"][-1]),
        "u_abs_max": float(sarr["u_abs_max"].max()),
        "elapsed_s": round(elapsed, 1),
        "settle_s": round(settle_s, 1),
        "entries": {
            "collide": "tensorlbm.solver.collide_bgk",
            "stream": "tensorlbm.solver.stream",
            "wall_velocity": "tensorlbm.thermal.pre_streaming_bounce_back",
            "force": "tensorlbm.turbulent_channel._apply_body_force_2d",
            "scalar_eq": "tensorlbm.thermal.temperature_equilibrium",
            "scalar_collide": "tensorlbm.thermal.temperature_collision",
            "scalar_stream": "tensorlbm.thermal.temperature_stream",
            "macro": "tensorlbm.d2q9.macroscopic / d2q9.equilibrium",
            "wall_scalar": "inline scalar_wall_reflect (unwrapped post-stream "
            "reflection at first/last fluid rows + zero dead rows; NOTES.md A.8)",
        },
        "samples_every10": [
            {k: samples[i][k] for k in ("step", "sigma2", "R", "u_mean")}
            for i in range(0, len(samples), 10)
        ],
    }
    if mode == "diffusion":
        res["v4_pass"] = bool(
            err <= V4_DEFF_MAX
            and float(sarr["u_abs_max"].max()) < V4_UMAX_ABS
            and mass_drift <= V2_MASS_MAX
        )
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=2))
    print(
        f"[{mode} pe={pe} H={H}] Deff_sim={deff_sim:.6f} Deff_theory={deff_theory:.6f} "
        f"err={err * 100:.3f}% | pe_sim={pe_sim:.3f} u_mean={u_mean_win:.6f} "
        f"(tgt {u_mean_target:.6f}, {abs(u_mean_win / u_mean_target - 1.0) * 100:.3f}%) "
        f"mass_drift={mass_drift:.2e} R_t2={res['R_at_t2']:.3f} "
        f"steps={n_steps} elapsed={elapsed:.0f}s",
        flush=True,
    )
    return res


def _periodized_gaussian(
    x: np.ndarray, x0: float, sigma: float, nx: int, n_image: int = 4
) -> np.ndarray:
    """Smooth periodic Gaussian on the circle (sum of periodic images).

    A truncated Gaussian placed directly on [0, L) has a seam discontinuity
    and is NOT the field the estimator faces in the simulation (which is
    smooth on the torus); the first V3 draft made that mistake and was fixed
    before the formal scan (NOTES.md B timeline).
    """
    c = np.zeros_like(x)
    for mm in range(-n_image, n_image + 1):
        c += np.exp(-((x - x0 + mm * nx) ** 2) / (2.0 * sigma**2))
    return c


def synthetic_check(out_json: str | None = None) -> dict:
    """V3: wrapped-moment estimator bias on synthetic lattices (fp64, numpy)."""
    nx = 1792
    x0 = nx / 3.0
    x = np.arange(nx, dtype=np.float64)
    rows = []
    for frac in (0.05, 0.10, 0.15, 0.20, 0.25):
        sigma = frac * nx
        c = _periodized_gaussian(x, x0, sigma, nx)
        s2, xc, r = wrapped_sigma2(c)
        rows.append(
            {
                "sigma_over_L": frac,
                "sigma_nodes": sigma,
                "sigma2_hat": s2,
                "bias": s2 / sigma**2 - 1.0,
                "naive_bias": naive_sigma2(c, xc) / sigma**2 - 1.0,
                "R": r,
            }
        )
    # non-Gaussian cross-check: two separated periodized blobs with MASS
    # weights 0.6/0.4 (amplitude ~ 1/sigma so each blob carries its weight);
    # reference is the exact continuous circular variance of the wrapped mixture
    s1, s2n = 0.08 * nx, 0.06 * nx
    k = 2.0 * math.pi / nx
    delta = 0.3 * nx
    c2 = (0.6 / s1) * _periodized_gaussian(x, x0, s1, nx) + (0.4 / s2n) * (
        _periodized_gaussian(x, (x0 + delta) % nx, s2n, nx)
    )
    r_ex = abs(
        0.6 * np.exp(-((s1 * k) ** 2) / 2.0)
        + 0.4 * np.exp(-((s2n * k) ** 2) / 2.0) * np.exp(1j * k * delta)
    )
    exact2 = -2.0 * math.log(r_ex) / (k * k)
    s2c, xcc, rc = wrapped_sigma2(c2)
    bias_max = max(abs(r["bias"]) for r in rows)
    res = {
        "mode": "synthetic",
        "nx": nx,
        "rows": rows,
        "two_blob": {
            "sigma2_hat": s2c,
            "sigma2_exact_circular": exact2,
            "bias": s2c / exact2 - 1.0,
            "R": rc,
        },
        "bias_max": bias_max,
        "v3_pass": bool(bias_max <= V3_BIAS_MAX),
    }
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=2))
    print(f"[synthetic] bias_max={bias_max:.2e} pass={res['v3_pass']}", flush=True)
    for r in rows:
        print(
            f"  sigma/L={r['sigma_over_L']:.2f} bias={r['bias']:+.2e} "
            f"naive_bias={r['naive_bias']:+.2e} R={r['R']:.3f}",
            flush=True,
        )
    print(
        f"  two-blob bias={res['two_blob']['bias']:+.2e} (non-Gaussian cross-check)",
        flush=True,
    )
    return res


def scan(device_str: str, out_dir: str) -> dict:
    device = torch.device(device_str)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for pe in PE_LIST:
        for H in H_LIST:
            r = run_case(
                float(pe),
                H,
                device,
                mode="dispersion",
                out_json=str(out / f"case_pe{pe}_H{H}.json"),
            )
            cases.append(r)
    summary = {}
    for pe in PE_LIST:
        c64 = next(c for c in cases if c["pe_target"] == pe and c["H"] == 64)
        c128 = next(c for c in cases if c["pe_target"] == pe and c["H"] == 128)
        summary[f"pe{pe}"] = {
            "err_H64_pct": c64["err_pct"],
            "err_H128_pct": c128["err_pct"],
            "monotone_decreasing": bool(c128["err_pct"] < c64["err_pct"]),
            "both_within_tol": bool(c64["err_pct"] <= TOL_PCT and c128["err_pct"] <= TOL_PCT),
        }
    all_within = all(c["err_pct"] <= TOL_PCT for c in cases)
    all_mono = all(v["monotone_decreasing"] for v in summary.values())
    result = {
        "case": "taylor_aris_dispersion",
        "verdict_pass": bool(all_within and all_mono),
        "all_within_tol": bool(all_within),
        "all_monotone": bool(all_mono),
        "tol_pct": TOL_PCT,
        "theory": "Deff/D = 1 + Pe^2/210 (Taylor 1953 / Aris 1956, plane slit)",
        "measurement": "slope of wrapped-moment sigma_x^2(t) over [2,3]H^2/D, "
        "Deff = slope/2, cross-section (fluid-row) averaged concentration",
        "grid_levels": list(H_LIST),
        "pe_levels": list(PE_LIST),
        "tau": TAU,
        "tau_T": TAU_T,
        "alpha": ALPHA,
        "nu": NU,
        "per_case": [
            {
                "pe_target": c["pe_target"],
                "H": c["H"],
                "nx": c["nx"],
                "pe_sim": c["pe_sim"],
                "u_mean_target": c["u_mean_target"],
                "u_mean_measured": c["u_mean_measured_window"],
                "deff_sim": c["deff_sim"],
                "deff_theory": c["deff_theory"],
                "err_pct": c["err_pct"],
                "pass_tol": c["pass_tol"],
                "sensitivity": c["sensitivity"],
                "mass_drift_max": c["mass_drift_max"],
                "v2_pass": c["v2_pass"],
                "v1_l2": c["v1_l2"],
                "Ma": c["Ma"],
            }
            for c in cases
        ],
        "per_pe_summary": summary,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print("=" * 78, flush=True)
    print(
        f"VERDICT pass={result['verdict_pass']} (all_within={all_within}, all_monotone={all_mono})",
        flush=True,
    )
    for pe, v in summary.items():
        print(
            f"  {pe}: H64 {v['err_H64_pct']:.3f}% -> H128 {v['err_H128_pct']:.3f}% "
            f"mono={v['monotone_decreasing']} within={v['both_within_tol']}",
            flush=True,
        )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sy = sub.add_parser("synthetic")
    sy.add_argument("--out", default=None)
    sp = sub.add_parser("profile")
    sp.add_argument("--pe", type=float, required=True)
    sp.add_argument("--H", type=int, required=True)
    sp.add_argument("--device", default="cuda")
    sp.add_argument("--out", default=None)
    dp = sub.add_parser("diffusion")
    dp.add_argument("--device", default="cuda")
    dp.add_argument("--out", default=None)
    cp = sub.add_parser("case")
    cp.add_argument("--pe", type=float, required=True)
    cp.add_argument("--H", type=int, required=True)
    cp.add_argument("--device", default="cuda")
    cp.add_argument("--dtype", default="fp32", choices=("fp32", "fp64"))
    cp.add_argument("--out", default=None)
    sc = sub.add_parser("scan")
    sc.add_argument("--device", default="cuda")
    sc.add_argument("--out-dir", default=".")
    a = p.parse_args()
    if a.cmd == "synthetic":
        synthetic_check(a.out)
    elif a.cmd == "profile":
        run_case(a.pe, a.H, torch.device(a.device), mode="profile", out_json=a.out)
    elif a.cmd == "diffusion":
        run_case(10.0, 64, torch.device(a.device), mode="diffusion", out_json=a.out)
    elif a.cmd == "case":
        dt = torch.float32 if a.dtype == "fp32" else torch.float64
        run_case(a.pe, a.H, torch.device(a.device), mode="dispersion", out_json=a.out, dtype=dt)
    elif a.cmd == "scan":
        scan(a.device, a.out_dir)


if __name__ == "__main__":
    main()
