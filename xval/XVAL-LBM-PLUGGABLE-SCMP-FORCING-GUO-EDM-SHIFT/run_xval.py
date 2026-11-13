#!/usr/bin/env python
"""XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT — Round 1 cross-benchmark.

Cross-validate TensorLBM's pluggable SCMP forcing schemes (velocity_shift,
EDM, Guo) against a clean reference implementation derived from the
published literature (zero hand-copy: formulas transcribed directly from
papers).

OSS / literature reference formulas:
  - Velocity shift:  u_eq = u + τ·F/ρ  (Shan & Chen 1993)
  - EDM:  Δf = feq(ρ, u + F/ρ) − feq(ρ, u)  (Kupershtokh 2004)
  - Guo:  S_i = w_i(1−1/(2τ))[(c_i−u*)·F/cs² + (c_i·u*)(c_i·F)/cs⁴]
          with u* = u + F/(2ρ)  (Guo, Zheng & Shi 2002, PRE 65, 046308)

Benchmark: static droplet Laplace law test (2-D D2Q9, periodic domain).
  ΔP = σ / R_eff  →  σ = slope of ΔP vs 1/R_eff

Criterion (owner): benchmark error < 15 %; cross-arm < 3 %.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Repo import — use the work directory directly
# ---------------------------------------------------------------------------
_WORK = Path(__file__).resolve().parents[2]  # m18371 root
sys.path.insert(0, str(_WORK / "src"))

from tensorlbm.d2q9 import C, W, equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import (  # noqa: E402
    collide_sc_single_component,
    psi_exp,
    sc_single_component_force,
)
from tensorlbm.solver import stream  # noqa: E402

# Verify we're using the right repo
import tensorlbm
assert tensorlbm.__file__.startswith(str(_WORK / "src")), (
    f"Wrong repo: {tensorlbm.__file__} (expected {_WORK / 'src'})"
)

# ---------------------------------------------------------------------------
# Lattice constants
# ---------------------------------------------------------------------------
CS2 = 1.0 / 3.0
C_ARR = C.numpy()  # (9, 2)
W_ARR = W.numpy()  # (9,)

# ---------------------------------------------------------------------------
# Reference (OSS-style) implementations — formulas from papers, zero hand-copy
# ---------------------------------------------------------------------------

def _feq_2d(rho, ux, uy):
    """D2Q9 equilibrium, numpy, shape (9, ny, nx)."""
    cu = C_ARR[0] * ux[np.newaxis] + C_ARR[1] * uy[np.newaxis]
    u2 = ux**2 + uy**2
    return W_ARR[:, np.newaxis, np.newaxis] * rho[np.newaxis] * (
        1.0 + cu / CS2 + 0.5 * (cu / CS2) ** 2 - 0.5 * u2[np.newaxis] / CS2
    )


def _macroscopic_2d(f):
    """Return (rho, ux, uy) from f of shape (9, ny, nx)."""
    rho = f.sum(axis=0)
    rho_s = np.clip(rho, 1e-12, None)
    ux = (C_ARR[0] @ f) / rho_s
    uy = (C_ARR[1] @ f) / rho_s
    return rho, ux, uy


def _sc_force_2d(rho, G, psi_fn, gx=0.0, gy=0.0):
    """Standard SC self-interaction force (backward-gather convention).

    F = G·cs²·ψ(x)·Σ_i w_i c_i ψ(x−c_i) + ρ·g

    With G > 0 and backward gather this gives attraction (library convention).
    """
    ny, nx = rho.shape
    psi = psi_fn(rho)
    Fx = np.zeros_like(rho)
    Fy = np.zeros_like(rho)
    for i in range(9):
        cx, cy = int(C_ARR[i, 0]), int(C_ARR[i, 1])
        psi_nb = np.roll(np.roll(psi, -cy, axis=0), -cx, axis=1)
        Fx += W_ARR[i] * cx * psi_nb
        Fy += W_ARR[i] * cy * psi_nb
    Fx = G * CS2 * psi * Fx + rho * gx
    Fy = G * CS2 * psi * Fy + rho * gy
    return Fx, Fy


def collide_ref_velocity_shift(f, G, tau, psi_fn, gx=0.0, gy=0.0):
    """Reference velocity-shift SCMP collision (Shan & Chen 1993)."""
    rho, ux, uy = _macroscopic_2d(f)
    Fx, Fy = _sc_force_2d(rho, G, psi_fn, gx, gy)
    rho_s = np.clip(rho, 1e-12, None)
    feq = _feq_2d(rho, ux + tau * Fx / rho_s, uy + tau * Fy / rho_s)
    return f - (f - feq) / tau


def collide_ref_edm(f, G, tau, psi_fn, gx=0.0, gy=0.0):
    """Reference EDM collision (Kupershtokh 2004)."""
    rho, ux, uy = _macroscopic_2d(f)
    Fx, Fy = _sc_force_2d(rho, G, psi_fn, gx, gy)
    rho_s = np.clip(rho, 1e-12, None)
    feq_u = _feq_2d(rho, ux, uy)
    feq_d = _feq_2d(rho, ux + Fx / rho_s, uy + Fy / rho_s)
    return f - (f - feq_u) / tau + (feq_d - feq_u)


def collide_ref_guo(f, G, tau, psi_fn, gx=0.0, gy=0.0):
    """Reference Guo (2002) second-order forcing collision.

    S_i = w_i (1 − 1/(2τ)) [ (c_i − u*)·F / cs² + (c_i·u*)(c_i·F) / cs⁴ ]
    with u* = u + F/(2ρ).
    """
    rho, ux, uy = _macroscopic_2d(f)
    Fx, Fy = _sc_force_2d(rho, G, psi_fn, gx, gy)
    rho_s = np.clip(rho, 1e-12, None)

    # Physical velocity with half-force correction
    uxp = ux + 0.5 * Fx / rho_s
    uyp = uy + 0.5 * Fy / rho_s

    feq = _feq_2d(rho, uxp, uyp)

    # Guo source term
    cu = C_ARR[0] * uxp[np.newaxis] + C_ARR[1] * uyp[np.newaxis]  # (9, ny, nx)
    cf = C_ARR[0] * Fx[np.newaxis] + C_ARR[1] * Fy[np.newaxis]    # (9, ny, nx)
    uf = uxp * Fx + uyp * Fy  # (ny, nx)

    pref = 1.0 - 0.5 / tau
    S = W_ARR[:, np.newaxis, np.newaxis] * pref * (
        (cf - uf[np.newaxis]) / CS2 + cu * cf / (CS2 * CS2)
    )
    return f - (f - feq) / tau + S


# ---------------------------------------------------------------------------
# TensorLBM wrappers (torch)
# ---------------------------------------------------------------------------

def collide_tl_velocity_shift(f, G, tau, psi_fn, gx=0.0, gy=0.0):
    """TensorLBM velocity-shift SCMP collision."""
    return collide_sc_single_component(
        f, G=G, tau=tau, psi_fn=psi_fn, gx=gx, gy=gy,
        forcing="velocity_shift", scheme="standard"
    )


def collide_tl_edm(f, G, tau, psi_fn, gx=0.0, gy=0.0):
    """TensorLBM EDM SCMP collision."""
    return collide_sc_single_component(
        f, G=G, tau=tau, psi_fn=psi_fn, gx=gx, gy=gy,
        forcing="edm", scheme="standard"
    )


def collide_tl_guo(f, G, tau, psi_fn, gx=0.0, gy=0.0):
    """TensorLBM Guo SCMP collision."""
    return collide_sc_single_component(
        f, G=G, tau=tau, psi_fn=psi_fn, gx=gx, gy=gy,
        forcing="guo", scheme="standard"
    )


# ---------------------------------------------------------------------------
# Droplet initialization
# ---------------------------------------------------------------------------

def init_droplet(L, R, rho_l, rho_v, width=4.0, dtype=np.float64):
    """Initialize a circular droplet with tanh interface."""
    ys = np.arange(L, dtype=dtype)
    xs = np.arange(L, dtype=dtype)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    r = np.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    rho = rho_v + 0.5 * (rho_l - rho_v) * (1.0 + np.tanh((R - r) / width))
    zero = np.zeros_like(rho)
    return _feq_2d(rho, zero, zero)


def init_droplet_torch(L, R, rho_l, rho_v, device, width=4.0):
    """Initialize a circular droplet with tanh interface (torch)."""
    ys = torch.arange(L, dtype=torch.float64, device=device)
    xs = torch.arange(L, dtype=torch.float64, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    rho = rho_v + 0.5 * (rho_l - rho_v) * (1.0 + torch.tanh((R - r) / width))
    zero = torch.zeros_like(rho)
    return equilibrium(rho, zero, zero)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def stream_numpy(f):
    """AA-pattern streaming for numpy (9, ny, nx)."""
    f_new = np.empty_like(f)
    for i in range(9):
        cx, cy = int(C_ARR[i, 0]), int(C_ARR[i, 1])
        f_new[i] = np.roll(np.roll(f[i], -cy, axis=0), -cx, axis=1)
    return f_new


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_droplet(collide_fn, L, R, rho_l, rho_v, G, tau, psi_fn,
                max_steps=8000, sample=200, min_steps=2000, conv_tol=1e-5,
                device=None, is_torch=False):
    """Run a static droplet simulation to steady state."""
    if is_torch:
        f = init_droplet_torch(L, R, rho_l, rho_v, device)
        mass0 = float(f.sum().item())
        ys = torch.arange(L, dtype=torch.float64, device=device)
        yy, xx = torch.meshgrid(ys, ys, indexing="ij")
        r_grid = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    else:
        f = init_droplet(L, R, rho_l, rho_v)
        mass0 = float(f.sum())
        ys = np.arange(L)
        yy, xx = np.meshgrid(ys, ys, indexing="ij")
        r_grid = np.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)

    core_mask = r_grid <= 0.25 * L / 2
    far_mask = r_grid >= 0.80 * L / 2

    hist = []
    stable = True
    err_msg = None
    step_i = 0

    for step_i in range(1, max_steps + 1):
        if is_torch:
            f = collide_fn(f)
            f = stream(f)
        else:
            f = stream_numpy(collide_fn(f))

        if step_i % sample != 0:
            continue

        if is_torch:
            rho = f.sum(dim=0).cpu().numpy()
        else:
            rho = f.sum(axis=0)

        if not np.isfinite(rho).all() or rho.min() < 0:
            stable = False
            err_msg = f"nonfinite/negative rho at step {step_i}"
            break

        rho_l_cur = float(rho[core_mask].mean())
        rho_v_cur = float(rho[far_mask].mean())

        # Iterative R_eff refinement
        r_eff = 0.0
        for _ in range(2):
            r_eff = float(np.sqrt(
                max((rho - rho_v_cur).sum(), 0.0) /
                (math.pi * max(rho_l_cur - rho_v_cur, 1e-12))
            ))
            if r_eff <= 0:
                break
            core_mask = r_grid <= 0.5 * r_eff
            rho_l_cur = float(rho[core_mask].mean())
            rho_v_cur = float(rho[far_mask].mean())

        if r_eff <= 0:
            stable = False
            err_msg = f"degenerate droplet at step {step_i}"
            break

        # Mechanical pressure: p_mech = ρ·cs² − (G/2)·cs²·ψ²
        psi_val = psi_fn(torch.tensor(rho) if is_torch else rho)
        if is_torch:
            psi_val = psi_val.cpu().numpy()
        p_mech = rho * CS2 - 0.5 * G * CS2 * psi_val**2
        dp = float(p_mech[core_mask].mean() - p_mech[far_mask].mean())

        # Spurious velocity
        if is_torch:
            mom_x = (f * C.to(device)[:, 0].view(9, 1, 1)).sum(dim=0).cpu().numpy()
            mom_y = (f * C.to(device)[:, 1].view(9, 1, 1)).sum(dim=0).cpu().numpy()
        else:
            mom_x = C_ARR[0] @ f
            mom_y = C_ARR[1] @ f
        rho_s = np.clip(rho, 1e-12, None)
        u_max = float(np.sqrt((mom_x / rho_s)**2 + (mom_y / rho_s)**2).max())

        mass = float(rho.sum())
        mass_drift = abs(mass - mass0) / mass0

        hist.append(dict(
            step=step_i, rho_l=rho_l_cur, rho_v=rho_v_cur,
            dp=dp, r_eff=r_eff, u_max=u_max, mass=mass, mass_drift=mass_drift
        ))

        # Convergence check
        if len(hist) >= 4 and step_i >= min_steps:
            rec = hist[-3:]
            if all(
                abs(h["rho_l"] - hp["rho_l"]) / max(abs(h["rho_l"]), 1e-12) <= conv_tol
                and abs(h["rho_v"] - hp["rho_v"]) / max(abs(h["rho_v"]), 1e-12) <= conv_tol
                for h, hp in zip(rec[1:], rec[:-1])
            ):
                break

    out = dict(
        stable=stable, error=err_msg, steps_run=step_i,
        converged=len(hist) >= 4 and step_i < max_steps and stable,
        hist=hist,
    )
    if hist:
        tail = hist[-5:]
        for k in ("rho_l", "rho_v", "dp", "r_eff"):
            m = sum(t[k] for t in tail) / len(tail)
            out[k] = m
            out[k + "_std"] = (sum((t[k] - m)**2 for t in tail) / len(tail))**0.5
        out["ratio"] = out["rho_l"] / out["rho_v"]
        out["u_max"] = max(t["u_max"] for t in tail)
        out["mass_drift"] = max(t["mass_drift"] for t in tail)
        out["mass_rate_per_step"] = out["mass_drift"] / out["steps_run"]
    return out


# ---------------------------------------------------------------------------
# Sigma fitting
# ---------------------------------------------------------------------------

def fit_sigma(runs):
    """Fit σ from ΔP vs 1/R_eff linear regression."""
    xs = [1.0 / r["r_eff"] for r in runs if r.get("r_eff", 0) > 0]
    ys = [r["dp"] for r in runs if r.get("r_eff", 0) > 0]
    n = len(xs)
    if n < 2:
        return dict(n=n, sigma=float("nan"), intercept=float("nan"),
                    rms_residual=float("nan"), dp_max=max(ys) if ys else 0,
                    resid_gate=False, points=[])
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx)**2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    rms = math.sqrt(sum(e**2 for e in resid) / n)
    return dict(
        n=n, sigma=slope, intercept=intercept,
        rms_residual=rms, dp_max=max(ys),
        resid_gate=rms <= 0.03 * max(ys),
        points=[dict(r_eff=r["r_eff"], inv_r=1.0/r["r_eff"], dp=r["dp"], tag=r.get("tag",""))
                for r in runs if r.get("r_eff", 0) > 0],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cpu")

    # SCMP parameters (classic psi_exp, moderate density ratio)
    G = 5.0       # coupling (G > 0 = attraction in library convention)
    tau = 1.0
    psi_fn = psi_exp  # classic Shan-Chen pseudopotential

    # Coexistence densities for psi_exp at G=5 (from literature / repo calibration)
    rho_l = 4.5
    rho_v = 0.37

    L = 128
    radii = [12, 16, 20, 24]

    forcing_schemes = ["velocity_shift", "edm", "guo"]
    collide_ref = {
        "velocity_shift": collide_ref_velocity_shift,
        "edm": collide_ref_edm,
        "guo": collide_ref_guo,
    }
    collide_tl = {
        "velocity_shift": collide_tl_velocity_shift,
        "edm": collide_tl_edm,
        "guo": collide_tl_guo,
    }

    results = {"ref": {}, "tl": {}}

    for scheme in forcing_schemes:
        print(f"\n=== {scheme} ===")
        ref_runs = []
        tl_runs = []

        for R in radii:
            tag = f"{scheme}_L{L}_R{R}"
            print(f"  [{tag}] reference ...", end="", flush=True)
            t0 = time.time()
            r_ref = run_droplet(
                lambda f, s=scheme: collide_ref[s](f, G, tau, psi_fn),
                L, R, rho_l, rho_v, G, tau, psi_fn,
                max_steps=8000, sample=200, is_torch=False
            )
            r_ref["tag"] = tag
            r_ref["wall_s"] = round(time.time() - t0, 1)
            ref_runs.append(r_ref)
            print(f" stable={r_ref['stable']} dp={r_ref.get('dp', float('nan')):.5f} "
                  f"r_eff={r_ref.get('r_eff', float('nan')):.2f} "
                  f"u_max={r_ref.get('u_max', float('nan')):.2e} "
                  f"t={r_ref['wall_s']}s")

            print(f"  [{tag}] TensorLBM ...", end="", flush=True)
            t0 = time.time()
            r_tl = run_droplet(
                lambda f, s=scheme: collide_tl[s](f, G, tau, psi_fn),
                L, R, rho_l, rho_v, G, tau, psi_fn,
                max_steps=8000, sample=200, is_torch=True, device=device
            )
            r_tl["tag"] = tag
            r_tl["wall_s"] = round(time.time() - t0, 1)
            tl_runs.append(r_tl)
            print(f" stable={r_tl['stable']} dp={r_tl.get('dp', float('nan')):.5f} "
                  f"r_eff={r_tl.get('r_eff', float('nan')):.2f} "
                  f"u_max={r_tl.get('u_max', float('nan')):.2e} "
                  f"t={r_tl['wall_s']}s")

        results["ref"][scheme] = ref_runs
        results["tl"][scheme] = tl_runs

    # Fit sigma for each arm + scheme
    sigma_results = {}
    for arm in ("ref", "tl"):
        sigma_results[arm] = {}
        for scheme in forcing_schemes:
            runs = results[arm][scheme]
            sigma_results[arm][scheme] = fit_sigma(runs)

    # Cross-arm comparison
    comparison = {}
    for scheme in forcing_schemes:
        sig_ref = sigma_results["ref"][scheme]["sigma"]
        sig_tl = sigma_results["tl"][scheme]["sigma"]
        if sig_ref and sig_tl and not (math.isnan(sig_ref) or math.isnan(sig_tl)):
            err = abs(sig_tl - sig_ref) / abs(sig_ref) * 100
        else:
            err = float("nan")
        comparison[scheme] = dict(
            sigma_ref=sig_ref, sigma_tl=sig_tl, error_pct=err,
            pass_15=err < 15 if not math.isnan(err) else False,
            pass_3=err < 3 if not math.isnan(err) else False,
        )

    # Output
    out = dict(
        parameters=dict(G=G, tau=tau, L=L, radii=radii, rho_l=rho_l, rho_v=rho_v),
        sigma=sigma_results,
        comparison=comparison,
        verdict=dict(
            all_pass_15=all(c["pass_15"] for c in comparison.values()),
            all_pass_3=all(c["pass_3"] for c in comparison.values()),
        ),
    )

    outpath = Path(__file__).parent / "results.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults written to {outpath}")

    # Print summary table
    print("\n=== Cross-arm comparison (σ from Laplace fit) ===")
    print(f"{'Scheme':>16} | {'σ_ref':>10} | {'σ_TL':>10} | {'Error %':>8} | <15% | <3%")
    print("-" * 65)
    for scheme in forcing_schemes:
        c = comparison[scheme]
        print(f"{scheme:>16} | {c['sigma_ref']:>10.6f} | {c['sigma_tl']:>10.6f} | "
              f"{c['error_pct']:>7.2f}% | {'✓' if c['pass_15'] else '✗'}    | "
              f"{'✓' if c['pass_3'] else '✗'}")

    return out


if __name__ == "__main__":
    main()
