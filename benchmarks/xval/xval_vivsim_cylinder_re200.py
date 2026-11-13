#!/usr/bin/env python3
"""XV AL vs OSS `haimingz/vivsim` — free-stream cylinder at Re=200, DOI-anchored Cd/Cl.

Anchor: ``FEAT-ANCHOR-FEEDER`` / ``XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS``.

What the OSS implementation does (source-confirmed, zero hand-copy)
------------------------------------------------------------------
``vivsim/examples/2d/flow_pass_cylinder.py`` (commit fb84b12):
  * geometry : D=60, NX=30D, NY=20D, cylinder centre (10D,10D)      (:37-44)
  * physics  : U0=0.05, RE=200, TM=int(30*5/U0*D) = 150 D/U0 steps  (:53-62)
  * coupling : IB multi-direct forcing, 3 iterations, Peskin 4-pt
               kernel, 4D markers, IB_PAD=2                        (:66-78)
  * collision: KBC (`lbm.collision_kbc`) + EDM forcing (`forcing_edm`)
  * BCs      : left NEBB ux=U0, right equilibrium ux=U0             (:131-133)
  * reference: CD_REF=1.389, CL_REF=0.718 "averaged from" 3 DOIs    (:78-85)
  * post-proc: cd = h0*2/(D*U0**2), cl = h1*2/(D*U0**2)             (:195-196)
               cd_mean = mean(cd[TM//2:])                           (:207)
               cl_max  = max(|cl[TM//2:]|)                          (:208)
  * outputs  : prints "Cd = ... err: x%" into the figure; **no assert,
               no pytest, no tolerance gate anywhere in the repo**.

This script runs the *same case* with TensorLBM primitives, evaluates the
*vivsim calibers* (cd_mean over the last half of the record, cl_peak =
max|Cl| over the last half) plus two reference-free calibers of our own
(harmonic amplitude at the shedding frequency, per-period median half
peak-to-peak), and prints the running error continuously -- i.e. it
reproduces the OSS practice -- but ends with a machine-checkable verdict
(the half of the practice vivsim is missing).

Usage
-----
    PYTHONPATH=src python benchmarks/xval/xval_vivsim_cylinder_re200.py \\
        --D 16 --u 0.1 --out out.json --series series.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensorlbm.boundaries import (
    compute_obstacle_forces,
    cylinder_mask,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import C as C2D
from tensorlbm.d2q9 import equilibrium
from tensorlbm.solver import collide_mrt, stream

# ---------------------------------------------------------------- references
# Source of the reference values (NOT hand-copied from a paper PDF): the OSS
# implementation itself carries them, together with the three DOIs it averaged
# them from -- vivsim/examples/2d/flow_pass_cylinder.py:78-85 @ fb84b12.
VIVSIM_SRC = "vivsim/examples/2d/flow_pass_cylinder.py:78-85 @ fb84b12"
CD_REF = 1.389  # mean drag coefficient
CL_REF = 0.718  # maximum lift coefficient amplitude
CD_REFS_DOI = (
    "10.1016/j.ijmecsci.2024.108961",
    "10.1016/j.camwa.2014.05.013",
    "10.1017/S0022112086003014",
)
# Secondary, independent-of-vivsim sanity anchor (not a pass/fail criterion).
ST_LIT = 0.197  # Williamson & Brown 1998 / Williamson 1996, Re=200

TOL_PCT = 15.0  # owner precision standard: |err| > 15% is not acceptable


# --------------------------------------------------------------- calibers
def harmonic_amplitude(t: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Least-squares amplitude at the dominant frequency + that frequency.

    Same estimator family as ``benchmarks/verified/cylinder/re100_dfg``
    post-audit (m16827): fit ``s = a*cos(2*pi*f*t) + b*sin(2*pi*f*t) + c``
    at the FFT peak; amplitude = hypot(a, b).  Unbiased by single-sample noise.
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    if t.size < 16:
        return float("nan"), float("nan")
    dt = float(np.median(np.diff(t)))
    sig = s - s.mean()
    win = np.hanning(sig.size)
    spec = np.abs(np.fft.rfft(sig * win))
    freqs = np.fft.rfftfreq(sig.size, d=dt)
    k = int(np.argmax(spec[1:])) + 1
    f0 = float(freqs[k])
    # parabolic refinement on the log spectrum
    if 0 < k < spec.size - 1:
        lp = np.log(np.maximum(spec[k - 1 : k + 2], 1e-30))
        den = lp[0] - 2.0 * lp[1] + lp[2]
        if abs(den) > 1e-12:
            d = float(np.clip(0.5 * (lp[0] - lp[2]) / den, -0.5, 0.5))
            f0 = float(freqs[k] + d * (freqs[1] - freqs[0]))
    w = 2.0 * math.pi * f0
    A = np.stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)], axis=1)
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    return float(math.hypot(coef[0], coef[1])), f0


def per_period_half_pp(t: np.ndarray, s: np.ndarray, f0: float) -> tuple[float, float]:
    """Median over whole shedding periods of (max-min)/2, and the IQR spread."""
    if not np.isfinite(f0) or f0 <= 0:
        return float("nan"), float("nan")
    T = 1.0 / f0
    t0, t1 = t[0], t[-1]
    n = int(math.floor((t1 - t0) / T)) - 1
    if n < 2:
        return float("nan"), float("nan")
    amps = []
    for j in range(n):
        m = (t >= t0 + j * T) & (t < t0 + (j + 1) * T)
        if m.sum() >= 4:
            amps.append(0.5 * (s[m].max() - s[m].min()))
    if len(amps) < 2:
        return float("nan"), float("nan")
    a = np.asarray(amps)
    return float(np.median(a)), float(np.subtract(*np.percentile(a, [75, 25])))


def vivsim_calibers(cd: np.ndarray, cl: np.ndarray) -> dict:
    """Cd mean and |Cl| peak over the **second half of the record**.

    Exactly the OSS calibers (flow_pass_cylinder.py:207-208), so that the
    comparison is caliber-matched: any residual disagreement is numerics.
    """
    h = cd.size // 2
    return {
        "cd_mean_last_half": float(np.mean(cd[h:])),
        "cl_peak_last_half": float(np.max(np.abs(cl[h:]))),
        "window": [int(h), int(cd.size)],
    }


def ours_calibers(t: np.ndarray, cd: np.ndarray, cl: np.ndarray, u_in: float, D: float) -> dict:
    """Transient-explicit calibers of our own (post-m16827 discipline).

    The OSS 'second half of the record' window is data-independent but blind to
    whether shedding has actually developed; here the transient is cut by a
    declared convective time and the amplitude is a frequency-domain estimate.
    """
    t_tot = t[-1] - t[0]
    keep = t >= (t[0] + 0.5 * t_tot)  # same 50% cut, then harmonic estimate
    a_h, f0 = harmonic_amplitude(t[keep], cl[keep])
    a_pp, iqr = per_period_half_pp(t[keep], cl[keep], f0)
    st = f0 * D / u_in if math.isfinite(f0) else float("nan")
    h = cd.size // 2
    return {
        "cd_mean": float(np.mean(cd[keep])),
        "cl_amp_harmonic": a_h,
        "cl_amp_period_median": a_pp,
        "cl_amp_period_iqr": iqr,
        "st": st,
        "nfreq_hz_lattice": f0,
        "cd_half1": float(np.mean(cd[:h])),
        "cd_half2": float(np.mean(cd[h:])),
        "cl_std_last_half": float(np.std(cl[h:])),
    }


def verdict(res: dict) -> dict:
    """Predeclared gate: |err| < 15% on both vivsim calibers."""
    e_cd = (res["vivsim"]["cd_mean_last_half"] - CD_REF) / CD_REF * 100.0
    e_cl = (res["vivsim"]["cl_peak_last_half"] - CL_REF) / CL_REF * 100.0
    e_cl_h = (
        (res["ours"]["cl_amp_harmonic"] - CL_REF) / CL_REF * 100.0
        if math.isfinite(res["ours"]["cl_amp_harmonic"])
        else float("nan")
    )
    e_st = (
        (res["ours"]["st"] - ST_LIT) / ST_LIT * 100.0
        if math.isfinite(res["ours"]["st"])
        else float("nan")
    )
    return {
        "err_pct_cd_mean": e_cd,
        "err_pct_cl_peak": e_cl,
        "err_pct_cl_amp_harmonic_vs_CL_REF": e_cl_h,
        "st_vs_lit_err_pct": e_st,
        "tol_pct": TOL_PCT,
        "pass_cd": bool(abs(e_cd) < TOL_PCT),
        "pass_cl": bool(abs(e_cl) < TOL_PCT),
        "pass": bool(abs(e_cd) < TOL_PCT and abs(e_cl) < TOL_PCT),
    }


# ---------------------------------------------------------------- solver run
def run_case(
    D: int = 16,
    u_in: float = 0.1,
    re: float = 200.0,
    nx_D: float = 30.0,
    ny_D: float = 20.0,
    cyl_x_D: float = 10.0,
    conv_lengths: float = 150.0,
    n_steps: int | None = None,
    sponge_d: float = 5.0,
    alpha: float = 10.0,
    power: float = 2.0,
    st_seed: float = 0.197,
    seed_amp: float = 0.10,
    seed_cols: int = 4,
    sample_every: int = 10,
    progress_every: int = 2000,
    threads: int = 32,
    dtype: str = "float32",
):
    if threads:
        torch.set_num_threads(threads)
    tdt = torch.float32 if dtype == "float32" else torch.float64

    nu = u_in * D / re
    tau = 3.0 * nu + 0.5
    nx, ny = int(nx_D * D), int(ny_D * D)
    cx, cy = cyl_x_D * D, ny / 2.0
    radius = D / 2.0
    if n_steps is None:
        n_steps = int(conv_lengths * D / u_in)

    dev = torch.device("cpu")
    solid = cylinder_mask(nx, ny, cx, cy, radius, dev)
    fluid = ~solid
    surface = solid & (
        torch.roll(fluid, 1, 0)
        | torch.roll(fluid, -1, 0)
        | torch.roll(fluid, 1, 1)
        | torch.roll(fluid, -1, 1)
    )
    dyn_p = 0.5 * u_in**2 * D

    x0 = int(nx - sponge_d * D)
    W = max(nx - x0, 1)
    sigma = make_sponge_strength(ny, nx, x0, W, power=power, device=dev)
    tau_field = (tau * (1.0 + alpha * sigma)).to(tdt)

    T_seed = D / (st_seed * u_in)
    n_seed = int(2 * T_seed)

    rho0 = torch.ones((ny, nx), dtype=tdt)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    uy0 = torch.zeros_like(rho0)
    f = equilibrium(rho0, ux0, uy0)
    initial_mass = float(f.sum().item())

    c2d = C2D.to(dev).to(tdt)
    cx_w = c2d[:, 0].view(9, 1, 1)
    cy_w = c2d[:, 1].view(9, 1, 1)

    print(
        f"[xval] D={D} u={u_in} Re={re} nu={nu:.5f} tau={tau:.5f} grid={nx}x{ny} "
        f"steps={n_steps} (conv_lengths={conv_lengths}) sponge={sponge_d}D "
        f"({nx - x0}/{nx} cols) threads={torch.get_num_threads()}",
        flush=True,
    )
    print(f"[xval] refs from {VIVSIM_SRC}: CD_REF={CD_REF} CL_REF={CL_REF}", flush=True)

    t0 = time.time()
    ts, cds, cls = [], [], []
    nan_at = None
    steps_done = 0
    for step in range(1, n_steps + 1):
        before = f.clone()
        collided = collide_mrt(f, tau, tau_field=tau_field)
        f = torch.where(solid.unsqueeze(0), before, collided)
        f = stream(f)
        fx, fy = compute_obstacle_forces(f, surface)
        f = far_field_bc_2d(f, u_in, obstacle_mask=solid)
        if step % 2000 == 0:
            f = f * (initial_mass / f.sum().item())

        if step <= n_seed:
            phase = 2 * math.pi * step / T_seed
            rho_col = f.sum(0)[:, :seed_cols]
            ux_col = (f * cx_w).sum(0)[:, :seed_cols] / rho_col.clamp(min=1e-12)
            uy_col = (f * cy_w).sum(0)[:, :seed_cols] / rho_col.clamp(min=1e-12)
            uy_col = uy_col + seed_amp * u_in * math.sin(phase)
            f[:, :, :seed_cols] = equilibrium(rho_col, ux_col, uy_col)

        if step % sample_every == 0:
            ts.append(step)
            cds.append(float(fx.item()) / dyn_p)
            cls.append(float(fy.item()) / dyn_p)

        if step % 500 == 0 and torch.isnan(f).any():
            nan_at = step
            print(f"[xval] *** NaN at step {step} ***", flush=True)
            break
        if step % progress_every == 0:
            # the OSS practice: continuous error print against the DOI refs
            cd_now = float(np.mean(cds[len(cds) // 2 :])) if cds else float("nan")
            cl_now = float(np.max(np.abs(cls[len(cls) // 2 :]))) if cls else float("nan")
            el = time.time() - t0
            mlups = (step * nx * ny) / el / 1e6
            print(
                f"[xval] step {step}/{n_steps} Cd={cd_now:.4f} "
                f"(ref {CD_REF}, err {(cd_now - CD_REF) / CD_REF * 100:+.1f}%) "
                f"Cl_peak={cl_now:.4f} (ref {CL_REF}, err "
                f"{(cl_now - CL_REF) / CL_REF * 100:+.1f}%) "
                f"[{el:.0f}s {mlups:.1f} MLUPS]",
                flush=True,
            )
        steps_done = step

    wall = time.time() - t0
    t = np.asarray(ts, dtype=float)
    cd = np.asarray(cds, dtype=float)
    cl = np.asarray(cls, dtype=float)
    if nan_at is not None:
        bad = np.searchsorted(t, nan_at, side="left")
        t, cd, cl = t[:bad], cd[:bad], cl[:bad]

    res = {
        "case": "xval_vivsim_cylinder_re200",
        "anchor": "XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS",
        "oss": "haimingz/vivsim @ fb84b12 examples/2d/flow_pass_cylinder.py",
        "refs": {"CD_REF": CD_REF, "CL_REF": CL_REF, "source": VIVSIM_SRC, "dois": list(CD_REFS_DOI)},
        "setup": {
            "D": D,
            "u_in": u_in,
            "re": re,
            "nu": nu,
            "tau": tau,
            "nx": nx,
            "ny": ny,
            "nx_D": nx_D,
            "ny_D": ny_D,
            "cyl_x_D": cyl_x_D,
            "blockage": 1.0 / ny_D,
            "n_steps": n_steps,
            "conv_lengths": n_steps * u_in / D,
            "sponge": {"x0": x0, "width": nx - x0, "alpha": alpha, "power": power,
                       "tau_max": float(tau_field.max())},
            "seed": {"st_seed": st_seed, "amp": seed_amp, "cols": seed_cols,
                     "n_seed_steps": n_seed},
            "collision": "MRT (s_e=1.64,s_eps=1.54,s_q=1.7)",
            "bc": "far_field_bc_2d (inlet u_in / outlet zero-grad / top-bottom far field)",
            "wall": "solid freeze + half-way bounce-back + Ladd MEM",
            "sample_every": sample_every,
            "dtype": dtype,
            "torch_threads": torch.get_num_threads(),
        },
        "runtime": {"wall_s": wall, "steps_done": steps_done, "nan_at": nan_at,
                    "mlups": (steps_done * nx * ny) / max(wall, 1e-9) / 1e6},
        "vivsim": vivsim_calibers(cd, cl),
        "ours": ours_calibers(t, cd, cl, u_in, D),
        "units": "lattice; Cd=2Fx/(rho*u^2*D), Cl=2Fy/(rho*u^2*D)",
        "dt_lattice": 1.0,
    }
    res["verdict"] = verdict(res)
    return res, t, cd, cl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--u", type=float, default=0.1)
    ap.add_argument("--re", type=float, default=200.0)
    ap.add_argument("--conv-lengths", type=float, default=150.0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--sponge-d", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--st-seed", type=float, default=0.197)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", default="xval_vivsim_cylinder_re200.json")
    ap.add_argument("--series", default="")
    ap.add_argument("--strict", action="store_true", help="exit 1 when the gate fails")
    a = ap.parse_args()

    res, t, cd, cl = run_case(
        D=a.D, u_in=a.u, re=a.re, conv_lengths=a.conv_lengths,
        n_steps=(a.steps or None), sponge_d=a.sponge_d, alpha=a.alpha,
        st_seed=a.st_seed, threads=a.threads, dtype=a.dtype,
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    if a.series:
        np.savez_compressed(a.series, t=t, cd=cd, cl=cl)
        csvp = Path(a.series).with_suffix(".csv")
        with csvp.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "cd", "cl"])
            w.writerows(zip(t.tolist(), cd.tolist(), cl.tolist(), strict=False))
    v = res["verdict"]
    print(json.dumps({k: res[k] for k in ("setup", "runtime", "vivsim", "ours", "verdict")}, indent=2), flush=True)
    print(
        f"[xval] GATE: Cd {res['vivsim']['cd_mean_last_half']:.4f} vs {CD_REF} "
        f"({v['err_pct_cd_mean']:+.2f}%) | Cl_peak {res['vivsim']['cl_peak_last_half']:.4f} "
        f"vs {CL_REF} ({v['err_pct_cl_peak']:+.2f}%) | tol {TOL_PCT}% -> "
        f"{'PASS' if v['pass'] else 'FAIL'}",
        flush=True,
    )
    if a.strict and not v["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())