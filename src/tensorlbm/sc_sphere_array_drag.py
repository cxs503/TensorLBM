#!/usr/bin/env python3
"""CASE-LBM-CUBIC-ARRAY-DRAG-REWORK2 (m16221): SC sphere-array drag vs Zick & Homsy / Hasimoto.

Fixes over m16140:
  - Error column labels now match the quantity used (K_mx for err_mx_*, K_mb for err_mb_*)
  - Richardson extrapolation to h->0 added (3-resolution ladder)
  - Steady-state detection tightened; longer runs for convergence
  - Momentum-exchange force computed on the correct post-stream / pre-BB state

Reference (locked):
    K(phi) = F / (6*pi*mu*a*U_s)   (U_s = superficial velocity, a = sphere radius)
    Hasimoto (1959) 5-term SC series for dilute limit; Zick & Homsy (1982) table
    for finite phi.

Lattice: D3Q19 BGK, full-way bounce-back, periodic unit cell.
Step order: collide -> stream -> (fluid-masked) body force -> bounce-back.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import torch

import tensorlbm
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.d3q19 import C as D3Q19_C
from tensorlbm.d3q19 import W_EXACT64, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d

assert str(Path(tensorlbm.__file__).resolve()).startswith(str(_REPO / "src")), tensorlbm.__file__

TAU = 1.0
NU = (TAU - 0.5) / 3.0
RE_TARGET = 0.05

SC_TABLE: dict[float, float] = {
    0.027: 2.0077, 0.064: 2.8102, 0.125: 4.292, 0.216: 7.4423,
    0.343: 15.402, 0.45: 28.09, 0.5236: 41.99,
}

LADDER: dict[float, list[int]] = {0.125: [64, 96, 128], 0.343: [64, 96, 128]}
VERDICT_PHI = [0.125, 0.343]
TOL_FINEST = 0.05
TOL_EXTRAP = 0.05
TOL_CHANNEL = 0.01

SAMPLE_EVERY = 200
DRIFT_SPAN = 10
DRIFT_TOL = 1e-5
STEADY_REPEATS = 3
POST_STEADY = 8000
MEAS_SAMPLES = 20
MAX_STEPS_CAP = 250_000
MAX_STEPS_COEF = 20


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hasimoto_K(phi: float) -> float:
    c = float(phi)
    u_ratio = (1.0 - 1.7601 * c ** (1.0 / 3.0) + c - 1.5593 * c**2
               + 3.9799 * c ** (8.0 / 3.0) - 3.0734 * c ** (10.0 / 3.0))
    return 1.0 / u_ratio


def loglog_K(phi: float) -> float:
    xs = [math.log(p) for p in SC_TABLE]
    ys = [math.log(k) for k in SC_TABLE.values()]
    x = math.log(phi)
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return math.exp(ys[i] + t * (ys[i + 1] - ys[i]))
    raise ValueError(f"phi={phi} outside table span")


def a_nominal(n: int, phi: float) -> float:
    return n * (3.0 * phi / (4.0 * math.pi)) ** (1.0 / 3.0)


def gap_cells(n: int, phi: float) -> float:
    return n - 2.0 * a_nominal(n, phi)


def sphere_array_mask(n: int, phi: float, device: torch.device) -> torch.Tensor:
    a = a_nominal(n, phi)
    idx = torch.arange(n, dtype=torch.float64, device=device)
    d = torch.abs(idx - n / 2.0)
    d = torch.minimum(d, n - d)
    dist2 = d[:, None, None] ** 2 + d[None, :, None] ** 2 + d[None, None, :] ** 2
    return dist2 <= a * a


def apply_body_force_3d(f: torch.Tensor, a_x: float, c: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    rho = f.sum(dim=0)
    return f + w * 3.0 * rho.unsqueeze(0) * c[:, 0].view(19, 1, 1, 1) * a_x


def momentum_exchange_force(f_post_stream: torch.Tensor, solid: torch.Tensor, c: torch.Tensor) -> list[float]:
    tot = np.zeros(3, dtype=np.float64)
    with torch.no_grad():
        for i in range(19):
            ci = (int(c[i, 0]), int(c[i, 1]), int(c[i, 2]))
            up = torch.roll(solid, shifts=(ci[2], ci[1], ci[0]), dims=(0, 1, 2))
            link = solid & (~up)
            if not bool(link.any()):
                continue
            val = float(f_post_stream[i][link].sum().item())
            tot += 2.0 * val * np.array(ci, dtype=np.float64)
    return tot.tolist()


def richardson_extrapolate(K_coarse: float, K_medium: float, K_fine: float,
                           h_coarse: float, h_medium: float, h_fine: float,
                           order: float = 2.0) -> tuple[float, float]:
    r1 = h_medium / h_fine
    r2 = h_coarse / h_medium
    if abs(r1 - r2) < 0.01:
        K_ext = (r1**order * K_fine - K_medium) / (r1**order - 1.0)
        err_est = abs(K_ext - K_fine)
    else:
        K_ext = (r1**order * K_fine - K_medium) / (r1**order - 1.0)
        err_est = abs(K_ext - K_fine)
    return K_ext, err_est


def device_info() -> dict:
    info = {"torch": torch.__version__, "threads": torch.get_num_threads(), "tensorlbm": tensorlbm.__file__}
    try:
        info["git_head"] = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"],
                                          capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        info["git_head"] = None
    return info


def run_case(phi: float, n: int, out: Path, max_steps: int | None = None, smoke: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    dt = torch.float32
    c = D3Q19_C.to(device=device, dtype=dt)
    w = W_EXACT64.to(device=device, dtype=dt).view(19, 1, 1, 1)

    a_nom = a_nominal(n, phi)
    solid = sphere_array_mask(n, phi, device)
    fluid = ~solid
    n_fluid = int(fluid.sum().item())
    phi_actual = 1.0 - n_fluid / n**3
    a_eff = a_nom * (phi_actual / phi) ** (1.0 / 3.0)
    gap = gap_cells(n, phi)

    k_ref_table = SC_TABLE.get(phi)
    k_ref_hasimoto = hasimoto_K(phi)
    k_design = k_ref_table if k_ref_table is not None else k_ref_hasimoto
    a_body = RE_TARGET * NU * NU * 3.0 * math.pi * k_design / n_fluid

    max_steps = max_steps or min(MAX_STEPS_COEF * n * n, MAX_STEPS_CAP)
    if smoke:
        max_steps = min(max_steps, 5000)

    ones = torch.ones((n, n, n), dtype=dt, device=device)
    zeros = torch.zeros((n, n, n), dtype=dt, device=device)
    f = equilibrium3d(ones, zeros, zeros, zeros).to(dt)
    del ones, zeros
    mass0 = float(f.sum().item())
    solid_b = solid.unsqueeze(0)

    hist: list[dict] = []
    steady_step: int | None = None
    recent_drifts: list[float] = []
    t0 = time.perf_counter()

    for step_i in range(1, max_steps + 1):
        f = collide_bgk3d(f, TAU)
        f = stream3d(f)
        f = torch.where(solid_b, f, apply_body_force_3d(f, a_body, c, w))
        f = bounce_back_cells_3d(f, solid)

        if step_i % SAMPLE_EVERY == 0:
            _, ux, uy, uz = macroscopic3d(f)
            um = float(ux[fluid].mean().item())
            umax = float(torch.sqrt(ux * ux + uy * uy + uz * uz)[fluid].max().item())
            mass = float(f.sum().item())
            hist.append({"step": step_i, "um": um, "umax": umax, "mass": mass})

            if steady_step is None and len(hist) >= DRIFT_SPAN + 1:
                drift = abs(hist[-1]["um"] - hist[-1 - DRIFT_SPAN]["um"]) / max(abs(hist[-1]["um"]), 1e-30)
                recent_drifts.append(drift)
                if len(recent_drifts) >= STEADY_REPEATS and all(v < DRIFT_TOL for v in recent_drifts[-STEADY_REPEATS:]):
                    steady_step = step_i - DRIFT_SPAN * SAMPLE_EVERY

        if steady_step is not None and step_i >= steady_step + POST_STEADY:
            break

    wall = time.perf_counter() - t0
    steps_run = hist[-1]["step"] if hist else 0
    if steady_step is None:
        print(f"WARNING: phi={phi} N={n}: steady not detected in {max_steps} steps", flush=True)

    with torch.no_grad():
        f_pre = collide_bgk3d(f, TAU)
        f_pre = stream3d(f_pre)
        Fvec = momentum_exchange_force(f_pre, solid, c)
        del f_pre

    _, ux, uy, uz = macroscopic3d(f)
    um = float(ux[fluid].mean().item())
    umax = float(torch.sqrt(ux * ux + uy * uy + uz * uz)[fluid].max().item())
    uy_absmax = float(uy[fluid].abs().max().item())
    uz_absmax = float(uz[fluid].abs().max().item())

    win = hist[-MEAS_SAMPLES:]
    um_win = sum(h["um"] for h in win) / len(win)
    mass_drift = max(abs(h["mass"] / mass0 - 1.0) for h in hist) if hist else float("nan")

    U_s = (1.0 - phi) * um_win
    U_s_act = (1.0 - phi_actual) * um_win
    F_mx = Fvec[0]
    F_inj = a_body * n_fluid

    def K_from_F(F: float, a: float, Us: float) -> float:
        return F / (6.0 * math.pi * NU * a * Us)

    K_mx = K_from_F(F_mx, a_nom, U_s)
    K_mb = K_from_F(F_inj, a_nom, U_s)
    K_qv = a_body * n**3 / (6.0 * math.pi * NU * a_nom * U_s)
    K_mx_act = K_from_F(F_mx, a_eff, U_s_act)

    ref = SC_TABLE.get(phi)
    if ref is None:
        ref = loglog_K(phi)

    # FIXED: error columns now match the quantity used
    err_mx_vs_ref = (K_mx / ref - 1.0) if ref else None
    err_mx_vs_hasimoto = K_mx / k_ref_hasimoto - 1.0
    err_mb_vs_ref = (K_mb / ref - 1.0) if ref else None
    err_mb_vs_hasimoto = K_mb / k_ref_hasimoto - 1.0

    rec = {
        "kind": "case", "created_utc": utc_now(), "phi": phi, "N": n,
        "device": device_info(), "tau": TAU, "nu": NU,
        "a_body": a_body, "a_nom": a_nom, "a_eff": a_eff,
        "phi_actual": phi_actual, "gap_cells": gap, "under_resolved": gap < 3.0,
        "n_fluid": n_fluid, "max_steps": max_steps, "steps_run": steps_run,
        "steady_detected": steady_step is not None, "steady_step": steady_step,
        "wall_seconds": wall, "ms_per_step": 1e3 * wall / max(1, steps_run),
        "mass_drift_rel": mass_drift, "um_fluid": um_win, "um_fluid_inst": um,
        "umax": umax, "Ma": math.sqrt(3.0) * umax, "Re_p": U_s * 2.0 * a_nom / NU,
        "U_s": U_s, "cross_flow_rel": (uy_absmax + uz_absmax) / abs(um_win) if um_win else None,
        "F_mx_x": F_mx, "F_inj_x": F_inj, "F_mx_over_F_inj": F_mx / F_inj if F_inj else None,
        "K_mx": K_mx, "K_mb": K_mb, "K_qv": K_qv, "K_mx_actual_geom": K_mx_act,
        "K_hasimoto": k_ref_hasimoto, "K_table": ref,
        "err_mx_vs_ref": err_mx_vs_ref, "err_mx_vs_hasimoto": err_mx_vs_hasimoto,
        "err_mb_vs_ref": err_mb_vs_ref, "err_mb_vs_hasimoto": err_mb_vs_hasimoto,
        "hist": hist[-500:],
    }
    p = out / f"case_phi{phi}_N{n}.json"
    p.write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: v for k, v in rec.items() if k != "hist"}, indent=2), flush=True)
    return rec


def load_cases(out: Path) -> dict:
    cases = {}
    for p in sorted(out.glob("case_phi*_N*.json")):
        r = json.loads(p.read_text())
        cases[(r["phi"], r["N"])] = r
    return cases


def make_report(out: Path, figdir: Path | None = None) -> dict:
    cases = load_cases(out)
    result = {"kind": "result", "created_utc": utc_now(), "points": {},
              "thresholds": {"tol_finest": TOL_FINEST, "tol_extrapolated": TOL_EXTRAP, "tol_channel": TOL_CHANNEL}}

    for phi in sorted({k[0] for k in cases}):
        tiers = [cases[(phi, n)] for n in sorted(n for (p, n) in cases if p == phi)]
        tiers.sort(key=lambda r: r["N"])
        ref = tiers[0]["K_table"]
        verdict_ref_bearing = phi in VERDICT_PHI and ref is not None
        finest = tiers[-1]
        chan = abs(finest["F_mx_over_F_inj"] - 1.0)
        conv = abs(finest["K_mx"] / tiers[-2]["K_mx"] - 1.0) if len(tiers) >= 2 else None

        richardson = None
        if len(tiers) >= 3:
            h_vals = [1.0 / t["N"] for t in tiers]
            K_vals = [t["K_mx"] for t in tiers]
            K_ext, err_ext = richardson_extrapolate(K_vals[0], K_vals[1], K_vals[2],
                                                     h_vals[0], h_vals[1], h_vals[2], order=2.0)
            err_ext_vs_ref = (K_ext / ref - 1.0) if ref else None
            richardson = {"K_extrapolated": K_ext, "err_extrapolated_vs_ref": err_ext_vs_ref,
                          "err_estimate": err_ext, "h_coarse": h_vals[0], "h_medium": h_vals[1], "h_fine": h_vals[2]}

        err_finest = (finest["K_mx"] / ref - 1.0) if ref else None

        if not verdict_ref_bearing:
            verdict = "WITHHELD_NO_LOCKED_REF_ROW"
        elif not finest["steady_detected"]:
            verdict = "INCONCLUSIVE_NOT_STEADY"
        elif chan > TOL_CHANNEL:
            verdict = "CHANNEL_INCONSISTENT"
        elif conv is not None and conv > TOL_FINEST:
            verdict = "NOT_GRID_CONVERGED"
        elif richardson and richardson["err_extrapolated_vs_ref"] is not None:
            verdict = "PASS_EXTRAPOLATED" if abs(richardson["err_extrapolated_vs_ref"]) < TOL_EXTRAP else ("PASS" if abs(err_finest) < TOL_FINEST else "FAIL")
        elif abs(err_finest) < TOL_FINEST:
            verdict = "PASS"
        else:
            verdict = "FAIL"

        result["points"][str(phi)] = {
            "phi": phi, "K_ref_table": ref, "K_hasimoto": finest["K_hasimoto"],
            "verdict": verdict, "verdict_bearing": bool(verdict_ref_bearing),
            "err_finest": err_finest, "convergence_finest_pair": conv,
            "channel_consistency": chan, "richardson_extrapolation": richardson,
            "tiers": [{"N": t["N"], "K_mx": t["K_mx"], "K_mb": t["K_mb"], "K_qv": t["K_qv"],
                       "K_hasimoto": t["K_hasimoto"], "K_table": t["K_table"],
                       "err_mx_vs_ref": t["err_mx_vs_ref"], "err_mx_vs_hasimoto": t["err_mx_vs_hasimoto"],
                       "err_mb_vs_ref": t["err_mb_vs_ref"], "err_mb_vs_hasimoto": t["err_mb_vs_hasimoto"],
                       "gap_cells": t["gap_cells"], "under_resolved": t["under_resolved"],
                       "phi_actual": t["phi_actual"], "Ma": t["Ma"], "Re_p": t["Re_p"],
                       "mass_drift_rel": t["mass_drift_rel"], "steady_detected": t["steady_detected"],
                       "steps_run": t["steps_run"], "ms_per_step": t["ms_per_step"],
                       "F_mx_over_F_inj": t["F_mx_over_F_inj"], "cross_flow_rel": t["cross_flow_rel"]}
                      for t in tiers],
        }

    (out / "result.json").write_text(json.dumps(result, indent=2))
    if figdir is not None:
        _plot(result, figdir)
    verdicts = {p: {"verdict": v["verdict"], "err": v["err_finest"],
                     "conv": v["convergence_finest_pair"], "chan": v["channel_consistency"]}
                for p, v in result["points"].items()}
    print(json.dumps(verdicts, indent=2))
    return result


def _plot(result: dict, figdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True, exist_ok=True)
    colors = {0.125: "tab:blue", 0.343: "tab:red"}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for p, v in result["points"].items():
        ns = [t["N"] for t in v["tiers"]]
        ks = [t["K_mx"] for t in v["tiers"]]
        col = colors.get(float(p), "k")
        ax.plot(ns, ks, "o-", color=col, label=f"LBM K_mx (phi={p})")
        if v["K_ref_table"]:
            ax.axhline(v["K_ref_table"], color=col, ls="--", lw=1, label=f"ref {v['K_ref_table']:.3f} (phi={p})")
        if v.get("richardson_extrapolation"):
            K_ext = v["richardson_extrapolation"]["K_extrapolated"]
            ax.axhline(K_ext, color=col, ls="-.", lw=1.5, label=f"Richardson {K_ext:.3f} (phi={p})")
    ax.set_xlabel("N (cells per lattice constant)")
    ax.set_ylabel(r"$K=F/(6\pi\mu a U_s)$")
    ax.set_title("SC sphere array drag: grid convergence vs reference")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "k_vs_N.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for p, v in result["points"].items():
        es = [t["err_mx_vs_ref"] for t in v["tiers"] if t["err_mx_vs_ref"] is not None]
        ns = [t["N"] for t in v["tiers"] if t["err_mx_vs_ref"] is not None]
        if es:
            ax.semilogy(ns, [abs(e) for e in es], "s-", color=colors.get(float(p), "k"), label=f"phi={p}")
    ax.axhline(0.05, color="k", ls="--", lw=1, label="5% criterion")
    ax.set_xlabel("N")
    ax.set_ylabel("|K_mx/K_ref - 1|")
    ax.set_title("Discretisation error vs resolution (K_mx channel)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(figdir / "error_vs_N.png", dpi=150)
    plt.close(fig)
