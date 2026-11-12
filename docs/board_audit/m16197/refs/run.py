#!/usr/bin/env python3
"""CASE-LBM-CUBIC-ARRAY-DRAG — simple-cubic sphere array, Stokes drag vs Hasimoto (1959)
and the locked Sangani-Acrivos / Zick-Homsy SC table.

Registered design: see NOTES.md (reference lock, channels, thresholds, ladder).

Subcommands
    case    one (phi, N) case -> out/case_phi{phi}_N{N}.json  (+ optional checkpoint)
    ladder  run all preregistered tiers of one phi
    report  aggregate case files -> out/result.json (+ figures)

All physics kernels are library calls from this worktree's ``src/tensorlbm``.
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

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch

import tensorlbm
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.d3q19 import C as D3Q19_C
from tensorlbm.d3q19 import W_EXACT64, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d

assert str(Path(tensorlbm.__file__).resolve()).startswith(str(REPO / "src")), tensorlbm.__file__

TAU = 1.0
NU = (TAU - 0.5) / 3.0
RE_TARGET = 0.05

# ---- reference values (NOTES.md 1.1-1.2); values are *computed* in code where possible
SC_TABLE = {
    0.027: 2.0077,
    0.064: 2.8102,
    0.125: 4.292,
    0.216: 7.4423,
    0.343: 15.402,
    0.45: 28.09,
    0.5236: 41.99,
}
LADDER = {0.125: [32, 48, 64, 96], 0.45: [48, 64, 96, 128], 0.5: [64, 96, 128]}
VERDICT_PHI = [0.125, 0.45]
TOL_FINEST = 0.05
TOL_CONV = 0.02
TOL_CHANNEL = 0.01

SAMPLE_EVERY = 100
DRIFT_SPAN = 20  # samples = 2000 steps
DRIFT_TOL = 1e-6
STEADY_REPEATS = 3
POST_STEADY = 4000
MEAS_SAMPLES = 40  # = 4000 steps


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hasimoto_K(phi: float) -> float:
    """Hasimoto (1959) SC sedimentation series, K = U0/U (NOTES 1.1). In code, not typed."""
    c = float(phi)
    u_ratio = (
        1.0
        - 1.7601 * c ** (1.0 / 3.0)
        + c
        - 1.5593 * c**2
        + 3.9799 * c ** (8.0 / 3.0)
        - 3.0734 * c ** (10.0 / 3.0)
    )
    return 1.0 / u_ratio


def loglog_K(phi: float) -> float:
    """Diagnostic log-log interpolation of the locked table (inside its span only)."""
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
    """Surface-to-surface gap of neighbouring spheres, in cells (NOTES 5)."""
    a = a_nominal(n, phi)
    return n - 2.0 * a


def sphere_array_mask(n: int, phi: float) -> torch.Tensor:
    """Periodic SC unit cell, sphere of radius a_nom at the cell centre (float64 test)."""
    a = a_nominal(n, phi)
    idx = np.arange(n, dtype=np.float64)
    d = np.abs(idx - n / 2.0)
    d = np.minimum(d, n - d)
    dist2 = d[:, None, None] ** 2 + d[None, :, None] ** 2 + d[None, None, :] ** 2
    return torch.as_tensor(dist2 <= a * a)


def apply_body_force_3d(f: torch.Tensor, a_x: float, c: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """First-order Guo body force, D3Q19 (disclosed driver-level kernel, NOTES 3)."""
    rho = f.sum(dim=0)
    return f + w * 3.0 * rho.unsqueeze(0) * c[:, 0].view(19, 1, 1, 1) * a_x


def momentum_exchange_force(f_post_stream: torch.Tensor, solid: torch.Tensor, c: torch.Tensor) -> list[float]:
    """Drag on the spheres by momentum exchange over interface links (NOTES 2).

    For every solid node x_s and direction i whose upstream neighbour x_s - c_i is
    fluid, the incoming population carries momentum f_i(x_s) * c_i and leaves with
    -f_i(x_s) * c_i, so the fluid loses 2 f_i(x_s) c_i on the solid.
    """
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


def device_info() -> dict:
    info = {
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "cuda": torch.cuda.is_available(),
        "tensorlbm": tensorlbm.__file__,
    }
    try:
        info["git_head"] = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        info["git_head"] = None
    return info


def run_case(phi: float, n: int, out: Path, max_steps: int | None, smoke: bool = False,
             ckpt_every: int = 2000, resume: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    dt = torch.float32
    c = D3Q19_C.to(dtype=dt)
    w = W_EXACT64.to(dtype=dt).view(19, 1, 1, 1)

    a_nom = a_nominal(n, phi)
    solid = sphere_array_mask(n, phi)
    fluid = ~solid
    n_fluid = int(fluid.sum().item())
    phi_actual = 1.0 - n_fluid / n**3
    a_eff = a_nom * (phi_actual / phi) ** (1.0 / 3.0)
    gap = gap_cells(n, phi)

    k_ref_hasimoto = hasimoto_K(phi)
    k_table = SC_TABLE.get(phi)
    k_design = k_table if k_table is not None else k_ref_hasimoto  # for a_body sizing only

    # a_body sized so that U_s = Re*nu/(2a) under the *first-principles* channel
    # K = F/(6 pi nu a U_s), F = a_body * n_fluid (momentum balance): U_s = a_body n_fluid/(6 pi nu a K)
    a_body = RE_TARGET * NU * NU * 3.0 * math.pi * k_design / n_fluid

    f = None
    hist: list[dict] = []
    start_step = 0
    ck = out / f"ckpt_phi{phi}_N{n}.pt"
    if resume and ck.exists():
        state = torch.load(ck, map_location="cpu", weights_only=False)
        f = state["f"]
        hist = state["hist"]
        start_step = state["step"]
        print(f"[resume] {ck} step={start_step}", flush=True)

    if f is None:
        ones = torch.ones((n, n, n), dtype=dt)
        zeros = torch.zeros((n, n, n), dtype=dt)
        f = equilibrium3d(ones, zeros, zeros, zeros).to(dt)
        del ones, zeros
    mass0 = float(f.sum().item())
    solid_b = solid.unsqueeze(0)

    if max_steps is None:
        max_steps = min(4 * n * n, 200_000)
    if smoke:
        max_steps = min(max_steps, 1500)

    def step(f_t: torch.Tensor) -> torch.Tensor:
        f_t = collide_bgk3d(f_t, TAU)
        f_t = stream3d(f_t)
        f_t = bounce_back_cells_3d(torch.where(solid_b, f_t, apply_body_force_3d(f_t, a_body, c, w)), solid)
        return f_t

    steady_step = None
    recent: list[float] = []
    t0 = time.perf_counter()
    step_i = start_step
    while step_i < max_steps:
        f = step(f)
        step_i += 1
        if step_i % SAMPLE_EVERY == 0:
            with torch.no_grad():
                _, ux, uy, uz = macroscopic3d(f)
                um = float(ux[fluid].mean().item())
                umax = float(torch.sqrt(ux * ux + uy * uy + uz * uz)[fluid].max().item())
                mass = float(f.sum().item())
            hist.append({"step": step_i, "um": um, "umax": umax, "mass": mass})
            if steady_step is None and len(hist) >= DRIFT_SPAN + 1:
                drift = abs(hist[-1]["um"] - hist[-1 - DRIFT_SPAN]["um"]) / abs(hist[-1]["um"])
                recent.append(drift)
                if len(recent) >= STEADY_REPEATS and all(v < DRIFT_TOL for v in recent[-STEADY_REPEATS:]):
                    steady_step = step_i - DRIFT_SPAN * SAMPLE_EVERY
            if step_i % ckpt_every == 0 and not smoke:
                torch.save({"f": f, "hist": hist, "step": step_i}, ck)
        if steady_step is not None and step_i >= steady_step + POST_STEADY:
            break

    wall = time.perf_counter() - t0

    # ---- measurement: force by momentum exchange on the post-stream state of one step
    with torch.no_grad():
        f_pre = collide_bgk3d(f, TAU)
        f_pre = stream3d(f_pre)
        Fvec = momentum_exchange_force(f_pre, solid, c)
        f_pre = bounce_back_cells_3d(
            torch.where(solid_b, f_pre, apply_body_force_3d(f_pre, a_body, c, w)), solid
        )
        del f_pre
        _, ux, uy, uz = macroscopic3d(f)
        um = float(ux[fluid].mean().item())
        umax = float(torch.sqrt(ux * ux + uy * uy + uz * uz)[fluid].max().item())
        uy_absmax = float(uy[fluid].abs().max().item())
        uz_absmax = float(uz[fluid].abs().max().item())

    win = hist[-MEAS_SAMPLES:]
    um_win = sum(h["um"] for h in win) / len(win)
    mass_drift = max(abs(h["mass"] / mass0 - 1.0) for h in hist) if hist else float("nan")

    U_s = (1.0 - phi) * um_win              # superficial velocity, nominal phi
    U_s_act = (1.0 - phi_actual) * um_win
    F_mx = Fvec[0]
    F_inj = a_body * n_fluid                # steady-state momentum injected per step (rho~1)

    def K(F: float, a: float, Us: float) -> float:
        return F / (6.0 * math.pi * NU * a * Us)

    K_mx = K(F_mx, a_nom, U_s)
    K_mb = K(F_inj, a_nom, U_s)
    K_qv = a_body * n**3 / (6.0 * math.pi * NU * a_nom * U_s)      # legacy in-repo channel
    K_mx_act = K(F_mx, a_eff, U_s_act)                              # diagnostic (actual geometry)
    K_ref_act = SC_TABLE.get(phi) if phi in SC_TABLE else None
    if K_ref_act is None:
        K_ref_act = loglog_K(phi)

    ref = SC_TABLE.get(phi)
    rec = {
        "kind": "case",
        "created_utc": utc_now(),
        "phi": phi,
        "N": n,
        "device": device_info(),
        "tau": TAU,
        "nu": NU,
        "a_body": a_body,
        "a_nom": a_nom,
        "a_eff": a_eff,
        "phi_actual": phi_actual,
        "gap_cells": gap,
        "under_resolved": gap < 3.0,
        "n_fluid": n_fluid,
        "max_steps": max_steps,
        "steps_run": step_i,
        "steady_detected": steady_step is not None,
        "steady_step": steady_step,
        "wall_seconds": wall,
        "ms_per_step": 1e3 * wall / max(1, step_i - start_step),
        "mass_drift_rel": mass_drift,
        "um_fluid": um_win,
        "um_fluid_inst": um,
        "umax": umax,
        "Ma": math.sqrt(3.0) * umax,
        "Re_p": U_s * 2.0 * a_nom / NU,
        "U_s": U_s,
        "cross_flow_rel": (uy_absmax + uz_absmax) / abs(um_win) if um_win else None,
        "F_mx_x": F_mx,
        "F_inj_x": F_inj,
        "F_mx_over_F_inj": F_mx / F_inj if F_inj else None,
        "K_mx": K_mx,
        "K_mb": K_mb,
        "K_qv": K_qv,
        "K_mx_actual_geom": K_mx_act,
        "K_hasimoto": k_ref_hasimoto,
        "K_table": ref,
        "K_used_ref": ref if ref is not None else None,
        "interp_diag": None if ref is not None else K_ref_act,
        "err_mx_vs_ref": (K_mb / ref - 1.0) if ref else None,
        "err_mx_vs_hasimoto": K_mb / k_ref_hasimoto - 1.0,
        "err_actualgeom_vs_interp": K_mx_act / K_ref_act - 1.0,
        "hist": hist[-500:],
    }
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"case_phi{phi}_N{n}.json"
    p.write_text(json.dumps(rec, indent=1))
    if ck.exists() and smoke:
        ck.unlink()
    print(json.dumps({k: v for k, v in rec.items() if k != "hist"}, indent=1), flush=True)
    return rec


def load_cases(out: Path) -> dict:
    cases = {}
    for p in sorted(out.glob("case_phi*_N*.json")):
        r = json.loads(p.read_text())
        cases[(r["phi"], r["N"])] = r
    return cases


def make_report(out: Path, figdir: Path | None) -> dict:
    cases = load_cases(out)
    result = {"kind": "result", "created_utc": utc_now(), "points": {}, "thresholds": {
        "tol_finest": TOL_FINEST, "tol_convergence": TOL_CONV, "tol_channel": TOL_CHANNEL}}
    for phi in sorted({k[0] for k in cases}):
        tiers = [cases[(phi, n)] for n in sorted(n for (p, n) in cases if p == phi)]
        tiers.sort(key=lambda r: r["N"])
        ref = tiers[0]["K_table"]
        verdict_ref_bearing = phi in VERDICT_PHI and ref is not None
        finest = tiers[-1]
        conv = None
        if len(tiers) >= 2:
            conv = abs(finest["K_mb"] / tiers[-2]["K_mb"] - 1.0)
        chan = abs(finest["F_mx_over_F_inj"] - 1.0)
        err = (finest["K_mb"] / ref - 1.0) if ref else None
        if not verdict_ref_bearing:
            verdict = "WITHHELD_NO_LOCKED_REF_ROW"
        elif not finest["steady_detected"]:
            verdict = "INCONCLUSIVE_NOT_STEADY"
        elif chan > TOL_CHANNEL:
            verdict = "CHANNEL_INCONSISTENT"
        elif conv is not None and conv > TOL_CONV:
            verdict = "NOT_GRID_CONVERGED"
        elif abs(err) < TOL_FINEST:
            verdict = "PASS"
        else:
            verdict = "FAIL"
        result["points"][str(phi)] = {
            "phi": phi,
            "K_ref_table": ref,
            "K_hasimoto": finest["K_hasimoto"],
            "verdict": verdict,
            "verdict_bearing": bool(verdict_ref_bearing),
            "err_finest": err,
            "convergence_finest_pair": conv,
            "channel_consistency": chan,
            "tiers": [
                {
                    "N": t["N"],
                    "K_mx": t["K_mx"],
                    "K_mb": t["K_mb"],
                    "K_qv": t["K_qv"],
                    "K_hasimoto": t["K_hasimoto"],
                    "K_table": t["K_table"],
                    "err_mx_vs_ref": t["err_mx_vs_ref"],
                    "err_mx_vs_hasimoto": t["err_mx_vs_hasimoto"],
                    "gap_cells": t["gap_cells"],
                    "under_resolved": t["under_resolved"],
                    "phi_actual": t["phi_actual"],
                    "Ma": t["Ma"],
                    "Re_p": t["Re_p"],
                    "mass_drift_rel": t["mass_drift_rel"],
                    "steady_detected": t["steady_detected"],
                    "steps_run": t["steps_run"],
                    "ms_per_step": t["ms_per_step"],
                    "F_mx_over_F_inj": t["F_mx_over_F_inj"],
                    "cross_flow_rel": t["cross_flow_rel"],
                }
                for t in tiers
            ],
        }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    if figdir is not None:
        plot(result, out, figdir)
    print(json.dumps({p: {"verdict": v["verdict"], "err": v["err_finest"],
                          "conv": v["convergence_finest_pair"], "chan": v["channel_consistency"]}
                      for p, v in result["points"].items()}, indent=1))
    return result


def plot(result: dict, out: Path, figdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    colors = {0.125: "tab:blue", 0.45: "tab:red", 0.5: "tab:green"}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for p, v in result["points"].items():
        ns = [t["N"] for t in v["tiers"]]
        ks = [t["K_mx"] for t in v["tiers"]]
        col = colors.get(float(p), "k")
        ax.plot(ns, ks, "o-", color=col, label=f"LBM K (phi={p})")
        if v["K_ref_table"]:
            ax.axhline(v["K_ref_table"], color=col, ls="--", lw=1,
                       label=f"ref table {v['K_ref_table']:.3f} (phi={p})")
        ax.axhline(v["K_hasimoto"], color=col, ls=":", lw=1,
                   label=f"Hasimoto series {v['K_hasimoto']:.2f} (phi={p})")
    ax.set_xlabel("N (cells per lattice constant)")
    ax.set_ylabel(r"$K=F/(6\pi\mu a U_s)$")
    ax.set_title("SC sphere array drag: grid convergence vs reference")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "k_vs_N.png", dpi=150)
    plt.close(fig)

    # error convergence
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for p, v in result["points"].items():
        es = [t["err_mx_vs_ref"] for t in v["tiers"] if t["err_mx_vs_ref"] is not None]
        ns = [t["N"] for t in v["tiers"] if t["err_mx_vs_ref"] is not None]
        if es:
            ax.semilogy(ns, [abs(e) for e in es], "s-", color=colors.get(float(p), "k"), label=f"phi={p}")
    ax.axhline(0.05, color="k", ls="--", lw=1, label="5% criterion")
    ax.set_xlabel("N")
    ax.set_ylabel("|K_sim/K_ref - 1|")
    ax.set_title("Discretisation error vs resolution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(figdir / "error_vs_N.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["case", "ladder", "report"])
    ap.add_argument("--phi", type=float)
    ap.add_argument("--n", type=int)
    ap.add_argument("--out", default=str(HERE.parent / "out"))
    ap.add_argument("--figures", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out)
    if a.cmd == "case":
        assert a.phi is not None and a.n is not None
        run_case(a.phi, a.n, out, a.max_steps, a.smoke, resume=a.resume)
    elif a.cmd == "ladder":
        assert a.phi is not None
        for n in LADDER[a.phi]:
            p = out / f"case_phi{a.phi}_N{n}.json"
            if p.exists() and not a.smoke:
                print(f"[skip] {p} exists", flush=True)
                continue
            run_case(a.phi, n, out, a.max_steps, a.smoke, resume=a.resume)
    else:
        make_report(out, Path(a.figures) if a.figures else None)


if __name__ == "__main__":
    main()