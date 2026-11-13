#!/usr/bin/env python3
"""XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY — reference harness.

Cross-validation of this library's ``u_eq="viscous"`` (τ-mixed, i.e.
viscosity-weighted) two-component MCMP collision against the OSS reference
recorded in platform ``oss_solutions`` entry id=35 (repo
``loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow``, tag
``shan-chen-mcmp-tau-mixed-velocity``, evidence report 9555):

    ``vel_mix = Σ(Vel*Rho/tau)/Σ(Rho/tau)``  (their ``Domain.py:151-153``)

Arms (each writes JSON; ``--arm all`` runs everything):

``parity``     black-box recovery of the mixed velocity the library collision
               actually imposes, compared with the REF-OSS formula on random
               fields (float64 and float32).
``channel``    ideal (``G_12=0``) co-moving two-component channel driven by a
               body force: the continuum limit of the REF-OSS formula predicts a
               mixture viscosity ``ν_eff = Σρ_kν_k/ρ``; the measured parabolic
               profile is compared with that analytic solution (acceptance
               < 15 %), including grid and force sensitivity.
``momentum``   total-momentum conservation of the three velocity-coupling modes
               (``viscous`` / ``mixture`` / ``self``) with unequal τ.
``separation`` control arm reproducing the OSS draft (no non-local Shan-Chen
               force ⇒ no phase separation) versus the library SC arm that has
               the force, over a ``G_12`` scan.

All numbers land in ``results/xval_results.json``; figures in ``figures/``.
Thresholds are in ``predeclared.json`` and are never edited after a run.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.mcmp_tau_mixed import (  # noqa: E402
    CS2,
    channel_poiseuille_profile,
    collision_momentum_probe,
    mixture_effective_viscosity,
    oss_vel_mix,
)
from tensorlbm.multiphase import collide_sc_two_component  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

DTYPE_F64 = torch.float64


def _declared() -> dict:
    raw = json.loads((HERE / "predeclared.json").read_text())
    return {"thresholds": raw["predeclared_thresholds"], "raw": raw}


# ---------------------------------------------------------------------------
# arm 1 — formula parity (REF-OSS vs library collision, black box)
# ---------------------------------------------------------------------------

PARITY_CASES = [
    # (rho1, rho2, u1, u2, tau1, tau2)
    (1.0, 1.0, 0.05, -0.02, 1.0, 0.70),
    (0.3, 1.7, 0.04, 0.01, 1.0, 0.60),
    (2.0, 0.2, -0.03, 0.06, 1.20, 0.80),
    (1.0, 1.0, 0.01, 0.01, 1.0, 1.00),  # equal-τ control: mixing must vanish
    (0.5, 0.5, 0.02, -0.02, 3.0, 0.55),  # viscosity ratio 16.7:1
]


def arm_parity(decl: dict) -> dict:
    thr64 = decl["thresholds"]["parity"]["max_rel_dev_float64"]
    thr32 = decl["thresholds"]["parity"]["max_rel_dev_float32"]
    cases = []
    for dtype in (torch.float64, torch.float32):
        for seed, (r1, r2, u1, u2, t1, t2) in enumerate(PARITY_CASES):
            torch.manual_seed(1000 + seed)
            ny, nx = 10, 8
            noise = 0.1
            rho1 = (r1 * (1.0 + noise * torch.rand((ny, nx), dtype=dtype))).to(dtype)
            rho2 = (r2 * (1.0 + noise * torch.rand((ny, nx), dtype=dtype))).to(dtype)
            ux1 = (u1 * torch.rand((ny, nx), dtype=dtype)).to(dtype)
            uy1 = (0.3 * u1 * torch.rand((ny, nx), dtype=dtype)).to(dtype)
            ux2 = (u2 * torch.rand((ny, nx), dtype=dtype)).to(dtype)
            uy2 = (0.3 * u2 * torch.rand((ny, nx), dtype=dtype)).to(dtype)
            f1 = equilibrium(rho1, ux1, uy1)
            f2 = equilibrium(rho2, ux2, uy2)
            probe = collision_momentum_probe(f1, f2, t1, t2)
            thr = thr64 if dtype is torch.float64 else thr32
            # the reference must also be reproducible from the pre-collision state
            refx = oss_vel_mix([rho1, rho2], [ux1, ux2], [t1, t2])
            cases.append(
                {
                    "dtype": str(dtype).replace("torch.", ""),
                    "seed": seed,
                    "rho": [r1, r2],
                    "u": [u1, u2],
                    "tau": [t1, t2],
                    "vel_mix_ref": float(refx.abs().mean()),
                    **probe,
                    "pass": probe["max_rel_dev"] <= thr,
                }
            )
    worst64 = max(c["max_rel_dev"] for c in cases if c["dtype"] == "float64")
    worst32 = max(c["max_rel_dev"] for c in cases if c["dtype"] == "float32")
    return {
        "arm": "parity",
        "ref": "oss_solutions id=35 vel_mix = Σ(Vel·Rho/τ)/Σ(Rho/τ) (Domain.py:151-153)",
        "threshold": {"float64": thr64, "float32": thr32},
        "cases": cases,
        "worst_rel_dev_float64": worst64,
        "worst_rel_dev_float32": worst32,
        "verdict": "PASS" if (worst64 <= thr64 and worst32 <= thr32) else "FAIL",
    }


# ---------------------------------------------------------------------------
# arm 2 — ideal co-moving mixture channel (continuum limit of REF-OSS formula)
# ---------------------------------------------------------------------------


def _run_channel(
    ny: int,
    gx: float,
    tau1: float,
    tau2: float,
    steps: int,
    *,
    nx: int = 4,
    warm_start: bool = False,
    u_eq: str = "viscous",
    force_scheme: str = "guo",
    dtype: torch.dtype = DTYPE_F64,
    probe_every: int = 500,
) -> dict:
    """Body-force driven channel of an ideal (G_12 = 0) two-component mixture."""
    rho1 = torch.ones((ny, nx), dtype=dtype)
    rho2 = torch.ones((ny, nx), dtype=dtype)
    zero = torch.zeros_like(rho1)
    ny_ana = ny
    nu_eff = mixture_effective_viscosity(tau1, tau2, 1.0, 1.0)
    analytic = channel_poiseuille_profile(ny_ana, gx, nu_eff, nx=nx, dtype=dtype)
    if warm_start:
        u0 = analytic
        f1 = equilibrium(rho1, u0, zero)
        f2 = equilibrium(rho2, u0, zero)
    else:
        f1 = equilibrium(rho1, zero, zero)
        f2 = equilibrium(rho2, zero, zero)
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[0, :] = True
    solid[-1, :] = True
    mass0 = float(rho1.sum() + rho2.sum())

    hist = []
    u_prev = None
    for it in range(steps):
        f1, f2 = collide_sc_two_component(
            f1,
            f2,
            G_12=0.0,
            tau1=tau1,
            tau2=tau2,
            gx=gx,
            gy=0.0,
            solid_mask=solid,
            u_eq=u_eq,
            force_scheme=force_scheme,
        )
        f1 = bounce_back_cells(stream(f1), solid)
        f2 = bounce_back_cells(stream(f2), solid)
        if (it + 1) % probe_every == 0 or it == steps - 1:
            r1, ux1, _ = macroscopic(f1)
            r2, ux2, _ = macroscopic(f2)
            um = (r1 * ux1 + r2 * ux2) / (r1 + r2)
            prof = um[:, nx // 2]
            resid = (
                float((prof - u_prev).abs().max() / max(float(prof.abs().max()), 1e-300))
                if u_prev is not None
                else float("nan")
            )
            u_prev = prof
            hist.append({"step": it + 1, "u_center": float(prof.max()), "resid": resid})

    r1, ux1, _ = macroscopic(f1)
    r2, ux2, _ = macroscopic(f2)
    um = ((r1 * ux1 + r2 * ux2) / (r1 + r2))[:, nx // 2]
    fluid = slice(1, ny - 1)
    ana = analytic[:, 0]
    err = (um - ana)[fluid]
    denom = float(ana[fluid].abs().max())
    l2 = math.sqrt(float((err**2).sum()) / float((ana[fluid] ** 2).sum()))
    # direct measurement of the mixture viscosity, robust to the wall-adjacent
    # rows: fit the full quadratic A0 + A1 y + A2 y^2 on the fluid rows; the
    # steady solution u = gx/(2 nu) y (H - y) gives A2 = -gx/(2 nu), and the
    # roots give the effective no-slip planes (wall-position diagnostic).
    ys = torch.arange(1, ny - 1, dtype=dtype)
    uu = um[1:-1]
    M = torch.stack([torch.ones_like(ys), ys, ys * ys], dim=1)
    coef = torch.linalg.lstsq(M, uu).solution
    a2 = float(coef[2])
    nu_meas = -gx / (2.0 * a2) if a2 != 0.0 else float("nan")
    wall_left = wall_right = float("nan")
    if a2 != 0.0:
        a0, a1 = float(coef[0]), float(coef[1])
        cc = -a2
        ssum, prod = a1 / cc, -a0 / cc
        disc = ssum * ssum - 4.0 * prod
        if disc >= 0.0:
            wall_left = (ssum - math.sqrt(disc)) / 2.0
            wall_right = (ssum + math.sqrt(disc)) / 2.0

    nu_tau_mean = CS2 * ((tau1 + tau2) / 2.0 - 0.5)
    nu_harm = 2.0 / (1.0 / (CS2 * (tau1 - 0.5)) + 1.0 / (CS2 * (tau2 - 0.5)))
    alt = {}
    for name, nu in (
        ("rho_weighted_arithmetic", nu_eff),
        ("tau_arithmetic", nu_tau_mean),
        ("harmonic", nu_harm),
    ):
        alt[name] = {
            "nu": nu,
            "u_center": float(channel_poiseuille_profile(ny, gx, nu, dtype=dtype)[:, 0].max()),
        }
    return {
        "ny": ny,
        "nx": nx,
        "gx": gx,
        "tau": [tau1, tau2],
        "steps": steps,
        "u_eq": u_eq,
        "force_scheme": force_scheme,
        "warm_start": warm_start,
        "nu_eff_pred": nu_eff,
        "nu_meas": nu_meas,
        "wall_left": wall_left,
        "wall_right": wall_right,
        "profile_lsq_resid": float((M @ coef - uu).abs().max()),
        "nu_rel_err": abs(nu_meas - nu_eff) / nu_eff,
        "u_center_got": float(um.max()),
        "u_center_analytic": float(ana.abs().max()),
        "rel_err_max": float(err.abs().max() / denom),
        "rel_err_l2": l2,
        "mass_drift": abs(float(r1.sum() + r2.sum()) - mass0) / mass0,
        "steady_resid": hist[-1]["resid"],
        "history": hist,
        "alternatives": alt,
        "profile_got": um.tolist(),
        "profile_analytic": ana.tolist(),
    }


CHANNEL_CASES = [
    # tag, ny, gx, tau1, tau2, steps, warm
    ("primary", 33, 1e-6, 1.0, 0.70, 12000, False),
    ("refine_17", 17, 1e-6, 1.0, 0.70, 8000, False),
    ("refine_65", 65, 1e-6, 1.0, 0.70, 25000, False),
    ("tau_equal_control", 33, 1e-6, 1.0, 1.00, 12000, False),
    ("tau_ratio_1p11", 33, 1e-6, 1.0, 0.90, 12000, False),
    ("tau_ratio_3", 33, 1e-6, 1.0, 0.60, 12000, False),
    ("force_half", 33, 5e-7, 1.0, 0.70, 12000, False),
    ("force_double", 33, 2e-6, 1.0, 0.70, 12000, False),
]


def arm_channel(decl: dict) -> dict:
    th = decl["thresholds"]["channel"]
    thr = th["rel_err_max"]
    rows = []
    for tag, ny, gx, t1, t2, steps, warm in CHANNEL_CASES:
        t0 = time.time()
        row = _run_channel(ny, gx, t1, t2, steps, warm_start=warm)
        row["tag"] = tag
        row["wall_seconds"] = round(time.time() - t0, 1)
        row["pass"] = (
            row["rel_err_max"] <= thr
            and row["nu_rel_err"] <= th["nu_rel_err_max"]
            and row["steady_resid"] <= th["steady_resid_max"]
        )
        rows.append(row)
        print(
            f"[channel] {tag:18s} ny={ny:3d} tau=({t1},{t2}) "
            f"rel_max={100*row['rel_err_max']:6.2f}% L2={100*row['rel_err_l2']:6.2f}% "
            f"nu={row['nu_meas']:.5f} vs {row['nu_eff_pred']:.5f} "
            f"({100*row['nu_rel_err']:.2f}%) {row['wall_seconds']}s",
            flush=True,
        )
    worst = max(r["rel_err_max"] for r in rows)
    return {
        "arm": "channel",
        "ref": "REF-OSS continuum limit: nu_eff = Σρ_k ν_k/ρ; analytic plane Poiseuille, H=ny-1",
        "thresholds": th,
        "cases": rows,
        "worst_rel_err_max": worst,
        "verdict": "PASS" if worst <= thr else "FAIL",
    }


# ---------------------------------------------------------------------------
# arm 3 — momentum conservation of the three coupling modes
# ---------------------------------------------------------------------------


def arm_momentum(decl: dict, n: int = 16, steps: int = 200, tau=(1.0, 0.7)) -> dict:
    thr = decl["thresholds"]["momentum"]["max_rel_drift"]
    rows = []
    for mode in ("viscous", "mixture", "self"):
        rho = torch.ones((n, n), dtype=DTYPE_F64)
        u1 = torch.full((n, n), 0.02, dtype=DTYPE_F64)
        u2 = torch.full((n, n), -0.01, dtype=DTYPE_F64)
        zero = torch.zeros_like(rho)
        f1 = equilibrium(rho, u1, zero)
        f2 = equilibrium(rho, u2, zero)
        jx0 = float((rho * u1 + rho * u2).sum())
        for _ in range(steps):
            f1, f2 = collide_sc_two_component(
                f1, f2, G_12=0.0, tau1=tau[0], tau2=tau[1], u_eq=mode, force_scheme="shift"
            )
            f1 = stream(f1)
            f2 = stream(f2)
        r1, ux1, _ = macroscopic(f1)
        r2, ux2, _ = macroscopic(f2)
        jx1 = float((r1 * ux1 + r2 * ux2).sum())
        drift = abs(jx1 - jx0) / abs(jx0)
        rows.append(
            {
                "u_eq": mode,
                "momentum0": jx0,
                "momentum1": jx1,
                "rel_drift": drift,
                "per_step": drift / steps,
                "pass": drift <= thr,
            }
        )
        print(f"[momentum] {mode:8s} rel drift {drift:.3e} over {steps} steps", flush=True)
    return {
        "arm": "momentum",
        "ref": "docstring claim of collide_sc_two_component: tau-mixed coupling conserves "
        "total momentum exactly for unequal tau",
        "tau": list(tau),
        "steps": steps,
        "threshold": thr,
        "cases": rows,
        "verdict": "PASS" if rows[0]["pass"] else "FAIL",
    }


# ---------------------------------------------------------------------------
# arm 4 — separation: OSS-equivalent control (no SC force) vs library SC arm
# ---------------------------------------------------------------------------


def _run_separation(
    n: int,
    G_12: float,
    tau1: float,
    tau2: float,
    steps: int,
    *,
    seed: int = 7,
    u_eq: str = "viscous",
) -> dict:
    g = torch.Generator().manual_seed(seed)
    pert = 1.0 + 0.01 * (2.0 * torch.rand((n, n), dtype=DTYPE_F64, generator=g) - 1.0)
    rho1 = pert.clone()
    rho2 = (2.0 - pert)
    zero = torch.zeros_like(rho1)
    f1 = equilibrium(rho1, zero, zero)
    f2 = equilibrium(rho2, zero, zero)
    c0 = float(rho1.max() - rho1.min())
    m0 = float(rho1.sum() + rho2.sum())
    diverged = False
    umax = 0.0
    for _ in range(steps):
        f1, f2 = collide_sc_two_component(
            f1, f2, G_12=G_12, tau1=tau1, tau2=tau2, u_eq=u_eq, force_scheme="guo"
        )
        f1 = stream(f1)
        f2 = stream(f2)
        if not torch.isfinite(f1).all() or not torch.isfinite(f2).all():
            diverged = True
            break
        r1, ux1, uy1 = macroscopic(f1)
        umax = max(umax, float((ux1.abs().max() + uy1.abs().max())))
    if diverged:
        return {
            "n": n,
            "G_12": G_12,
            "tau": [tau1, tau2],
            "steps": steps,
            "diverged": True,
            "u_eq": u_eq,
        }
    r1, ux1, uy1 = macroscopic(f1)
    r2, ux2, uy2 = macroscopic(f2)
    return {
        "n": n,
        "G_12": G_12,
        "tau": [tau1, tau2],
        "steps": steps,
        "u_eq": u_eq,
        "diverged": False,
        "contrast_initial": c0,
        "contrast_final": float(r1.max() - r1.min()),
        "contrast_ratio": float((r1.max() - r1.min()) / c0),
        "rho1_mean": float(r1.mean()),
        "rho1_min": float(r1.min()),
        "rho1_max": float(r1.max()),
        "mass_drift": abs(float(r1.sum() + r2.sum()) - m0) / m0,
        "max_speed": umax,
        "final_speed": float(max(ux1.abs().max(), uy1.abs().max())),
        "field_rho1_row": r1[n // 2, :].tolist(),
    }


SEPARATION_G = [-0.4, -0.8, -1.2, -1.6, -2.0]


def arm_separation(decl: dict) -> dict:
    thr = decl["thresholds"]["separation"]
    rows = []
    # control: OSS-equivalent (no non-local Shan-Chen force)
    control = _run_separation(48, 0.0, 1.0, 0.7, thr["steps"])
    control["tag"] = "oss_control_no_force"
    control["pass"] = control["contrast_final"] <= thr["control_max_contrast"]
    rows.append(control)
    print(
        f"[separation] G=0.0 (OSS control) contrast={control['contrast_final']:.3e} "
        f"({control['contrast_ratio']:.2f}x) max|u|={control['max_speed']:.2e}",
        flush=True,
    )
    for G in SEPARATION_G:
        t0 = time.time()
        row = _run_separation(48, G, 1.0, 0.7, thr["steps"])
        row["tag"] = f"sc_G{G:+.1f}"
        row["pass"] = (not row["diverged"]) and row["contrast_final"] >= thr["min_contrast"]
        row["wall_seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        if row["diverged"]:
            print(f"[separation] G={G:+.1f} DIVERGED", flush=True)
        else:
            print(
                f"[separation] G={G:+.1f} contrast={row['contrast_final']:.3e} "
                f"({row['contrast_ratio']:.1f}x) max|u|={row['max_speed']:.2e} "
                f"mass_drift={row['mass_drift']:.2e} {row['wall_seconds']}s",
                flush=True,
            )
    ok = [r for r in rows if r["tag"].startswith("sc_") and r.get("pass")]
    return {
        "arm": "separation",
        "ref": "oss_solutions id=35 conventions: no non-local Shan-Chen force ⇒ no phase "
        "separation (contrast stays at the 1% initial level)",
        "thresholds": thr,
        "cases": rows,
        "separating_G": [r["G_12"] for r in ok],
        "verdict": "PASS" if (rows[0]["pass"] and len(ok) > 0) else "FAIL",
    }


# ---------------------------------------------------------------------------
# figures + environment
# ---------------------------------------------------------------------------


def make_figures(res: dict, outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    made = []
    ch = res.get("channel")
    if ch:
        by_tag = {c["tag"]: c for c in ch["cases"]}
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        p = by_tag["primary"]
        ax[0].plot(p["profile_analytic"], color="k", lw=1.6, label="analytic (ν_eff = Σρν/ρ)")
        ax[0].plot(p["profile_got"], "o", ms=3.5, color="tab:red", label="TensorLBM u_eq=viscous")
        for name, style in (("tau_arithmetic", "--"), ("harmonic", ":")):
            alt = p["alternatives"][name]
            ys = channel_poiseuille_profile(p["ny"], p["gx"], alt["nu"])
            ax[0].plot(ys.numpy(), style, lw=1.2, label=f"analytic if ν={name}")
        ax[0].set_xlabel("y (lattice)")
        ax[0].set_ylabel("u_x")
        ax[0].set_title(f"co-moving ideal mixture channel, τ=(1.0,0.7), ny={p['ny']}")
        ax[0].legend(fontsize=7)
        ax[0].grid(alpha=0.3)
        labels, errs, cols = [], [], []
        for c in ch["cases"]:
            labels.append(c["tag"])
            errs.append(100 * c["rel_err_max"])
            cols.append("tab:green" if c["pass"] else "tab:red")
        ax[1].bar(range(len(errs)), errs, color=cols)
        ax[1].axhline(15.0, color="k", ls="--", lw=1.0, label="15% acceptance")
        ax[1].set_xticks(range(len(labels)))
        ax[1].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax[1].set_ylabel("max rel. error of u profile [%]")
        ax[1].set_title("channel arm: all cases vs analytic")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        f1 = outdir / "channel_profiles.png"
        fig.savefig(f1, dpi=150)
        plt.close(fig)
        made.append(str(f1))

        # convergence + viscosity measurement
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ref = [by_tag[t] for t in ("refine_17", "primary", "refine_65") if t in by_tag]
        ax[0].plot([r["ny"] for r in ref], [100 * r["rel_err_l2"] for r in ref], "o-")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("ny")
        ax[0].set_ylabel("L2 rel. error [%]")
        ax[0].set_title("grid refinement (τ=(1.0,0.7))")
        ax[0].grid(alpha=0.3, which="both")
        taus = [c for c in ch["cases"] if c["tag"].startswith("tau_")]
        sets = [(1.0, 1.0), (1.0, 0.9), (1.0, 0.7), (1.0, 0.6)]
        pred = [mixture_effective_viscosity(a, b) for a, b in sets]
        ax[1].plot(pred, pred, "k--", lw=1.0, label="ν_eff = Σρν/ρ")
        for c in taus:
            ax[1].plot(c["nu_eff_pred"], c["nu_meas"], "o", ms=5)
            ax[1].annotate(f"τ={tuple(c['tau'])}", (c["nu_eff_pred"], c["nu_meas"]), fontsize=6)
        ax[1].set_xlabel("predicted ν_eff")
        ax[1].set_ylabel("measured ν from profile")
        ax[1].set_title("mixture viscosity: measured vs REF-OSS continuum limit")
        ax[1].legend(fontsize=7)
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        f2 = outdir / "convergence_viscosity.png"
        fig.savefig(f2, dpi=150)
        plt.close(fig)
        made.append(str(f2))

    sp = res.get("separation")
    if sp:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ok = [c for c in sp["cases"] if not c.get("diverged")]
        ax[0].semilogy(
            [c["G_12"] for c in ok],
            [max(c["contrast_final"], 1e-12) for c in ok],
            "o-",
        )
        ax[0].axhline(sp["thresholds"]["min_contrast"], color="k", ls="--", lw=1.0,
                      label=f"min contrast {sp['thresholds']['min_contrast']}")
        ax[0].set_xlabel("G_12 (negative = repulsive)")
        ax[0].set_ylabel("density contrast max−min (ρ1)")
        ax[0].set_title("separation arm (48×48, 4000 steps, τ=(1.0,0.7))")
        ax[0].legend(fontsize=7)
        ax[0].grid(alpha=0.3, which="both")
        sel = [c for c in ok if c["tag"] in ("oss_control_no_force", "sc_G-1.2")]
        for c in sel:
            ax[1].plot(c["field_rho1_row"], label=f"{c['tag']}: contrast {c['contrast_final']:.2f}")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("ρ1 (mid row)")
        ax[1].set_title("density field: OSS control vs SC arm")
        ax[1].legend(fontsize=7)
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        f3 = outdir / "separation.png"
        fig.savefig(f3, dpi=150)
        plt.close(fig)
        made.append(str(f3))

    mo = res.get("momentum")
    if mo:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.bar(
            [c["u_eq"] for c in mo["cases"]],
            [max(c["rel_drift"], 1e-17) for c in mo["cases"]],
            color=["tab:green" if c["pass"] else "tab:red" for c in mo["cases"]],
        )
        ax.set_yscale("log")
        ax.set_ylabel("relative total-momentum drift")
        ax.set_title("momentum conservation, τ=(1.0,0.7), 200 steps")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        f4 = outdir / "momentum.png"
        fig.savefig(f4, dpi=150)
        plt.close(fig)
        made.append(str(f4))
    return made


def environment() -> dict:
    import subprocess

    try:
        sha = (
            subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        sha = "unknown"
    return {
        "torch": torch.__version__,
        "numpy": __import__("numpy").__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "git_head": sha,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all", choices=["all", "parity", "channel", "momentum", "separation"])
    ap.add_argument("--out", default=str(HERE / "results" / "xval_results.json"))
    ap.add_argument("--figures-dir", default=str(HERE / "figures"))
    args = ap.parse_args()

    decl = _declared()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {}
    if out.exists():
        try:
            res = json.loads(out.read_text())
        except Exception:
            res = {}
    res["mission"] = 18327
    res["anchor"] = "XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY"
    res["environment"] = environment()
    res["predeclared"] = decl["raw"]
    t0 = time.time()
    if args.arm in ("all", "parity"):
        res["parity"] = arm_parity(decl)
        print("[parity]", res["parity"]["verdict"], res["parity"]["worst_rel_dev_float64"])
    if args.arm in ("all", "momentum"):
        res["momentum"] = arm_momentum(decl)
        print("[momentum]", res["momentum"]["verdict"])
    if args.arm in ("all", "channel"):
        res["channel"] = arm_channel(decl)
        print("[channel]", res["channel"]["verdict"], res["channel"]["worst_rel_err_max"])
    if args.arm in ("all", "separation"):
        res["separation"] = arm_separation(decl)
        print("[separation]", res["separation"]["verdict"])
    res["wall_seconds"] = round(time.time() - t0, 1)
    verdicts = {k: res[k]["verdict"] for k in ("parity", "momentum", "channel", "separation") if k in res}
    res["verdicts"] = verdicts
    res["overall"] = "PASS" if all(v == "PASS" for v in verdicts.values()) else "FAIL"
    figs = make_figures(res, Path(args.figures_dir))
    res["figures"] = figs
    out.write_text(json.dumps(res, indent=2))
    print("overall:", res["overall"], "->", out)
    return 0 if res["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())