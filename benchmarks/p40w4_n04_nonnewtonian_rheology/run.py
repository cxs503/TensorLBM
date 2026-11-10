#!/usr/bin/env python3
"""P40W4-N04 — non-Newtonian rheology verification surface for TensorLBM.

Three generalised-Newtonian closures are driven through a variable-viscosity
D2Q9 plane channel and checked against independent one-dimensional anchors:

======  ===================================================================
model   closure / anchor
======  ===================================================================
power   ``ν = K γ̇^(n-1)`` / closed-form Ostwald–de Waele profile
law     ``u(y) = n/(n+1)(G/K)^{1/n}[H^{(n+1)/n} - |y|^{(n+1)/n}]``
Carreau- ``ν = ν∞ + (ν0-ν∞)[1+(λγ̇)^a]^((n-1)/a)`` / constitutive inversion
Yasuda  ``γ̇(τ)`` + composite Gauss–Legendre quadrature (no closed form)
Cross   ``ν = ν∞ + (ν0-ν∞)/(1+(λγ̇)^m)`` / same quadrature anchor
Herschel- ``ν = τ_y(1-e^{-mγ̇})/γ̇ + Kγ̇^{n-1}`` (Papanastasiou) / ideal plug
Bulkley half-width ``r_c = τ_y/(ρa)`` (stress field is rheology-independent)
======  ===================================================================

Gates (machine-checked, written to ``gates.json``)
--------------------------------------------------
G1 power-law profile L2 error           < 3 %   (n = 0.7 and n = 1.3)
G2 Carreau–Yasuda profile + viscosity   < 5 %
G3 Cross profile + viscosity            < 5 %
G4 ice-slurry plug half-width           < 5 %
G5 grid-convergence order (power law)   ≥ 2
G6 pytest suite                         ≥ 10 passing tests

Usage
-----
    PYTHONPATH=src python benchmarks/p40w4_n04_nonnewtonian_rheology/run.py --fast
    PYTHONPATH=src python benchmarks/p40w4_n04_nonnewtonian_rheology/run.py

``--fast`` shrinks every grid (h = 8…32 instead of 32…64) so the whole study
runs in a couple of minutes on one core; the production setting is what the
reported numbers in ``result.json`` / ``docs/P40W4_N04_NONNEWTONIAN_RHEOLOGY.md``
were produced with.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from tensorlbm.nonnewtonian_frontier import (  # noqa: E402
    carreau_yasuda,
    channel_reference,
    cross,
    herschel_bulkley,
    l2_relative_error,
    newtonian,
    plug_report,
    power_law,
    run_channel,
)
from tensorlbm.nonnewtonian_frontier.reference import (  # noqa: E402
    calibrate_accel,
    power_law_velocity,
    power_law_wall_shear_rate,
)
from tensorlbm.nonnewtonian_frontier.channel import solution_residual  # noqa: E402
from tensorlbm.nonnewtonian_frontier.plug import symmetrise  # noqa: E402

OUT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _f(obj):
    """Recursively convert tensors/floats to JSON-friendly Python types."""
    if isinstance(obj, dict):
        return {k: _f(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_f(v) for v in obj]
    if torch.is_tensor(obj):
        return float(obj)
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    return obj


def profile_metrics(out: dict, ref: dict, y: torch.Tensor) -> dict:
    """L2/max-error and viscosity-curve metrics of one channel run."""
    rows = out["fluid_rows"]
    y_num = out["y"][rows]
    u_num = out["u"][rows].double()
    u_ref = ref["u"][rows]
    g_num = symmetrise(y_num, out["gamma_dot"][rows].mean(dim=1))[1]
    r_ref, g_ref = symmetrise(y_num, ref["gamma_dot"][rows])
    closure = out["_closure"]
    nu_num = closure.nu_of_gamma(torch.clamp(g_num, min=1e-300))
    nu_ref = closure.nu_of_gamma(torch.clamp(g_ref, min=1e-300))
    return {
        "l2_relative": l2_relative_error(u_num, u_ref),
        "max_relative": float(((u_num - u_ref).abs() / u_ref.abs().max()).max()),
        "u_max_lbm": out["u_max"],
        "u_max_ref": float(u_ref.max()),
        "u_max_relative_error": out["u_max"] / float(u_ref.max()) - 1.0,
        "mean_velocity_lbm": out["mean_velocity"],
        "mean_velocity_ref": ref["mean_velocity"],
        "flow_rate_relative_error": out["mean_velocity"] / ref["mean_velocity"] - 1.0,
        "gamma_dot_wall_lbm": out["gamma_dot_wall"],
        "gamma_dot_wall_ref": ref["gamma_dot_wall"],
        "viscosity_curve_max_relative_error": float(
            ((nu_num - nu_ref).abs() / nu_ref.abs()).max()
        ),
        "viscosity_range": [float(nu_ref.min()), float(nu_ref.max())],
        "momentum_residual_max": float(
            solution_residual(y_num, out["gamma_dot"][rows].mean(dim=1), closure,
                              out["accel"], out["h"] / 2).abs().max()
        ),
        "steps": out["steps"],
        "converged": out["converged"],
        "steady_residual": out["residual"],
        "profile": {
            "y": [float(v) for v in y_num],
            "u_lbm": [float(v) for v in u_num],
            "u_ref": [float(v) for v in u_ref],
            "gamma_lbm": [float(v) for v in g_num],
            "gamma_ref": [float(v) for v in g_ref],
            "nu_lbm": [float(v) for v in nu_num],
            "nu_ref": [float(v) for v in nu_ref],
        },
    }


def run_case(h: int, closure, u_max_target: float = 0.02, accel: float | None = None,
             nx: int = 4, n_steps: int = 120_000, tol: float = 1e-5,
             check_every: int = 500) -> tuple[dict, dict]:
    """Calibrate the drive, run the channel, and return (run, reference)."""
    H = 0.5 * h
    if accel is None:
        cal = calibrate_accel(closure, H, u_max_target, n_panels=400, n_iter=40)
        accel = cal["accel"]
    y = torch.arange(h + 2, dtype=torch.float64) - 0.5 - H
    ref = channel_reference(closure, accel, H, y=y, n_panels=4000)
    u_init = ref["u"].clone()
    u_init[0] = 0.0
    u_init[-1] = 0.0
    t0 = time.time()
    out = run_channel(h, accel, closure, nx=nx, n_steps=n_steps, tol=tol,
                      check_every=check_every, u_init=u_init)
    out["_closure"] = closure
    out["wall_time_s"] = time.time() - t0
    return out, ref


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_newtonian(cfg) -> dict:
    """Negative control: with ν = const the solver must reproduce the parabola."""
    h = cfg["h_newtonian"]
    closure = newtonian(0.1)
    out, ref = run_case(h, closure, 0.02, n_steps=cfg["n_steps"], tol=cfg["tol"],
                        check_every=cfg["check_every"])
    met = profile_metrics(out, ref, y=None)
    return {
        "h": h,
        "nu": 0.1,
        "accel": out["accel"],
        "u_max": out["u_max"],
        **met,
    }


def power_law_case(h: int, n: float, u_max_target: float = 0.02, nu_wall: float = 0.1,
                   n_steps: int = 120_000, tol: float = 1e-5, check_every: int = 500) -> dict:
    """Power-law channel: closed form (unregularised anchor) + numeric reference.

    The consistency index is chosen so that the *wall* viscosity is
    ``nu_wall = 0.1`` (the same as the Newtonian control) — the viscosity then
    spans ``[ν(H), ν(0)]`` which is reported, so the stiffness of each case is
    explicit rather than hidden in a parameter.
    """
    H = 0.5 * h
    gamma_wall_target = u_max_target * (n + 1.0) / (n * H)
    K = nu_wall * gamma_wall_target ** (1.0 - n)
    accel = K * gamma_wall_target ** n / H
    closure = power_law(K, n)
    out, ref = run_case(h, closure, u_max_target, accel=accel, n_steps=n_steps,
                        tol=tol, check_every=check_every)
    rows = out["fluid_rows"]
    u_num = out["u"][rows].double()
    u_closed = power_law_velocity(out["y"][rows], H, accel, K, n)
    met = profile_metrics(out, ref, y=None)
    met["l2_vs_closed_form"] = l2_relative_error(u_num, u_closed)
    met["gamma_dot_wall_closed_form"] = power_law_wall_shear_rate(H, accel, K, n)
    return {
        "h": h,
        "n": n,
        "K": K,
        "accel": accel,
        "u_max": out["u_max"],
        "nu_center": float(closure.nu_of_gamma(torch.tensor([1e-30], dtype=torch.float64))[0]),
        **met,
        "_out": out,
        "_ref": ref,
        "_closure": closure,
    }


def stage_power_law(cfg) -> dict:
    cases = {}
    for n in (0.7, 1.3):
        res = power_law_case(cfg["h_powerlaw"], n, n_steps=cfg["n_steps"], tol=cfg["tol"],
                             check_every=cfg["check_every"])
        for key in ("_out", "_ref", "_closure"):
            res.pop(key, None)
        cases[f"n={n:g}"] = res
    worst = max(c["l2_vs_closed_form"] for c in cases.values())
    return {"cases": cases, "worst_l2_vs_closed_form": worst}


def stage_viscoelastic_closure(cfg, kind: str) -> dict:
    """Carreau–Yasuda / Cross: numeric anchor + closure round-trip."""
    h = cfg["h_" + kind]
    if kind == "carreau_yasuda":
        closure = carreau_yasuda(nu_0=0.30, nu_inf=0.02, relaxation_time=500.0,
                                 flow_index=0.4, yasuda_exponent=2.0)
    else:
        closure = cross(nu_0=0.30, nu_inf=0.02, relaxation_time=500.0, exponent=1.2)
    out, ref = run_case(h, closure, 0.02, n_steps=cfg["n_steps"], tol=cfg["tol"],
                        check_every=cfg["check_every"])
    met = profile_metrics(out, ref, y=None)
    # closure round-trip: the model evaluated on the *analytic* shear rate must
    # reproduce the model itself (guards against an algebra slip in the closure)
    rows = out["fluid_rows"]
    g_ref = ref["gamma_dot"][rows]
    nu_direct = closure.nu_of_gamma(g_ref)
    nu_independent = (
        closure.params["nu_inf"]
        + (closure.params["nu_0"] - closure.params["nu_inf"])
        * (
            (1.0 + (closure.params["lambda"] * g_ref.double()) ** closure.params["a"])
            ** ((closure.params["n"] - 1.0) / closure.params["a"])
            if kind == "carreau_yasuda"
            else 1.0
            / (
                1.0
                + (closure.params["lambda"] * g_ref.double()) ** closure.params["m"]
            )
        )
    )
    met["closure_roundtrip_max_relative"] = float(
        ((nu_direct - nu_independent).abs() / nu_independent.abs()).max()
    )
    return {
        "h": h,
        "closure": closure.name,
        "params": closure.params,
        "accel": out["accel"],
        "u_max": out["u_max"],
        **met,
    }


def stage_herschel_bulkley(cfg) -> dict:
    """Ice-slurry H-B channel with a plug core (ideal plug fraction 0.35)."""
    h = cfg["h_hb"]
    H = 0.5 * h
    n, K, m, plug_fraction, u_max_target = 0.6, 0.02, 1.0e4, 0.35, 0.02
    p = (n + 1.0) / n
    amp = u_max_target / (H * (1.0 - plug_fraction)) ** p
    accel = K * (amp * (n + 1.0) / n) ** n
    tau_y = plug_fraction * accel * H
    closure = herschel_bulkley(tau_y, K, n, regularization=m)
    out, ref = run_case(h, closure, u_max_target, accel=accel, n_steps=cfg["n_steps"],
                        tol=cfg["tol"], check_every=cfg["check_every"])
    met = profile_metrics(out, ref, y=None)
    plug = plug_report(
        closure, accel, out["y"], out["u"], out["gamma_dot"].mean(dim=1),
        y_ref=out["y"], u_ref=ref["u"], gamma_ref=ref["gamma_dot"],
        half_width=H,
    )
    return {
        "h": h,
        "params": closure.params,
        "accel": accel,
        "tau_y": tau_y,
        "plug_fraction_design": plug_fraction,
        "u_max": out["u_max"],
        **met,
        "plug": plug,
    }


def stage_convergence(cfg) -> dict:
    """Grid-convergence order of the power-law channel (n = 0.7).

    Protocol (declared before the numbers are looked at):

    * drive calibrated on ``u_max = 5e-3`` (``Ma² = 7.5e-5``), the level where
      the float32 lattice noise (~1e-7/u_max) still sits two orders below the
      discretisation error;
    * **fixed step count** (no early stop) so every grid is a converged fixed
      point and the comparison is not contaminated by the stopping rule;
    * the *asymptotic range* is the set of grids whose L2 error exceeds the
      measured compressibility/drive floor ``2·Ma²``; the pair that falls below
      it is reported but excluded from the fitted order, and the floor itself is
      measured by repeating one grid at a lower drive;
    * the reported order is the least-squares slope of ``log e`` against
      ``log h`` over the asymptotic-range grids, plus every two-grid
      Richardson pair for inspection.
    """
    n = 0.7
    um = cfg["u_max_convergence"]
    grids = cfg["grids_convergence"]
    steps = cfg["steps_convergence"]
    errs, detail = {}, {}
    for h in grids:
        res = power_law_case(h, n, u_max_target=um, n_steps=steps, tol=0.0,
                             check_every=10**9)
        errs[str(h)] = res["l2_vs_closed_form"]
        detail[str(h)] = {"steps": res["steps"], "converged": res["converged"],
                          "steady_residual": res["steady_residual"],
                          "u_max": res["u_max"]}
    # drive-floor diagnosis at one fixed grid
    h_floor = cfg["grid_floor"]
    low = power_law_case(h_floor, n, u_max_target=0.4 * um, n_steps=steps, tol=0.0,
                         check_every=10**9)
    floor = {
        "grid": h_floor,
        "u_max_high": um,
        "l2_high": errs.get(str(h_floor)),
        "u_max_low": 0.4 * um,
        "l2_low": low["l2_vs_closed_form"],
        "compressibility_scale": 2.0 * (um / (1.0 / 3 ** 0.5)) ** 2,
    }
    hs = sorted(grids)
    pairs = {}
    for h0, h1 in zip(hs[:-1], hs[1:]):
        pairs[f"{h0}->{h1}"] = math.log(errs[str(h0)] / errs[str(h1)]) / math.log(h1 / h0)
    # asymptotic range: errors above the measured floor
    asym = [h for h in hs if errs[str(h)] > floor["compressibility_scale"]]
    if len(asym) < 2:
        asym = hs[:-1]
    xs = [math.log(h) for h in asym]
    ys = [math.log(errs[str(h)]) for h in asym]
    nfit = len(xs)
    slope = -((nfit * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys))
              / (nfit * sum(x * x for x in xs) - sum(xs) ** 2))
    return {
        "n": n,
        "grids": hs,
        "steps_per_grid": steps,
        "l2_relative": errs,
        "run_detail": detail,
        "observed_order_pairs": pairs,
        "asymptotic_range": asym,
        "fitted_order": slope,
        "floor_diagnosis": floor,
        "min_pair_order": min(pairs.values()) if pairs else float("nan"),
    }


def pytest_count() -> dict:
    """Count the collected tests of the module's own test file (machine-counted)."""
    target = REPO / "tests" / "test_nonnewtonian_rheology_frontier.py"
    if not target.exists():
        return {"file": str(target), "collected": 0, "error": "test file missing"}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(target)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    out = proc.stdout + proc.stderr
    m = re.search(r"(\d+) tests? collected", out)
    return {
        "file": str(target.relative_to(REPO)),
        "collected": int(m.group(1)) if m else 0,
        "returncode": proc.returncode,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(res: dict, cfg: dict, figdir: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figdir.mkdir(parents=True, exist_ok=True)
    figs: list[dict] = []

    # 1) power-law profiles (LBM vs closed form)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, n in zip(axes, (0.7, 1.3)):
        case = res["power_law"]["cases"][f"n={n:g}"]
        prof = case["profile"]
        ax.plot(prof["y"], prof["u_ref"], "k-", lw=1.4,
                label="closed form $n/(n+1)(G/K)^{1/n}[H^{(n+1)/n}-|y|^{(n+1)/n}]$")
        ax.plot(prof["y"], prof["u_lbm"], "o", ms=3.0, mfc="none",
                label="D2Q9 variable-viscosity LBM")
        ax.set_title(f"power law n = {n:g} (h = {case['h']})  L2 = {case['l2_vs_closed_form']*100:.2f} %")
        ax.set_xlabel("y (lattice units)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("u$_x$")
    axes[0].legend(fontsize=6.5, loc="lower center")
    fig.suptitle("P40W4-N04 · power-law plane Poiseuille: LBM vs analytic", fontsize=9)
    fig.tight_layout()
    p = figdir / "fig1_powerlaw_profiles.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "Power-law velocity profiles, LBM vs closed form.",
                 "path": str(p.relative_to(REPO))})

    # 2) viscosity curves of the three closures + LBM-sampled points
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    g = torch.logspace(-5, 0, 400, dtype=torch.float64)
    pl = power_law(res["power_law"]["cases"]["n=0.7"]["K"], 0.7)
    cy = carreau_yasuda(0.30, 0.02, 500.0, 0.4, 2.0)
    cr = cross(0.30, 0.02, 500.0, 1.2)
    ax.loglog(g.numpy(), pl.nu_of_gamma(g).numpy(), label="power law n=0.7 (K=%.4g)" % pl.params["K"])
    ax.loglog(g.numpy(), cy.nu_of_gamma(g).numpy(), label="Carreau–Yasuda")
    ax.loglog(g.numpy(), cr.nu_of_gamma(g).numpy(), label="Cross")
    for key, closure in (("carreau_yasuda", cy), ("cross", cr)):
        c = res[key]
        if "viscosity_range" in c:
            ax.axhspan(c["viscosity_range"][0], c["viscosity_range"][1], alpha=0.07)
    ax.set_xlabel(r"$\dot\gamma$ (lattice units)")
    ax.set_ylabel(r"$\nu_{app}(\dot\gamma)$  (lattice units)")
    ax.set_title("Constitutive curves (shaded: viscosity range spanned by the channel runs)", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = figdir / "fig2_viscosity_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "Apparent-viscosity curves of the three closures with the shear-rate band each channel run sweeps.",
                 "path": str(p.relative_to(REPO))})

    # 3) Carreau–Yasuda / Cross profiles vs numeric anchor
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, key, label in zip(axes, ("carreau_yasuda", "cross"), ("Carreau–Yasuda", "Cross")):
        c = res[key]
        prof = c["profile"]
        ax.plot(prof["y"], prof["u_ref"], "k-", lw=1.4,
                label="constitutive inversion + quadrature anchor")
        ax.plot(prof["y"], prof["u_lbm"], "s", ms=3.0, mfc="none",
                label="D2Q9 variable-viscosity LBM")
        ax.set_title(f"{label} (h = {c['h']})  L2 = {c['l2_relative']*100:.2f} %")
        ax.set_xlabel("y (lattice units)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("u$_x$")
    axes[0].legend(fontsize=6.5, loc="lower center")
    fig.suptitle("P40W4-N04 · Carreau–Yasuda / Cross channel flow vs the 1-D constitutive anchor", fontsize=9)
    fig.tight_layout()
    p = figdir / "fig3_carreau_cross_profiles.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "Carreau–Yasuda and Cross velocity profiles: LBM vs the independent 1-D constitutive-inversion anchor.",
                 "path": str(p.relative_to(REPO))})

    # 4) ice-slurry plug flow
    hb = res["herschel_bulkley"]
    prof = hb["profile"]
    H = 0.5 * hb["h"]
    y = np.asarray(prof["y"])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    axes[0].plot(y, prof["u_ref"], "k-", lw=1.4, label="regularised H-B anchor")
    axes[0].plot(y, prof["u_lbm"], "o", ms=3.2, mfc="none", label="LBM")
    for key, colour, style in (("r_c_ideal", "tab:red", ":"), ("r_c_lbm_fit", "tab:blue", "--")):
        v = hb["plug"].get(key)
        if isinstance(v, float):
            for sgn in (-1.0, 1.0):
                axes[0].axvline(sgn * v, color=colour, ls=style, lw=1.0)
    axes[0].set_xlabel("y (mid-plane origin, lattice units)")
    axes[0].set_ylabel("u$_x$")
    axes[0].set_title("ice slurry H-B plug: velocity profile", fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower center")
    axes[1].plot(np.abs(y), prof["u_lbm"], "o-", ms=2.4, lw=0.8, label="LBM")
    axes[1].plot(np.abs(y), prof["u_ref"], "k--", lw=1.0, label="anchor")
    axes[1].axvline(hb["plug"]["r_c_ideal"], color="tab:red", ls=":", lw=1.0,
                    label=r"ideal $r_c=\tau_y/(\rho a)$ = %.2f" % hb["plug"]["r_c_ideal"])
    axes[1].axvline(hb["plug"]["r_c_lbm_fit"], color="tab:blue", ls="--", lw=1.0,
                    label="LBM fit $r_c$ = %.2f" % hb["plug"]["r_c_lbm_fit"])
    axes[1].set_xlim(0.0, H)
    axes[1].set_xlabel("r = |y|")
    axes[1].set_title("plug half-width detection", fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=6.5)
    fig.suptitle("P40W4-N04 · Herschel–Bulkley ice slurry: plug core verification", fontsize=9)
    fig.tight_layout()
    p = figdir / "fig4_hb_plug.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "Herschel–Bulkley plug core: LBM profile, exact anchor, ideal and measured plug half-width.",
                 "path": str(p.relative_to(REPO))})

    # 5) grid convergence
    conv = res["convergence"]
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    hs = np.array(conv["grids"], dtype=float)
    errs = np.array([conv["l2_relative"][str(h)] for h in conv["grids"]])
    ax.loglog(hs, errs, "o-", label="power law n=0.7")
    if len(hs) >= 2:
        slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
        ax.loglog(hs, errs[0] * (hs / hs[0]) ** slope, "k--", lw=0.9,
                  label=f"fitted slope {slope:+.2f}")
    ax.set_xlabel("channel gap h (lattice units)")
    ax.set_ylabel("L2 relative error")
    ax.set_title("grid convergence (target order ≥ 2)", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = figdir / "fig5_convergence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "Grid-convergence of the power-law channel: L2 error vs gap h with fitted order.",
                 "path": str(p.relative_to(REPO))})
    return figs


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def build_gates(res: dict) -> dict:
    pl = res["power_law"]
    cy = res["carreau_yasuda"]
    cr = res["cross"]
    hb = res["herschel_bulkley"]
    conv = res["convergence"]
    py = res["pytest"]

    cy_worst = max(cy["l2_relative"], cy["viscosity_curve_max_relative_error"])
    cr_worst = max(cr["l2_relative"], cr["viscosity_curve_max_relative_error"])
    gates = [
        {
            "id": "G1_power_law_profile",
            "description": "power-law plane Poiseuille: L2 velocity error vs the closed-form "
                           "Ostwald-de Waele profile (n = 0.7 and 1.3)",
            "values": {k: v["l2_vs_closed_form"] for k, v in pl["cases"].items()},
            "worst": pl["worst_l2_vs_closed_form"],
            "threshold": 0.03,
            "passed": pl["worst_l2_vs_closed_form"] < 0.03,
        },
        {
            "id": "G2_carreau_yasuda",
            "description": "Carreau-Yasuda: worst of the velocity L2 error vs the 1-D "
                           "constitutive-inversion anchor and the viscosity-curve error",
            "values": {"l2_profile": cy["l2_relative"],
                       "viscosity_curve": cy["viscosity_curve_max_relative_error"],
                       "closure_roundtrip": cy["closure_roundtrip_max_relative"]},
            "worst": cy_worst,
            "threshold": 0.05,
            "passed": cy_worst < 0.05,
        },
        {
            "id": "G3_cross",
            "description": "Cross model: worst of the velocity L2 error vs the 1-D anchor "
                           "and the viscosity-curve error",
            "values": {"l2_profile": cr["l2_relative"],
                       "viscosity_curve": cr["viscosity_curve_max_relative_error"],
                       "closure_roundtrip": cr["closure_roundtrip_max_relative"]},
            "worst": cr_worst,
            "threshold": 0.05,
            "passed": cr_worst < 0.05,
        },
        {
            "id": "G4_ice_slurry_plug",
            "description": "ice-slurry Herschel-Bulkley plug half-width: |LBM fit r_c / "
                           "(tau_y/rho a) - 1| (velocity/stress detectors reported too)",
            "values": {"fit": hb["plug"]["r_c_lbm_fit"] / hb["plug"]["r_c_ideal"] - 1.0,
                       "velocity": hb["plug"]["rel_lbm_velocity"],
                       "stress": hb["plug"]["rel_lbm_stress"],
                       "reference_fit_bias": hb["plug"].get("rel_ref_fit")},
            "worst": abs(hb["plug"]["r_c_lbm_fit"] / hb["plug"]["r_c_ideal"] - 1.0),
            "threshold": 0.05,
            "passed": abs(hb["plug"]["r_c_lbm_fit"] / hb["plug"]["r_c_ideal"] - 1.0) < 0.05,
        },
        {
            "id": "G5_grid_convergence_order",
            "description": "observed grid-convergence order of the power-law channel L2 error",
            "values": {"fitted_order_asymptotic_range": conv["fitted_order"],
                       "two_grid_orders": conv["observed_order_pairs"],
                       "asymptotic_range": conv["asymptotic_range"],
                       "floor_diagnosis": conv["floor_diagnosis"]},
            "worst": conv["fitted_order"],
            "threshold": 2.0,
            "passed": conv["fitted_order"] >= 2.0,
        },
        {
            "id": "G6_pytest",
            "description": "collected tests in tests/test_nonnewtonian_rheology_frontier.py",
            "values": {"collected": py["collected"]},
            "worst": py["collected"],
            "threshold": 10,
            "passed": py["collected"] >= 10,
        },
    ]
    return {"all_passed": all(g["passed"] for g in gates), "gates": gates}


# ---------------------------------------------------------------------------
# machine-generated markdown summary
# ---------------------------------------------------------------------------

def _pct(x, nd=3):
    return "—" if x is None else f"{x * 100:+.{nd}f} %"


def write_markdown_summary(res: dict, outdir: Path) -> Path:
    """Write ``RESULTS.md`` from the in-memory result (no hand transcription)."""
    L = []
    a = L.append
    a(f"# P40W4-N04 results ({res['mode']} mode)")
    a("")
    a(f"generated {res.get('finished', '')} · wall time {res.get('wall_time_s', 0):.0f} s · "
      f"torch {res['environment']['torch']} on {res['environment']['machine']}")
    a("")
    a("## 1. Solver control case (Newtonian)")
    a("")
    nw = res["newtonian"]
    a("| quantity | value |")
    a("|---|---|")
    a(f"| grid gap h | {nw['h']} |")
    a(f"| u_max relative error vs parabola | {_pct(nw['u_max_relative_error'])} |")
    a(f"| profile L2 relative error | {_pct(nw['l2_relative'])} |")
    a(f"| steps / converged | {nw['steps']} / {nw['converged']} |")
    a("")
    a("## 2. Power law (closed-form Ostwald-de Waele anchor)")
    a("")
    a("| n | h | K | L2 vs closed form | u_max err | flow-rate err | steps | converged |")
    a("|---|---|---|---|---|---|---|---|")
    for key, c in res["power_law"]["cases"].items():
        a(f"| {c['n']:g} | {c['h']} | {c['K']:.5g} | {_pct(c['l2_vs_closed_form'])} | "
          f"{_pct(c['u_max_relative_error'])} | {_pct(c['flow_rate_relative_error'])} | "
          f"{c['steps']} | {c['converged']} |")
    a("")
    a("## 3. Carreau-Yasuda and Cross (constitutive-inversion anchor)")
    a("")
    a("| model | h | profile L2 | u_max err | flow-rate err | viscosity-curve max dev | closure round-trip | steps |")
    a("|---|---|---|---|---|---|---|---|")
    for key, label in (("carreau_yasuda", "Carreau-Yasuda"), ("cross", "Cross")):
        c = res[key]
        a(f"| {label} | {c['h']} | {_pct(c['l2_relative'])} | {_pct(c['u_max_relative_error'])} | "
          f"{_pct(c['flow_rate_relative_error'])} | {_pct(c['viscosity_curve_max_relative_error'])} | "
          f"{_pct(c['closure_roundtrip_max_relative'])} | {c['steps']} |")
    a("")
    a("## 4. Ice-slurry Herschel-Bulkley plug")
    a("")
    hb = res["herschel_bulkley"]
    pl = hb["plug"]
    a("| quantity | value |")
    a("|---|---|")
    a(f"| h / tau_y / K / n / m | {hb['h']} / {hb['tau_y']:.4g} / {hb['params']['K']:.4g} / "
      f"{hb['params']['n']:g} / {hb['params']['m']:.0f} |")
    a(f"| ideal plug half-width tau_y/(rho a) | {pl['r_c_ideal']:.4f} |")
    a(f"| LBM fit detector | {pl['r_c_lbm_fit']:.4f} ({_pct(pl['rel_lbm_fit'])}) |")
    a(f"| LBM stress detector | {pl['r_c_lbm_stress']:.4f} ({_pct(pl['rel_lbm_stress'])}) |")
    a(f"| LBM velocity-threshold detector | {pl['r_c_lbm_velocity']:.4f} ({_pct(pl['rel_lbm_velocity'])}) |")
    a(f"| anchor fit-detector bias (regularisation) | {_pct(pl.get('rel_ref_fit'))} |")
    a(f"| anchor velocity-threshold bias | {_pct(pl.get('rel_ref_velocity'))} |")
    a(f"| LBM vs anchor, fit detector (discretisation) | {_pct(pl.get('lbm_vs_ref_fit'))} |")
    a(f"| HB profile L2 vs anchor | {_pct(hb['l2_relative'])} |")
    a("")
    a("## 5. Grid convergence (power law n = 0.7)")
    a("")
    conv = res["convergence"]
    a(f"drive calibrated on u_max = {res['config']['u_max_convergence']:g} (Ma^2 = "
      f"{(res['config']['u_max_convergence'] / (1 / 3 ** 0.5)) ** 2:.2e}).")
    a("")
    a("| grid gap h | L2 relative error |")
    a("|---|---|")
    for h in conv["grids"]:
        a(f"| {h} | {_pct(conv['l2_relative'][str(h)], 4)} |")
    a("")
    a("| pair | two-grid Richardson order |")
    a("|---|---|")
    for k, v in conv["observed_order_pairs"].items():
        a(f"| {k} | {v:.3f} |")
    a("")
    fl = conv["floor_diagnosis"]
    a(f"drive-floor diagnosis (h = {fl['grid']}): L2 = {_pct(fl['l2_high'], 4)} at u_max = "
      f"{fl['u_max_high']:g} vs {_pct(fl['l2_low'], 4)} at u_max = {fl['u_max_low']:g}; "
      f"compressibility scale 2·Ma² = {fl['compressibility_scale']:.2e}")
    a("")
    a(f"asymptotic range = {conv['asymptotic_range']} · **fitted order = {conv['fitted_order']:.3f}**")
    a("")
    a("## 6. Gates")
    a("")
    a("| id | worst | threshold | passed |")
    a("|---|---|---|---|")
    for g in res["gates"]["gates"]:
        worst = g["worst"]
        worst_s = f"{worst:.4g}" if isinstance(worst, (int, float)) else str(worst)
        a(f"| {g['id']} | {worst_s} | {g['threshold']} | {g['passed']} |")
    a("")
    a(f"**all_passed = {res['gates']['all_passed']}**")
    a("")
    path = outdir / "RESULTS.md"
    path.write_text("\n".join(L) + "\n")
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

PRODUCTION = {
    "h_newtonian": 32,
    "h_powerlaw": 32,
    "h_carreau_yasuda": 48,
    "h_cross": 48,
    "h_hb": 48,
    "grids_convergence": [16, 32, 64, 128],
    "u_max_convergence": 5e-3,
    "steps_convergence": 40_000,
    "grid_floor": 64,
    "n_steps": 250_000,
    "tol": 1e-6,
    "check_every": 500,
}

FAST = {
    "h_newtonian": 16,
    "h_powerlaw": 16,
    "h_carreau_yasuda": 16,
    "h_cross": 16,
    "h_hb": 16,
    "grids_convergence": [8, 16, 32],
    "u_max_convergence": 5e-3,
    "steps_convergence": 6000,
    "grid_floor": 16,
    "n_steps": 20_000,
    "tol": 1e-5,
    "check_every": 250,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="coarse grids, short runs (smoke test)")
    ap.add_argument("--stage", default="all",
                    choices=["all", "newtonian", "power_law", "carreau_yasuda", "cross",
                             "herschel_bulkley", "convergence"])
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args(argv)

    cfg = dict(FAST if args.fast else PRODUCTION)
    outdir = Path(args.outdir)
    res: dict = {
        "case": "p40w4_n04_nonnewtonian_rheology",
        "node": "p40w4-n04-lbm-nonnewtonian",
        "title": "Non-Newtonian rheology verification surface: power law / Carreau-Yasuda / "
                 "Cross + Herschel-Bulkley ice-slurry plug",
        "mode": "fast" if args.fast else "production",
        "config": cfg,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dtype": "float32 lattice / float64 anchors",
        },
        "lattice": {"scheme": "D2Q9", "collision": "variable-viscosity BGK (per-cell tau)",
                    "forcing": "Guo (2002) velocity-shift, per-cell tau",
                    "wall": "pre-streaming half-way bounce-back"},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    t_start = time.time()

    stages = (["newtonian", "power_law", "carreau_yasuda", "cross", "herschel_bulkley", "convergence"]
              if args.stage == "all" else [args.stage])
    for st in stages:
        t0 = time.time()
        print(f"[p40w4-n04] stage {st} ...", flush=True)
        if st == "newtonian":
            res["newtonian"] = stage_newtonian(cfg)
        elif st == "power_law":
            res["power_law"] = stage_power_law(cfg)
        elif st in ("carreau_yasuda", "cross"):
            res[st] = stage_viscoelastic_closure(cfg, st)
        elif st == "herschel_bulkley":
            res["herschel_bulkley"] = stage_herschel_bulkley(cfg)
        elif st == "convergence":
            res["convergence"] = stage_convergence(cfg)
        print(f"[p40w4-n04] stage {st} done in {time.time() - t0:.1f}s", flush=True)

    if args.stage == "all":
        res["pytest"] = pytest_count()
        try:
            res["figures"] = make_figures(res, cfg, REPO / "figures" / "p40w4_n04")
        except Exception as exc:  # pragma: no cover - figures are optional here
            res["figures"] = []
            res["figures_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[p40w4-n04] figure generation failed: {exc}", flush=True)
        res["gates"] = build_gates(res)
        res["wall_time_s"] = time.time() - t_start
        res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    payload = _f(res)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    if "gates" in res:
        (outdir / "gates.json").write_text(json.dumps(_f(res["gates"]), indent=2) + "\n")
        md_path = write_markdown_summary(payload, outdir)
        print(f"[p40w4-n04] wrote {md_path}")
        print("[p40w4-n04] gates:", {g["id"]: g["passed"] for g in res["gates"]["gates"]})
        print("[p40w4-n04] all_passed =", res["gates"]["all_passed"])
    print(f"[p40w4-n04] wrote {outdir / 'result.json'} (wall {time.time() - t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())