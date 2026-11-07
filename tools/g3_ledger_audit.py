#!/usr/bin/env python
"""G3 ledger audit — Shan–Chen coexistence of the cavitation pseudopotential.

Ledger entry (``benchmarks/gaps.md``, G3)::

    psi_cavitation 默认 G=−5.5 无共存态（cavitation.py）
    EOS p=ρ/3+Gψ²/6 在 G=−5.5 时液枝永不恢复稳定（p(2.65)=−4.435），
    模块注释声称的 ρ_l=2.65 非麦克斯韦共存态；
    有效共存：G=−1.0 → ρ_v=0.0296/ρ_l=4.330/密度比 146

This script executes the ledger's own verification surface: every quantitative
claim is recomputed from the equation of state of the force the library
actually applies, cross-checked against an independent direct SCMP relaxation,
and written to ``artifacts/g3_ledger/`` (JSON + figures).  The markdown report
(``docs/g_ledger_g3_cavitation_coexistence.md``) is generated from the same
numbers in the same run — nothing in it is typed by hand.

Usage::

    PYTHONPATH=src python tools/g3_ledger_audit.py             # full audit
    PYTHONPATH=src python tools/g3_ledger_audit.py --quick     # smoke run
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorlbm import d2q9, solver  # noqa: E402
from tensorlbm.cavitation import (  # noqa: E402
    COEXISTENCE_G,
    COEXISTENCE_RHO_L,
    COEXISTENCE_RHO_V,
    CavitationConfig,
    psi_cavitation,
)
from tensorlbm.multiphase import collide_sc_single_component  # noqa: E402
from tensorlbm.sc_eos_coexistence import (  # noqa: E402
    CS2,
    chemical_potential,
    maxwell_coexistence,
    mechanical_pressure,
    relax_scmp,
    spinodal_bounds,
)

# --- Ledger numbers, transcribed verbatim from benchmarks/gaps.md (entry G3) ---
LEDGER_G_CLASSIC_BAD = -5.5
LEDGER_P_AT_RHO_2_65 = -4.435
LEDGER_RHO_L_CLAIMED = 2.65
LEDGER_G_CLASSIC_VALID = -1.0
LEDGER_RHO_V = 0.0296
LEDGER_RHO_L = 4.330
LEDGER_RATIO = 146.0

# --- Acceptance criteria, fixed before the run (report section 6) ---
CRITERIA = {
    "claim1_pressure_rel_tol": 0.01,
    "claim3_ledger_rule_rel_tol": 0.01,
    "three_percent_tol": 0.03,
    "threshold_agree_abs_G": 0.10,
}


def psi(rho: torch.Tensor) -> torch.Tensor:
    """The cavitation module's pseudopotential, with its module defaults."""
    return psi_cavitation(rho)


def ledger_rho_space_rule(G: float, n_psat: int = 400) -> dict[str, Any]:
    """Reproduce the ledger's coexistence numbers with its own area rule.

    The ledger's ``G_c=-1.0 → ρ_v=0.0296 / ρ_l=4.330 / 146`` triple is *not* the
    Maxwell construction: it is what one gets by zeroing ``∫(p − p_sat) dρ``
    instead of ``∫(p − p_sat) dv`` (``v = 1/ρ``).  Kept here only to identify the
    rule behind the ledger digits; the library uses the correct rule
    (:func:`tensorlbm.sc_eos_coexistence.maxwell_coexistence`).
    """
    bounds = spinodal_bounds(G, psi, 1e-5, 2e2, 12000)
    if bounds is None:
        return {"found": False, "reason": "no spinodal"}
    r_sp_v, r_sp_l = bounds
    rs = torch.exp(torch.linspace(math.log(1e-5), math.log(2e2), 12000))
    ps = mechanical_pressure(rs, G, psi)
    i_v = int(torch.nonzero(rs <= r_sp_v).flatten()[-1])
    i_l = int(torch.nonzero(rs >= r_sp_l).flatten()[0])
    lo = max(float(ps[: i_v + 1].min()), float(ps[i_l:].min()))
    hi = min(float(ps[: i_v + 1].max()), float(ps[i_l:].max()))

    def p_of(r: float) -> float:
        return float(mechanical_pressure(torch.tensor(r), G, psi))

    def bisect(f, a: float, b: float, it: int = 200) -> float:
        fa = f(a)
        for _ in range(it):
            m = 0.5 * (a + b)
            fm = f(m)
            if (fa > 0) == (fm > 0):
                a, fa = m, fm
            else:
                b = m
            if b - a <= 1e-14 * max(1.0, abs(m)):
                break
        return 0.5 * (a + b)

    def area(p_sat: float) -> float:
        r_v = bisect(lambda r: p_of(r) - p_sat, 1e-5, r_sp_v)
        r_l = bisect(lambda r: p_of(r) - p_sat, r_sp_l, 2e2)
        u = torch.linspace(math.log(r_v), math.log(r_l), 4000, dtype=torch.float64)
        p = mechanical_pressure(torch.exp(u), G, psi)
        return float(torch.trapezoid((p - p_sat) * torch.exp(u), u))

    trials = torch.linspace(lo, hi, n_psat, dtype=torch.float64)
    vals = [area(float(p)) for p in trials]
    for k in range(len(vals) - 1):
        if (vals[k] > 0) != (vals[k + 1] > 0):
            p_sat = bisect(area, float(trials[k]), float(trials[k + 1]))
            r_v = bisect(lambda r: p_of(r) - p_sat, 1e-5, r_sp_v)
            r_l = bisect(lambda r: p_of(r) - p_sat, r_sp_l, 2e2)
            return {
                "found": True,
                "G": G,
                "G_classic": -G,
                "p_sat": p_sat,
                "rho_vapour": r_v,
                "rho_liquid": r_l,
                "density_ratio": r_l / r_v,
                "rule": "int (p - p_sat) d(rho) = 0  [dimensionally inconsistent]",
            }
    return {"found": False, "reason": "no sign change of the rho-space area"}


def run_field(
    G: float,
    n: int,
    steps: int,
    rho0: float = 1.5,
    tau: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """SCMP relaxation returning the last finite density field (contour figure)."""
    gen = torch.Generator().manual_seed(seed)
    rho = rho0 * (1.0 + 0.02 * (torch.rand((n, n), generator=gen) - 0.5))
    zeros = torch.zeros((n, n))
    f = d2q9.equilibrium(rho, zeros, zeros)
    diverged_at: int | None = None
    last_finite = rho.clone()
    for step in range(1, steps + 1):
        f = collide_sc_single_component(f, G=G, tau=tau, psi_fn=psi, forcing="edm")
        f = solver.stream(f)
        if step % 25 == 0 or step == steps:
            r, _, _ = d2q9.macroscopic(f)
            if bool(torch.isnan(r).any() or torch.isinf(r).any()):
                diverged_at = step
                break
            last_finite = r.clone()
    return {
        "G": G,
        "G_classic": -G,
        "field": last_finite,
        "diverged_at_step": diverged_at,
    }


def rel_err(value: float | None, ref: float | None) -> float | None:
    """Signed relative error ``(value − ref)/ref``; ``None`` when undefined."""
    if value is None or ref in (None, 0.0):
        return None
    return (value - ref) / ref


def _figures(audit: dict[str, Any], outdir: Path, nx: int, steps: int) -> dict[str, Any]:
    G_bad = -LEDGER_G_CLASSIC_BAD
    G_ok = -LEDGER_G_CLASSIC_VALID
    mw_ok = maxwell_coexistence(G_ok, psi)
    rule = audit["claims"]["C3"]["ledger_rho_space_rule"]
    eos_rows = audit["eos_scan"]["rows"]
    sim_rows = audit["simulation"]["rows"]

    fig1 = outdir / "fig1_eos_spinodal.png"
    rs = torch.exp(torch.linspace(math.log(1e-2), math.log(2e1), 2000))
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for g, colour, label in (
        (G_ok, "tab:blue", f"G={G_ok:+.1f} (classic {-G_ok:+.1f}): coexistence exists"),
        (G_bad, "tab:red", f"G={G_bad:+.1f} (classic {-G_bad:+.1f}): ledger default"),
    ):
        p = mechanical_pressure(rs, g, psi)
        ax.plot(rs.numpy(), p.numpy(), color=colour, label=label)
        bounds = spinodal_bounds(g, psi, 1e-5, 2e2, 12000)
        if bounds:
            ax.axvspan(*bounds, color=colour, alpha=0.10)
    if mw_ok.found:
        ax.axhline(mw_ok.p_sat, color="0.35", ls=":", lw=1.0)
        ax.plot(
            [mw_ok.rho_vapour, mw_ok.rho_liquid],
            [mw_ok.p_sat, mw_ok.p_sat],
            "o",
            ms=4,
            color="tab:blue",
        )
        ax.annotate(
            f"$p_{{sat}}$={mw_ok.p_sat:.4f}",
            xy=(1.0, mw_ok.p_sat),
            xytext=(1.2, mw_ok.p_sat + 1.0),
            fontsize=8,
            color="0.35",
        )
    ax.axvline(LEDGER_RHO_L_CLAIMED, color="k", ls="--", lw=0.8)
    ax.annotate(
        f"ledger $\\rho_l$={LEDGER_RHO_L_CLAIMED}",
        xy=(LEDGER_RHO_L_CLAIMED, -6.0),
        xytext=(3.1, -6.4),
        fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_xlabel("density ρ (lattice units)")
    ax.set_ylabel("mechanical pressure p(ρ)")
    ax.set_title("Shan–Chen EOS of psi_cavitation at the ledger couplings (spinodal shaded)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig1, dpi=140)
    plt.close(fig)

    fig2 = outdir / "fig2_coexistence_vs_coupling.png"
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    found = [row for row in eos_rows if row["found"]]
    gs = [row["G"] for row in found]
    sim_ok = [row for row in sim_rows if not row["diverged"]]
    axes[0].plot(gs, [row["rho_liquid"] for row in found], "-", color="tab:green",
                 label="Maxwell $\\rho_l$ (correct rule)")
    axes[0].plot(gs, [row["rho_vapour"] for row in found], "-", color="tab:orange",
                 label="Maxwell $\\rho_v$ (correct rule)")
    axes[0].plot([row["G"] for row in sim_ok], [row["rho_max"] for row in sim_ok], "s",
                 ms=5, mfc="none", color="tab:green", label="SCMP measured max ρ")
    axes[0].plot([row["G"] for row in sim_ok], [row["rho_min"] for row in sim_ok], "^",
                 ms=5, mfc="none", color="tab:orange", label="SCMP measured min ρ")
    if rule.get("found"):
        axes[0].plot([G_ok], [rule["rho_liquid"]], "x", color="k", ms=8,
                     label=f"ledger ρ_l={LEDGER_RHO_L}")
        axes[0].plot([G_ok], [rule["rho_vapour"]], "x", color="0.45", ms=8,
                     label=f"ledger ρ_v={LEDGER_RHO_V}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("G (library convention, >0 attractive)")
    axes[0].set_ylabel("density ρ")
    axes[0].set_title("Coexistence densities: EOS vs direct SCMP")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=6.8)

    axes[1].plot(gs, [row["density_ratio"] for row in found], "-", color="tab:blue",
                 label="Maxwell ratio (correct rule)")
    axes[1].plot([row["G"] for row in sim_ok], [row["density_ratio"] for row in sim_ok], "o",
                 ms=4, color="tab:red", label="SCMP measured ratio")
    axes[1].axhline(LEDGER_RATIO, color="k", ls="--", lw=0.9,
                    label=f"ledger ratio {LEDGER_RATIO:g}")
    if rule.get("found"):
        axes[1].plot([G_ok], [rule["density_ratio"]], "x", color="k", ms=9,
                     label="ratio from ledger's rule")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("G (library convention, >0 attractive)")
    axes[1].set_ylabel("ρ_liquid / ρ_vapour")
    axes[1].set_title("Density ratio: correct Maxwell vs ledger rule vs SCMP")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=6.8)
    fig.tight_layout()
    fig.savefig(fig2, dpi=140)
    plt.close(fig)

    fig3 = outdir / "fig3_coupling_gate.png"
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.plot([row["G"] for row in sim_ok], [row["density_ratio"] for row in sim_ok], "o-",
            color="tab:blue", label="SCMP measured ρ_max/ρ_min")
    for row in sim_rows:
        if row["diverged"]:
            ax.axvspan(row["G"] - 0.06, row["G"] + 0.06, color="tab:red", alpha=0.25)
            ax.annotate("NaN", xy=(row["G"], 1.06), ha="center", fontsize=8, color="tab:red")
    threshold = audit["eos_scan"]["first_coexisting_G"]
    if threshold is not None:
        ax.axvline(threshold, color="tab:green", ls="--",
                   label=f"Maxwell coexistence onset G≈{threshold:.2f}")
    ax.axhline(2.0, color="0.5", lw=0.8, ls=":", label="separation criterion ratio>2")
    ax.set_yscale("log")
    ax.set_xlabel("G (library convention, >0 attractive;  classic G = -G)")
    ax.set_ylabel("density ratio")
    ax.set_title("Coupling gate: homogeneous / separated / divergent (red = NaN)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(fig3, dpi=140)
    plt.close(fig)

    fig4 = outdir / "fig4_density_fields.png"
    field_bad = run_field(G_bad, nx, steps)
    field_ok = run_field(G_ok, nx, steps)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    im0 = axes[0].imshow(field_ok["field"].numpy(), origin="lower", cmap="viridis")
    axes[0].set_title(
        f"G={G_ok:+.1f} (classic {-G_ok:+.1f}): phase separated\n"
        f"ρ ∈ [{float(field_ok['field'].min()):.2f}, {float(field_ok['field'].max()):.2f}]"
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, label="ρ")
    im1 = axes[1].imshow(field_bad["field"].numpy(), origin="lower", cmap="magma")
    axes[1].set_title(
        f"G={G_bad:+.1f} (classic {-G_bad:+.1f}): ledger default\n"
        f"NaN from step {field_bad['diverged_at_step']} (last finite field shown)"
    )
    fig.colorbar(im1, ax=axes[1], fraction=0.046, label="ρ")
    for axx in axes:
        axx.set_xlabel("x")
        axx.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(fig4, dpi=140)
    plt.close(fig)

    return {
        "figures": [
            {"name": fig1.name,
             "caption": "EOS p(ρ) of psi_cavitation at the ledger couplings; spinodal bands "
                        "shaded; Maxwell p_sat and coexistence pair at G=+1.0 (classic −1.0)."},
            {"name": fig2.name,
             "caption": "Coexistence densities (left) and density ratio (right) vs coupling: "
                        "correct Maxwell rule, direct SCMP measurement, and the ledger's pair."},
            {"name": fig3.name,
             "caption": "Coupling gate: measured density ratio vs G, NaN bands (red) and the "
                        "Maxwell coexistence onset."},
            {"name": fig4.name,
             "caption": "Density fields after relaxation at the audited coupling (left) and at "
                        "the ledger default coupling (right, NaN onset annotated)."},
        ],
        "fields": {
            "G_valid": {
                "G": field_ok["G"],
                "G_classic": field_ok["G_classic"],
                "diverged_at_step": field_ok["diverged_at_step"],
                "rho_min": float(field_ok["field"].min()),
                "rho_max": float(field_ok["field"].max()),
            },
            "G_ledger_default": {
                "G": field_bad["G"],
                "G_classic": field_bad["G_classic"],
                "diverged_at_step": field_bad["diverged_at_step"],
                "rho_min": float(field_bad["field"].min()),
                "rho_max": float(field_bad["field"].max()),
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="artifacts/g3_ledger")
    ap.add_argument("--doc", default="docs/g_ledger_g3_cavitation_coexistence.md")
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-doc", action="store_true", help="skip report generation")
    ap.add_argument("--quick", action="store_true", help="small grid/steps for smoke runs")
    args = ap.parse_args()
    if args.quick:
        args.nx, args.steps = 32, 400
    torch.set_num_threads(max(1, args.threads))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    G_bad = -LEDGER_G_CLASSIC_BAD
    G_ok = -LEDGER_G_CLASSIC_VALID

    audit: dict[str, Any] = {
        "ledger": {
            "file": "benchmarks/gaps.md",
            "entry": "G3",
            "title": "psi_cavitation 默认 G=−5.5 无共存态 (cavitation.py)",
            "claims": {
                "C1": f"EOS p=ρ/3+Gψ²/6 at classic G={LEDGER_G_CLASSIC_BAD} gives "
                      f"p(ρ=2.65)={LEDGER_P_AT_RHO_2_65}",
                "C2": f"no liquid/vapour coexistence at classic G={LEDGER_G_CLASSIC_BAD}; "
                      f"ρ_l={LEDGER_RHO_L_CLAIMED} is not a Maxwell coexistence density",
                "C3": f"valid coexistence at classic G={LEDGER_G_CLASSIC_VALID}: "
                      f"ρ_v={LEDGER_RHO_V}, ρ_l={LEDGER_RHO_L}, ratio={LEDGER_RATIO:g}",
            },
            "sign_convention": "library: F=(G cs²/2)∇(ψ²), G>0 attractive; classic G_c=-G",
        },
        "criteria": CRITERIA,
        "settings": {
            "nx": args.nx,
            "steps": args.steps,
            "rho0": 1.5,
            "tau": 1.0,
            "forcing": "edm",
            "torch_version": torch.__version__,
            "threads": args.threads,
            "psi": "psi_cavitation(rho, rho_l=2.65, rho_v=0.038, psi0=4.0)",
        },
        "claims": {},
    }

    # --- C1: ledger pressure anchor -------------------------------------------------
    p_anchor = float(mechanical_pressure(torch.tensor(LEDGER_RHO_L_CLAIMED), G_bad, psi))
    c1_err = rel_err(p_anchor, LEDGER_P_AT_RHO_2_65)
    audit["claims"]["C1"] = {
        "G_library": G_bad,
        "G_classic": -G_bad,
        "rho": LEDGER_RHO_L_CLAIMED,
        "psi": float(psi(torch.tensor(LEDGER_RHO_L_CLAIMED))),
        "p_computed": p_anchor,
        "p_ledger": LEDGER_P_AT_RHO_2_65,
        "rel_error": c1_err,
        "verdict": "PASS" if abs(c1_err) <= CRITERIA["claim1_pressure_rel_tol"] else "FAIL",
    }

    # --- C2: no coexistence at the old default -------------------------------------
    mw_bad = maxwell_coexistence(G_bad, psi)
    audit["claims"]["C2"] = {
        "G_library": G_bad,
        "G_classic": -G_bad,
        "maxwell": mw_bad.as_dict(),
        "spinodal": spinodal_bounds(G_bad, psi, 1e-5, 2e2, 12000),
        "coexistence_found": mw_bad.found,
        "verdict": "PASS" if not mw_bad.found else "FAIL",
    }

    # --- C3: the ledger's "valid" coupling ----------------------------------------
    mw_ok = maxwell_coexistence(G_ok, psi)
    ledger_rule = ledger_rho_space_rule(G_ok)
    mu = chemical_potential(torch.tensor([mw_ok.rho_vapour, mw_ok.rho_liquid]), G_ok, psi,
                            n=40000)
    audit["claims"]["C3"] = {
        "G_library": G_ok,
        "G_classic": -G_ok,
        "maxwell_correct_rule": mw_ok.as_dict(),
        "ledger_rho_space_rule": ledger_rule,
        "mu_equality_check": {
            "mu_vapour": float(mu[0]),
            "mu_liquid": float(mu[1]),
            "abs_diff_over_cs2": float(abs(mu[1] - mu[0]) / CS2),
        },
        "ledger_vs_maxwell": {
            "rho_vapour_ledger": LEDGER_RHO_V,
            "rho_vapour_maxwell": mw_ok.rho_vapour,
            "rho_vapour_rel_error": rel_err(mw_ok.rho_vapour, LEDGER_RHO_V),
            "rho_liquid_ledger": LEDGER_RHO_L,
            "rho_liquid_maxwell": mw_ok.rho_liquid,
            "rho_liquid_rel_error": rel_err(mw_ok.rho_liquid, LEDGER_RHO_L),
            "ratio_ledger": LEDGER_RATIO,
            "ratio_maxwell": mw_ok.density_ratio,
            "ratio_rel_error": rel_err(mw_ok.density_ratio, LEDGER_RATIO),
        },
        "ledger_vs_ledger_rule": {
            "rho_vapour_rel_error": rel_err(ledger_rule.get("rho_vapour"), LEDGER_RHO_V),
            "rho_liquid_rel_error": rel_err(ledger_rule.get("rho_liquid"), LEDGER_RHO_L),
            "ratio_rel_error": rel_err(ledger_rule.get("density_ratio"), LEDGER_RATIO),
        },
    }

    # --- EOS scan: coexistence onset ------------------------------------------------
    g_scan = [round(0.05 * k, 2) for k in range(1, 41)]
    eos_rows = []
    for g in g_scan:
        res = maxwell_coexistence(g, psi, n=8000)
        eos_rows.append({
            "G": g,
            "G_classic": -g,
            "found": res.found,
            "rho_vapour": res.rho_vapour,
            "rho_liquid": res.rho_liquid,
            "density_ratio": res.density_ratio,
            "p_sat": res.p_sat,
        })
    eos_threshold = next((row["G"] for row in eos_rows if row["found"]), None)
    audit["eos_scan"] = {"rows": eos_rows, "first_coexisting_G": eos_threshold}

    # --- independent SCMP relaxation ----------------------------------------------
    sim_g = [-5.5, 0.5, 0.6, 0.7, 0.9, 1.0, 1.2, 1.5]
    sim = [relax_scmp(g, psi, nx=args.nx, ny=args.nx, rho0=1.5, steps=args.steps)
           for g in sim_g]
    separated = [row["G"] for row in sim if not row["diverged"] and row["density_ratio"] > 2.0]
    sim_threshold = min(separated) if separated else None
    agreement = None if (sim_threshold is None or eos_threshold is None) else abs(
        sim_threshold - eos_threshold
    )
    audit["simulation"] = {
        "rows": sim,
        "separated_at": separated,
        "first_separating_G": sim_threshold,
        "threshold_agreement": agreement,
        "verdict": "PASS" if agreement is not None and agreement <= CRITERIA[
            "threshold_agree_abs_G"] else "FAIL",
    }

    # --- module defaults -----------------------------------------------------------
    mw_default = maxwell_coexistence(COEXISTENCE_G, psi)
    audit["module_defaults"] = {
        "cavitation_COEXISTENCE_G": COEXISTENCE_G,
        "cavitation_COEXISTENCE_RHO_L": COEXISTENCE_RHO_L,
        "cavitation_COEXISTENCE_RHO_V": COEXISTENCE_RHO_V,
        "CavitationConfig_G": CavitationConfig().G,
        "maxwell_at_default": mw_default.as_dict(),
        "rho_l_rel_error_vs_constant": rel_err(mw_default.rho_liquid, COEXISTENCE_RHO_L),
        "rho_v_rel_error_vs_constant": rel_err(mw_default.rho_vapour, COEXISTENCE_RHO_V),
    }

    # --- figures -------------------------------------------------------------------
    audit |= _figures(audit, outdir, args.nx, args.steps)

    # --- 3 % error table -----------------------------------------------------------
    c3 = audit["claims"]["C3"]
    lv, lr = c3["ledger_vs_maxwell"], c3["ledger_vs_ledger_rule"]
    mw_0p7 = maxwell_coexistence(0.7, psi)
    sim_0p7 = next((row for row in sim if row["G"] == 0.7), None)
    sim_1p0 = next((row for row in sim if row["G"] == 1.0), None)
    checks = [
        {"quantity": "p(ρ=2.65) at classic G=-5.5",
         "reference": f"ledger {LEDGER_P_AT_RHO_2_65}",
         "value": audit["claims"]["C1"]["p_computed"],
         "rel_error": audit["claims"]["C1"]["rel_error"]},
        {"quantity": "ρ_l at classic G=-1.0 (correct Maxwell)",
         "reference": f"ledger ρ_l={LEDGER_RHO_L}",
         "value": lv["rho_liquid_maxwell"],
         "rel_error": lv["rho_liquid_rel_error"]},
        {"quantity": "ρ_v at classic G=-1.0 (correct Maxwell)",
         "reference": f"ledger ρ_v={LEDGER_RHO_V}",
         "value": lv["rho_vapour_maxwell"],
         "rel_error": lv["rho_vapour_rel_error"]},
        {"quantity": "density ratio at classic G=-1.0 (correct Maxwell)",
         "reference": f"ledger ratio {LEDGER_RATIO:g}",
         "value": lv["ratio_maxwell"],
         "rel_error": lv["ratio_rel_error"]},
        {"quantity": "ledger triple reproduced by the ledger's own area rule",
         "reference": f"{LEDGER_RHO_V} / {LEDGER_RHO_L} / {LEDGER_RATIO:g}",
         "value": f"{ledger_rule.get('rho_vapour')} / {ledger_rule.get('rho_liquid')} / "
                  f"{ledger_rule.get('density_ratio')}",
         "rel_error": max(abs(lr["rho_vapour_rel_error"]), abs(lr["rho_liquid_rel_error"]),
                          abs(lr["ratio_rel_error"]))},
        {"quantity": "μ(ρ_v) vs μ(ρ_l) at the Maxwell pair (relative to cs²)",
         "reference": "0 (coexistence condition)",
         "value": c3["mu_equality_check"]["abs_diff_over_cs2"],
         "rel_error": c3["mu_equality_check"]["abs_diff_over_cs2"]},
    ]
    if sim_0p7 is not None and mw_0p7.found:
        checks += [
            {"quantity": "SCMP ρ_l at G=+0.7 vs correct Maxwell",
             "reference": f"Maxwell {mw_0p7.rho_liquid:.4f}",
             "value": sim_0p7["rho_max"],
             "rel_error": rel_err(sim_0p7["rho_max"], mw_0p7.rho_liquid)},
            {"quantity": "SCMP ρ_v at G=+0.7 vs correct Maxwell",
             "reference": f"Maxwell {mw_0p7.rho_vapour:.4f}",
             "value": sim_0p7["rho_min"],
             "rel_error": rel_err(sim_0p7["rho_min"], mw_0p7.rho_vapour)},
        ]
    if sim_1p0 is not None:
        checks += [
            {"quantity": "SCMP ρ_l at G=+1.0 (module default) vs correct Maxwell",
             "reference": f"Maxwell {mw_ok.rho_liquid:.4f}",
             "value": sim_1p0["rho_max"],
             "rel_error": rel_err(sim_1p0["rho_max"], mw_ok.rho_liquid)},
            {"quantity": "SCMP ρ_v at G=+1.0 (module default) vs correct Maxwell",
             "reference": f"Maxwell {mw_ok.rho_vapour:.4f}",
             "value": sim_1p0["rho_min"],
             "rel_error": rel_err(sim_1p0["rho_min"], mw_ok.rho_vapour)},
        ]
    for row in checks:
        err = row["rel_error"]
        row["within_3pct"] = (None if err is None
                              else abs(err) < CRITERIA["three_percent_tol"])
    audit["error_table"] = checks
    audit["runtime_seconds"] = round(time.time() - t_start, 2)

    (outdir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    if not args.no_doc:
        from g3_ledger_report import write_report  # noqa: PLC0415

        write_report(Path(args.doc), audit)
    print(json.dumps({"outdir": str(outdir), "doc": None if args.no_doc else args.doc,
                      "runtime_s": audit["runtime_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())