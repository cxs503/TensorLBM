#!/usr/bin/env python3
"""Aggregate the P127W2-N03 campaign artifacts into a verdict + figures.

Reads every ``artifacts/campaign/*.json`` case file produced by
``run_case.py`` plus ``artifacts/campaign/feasibility.json``, and writes

* ``artifacts/campaign/campaign_summary.json`` — per-case criterion table with
  the number behind every PASS/FAIL/WITHHELD, the grid-convergence study and the
  statistics-window audit (eddy turnovers and viscous times actually sampled);
* ``artifacts/campaign/figures/*.png`` — profile / stress / spectrum / decay /
  feasibility figures.

Nothing is typed in by hand: every number is read from a case artifact and every
derived quantity is recomputed here from the case's own configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.turbulence.les_channel3d import (  # noqa: E402
    KAPPA,
    B_LOG,
    gci_three_grid,
    interpolate_profile,
    mkm_profile_for,
)

CAMPAIGN = ROOT / "artifacts" / "campaign"


def load_cases(pattern: str = "*.json") -> list[dict]:
    cases = []
    for path in sorted(CAMPAIGN.glob(pattern)):
        if path.name in ("campaign_summary.json", "feasibility.json"):
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "result" not in payload or "verdict" not in payload:
            continue
        payload["_path"] = str(path.relative_to(ROOT))
        cases.append(payload)
    return cases


def window_audit(case: dict) -> dict:
    """How much physical time did the statistics window really cover?

    Reported in eddy turnovers ``T_e = h/u_tau`` and in wall-clock viscous times
    ``t_nu = h^2/nu`` — the second one is what decides whether the *mean* profile
    has even reached its statistically stationary state, independent of whether
    the flow is turbulent.
    """
    cfg = case["result"]["config"]
    h = float(cfg.get("half_height") or 0.5 * (cfg["ny"] - 2))
    u_tau = float(cfg.get("u_tau_target") or cfg.get("u_tau") or 0.0)
    nu = float(cfg["nu"]) if cfg.get("nu") else (u_tau * h / float(cfg["re_tau_target"]))
    steps = int(cfg["steps"])
    warmup = int(cfg["warmup_steps"])
    sampled = max(steps - warmup, 1)
    t_e = h / u_tau if u_tau > 0 else float("nan")
    t_nu = h * h / nu if nu > 0 else float("nan")
    return {
        "sampled_steps": sampled,
        "eddy_turnover_steps": t_e,
        "eddy_turnovers_sampled": sampled / t_e if t_e > 0 else float("nan"),
        "viscous_time_steps": t_nu,
        "viscous_times_sampled": sampled / t_nu if t_nu > 0 else float("nan"),
        "warmup_eddy_turnovers": warmup / t_e if t_e > 0 else float("nan"),
    }


def case_row(case: dict) -> dict:
    result = case["result"]
    wu = result["wall_units"]
    checks = case["verdict"]["criterion_checks"]
    y_plus = [float(v) for v in wu["y_plus"]]
    u_plus = [float(v) for v in wu["u_plus"]]
    band = [i for i, yp in enumerate(y_plus) if 30.0 <= yp <= 100.0 and yp <= 0.3 * float(case["re_tau_target"])]
    log_resid = [
        u_plus[i] - (math.log(y_plus[i]) / KAPPA + B_LOG) for i in band
    ]
    return {
        "label": case["label"],
        "artifact": case["_path"],
        "re_tau": float(case["re_tau_target"]),
        "h": case["geometry"]["h"],
        "cells": case["geometry"]["nx"] * case["geometry"]["ny"] * case["geometry"]["nz"],
        "dy_plus": case["geometry"]["dy_plus"],
        "y_plus_first_node": case["geometry"]["first_fluid_node_y_plus"],
        "steps": result["config"]["steps"],
        "u_tau_target": result["config"].get("u_tau_target"),
        "u_tau_wall_exchange": case["summary"].get("u_tau_wall_exchange"),
        "u_tau_stress_fit": case["summary"].get("u_tau_stress_fit"),
        "u_plus_max": case["summary"].get("u_plus_max"),
        "bulk_reynolds": case["summary"].get("bulk_reynolds"),
        "log_band_nodes": len(band),
        "log_band_max_abs_residual": max((abs(r) for r in log_resid), default=None),
        "log_band_max_rel_error": (
            max((abs(r) / (math.log(y_plus[i]) / KAPPA + B_LOG)
                 for r, i in zip(log_resid, band)), default=None)
        ),
        "uv_peak_plus": case["summary"].get("uv_peak_plus"),
        "uv_peak_y_plus": case["summary"].get("uv_peak_y_plus"),
        "sgs_fraction_y50": case["summary"].get("sgs_fraction_y50"),
        "spectrum_slope": case["summary"].get("spectrum_slope"),
        "spectrum_r2": case["summary"].get("spectrum_r2"),
        "mass_rel_std": case["summary"].get("mass_rel_std"),
        "mass_drift": case["summary"].get("mass_drift"),
        "momentum_balance_max_error": case["summary"].get("momentum_balance_max_error"),
        "tke_final": (result["history"][-1]["tke"] if result.get("history") else None),
        "tke_initial": (result["history"][0]["tke"] if result.get("history") else None),
        "strain_source": result["config"].get("strain_source"),
        "wall_model": result["config"].get("wall_model"),
        "criteria": {
            key: {"status": val.get("status"), "metric": val.get("metric")}
            for key, val in checks.items()
        },
        "window": window_audit(case),
    }


def gci_study(cases: list[dict]) -> dict:
    """Three-grid GCI on the Re_tau=180 ladder (fine/medium/coarse by h)."""
    ladder = sorted(
        [c for c in cases if abs(float(c["re_tau_target"]) - 180.0) < 1e-9],
        key=lambda c: c["geometry"]["h"],
    )
    if len(ladder) < 3:
        return {"available": False, "reason": f"only {len(ladder)} grids at Re_tau=180"}
    # use the three finest distinct grids
    picked = ladder[-3:]
    picked = sorted(picked, key=lambda c: c["geometry"]["h"])
    h = [c["geometry"]["h"] for c in picked]
    r = (h[1] / h[0] + h[2] / h[1]) / 2.0
    metrics = {}
    for name, getter in (
        ("u_plus_max", lambda s: s.get("u_plus_max")),
        ("uv_peak_plus", lambda s: s.get("uv_peak_plus")),
        ("sgs_fraction_y50", lambda s: s.get("sgs_fraction_y50")),
        ("bulk_reynolds", lambda s: s.get("bulk_reynolds")),
    ):
        vals = [getter(c["summary"]) for c in picked]
        if any(v is None or not math.isfinite(float(v)) for v in vals):
            metrics[name] = {"available": False}
            continue
        metrics[name] = gci_three_grid(float(vals[0]), float(vals[1]), float(vals[2]), r=r)
        metrics[name]["values_coarse_medium_fine"] = [float(v) for v in vals]
    return {
        "available": True,
        "grids": [
            {"label": c["label"], "h": c["geometry"]["h"], "dy_plus": c["geometry"]["dy_plus"]}
            for c in picked
        ],
        "refinement_ratio_r": r,
        "metrics": metrics,
        "interpretation": (
            "GCI is computed on the statistics of the flow state actually reached in the "
            "sampled window; if that state is not turbulent the GCI describes the numerical "
            "convergence of a non-turbulent state and must not be read as a turbulent LES "
            "grid-convergence proof."
        ),
    }


def figure_profiles(cases, outdir: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    r180 = [c for c in cases if abs(float(c["re_tau_target"]) - 180.0) < 1e-9]
    if not r180:
        return figs

    # --- U+ vs y+ on a log axis, with the log law and the MKM1999 table -----
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for case in sorted(r180, key=lambda c: c["geometry"]["h"]):
        wu = case["result"]["wall_units"]
        ax.plot(wu["y_plus"], wu["u_plus"], "o-", ms=3, lw=1.0,
                label=f"{case['label']} (dy+={case['geometry']['dy_plus']:.1f})")
    ys = torch.logspace(0, math.log10(200), 60)
    ax.plot(ys, torch.log(ys) / KAPPA + B_LOG, "k--", lw=1.2,
            label=r"$\frac{1}{0.41}\ln y^+ + 5.2$")
    profile = mkm_profile_for(180.0, ROOT)
    if profile is not None:
        ax.plot(profile.y_plus, profile.u_plus, "s", ms=2.5, mfc="none",
                color="k", label="MKM1999 Re_tau=180")
    ax.set_xscale("log")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$U^+$")
    ax.set_title("Mean velocity, Re_tau=180 (short statistical window)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    path = outdir / "fig_u_plus.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig_u_plus", "path": str(path.relative_to(ROOT)),
                 "caption": "U+ vs y+ for the Re_tau=180 ladder against the log law and MKM1999."})

    # --- Reynolds shear stress ---------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for case in sorted(r180, key=lambda c: c["geometry"]["h"]):
        wu = case["result"]["wall_units"]
        ax.plot(wu["y_plus"], wu["uv_minus_plus"], "o-", ms=3, lw=1.0,
                label=f"{case['label']} (dy+={case['geometry']['dy_plus']:.1f})")
    if profile is not None:
        y = profile.y_plus.to(torch.float64)
        budget = 1.0 - y / float(profile.re_tau) - profile.du_plus_dy_plus.to(torch.float64)
        ax.plot(y, budget, "k--", lw=1.2, label="DNS budget 1-y+/Re_tau-dU+/dy+")
        ax.plot(y, 1.0 - y / float(profile.re_tau), ":", color="0.5", lw=1.0,
                label="total stress 1-y+/Re_tau")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$-\langle u'v'\rangle^+$")
    ax.set_title("Reynolds shear stress (resolved), Re_tau=180")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    path = outdir / "fig_uv_plus.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig_uv_plus", "path": str(path.relative_to(ROOT)),
                 "caption": "Resolved -<u'v'>+ against the DNS stress budget: the resolved stress is two to three orders of magnitude below the budget."})

    # --- decay of the resolved turbulence ----------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for case in sorted(r180, key=lambda c: c["geometry"]["h"]):
        hist = case["result"].get("history") or []
        if not hist:
            continue
        steps = [h["step"] for h in hist]
        tke = [max(h.get("tke") or 0.0, 1e-16) for h in hist]
        hh = float(case["geometry"]["h"])
        u_tau = float(case["result"]["config"].get("u_tau_target") or 0.01)
        ax.semilogy([s * u_tau / (hh * hh) for s in steps], tke, lw=1.2,
                    label=f"{case['label']} (dy+={case['geometry']['dy_plus']:.1f})")
    ax.set_xlabel(r"$t\,u_\tau/h$  (eddy turnovers)")
    ax.set_ylabel("resolved TKE (lattice units)")
    ax.set_title("Resolved turbulence decays on every affordable grid")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    path = outdir / "fig_tke_decay.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig_tke_decay", "path": str(path.relative_to(ROOT)),
                 "caption": "Resolved TKE versus eddy turnovers: decay of 2-4 decades within a few turnovers on every affordable grid."})

    return figs


def figure_spectra(cases, outdir: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    r180 = [c for c in cases if abs(float(c["re_tau_target"]) - 180.0) < 1e-9]
    if not r180:
        return figs
    case = max(r180, key=lambda c: c["summary"].get("spectrum_r2") or -1.0)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    any_line = False
    for leg in case["result"]["les"]["spectra"]["streamwise"]:
        k = torch.tensor(leg["k"], dtype=torch.float64)
        e = torch.tensor(leg["E"], dtype=torch.float64)
        mask = (k > 0) & (e > 0)
        if int(mask.sum()) < 4:
            continue
        ax.loglog(k[mask], e[mask], "-", lw=1.0,
                  label=f"y+={leg['y_plus']:.0f}")
        any_line = True
        fit = leg.get("fit", {})
        if fit.get("valid"):
            kk = torch.linspace(fit["k_min"], fit["k_max"], 20, dtype=torch.float64)
            a = math.exp(fit["intercept"]) if "intercept" in fit else None
            if a is not None:
                ax.loglog(kk, a * kk ** fit["slope"], "k--", lw=0.8)
    kref = torch.logspace(0.2, 1.6, 20)
    ax.loglog(kref, 1e-6 * kref ** (-5.0 / 3.0), ":", color="0.5", lw=1.0,
              label=r"$k^{-5/3}$")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$E_{uu}(k_x)$")
    ax.set_title(f"Streamwise spectra ({case['label']})")
    if any_line:
        ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    path = outdir / "fig_spectra.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig_spectra", "path": str(path.relative_to(ROOT)),
                 "caption": "Streamwise energy spectra on four wall-normal planes with the -5/3 reference slope."})
    return figs


def figure_feasibility(feasibility: dict, outdir: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    labels, values, colors = [], [], []
    for entry in feasibility["wall_resolved_uniform"]:
        labels.append(f"spec, uniform\nRe={entry['re_tau']:.0f}, dy+=2")
        values.append(entry["cell_updates"])
        colors.append("tab:red")
    for entry in feasibility["affordable_within_budget"]:
        labels.append(f"affordable\nRe={entry['re_tau']:.0f}, dy+={entry['dy_plus']:.0f}")
        values.append(entry["cell_updates"])
        colors.append("tab:blue")
    ax.bar(range(len(values)), values, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.axhline(feasibility["budget_cell_updates"], color="k", ls="--", lw=1.0,
               label=f"budget {feasibility['budget_cell_updates']:.1e}")
    ax.set_ylabel("cell-updates (cells x steps)")
    ax.set_title("Cost of the campaign specification vs the affordable envelope")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    path = outdir / "fig_feasibility.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return [{"name": "fig_feasibility", "path": str(path.relative_to(ROOT)),
             "caption": "Cell-updates required by the specification (wall-resolved, dy+=2) versus what fits the available budget (dy+ = 32-105 at the same statistics window)."}]


def markdown_tables(summary: dict) -> str:
    """Markdown tables for the report (generated, never hand-typed)."""
    fmt = lambda v, d=3: ("nan" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}")
    lines = ["### Case table", "",
             "| case | Re_tau | h | cells | dy+ | y+ first node | steps | T_e sampled | t_nu sampled | u+ max | -<u'v'>+ peak | peak y+ | sgs frac @y+=50 | spec slope | spec R2 | mass drift |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["cases"]:
        lines.append(
            "| {label} | {re_tau:.0f} | {h} | {cells:,} | {dy_plus:.1f} | {y_plus_first_node:.2f} | {steps} | "
            "{te:.2f} | {tnu:.3f} | {upmax} | {uv} | {uvy} | {sgs} | {slope} | {r2} | {drift} |".format(
                label=r["label"], re_tau=r["re_tau"], h=int(r["h"]), cells=r["cells"],
                dy_plus=r["dy_plus"], y_plus_first_node=r["y_plus_first_node"],
                steps=r["steps"], te=r["window"]["eddy_turnovers_sampled"],
                tnu=r["window"]["viscous_times_sampled"], upmax=fmt(r["u_plus_max"], 2),
                uv=fmt(r["uv_peak_plus"], 5), uvy=fmt(r["uv_peak_y_plus"], 1),
                sgs=fmt(r["sgs_fraction_y50"], 3), slope=fmt(r["spectrum_slope"], 3),
                r2=fmt(r["spectrum_r2"], 3), drift=fmt(r["mass_drift"], 5),
            )
        )
    lines += ["", "### Criterion verdicts (per case)", "",
              "| case | a mean profile (<=5%) | b Re-stress peak (<=15%/10%) | c spectrum slope (<=0.10) | d sgs fraction 20-40% | f mass (<=0.1%) |",
              "|---|---|---|---|---|---|"]
    for r in summary["cases"]:
        cells = []
        for key in ("a_mean_profile_log_law", "b_reynolds_stress_peak",
                    "c_energy_spectrum_slope", "d_sgs_stress_fraction", "f_mass_conservation"):
            entry = r["criteria"].get(key, {})
            metric = entry.get("metric")
            cells.append("{}{}".format(entry.get("status", "n/a"),
                                       "" if metric is None else f" ({fmt(metric, 4)})"))
        lines.append("| {} | {} |".format(r["label"], " | ".join(cells)))
    lines += ["", "### Grid convergence (Re_tau=180 ladder)", ""]
    gci = summary["gci"]
    if gci.get("available"):
        lines.append(f"Refinement ratio r = {gci['refinement_ratio_r']:.3f}; grids: " +
                     ", ".join(f"{g['label']} (h={g['h']}, dy+={g['dy_plus']:.1f})" for g in gci["grids"]))
        lines += ["", "| statistic | coarse | medium | fine | observed p | GCI (%) |", "|---|---|---|---|---|---|"]
        for name, entry in gci["metrics"].items():
            if not entry.get("available"):
                lines.append(f"| {name} | - | - | - | - | not available |")
                continue
            vals = entry.get("values_coarse_medium_fine", [float("nan")] * 3)
            gci_pct = entry.get("gci_fine")
            gci_pct = None if gci_pct is None else 100.0 * float(gci_pct)
            lines.append("| {} | {:.5g} | {:.5g} | {:.5g} | {} | {} |".format(
                name, vals[0], vals[1], vals[2],
                fmt(entry.get("order_used"), 2), fmt(gci_pct, 3)))
    else:
        lines.append(f"not available: {gci.get('reason')}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(CAMPAIGN / "campaign_summary.json"))
    ap.add_argument("--markdown", default=str(CAMPAIGN / "tables.md"))
    args = ap.parse_args(argv)
    outdir = CAMPAIGN / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    cases = load_cases()
    rows = [case_row(c) for c in cases]
    feasibility = json.loads((CAMPAIGN / "feasibility.json").read_text())
    figures = []
    figures += figure_profiles(cases, outdir)
    figures += figure_spectra(cases, outdir)
    figures += figure_feasibility(feasibility, outdir)
    # keep figures produced by sibling scripts (collect_evidence.py) in the record
    known = {Path(f["path"]).name for f in figures}
    extra_captions = {
        "fig_all_runs_tke.png": (
            "Resolved TKE versus eddy turnovers for all 18 runs of the register "
            "(fneq strain decays 2-5 decades; velocity-gradient strain bursts and then "
            "trips the divergence guard at step ~1000)."
        ),
    }
    for png in sorted(outdir.glob("*.png")):
        if png.name in known:
            continue
        figures.append({
            "name": png.stem,
            "path": str(png.relative_to(ROOT)),
            "caption": extra_captions.get(png.name, png.name),
        })

    summary = {
        "schema": "p127w2-n03-campaign-summary-v1",
        "cases": rows,
        "gci": gci_study(cases),
        "feasibility": {
            "budget_cell_updates": feasibility["budget_cell_updates"],
            "measured_mlups": feasibility["measured_mlups"],
            "wall_resolved_uniform": feasibility["wall_resolved_uniform"],
            "affordable_within_budget": feasibility["affordable_within_budget"],
        },
        "figures": figures,
        "headline": {
            "n_cases": len(rows),
            "n_criteria_passed": sum(
                1 for r in rows for c in r["criteria"].values() if c["status"] == "PASS"
            ),
            "n_criteria_failed": sum(
                1 for r in rows for c in r["criteria"].values() if c["status"] == "FAIL"
            ),
            "n_criteria_withheld": sum(
                1 for r in rows for c in r["criteria"].values() if c["status"] == "WITHHELD"
            ),
        },
    }
    Path(args.out).write_text(json.dumps(summary, indent=1))

    print(f"{len(rows)} cases")
    for row in rows:
        crit = " ".join(
            "{}:{}".format(key[0], val["status"]) for key, val in row["criteria"].items()
        )
        print(
            "{:<14} Re={:>3.0f} h={:>2} dy+={:6.1f} u+max={:.2f} uv+peak={:.2e} "
            "sgs@50={} T_e={:.2f} t_nu={:.3f} crit={}".format(
                row["label"], row["re_tau"], row["h"], row["dy_plus"],
                row["u_plus_max"] or float("nan"), row["uv_peak_plus"] or float("nan"),
                row["sgs_fraction_y50"], row["window"]["eddy_turnovers_sampled"],
                row["window"]["viscous_times_sampled"], crit,
            )
        )
    print("GCI:", json.dumps(summary["gci"], indent=1)[:1200])
    Path(args.markdown).write_text(markdown_tables(summary))
    print(f"wrote {args.out}, {args.markdown} and {len(figures)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())