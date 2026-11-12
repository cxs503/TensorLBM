#!/usr/bin/env python3
"""m15891 board audit — independent recomputation of CASE-MCMP-LAPLACE (m15467).

Reads only committed artifacts (no solver call), recomputes every headline
number of the case from scratch, extracts one droplet state from
``hero_fields.npz`` through the library ``measure_laplace`` and renders the
audit figure.  Writes ``artifacts/m15891_mcmp_audit/evidence.json`` (``runs/`` is gitignored).

Usage: ``python scripts/audit_m15891_mcmp_laplace.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CASE = ROOT / "benchmarks" / "case_pool" / "mcmp_laplace"
OUT = ROOT / "artifacts" / "m15891_mcmp_audit"
FIG = ROOT / "figures"

sys.path.insert(0, str(ROOT / "src"))
from tensorlbm.mcmp_verification import measure_laplace  # noqa: E402


def through_origin(xs, ys):
    return sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)


def free_line(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    a = sxy / sxx
    b = my - a * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    return a, b, 1.0 - ss_res / ss_tot


def main() -> int:
    res = json.loads((CASE / "artifacts" / "summary.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)

    levels = []
    for lv in res["levels"]:
        G = lv["G_12"]
        cases = [c for c in res["cases"] if round(c["G_12"], 4) == round(G, 4)]
        xs = [1.0 / c["radius"] for c in cases]
        ys = [c["dp"] for c in cases]
        s0 = through_origin(xs, ys)
        errs = [abs(y - s0 * x) / (s0 * x) for x, y in zip(xs, ys)]
        a, b, r2 = free_line(xs, ys)
        off = max(abs(b) * c["radius"] / abs(a) for c in cases)
        levels.append(
            {
                "G_12": G,
                "n_radii": len(cases),
                "sigma_through_origin_auditor": s0,
                "sigma_through_origin_stored": lv["sigma_origin"],
                "delta_vs_stored": s0 - lv["sigma_origin"],
                "max_rel_err_auditor_pct": 100 * max(errs),
                "max_rel_err_stored_pct": 100 * lv["max_rel_err"],
                "per_point_err_pct": [100 * e for e in errs],
                "per_point_err_vs_free_line_pct": [
                    100 * (y - a * x - b) / (a * x + b) for x, y in zip(xs, ys)
                ],
                "tol_pct": 5.0,
                "within_tol": max(errs) < 0.05,
                "free_line_slope_sigma": a,
                "free_line_intercept": b,
                "free_line_r2": r2,
                "free_line_slope_vs_through_origin_pct": 100 * (a - s0) / s0,
                "max_offset_fraction_pct": 100 * off,
                "offset_guardrail_le_5pct": off <= 0.05,
                "sigma_i_dpR": [c["dp"] * c["radius"] for c in cases],
                "radii": [c["radius"] for c in cases],
                "dp": ys,
            }
        )

    # --- raw-field spot check: hero droplet re-measured from the committed npz ---
    z = np.load(CASE / "artifacts" / "hero_fields.npz")
    run = {
        "G_12": float(z["G_12"]),
        "n": int(z["n"]),
        "radius_init": float(z["radius_init"]),
        "steps_run": -1,
        "converged": True,
        "u_max": 0.0,
        "rho1": torch.tensor(z["rho1"], dtype=torch.float64),
        "rho2": torch.tensor(z["rho2"], dtype=torch.float64),
    }
    hero = measure_laplace(run, dims=2)
    stored_hero = next(
        c
        for c in res["cases"]
        if round(c["G_12"], 4) == round(run["G_12"], 4)
        and abs(c["radius_init"] - run["radius_init"]) < 1e-9
    )
    spot = {
        "hero_selection": {"G_12": run["G_12"], "radius_init": run["radius_init"], "n": run["n"]},
        "dp_auditor": hero["dp"],
        "dp_stored": stored_hero["dp"],
        "dp_rel_diff": (hero["dp"] - stored_hero["dp"]) / stored_hero["dp"],
        "radius_auditor": hero["radius"],
        "radius_stored": stored_hero["radius"],
        "sigma_auditor": hero["sigma"],
        "sigma_stored": stored_hero["sigma"],
        "density_ratio_auditor": hero["density_ratio"],
        "laplace_dims": hero["laplace_dims"],
        "doc_mislabel": "README/docs say the hero case is G12=-2.500; npz G_12=-2.750, R_init=20",
    }

    evidence = {
        "audit_mission": 15891,
        "audited_mission": 15467,
        "audited_commit": "e547c37c (PR #1822 merged as cf959e0d)",
        "case_dir": str(CASE.relative_to(ROOT)),
        "protocol": res["protocol"],
        "thresholds": res["thresholds"],
        "levels_auditor": levels,
        "routeB": [
            {
                "G_12": r["G_12"],
                "sigma_mech": r["sigma_mech"],
                "flatness": r["flatness"],
                "converged": r["converged"],
                "valid": bool(r["flatness"] <= res["thresholds"]["flat_flatness_max"]),
            }
            for r in res["routeB"]
        ],
        "tol_family_pass_reported": res["verdict"]["tol_family_pass"],
        "guard_family_reported": res["verdict"].get("guard_family"),
        "hero_spot_check": spot,
        "resolution_study": res["resolution"],
        "coexistence": res["coexistence"],
        "verdict_all_tol_items_pass": all(l["within_tol"] for l in levels),
        "verdict_offset_guardrail_pass": all(l["offset_guardrail_le_5pct"] for l in levels),
    }
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=1, default=float))

    # ---------------- figure: ref-vs-got + sigma drift ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    colors = {-2.5: "#c0392b", -2.75: "#2980b9", -3.0: "#27ae60"}
    ax = axes[0]
    allx = []
    for lv in levels:
        xs = [1.0 / r for r in lv["radii"]]
        allx += xs
        ax.plot(xs, lv["dp"], "o", color=colors[lv["G_12"]], label=f"got G12={lv['G_12']:g}")
        xx = np.linspace(0, max(allx) * 1.05, 50)
        ax.plot(
            xx,
            lv["sigma_through_origin_auditor"] * xx,
            "-",
            color=colors[lv["G_12"]],
            lw=1.4,
            label=f"ref σ₀/R (σ₀={lv['sigma_through_origin_auditor']:.4f})",
        )
    ax.set_xlabel("1/R  [lattice units⁻¹]")
    ax.set_ylabel(r"$\Delta p$  [lattice units]")
    ax.set_title("(a) Laplace law: Δp vs 1/R, audited through-origin reference")
    ax.set_xlim(0, max(allx) * 1.05)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    for lv in levels:
        sig = np.array(lv["sigma_i_dpR"])
        rel = 100 * (sig - lv["sigma_through_origin_auditor"]) / lv["sigma_through_origin_auditor"]
        ax.plot(lv["radii"], rel, "o-", color=colors[lv["G_12"]], lw=1.2,
                label=f"G12={lv['G_12']:g}: max {np.abs(rel).max():.2f}%, "
                      f"free-line σ {lv['free_line_slope_vs_through_origin_pct']:+.1f}%, "
                      f"|b|R/|a| {lv['max_offset_fraction_pct']:.1f}%")
    ax.axhspan(-5, 5, color="0.85", zorder=0, label="predeclared ±5% door")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("measured droplet radius R  [lattice units]")
    ax.set_ylabel(r"$(\sigma_i-\sigma_0)/\sigma_0$  [%]")
    ax.set_title("(b) radius independence σ_i=Δp·R: drift + offset diagnostic")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.suptitle(
        "m15891 audit of m15467 CASE-MCMP-LAPLACE — independent recompute from committed summary.json",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG / "audit_m15891_mcmp_laplace.png", dpi=150)
    print(json.dumps({"evidence": str(OUT / "evidence.json"), "figure": str(FIG / "audit_m15891_mcmp_laplace.png"),
                      "levels": [{k: l[k] for k in ("G_12", "max_rel_err_auditor_pct", "free_line_slope_vs_through_origin_pct", "max_offset_fraction_pct")} for l in levels],
                      "spot_check": spot}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())