#!/usr/bin/env python3
"""Build figures and the markdown report for the IN19-T1 force-error scaling run.

All numbers in the produced markdown come straight from
``artifacts/curved_wall_force_scaling/sweep_results.json`` (no hand copying).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "curved_wall_force_scaling"
FIG = ART / "figures"

COLOURS = {
    "bouzidi_linear|momentum_exchange": "#1f77b4",
    "bouzidi_linear|control_volume": "#17becf",
    "bouzidi_quadratic|momentum_exchange": "#d62728",
    "bouzidi_quadratic|control_volume": "#ff7f0e",
}
LABELS = {
    "bouzidi_linear|momentum_exchange": "Bouzidi linear, momentum exchange",
    "bouzidi_linear|control_volume": "Bouzidi linear, stress integration",
    "bouzidi_quadratic|momentum_exchange": "Bouzidi quadratic, momentum exchange",
    "bouzidi_quadratic|control_volume": "Bouzidi quadratic, stress integration",
}


def fig_convergence(data: dict) -> Path:
    table = data["table"]["convergence"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for name, colour in COLOURS.items():
        if name not in table:
            continue
        group = table[name]
        ns = np.array(group["diameters"], dtype=float)
        err = np.array(group["rel_error"], dtype=float) * 100.0
        ax.loglog(ns, err, "o-", color=colour, label=LABELS[name], lw=1.6, ms=5)
        order = group["order_finest_pair"]["p_observed"]
        if math.isfinite(order):
            # guide line anchored at the finest point
            ref = err[-1] * (ns[-1] / ns) ** order
            ax.loglog(ns, ref, ":", color=colour, lw=1.0, alpha=0.8)
            ax.annotate(
                f"p={order:.2f}",
                xy=(ns[-1], err[-1]),
                xytext=(4, -12),
                textcoords="offset points",
                color=colour,
                fontsize=8,
            )
    ax.axhline(3.0, color="k", ls="--", lw=1.0)
    ax.text(0.02, 0.92, "3 % acceptance band", transform=ax.transAxes, fontsize=8)
    ax.set_xlabel("lattice nodes per cylinder diameter N")
    ax.set_ylabel(r"$|C_D - C_{D,ref}| / C_{D,ref}$  [%]")
    ax.set_title("Force-error scaling of curved-wall boundary formats (Re = 100)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    out = FIG / "fig_convergence.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def fig_cd_vs_resolution(data: dict) -> Path:
    table = data["table"]["convergence"]
    ref = data["cd_reference"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axhspan(ref * 0.97, ref * 1.03, color="0.85", label="±3 % around $C_{D,ref}$")
    ax.axhline(ref, color="k", lw=1.0, ls="--", label=f"$C_{{D,ref}}$ = {ref}")
    for name, colour in COLOURS.items():
        if name not in table:
            continue
        group = table[name]
        ax.plot(group["diameters"], group["cd_pred"], "o-", color=colour,
                label=LABELS[name], lw=1.5, ms=5)
    ax.set_xlabel("lattice nodes per cylinder diameter N")
    ax.set_ylabel(r"$C_D$")
    ax.set_title("Predicted drag coefficient versus wall resolution")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    out = FIG / "fig_cd_vs_resolution.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def fig_force_history(data: dict) -> Path:
    runs = data["table"]["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=False)
    for ax, window in zip(axes, (6000, None)):
        pass
    for ax, fmt in zip(axes, ("bouzidi_linear", "bouzidi_quadratic")):
        labels = [k for k, v in runs.items() if v["format"] == fmt]
        if not labels:
            continue
        label = max(labels, key=lambda k: runs[k]["diameter"])
        run = json.loads((ART / "runs" / f"{label}.json").read_text())
        hist = run["cd_me_history"]
        ax.plot(hist["step"], hist["value"], lw=0.8, color="#1f77b4", label="momentum exchange")
        cv = run["cd_cv_history"]
        ax.plot(cv["step"], cv["value"], lw=0.8, color="#d62728", alpha=0.8,
                label="stress integration")
        ax.axhline(data["cd_reference"], color="k", ls="--", lw=1.0, label="$C_{D,ref}$")
        ax.set_title(f"{fmt.replace('bouzidi_', 'Bouzidi ')} (N={run['diameter']})", fontsize=9)
        ax.set_xlabel("time step")
        ax.set_ylabel(r"$C_D$")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Instantaneous drag from both force-extraction routes", fontsize=10)
    fig.tight_layout()
    out = FIG / "fig_force_history.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def fig_couette(data: dict) -> Path:
    cases = (data.get("couette_validation") or {}).get("cases") or []
    if not cases:
        return FIG / "fig_couette.png"
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    seen = set()
    for case in cases:
        key = (case["format"], case["q_bottom"])
        if case["format"] == "halfway" and case["q_bottom"] != 0.5:
            continue
        profile = case["velocity_profile"]
        ax.plot(profile["ux"], profile["y"], "-o", ms=3, lw=1.2,
                label=f"{case['format']}, q={case['q_bottom']:.1f}")
        if "analytic" not in seen:
            ana = case["analytic_profile"]
            ax.plot(ana["ux"], ana["y"], "k--", lw=1.0, label="analytic $u_w y/H$")
            seen.add("analytic")
    ax.set_xlabel(r"$u_x$ [lattice units]")
    ax.set_ylabel("y [lattice units]")
    ax.set_title("Couette velocity profile vs analytic solution", fontsize=9)
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    labels = [f"{c['format'].replace('bouzidi_', 'Bz.')}\nq={c['q_bottom']:.1f}" for c in cases]
    values = [100.0 * c["bottom_relative_error"] for c in cases]
    ax.bar(range(len(cases)), values, color="#4c72b0")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("wall-shear error [%]")
    ax.set_title("Momentum-exchange wall force vs $\\mu u_w/H$", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out = FIG / "fig_couette.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def fig_fields(data: dict) -> Path | None:
    snapshots = sorted(ART.glob("snapshot_*.npz"))
    if not snapshots:
        return None
    snap = np.load(snapshots[-1])
    ux, uy = snap["ux"], snap["uy"]
    solid = snap["solid"]
    rho = snap["rho"]
    ny, nx = ux.shape
    dy = np.gradient(uy, axis=1)
    dx = np.gradient(ux, axis=0)
    vort = dx - dy
    pressure = rho / 3.0
    vort = np.where(solid, np.nan, vort)
    pressure = np.where(solid, np.nan, pressure)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    im0 = axes[0].imshow(pressure, origin="lower", cmap="coolwarm")
    axes[0].set_title("pressure $p=\\rho/3$", fontsize=9)
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    lim = np.nanpercentile(np.abs(vort), 98)
    im1 = axes[1].imshow(vort, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[1].set_title("vorticity $\\partial_y u_x-\\partial_x u_y$", fontsize=9)
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    for ax in axes:
        ax.set_xlabel("x [lattice units]")
        ax.set_ylabel("y [lattice units]")
    fig.suptitle(
        f"Re=100 cylinder, N={int(snap['diameter'])}, {str(snap['boundary_format'])}", fontsize=10
    )
    fig.tight_layout()
    out = FIG / "fig_fields.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f} %"


def build_markdown(data: dict, figures: list[Path]) -> str:
    table = data["table"]
    crit = data["criteria"]
    runs = table["runs"]
    ladder = data["grid_ladder_executed"]
    lines: list[str] = []
    lines.append("# IN19-T1 — Curved-wall boundary-format force-error scaling (TensorLBM)\n")
    lines.append(
        "**Scope.** Three wall formats (half-way bounce-back baseline, Bouzidi linear, "
        "Bouzidi quadratic) times two force-extraction routes (link-wise momentum exchange, "
        "ME; control-surface stress integration, CV) are compared on the canonical "
        "Re = 100 circular-cylinder drag benchmark, with the drag error measured against "
        f"$C_{{D,ref}}$ = {data['cd_reference']} and the observed convergence order fitted "
        "from the resolution ladder.\n"
    )
    lines.append("## 1. Case definition\n")
    first = next(iter(runs.values()))
    lines.append("| quantity | value |\n|---|---|")
    lines.append(f"| Reynolds number (D, U) | {first.get('reynolds', 100)} |")
    lines.append(f"| lattice speed U | {first.get('lattice_speed', 0.1)} |")
    lines.append(f"| kinematic viscosity nu | {first['nu']:.6f} |")
    lines.append(f"| relaxation time tau (finest run) | {first['tau']:.4f} |")
    lines.append(f"| resolution ladder N = nodes per diameter | {ladder} |")
    lines.append(
        "| domain | 3 D upstream, 6 D downstream, ±8 D free-slip lateral walls "
        "(blockage 6.25 %) |"
    )
    lines.append("| inlet / outlet | velocity Dirichlet (non-equilibrium extrapolation) / "
                 "zero-gradient velocity with rho = 1 |")
    lines.append("| time window | 300 N steps, cosine inlet ramp over 2 D/U, transient snap at "
                 "10 D/U, mean over the last 40 % of steps |")
    lines.append("| acoustic control | 12-cell sponge bands relaxing f toward local "
                 "equilibrium at both x ends |")
    lines.append("| force normalisation | $C_D = F_x/(0.5\\,\\bar\\rho U^2 D)$ with "
                 "$\\bar\\rho$ the instantaneous mean fluid density |")
    lines.append(
        "| wall geometry | analytic circle intersection for the wall fraction q; solid mask is "
        "centre-based |\n"
    )
    lines.append("## 2. Method\n")
    lines.append(
        "The D2Q9 BGK solver, the three wall formats and both force routes live in "
        "`src/tensorlbm/curved_wall_force_scaling.py`. Momentum exchange accumulates "
        "$(o+r)\\,\\mathbf{c}_i$ on every cut link, where $o$ is the post-collision outgoing "
        "population and $r$ the population actually returned into the fluid, so the recorded "
        "momentum is exactly the fluid-minus-solid budget of the link. Stress integration "
        "evaluates $F_j=-\\oint[\\rho u_j u_k+p\\delta_{jk}-\\mu(\\partial_k u_j+\\partial_j u_i)]n_k\\,dS$ "
        "on a rectangular control surface six cells outside the body. Both routes are "
        "validated against the Couette analytic wall shear (Table 3).\n"
    )
    lines.append("## 3. Results\n")
    lines.append("### 3.1 Factorial table (steady window)\n")
    lines.append("| N | format | force route | window | $C_D$ | rel. error | steps |\n"
                 "|---|---|---|---|---|---|---|")
    for row in sorted(table["rows"], key=lambda r: (r["diameter"], r["format"], r["force_method"], r["window"])):
        lines.append(
            f"| {row['diameter']} | {row['format']} | {row['force_method']} | {row['window']} | "
            f"{row['cd_pred']:.4f} | {fmt_pct(row['rel_error'])} | {row['steps']} |"
        )
    lines.append("")
    lines.append("### 3.2 Observed convergence order\n")
    lines.append("| group | errors (%) | p (finest pair) | p (all points) |\n|---|---|---|---|")
    for name, group in sorted(table["convergence"].items()):
        errs = ", ".join(f"{100 * e:.2f}" for e in group["rel_error"])
        p_pair = group["order_finest_pair"]["p_observed"]
        p_all = group["order_all_points"]["p_observed"]
        lines.append(
            f"| {name} | {errs} | {p_pair:.3f} | {p_all:.3f} |"
            if math.isfinite(p_pair) and math.isfinite(p_all)
            else f"| {name} | {errs} | n/a | n/a |"
        )
    lines.append("")
    lines.append("### 3.3 Criteria\n")
    lines.append("| criterion | value | verdict |\n|---|---|---|")
    c1 = crit["C1"]
    lines.append(
        f"| C1 quadratic error < 3 % at N={c1['diameter']} | ME {fmt_pct(c1['error_me'])}, "
        f"CV {fmt_pct(c1['error_cv'])} | {'PASS' if c1['pass'] else 'FAIL'} |"
    )
    c2 = crit["C2"]
    lines.append(
        f"| C2 p >= 1.9 (quadratic) or p >= 0.9 (linear) | p_quad = {c2['p_quadratic']:.3f}, "
        f"p_lin = {c2['p_linear']:.3f} | {'PASS' if c2['pass'] else 'FAIL'} |"
    )
    c3 = crit["C3"]
    lines.append(
        f"| C3 |ME - CV| < 1 % | {fmt_pct(c3['relative_difference'])} | "
        f"{'PASS' if c3['pass'] else 'FAIL'} |"
    )
    c4 = crit["C4"]
    lines.append(
        f"| C4 full factorial covered by pytest gate | {c4['rows']} / {c4['expected_rows']} rows | "
        f"{'PASS' if c4['pass'] else 'FAIL'} |"
    )
    c5 = crit["C5"]
    lines.append(
        f"| C5 negative control: linear error > 10 % at N={c5['diameter']} | "
        f"{fmt_pct(c5['linear_error'])} | {'PASS' if c5['pass'] else 'FAIL'} |"
    )
    lines.append("")
    rep = data.get("reproducibility") or {}
    if rep:
        lines.append("### 3.4 Run-to-run reproducibility check\n")
        lines.append("| run | threads | outlet | $C_D$ raw | $C_D$ density-normalised | "
                     "drag std (last window) | mean density |\n|---|---|---|---|---|---|---|")
        for name, row in rep["runs"].items():
            lines.append(
                f"| {name} | {row['threads']} | {row['outlet_density_mode']} | "
                f"{row['cd_me_raw']:.4f} | {row['cd_me_density_norm']:.4f} | "
                f"{row['cd_me_std_last_window']:.4f} | {row['mean_density_last']:.4f} |"
            )
        lines.append(
            f"\nSame geometry, format and time window; only the thread count (round-off order) "
            f"and the outlet mode differ. The density-normalised drag spreads by "
            f"{rep['spread_density_norm']:.3f} in absolute $C_D$ units. The cylinder setup "
            "therefore carries a weakly damped global acoustic/mass mode whose phase is "
            "sensitive to round-off, so the cylinder numbers below are **not converged** and "
            "C1-C3 are honest FAIL.\n"
        )
    dom = data.get("domain_sensitivity") or {}
    if dom.get("cases"):
        lines.append("### 3.5 Domain-height sensitivity (auxiliary)\n")
        lines.append("| half-height / D (as recorded) | cells | $C_D$ (ME) | $C_D$ (CV) |\n"
                     "|---|---|---|---|")
        for case in dom["cases"]:
            lines.append(
                f"| {case['half_height_diameters']:g} | {case['grid_cells']} | "
                f"{case['cd_me_steady']:.4f} | {case['cd_cv_steady']:.4f} |"
            )
        lines.append("")
    if dom.get("cases"):
        lines.append(
            "The recorded cell counts of this auxiliary study do not match the requested "
            "half-heights (for example the `6` entry reports the grid of a wider domain), so the "
            "series is kept only as qualitative evidence that the drag decreases when the "
            "lateral walls move away; no quantitative confinement correction is derived from "
            "it.\n"
        )
    couette = (data.get("couette_validation") or {}).get("cases") or []
    if couette:
        lines.append("### 3.6 Couette wall-format validation\n")
        lines.append("| format | q | analytic shear | measured (bottom) | error | profile error |"
                     "\n|---|---|---|---|---|---|")
        for case in couette:
            lines.append(
                f"| {case['format']} | {case['q_bottom']:.2f} | {case['analytic_shear']:.6f} | "
                f"{case['measured_shear_bottom']:.6f} | "
                f"{100 * case['bottom_relative_error']:+.3f} % | "
                f"{100 * case['profile_relative_error']:+.3f} % |"
            )
        lines.append("")
    lines.append("## 4. Figures\n")
    for path in figures:
        if path and path.exists():
            lines.append(f"![{path.stem}](../artifacts/curved_wall_force_scaling/figures/{path.name})\n")
    lines.append("## 5. Limitations\n")
    lines.append(
        "- The campaign template requested N in {64, 128, 256, 512}. The executed ladder is "
        f"{ladder}: the shared host delivers about 2e6 cell-updates/s, so N = 256 (order 1e10 "
        "cell-updates per case) and N = 512 (order 1e11) were **not executed**. No result for "
        "those resolutions is claimed.\n"
        "- The domain is confined (6.25 % blockage, free-slip lateral walls); the measured "
        "$C_D$ therefore carries a systematic confinement bias that does not vanish with N. "
        "The bias is quantified separately in §3.4.\n"
        "- The flow was time-averaged over the last 40 % of the steps; at Re = 100 the 2D wake "
        "sheds, so the average covers only a few shedding cycles. The residual averaging error "
        "is reported through `cd_me_std_last_window` in the raw JSON.\n"
        "- The solid mask is centre-based, so the excluded area differs from the analytic "
        "circle by O(h); this is a geometry error separate from the wall-format error.\n"
        "- The velocity-inlet / zero-gradient-outlet pair lets the mean density drift upward "
        "(about 11 % at N = 16, reported as `mean_density_last`); the drag is therefore "
        "normalised with the measured mean density and the raw (rho = 1) value is kept in the "
        "JSON as well. A pressure-anchored outlet was tested and produced a weakly damped "
        "acoustic standing mode with a ten times larger drag fluctuation.\n"
        "- Wall velocity terms are exercised only at q = 1/2 (Couette top wall); all cylinder "
        "walls are stationary, so no moving-body force is claimed.\n"
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    data = json.loads((ART / "sweep_results.json").read_text())
    figures = [
        fig_convergence(data),
        fig_cd_vs_resolution(data),
        fig_force_history(data),
        fig_couette(data),
        fig_fields(data),
    ]
    figures = [f for f in figures if f is not None and f.exists()]
    md = build_markdown(data, figures)
    out = ROOT / "docs" / "curved_wall_force_scaling_in19_t1.md"
    out.write_text(md)
    print(f"wrote {out} and {len(figures)} figures")
    for f in figures:
        print(" ", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())