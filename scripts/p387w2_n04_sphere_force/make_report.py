#!/usr/bin/env python3
"""Assemble the acceptance record and the figures for node ``p387w2-n04``.

Reads the per-case JSON records written by ``run_validation.py`` and produces

    artifacts/p387w2_n04/acceptance.json      machine-checked criteria + numbers
    artifacts/p387w2_n04/figures/*.png        layer plots

The acceptance logic is written so that a *missing* case is a failure with an
explicit reason, never a silent pass.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.drag_closure import cd_clift_gauvin, cd_schiller_naumann  # noqa: E402
from tensorlbm.ibm_sphere_surface_force import (  # noqa: E402
    observed_order,
    stokes_torque_coefficient,
)

CASES = ROOT / "artifacts" / "p387w2_n04" / "cases"
OUT = ROOT / "artifacts" / "p387w2_n04"
FIGS = OUT / "figures"

# --- acceptance tolerances (from the campaign node contract) ---------------
CD_TOL_LOW = 10.0    # % for Re <= 100
CD_TOL_HIGH = 15.0   # % for 100 < Re <= 1000
CM_TOL = 10.0        # %
ORDER_MIN = 0.8
MASS_TOL = 1e-12


def load_cases() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(CASES.glob("*.json")):
        rec = json.loads(path.read_text())
        out[rec["name"]] = rec
    return out


def _band_error(cd: float, re: float) -> tuple[float, float, float]:
    sn = cd_schiller_naumann(re)
    cg = cd_clift_gauvin(re)
    lo, hi = min(sn, cg), max(sn, cg)
    mid = 0.5 * (lo + hi)
    # signed error against the band mid-point, and the distance to the band
    err_mid = 100.0 * (cd - mid) / mid
    if cd < lo:
        err_band = 100.0 * (cd - lo) / mid
    elif cd > hi:
        err_band = 100.0 * (cd - hi) / mid
    else:
        err_band = 0.0
    return err_mid, err_band, 100.0 * (hi - lo) / mid


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    drag = {
        k: v
        for k, v in cases.items()
        if v.get("layer") == "drag" and "cd_mean" in v and v.get("status") is None
    }
    torque = {
        k: v
        for k, v in cases.items()
        if v.get("layer") == "torque" and "cm_mean" in v and v.get("status") is None
    }
    failed = [v for v in cases.values() if v.get("status")]
    controls = cases.get("negative_controls", {})

    # ---------------- layer 1: drag vs the adjudicated band ----------------
    drag_rows = []
    for name, rec in sorted(drag.items(), key=lambda kv: kv[1]["reynolds_measured"]):
        re = rec["reynolds_measured"]
        cd = rec["cd_mean"]
        err_mid, err_band, spread = _band_error(cd, re)
        tol = CD_TOL_LOW if re <= 100.0 else CD_TOL_HIGH
        drag_rows.append(
            {
                "case": name,
                "reynolds_target": rec["config"]["reynolds_target"],
                "reynolds": re,
                "diameter": rec["config"]["diameter"],
                "lateral_diameters": rec["config"]["lateral_diameters"],
                "cd": cd,
                "cd_sem": rec["cd_std"] / max(math.sqrt(max(rec["config"]["steps"] // 200, 1)), 1.0),
        "partial_from_log": bool(rec.get("partial_from_log", False)),
                "cd_std": rec["cd_std"],
                "cd_drive": rec["cd_drive_mean"],
                "sn": cd_schiller_naumann(re),
                "cg": cd_clift_gauvin(re),
                "band_mid": 0.5 * (cd_schiller_naumann(re) + cd_clift_gauvin(re)),
                "band_spread_pct": spread,
                "err_vs_band_mid_pct": err_mid,
                "err_outside_band_pct": err_band,
                "tolerance_pct": tol,
                "pass": abs(err_band) <= tol,
                "steps": rec["config"]["steps"],
                "tau": rec["config"].get("tau"),
                "dtype": rec["config"].get("dtype"),
                "collision": rec["config"].get("collision"),
                "samples": rec.get("samples"),
                "stable": math.isfinite(cd),
            }
        )

    # ---------------- layer 2: observed order at Re = 100 ------------------
    # the refinement study must hold the *box in diameters* fixed, otherwise the
    # observed order mixes mesh refinement with blockage
    conv = [
        r
        for r in drag_rows
        if abs(r["reynolds"] - 100.0) < 25.0
        and abs(r["lateral_diameters"] - 5.0) < 1e-9
        and r["diameter"] is not None
        and not r["partial_from_log"]
    ]
    order = None
    if len(conv) >= 3:
        hs = [1.0 / r["diameter"] for r in conv]
        vals = [r["cd"] for r in conv]
        # The three grids do not settle at exactly the same Reynolds number under
        # a fixed drive, and the reference C_D depends on Re.  The convergence
        # study is therefore run on the *normalised* drag
        #     ratio(h) = C_D(h) / C_D,band(Re_measured)
        # which is 1 for a perfect scheme at every Re; the raw-C_D order is
        # reported next to it for completeness.
        ratios = [r["cd"] / r["band_mid"] for r in conv]
        order = observed_order(hs, ratios)
        order["quantity"] = "C_D / C_D,band(Re_measured)"
        order_raw = observed_order(hs, vals)
        order["order_raw_cd"] = order_raw["order"]
        order["extrapolated_raw_cd"] = order_raw["extrapolated"]
        order["grids"] = [
            {
                "diameter": r["diameter"],
                "reynolds": round(r["reynolds"], 2),
                "cd": r["cd"],
                "cd_band": round(r["band_mid"], 4),
                "ratio": round(r["cd"] / r["band_mid"], 4),
                "steps": r["steps"],
            }
            for r in conv
        ]

    # ---------------- layer 3: torque vs Stokes ----------------------------
    torque_rows = []
    for name, rec in sorted(torque.items(), key=lambda kv: kv[1]["config"]["re_omega_target"]):
        re_o = rec["config"]["re_omega_target"]
        cm = rec["cm_mean"]
        ref = stokes_torque_coefficient(re_o)
        torque_rows.append(
            {
                "case": name,
                "re_omega": re_o,
                "cm": cm,
                "cm_std": rec["cm_std"],
                "cm_stokes": ref,
                "err_pct": 100.0 * (cm - ref) / ref,
                "cm_pressure_only": rec["cm_pressure_only_mean"],
                "diameter": rec["config"]["diameter"],
                "box_diameters": rec["config"]["box_diameters"],
                "steps": rec["config"]["steps"],
                "pass": abs(cm - ref) / ref * 100.0 <= CM_TOL,
            }
        )

    # ---------------- mass conservation ------------------------------------
    f64 = [r for r in drag.values() if r["config"]["dtype"] == "float64"]
    mass = {
        "float64_runs": [
            {
                "case": r["name"],
                "max_step_residual": max(
                    (h.get("mass_step_residual", 0.0) for h in r["history"]), default=0.0
                ),
                "max_cumulative_residual": r["max_mass_residual"],
                "max_closure_residual": r["max_closure_residual"],
                "steps": r["config"]["steps"],
            }
            for r in f64
        ],
        "float32_max_cumulative": max(
            (r["max_mass_residual"] for r in drag.values() if r["config"]["dtype"] == "float32"),
            default=float("nan"),
        ),
    }
    mass_pass = bool(
        f64
        and all(m["max_step_residual"] <= MASS_TOL for m in mass["float64_runs"])
        and all(m["max_closure_residual"] <= MASS_TOL for m in mass["float64_runs"])
    )

    # ---------------- primary sweep (best-resolved case per target Re) ------
    primary: list[dict] = []
    for target in sorted({r["reynolds_target"] for r in drag_rows}):
        group = [r for r in drag_rows if r["reynolds_target"] == target]
        primary.append(max(group, key=lambda r: r["diameter"]))
    missing_targets = sorted(
        {v["config"]["reynolds_target"] for v in (cases.values()) if v.get("status")}
    )

    # ---------------- criteria ---------------------------------------------
    tested = [r for r in primary if r["stable"]]
    criteria = {
        "a_cd_error": {
            "rule": "C_D outside the Schiller-Naumann / Clift-Gauvin band <= 10 % (Re<=100), <= 15 % (Re<=1000)",
            "pass": bool(tested) and all(r["pass"] for r in tested) and not missing_targets,
            "selection_rule": "one primary case per target Re = the best-resolved (largest D) case; every primary must be inside the band",
            "diverged_targets": missing_targets,
            "per_case": [
                {
                    "re": round(r["reynolds"], 2),
                    "cd": round(r["cd"], 4),
                    "err_outside_band_pct": round(r["err_outside_band_pct"], 2),
                    "tol": r["tolerance_pct"],
                    "pass": r["pass"],
                }
                for r in primary
            ],
        },
        "b_grid_order": {
            "rule": "observed order at Re=100 from >= 3 grids >= 0.8",
            "pass": bool(order and order["order"] >= ORDER_MIN),
            "observed_order": None if not order else round(order["order"], 3),
            "extrapolated_cd": None if not order else round(order["extrapolated"], 4),
            "fit_rms_pct": None if not order else round(order.get("fit_rms_pct", float("nan")), 3),
        },
        "c_torque": {
            "rule": "C_M within 10 % of 8/Re_omega for Re_omega <= 2",
            "pass": bool(torque_rows)
            and all(r["pass"] for r in torque_rows if r["re_omega"] <= 2.0),
            "per_case": [
                {"re_omega": r["re_omega"], "cm": round(r["cm"], 4), "err_pct": round(r["err_pct"], 2)}
                for r in torque_rows
            ],
        },
        "d_mass_conservation": {
            "rule": "per-step lattice-mass residual <= 1e-12 (float64) and force-budget closure <= 1e-12",
            "pass": mass_pass,
            "detail": mass,
        },
        "e_negative_controls": {
            "rule": "two negative controls: (i) rigid translation is a null vector and (ii) a uniform free stream is Galilean invariant, both <= 1e-12; the legacy estimators must FAIL the same operators (legacy marker sum biased by M/A >> 1, pressure-only torque = 0 against a finite analytic torque)",
            "pass": bool(controls)
            and controls.get("nc1_uniform_translation_force_rel", 1.0) <= 1e-12
            and controls.get("nc2_galilean_force_rel_change", 1.0) <= 1e-12
            and abs(
                controls.get("legacy_g15_uniform_force_rel", 0.0)
                - controls.get("legacy_marker_density_ratio", 0.0)
            )
            < 1e-6
            and controls.get("legacy_g15_uniform_force_rel", 0.0) > 1.0
            and controls.get("g12_pressure_only_torque_rel", 1.0) <= 1e-12,
            "detail": {k: v for k, v in controls.items() if k != "band_re100"},
        },
    }
    accepted = all(c["pass"] for c in criteria.values())

    record = {
        "node": "p387w2-n04",
        "repo": "TensorLBM",
        "module": "src/tensorlbm/ibm_sphere_surface_force.py",
        "driver": "scripts/p387w2_n04_sphere_force/run_validation.py",
        "generated_from": sorted(cases),
        "conventions": {
            "lattice": "D3Q19, dt = dx = rho0 = 1, u_lattice = 0.10 (Ma = 0.17)",
            "drag_normalisation": "C_D = |F_x| / (0.5 rho U^2 pi a^2) (frontal area)",
            "torque_normalisation": "C_M = |T| / (0.5 rho U_tip^2 4 pi a^2 a) (sphere surface area) -> Stokes 8/Re_omega",
            "reference": "Schiller-Naumann (1933) and Clift-Gauvin (1971) mutually adjudicated; band = [min, max]",
        },
        "criteria": criteria,
        "accepted": accepted,
        "drag": drag_rows,
        "drag_primary": [r["case"] for r in primary],
        "grid_order": order,
        "torque": torque_rows,
        "controls": controls,
        "mass": mass,
        "failed_runs": failed,
    }
    (OUT / "acceptance.json").write_text(json.dumps(record, indent=1, default=float))
    print(json.dumps({"accepted": accepted, "criteria": {k: v["pass"] for k, v in criteria.items()}}, indent=1))

    # ---------------- figures ---------------------------------------------
    (OUT / "tables.md").write_text(_render_tables(record), encoding="utf-8")

    if order:
        _plot_normalised_convergence(conv, order)
    _plot_cd_re(drag_rows)
    if order:
        _plot_convergence(conv, order)
    if torque_rows:
        _plot_torque(torque_rows)
    _plot_history(drag, controls)
    return 0


def _plot_cd_re(rows) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    res = sorted(r["reynolds"] for r in rows)
    grid = [10 ** (math.log10(res[0]) + i * (math.log10(res[-1]) - math.log10(res[0])) / 200) for i in range(201)]
    ax.fill_between(
        grid,
        [cd_schiller_naumann(r) for r in grid],
        [cd_clift_gauvin(r) for r in grid],
        color="0.8",
        label="Schiller-Naumann / Clift-Gauvin band",
    )
    ax.plot(grid, [cd_schiller_naumann(r) for r in grid], "k--", lw=1)
    ax.errorbar(
        [r["reynolds"] for r in rows],
        [r["cd"] for r in rows],
        yerr=[r["cd_std"] for r in rows],
        fmt="o",
        color="C3",
        capsize=3,
        label="IBM surface quadrature (this node)",
    )
    ax.plot(
        [r["reynolds"] for r in rows],
        [r["cd_drive"] for r in rows],
        "s",
        mfc="none",
        color="C0",
        label="momentum-balance drive force",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Re = U D / nu")
    ax.set_ylabel("$C_D$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "cd_vs_re.png", dpi=160)
    plt.close(fig)


def _plot_convergence(rows, order) -> None:
    rows = sorted(rows, key=lambda r: r["diameter"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    d = [r["diameter"] for r in rows]
    cd = [r["cd"] for r in rows]
    ax.plot([1.0 / v for v in d], cd, "o-", color="C3", label="Re = 100, box 5D")
    hs = torch.tensor([1.0 / v for v in d])
    p = order["order"]
    c0 = order["extrapolated"]
    # one-parameter fit through the finest point for display
    basis = hs**p
    c = (torch.tensor(cd) - c0) / basis
    ax.plot(hs.numpy(), (c0 + float(c[-1]) * basis).numpy(), "k--", label=f"fit p = {p:.2f}")
    ax.axhline(c0, color="0.4", ls=":", lw=1, label=f"h->0 extrapolation = {c0:.3f}")
    for r, val in zip(rows, cd):
        ax.annotate(f"D={r['diameter']:g}", (1.0 / r["diameter"], val), textcoords="offset points",
                    xytext=(4, 4), fontsize=8)
    ax.set_xlabel("1 / D  [1/lattice]")
    ax.set_ylabel("$C_D$")
    ax.set_title("Grid convergence at Re = 100")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "grid_convergence.png", dpi=160)
    plt.close(fig)


def _plot_normalised_convergence(rows, order) -> None:
    rows = sorted(rows, key=lambda r: r["diameter"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    d = [r["diameter"] for r in rows]
    ratio = [r["cd"] / r["band_mid"] for r in rows]
    ax.plot([1.0 / v for v in d], [100.0 * (v - 1.0) for v in ratio], "o-", color="C3",
            label=r"$C_D / C_{D,band} - 1$")
    p = order["order"]
    hs = torch.tensor([1.0 / v for v in d])
    c0 = order["extrapolated"]
    basis = hs**p
    c = (torch.tensor(ratio) - c0) / basis
    ax.plot(hs.numpy(), (100.0 * (c0 + float(c[-1]) * basis - 1.0)).numpy(), "k--",
            label=f"fit p = {p:.2f}, h->0 = {100.0 * (c0 - 1.0):+.1f} %")
    ax.axhline(0.0, color="0.4", ls=":", lw=1)
    for r, val in zip(rows, ratio):
        ax.annotate(f"D={r['diameter']:g}\nRe={r['reynolds']:.0f}", (1.0 / r["diameter"], 100.0 * (val - 1.0)),
                    textcoords="offset points", xytext=(4, 6), fontsize=7)
    ax.set_xlabel("1 / D  [1/lattice]")
    ax.set_ylabel("normalised drag error [%]")
    ax.set_title("Grid convergence of the normalised drag at Re ~ 100")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "grid_convergence_normalised.png", dpi=160)
    plt.close(fig)


def _plot_torque(rows) -> None:
    rows = sorted(rows, key=lambda r: r["re_omega"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    grid = [10 ** (-0.5 + i * (2.0 / 200)) for i in range(201)]
    ax.plot(grid, [stokes_torque_coefficient(r) for r in grid], "k--", label="$8/Re_\\omega$ (Stokes)")
    ax.errorbar(
        [r["re_omega"] for r in rows],
        [r["cm"] for r in rows],
        yerr=[r["cm_std"] for r in rows],
        fmt="o",
        color="C3",
        capsize=3,
        label="rotating sphere, traction quadrature",
    )
    ax.plot(
        [r["re_omega"] for r in rows],
        [r["cm_pressure_only"] for r in rows],
        "v",
        color="C0",
        label="pressure-only estimator (G12)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$Re_\\omega = U_{tip} D / \\nu$")
    ax.set_ylabel("$C_M$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "torque_vs_re_omega.png", dpi=160)
    plt.close(fig)


def _plot_history(drag: dict[str, dict], controls: dict) -> None:
    if not drag:
        return
    # pick the case closest to Re = 100 as the exhibit
    with_hist = [r for r in drag.values() if r.get("history")]
    if not with_hist:
        return
    rec = min(with_hist, key=lambda r: abs(r["reynolds_measured"] - 100.0))
    hist = rec["history"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot([h["step"] for h in hist], [h["cd"] for h in hist], lw=0.9, color="C3",
            label="instantaneous $C_D$ (marker quadrature)")
    ax.plot([h["step"] for h in hist], [h["cd_drive"] for h in hist], lw=1.2, color="C0",
            label="$C_D$ from the momentum-balance drive force")
    ax.set_xlabel("lattice step")
    ax.set_ylabel("$C_D$")
    ax.set_title(f"Re = {rec['reynolds_measured']:.1f}, D = {rec['config']['diameter']:g}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "cd_history.png", dpi=160)
    plt.close(fig)

    if controls:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        labels = ["rigid\ntranslation\n(null)", "Galilean\ninvariance", "legacy marker\nsum (M/A bias)"]
        values = [
            controls.get("nc1_uniform_translation_force_rel", 0.0),
            controls.get("nc2_galilean_force_rel_change", 0.0),
            controls.get("legacy_g15_uniform_force_rel", 0.0),
        ]
        floor = 1e-16
        ax.bar(labels, [max(v, floor) for v in values], color=["C2", "C2", "C3"])
        ax.set_yscale("log")
        ax.set_ylabel("residual / reference force")
        ax.axhline(1e-12, color="k", ls="--", lw=1, label="1e-12 acceptance line")
        ax.legend(fontsize=8)
        ax.set_title("Negative controls")
        fig.tight_layout()
        fig.savefig(FIGS / "negative_controls.png", dpi=160)
        plt.close(fig)


def _render_tables(record: dict) -> str:
    """Markdown tables generated from the records (no hand-copied numbers)."""
    lines: list[str] = []
    lines.append("### Layer 1 -- drag vs the mutually adjudicated empirical band\n")
    lines.append(
        "| Re (measured) | D | box | $C_D$ (surface quadrature) | sd | $C_D$ (drive/momentum) | "
        "Schiller-Naumann | Clift-Gauvin | err vs band [%] | tol [%] | pass | steps |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(record["drag"], key=lambda r: r["reynolds"]):
        lines.append(
            f"| {r['reynolds']:.2f} | {r['diameter']:g} | {r['lateral_diameters']:g}D | "
            f"{r['cd']:.4f} | {r['cd_std']:.3f} | {r['cd_drive']:.4f} | {r['sn']:.4f} | {r['cg']:.4f} | "
            f"{r['err_outside_band_pct']:+.1f} | {r['tolerance_pct']:g} | {'PASS' if r['pass'] else 'FAIL'} | "
            f"{r['steps']}{' (partial)' if r.get('partial_from_log') else ''} |"
        )
    order = record.get("grid_order")
    lines.append("\n### Layer 2 -- grid convergence at Re = 100\n")
    if order:
        lines.append(
            f"Observed order **p = {order['order']:.3f}**, extrapolated $C_D(h\\to0)$ = "
            f"{order['extrapolated']:.4f}, fit RMS = {order.get('fit_rms_pct', float('nan')):.2f} %, "
            f"GCI(finest) = {order.get('gci_finest', float('nan')):.2f} %\n"
        )
        lines.append("| D | $C_D$ | steps |")
        lines.append("|---|---|---|")
        for g in order["grids"]:
            lines.append(f"| {g['diameter']:g} | {g['cd']:.4f} | {g['steps']} |")
    else:
        lines.append(
            "Order **not established**: fewer than three *converged* grids at Re ~ 100 within "
            "the available wall clock. The measured normalised errors are reported below; they "
            "are not used to fit an order."
        )
    # the grid evidence that exists, converged or not
    conv_all = sorted(
        [r for r in record["drag"] if abs(r["reynolds"] - 100.0) < 25.0 and r["diameter"]],
        key=lambda r: r["diameter"],
    )
    if conv_all:
        lines.append("")
        lines.append("| D | Re (measured) | $C_D$ | band at that Re | normalised error [%] | steps |")
        lines.append("|---|---|---|---|---|---|")
        for r in conv_all:
            band = 0.5 * (cd_schiller_naumann(r["reynolds"]) + cd_clift_gauvin(r["reynolds"]))
            lines.append(
                f"| {r['diameter']:g} | {r['reynolds']:.1f} | {r['cd']:.4f} | {band:.4f} | "
                f"{100.0 * (r['cd'] / band - 1.0):+.1f} | "
                f"{r['steps']}{' (partial)' if r.get('partial_from_log') else ''} |"
            )
    lines.append("\n### Layer 3 -- rotating-sphere torque\n")
    lines.append(
        "| $Re_\\omega$ | D | box | $C_M$ (traction) | sd | $8/Re_\\omega$ (Stokes) | err [%] | "
        "$C_M$ (pressure-only, G12) | pass |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in record["torque"]:
        lines.append(
            f"| {r['re_omega']:g} | {r['diameter']:g} | {r['box_diameters']:g}D | {r['cm']:.4f} | "
            f"{r['cm_std']:.4f} | {r['cm_stokes']:.4f} | {r['err_pct']:+.1f} | "
            f"{r['cm_pressure_only']:.2e} | {'PASS' if r['pass'] else 'FAIL'} |"
        )
    lines.append("\n### Closures and controls\n")
    lines.append("| quantity | value | criterion |")
    lines.append("|---|---|---|")
    for run in record["mass"]["float64_runs"]:
        lines.append(
            f"| float64 per-step mass residual ({run['case']}, {run['steps']} steps) | "
            f"{run['max_step_residual']:.3e} | <= 1e-12 |"
        )
        lines.append(
            f"| float64 force-budget closure | {run['max_closure_residual']:.3e} | <= 1e-12 |"
        )
    c = record["controls"]
    if c:
        lines.append(f"| NC1 rigid translation (null) | {c['nc1_uniform_translation_force_rel']:.3e} | <= 1e-12 |")
        lines.append(f"| NC2 Galilean invariance | {c['nc2_galilean_force_rel_change']:.3e} | <= 1e-12 |")
        lines.append(
            f"| legacy marker sum bias | {c['legacy_g15_uniform_force_rel']:.4f} | = M/A = "
            f"{c['legacy_marker_density_ratio']:.4f} (must be >> 1) |"
        )
        lines.append(
            f"| G12 pressure-only torque | {c['g12_pressure_only_torque_rel']:.3e} | = 0 against a finite analytic torque |"
        )
        lines.append(
            f"| quadrature of a constant traction | {c['quadrature_constant_traction_rel_error']:.3e} | <= 1e-12 |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())