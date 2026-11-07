#!/usr/bin/env python3
"""IN45-N07 air-lubrication campaign driver.

Runs the five on-node criteria (C1-C5 of the node) and writes
``artifacts/in45_n07/campaign.json`` plus the figures used by the report.
C7 (the 3 Re x 5 alpha lattice scan) is executed on the loong64 cluster by
``re_scan_worker.py`` and merged from ``artifacts/in45_n07/re_scan.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.air_lubrication import LAMBDA_AIR_WATER  # noqa: E402
from tensorlbm.air_lubrication_frontier import (  # noqa: E402
    CF0_SHIP,
    ETA_COMPRESSOR,
    INJECTION_DEPTH,
    MADAVAN_DELTA_B_BAND,
    S_WET,
    U_SHIP,
    curvature_injection_study,
    curtain_drag_reduction,
    energy_balance,
    madavan1985_profile,
    optimize_air_flow,
    pal_formula_comparison,
    profile_deviation,
    re_scan_case,
    simulate_curtain,
)

OUT = ROOT / "artifacts" / "in45_n07"
FIG = OUT / "figures"
D_B_SWEEP = (30e-6, 50e-6, 100e-6, 200e-6, 500e-6)
Q_SWEEP = (1e-5, 3e-5, 6e-5, 1e-4, 2e-4, 4e-4, 8e-4)


def stage_c2_alpha_profile():
    """alpha(y) vs the Madavan 1985 wall-peaked family, over the bubble size."""
    rows = []
    for d_b in D_B_SWEEP:
        c = simulate_curtain(1e-4, d_b, l_over_delta=1000.0, dt_steps=12000)
        dev = profile_deviation(c.alpha_profile, c.eta)
        rows.append(
            {
                "d_b_um": d_b * 1e6,
                "alpha_wall": c.alpha_wall,
                "delta_b_over_delta": c.delta_b_over_delta,
                "capture_efficiency": c.capture_efficiency,
                "conservation_error_max": c.conservation_error_max,
                "deviation": dev,
                "eta": [float(v) for v in c.eta],
                "alpha_profile": [float(v) for v in c.alpha_profile],
                "alive": c.mass > 0.0,
            }
        )
    best = min(rows, key=lambda r: r["deviation"]["l2_rel"])
    return {
        "rows": rows,
        "best_d_b_um": best["d_b_um"],
        "band": list(MADAVAN_DELTA_B_BAND),
        "criterion": "wall-peaked, delta_b/delta in band, shape L2 < 30%",
    }


def stage_c1_drag_reduction(profile_row):
    """PAL empirical formula vs exact stratified law, and the lattice cross-check."""
    alpha = np.array(profile_row["alpha_profile"])
    cmp = pal_formula_comparison(alpha)
    lattice = [re_scan_case(500.0, a, steps=700) for a in (0.05, 0.10, 0.20)]
    worst = max(l["dr_rel_err"] for l in lattice)
    return {
        "pal_vs_exact": cmp,
        "lattice": lattice,
        "lattice_worst_rel_err": worst,
        "criterion": "|Delta C_f - PAL formula| / Delta C_f < 20%",
    }


def stage_c3_conservation(profile_row):
    return {
        "residual_max": profile_row["conservation_error_max"],
        "criterion": "< 1e-8",
    }


def stage_c4_optimum():
    opt = optimize_air_flow(
        Q_SWEEP,
        d_b=1e-4,
        l_over_delta=1000.0,
        dt_steps=8000,
    )
    # sensitivity of the optimum to the compressor depth / efficiency
    sens = []
    for depth, eta in ((6.0, 0.6), (6.0, 0.75), (4.0, 0.75), (3.0, 0.75)):
        rows = []
        for r in opt["table"]:
            e = energy_balance(r["q_air"], r["dr"], depth=depth, efficiency=eta)
            rows.append({"q_air": r["q_air"], "net_power": e["net_power"]})
        best = max(rows, key=lambda r: r["net_power"])
        sens.append(
            {
                "depth_m": depth,
                "efficiency": eta,
                "q_opt": best["q_air"],
                "net_power_opt": best["net_power"],
                "net_positive": bool(best["net_power"] > 0.0),
            }
        )
    return {
        "table": opt["table"],
        "q_air_opt": opt["q_air_opt"],
        "net_power_opt": opt["net_power_opt"],
        "net_positive": opt["net_positive"],
        "d_b_um": 100.0,
        "reference": {
            "depth_m": INJECTION_DEPTH,
            "efficiency": ETA_COMPRESSOR,
            "cf0": CF0_SHIP,
            "wetted_area_m2": S_WET,
            "u_ship_ms": U_SHIP,
        },
        "sensitivity": sens,
        "criterion": "net power at Q_opt > 0",
    }


def stage_c5_curvature():
    out = curvature_injection_study(
        q_air=1e-4,
        d_b=1e-4,
        radii=(float("inf"), 20.0, 10.0, 5.0),
        injection_positions=(0.0, 50.0, 100.0, 150.0),
        inj_length=20.0,
        l_over_delta=200.0,
        dt_steps=4000,
    )
    return {
        "grid": out["grid"],
        "per_radius": out["per_radius"],
        "per_position": out["per_position"],
        "criterion": "capture monotone in curvature and in injection position",
    }


# ------------------------------------------------------------------ figures
def make_figures(c2, c1, c4, c5, c7):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 1 -- void fraction vs Madavan family
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    eta = np.array(c2["rows"][0]["eta"])
    ref = madavan1985_profile(eta, 1.0, 0.5)["profile"]
    ax.plot(eta, ref, "k--", lw=1.6, label="Madavan 1985 family (delta_b=0.5 delta)")
    lo = madavan1985_profile(eta, 1.0, MADAVAN_DELTA_B_BAND[0])["profile"]
    hi = madavan1985_profile(eta, 1.0, MADAVAN_DELTA_B_BAND[1])["profile"]
    ax.fill_between(eta, lo, hi, color="0.85", label="delta_b/delta band 0.4-0.6")
    for r in c2["rows"]:
        a = np.array(r["alpha_profile"])
        ax.plot(eta, a / max(r["alpha_wall"], 1e-30), lw=1.2, label=f"$d_b$={r['d_b_um']:.0f} um")
    ax.set_xlim(0, 1.5)
    ax.set_xlabel("y / delta")
    ax.set_ylabel(r"$\alpha(y)/\alpha_{wall}$")
    ax.set_title("Void-fraction shape vs Madavan 1985")
    ax.legend(fontsize=7)
    fig.tight_layout()
    paths["alpha_profile"] = FIG / "alpha_profile.png"
    fig.savefig(paths["alpha_profile"], dpi=110)
    plt.close(fig)

    # 2 -- air-flow optimum
    tab = c4["table"]
    q = np.array([r["q_air"] for r in tab])
    net = np.array([r["net_power"] for r in tab]) / 1e3
    dr = np.array([r["dr"] for r in tab])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.semilogx(q, net, "o-", color="C0", label="net power [kW]")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(c4["q_air_opt"], color="C3", ls=":", label=r"$Q_{air}$ optimum")
    ax.set_xlabel(r"$q_{air}$ [m/s, superficial]")
    ax.set_ylabel("net power [kW]")
    ax2 = ax.twinx()
    ax2.semilogx(q, dr, "s--", color="C1", label=r"$\Delta C_f/C_{f0}$")
    ax2.set_ylabel(r"$\Delta C_f/C_{f0}$")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, fontsize=7)
    ax.set_title("Air-flow optimum: saved power vs compressor power")
    fig.tight_layout()
    paths["air_optimum"] = FIG / "air_optimum.png"
    fig.savefig(paths["air_optimum"], dpi=110)
    plt.close(fig)

    # 3 -- capture vs injection position
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for r, rec in c5["per_radius"].items():
        xs = [
            g["x_inj_over_delta"] for g in c5["grid"] if g["radius_m"] == float(r)
        ]
        label = "flat" if r == float("inf") else f"R={r:.0f} m"
        ax.plot(xs, rec["captures"], "o-", label=label)
    ax.set_xlabel(r"injection position $x_{inj}/\delta$")
    ax.set_ylabel("capture efficiency (inner half of the layer)")
    ax.set_title("Hull curvature and injection position")
    ax.legend(fontsize=7, title="radius of curvature [m]")
    fig.tight_layout()
    paths["curvature_capture"] = FIG / "curvature_capture.png"
    fig.savefig(paths["curvature_capture"], dpi=110)
    plt.close(fig)

    # 4 -- Re scan parity
    if c7 and c7.get("cases"):
        fig, ax = plt.subplots(figsize=(5.0, 4.4))
        for re_val in sorted({c["re"] for c in c7["cases"]}):
            pts = [c for c in c7["cases"] if c["re"] == re_val]
            ax.plot(
                [p["dr_ref"] for p in pts],
                [p["dr_num"] for p in pts],
                "o-",
                label=f"Re={re_val:.0f}",
            )
        lim = max(
            max(c["dr_ref"] for c in c7["cases"]),
            max(c["dr_num"] for c in c7["cases"]),
        ) * 1.15
        ax.plot([0, lim], [0, lim], "k--", lw=1.0, label="1:1")
        ax.plot([0, lim], [0, 0.8 * lim], ":", color="0.5", lw=1.0, label="-20%")
        ax.plot([0, lim], [0, 1.2 * lim], ":", color="0.5", lw=1.0, label="+20%")
        ax.set_xlabel(r"exact stratified $\Delta C_f/C_{f0}$")
        ax.set_ylabel(r"lattice $\Delta C_f/C_{f0}$")
        ax.set_title("Re scan: lattice vs exact (15 cases)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        paths["re_scan"] = FIG / "re_scan.png"
        fig.savefig(paths["re_scan"], dpi=110)
        plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c2 = stage_c2_alpha_profile()
    # C1 uses the microbubble case whose layer thickness sits in the
    # published band (not the millimetre-bubble film branch)
    best = min(c2["rows"], key=lambda r: abs(r["d_b_um"] - 100.0))
    c1 = stage_c1_drag_reduction(best)
    c3 = stage_c3_conservation(best)
    c4 = stage_c4_optimum()
    c5 = stage_c5_curvature()
    c7_path = OUT / "re_scan.json"
    c7 = json.loads(c7_path.read_text()) if c7_path.exists() else None

    figs = make_figures(c2, c1, c4, c5, c7)

    verdict = {
        "C1_pal_formula_lt_20pct": bool(c1["pal_vs_exact"]["rel_error"] < 0.20),
        "C1_lattice_lt_20pct": bool(c1["lattice_worst_rel_err"] < 0.20),
        "C2_shape_lt_30pct": bool(
            min(r["deviation"]["l2_rel"] for r in c2["rows"]) < 0.30
        ),
        "C2_wall_peaked": bool(all(r["deviation"]["wall_peaked"] for r in c2["rows"])),
        "C2_band_any": bool(any(r["deviation"]["band_ok"] for r in c2["rows"])),
        "C3_residual_lt_1e8": bool(c3["residual_max"] < 1e-8),
        "C4_net_positive": bool(c4["net_positive"]),
        "C5_monotone_curvature": bool(
            all(v["strictly_decreasing"] for v in c5["per_position"].values())
        ),
        "C5_monotone_injection": bool(
            all(v["strictly_increasing"] for v in c5["per_radius"].values())
        ),
        "C7_available": bool(c7 is not None),
    }
    if c7 and c7.get("cases"):
        verdict["C7_lattice_lt_20pct"] = bool(
            max(c["dr_rel_err"] for c in c7["cases"]) < 0.20
        )
    payload = {
        "node": "IN45-N07",
        "module": "src/tensorlbm/air_lubrication_frontier.py",
        "operating_point": {
            "U_ship_ms": U_SHIP,
            "delta_m": 0.1,
            "wetted_area_m2": S_WET,
            "cf0": CF0_SHIP,
            "lambda_air_water": LAMBDA_AIR_WATER,
            "depth_m": INJECTION_DEPTH,
            "compressor_efficiency": ETA_COMPRESSOR,
        },
        "C1": c1,
        "C2": c2,
        "C3": c3,
        "C4": c4,
        "C5": c5,
        "C7": c7,
        "verdict": verdict,
        "figures": figs,
    }
    (OUT / "campaign.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(verdict, indent=2))
    print("wrote", OUT / "campaign.json")


if __name__ == "__main__":
    main()