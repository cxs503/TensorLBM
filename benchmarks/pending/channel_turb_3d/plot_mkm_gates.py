#!/usr/bin/env python3
"""Figures for the P20-N01 channel-flow assessment (MKM-1999 comparison).

Every curve is read from artefacts produced by
``analyze_mkm_gates.py`` / ``laminar_wall_origin.py``; nothing is typed in by
hand.  Output PNGs go to ``figures/``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
NUMBERS = HERE / "mkm_gates_numbers.json"
MKM_CSV = HERE / "dns_ref" / "mkm180.csv"
OUT = HERE / "figures"

KAPPA, B = 0.41, 5.0


def load_numbers() -> dict:
    return json.loads(NUMBERS.read_text())


def load_mkm() -> tuple[np.ndarray, np.ndarray]:
    y_plus, u_plus = [], []
    with open(MKM_CSV, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            y_plus.append(float(row["y_plus"]))
            u_plus.append(float(row["u_plus"]))
    return np.asarray(y_plus), np.asarray(u_plus)


def figure_profiles(numbers: dict) -> Path:
    mkm_y, mkm_u = load_mkm()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    ax = axes[0]
    ax.plot(mkm_y, mkm_u, "k-", lw=2.2, label="MKM-1999 DNS (Re_tau=178.1)")
    grid = np.logspace(np.log10(1.0), np.log10(200.0), 200)
    ax.plot(grid, grid, ":", color="0.6", lw=1.2, label="u+ = y+ (viscous)")
    ax.plot(grid, np.log(grid) / KAPPA + B, "--", color="0.45", lw=1.2,
            label=r"log law $\kappa$=0.41, B=5.0")
    colours = {"L1": "#d1495b", "L2": "#edae49", "L3": "#00798c"}
    for name, profile in numbers["profiles"].items():
        y = np.asarray(profile["measured"]["y_plus"])
        u = np.asarray(profile["measured"]["u_plus"])
        ax.plot(y, u, "o-", ms=3.0, lw=1.4, color=colours.get(name, "C0"),
                label=f"{name} D3Q19, y_w measured = {profile['measured']['wall_origin']:.3f}")
    ax.set_xscale("log")
    ax.set_xlim(1, 250)
    ax.set_ylim(0, 25)
    ax.set_xlabel("y+")
    ax.set_ylabel("u+")
    ax.set_title("(a) mean velocity profile vs MKM-1999")
    ax.legend(fontsize=7.4, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(mkm_y, mkm_u, "k-", lw=2.2, label="MKM-1999 DNS")
    for name, profile in numbers["profiles"].items():
        for label, style, marker in (("assumed_0p5", "--", None), ("measured", "-", "o")):
            y = np.asarray(profile[label]["y_plus"])
            u = np.asarray(profile[label]["u_plus"])
            ax.plot(y, u, style, marker=marker, ms=3.0, lw=1.3,
                    color=colours.get(name, "C0"), alpha=0.9 if label == "measured" else 0.55,
                    label=f"{name} y_w={profile[label]['wall_origin']:.3f}")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 20)
    ax.axvspan(30, 100, color="0.9", zorder=0)
    ax.set_xlabel("y+")
    ax.set_ylabel("u+")
    ax.set_title("(b) near-wall zoom: assumed 0.5 (dashed) vs measured y_w")
    ax.legend(fontsize=6.6, loc="lower right", ncol=1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUT / "profiles_uplus_vs_mkm.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_wall_origin(numbers: dict) -> Path:
    table = numbers["calibration"]["table"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    styles = {"natural_kbc": ("o", "#00798c"), "cumulant": ("s", "#d1495b")}
    for model, (marker, colour) in styles.items():
        rows = [row for row in table if row["collision_model"] == model]
        if not rows:
            continue
        taus = np.array([row["tau"] for row in rows])
        order = np.argsort(taus)
        values = np.array([row["y_w_wall_stress_local"] for row in rows])[order]
        ax.plot(taus[order], values, marker + "-", color=colour, ms=6, lw=1.4,
                label=f"{model} (measured)")
    ax.axhline(0.5, color="k", ls="--", lw=1.2, label="assumed 0.5 (legacy pipeline)")
    ax.axhline(0.1, color="0.45", ls=":", lw=1.4,
               label="0.1 previously reported (not reproducible)")
    ax.set_xlabel("relaxation time tau")
    ax.set_ylabel("effective wall origin y_w [cells]")
    ax.set_title("post-streaming bounce-back wall position, laminar Poiseuille inversion")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 0.8)
    fig.tight_layout()
    path = OUT / "wall_origin_calibration.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_three_way(numbers: dict) -> Path:
    levels = list(numbers["profiles"])
    routes = ("pressure_gradient", "wall_shear", "log_law")
    colours = {"pressure_gradient": "#00798c", "wall_shear": "#edae49", "log_law": "#d1495b"}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    width = 0.24
    x = np.arange(len(levels))
    for index, route in enumerate(routes):
        values = [
            numbers["levels"][name]["three_way"]["u_tau"][route] for name in levels
        ]
        ax.bar(x + (index - 1) * width, values, width, color=colours[route],
               label=route.replace("_", " "))
    target = [numbers["levels"][name]["target_u_tau"] for name in levels]
    for position, value in zip(x, target, strict=True):
        ax.hlines(value, position - 0.45, position + 0.45, color="k", ls="--", lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{name}\n{numbers['levels'][name]['grid']}\n"
            f"max dev {numbers['levels'][name]['three_way']['max_pairwise_deviation_pct']:.1f}%"
            for name in levels
        ],
        fontsize=8,
    )
    ax.set_ylabel("u_tau [lattice units]")
    ax.set_title("three-way friction velocity (dashed: momentum-balance target)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    path = OUT / "u_tau_three_way.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_reynolds_stress(numbers: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    colours = {"L1": "#d1495b", "L2": "#edae49", "L3": "#00798c"}
    ax = axes[0]
    for name in numbers["reynolds_stress_plus"]:
        data = numbers["reynolds_stress_plus"][name]
        y = np.asarray(numbers["profiles"][name]["measured"]["y_plus"])
        ax.plot(y, np.asarray(data["uv"]), "o-", ms=2.6, lw=1.2,
                color=colours.get(name, "C0"), label=f"{name} <u'v'>+")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("y+")
    ax.set_ylabel(r"$\langle u'v'\rangle^+$")
    ax.set_title("(a) Reynolds shear stress (all levels)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for name in numbers["reynolds_stress_plus"]:
        data = numbers["reynolds_stress_plus"][name]
        y = np.asarray(numbers["profiles"][name]["measured"]["y_plus"])
        ax.plot(y, np.asarray(data["uu"]), "-", lw=1.3, color=colours.get(name, "C0"),
                label=f"{name} <u'u'>+")
        ax.plot(y, np.asarray(data["vv"]), ":", lw=1.1, color=colours.get(name, "C0"))
        ax.plot(y, np.asarray(data["ww"]), "--", lw=1.1, color=colours.get(name, "C0"))
    ax.set_xlabel("y+")
    ax.set_ylabel("normal stress+")
    ax.set_title("(b) normal stresses (solid uu, dotted vv, dashed ww)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUT / "reynolds_stress_profiles.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_error_trend(numbers: dict) -> Path:
    levels = list(numbers["profiles"])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    ax = axes[0]
    cells = [np.prod(numbers["levels"][name]["grid"]) for name in levels]
    measured = [
        100 * numbers["mkm_metrics"][name]["measured"]["u_plus_l2_relative"] for name in levels
    ]
    assumed = [
        100 * numbers["mkm_metrics"][name]["assumed_0p5"]["u_plus_l2_relative"] for name in levels
    ]
    ax.plot(cells, assumed, "s--", color="0.5", label="y_w assumed 0.5")
    ax.plot(cells, measured, "o-", color="#00798c", label="y_w measured")
    ax.axhline(10.0, color="#d1495b", ls="--", lw=1.1, label="C1 gate 10%")
    for x, y, name in zip(cells, measured, levels, strict=True):
        ax.annotate(f"{name}", (x, y), textcoords="offset points", xytext=(4, 6), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("grid cells")
    ax.set_ylabel("relative L2 error of u+ vs MKM [%]")
    ax.set_title("(a) profile error vs resolution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    turns = [numbers["levels"][name]["statistics_eddy_turnovers"] for name in levels]
    three = [
        numbers["levels"][name]["three_way"]["max_pairwise_deviation_pct"] for name in levels
    ]
    shear = [
        100
        * (
            numbers["levels"][name]["three_way"]["u_tau"]["wall_shear"]
            - numbers["levels"][name]["target_u_tau"]
        )
        / numbers["levels"][name]["target_u_tau"]
        for name in levels
    ]
    ax.plot(turns, three, "o-", color="#d1495b", label="three-way spread")
    ax.plot(turns, np.abs(shear), "s--", color="#00798c", label="|wall-shear error|")
    ax.axhline(3.0, color="k", ls="--", lw=1.1, label="C4 gate 3%")
    for x, y, name in zip(turns, three, levels, strict=True):
        ax.annotate(f"{name}", (x, y), textcoords="offset points", xytext=(4, 6), fontsize=7.5)
    ax.set_xlabel("statistics window [eddy turnovers]")
    ax.set_ylabel("error [%]")
    ax.set_title("(b) closure error vs statistical convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUT / "error_trends.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    numbers = load_numbers()
    paths = [
        figure_profiles(numbers),
        figure_wall_origin(numbers),
        figure_three_way(numbers),
        figure_reynolds_stress(numbers),
        figure_error_trend(numbers),
    ]
    manifest = {
        "figures": [
            {
                "path": str(path.relative_to(HERE)),
                "caption": caption,
                "sha256_size": path.stat().st_size,
            }
            for path, caption in zip(
                paths,
                [
                    "u+ vs y+ against MKM-1999 DNS for the three grid levels, with the "
                    "measured wall origin, and a near-wall zoom comparing the previously "
                    "assumed y_w=0.5 with the measured value.",
                    "Effective wall origin of the production bounce-back wall measured by "
                    "laminar Poiseuille inversion, versus relaxation time and collision "
                    "model, against the assumed 0.5 and the previously reported 0.1.",
                    "Three-way friction velocity (momentum balance / momentum exchange / "
                    "log-law fit at the measured origin) per grid level.",
                    "Reynolds shear stress <u'v'>+ and normal stresses in wall units for the "
                    "three grid levels.",
                    "Profile error versus resolution and closure error versus statistical "
                    "convergence, with the C1 (10%) and C4 (3%) gates.",
                ],
                strict=True,
            )
        ]
    }
    (OUT / "figures.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for path in paths:
        print(path.relative_to(HERE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())