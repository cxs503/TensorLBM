"""Audit the geometry of the Ghia (1982) metric used by the cavity benchmark.

Zero-simulation audit (m16740): the reported RMSE depends on an implicit
index->coordinate map, and the map that the solver's boundary recipe actually
implies is neither of the two historical candidates.  This script quantifies,
for every Reynolds number in the repository's Ghia tables and a resolution
ladder, how much error a *perfect* solver would still be charged under each
map (the "metric floor"), plus the geometric facts behind it:

* the effective cavity is ``H_x = nx - 2`` by ``H_y = ny - 1.5`` (bottom/left/
  right half-way bounce-back walls, lid on the Zou/He node row) -> not square;
* ``Re = u_lid * nx / nu`` uses ``L = nx``, which matches none of the three
  lengths, so the run's Reynolds number is itself ~1.5 % off the geometric one.

Outputs (programmatic, no hand-copied numbers):

* ``audit/m16740/cavity_metric_geometry.json``
* ``figures/m16740_cavity_metric_geometry.png``

Usage: ``python scripts/audit_cavity_metric_geometry.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorlbm.lid_driven_cavity import (
    CAVITY_BOUNDARY_RECIPE,
    GHIA_RE100,
    GHIA_RE400,
    GHIA_RE1000,
    cavity_wall_planes,
    ghia_metric_floor,
)

RE_TABLES = {"100": GHIA_RE100, "400": GHIA_RE400, "1000": GHIA_RE1000}
GRIDS = (32, 48, 64, 96, 128, 192, 256)
OUT_DIR = Path("audit/m16740")
FIG_DIR = Path("figures")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    geometry = {f"n{n}": cavity_wall_planes(n) for n in GRIDS}
    floors: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for re_label, ref in RE_TABLES.items():
        floors[re_label] = {
            f"n{n}": ghia_metric_floor(ref, n) for n in GRIDS
        }

    payload = {
        "audit": "m16740",
        "boundary_recipe": CAVITY_BOUNDARY_RECIPE,
        "geometry": geometry,
        "metric_floor_percent_of_u_lid": floors,
        "units": "RMSE normalised by u_lid, reported in percent",
        "maps": {
            "legacy_w0": "walls on the first/last node (historical default)",
            "symmetric_w+0.5": "nodes at cell centres (symmetric half-cell)",
            "recipe_planes": "map implied by the BC recipe of run_lid_driven_cavity",
        },
    }
    (OUT_DIR / "cavity_metric_geometry.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = {"legacy_w0": "#1f77b4", "symmetric_w+0.5": "#d62728", "recipe_planes": "#2ca02c"}
    labels = {
        "legacy_w0": "legacy map (w=0, walls on nodes)",
        "symmetric_w+0.5": "symmetric map (w=+0.5)",
        "recipe_planes": "recipe map (this solver's BC)",
    }

    ax = axes[0]
    for re_label in RE_TABLES:
        for name in colors:
            series = [
                floors[re_label][f"n{n}"][name]["floor_u"] * 100.0 for n in GRIDS
            ]
            ax.plot(
                GRIDS,
                series,
                marker="o",
                ms=3,
                color=colors[name],
                ls="-" if re_label == "1000" else "--",
                alpha=1.0 if re_label == "1000" else 0.45,
                lw=1.6 if re_label == "1000" else 1.0,
                label=f"{labels[name]}" + (" (Re=1000)" if re_label == "1000" else ""),
            )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("nodes per side n")
    ax.set_ylabel(r"u-centreline metric floor  [$\%$ of $u_{lid}$]")
    ax.set_title("(a) metric floor of a perfect solver, u profile")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    n = 128
    names = list(colors)
    x = np.arange(len(RE_TABLES))
    width = 0.26
    for k, name in enumerate(names):
        vals = [floors[re][f"n{n}"][name]["floor_v"] * 100.0 for re in RE_TABLES]
        ax.bar(x + (k - 1) * width, vals, width, color=colors[name], label=labels[name])
    ax.set_xticks(x, [f"Re={re}" for re in RE_TABLES])
    ax.set_ylabel(r"v-centreline metric floor  [$\%$ of $u_{lid}$]")
    ax.set_title(f"(b) floor at n={n}: the symmetric map is the trap")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2]
    ns = np.array(GRIDS, dtype=float)
    aspect = np.array([geometry[f"n{int(v)}"]["aspect_ratio_h_y_over_h_x"] for v in ns])
    ax.plot(ns, (aspect - 1.0) * 100.0, marker="s", color="#9467bd", label="H_y/H_x - 1")
    ax.plot(
        ns,
        (1.0 - np.array([geometry[f"n{int(v)}"]["implied_re_ratio_x"] for v in ns])) * 100.0,
        marker="^",
        color="#ff7f0e",
        label="1 - H_x/nx  (Re length deficit)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("nodes per side n")
    ax.set_ylabel("percent")
    ax.set_title("(c) geometry of the discretised cavity")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        "m16740 - Ghia metric geometry audit: no RMSE is reproducible without its coordinate map"
    )
    fig.tight_layout()
    fig_path = FIG_DIR / "m16740_cavity_metric_geometry.png"
    fig.savefig(fig_path, dpi=150)
    print(json.dumps({"json": str(OUT_DIR / "cavity_metric_geometry.json"), "figure": str(fig_path)}))

    # console summary, programmatic
    for re_label in RE_TABLES:
        row = {k: round(floors[re_label]["n128"][k]["floor_v"] * 100.0, 3) for k in names}
        print(f"Re={re_label} n=128 floor_v [%u_lid]: {row}")
    print("geometry n=128:", {k: round(v, 5) for k, v in geometry["n128"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
