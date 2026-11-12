"""m16339 audit: is the Ghia (1982) cavity metric convention-invariant?

Produces (all numbers programmatic, nothing hand copied):

* ``ghia_metric_audit.json`` -- convention sensitivity of the reported RMSE for
  Re = 100/400/1000 at 64..384 nodes, plus the perfect-solver floor;
* ``ghia_convention_bias.png`` -- (a) bias vs grid, (b) the node-map shift on
  the Re=1000 near-lid profile, (c) sensitivity bars at benchmark grids.

Run:  python audit/m16339/ghia_metric_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.lid_driven_cavity import (  # noqa: E402
    GHIA_RE100,
    GHIA_RE400,
    GHIA_RE1000,
    centreline_coordinates,
    ghia_coordinate_sensitivity,
)

OUT = Path(__file__).resolve().parent
TABLES = {"Re100": GHIA_RE100, "Re400": GHIA_RE400, "Re1000": GHIA_RE1000}
GRIDS = (64, 96, 128, 192, 256, 384)
OFFSETS = (0.0, 0.5, -0.5)


def main() -> int:
    payload: dict[str, object] = {
        "case": "GHIA_METRIC_CONVENTION_AUDIT",
        "question": "how much of a reported Ghia RMSE is coordinate-convention, not solver error",
        "convention": "y_j = (j + w) / (n - 1 + 2 w);  w=+0.5 cell-centred nodes, w=0 walls on nodes, w=-0.5 wall nodes inside",
        "perfect_solver_reference_map": "+0.5",
        "grids": list(GRIDS),
        "tables": {},
    }
    for name, ref in TABLES.items():
        rows = {}
        for n in GRIDS:
            rows[str(n)] = ghia_coordinate_sensitivity(ref, n, OFFSETS)
        payload["tables"][name] = rows
    (OUT / "ghia_metric_audit.json").write_text(json.dumps(payload, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    colours = {"Re100": "tab:blue", "Re400": "tab:orange", "Re1000": "tab:red"}

    ax = axes[0]
    for name in TABLES:
        b = [payload["tables"][name][str(n)]["+0.0"]["bias_u"] * 100 for n in GRIDS]
        ax.loglog(GRIDS, b, "o-", color=colours[name], label=name)
        b2 = [payload["tables"][name][str(n)]["+0.0"]["bias_v"] * 100 for n in GRIDS]
        ax.loglog(GRIDS, b2, "s--", color=colours[name], alpha=0.5)
    ax.set_xlabel("nodes per side n")
    ax.set_ylabel("perfect-solver RMSE floor [% of u_lid]")
    ax.set_title("(a) legacy map bias vs resolution\n(solid u, dashed v)")
    ax.axvspan(192, 256, color="0.85", zorder=0)
    ax.text(196, 0.02, "benchmark grids", fontsize=7, color="0.35")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ref = GHIA_RE1000
    y = np.asarray(ref["y"])[::-1]
    u = np.asarray(ref["u"])[::-1]
    from scipy.interpolate import PchipInterpolator

    f = PchipInterpolator(y, u)
    yy = np.linspace(0.0, 1.0, 400)
    ax.plot(f(yy), yy, "k-", lw=1.2, label="Ghia Re=1000 master curve")
    n = 128
    ax.plot(f(np.asarray(centreline_coordinates(n, 0.5))), centreline_coordinates(n, 0.5),
            "o", ms=3, color="tab:green", label="node truth (w=+0.5)")
    ax.plot(f(np.asarray(centreline_coordinates(n, 0.0))), centreline_coordinates(n, 0.0),
            "x", ms=4, color="tab:purple", label="legacy assumed position (w=0)")
    ax.set_xlabel("u / u_lid")
    ax.set_ylabel("y / H")
    ax.set_title("(b) 128²: where the legacy map samples\n(half-cell shift, largest near the lid)")
    ax.set_ylim(0.85, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2]
    width = 0.26
    xs = np.arange(len(GRIDS))
    for k, w in enumerate(OFFSETS):
        vals = [payload["tables"]["Re1000"][str(n)][f"{w:+.1f}"]["bias_u"] * 100 for n in GRIDS]
        ax.bar(xs + (k - 1) * width, vals, width, label=f"w={w:+.1f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(n) for n in GRIDS])
    ax.set_xlabel("nodes per side n")
    ax.set_ylabel("RMSE floor [% of u_lid]")
    ax.set_title("(c) Re=1000: convention spread at each grid")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Ghia (1982) cavity metric: coordinate-convention sensitivity (m16339 audit)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "ghia_convention_bias.png", dpi=150)

    print(json.dumps({name: {n: round(payload["tables"][name][n]["+0.0"]["bias_u"] * 100, 4) for n in ("128", "192", "256")} for name in TABLES}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())