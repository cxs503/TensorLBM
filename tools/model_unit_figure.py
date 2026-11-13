#!/usr/bin/env python
"""Figure for the model-directory-as-CI-unit benchmark artefacts.

Reads the artefacts written by ``tensorlbm.models.ci_unit all`` and
``tools/model_unit_sensitivity.py`` (JSON/CSV only: every number in the figure
is a value that was measured by the CI run, nothing is typed in by hand) and
writes a three-panel figure:

1. measured Poiseuille profile vs the analytic parabola (BGK and TRT units);
2. Poiseuille L2 error vs channel height (grid refinement);
3. shear-wave viscosity error vs relaxation time (parameter sensitivity).

    python tools/model_unit_figure.py --artifacts artifacts/model_units
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _profile(path: Path) -> tuple[list[float], list[float], list[float]]:
    rows = path.read_text().strip().splitlines()[1:]
    y, lbm, exact = [], [], []
    for row in rows:
        a, b, c = row.split(",")
        y.append(float(a))
        lbm.append(float(b))
        exact.append(float(c))
    return y, lbm, exact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts/model_units")
    args = parser.parse_args(argv)
    root = Path(args.artifacts)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    ax = axes[0]
    for unit_id, style, label in (
        ("d2q9_bgk_channel_poiseuille", "-", "D2Q9 BGK"),
        ("d2q9_trt_channel_poiseuille", "--", "D2Q9 TRT (Λ=3/16)"),
    ):
        y, lbm, exact = _profile(root / "series" / f"{unit_id}_profile.csv")
        ax.plot(lbm, y, style, lw=1.6, label=f"{label} (LBM)")
        if unit_id.startswith("d2q9_bgk"):
            ax.plot(exact, y, ":", color="k", lw=2.2, label="analytic parabola")
    ax.set_xlabel(r"$u_x$ (lattice units)")
    ax.set_ylabel(r"$y$ (lattice units)")
    ax.set_title("Poiseuille profile, $n_y=32$, $\\tau=0.65$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sens = json.loads((root / "sensitivity.json").read_text())
    for unit_id, marker in (
        ("d2q9_bgk_channel_poiseuille", "o"),
        ("d2q9_trt_channel_poiseuille", "s"),
    ):
        rows = [r for r in sens["channel_grid_refinement"] if r["unit_id"] == unit_id]
        ax.loglog(
            [r["variant"]["ny"] for r in rows],
            [r["metrics"]["profile_l2_rel_err"] for r in rows],
            marker + "-",
            label=unit_id.replace("d2q9_", "").replace("_channel_poiseuille", ""),
        )
    ax.axhline(0.15, color="r", ls="--", lw=1.0, label="owner 15% bar")
    ax.axhline(1e-2, color="orange", ls=":", lw=1.2, label="1% unit gate")
    ax.set_xlabel("channel height $n_y$")
    ax.set_ylabel("profile $L_2$ relative error")
    ax.set_title("Grid refinement (BGK vs TRT)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    for unit_id, marker in (
        ("d2q9_bgk_shear_wave", "o"),
        ("d2q9_mrt_shear_wave", "s"),
    ):
        rows = [r for r in sens["shear_tau_sweep"] if r["unit_id"] == unit_id]
        ax.semilogy(
            [r["variant"]["tau"] for r in rows],
            [r["metrics"]["nu_rel_err"] for r in rows],
            marker + "-",
            label=unit_id.replace("d2q9_", "").replace("_shear_wave", ""),
        )
    ax.axhline(1e-2, color="orange", ls=":", lw=1.2, label="1% unit gate")
    ax.set_xlabel(r"relaxation time $\tau$")
    ax.set_ylabel(r"$|\nu_{meas}-\nu_{ref}|/\nu_{ref}$")
    ax.set_title("Shear-wave viscosity vs $\\tau$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model_units_ci.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
