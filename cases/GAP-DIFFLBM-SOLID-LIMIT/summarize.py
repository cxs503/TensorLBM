"""Aggregate the GAP-DIFFLBM-SOLID-LIMIT campaign into the verdict + figures.

Reads ``artifacts/GAP-DIFFLBM-SOLID-LIMIT/runs/*.json`` (one file per
arm/resolution/mask-scale combo), applies the two ledger criteria through
:func:`cases.GAP-DIFFLBM-SOLID-LIMIT.run_case.evaluate`, and writes

``results.json``   every record + the criteria table + the verdict
``table.csv``      the ref-vs-got table (one row per criterion comparison)
``figures/*.png``  drag vs mask scale, drag vs resolution, solid
                   representation (volume fraction and reconstructed
                   interface) at two mask scales

Usage::

    PYTHONPATH=src python cases/GAP-DIFFLBM-SOLID-LIMIT/summarize.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from run_case import ARMS, build_volume_fraction, evaluate  # noqa: E402
from tensorlbm.volume_solid import signed_distance_from_fraction  # noqa: E402

ROOT = Path("artifacts/GAP-DIFFLBM-SOLID-LIMIT")


def load_runs(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(run_dir.glob("*.json")):
        rec = json.loads(path.read_text())
        if "series" in rec:
            rec.pop("series")
        records.append(rec)
    if not records:
        raise SystemExit(f"no run records under {run_dir}")
    return records


def figure_mask_scales(fig_dir: Path, D: int = 32) -> list[dict[str, Any]]:
    """Volume fraction and reconstructed interface at two mask scales."""
    figs: list[dict[str, Any]] = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2))
    for col, scale in enumerate((1.0, 0.5)):
        m, info = build_volume_fraction(D, scale, 1.0)
        phi = signed_distance_from_fraction(m)
        ax = axes[0, col]
        im = ax.imshow(m.numpy(), origin="lower", cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"volume fraction m, mask scale {scale:g} (sigma={info['sigma_cells']:g} cells)")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax = axes[1, col]
        im = ax.imshow(phi.numpy(), origin="lower", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        ax.contour(phi.numpy(), levels=[0.0], colors="k", linewidths=0.8)
        ax.set_title("reconstructed interface distance phi = (m-1/2)/|grad m|")
        plt.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    path = fig_dir / "solid_representation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append(
        {
            "name": "solid_representation",
            "caption": (
                "Volume fraction m and the reconstructed interface distance phi of the 2D-1 "
                "cylinder (D=" + str(D) + " cells) at mask scale 1.0 (solid limit) and 0.5 "
                "(volume preserving blur, sigma = 0.5 cell). The black contour is the "
                "reconstructed interface phi = 0; it is the same surface at both mask scales, "
                "which is what the geometry closure acts on and the amplitude closure ignores."
            ),
            "png": path,
        }
    )
    return figs


def figure_scale_sweep(records: Sequence[dict[str, Any]], verdict: dict[str, Any], fig_dir: Path):
    figs = []
    table = verdict["table"]
    resolutions = sorted({r["D_cells"] for r in records})
    fig, axes = plt.subplots(1, len(resolutions), figsize=(4.6 * len(resolutions), 3.8), squeeze=False)
    for ax, D in zip(axes[0], resolutions):
        for arm, style in (("amplitude", "o--"), ("interface", "s-")):
            pts = sorted(
                [(r["scale"], r["cd_cv"]) for r in records if r["arm"] == arm and r["D_cells"] == D]
            )
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], style, label=arm)
        hw = [r["cd_cv"] for r in records if r["arm"] == "hardwall" and r["D_cells"] == D]
        if hw:
            ax.axhline(hw[0], color="k", ls=":", label="hard-wall bounce-back")
        ref = [r["ref_cd"] for r in records if r["D_cells"] == D][0]
        ax.axhline(ref, color="tab:red", lw=1.0, ls="-.", label="Schäfer-Turek 2D-1 ref")
        ax.set_xlabel("mask scale s (1 = solid limit)")
        ax.set_ylabel("Cd (control-volume momentum balance)")
        ax.set_title(f"D = {D} cells, Re = 20")
        ax.legend(fontsize=7)
        ax.set_xticks([0.5, 0.75, 1.0])
    fig.tight_layout()
    path = fig_dir / "cd_vs_mask_scale.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append(
        {
            "name": "cd_vs_mask_scale",
            "caption": (
                "Drag coefficient against mask scale for the two volumetric closures, with the "
                "hard-wall bounce-back arm and the Schäfer-Turek 2D-1 reference as horizontal "
                "lines. The amplitude closure drifts as the volume representation is smeared; "
                "the geometry closure stays on the hard-wall value."
            ),
            "png": path,
        }
    )

    if table:
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        labels = [f"{r['arm']}\nD={r['D_cells']}" for r in table]
        xs = range(len(table))
        ax.bar([x - 0.2 for x in xs], [r["delta_scale_pct"] for r in table], width=0.4, label="D1: |s=0.5 - s=1.0|")
        ax.bar([x + 0.2 for x in xs], [r["delta_hardwall_pct"] for r in table], width=0.4, label="D2: |arm - hard wall|")
        ax.axhline(5.0, color="tab:green", ls="--", lw=1.0)
        ax.axhline(10.0, color="tab:orange", ls="--", lw=1.0)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("relative deviation [%]")
        ax.set_title("ledger criteria (dashed: 5 % and 10 % limits)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = fig_dir / "criteria.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figs.append(
            {
                "name": "criteria",
                "caption": "The two ledger criteria per arm and resolution; dashed lines are the 5 % (D1) and 10 % (D2) limits.",
                "png": path,
            }
        )
    return figs


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--D", type=int, default=32, help="resolution for the representation figure")
    args = ap.parse_args(argv)
    run_dir = args.root / "runs"
    fig_dir = args.root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    records = load_runs(run_dir)
    verdict = evaluate(records)

    figs = figure_mask_scales(fig_dir, args.D)
    figs += figure_scale_sweep(records, verdict, fig_dir)

    (args.root / "results.json").write_text(
        json.dumps({"records": records, "verdict": verdict}, indent=2, default=str)
    )
    with (args.root / "table.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(verdict["table"][0].keys()) if verdict["table"] else [])
        if verdict["table"]:
            wr.writeheader()
            wr.writerows(verdict["table"])
    print(json.dumps(verdict, indent=2, default=str))
    print("figures:", [str(f["png"]) for f in figs])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())