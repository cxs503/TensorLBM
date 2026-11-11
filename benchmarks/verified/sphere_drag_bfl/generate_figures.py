"""Figures + programmatic data tables for the BFL sphere-drag anchor.

Reads ``results.json`` (written by ``run.py``) and produces

* ``figures/fig1_cd_vs_re.png``      – C_D(Re): LBM vs Clift–Grace–Weber anchor
* ``figures/fig2_grid_convergence.png`` – C_D vs 1/D for Re = 20, 100
* ``figures/fig3_wake_velocity.png`` – |u| in the symmetry plane (field snapshot)
* ``docs/sphere_drag_bfl_anchor.md``  – numeric table, injected from the JSON
  (no hand-typed numbers in the documentation)

The field snapshot is produced by ``--field-only`` (a short Re=100, D=8 run that
stores rho/ux/uy/uz); the plotting pass reuses it when present.

Usage::

    PYTHONPATH=src python benchmarks/verified/sphere_drag_bfl/generate_figures.py --field-only
    PYTHONPATH=src python benchmarks/verified/sphere_drag_bfl/generate_figures.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorlbm.sphere_drag_bfl import (  # noqa: E402
    REFERENCE_CD_CGW,
    SphereBFLConfig,
    run_sphere_case,
    schiller_naumann_cd,
)

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "verified" / "sphere_drag_bfl"
FIG = BENCH / "figures"
DOCS = ROOT / "docs"
FIELD_DIR = ROOT / "artifacts" / "sphere_drag_bfl" / "fields"


# --------------------------------------------------------------------------- #
# field snapshot
# --------------------------------------------------------------------------- #

def make_field_snapshot(re: float = 100.0, diameter: int = 8, steps: int = 2500) -> Dict[str, Any]:
    cfg = SphereBFLConfig(
        re=re, diameter=diameter, u_lb=0.1, max_steps=steps,
        window=max(100, steps // 5), tol=1e9, threads=8,
        output_dir=str(FIELD_DIR), save_fields=True,
    )
    res = run_sphere_case(cfg, save_fields=True, progress=False)
    FIELD_DIR.mkdir(parents=True, exist_ok=True)
    (FIELD_DIR / "snapshot_meta.json").write_text(json.dumps(
        {k: v for k, v in res.items() if k != "cd_history"}, indent=2, default=float))
    return res


def _load_snapshot() -> Optional[Dict[str, Any]]:
    import torch
    for p in sorted(FIELD_DIR.glob("*_fields.pt")):
        return torch.load(p, map_location="cpu")
    return None


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def fig_cd_vs_re(results: Dict[str, Any], path: Path) -> None:
    rows = [r for r in results["summary"] if r.get("diameter") == 12]
    rows.sort(key=lambda r: r["re"])
    re_lbm = np.array([r["re"] for r in rows], dtype=float)
    cd_lbm = np.array([r["cd"] for r in rows], dtype=float)
    cd_ref = np.array([r["cd_ref"] for r in rows], dtype=float)

    re_cgw = np.array(sorted(REFERENCE_CD_CGW), dtype=float)
    cd_cgw = np.array([REFERENCE_CD_CGW[r] for r in re_cgw], dtype=float)
    re_sn = np.logspace(0.5, 3.1, 200)
    cd_sn = np.array([schiller_naumann_cd(r) for r in re_sn])

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.loglog(re_sn, cd_sn, "-", color="0.6", lw=1.2,
              label="Schiller–Naumann correlation")
    ax.loglog(re_cgw, cd_cgw, "ks", ms=6, mfc="none",
              label="Clift–Grace–Weber (1978) measurements")
    ax.loglog(re_lbm, cd_lbm, "o-", color="tab:red", ms=6, lw=1.4,
              label=f"LBM BFL/D3Q19 (D=12, u=0.1)")
    for r, c in zip(re_lbm, cd_lbm):
        ax.annotate(f"{r:g}", (r, c), textcoords="offset points", xytext=(4, 5),
                    fontsize=7, color="tab:red")
    ax.set_xlabel("Re = U D / ν")
    ax.set_ylabel("$C_D$")
    ax.set_title("Sphere drag coefficient: LBM (BFL bounce-back) vs literature")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_grid_convergence(results: Dict[str, Any], path: Path) -> None:
    grid = results.get("grid_convergence") or {}
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    markers = ["o", "s", "^"]
    for k, (key, entry) in enumerate(sorted(grid.items())):
        ds = np.array(entry["diameters"], dtype=float)
        cds = np.array(entry["cd"], dtype=float)
        ax.plot(ds, cds, marker=markers[k % 3], lw=1.3,
                label=f"{key} (finest change {entry['finest_change_pct']:.2f} %)")
        if entry.get("cd_ref") is not None:
            ax.axhline(entry["cd_ref"], ls=":", color="0.4", lw=0.9)
    ax.set_xlabel("lattice diameter D")
    ax.set_ylabel("$C_D$")
    ax.set_title("Grid refinement of $C_D$ (dotted: CGW anchor)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_wake(snapshot: Dict[str, Any], path: Path) -> bool:
    rho = snapshot["rho"].numpy()
    ux = snapshot["ux"].numpy()
    uy = snapshot["uy"].numpy()
    uz = snapshot["uz"].numpy()
    solid = snapshot["solid"].numpy()
    umag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    j = umag.shape[1] // 2            # symmetry plane z = const
    slice_u = umag[:, j, :].T         # (nz, nx)
    slice_s = solid[:, j, :].T

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    im = ax.imshow(slice_u, origin="lower", aspect="auto", cmap="viridis",
                   vmin=0.0, vmax=float(np.percentile(slice_u[~slice_s], 99)))
    ax.contour(slice_s, levels=[0.5], colors="w", linewidths=0.8)
    ax.set_xlabel("x / lattice unit")
    ax.set_ylabel("z / lattice unit")
    ax.set_title(f"|u| in the symmetry plane, sphere Re={snapshot['cfg']['re']:g}, "
                 f"D={snapshot['cfg']['diameter']} (white: sphere surface)")
    fig.colorbar(im, ax=ax, label="|u| / (lattice unit)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# documentation table (numbers injected programmatically)
# --------------------------------------------------------------------------- #

def write_docs(results: Dict[str, Any], path: Path, has_field: bool) -> None:
    rows = sorted(results["summary"], key=lambda r: (r["diameter"], r["re"]))
    lines: List[str] = []
    lines.append("# Sphere-drag anchor (GAP-LBM-SPHERE) — generated data table\n")
    lines.append("Auto-generated by `benchmarks/verified/sphere_drag_bfl/generate_figures.py` "
                 "from `results.json` (no hand-typed numbers).\n")
    lines.append(f"- method: {results['method']['collision']}, "
                 f"{results['method']['boundary']}, {results['method']['force']}")
    lines.append(f"- free-stream lattice velocity u = {results['method']['u_lb']:g}")
    lines.append(f"- external anchor: {results['external_anchor']['source']}")
    lines.append(f"- generated: {results['generated_at']}, "
                 f"total wall time {results['elapsed_s']:.0f} s\n")
    lines.append("## C_D vs anchor\n")
    lines.append("| case | D | Re | C_D (LBM) | C_D ref | err % | steps | converged | wall s |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['key']} | {r['diameter']} | {r['re']:g} | {r['cd']:.4f} | "
            f"{r['cd_ref']:.4f} | {r['cd_err_pct']:.2f} | {r['steps']} | "
            f"{r['converged']} | {r['wall_s']:.0f} |"
        )
    lines.append("\n## Grid refinement\n")
    lines.append("| Re | diameters | C_D | successive change % | finest change % | monotone | pass <5% |")
    lines.append("|---|---|---|---|---|---|---|")
    for key, g in sorted((results.get("grid_convergence") or {}).items()):
        cds = ", ".join(f"{c:.4f}" for c in g["cd"])
        ch = ", ".join(f"{c:.2f}" for c in g["successive_change_pct"])
        lines.append(f"| {key} | {g['diameters']} | {cds} | {ch} | "
                     f"{g['finest_change_pct']:.2f} | {g['monotone_convergence']} | "
                     f"{g['pass_finest_5pct']} |")
    acc = results["acceptance"]
    lines.append("\n## Acceptance (pre-declared bands)\n")
    lines.append(f"- criteria: `{json.dumps(acc['criteria'])}`")
    lines.append(f"- laminar max error: {acc['laminar_max_err_pct']:.2f} % "
                 f"(pass={acc['laminar_pass']})")
    lines.append(f"- transitional max error: {acc['transitional_max_err_pct']:.2f} % "
                 f"(pass={acc['transitional_pass']})")
    lines.append(f"- C_D decreases monotonically with Re: {acc['trend_monotone']}")
    lines.append(f"- all cases finite: {acc['all_finite']}")
    lines.append("\n## Figures\n")
    lines.append("- `benchmarks/verified/sphere_drag_bfl/figures/fig1_cd_vs_re.png`")
    lines.append("- `benchmarks/verified/sphere_drag_bfl/figures/fig2_grid_convergence.png`")
    if has_field:
        lines.append("- `benchmarks/verified/sphere_drag_bfl/figures/fig3_wake_velocity.png`")
    else:
        lines.append("- fig3 (wake field) not produced: no field snapshot available")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(BENCH / "results.json"))
    ap.add_argument("--field-only", action="store_true")
    ap.add_argument("--field-re", type=float, default=100.0)
    ap.add_argument("--field-diameter", type=int, default=8)
    ap.add_argument("--field-steps", type=int, default=2500)
    args = ap.parse_args(argv)

    FIG.mkdir(parents=True, exist_ok=True)

    if args.field_only:
        res = make_field_snapshot(args.field_re, args.field_diameter, args.field_steps)
        print(f"field snapshot: Cd={res['cd']:.4f} steps={res['steps']} "
              f"wall={res['wall_s']:.0f}s -> {res['fields']}")
        return 0

    results = json.loads(Path(args.results).read_text())
    fig_cd_vs_re(results, FIG / "fig1_cd_vs_re.png")
    fig_grid_convergence(results, FIG / "fig2_grid_convergence.png")
    snap = _load_snapshot()
    has_field = False
    if snap is not None:
        has_field = fig_wake(snap, FIG / "fig3_wake_velocity.png")
    write_docs(results, DOCS / "sphere_drag_bfl_anchor.md", has_field)
    print(f"figures written to {FIG}")
    print(f"data table written to {DOCS / 'sphere_drag_bfl_anchor.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())