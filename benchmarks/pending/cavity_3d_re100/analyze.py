"""Post-processing / figure generation for CASE-LBM-CAVITY-3D-RE100.

Reads the run_*.json produced by run.py, builds the reference comparison table,
convergence analysis and the report figures.

Reference policy (honest):
  * Ku, Hirsh & Taylor (1987) JCP 70:439 (DOI:10.1016/0021-9991(87)90190-2) is the
    mission anchor, but the paper is paywalled and every OA mirror probed from this
    worktree was unreachable (fatcat/scholar.archive.org/fatcat API -> network
    blocked; S2/OpenAlex report CLOSED access).  Its tabulated numbers were therefore
    NOT transcribed -> the numerical error column vs Ku is reported as WITHHELD,
    never as a guessed value.
  * What is actually compared against, with provenance inside this repo:
      - in-repo Ghia et al. (1982) Re=100 2D centreline data
        (tensorlbm.lid_driven_cavity.GHIA_RE100) -> quantifies the 3D side-wall
        correction of the mid-plane extrema;
      - grid triad 25^3 / 33^3 / 41^3 -> self-convergence + Richardson extrapolation.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def load_runs(art: Path) -> dict:
    runs = {}
    for p in sorted(art.glob("run_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        runs[p.stem.replace("run_", "")] = d
    return runs


def ghia_re100():
    """Extract (y, u) vertical and (x, v) horizontal centreline data of 2D Ghia Re=100."""
    from tensorlbm.lid_driven_cavity import GHIA_RE100

    g = GHIA_RE100
    # normalise container types seen in the repo: dict of lists, or dict of dicts
    def arr(key_options):
        for k in key_options:
            if k in g:
                return np.asarray(g[k], dtype=float)
        raise KeyError(key_options)

    y = arr(["y", "y_coords", "vertical_y"])
    u = arr(["u", "u_vertical", "vertical_u"])
    x = arr(["x", "x_coords", "horizontal_x"])
    v = arr(["v", "v_horizontal", "horizontal_v"])
    return x, v, y, u


def richardson(vals: dict, p: float = 2.0) -> dict:
    """Richardson extrapolation from the two finest grids."""
    keys = sorted(vals, key=lambda k: int(k))
    if len(keys) < 2:
        return {}
    n1, n2 = int(keys[-2]), int(keys[-1])
    f1, f2 = vals[keys[-2]], vals[keys[-1]]
    h1, h2 = 1.0 / n1, 1.0 / n2
    r = (h1 / h2) ** p
    f_ex = (r * f2 - f1) / (r - 1.0)
    return {"n_fine": n2, "n_coarse": n1, "p": p, "value": f_ex,
            "fine_value": f2, "correction": abs(f2 - f_ex)}


def make_figures(runs: dict, art: Path, figdir: Path) -> list[dict]:
    figdir.mkdir(parents=True, exist_ok=True)
    figs = []
    x_g, v_g, y_g, u_g = ghia_re100()

    main = runs.get("33_ulid010")
    if main is None and runs:
        main = list(runs.values())[0]

    # ---- Fig 1: mid-plane centreline profiles (3D vs 2D Ghia Re=100) ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(u_g, y_g, "o", ms=4, mfc="none", color="k", label="Ghia 1982 Re=100 (2D)")
    ax[1].plot(x_g, v_g, "o", ms=4, mfc="none", color="k", label="Ghia 1982 Re=100 (2D)")
    for key, d in runs.items():
        lab = f"TensorLBM {key} (n={d['n']}, u_lid={d['u_lid']})"
        ax[0].plot(d["u_centerline_mid"], d["y_coords"], "-", lw=1.2, label=lab)
        ax[1].plot(d["x_coords"], d["v_centerline_mid"], "-", lw=1.2, label=lab)
    ax[0].set_xlabel("$u_x/U_{lid}$"); ax[0].set_ylabel("$y$")
    ax[0].set_title("vertical centreline, mid plane z=0.5"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("$x$"); ax[1].set_ylabel("$u_y/U_{lid}$")
    ax[1].set_title("horizontal centreline, mid plane z=0.5"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
    fig.suptitle("3D cubic lid-driven cavity Re=100: TensorLBM vs 2D Ghia (side-wall correction)")
    p1 = figdir / "centerlines.png"; fig.tight_layout(); fig.savefig(p1, dpi=150); plt.close(fig)
    figs.append({"path": str(p1), "caption": "Mid-plane centreline profiles z=0.5 (u_x along y, u_y along x) for the grid/speed studies vs 2D Ghia Re=100 data."})

    # ---- Fig 2: grid convergence of the extrema ----
    ns, umins, vmins = [], [], []
    for key, d in runs.items():
        ns.append(d["n"]); umins.append(d["u_min"]); vmins.append(d["v_min"])
    order = np.argsort(ns)
    ns = np.array(ns)[order]; umins = np.array(umins)[order]; vmins = np.array(vmins)[order]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(1.0 / ns, umins, "s-", label="$u_{min}$ (vertical CL)")
    ax.plot(1.0 / ns, vmins, "o-", label="$v_{min}$ (horizontal CL)")
    ax.set_xlabel("$1/n$"); ax.set_ylabel("normalised extremum")
    ax.grid(alpha=.3); ax.legend()
    ax.set_title("grid convergence of mid-plane extrema (Re=100, u_lid=0.1)")
    p2 = figdir / "grid_convergence.png"; fig.tight_layout(); fig.savefig(p2, dpi=150); plt.close(fig)
    figs.append({"path": str(p2), "caption": "Grid convergence of the mid-plane extrema u_min(0.5,y,z=0.5) and v_min(x,0.5,z=0.5) versus 1/n."})

    # ---- Fig 3: mid-plane field of u_x and u_z (from npz dump) ----
    dump = art / "field_33.npz"
    if dump.exists():
        z = np.load(dump)
        ux, uz = z["ux"], z["uz"]
        kz = ux.shape[0] // 2
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
        im0 = ax[0].imshow(ux[kz], origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r", vmin=-0.22, vmax=1.0)
        ax[0].set_title("$u_x/U_{lid}$, mid plane z=0.5"); fig.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(uz[kz], origin="lower", extent=[0, 1, 0, 1], cmap="PuOr_r")
        ax[1].set_title("$u_z/U_{lid}$, mid plane z=0.5 (should be ~0 by symmetry)"); fig.colorbar(im1, ax=ax[1])
        for a in ax:
            a.set_xlabel("x"); a.set_ylabel("y")
        p3 = figdir / "field_midplane.png"; fig.tight_layout(); fig.savefig(p3, dpi=150); plt.close(fig)
        figs.append({"path": str(p3), "caption": "Mid-plane fields from the 33^3 run: axial velocity u_x (left) and spanwise u_z (right, symmetry check)."})
    return figs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="artifacts/cavity_3d_re100")
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--out", default="artifacts/cavity_3d_re100/summary.json")
    args = ap.parse_args(argv)
    art = Path(args.art)
    figdir = Path(args.figdir) if args.figdir else art / "figures"

    runs = load_runs(art)
    if not runs:
        print("no runs found in", art)
        return 1

    # grid study restricted to the base lid speed so the triad is comparable
    grid_runs = {str(d["n"]): d for k, d in runs.items() if abs(d["u_lid"] - 0.1) < 1e-12}
    rich = {
        "u_min": richardson({k: d["u_min"] for k, d in grid_runs.items()}),
        "v_min": richardson({k: d["v_min"] for k, d in grid_runs.items()}),
        "u_min_y": richardson({k: d["u_min_y"] for k, d in grid_runs.items()}),
        "v_min_x": richardson({k: d["v_min_x"] for k, d in grid_runs.items()}),
    }

    x_g, v_g, y_g, u_g = ghia_re100()
    iu = int(np.argmin(u_g[1:-1])); iv = int(np.argmin(v_g[1:-1]))
    ghia = {"u_min": float(u_g[1:-1][iu]), "u_min_y": float(y_g[1:-1][iu]),
            "v_min": float(v_g[1:-1][iv]), "v_min_x": float(x_g[1:-1][iv])}

    figs = make_figures(runs, art, figdir)
    summary = {
        "case": "cavity_3d_re100",
        "reference": {
            "anchor": "Ku, Hirsh & Taylor (1987), J. Comput. Phys. 70:439-462, DOI:10.1016/0021-9991(87)90190-2",
            "status": "WITHHELD",
            "reason": "paper is paywalled (S2/OpenAlex: CLOSED); OA mirrors (fatcat API, scholar.archive.org, MDPI/J-STAGE PDF copies) unreachable from the worktree network; tabulated values not transcribed (zero-hand-copy discipline)",
            "in_repo_secondary": "2D Ghia et al. (1982) Re=100 centreline data (tensorlbm.lid_driven_cavity.GHIA_RE100)",
        },
        "ghia_2d_re100": ghia,
        "runs": {k: {kk: d[kk] for kk in ["n", "re", "u_lid", "length_convention", "L_lattice", "nu_lattice", "tau",
                                          "steps", "converged", "last_resid", "elapsed_s", "mass_drift", "finite",
                                          "u_min", "u_min_y", "v_min", "v_min_x", "u_max", "u_max_y", "v_max", "v_max_x",
                                          "span_symmetry"]} for k, d in runs.items()},
        "richardson": rich,
        "figures": [{"path": f["path"], "caption": f["caption"]} for f in figs],
    }

    # 3D side-wall correction vs the 2D anchor
    for k, d in summary["runs"].items():
        d["dU_pct_vs_2d_ghia"] = 100.0 * (d["u_min"] - ghia["u_min"]) / abs(ghia["u_min"])
        d["dV_pct_vs_2d_ghia"] = 100.0 * (d["v_min"] - ghia["v_min"]) / abs(ghia["v_min"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary["runs"][k] for k in summary["runs"]}, indent=1, ensure_ascii=False))
    print(json.dumps({"richardson": rich, "ghia": ghia}, indent=1, ensure_ascii=False))
    print("figures:", [f["path"] for f in figs])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())