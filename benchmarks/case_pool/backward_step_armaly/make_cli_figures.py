#!/usr/bin/env python3
"""Figures for a *compatibility-CLI* (``run.py``) artefact -- no hand-copied numbers.

``run.py`` writes the same triple as ``run_case.py`` (``result.json``, ``wall_shear.csv``,
``field.npz``).  This script turns one such tag directory into

* ``field_<tag>.png``   mean ``u_x`` cloud + zero-velocity contour + the reattachment
  column implied by ``result.x1_over_s`` (numbers read from the artefact),
* ``convergence_<tag>.png``  instantaneous X1/s from ``convergence_history`` against the
  ``refs.json`` anchor band, so a transient run cannot be mistaken for a converged one.

Usage
-----
    PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/make_cli_figures.py \\
        --tag-dir artifacts/case_backward_step_armaly/cli_re100_m16 --out figures/CASE_BACKWARD_STEP_LBM_cli
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CASE = REPO / "benchmarks" / "case_pool" / "backward_step_armaly"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-dir", required=True)
    ap.add_argument("--out", default=str(REPO / "figures" / "CASE_BACKWARD_STEP_LBM_cli"))
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag_dir = Path(args.tag_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = json.loads((tag_dir / "result.json").read_text())
    tag = tag_dir.name
    res, der = run["result"], run["derived"]
    refs = json.loads((CASE / "refs.json").read_text())["adopted_numeric_reference"]
    ref = float(refs["points"][str(int(run["config"]["re"]))]["x1_over_s"])
    tol = float(refs["tolerance_pct"])
    made, prov = [], {"tag": tag, "sources": {}}

    npz = tag_dir / "field.npz"
    if npz.is_file():
        d = np.load(npz)
        ux = np.where(d["solid"], np.nan, d["ux_mean"])
        x_step, s = int(d["x_step"]), int(d["s"])
        fig, ax = plt.subplots(figsize=(9.0, 3.2))
        ys, xs = np.mgrid[0:ux.shape[0], 0:ux.shape[1]]
        im = ax.pcolormesh(xs, ys, ux, cmap="RdBu_r", shading="auto")
        ax.contour(xs, ys, np.nan_to_num(ux), levels=[0.0], colors="k", linewidths=0.6)
        if np.isfinite(res["x1_over_s"]):
            ax.axvline(x_step + res["x1_over_s"] * s, color="lime", lw=1.2, ls="--",
                       label=f"X1/s={res['x1_over_s']:.3f}")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal")
        ax.set_title(f"{tag}: time-mean $u_x$ [lattice] (Re={run['config']['re']:.0f}, "
                     f"steps={res['steps_run']}, steady={res['steady_at_step'] is not None})")
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("y (cells)")
        fig.colorbar(im, ax=ax, shrink=0.85)
        p = out / f"field_{tag}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        made.append(p)
        prov["sources"][npz.name] = sha256(npz)

    hist = [h for h in run.get("convergence_history", []) if h.get("x1_over_s_instant") == h.get("x1_over_s_instant")]
    if hist:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        steps = [h["step"] for h in hist]
        vals = [h["x1_over_s_instant"] for h in hist]
        ax.plot(steps, vals, "o-", label="instantaneous X1/s (artefact)")
        ax.axhline(ref, color="k", ls="--", label=f"Armaly anchor {ref:.2f}")
        ax.axhspan(ref * (1 - tol / 100), ref * (1 + tol / 100), color="k", alpha=0.08,
                   label=f"±{tol:.0f} % band")
        ax.axhline(res["x1_over_s"], color="tab:green", ls=":",
                   label=f"time-mean {res['x1_over_s']:.3f} ({100 * (res['x1_over_s'] - ref) / ref:+.1f} %)")
        ax.set_xlabel("step")
        ax.set_ylabel("$X_1/s$")
        ax.set_title(f"{tag}: convergence history (drift {100 * abs(vals[-1] - vals[0]) / abs(vals[0]):.1f} % over window)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        p = out / f"convergence_{tag}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

    prov["sources"]["result.json"] = sha256(tag_dir / "result.json")
    prov["figures"] = [str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p) for p in made]
    (out / f"provenance_{tag}.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())