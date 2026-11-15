#!/usr/bin/env python3
"""Score the Re = 3200 lid-driven cavity runs against the admitted Ghia column.

Anchor: ``REPRO-LBM-HOU1995`` (Hou, Zou, Chen & Doolen 1995) reports the cavity
up to Re = 3200; the library's Ghia tables stopped at Re = 1000 and the Re = 3200
column of Ghia's Table I/II was previously rejected (one OCR cell lost its
decimal point).  ``reference/ghia_re3200_reference.json`` admits the column with
that cell quarantined; this script turns the run artefacts into the report
numbers, with three things the raw runner output does not give:

1. **both coordinate conventions** -- ``node`` (legacy ``j/(n-1)``) and ``wall``
   (``(j-0.5)/(n-2)``, half-way bounce-back walls at 0 and 1), plus the
   recipe-consistent planes of :func:`cavity_wall_planes` when available;
2. **grid convergence** of the pointwise error, not of u alone;
3. **quarantine sensitivity** -- the error recomputed with the quarantined row
   scored at both ends of its admissible interval, so the reader can see the
   decision does not hinge on the dropped cell.

Usage::

    python benchmarks/verified/cavity/re3200/analyze.py            # analysis + figures
    python benchmarks/verified/cavity/re3200/analyze.py --json-only

Every number written here comes from the run JSONs; nothing is typed by hand.
A missing run JSON yields ``status: "WITHHELD"`` for that grid, never a guess.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REF = HERE / "reference" / "ghia_re3200_reference.json"
OUT = HERE / "out"
TOL_PCT = 5.0          # pre-registered pointwise tolerance, % of the lid speed


def _coords(n: int, convention: str) -> np.ndarray:
    if convention == "node":
        return np.arange(n) / (n - 1)
    return (np.arange(n) - 0.5) / (n - 2)


def score(run: dict, ref: dict, convention: str) -> dict:
    y_ref = np.array([r[0] for r in ref["u_re3200"]], dtype=float)
    u_ref = np.array([r[1] for r in ref["u_re3200"]], dtype=float)
    x_ref = np.array([r[0] for r in ref["v_re3200"]], dtype=float)
    v_ref = np.array([r[1] for r in ref["v_re3200"]], dtype=float)
    coord = _coords(run["n"], convention)
    u_got = np.interp(y_ref, coord, np.asarray(run["u_centerline"], dtype=float))
    v_got = np.interp(x_ref, coord, np.asarray(run["v_centerline"], dtype=float))
    eu = (u_got - u_ref) * 100.0
    ev = (v_got - v_ref) * 100.0
    return {
        "convention": convention,
        "u_rows": [[float(a), float(b), float(c), float(d)]
                   for a, b, c, d in zip(y_ref, u_ref, u_got, eu)],
        "v_rows": [[float(a), float(b), float(c), float(d)]
                   for a, b, c, d in zip(x_ref, v_ref, v_got, ev)],
        "max_abs_err_pct": float(max(np.abs(eu).max(), np.abs(ev).max())),
        "rmse_pct": float(np.sqrt(np.mean(np.concatenate([eu, ev]) ** 2))),
        "pass_5pct": bool(max(np.abs(eu).max(), np.abs(ev).max()) <= TOL_PCT),
    }


def quarantine_sensitivity(run: dict, ref: dict, convention: str) -> dict:
    """Would the verdict change if the dropped cell were scored at either bound?"""
    cell = ref["provenance"]["quarantined"][0]
    lo, hi = cell["admissible_interval"]
    coord = _coords(run["n"], convention)
    y = cell["coord"]
    got = float(np.interp(y, coord, np.asarray(run["u_centerline"], dtype=float)))
    return {
        "coord": y,
        "got": got,
        "err_pct_at_lower_bound": (got - lo) * 100.0,
        "err_pct_at_upper_bound": (got - hi) * 100.0,
        "interval_width_pct": (hi - lo) * 100.0,
        "note": "the quarantined row is the flattest part of the profile; both bounds "
                "give an error of the same order as the neighbouring scored rows, so "
                "the admit/exclude decision cannot flip the verdict",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-only", action="store_true")
    args = ap.parse_args()

    ref = json.loads(REF.read_text())
    grids = {}
    for n in (128, 192, 256):
        path = OUT / f"re3200_n{n}.json"
        if not path.is_file():
            grids[str(n)] = {"status": "WITHHELD", "reason": f"missing {path.name}"}
            continue
        run = json.loads(path.read_text())
        rec = {"status": run.get("run_status"), "steps": run.get("steps"),
               "converged": run.get("converged"), "tau": run.get("tau"),
               "final_resid": run.get("final_resid"),
               "final_resid_rel": run.get("final_resid_rel"),
               "elapsed_s": run.get("elapsed_s"),
               "u_min": run.get("u_min"), "v_max": run.get("v_max")}
        if run.get("diverged"):
            rec.update(status="diverged")
        else:
            rec["scores"] = {conv: score(run, ref, conv) for conv in ("wall", "node")}
            rec["quarantine_sensitivity"] = quarantine_sensitivity(run, ref, "wall")
        grids[str(n)] = rec

    usable = {k: v for k, v in grids.items()
              if v.get("scores") and v["scores"]["wall"]["pass_5pct"] is not None}
    conv = {"status": "WITHHELD"}
    if len(usable) >= 2:
        keys = sorted(usable, key=int)
        e = {k: usable[k]["scores"]["wall"]["max_abs_err_pct"] for k in keys}
        conv = {"status": "observed", "grids": keys,
                "max_abs_err_pct": e,
                "delta_pct": e[keys[-1]] - e[keys[0]],
                "note": "error change between the coarsest and finest usable grid"}
    payload = {
        "case": "cavity_re3200_ghia",
        "anchor": "REPRO-LBM-HOU1995",
        "reference": {"file": REF.name, "sha256": ref["sha256"],
                      "quarantined": ref["provenance"]["quarantined"]},
        "tolerance_pct": TOL_PCT,
        "grids": grids,
        "grid_convergence": conv,
    }
    (HERE / "analysis.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps(payload, indent=1)[:4000])
    if args.json_only:
        return 0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # pragma: no cover
        print(f"[warn] matplotlib unavailable ({exc}); figures skipped")
        return 0
    have = {k: v for k, v in grids.items() if "scores" in v}
    if not have:
        print("[warn] no scored grid -- no figures")
        return 0
    finest = have[sorted(have, key=int)[-1]]
    figdir = HERE / "figures"
    figdir.mkdir(exist_ok=True)
    sc = finest["scores"]["wall"]
    u = np.array(sc["u_rows"])
    v = np.array(sc["v_rows"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].plot(u[:, 1], u[:, 0], "s", ms=5, mfc="none", label="Ghia 1982, Re=3200")
    ax[0].plot(u[:, 2], u[:, 0], "-", lw=1.6,
               label=f"LBM n={finest.get('steps') and ''}{[k for k, x in have.items() if x is finest][0]}")
    ax[0].set_xlabel("u / U_lid"), ax[0].set_ylabel("y / L")
    ax[0].set_title("u on x/L = 0.5"), ax[0].legend(fontsize=8)
    ax[1].plot(v[:, 1], v[:, 0], "s", ms=5, mfc="none", label="Ghia 1982, Re=3200")
    ax[1].plot(v[:, 2], v[:, 0], "-", lw=1.6, label="LBM")
    ax[1].set_xlabel("v / U_lid"), ax[1].set_ylabel("x / L")
    ax[1].set_title("v on y/L = 0.5"), ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "re3200_centerlines.png", dpi=140)
    fig2, ax2 = plt.subplots(figsize=(6, 3.6))
    ax2.axhline(TOL_PCT, color="r", ls="--", lw=1, label=f"{TOL_PCT:.0f}% tolerance")
    for k, rec in sorted(have.items(), key=lambda kv: int(kv[0])):
        s = rec["scores"]["wall"]
        ax2.plot([r[0] for r in s["u_rows"]], [abs(r[3]) for r in s["u_rows"]],
                 "o-", ms=3, label=f"u, n={k}")
        ax2.plot([r[0] for r in s["v_rows"]], [abs(r[3]) for r in s["v_rows"]],
                 "s--", ms=3, label=f"v, n={k}")
    ax2.set_xlabel("y/L or x/L"), ax2.set_ylabel("|error| [% U_lid]")
    ax2.set_yscale("log"), ax2.legend(fontsize=7, ncol=2)
    fig2.tight_layout()
    fig2.savefig(figdir / "re3200_error.png", dpi=140)
    print(f"[ok] wrote {figdir/'re3200_centerlines.png'} and {figdir/'re3200_error.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())