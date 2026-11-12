#!/usr/bin/env python3
"""CASE-BACKWARD-STEP-LBM -- machine-checkable verdict over the Armaly ladder artefacts.

Round 2 of this case (``docs/CASE_BACKWARD_STEP_LBM_ROUND2.md``) delivered the
caliber fixes and the convergence ladder but produced **no converged point**, so the
anchor could not be judged.  This module turns the artefacts of that ladder into a
verdict that a machine can check: every number is read from
``artifacts/case_backward_step_armaly/<tag>/result.json`` (never hand-copied) and every
acceptance condition is evaluated separately, so a failing condition cannot be hidden
behind a passing average.

Conditions (all must hold for a point to be judged PASS)
-------------------------------------------------------
``error``      ``|X1/s - X1/s_ref| <= 15 %``  (anchor tolerance, refs.json)
``steady``     the instantaneous reattachment length stops drifting: either the run
               stopped itself on ``steady_tol`` or the drift of the last three sampled
               checks is ``<= 1 %`` -- a *converged* point is what round 2 lacked
``caliper``    the shear-zero-crossing caliper and the independent
               ``reversed_flow_extent`` bound agree: ``reverse_extent <= X1/s + 0.5``
               (a reversed-flow region may not outlast the detected reattachment point)
``mass``       inlet/outlet mass flux imbalance ``<= 1e-3`` (the averaging window is not
               polluted by a leaking outlet)
``lattice``    ``tau >= 0.55`` and inlet Mach ``<= 0.3`` (BGK/TRT validity window)

Usage
-----
    PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/verdict.py
    PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/verdict.py --figures

Writes ``artifacts/case_backward_step_armaly/verdict.json`` + ``verdict.md`` and exits
non-zero when any *judged* point fails, so it can be used as a CI gate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CASE = REPO / "benchmarks" / "case_pool" / "backward_step_armaly"
ART = REPO / "artifacts" / "case_backward_step_armaly"
FIG = REPO / "figures" / "CASE_BACKWARD_STEP_LBM_round3"

TOLERANCE_PCT = 15.0
STEADY_DRIFT_PCT = 1.0
CALIPER_SLACK = 0.5
MASS_FLUX_TOL = 1e-3
TAU_MIN = 0.55
MACH_MAX = 0.3


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def steadiness(run: dict) -> dict:
    """Drift of the instantaneous X1/s over the last three sampled checks."""
    hist = [h for h in (run.get("convergence_history") or []) if _finite(h.get("x1_over_s_instant"))]
    res = run.get("result", {})
    out = {"steady_at_step": res.get("steady_at_step"), "self_stopped": res.get("steady_at_step") is not None,
           "drift_pct": None, "n_checks": len(hist)}
    if len(hist) >= 3:
        tail = [h["x1_over_s_instant"] for h in hist[-3:]]
        base = abs(tail[0]) if tail[0] else None
        if base:
            out["drift_pct"] = 100.0 * max(abs(v - tail[0]) for v in tail) / base
        out["tail"] = tail
        out["tail_steps"] = [h["step"] for h in hist[-3:]]
    return out


def judge_run(tag: str, run: dict, refs: dict) -> dict:
    cfg, res, der = run.get("config", {}), run.get("result", {}), run.get("derived", {})
    re = float(cfg["re"])
    points = refs["adopted_numeric_reference"]["points"]
    ref = float(points[str(int(re))]["x1_over_s"])
    x1 = res.get("x1_over_s")
    err = 100.0 * (float(x1) - ref) / ref if _finite(x1) else None
    st = steadiness(run)
    ext = res.get("x1_over_s_reverse_extent")
    mass = res.get("mass_flux_rel_error")
    tau, mach = der.get("tau"), der.get("mach_inlet")

    checks = {
        "error": _finite(err) and abs(err) <= TOLERANCE_PCT,
        "steady": bool(st["self_stopped"]) or (st["drift_pct"] is not None
                                               and st["drift_pct"] <= STEADY_DRIFT_PCT),
        "caliper": _finite(ext) and _finite(x1) and float(ext) <= float(x1) + CALIPER_SLACK,
        "mass": _finite(mass) and abs(float(mass)) <= MASS_FLUX_TOL,
        "lattice": _finite(tau) and _finite(mach) and float(tau) >= TAU_MIN and float(mach) <= MACH_MAX,
    }
    return {
        "tag": tag, "re": re, "hi": cfg.get("hi"), "s": cfg.get("s"),
        "expansion_ratio": der.get("expansion_ratio"),
        "expansion_ratio_deviation_pct": der.get("expansion_ratio_deviation_pct"),
        "tau": tau, "mach_inlet": mach,
        "steps_run": res.get("steps_run"), "average_window_steps": res.get("average_window_steps"),
        "n_averaged_samples": res.get("n_averaged_samples"),
        "x1_over_s": x1, "ref_x1_over_s": ref, "error_pct": err,
        "x1_over_s_reverse_extent": ext, "x1_over_s_row1": res.get("x1_over_s_row1"),
        "mass_flux_rel_error": mass, "steadiness": st,
        "checks": checks, "pass": all(checks.values()),
        "note_missing": [k for k, ok in checks.items() if not ok],
    }


def evaluate(repo: Path = REPO, tags: list[str] | None = None) -> dict:
    art = Path(repo) / "artifacts" / "case_backward_step_armaly"
    refs = json.loads((Path(repo) / "benchmarks" / "case_pool" / "backward_step_armaly" / "refs.json").read_text())
    runs: dict[str, dict] = {}
    for p in sorted(art.glob("*/result.json")):
        try:
            runs[p.parent.name] = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
    if tags is not None:
        runs = {k: v for k, v in runs.items() if k in tags}
    rows = [judge_run(tag, run, refs) for tag, run in sorted(runs.items())]
    judged = [r for r in rows if r["re"] in (100.0, 200.0, 400.0)]
    return {
        "case": "CASE-BACKWARD-STEP-LBM",
        "reference": "Armaly et al. 1983 JFM 127:473 (in-repo anchor tier SECONDARY_IN_REPO)",
        "tolerance_pct": TOLERANCE_PCT,
        "conditions": {"steady_drift_pct": STEADY_DRIFT_PCT, "caliper_slack": CALIPER_SLACK,
                       "mass_flux_tol": MASS_FLUX_TOL, "tau_min": TAU_MIN, "mach_max": MACH_MAX},
        "runs": rows,
        "n_runs": len(rows),
        "n_pass": sum(1 for r in rows if r["pass"]),
        "max_abs_error_pct": max((abs(r["error_pct"]) for r in rows if _finite(r["error_pct"])), default=None),
        "all_judged_pass": bool(judged) and all(r["pass"] for r in judged),
    }


def _fmt(v, nd=3) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def render_md(v: dict) -> str:
    lines = [
        "# CASE-BACKWARD-STEP-LBM · converged-ladder verdict (machine generated)", "",
        f"reference: {v['reference']} · tolerance {v['tolerance_pct']:.0f} %", "",
        "| tag | Re | h_in | steps | steady@ | X1/s (shear) | reverse-extent | ref | err % | checks |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in v["runs"]:
        failed = ",".join(r["note_missing"]) or "-"
        lines.append(
            f"| {r['tag']} | {r['re']:.0f} | {r['hi']} | {r['steps_run']} | {r['steadiness']['steady_at_step']} | "
            f"{_fmt(r['x1_over_s'])} | {_fmt(r['x1_over_s_reverse_extent'])} | {r['ref_x1_over_s']:.2f} | "
            f"{_fmt(r['error_pct'], 2) if r['error_pct'] is not None else 'n/a'} | "
            f"{'PASS' if r['pass'] else 'FAIL(' + failed + ')'} |")
    lines += ["", f"runs={v['n_runs']} pass={v['n_pass']} "
                  f"max|err|={_fmt(v['max_abs_error_pct'], 2)} % "
                  f"all_judged_pass={v['all_judged_pass']}"]
    return "\n".join(lines) + "\n"


def field_figures(v: dict) -> list[str]:
    """Streamlines + near-wall shear of every committed field, and the X1/s-vs-Re panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIG.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    for r in v["runs"]:
        npz = ART / r["tag"] / "field.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        ux, solid, x_step, s = d["ux_mean"], d["solid"], int(d["x_step"]), int(d["s"])
        uy = d["uy_mean"] if "uy_mean" in d else np.zeros_like(ux)
        ux = np.where(solid, np.nan, ux)
        fig, ax = plt.subplots(figsize=(9.0, 3.2))
        H = ux.shape[0]
        ys, xs = np.mgrid[0:ux.shape[0], 0:ux.shape[1]]
        im = ax.pcolormesh(xs, ys, ux, cmap="RdBu_r", shading="auto",
                           vmin=-0.1, vmax=0.15)
        ax.contour(xs, ys, np.nan_to_num(ux), levels=[0.0], colors="k", linewidths=0.6)
        ax.set_aspect("equal")
        if _finite(r["x1_over_s"]):
            ax.axvline(x_step + r["x1_over_s"] * s, color="lime", lw=1.2, ls="--",
                       label=f"X1/s={r['x1_over_s']:.2f}")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{r['tag']} (Re={r['re']:.0f}): mean u_x [lattice], y=0..{H - 1}")
        ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")
        fig.colorbar(im, ax=ax, shrink=0.85)
        p = FIG / f"field_{r['tag']}.png"
        fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
        made.append(str(p.relative_to(REPO)))

    pts = [(r["re"], r["x1_over_s"], r["error_pct"], r["hi"]) for r in v["runs"] if _finite(r["x1_over_s"])]
    if pts:
        refs = json.loads((CASE / "refs.json").read_text())
        ref = refs["adopted_numeric_reference"]["points"]
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        rx = sorted(float(k) for k in ref)
        ax.plot(rx, [float(ref[str(int(k))]["x1_over_s"]) for k in rx], "k--o", label="Armaly 1983 (anchor)")
        for hi, marker in ((52, "s"), (104, "^")):
            q = [(x, y) for x, y, _, h in pts if h == hi]
            if q:
                ax.plot(*zip(*q), marker, label=f"LBM $h_{{in}}$={hi} cells")
        for x, y, e, _ in pts:
            ax.annotate(f"{e:+.1f}%", (x, y), textcoords="offset points", xytext=(5, -10), fontsize=8)
        ax.axhspan(0, 0, alpha=0)
        ax.set_xlabel("Re = $\\bar u D_h/\\nu$"); ax.set_ylabel("$X_1/s$")
        ax.set_title("Backward-facing step reattachment (converged ladder)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        p = FIG / "x1_vs_re_converged.png"
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        made.append(str(p.relative_to(REPO)))
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--tags", nargs="*", default=None)
    args = ap.parse_args(argv)
    v = evaluate(tags=args.tags)
    if args.figures:
        v["figures"] = field_figures(v)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "verdict.json").write_text(json.dumps(v, indent=2, default=str))
    md = render_md(v)
    (ART / "verdict.md").write_text(md)
    print(md)
    for f in v.get("figures", []):
        print("figure:", f)
    return 0 if v["all_judged_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())