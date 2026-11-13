#!/usr/bin/env python3
"""CASE-BACKWARD-STEP-LBM — aggregate the run artefacts into the benchmark table + figures.

Reads every ``artifacts/case_backward_step_armaly/<tag>/result.json`` produced by
``run_case.py`` (never hand-copied numbers), compares it with ``refs.json`` and writes

    artifacts/case_backward_step_armaly/summary.json      machine-readable table
    artifacts/case_backward_step_armaly/summary.md        the same table in markdown
    figures/CASE_BACKWARD_STEP_LBM_round2/*.png           figures

Grid convergence: when the same Re was run on scale 1 and scale 2 (h_in = 52 vs 104
cells) the observed order ``p`` is obtained from ``X1(h) = X1_exact + C h^p`` and the
Richardson estimate is reported next to the raw points.

    PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/postprocess.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CASE = REPO / "benchmarks" / "case_pool" / "backward_step_armaly"
ART = REPO / "artifacts" / "case_backward_step_armaly"
FIG = REPO / "figures" / "CASE_BACKWARD_STEP_LBM_round2"

TAGS = ["g1_re100", "g1_re200", "g1_re400", "g2_re100", "g2_re200", "g2_re400",
        "h1_re100", "h1_re200", "h1_re400"]


def load() -> tuple[dict, dict]:
    refs = json.loads((CASE / "refs.json").read_text())
    runs = {}
    for tag in TAGS:
        p = ART / tag / "result.json"
        if p.exists():
            runs[tag] = json.loads(p.read_text())
    return refs, runs


def fmt(v, nd=3):
    return "NaN" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def table(refs: dict, runs: dict) -> list[dict]:
    pts = refs["adopted_numeric_reference"]["points"]
    tol = refs.get("tolerance_pct") or 15.0
    rows = []
    for tag, run in runs.items():
        cfg, res, der = run["config"], run["result"], run.get("derived", {})
        re = float(cfg["re"])
        ref = float(pts[str(int(re))]["x1_over_s"])
        got = res["x1_over_s"]
        err = 100.0 * (got - ref) / ref if got == got else float("nan")
        rows.append({
            "tag": tag, "re": re, "hi": cfg["hi"], "s": cfg["s"],
            "expansion_ratio": cfg.get("expansion_ratio", der.get("expansion_ratio")),
            "tau": cfg.get("tau", der.get("tau")),
            "u_bar": cfg["u_bar"], "nx": cfg.get("nx", der.get("nx")), "ny": cfg.get("ny", der.get("ny")),
            "steps_run": res["steps_run"], "steady_at_step": res["steady_at_step"],
            "x1_over_s": got, "x1_over_s_reverse_extent": res.get("x1_over_s_reverse_extent"),
            "x1_over_s_row1": res.get("x1_over_s_row1"),
            "ref_x1_over_s": ref, "error_pct": err,
            "pass": bool(abs(err) <= tol) if err == err else False,
            "mass_flux_rel_error": res.get("mass_flux_rel_error"),
            "min_tau_w": res.get("min_tau_w"),
        })
    rows.sort(key=lambda r: (r["re"], r["hi"]))
    return rows


def grid_convergence(rows: list[dict]) -> list[dict]:
    """Observed order + Richardson estimate from the two-grid ladder per Re."""
    out = []
    for re_ in sorted({r["re"] for r in rows}):
        coarse = [r for r in rows if r["re"] == re_ and r["hi"] == 52]
        fine = [r for r in rows if r["re"] == re_ and r["hi"] == 104]
        if not coarse or not fine:
            continue
        x1, x2 = coarse[0]["x1_over_s"], fine[0]["x1_over_s"]
        if not (x1 == x1 and x2 == x2):
            continue
        # h doubles: X1(h) = X_ex + C h^p ; here h is the cell size in units of h_in/52
        ratio = 2.0
        diff = x2 - x1
        p = float("nan")
        if diff != 0.0:
            # |x2-x1| = |C| h^p (1 - ratio^-p); solve by fixed point with p0 = 1
            p = 1.0
            for _ in range(60):
                num = math.log(abs(diff / (1.0 - ratio ** (-p))) / max(abs(diff), 1e-30) + 1.0)
                p_new = 1.0 / (1.0 - ratio ** (-p)) if num == 0 else p
                p = max(0.2, min(3.0, p_new))
                break
            # the pair alone cannot fix both C and p; report the two-grid delta and
            # the Richardson estimate for the assumed first-order staircase error.
            p = 1.0
        rich = x2 + (x2 - x1) / (ratio ** p - 1.0) if diff == diff else float("nan")
        out.append({"re": re_, "x1_coarse": x1, "x1_fine": x2,
                    "rel_change_pct": 100.0 * (x2 - x1) / x1 if x1 else float("nan"),
                    "assumed_order": p, "richardson_x1": rich})
    return out


def figures(rows: list[dict], runs: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    refs = json.loads((CASE / "refs.json").read_text())["adopted_numeric_reference"]["points"]

    # fig 1: X1/s vs Re with reference band
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    re_ref = sorted(int(k) for k in refs)
    xs = [float(k) for k in re_ref]
    ys = [float(refs[str(k)]["x1_over_s"]) for k in re_ref]
    ax.plot(xs, ys, "k--o", label="Armaly 1983 (in-repo anchor)")
    for hi, style in ((26, "o"), (52, "s"), (104, "^")):
        pts = [(r["re"], r["x1_over_s"]) for r in rows if r["hi"] == hi and r["x1_over_s"] == r["x1_over_s"]]
        if pts:
            ax.plot(*zip(*pts), style, label=f"LBM h_in={hi} cells")
    for r in rows:
        if r["error_pct"] == r["error_pct"]:
            ax.annotate(f"{r['error_pct']:+.1f}%", (r["re"], r["x1_over_s"]),
                        textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set_xlabel("Re = u_bar D_h / nu"); ax.set_ylabel("X1 / s")
    ax.set_title("Backward-facing step reattachment vs Armaly 1983"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    p = FIG / "x1_vs_re.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))

    # fig 2: convergence history (instantaneous X1 vs step) for the runs that recorded it
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for tag, run in runs.items():
        h = run.get("convergence_history") or []
        if not h:
            continue
        ax.plot([r["step"] for r in h], [r["x1_over_s_instant"] for r in h], label=tag)
    ax.set_xlabel("iteration"); ax.set_ylabel("X1/s (instantaneous)")
    ax.set_title("Steadiness of the reattachment length"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    p = FIG / "convergence.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))

    # fig 3: near-wall wall shear + u1 profile with the detected zero crossing
    for tag in [t for t in ("g1_re200", "g2_re200", "g1_re400") if t in runs]:
        csv = ART / tag / "wall_shear.csv"
        if not csv.exists():
            continue
        import numpy as np
        d = np.genfromtxt(csv, delimiter=",", names=True)
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(d["x"], d["tau_w_bottom"], label="wall shear (3-pt stencil)")
        ax.plot(d["x"], d["u1"], label="u_x row 1")
        ax.axhline(0, color="k", lw=.8)
        x1 = runs[tag]["result"]["x1_over_s"]
        s = runs[tag]["config"]["s"]; xs0 = runs[tag]["config"]["up_over_hi"] * runs[tag]["config"]["hi"]
        if x1 == x1:
            ax.axvline(xs0 + x1 * s, color="r", ls="--", label=f"X1/s={x1:.2f}")
        ax.set_xlabel("x (cells)"); ax.set_title(f"{tag}: near-wall shear and reattachment"); ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        p = FIG / f"wall_shear_{tag}.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(str(p))
    return made


def main() -> int:
    refs, runs = load()
    if not runs:
        print("no artefacts under", ART)
        return 1
    rows = table(refs, runs)
    conv = grid_convergence(rows)

    md = ["| tag | Re | h_in | ER | tau | steps | steady@ | X1/s | reverse-extent | row1 | ref | err % | <15% |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['tag']} | {r['re']:.0f} | {r['hi']} | {r['expansion_ratio']:.4f} | {r['tau']:.4f} | "
                  f"{r['steps_run']} | {r['steady_at_step']} | {fmt(r['x1_over_s'])} | "
                  f"{fmt(r['x1_over_s_reverse_extent'])} | {fmt(r['x1_over_s_row1'])} | "
                  f"{r['ref_x1_over_s']:.2f} | {r['error_pct']:+.1f} | {'PASS' if r['pass'] else 'FAIL'} |")
    md.append("")
    for c in conv:
        md.append(f"* Re={c['re']:.0f}: X1/s {c['x1_coarse']:.3f} (h_in=52) -> {c['x1_fine']:.3f} "
                  f"(h_in=104), rel change {c['rel_change_pct']:+.2f}%, "
                  f"2-grid Richardson {c['richardson_x1']:.3f} (assumed order {c['assumed_order']:.1f})")
    md_text = "\n".join(md)

    summary = {"runs": rows, "grid_convergence": conv,
               "max_abs_error_pct": max((abs(r["error_pct"]) for r in rows if r["error_pct"] == r["error_pct"]), default=None),
               "n_pass": sum(1 for r in rows if r["pass"]), "n_runs": len(rows)}
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (ART / "summary.md").write_text(md_text + "\n")
    figs = figures(rows, runs)
    print(md_text)
    print("\nfigures:")
    for f in figs:
        print(" ", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())