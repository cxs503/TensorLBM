#!/usr/bin/env python3
"""m23014 — score the DFG 2D-2 refinement ladder against the PRE-REGISTERED gates.

Reads level artifacts (including levels produced by this mission), computes the
four observables' errors against the Schäfer-Turek bands, the convergence spans
between adjacent levels, applies the pre-declared decision rules from
docs/m23014/PREREG.json, and renders the figures with numbers injected
programmatically (no hand-copied values).

Usage:  python docs/m23014/analyse_m23014.py
Output: docs/m23014/analysis.json, docs/figures/m23014_*.png
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.validation import cylinder_dfg as cdfg  # noqa: E402
from tensorlbm.validation.cylinder_dfg import band_error, band_mid  # noqa: E402

ART = REPO / "benchmarks/verified/cylinder/re100_dfg/artifacts"
FIGDIR = REPO / "docs/figures"
METRICS = ("cd", "cl_amp", "st", "dp")
BANDS = cdfg.REFERENCE_BANDS["2D-2"]
TOL = 5.0          # frozen matrix tolerance (percent)
TASK_TOL_CL = 10.0  # task-stated Cl tolerance
SPAN_TOL = 3.0     # convergence span criterion


def load_level(D: int, tag: str) -> dict | None:
    p = ART / f"level_D{D}_{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    d["_file"] = p.name
    return d


def ladder(tag: str = "u0833", levels=(20, 28, 40, 48, 56)) -> dict:
    out = {}
    for D in levels:
        d = load_level(D, tag)
        if d is None:
            continue
        row = {
            "level": f"D{D}",
            "D_cells": D,
            "n_steps": d.get("n_steps"),
            "tau": d.get("tau"),
            "nu_lb": d.get("nu_lb"),
            "nx": d.get("nx"),
            "ny": d.get("ny"),
            "wall_s": d.get("wall_s"),
            "values": {},
            "err_pct": {},
            "source": d["_file"],
        }
        for m in METRICS:
            v = d.get(m)
            if v is None or not np.isfinite(v):
                continue
            row["values"][m] = float(v)
            if m == "cl_amp":
                row["cl_amp_estimator"] = (
                    "fundamental (run_case, transient-free)"
                    if "cl_amp_stats" in d else "legacy peak-pick (whole record)"
                )
                row["cl_amp_peakpick_legacy"] = d.get("cl_amp_peakpick_legacy")
            key = "dp_half_rho_u2" if m == "dp" else m
            row["err_pct"][m] = float(band_error(float(v), BANDS[key])["err_pct_vs_mid"])
        out[f"D{D}"] = row

    # Estimator provenance: the frozen levels D20/D28/D40 were written by the code
    # BEFORE the m16827 estimator fix, so their ``cl_amp`` field is the legacy
    # whole-record peak-pick.  Scoring the new levels (fundamental) against those
    # would be an apples-to-oranges comparison -- exactly the defect class this
    # mission is about.  Re-score the frozen levels with the frozen re-judge.
    rj_path = ART / "rejudge_cl_amp.json"
    if rj_path.exists():
        rj = json.loads(rj_path.read_text())
        for lv, row in out.items():
            entry = rj.get("levels", {}).get(lv)
            if entry is None:
                continue
            row["cl_amp_legacy_field"] = row["values"].get("cl_amp")
            row["values_legacy_window"] = dict(row["values"])
            row["cl_amp_peakpick_legacy"] = float(entry["cl_amp_old_peakpick_all_window"])
            row["cl_amp_estimator"] = "fundamental (m16827 re-judge of frozen series)"
            # ALL four observables are re-scored from the frozen series on the same
            # transient-free window, so the ladder is estimator-consistent.
            for m, v in entry.get("values_new", {}).items():
                if m in METRICS:
                    row["values"][m] = float(v)
            for m in METRICS:
                key = "dp_half_rho_u2" if m == "dp" else m
                row["err_pct"][m] = float(
                    band_error(row["values"][m], BANDS[key])["err_pct_vs_mid"]
                )
            row["window_provenance"] = "frozen series re-scored, transient-free window"
        out["_estimator_note"] = (
            "All four observables for D20/D28/D40 are taken from rejudge_cl_amp.json "
            "(m16827: the frozen series re-scored on the transient-free window, Cl amplitude = "
            "fundamental). Levels written by the current code path are scored by run_case's own "
            "estimator, which uses the same window convention. Same convention on both sides; "
            "the legacy whole-record peak-pick is kept alongside for audit only."
        )
    return out


def spans(lad: dict) -> dict:
    """Change in each metric between adjacent levels, percent of the finer value."""
    keys = sorted((k for k in lad if not k.startswith("_")), key=lambda k: lad[k]["D_cells"])
    out = {m: [] for m in METRICS}
    for a, b in zip(keys, keys[1:]):
        for m in METRICS:
            va, vb = lad[a]["values"].get(m), lad[b]["values"].get(m)
            if va is None or vb is None:
                continue
            out[m].append({
                "pair": f"{a}->{b}",
                "coarse": va, "fine": vb,
                "span_pct": 100.0 * abs(vb - va) / abs(vb),
            })
    return out


def decide(lad: dict, sp: dict) -> dict:
    """Apply the PRE-REGISTERED decision rules. No post-hoc threshold edits."""
    keys = sorted((k for k in lad if not k.startswith("_")), key=lambda k: lad[k]["D_cells"])
    if not keys:
        return {"verdict": "NO_DATA"}
    finest = keys[-1]
    f = lad[finest]
    e = f["err_pct"]
    gate = {
        "finest_level": finest,
        "cl_amp_err_pct": e.get("cl_amp"),
        "dp_err_pct": e.get("dp"),
        "cd_err_pct": e.get("cd"),
        "cl_lt_10": (e.get("cl_amp") is not None and abs(e["cl_amp"]) < TASK_TOL_CL),
        "cl_lt_5": (e.get("cl_amp") is not None and abs(e["cl_amp"]) < TOL),
        "dp_lt_5": (e.get("dp") is not None and abs(e["dp"]) < TOL),
        "cd_lt_5": (e.get("cd") is not None and abs(e["cd"]) < TOL),
    }
    gate["primary_gate_pass"] = bool(
        gate["cl_lt_10"] and gate["dp_lt_5"] and gate["cd_lt_5"]
    )
    finest_span = {}
    for m in METRICS:
        if sp.get(m):
            finest_span[m] = sp[m][-1]["span_pct"]
    gate["finest_span_pct"] = finest_span
    gate["secondary_gate_pass"] = bool(
        finest_span and all(v < SPAN_TOL for v in finest_span.values())
    )
    if gate["primary_gate_pass"] and gate["secondary_gate_pass"]:
        verdict = "PASS_CONVERGED"
        rule = "PREREG rule 1: primary gate + span gate both met at finest level"
    elif gate["primary_gate_pass"]:
        verdict = "HONEST_FAIL_CONVERGENCE"
        rule = "PREREG rule 2: primary gate met but span gate failed -> no converged level"
    elif gate["cl_amp_err_pct"] is not None and abs(gate["cl_amp_err_pct"]) >= TASK_TOL_CL:
        verdict = "H1_FALSIFIED_RESIDUAL_NOT_GRID"
        rule = "PREREG rule 3: Cl error >= 10% at the finest level -> grid is not the root cause"
    else:
        verdict = "HONEST_FAIL_OTHER"
        rule = "not covered by a pre-declared branch; reported as-is"
    return {"verdict": verdict, "rule_applied": rule, "gate": gate}


def tau_sensitivity(lad: dict) -> dict | None:
    """H2: same D40 geometry, U_mean lowered -> tau 0.600 -> 0.572 (Re fixed at 100)."""
    hi = load_level(40, "u0833")
    lo = load_level(40, "u0600")
    if not (hi and lo):
        return None
    out = {"tau_hi": hi.get("tau"), "tau_lo": lo.get("tau"),
           "u_mean_hi": 0.0833, "u_mean_lo": 0.06, "metrics": {}}
    for m in METRICS:
        a, b = hi.get(m), lo.get(m)
        if a is None or b is None:
            continue
        out["metrics"][m] = {
            "tau_0600": float(a), "tau_0571": float(b),
            "delta_pct": 100.0 * (b - a) / a,
        }
    if "cl_amp" in out["metrics"]:
        d = abs(out["metrics"]["cl_amp"]["delta_pct"])
        out["h2_verdict"] = (
            "tau_model_dominant" if d >= 5.0
            else "tau_model_not_dominant" if d < 2.0
            else "tau_model_marginal"
        )
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def render(lad: dict, sp: dict, dec: dict, h2: dict | None, h4: dict, audit_h3: dict) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    keys = sorted((k for k in lad if not k.startswith("_")), key=lambda k: lad[k]["D_cells"])
    D = [lad[k]["D_cells"] for k in keys]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # (a) observables normalised
    ax = axes[0, 0]
    for m, c, lab in (("cd", "#1f77b4", "C_D"), ("cl_amp", "#d62728", "C_L amplitude"),
                      ("dp", "#2ca02c", r"$\Delta p^*$"), ("st", "#9467bd", "St")):
        if not all(m in lad[k]["values"] for k in keys):
            continue
        band = "dp_half_rho_u2" if m == "dp" else m
        ax.plot(D, [lad[k]["values"][m] / band_mid(BANDS[band]) for k in keys],
                "o-", color=c, label=lab)
    ax.axhspan(0.95, 1.05, color="k", alpha=0.08, label="±5% tolerance")
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.set_xlabel("cylinder diameter [lattice cells]")
    ax.set_ylabel("value / reference mid-point")
    ax.set_title("(a) DFG 2D-2 observables vs resolution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) signed error
    ax = axes[0, 1]
    for m, c, lab in (("cd", "#1f77b4", "C_D"), ("cl_amp", "#d62728", "C_L amplitude"),
                      ("dp", "#2ca02c", r"$\Delta p^*$"), ("st", "#9467bd", "St")):
        if not all(m in lad[k]["err_pct"] for k in keys):
            continue
        ax.plot(D, [lad[k]["err_pct"][m] for k in keys], "o-", color=c, label=lab)
    ax.axhspan(-TOL, TOL, color="k", alpha=0.08, label="±5% (matrix)")
    ax.axhline(0, color="k", lw=1)
    ax.axhline(-TASK_TOL_CL, color="r", lw=0.8, ls=":")
    ax.axhline(TASK_TOL_CL, color="r", lw=0.8, ls=":", label="±10% (task Cl gate)")
    ax.set_xlabel("cylinder diameter [lattice cells]")
    ax.set_ylabel("error vs band mid-point [%]")
    ax.set_title("(b) Signed error vs resolution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) convergence span between adjacent levels
    ax = axes[1, 0]
    for m, c, lab in (("cd", "#1f77b4", "C_D"), ("cl_amp", "#d62728", "C_L amplitude"),
                      ("dp", "#2ca02c", r"$\Delta p^*$"), ("st", "#9467bd", "St")):
        rows = sp.get(m) or []
        if not rows:
            continue
        xs = [int(r["pair"].split("->")[1][1:]) for r in rows]
        ax.plot(xs, [r["span_pct"] for r in rows], "o-", color=c, label=lab)
    ax.axhline(SPAN_TOL, color="r", ls="--", lw=1.2, label="3% span criterion")
    ax.set_xlabel("finer level of the pair [cells]")
    ax.set_ylabel("|change| between adjacent levels [%]")
    ax.set_title("(c) Convergence span — does refinement settle?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) H4 window sweep + H3 geometric bias
    ax = axes[1, 1]
    for lv, v in h4.items():
        ws = v["windows"]
        ax.plot([w["n_periods"] for w in ws], [w["err_pct"] for w in ws], "o-",
                label=f"{lv} window sweep")
    ax.axhspan(-TOL, TOL, color="k", alpha=0.08, label="±5% tolerance")
    if h4:
        Ds = sorted((int(k[1:]) for k in audit_h3), key=int)
        bias = [audit_h3[f"D{d}"]["cd_bias_predicted_pct_from_effective_Da"] for d in Ds]
        ax.plot(Ds, bias, "s--", color="gray",
                label=r"H3: half-way BB $D_{eff}=D+1$ bias")
    ax.set_xlabel("averaging window [shedding periods]  /  resolution D [cells]")
    ax.set_ylabel("error [%]")
    ax.set_title("(d) H4 window-length stability vs H3 geometric bias")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"m23014 · DFG Re=100 cylinder · verdict {dec['verdict']} · "
        f"finest {dec['gate']['finest_level']}: "
        f"Cl {dec['gate']['cl_amp_err_pct']:+.2f}%  Cd {dec['gate']['cd_err_pct']:+.2f}%  "
        f"dp {dec['gate']['dp_err_pct']:+.2f}%",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGDIR / "m23014_dfg_hypotheses.png"
    fig.savefig(out, dpi=125)
    plt.close(fig)
    print("figure:", out)

    # second figure: lift history of the finest resolved level (physics, not just numbers)
    make_series_figure(keys, lad)


def make_series_figure(keys, lad) -> None:
    """C_L(t) for the coarsest and the finest level, with the discarded transient shaded."""
    try:
        picks = [k for k in keys if (ART / f"level_D{lad[k]['D_cells']}_u0833.series.npz").exists()]
        picks = [picks[0], picks[-1]] if len(picks) >= 2 else picks
        if not picks:
            return
        fig, axes = plt.subplots(len(picks), 1, figsize=(11, 3.2 * len(picks)), squeeze=False)
        for ax, k in zip(axes[:, 0], picks):
            D = lad[k]["D_cells"]
            z = np.load(ART / f"level_D{D}_u0833.series.npz")
            cl = z["cl"].astype(float)
            cfg = cdfg.CaseConfig(D_cells=D, u_mean_lb=0.0833, n_steps=max(len(cl), 100_000))
            per = cfg.period_steps
            n_tr = int(round(cfg.transient_periods * per))
            t = np.arange(cl.size) / per
            ax.plot(t, cl, lw=0.8, color="#333")
            ax.axvspan(0, n_tr / per, color="tab:red", alpha=0.18,
                       label=f"start-up transient ({cfg.transient_periods:g} periods)")
            ax.axhline(0, color="k", lw=0.6)
            e = lad[k]["err_pct"].get("cl_amp")
            ax.set_title(f"{k}: C_L(t), re-judged amplitude error {e:+.2f}%" if e is not None
                         else f"{k}: C_L(t)")
            ax.set_xlabel("time [shedding periods]")
            ax.set_ylabel("$C_L$")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=0.3)
        fig.suptitle("C_L(t) at the coarsest and finest level — the transient is why "
                     "a whole-record peak-pick overstates the amplitude", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = FIGDIR / "m23014_cl_series.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print("figure:", out)
    except Exception as exc:  # pragma: no cover - figure is best-effort
        print("series figure skipped:", exc)


def main() -> int:
    lad = ladder()
    sp = spans(lad)
    dec = decide(lad, sp)
    h2 = tau_sensitivity(lad)
    audit_json = json.loads((HERE / "h3_h4_audit.json").read_text())
    h4 = audit_json["H4_window_sweep"]
    h3 = audit_json["H3_geometry"]

    payload = {
        "mission": "m23014",
        "case": "CASE-CYLINDER-DRAG-RE100",
        "prereg": json.loads((HERE / "PREREG.json").read_text()),
        "reference_bands": {k: (list(v) if isinstance(v, (tuple, list)) else v)
                        for k, v in BANDS.items()},
        "tolerance": {"matrix_pct": TOL, "task_cl_pct": TASK_TOL_CL, "span_pct": SPAN_TOL},
        "ladder": lad,
        "spans": sp,
        "decision": dec,
        "H2_tau_sensitivity": h2,
        "H2_tau_pair_frozen": audit_json.get("H2_tau_pair_frozen"),
        "H3_geometry": h3,
        "H3_note": (
            "mask area matches pi*r^2 within 1% at every level (so 'the mask is wrong' is "
            "falsified), nominal blockage equals the 0.1/0.41 spec, but half-way bounce-back "
            "puts the hydrodynamic surface at r+0.5 -> effective diameter D+1, an O(1/D) bias "
            "(5.00% at D20 -> 1.79% at D56) that can only be a MONOTONE contributor. The "
            "observed Cl error is non-monotone (17.9 / 17.7 / 2.8), so H3 alone cannot explain it."
        ),
        "H4_transient_decomposition": audit_json["H4_transient_decomposition"],
        "H4_window_sweep": h4,
    }
    out = HERE / "analysis.json"
    out.write_text(json.dumps(payload, indent=2, default=float))

    print(json.dumps(dec, indent=2, default=float))
    if h2:
        print("H2 tau sensitivity:", json.dumps(h2, indent=2, default=float))
    print("\nspans:")
    for m, rows in sp.items():
        print(" ", m, [(r["pair"], round(r["span_pct"], 2)) for r in rows])
    render(lad, sp, dec, h2, h4, h3)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())