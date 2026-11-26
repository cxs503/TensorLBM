"""R4 analysis for anchor XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS.

The question R2/R3 left open
----------------------------
R2 ran a resolution ladder a = 16/24/32/48/64 at a *fixed lattice* relaxation
time (tau = 1) and reported a non-monotone error: 28.16 % (a = 16), 13.81 %
(a = 32), then rising again at a = 48/64.  R2 read that as "no convergence" and
fell back on a Richardson extrapolation.  R3 repeated the a = 16 rung and
changed only the delivery form.  Neither round asked the prior question:

    at fixed tau the *lattice* viscosity nu = (tau - 1/2)/3 is fixed, while the
    depth h0 = 2a and the gravity-wave speed U0 = sqrt(g h0) both grow with a,
    so the Reynolds number Re = U0 h0 / nu grows like a**(3/2).

The "resolution ladder" is therefore also a Reynolds sweep (Re = 7.68 / 14.1 /
21.7 / 39.9 / 61.4 for a = 16..64).  A dam-break front is a balance between
gravity-driven inertia and viscous drag, so a monotone-in-a error is not
expected at all: with every rung a different Re, E(a) cannot be read as
discretisation error, and the H1 extrapolation E(a) -> E_inf is fitted to a
curve that has no reason to be a function of a alone.

What this module does (all from artifacts, no solver time)
----------------------------------------------------------
* Reynolds number per run, from the run's own (a, tau) -- ``R.reynolds``.
* Signed front error per sampled T against Martin & Moyce (1952), so the sign
  of the largest deviation is reported instead of an unsigned maximum.
* Distance from the front to the downstream wall, in units of a -- the declared
  domain is only 5a wide, so at high a the late-window front comes close to the
  wall and the comparison is contaminated by reflection.
* ``collapse``: E(Re) over the ladder and the a = 32 tau sweep pooled together.
  If both families fall on one curve, the error is set by Re (physics mismatch
  of the comparison), not by the grid.
* ``fixed_re_trio``: a = 24/32/48 with tau from ``R.tau_for_re`` so all three
  run at the *same* Re = Re(a = 32, tau = 1) = 21.72.  Flat E across the trio
  is the decisive test: it isolates the discretisation from the Re effect.
* ``width_control``: a = 64 at nx = 5a versus nx = 8a -- separates the
  late-window overshoot from downstream-wall reflection.

Salvaged data
-------------
``artifacts/recovered_cj3126_tau_sweep.json`` holds the max |err| scalars of
job cj3126's tau sweep (that job died in its analysis stage and lost the per-T
series); it is used only for magnitude, never for sign, and the re-run cj3218
supersedes it wherever both exist.

Usage
    python benchmarks/xval_oracle_minimal_d2q9_fs/reference_study/re_confound.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import resolution_study as R  # noqa: E402  (metric/Re definitions live there)

ART = HERE / "artifacts"
KEY_PRIMARY = "Z_sub0.05"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_runs() -> list[dict]:
    """All per-run artifacts that carry a usable in-window series."""
    runs = []
    for f in sorted((ART / "runs_r3").glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if r.get("status") != "ok" or not r.get("series"):
            continue
        r["source"] = f.name
        runs.append(r)
    return runs


def load_salvaged() -> list[dict]:
    f = ART / "recovered_cj3126_tau_sweep.json"
    if not f.exists():
        return []
    return json.loads(f.read_text()).get("runs", [])


def metrics(r: dict) -> dict:
    """Signed in-window error of one run + wall clearance of the final front."""
    a = float(r["a"])
    tau = float(r["tau"])
    w = float(r.get("width_in_a", 5.0) or 5.0)
    lo, hi = R.DESIGN["window_T"]
    rows = []
    for s in r["series"]:
        T = float(s["T"])
        if not (lo - 1e-9 <= T <= hi + 1e-9):
            continue
        zref = R.mm_interp(T)
        if zref is None:
            continue
        rows.append(
            {
                "T": round(T, 4),
                "Z_ref": round(float(zref), 4),
                "Z_cell": round(float(s["Z_cell"]), 4),
                "Z_sub": round(float(s[KEY_PRIMARY]), 4),
                "e_cell_pct": round(100.0 * (s["Z_cell"] - zref) / zref, 3),
                "e_sub_pct": round(100.0 * (s[KEY_PRIMARY] - zref) / zref, 3),
            }
        )
    out = {
        "label": r.get("label") or R.result_label(r),
        "a": a,
        "tau": tau,
        "width_in_a": w,
        "nx": r.get("nx"),
        "ny": r.get("ny"),
        "steps": r.get("steps"),
        "re": round(R.reynolds(a, tau), 4),
        "nu": round((tau - 0.5) / 3.0, 6),
        "u_max": round(float(r["u_max"]), 5) if r.get("u_max") is not None else None,
        "source": r.get("source", "recovered"),
        "points": rows,
    }
    if rows:
        e = np.array([p["e_sub_pct"] for p in rows])
        i = int(np.argmax(np.abs(e)))
        out.update(
            {
                "max_abs_err_pct": round(float(abs(e[i])), 3),
                "signed_err_at_max_pct": round(float(e[i]), 3),
                "T_at_max": rows[i]["T"],
                "signed_err_end_pct": rows[-1]["e_sub_pct"],
                "Z_cell_end": rows[-1]["Z_cell"],
                "T_end": rows[-1]["T"],
                "late_signed_mean_pct": round(
                    float(e[np.array([p["T"] for p in rows]) >= 2.0 - 1e-9].mean()), 3
                )
                if any(p["T"] >= 2.0 - 1e-9 for p in rows)
                else None,
                "pass_15pct": bool(abs(e[i]) <= R.DESIGN["thresholds_pct"]["z_max_rel_err"]),
                "pass_3pct": bool(abs(e[i]) <= R.DESIGN["thresholds_pct"]["asymptotic_target"]),
            }
        )
        if r.get("nx"):
            out["wall_clearance_a"] = round((float(r["nx"]) - 2.0 - rows[-1]["Z_cell"] * a) / a, 3)
    return out


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def widening_pairs(runs: list[dict]) -> dict:
    """Same (a, tau) at two domain widths -> isolates downstream-wall effects."""
    by = {}
    for m in runs:
        by.setdefault((m["a"], m["tau"]), []).append(m)
    out = {}
    for (a, tau), ms in sorted(by.items()):
        ws = sorted({m["width_in_a"] for m in ms})
        if len(ws) < 2:
            continue
        out[f"a{a}_tau{tau:g}"] = {
            "widths": ws,
            "max_abs_err_pct": {m["width_in_a"]: m["max_abs_err_pct"] for m in ms},
            "signed_err_end_pct": {m["width_in_a"]: m["signed_err_end_pct"] for m in ms},
            "wall_clearance_a": {m["width_in_a"]: m.get("wall_clearance_a") for m in ms},
        }
    return out


def zero_crossing(pooled: list[dict], key: str = "late_signed_mean_pct") -> dict:
    """First Re at which the signed late-window error changes sign (linear interp)."""
    pts = sorted((m["re"], m[key], m["label"]) for m in pooled if m.get(key) is not None)
    for (r0, e0, l0), (r1, e1, l1) in zip(pts, pts[1:]):
        if e0 == 0 or e0 * e1 < 0:
            re_star = r0 - e0 * (r1 - r0) / (e1 - e0)
            return {
                "re_star": round(float(re_star), 3),
                "bracket": [l0, l1],
                "bracket_re": [r0, r1],
                "bracket_err_pct": [e0, e1],
            }
    return {"re_star": None, "reason": "no sign change in the available Re range"}


def analyze(runs: list[dict], salvaged: list[dict]) -> dict:
    allruns = [metrics(r) for r in runs]
    # a run with no sample inside the declared window carries no error and must
    # not enter any error aggregate (cj3126 regression)
    ms = [m for m in allruns if m.get("points")]
    ladder = sorted(
        [m for m in ms if m["tau"] == 1.0 and m["width_in_a"] == 5.0], key=lambda m: m["a"]
    )
    sweeps = sorted(
        [m for m in ms if m["a"] == R.DESIGN["tau_sweep"]["a"] and m["width_in_a"] == 5.0],
        key=lambda m: m["tau"],
    )
    wide = [m for m in ms if m["width_in_a"] != 5.0]
    re0 = R.re_reference()
    tol = 0.02 * re0
    trio = sorted(
        [m for m in ms if m["width_in_a"] == 5.0 and abs(m["re"] - re0) <= tol],
        key=lambda m: m["a"],
    )
    pooled = ladder + [m for m in sweeps if all(m["label"] != x["label"] for x in ladder)]
    out = {
        "design": {
            "re_reference": round(re0, 4),
            "re_reference_rung": "a=32, tau=1",
            "re_formula": "Re = sqrt(g*h0)*h0/nu, h0=2a, nu=(tau-0.5)/3",
            "window_T": R.DESIGN["window_T"],
            "thresholds_pct": R.DESIGN["thresholds_pct"],
            "primary_metric": KEY_PRIMARY,
        },
        "runs": allruns,
        "ladder_tau1_w5": ladder,
        "tau_sweep_a32_w5": sweeps,
        "fixed_re_trio": trio,
        "width_control": wide,
        "widening_pairs": widening_pairs(ms),
        "salvaged_cj3126": salvaged,
    }
    # pooled E(Re) collapse
    out["collapse"] = [
        {
            "label": m["label"],
            "re": m["re"],
            "max_abs_err_pct": m["max_abs_err_pct"],
            "signed_err_at_max_pct": m["signed_err_at_max_pct"],
            "late_signed_mean_pct": m.get("late_signed_mean_pct"),
        }
        for m in sorted(pooled, key=lambda m: m["re"])
    ]
    out["zero_crossing"] = zero_crossing(pooled)
    # spread of the error at fixed Re vs the spread over the whole ladder
    concl = {}
    if len(trio) >= 2:
        vals = [m["max_abs_err_pct"] for m in trio]
        concl["fixed_re_trio_spread_pct_points"] = round(max(vals) - min(vals), 3)
        concl["fixed_re_trio_a"] = [m["a"] for m in trio]
        concl["fixed_re_trio_re"] = [m["re"] for m in trio]
        if len(ladder) >= 2:
            lv = [m["max_abs_err_pct"] for m in ladder]
            concl["ladder_spread_pct_points"] = round(max(lv) - min(lv), 3)
            concl["trio_over_ladder_spread_ratio"] = round(
                (max(vals) - min(vals)) / max(max(lv) - min(lv), 1e-12), 3
            )
        if len(trio) == 3:
            concl["H2_fixed_re_flat"] = bool(max(vals) - min(vals) <= 3.0)
            concl["H2_rule"] = (
                "H2 (residual error is Re mismatch, not discretisation) "
                "is supported if the a-spread at matched Re is <= 3 "
                "percentage points"
            )
    if out["widening_pairs"]:
        concl["width_effect"] = out["widening_pairs"]
    total = [m for m in ms if ms]
    if total:
        concl["all_runs_pass_15pct"] = all(m["pass_15pct"] for m in total)
        concl["n_runs"] = len(total)
        best = min(total, key=lambda m: m["max_abs_err_pct"])
        concl["best_run"] = {
            "label": best["label"],
            "re": best["re"],
            "max_abs_err_pct": best["max_abs_err_pct"],
        }
    out["conclusions"] = concl
    return out


def render_md(an: dict) -> str:
    L = ["# R4 — Re-confound analysis of the a-ladder (programmatic dump)", ""]
    d = an["design"]
    L.append(
        f"* Re = sqrt(g h0) h0 / nu, h0 = 2a, nu = (tau-1/2)/3; "
        f"Re reference (a=32, tau=1) = **{d['re_reference']}**"
    )
    L.append(
        f"* window T = {d['window_T']}, primary metric = {d['primary_metric']}, "
        f"gates = {d['thresholds_pct']}"
    )
    L.append("* reference: Martin & Moyce (1952), 17 digitised points")
    L.append("")
    L.append("## Runs (signed error sign is reported; a '<' zero run lags the experiment)")
    L.append("")
    L.append(
        "| label | a | tau | Re | max \\|err\\| % | signed err @ max % | T@max | err @ T_end % | "
        "late mean % | wall clearance (a) |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for m in an["runs"]:
        if "max_abs_err_pct" not in m:  # no sample inside the declared window
            L.append(
                f"| {m['label']} | {m['a']:g} | {m['tau']:g} | {m['re']:.2f} | "
                f"n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        L.append(
            f"| {m['label']} | {m['a']:g} | {m['tau']:g} | {m['re']:.2f} | "
            f"{m['max_abs_err_pct']:.2f} | {m['signed_err_at_max_pct']:+.2f} | "
            f"{m['T_at_max']:.2f} | {m['signed_err_end_pct']:+.2f} | "
            f"{m.get('late_signed_mean_pct')} | {m.get('wall_clearance_a')} |"
        )
    L.append("")
    L.append("## E(Re) collapse (ladder tau=1 + a=32 tau sweep)")
    L.append("")
    L.append("| Re | run | max \\|err\\| % | signed @ max % | late mean % |")
    L.append("|---|---|---|---|---|")
    for c in an["collapse"]:
        L.append(
            f"| {c['re']:.2f} | {c['label']} | {c['max_abs_err_pct']:.2f} | "
            f"{c['signed_err_at_max_pct']:+.2f} | {c.get('late_signed_mean_pct')} |"
        )
    L.append("")
    zc = an["zero_crossing"]
    L.append(
        f"* zero crossing of the signed late-window error: Re* = {zc.get('re_star')} "
        f"({zc.get('bracket_re')}, {zc.get('bracket_err_pct')})"
    )
    L.append("")
    L.append("## Fixed-Re trio (decisive resolution test)")
    L.append("")
    if an["fixed_re_trio"]:
        L.append("| a | tau | Re | max \\|err\\| % | signed @ max % |")
        L.append("|---|---|---|---|---|")
        for m in an["fixed_re_trio"]:
            L.append(
                f"| {m['a']:g} | {m['tau']:g} | {m['re']:.2f} | "
                f"{m['max_abs_err_pct']:.2f} | {m['signed_err_at_max_pct']:+.2f} |"
            )
    else:
        L.append("* (no run yet at Re = Re_ref)")
    L.append("")
    L.append("## Domain-width control")
    L.append("")
    if an["widening_pairs"]:
        for k, v in an["widening_pairs"].items():
            L.append(
                f"* {k}: max|err| by width {v['max_abs_err_pct']}, "
                f"signed end error {v['signed_err_end_pct']}, "
                f"wall clearance {v['wall_clearance_a']}"
            )
    else:
        L.append("* (no widened run yet)")
    L.append("")
    L.append("## Conclusions (computed)")
    L.append("")
    for k, v in an["conclusions"].items():
        L.append(f"* `{k}` = {v}")
    L.append("")
    return "\n".join(L)


def main() -> int:
    runs = load_runs()
    salvaged = load_salvaged()
    an = analyze(runs, salvaged)
    (ART / "re_analysis.json").write_text(json.dumps(an, indent=1))
    (ART / "re_numbers.md").write_text(render_md(an))
    print(f"[r4] {len(runs)} run artifacts -> artifacts/re_analysis.json + re_numbers.md")
    for k, v in an["conclusions"].items():
        print(f"  {k} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
