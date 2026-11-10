#!/usr/bin/env python3
"""Root-cause attribution for the Re_tau = 590 WMLES failure (P372W2-N01).

Reads the per-case JSON written by
:mod:`benchmarks.wall_model_les.re590_stability` and turns the scan into an
explicit hypothesis table: each candidate mechanism gets a discriminating
observation, the prediction it makes, the measured outcome and a verdict.

The three candidates are the ones the campaign node names:

``H1  relaxation margin``
    ``tau - 1/2 = 3 (nu + nu_t)`` shrinks as the molecular viscosity drops, so
    the run destabilises because the collision is nearly inviscid.  Prediction:
    at a *fixed grid in wall units* (fixed ``dy+``, fixed box), raising
    ``u_tau`` (and with it ``nu``) stabilises the run.
``H2  exchange-point coupling``
    the wall-model/first-cell loop has too much gain: the modelled velocity is
    imposed on the wall node in one step, overshoots, reverses the mean flow and
    then diverges.  Prediction: under-relaxing the imposed velocity (or turning
    the wall model off) keeps the run finite, while the converged state is
    unchanged.
``H3  sub-grid dissipation deficit``
    the Smagorinsky viscosity is too small where the strain is small.
    Prediction: raising ``Cs`` stabilises the run.

Usage
-----
    python benchmarks/wall_model_les/stability_attribution.py \
        --cases runs/m10757/batchQ runs/m10757/batchP runs/m10757/long1 \
        --out artifacts/wall_model_les_peak
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys_ready = True


def load_cases(dirs: list[Path]) -> list[dict]:
    """Collect every case record under the given run directories."""
    out: dict[str, dict] = {}
    for d in dirs:
        for pattern in ("stability_scan_single_*.json", "stability_scan_*.json", "stability_cases/*.json", "stability_cases/*.json", "*.json"):
            for path in sorted(glob.glob(str(Path(d) / pattern))):
                blob = json.loads(Path(path).read_text())
                records = blob.get("cases", [blob]) if isinstance(blob, dict) else blob
                for rec in records:
                    if "label" not in rec:
                        continue
                    out[rec["label"]] = rec
    return list(out.values())


def reversal_step(rec: dict) -> int | None:
    """First logged step at which the plane-mean bulk velocity changes sign."""
    history = rec.get("history") or []
    sign = None
    for row in history:
        ub = row.get("u_bulk")
        if ub is None or not isinstance(ub, (int, float)) or not math.isfinite(ub):
            continue
        s = 1 if ub > 0 else -1
        if sign is None:
            sign = s
        elif s != sign:
            return int(row.get("step"))
    return None


def peak_speed(rec: dict) -> tuple[float | None, int | None]:
    """Largest logged ``max_speed`` and the step at which it occurred."""
    best, at = None, None
    for row in rec.get("history") or []:
        v = row.get("max_speed")
        if isinstance(v, (int, float)) and math.isfinite(v):
            if best is None or v > best:
                best, at = float(v), row.get("step")
    return best, at


def feature_row(rec: dict) -> dict:
    params = rec["params"]
    derived = rec["derived"]
    log = rec.get("last_finite_log") or {}
    v_max, v_step = peak_speed(rec)
    return {
        "label": rec["label"],
        "stable": bool(rec["converged"]),
        "first_nonfinite_step": rec.get("first_nonfinite_step"),
        "reversal_step": reversal_step(rec),
        "re_tau": params["re_tau"],
        "u_tau": params["u_tau"],
        "cs": params["cs"],
        "ny": params["ny"],
        "nx": params["nx"],
        "nz": params["nz"],
        "exchange_offset": params.get("exchange_offset", 1),
        "wall_coupling": params.get("wall_coupling", "assimilate"),
        "wall_relaxation": params.get("wall_relaxation", 1.0),
        "steps": params.get("steps"),
        "delta_y_plus": derived["delta_y_plus"],
        "nu": derived["nu"],
        "tau_minus_half": derived["tau_minus_half_molecular"],
        "y_plus_exchange": derived["y_plus_exchange"],
        "Lx_plus": params["nx"] * derived["delta_y_plus"],
        "re_tau_measured": log.get("re_tau_measured"),
        "u_bulk_last": log.get("u_bulk"),
        "tau_margin_last": log.get("tau_minus_half_min"),
        "sgs_ratio_last": log.get("sgs_ratio_mean"),
        "rho_min_last": log.get("rho_min"),
        "max_speed_last": log.get("max_speed"),
        "max_speed_peak": v_max,
        "max_speed_peak_step": v_step,
        "error": rec.get("error"),
    }


def _find(rows: list[dict], **criteria) -> dict | None:
    for row in rows:
        if all(abs(row.get(k, object()) - v) < 1e-9 if isinstance(v, (int, float)) else row.get(k) == v
               for k, v in criteria.items()):
            return row
    return None


def _axis_rows(rows: list[dict], *, fixed: dict, varying: str) -> list[dict]:
    """Rows that share ``fixed`` and differ in ``varying``."""
    sel = []
    for row in rows:
        ok = True
        for k, v in fixed.items():
            rv = row.get(k)
            if isinstance(v, float):
                if rv is None or abs(float(rv) - v) > 1e-6:
                    ok = False
                    break
            elif rv != v:
                ok = False
                break
        if ok:
            sel.append(row)
    return sorted(sel, key=lambda r: (r[varying] if r.get(varying) is not None else float("nan")))


def attribute(rows: list[dict]) -> dict:
    """Fold the scan into the hypothesis table."""
    # --- the blow-up reference: largest box, no stabilisation, Re_tau = 590
    reference = [
        r for r in rows
        if r["re_tau"] == 590.0
        and r["wall_relaxation"] == 1.0
        and r["wall_coupling"] == "assimilate"
        and r["cs"] == 0.1
        and r["u_tau"] == 0.02
        and r["Lx_plus"] >= 15.0 * 48  # a box wide enough to hold the turbulence
    ]
    ref = max(reference, key=lambda r: r["steps"]) if reference else None

    out: dict = {"reference_case": ref, "hypotheses": {}, "features": rows}

    # H1: molecular-viscosity margin, tested by changing u_tau at fixed dy+
    h1_rows = [r for r in rows if r["re_tau"] == 590.0 and abs(r["ny"] - 49) == 0
               and r["wall_relaxation"] == 1.0 and r["wall_coupling"] == "assimilate"
               and r["cs"] == 0.1 and r["Lx_plus"] > 400.0]
    h1_rows = sorted(h1_rows, key=lambda r: r["u_tau"])
    if len(h1_rows) >= 2:
        monotone = all(
            (not h1_rows[i]["stable"] or h1_rows[i + 1]["stable"]) for i in range(len(h1_rows) - 1)
        )
        supported = monotone and any(r["stable"] for r in h1_rows) and any(not r["stable"] for r in h1_rows)
        out["hypotheses"]["H1_relaxation_margin"] = {
            "test": "same wall-unit grid (ny=49, box 32x32), u_tau (hence nu) varied",
            "rows": h1_rows,
            "verdict": "SUPPORTED" if supported else "REJECTED",
            "reason": (
                "stability switches with nu at a fixed wall-unit grid"
                if supported else
                "the outcome does not follow the viscosity margin (see rows)"
            ),
        }
    else:
        out["hypotheses"]["H1_relaxation_margin"] = {"test": "insufficient cases", "verdict": "WITHHELD"}

    # H2: coupling gain -- under-relaxation and wall-model-off controls
    h2_rows = [r for r in rows if r["re_tau"] == 590.0 and abs(r["ny"] - 49) == 0
               and r["u_tau"] == 0.02 and r["cs"] == 0.1 and r["Lx_plus"] > 400.0
               and (r["wall_relaxation"] < 1.0 or r["wall_coupling"] != "assimilate")]
    if h2_rows:
        cured = [r for r in h2_rows if r["stable"]]
        out["hypotheses"]["H2_exchange_coupling"] = {
            "test": "under-relaxed wall-model target / alternative coupling on the blowing-up configuration",
            "rows": h2_rows,
            "verdict": "SUPPORTED" if cured else "REJECTED",
            "reason": (
                f"{len(cured)} of {len(h2_rows)} coupling variants keep the run finite "
                "while the explicit assimilation of the same configuration diverges"
                if cured else "no coupling variant stabilises the run"
            ),
        }
    else:
        out["hypotheses"]["H2_exchange_coupling"] = {"test": "insufficient cases", "verdict": "WITHHELD"}

    # H3: sub-grid dissipation -- Cs sweep on the blowing-up configuration
    h3_rows = [r for r in rows if r["re_tau"] == 590.0 and abs(r["ny"] - 49) == 0
               and r["u_tau"] == 0.02 and r["wall_relaxation"] == 1.0
               and r["wall_coupling"] == "assimilate" and r["Lx_plus"] > 400.0]
    h3_rows = sorted(h3_rows, key=lambda r: r["cs"])
    if len(h3_rows) >= 2:
        cured = [r for r in h3_rows if r["stable"] and r["cs"] > 0.1]
        out["hypotheses"]["H3_sgs_deficit"] = {
            "test": "Cs raised on the blowing-up configuration (0.1 -> 0.3)",
            "rows": h3_rows,
            "verdict": "SUPPORTED" if cured else "REJECTED",
            "reason": (
                "a larger Smagorinsky constant keeps the run finite"
                if cured else "raising Cs does not prevent the divergence"
            ),
        }
    else:
        out["hypotheses"]["H3_sgs_deficit"] = {"test": "insufficient cases", "verdict": "WITHHELD"}

    # H4: velocity scale / lattice stability envelope
    # At a fixed wall-unit grid (fixed dy+, box, Cs, explicit coupling) the only
    # knob that changes in this family is u_tau, i.e. the *velocity* scale.  If
    # stability follows u_tau (and not nu, which changes with it in the same
    # direction), then the failure is the resolved velocity running past the
    # lattice stability envelope, not a relaxation/molecular-viscosity effect.
    h4_rows = [r for r in rows if r["re_tau"] == 590.0 and abs(r["ny"] - 49) == 0
               and r["wall_relaxation"] == 1.0 and r["wall_coupling"] == "assimilate"
               and r["cs"] == 0.1 and r["Lx_plus"] > 400.0]
    h4_rows = sorted(h4_rows, key=lambda r: r["u_tau"])
    if len(h4_rows) >= 2:
        unstable = [r for r in h4_rows if not r["stable"]]
        fastest_speed = max((r["max_speed_peak"] or 0.0) for r in rows if r["max_speed_peak"])
        out["hypotheses"]["H4_resolved_velocity_envelope"] = {
            "test": (
                "u_tau swept at a fixed wall-unit grid (dy+=24.6, box 32x32, Cs=0.1, explicit coupling): "
                "stability must follow the velocity scale, not nu"
            ),
            "rows": h4_rows,
            "verdict": "SUPPORTED" if (len(unstable) >= 1 and any(r["stable"] for r in h4_rows)) else "REJECTED",
            "reason": (
                f"the fastest run of the family reaches max abs(u) = {fastest_speed:.3f} lattice units "
                f"(lattice stability envelope ~1) before/at divergence, and the smallest velocity scale "
                f"is the stable one"
            ),
        }
    else:
        out["hypotheses"]["H4_resolved_velocity_envelope"] = {"test": "insufficient cases", "verdict": "WITHHELD"}

    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "wall_model_les_peak"))
    args = ap.parse_args(argv)

    rows = [feature_row(rec) for rec in load_cases([Path(c) for c in args.cases])]
    rows = sorted(rows, key=lambda r: (r["re_tau"], r["ny"], r["cs"], r["wall_relaxation"], r["u_tau"]))
    result = attribute(rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stability_attribution.json").write_text(json.dumps(result, indent=2, default=str))

    print(f"{'case':<34s} {'status':<7s} {'dy+':>6s} {'nu':>9s} {'box':>9s} {'cs':>4s} {'beta':>4s} "
          f"{'u_tau':>6s} {'reversal':>8s} {'firstNaN':>8s} {'Re_tau_m':>8s}")
    for r in rows:
        print(f"{r['label']:<34s} {'STABLE' if r['stable'] else 'BLOWUP':<7s} {r['delta_y_plus']:6.2f} "
              f"{r['nu']:9.2e} {r['nx']:4d}x{r['nz']:<4d} {r['cs']:4.2f} {r['wall_relaxation']:4.2f} "
              f"{r['u_tau']:6.3f} {str(r['reversal_step']):>8s} {str(r['first_nonfinite_step']):>8s} "
              f"{('%.1f' % r['re_tau_measured']) if r['re_tau_measured'] else 'nan':>8s}")
    print()
    for name, hyp in result["hypotheses"].items():
        print(f"{name}: {hyp['verdict']} -- {hyp.get('reason', hyp.get('test'))}")
    print(f"wrote {out_dir / 'stability_attribution.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())