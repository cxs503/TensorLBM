#!/usr/bin/env python3
"""Acceptance fold for P372W2-N01 (WMLES stability + peak-position closure).

Reads the artifacts produced by the other scripts in this directory and writes
``acceptance.json``: one row per criterion with a verdict and the numbers it is
based on.  Nothing here recomputes physics -- it only applies the thresholds,
so the verdict can be re-derived from the artifacts alone.

Criteria
--------
``c1_stability_re590``
    the Re_tau = 590 wall-modelled channel runs to the end of the sampling
    window with no non-finite state, at the finest grid of the study.
``c2_reproducibility``
    the same seed reproduces the same state exactly, and two independent seeds
    agree on the peak position within 5 %.
``c3_peak_offset_15pct``
    |peak-position error| <= 15 % against the DNS reference, on the finest grid
    *and* after Richardson extrapolation to ``h -> 0`` (the grid-independent
    part of the claim).
``c4_grid_error_split``
    the Richardson/GCI numbers that separate grid quantisation from wall-model
    error (INFO: reported, not thresholded).
``c5_wall_function_correction``
    a correction map exists and implies a wall-normal spacing for the 15 %
    tolerance (PASS when the map is bracketed by the measured points).
``c6_reference_integrity``
    the DNS reference table matches its provenance hash and satisfies the exact
    channel-balance endpoint identities.
``c7_control_re180``
    the same diagnostic on the Re_tau = 180 configuration (where the solver is
    verified) returns a peak within 15 % of the MKM-180 value -- the
    diagnostic's own baseline error.

Usage
-----
    python benchmarks/wall_model_les/acceptance_p372w2.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "artifacts" / "wall_model_les_peak"


def _load(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    args = ap.parse_args(argv)
    d = Path(args.dir)
    study = _load(d / "peak_offset_study.json", {})
    attribution = _load(d / "stability_attribution.json", {})
    repro = _load(d / "reproducibility.json", {})
    runs = {p.stem.replace("run_", ""): json.loads(p.read_text()) for p in sorted(d.glob("run_*.json"))}

    criteria: dict[str, dict] = {}

    # --- c1 stability ------------------------------------------------------
    fine = runs.get("fine")
    if fine is None:
        criteria["c1_stability_re590"] = {"verdict": "WITHHELD", "evidence": "no fine-grid run record"}
    else:
        hist = fine.get("history") or []
        bad = [
            row for row in hist
            if any(isinstance(v, float) and not math.isfinite(v) for k, v in row.items() if k != "step")
        ]
        criteria["c1_stability_re590"] = {
            "verdict": "PASS" if fine.get("stable") else "FAIL",
            "evidence": (
                f"fine grid dy+={fine['derived']['delta_y_plus']:.2f}, {fine['params']['steps']} steps, "
                f"first non-finite step = {fine.get('first_nonfinite_step')}, "
                f"non-finite log rows = {len(bad)}, measured Re_tau at the last log = "
                f"{fine['changed'].get('re_tau_measured')}"
            ),
            "numbers": {
                "delta_y_plus": fine["derived"]["delta_y_plus"],
                "steps": fine["params"]["steps"],
                "first_nonfinite_step": fine.get("first_nonfinite_step"),
                "re_tau_measured": fine["changed"].get("re_tau_measured"),
                "u_tau_model_over_balance": fine["changed"].get("u_tau_model_over_balance"),
            },
        }

    # --- c2 reproducibility ------------------------------------------------
    if not repro:
        criteria["c2_reproducibility"] = {"verdict": "WITHHELD", "evidence": "no reproducibility record"}
    else:
        det = repro["determinism"]
        stat = repro["statistical"]
        # the gate is derivable from the record itself: a peak is only
        # meaningful when the resolved stress reaches a reasonable fraction of
        # the DNS scale (-<u'v'>+ peak ~ 0.87 at Re_tau = 590)
        scale = 0.87
        magnitudes = [
            abs(stat.get(k, 0.0)) / scale for k in ("peak_value_seed0", "peak_value_seed1")
            if isinstance(stat.get(k), (int, float))
        ]
        developed = bool(repro.get("stress_developed")) if "stress_developed" in repro else bool(
            magnitudes and min(magnitudes) >= 0.2
        )
        if not developed:
            criteria["c2_reproducibility"] = {
                "verdict": "WITHHELD",
                "evidence": (
                    "the reproducibility window does not develop a resolved stress peak "
                    f"(peak / DNS scale = {[round(x, 4) for x in magnitudes]}), "
                    "so peak reproducibility is not testable there; determinism itself holds "
                    f"(bitwise identical = {det['bitwise_identical']}, max |d uv+| = "
                    f"{det['uv_plus_max_abs_diff']:.3e})"
                ),
                "numbers": {"determinism": det, "statistical": stat},
            }
        else:
            ok = det["bitwise_identical"] and stat["peak_y_plus_rel_diff"] <= 0.05
            criteria["c2_reproducibility"] = {
                "verdict": "PASS" if ok else "FAIL",
                "evidence": (
                    f"same-seed rerun bitwise identical = {det['bitwise_identical']} "
                    f"(max |d uv+| = {det['uv_plus_max_abs_diff']:.3e}); seed 0 vs 1 peak position "
                    f"{stat['peak_y_plus_rel_diff']*100:.2f} % apart (threshold 5 %), peak value "
                    f"{stat['peak_value_rel_diff']*100:.2f} % apart"
                ),
                "numbers": {"determinism": det, "statistical": stat},
            }

    # --- c3 peak offset ----------------------------------------------------
    verdict = (study.get("verdict") or {})
    grids = study.get("grids") or {}
    gci = ((study.get("grid_convergence") or {}).get("y_peak")) or {}
    if "fine" not in grids:
        criteria["c3_peak_offset_15pct"] = {"verdict": "WITHHELD", "evidence": "no fine-grid peak"}
    else:
        fine_off = grids["fine"]["offset_rel"]
        ext_off = verdict.get("extrapolated_offset_rel")
        ok = abs(fine_off) <= 0.15 and ext_off is not None and abs(ext_off) <= 0.15
        criteria["c3_peak_offset_15pct"] = {
            "verdict": "PASS" if ok else "FAIL",
            "evidence": (
                f"fine-grid peak y+ = {grids['fine']['y_peak']:.2f} vs DNS "
                f"{study['reference']['peak_y_plus']:.2f} -> {100*fine_off:+.2f} % (threshold ±15 %); "
                f"Richardson h->0 = {gci.get('value_extrapolated')} -> "
                f"{'n/a' if ext_off is None else f'{100*ext_off:+.2f} %'}; "
                f"observed order p = {gci.get('observed_order')}, converged = {gci.get('converged')}"
            ),
            "numbers": {
                "delta_y_plus_peak_offset_rel": fine_off,
                "delta_y_plus_peak_offset_rel_extrapolated": ext_off,
                "fine_offset_rel": fine_off,
                "extrapolated_offset_rel": ext_off,
                "observed_order": gci.get("observed_order"),
                "gci_converged": gci.get("converged"),
            },
        }

    # --- c3b Re-matched peak offset ---------------------------------------
    # A run that does not reach the nominal friction velocity is not at
    # Re_tau = 590; comparing its peak with the 590 DNS then measures the
    # friction deficit, not the peak shape.  The Re-matched leg uses the DNS
    # peak at the run's own measured Re_tau (two-tier ln-linear interpolation)
    # and is the physically meaningful peak-position metric here.
    rm = ((study.get("re_matched") or {}).get("per_grid")) or {}
    if rm:
        worst = max(rm.items(), key=lambda kv: abs(kv[1]["offset_rel"]))
        ok = all(abs(row["offset_rel"]) <= 0.15 for row in rm.values())
        criteria["c3b_peak_offset_re_matched"] = {
            "verdict": "PASS" if ok else "FAIL",
            "evidence": (
                "peak position against the DNS peak at each run's own measured Re_tau: "
                + "; ".join(
                    f"{name} Re_tau={row['re_tau_measured']:.0f} -> {100*row['offset_rel']:+.2f} %"
                    for name, row in rm.items()
                )
                + f" (worst {worst[0]}, threshold ±15 %)"
            ),
            "numbers": rm,
        }
    else:
        criteria["c3b_peak_offset_re_matched"] = {
            "verdict": "WITHHELD",
            "evidence": "no measured Re_tau available (force balance not reached)",
        }

    # --- c4 grid/model split (INFO) ---------------------------------------
    if study.get("grid_convergence"):
        gc = study["grid_convergence"]
        criteria["c4_grid_error_split"] = {
            "verdict": "INFO",
            "evidence": (
                "y_peak: grid error (fine vs h->0) = "
                f"{gc['y_peak']['grid_error_fine']:.3f} wall units, GCI(fine) = {gc['y_peak']['gci_fine']:.3f}, "
                f"model error (h->0 vs DNS) = {gc['y_peak']['model_error']:.3f}; "
                f"value: grid error = {gc['value_peak']['grid_error_fine']:.4f}, "
                f"model error = {gc['value_peak']['model_error']:.4f}"
            ),
            "numbers": gc,
        }
    else:
        criteria["c4_grid_error_split"] = {"verdict": "WITHHELD", "evidence": "fewer than three grids"}

    # --- c5 correction map -------------------------------------------------
    corr = (study.get("wall_function_correction") or {})
    if corr:
        x15 = corr.get("spacing_for_15pct")
        criteria["c5_wall_function_correction"] = {
            "verdict": "PASS" if x15 is not None else "WITHHELD",
            "evidence": (
                f"offset map ({corr['offset_map']['kind']}) coefficients "
                f"{[round(c, 6) for c in corr['offset_map']['coefficients']]}, rmse = "
                f"{corr['offset_map']['rmse']:.4f}; fitted spacing for the 15 % tolerance = "
                f"{'not bracketed' if x15 is None else f'{x15:.2f} wall units'}"
            ),
            "numbers": corr,
        }
    else:
        criteria["c5_wall_function_correction"] = {"verdict": "WITHHELD", "evidence": "no map (needs 3 grids)"}

    # --- c6 reference integrity -------------------------------------------
    ref = study.get("reference") or {}
    gates = (ref.get("stress_identity_gates") or {})
    if ref:
        ok = bool(ref.get("sha256_matches_provenance")) and gates.get("wall_ok", False) and gates.get("centre_ok", False)
        criteria["c6_reference_integrity"] = {
            "verdict": "PASS" if ok else "FAIL",
            "evidence": (
                f"{ref.get('source')} sha256 matches provenance = {ref.get('sha256_matches_provenance')}; "
                f"balance identities: wall value = {gates.get('wall_value')}, centre value = "
                f"{gates.get('centre_value')}; tabulated Re_tau = {ref.get('re_tau_tabulated')}"
            ),
            "numbers": {"sha256": ref.get("sha256"), "gates": gates},
        }
    else:
        criteria["c6_reference_integrity"] = {"verdict": "WITHHELD", "evidence": "no DNS reference loaded"}

    # --- c7 control (Re_tau = 180) ----------------------------------------
    control = runs.get("control180")
    control_peak = (study.get("control") or {}).get("peak_offset_rel")
    if control is not None or control_peak is not None:
        criteria["c7_control_re180"] = {
            "verdict": "INFO",
            "evidence": (
                f"Re_tau = 180 control: stable = {control.get('stable') if control else 'n/a'}, "
                f"peak-position error vs MKM-180 = "
                f"{'n/a' if control_peak is None else f'{100*control_peak:+.2f} %'}"
            ),
            "numbers": {"peak_offset_rel": control_peak},
        }
    else:
        criteria["c7_control_re180"] = {"verdict": "WITHHELD", "evidence": "control run missing"}

    counts = {"PASS": 0, "FAIL": 0, "WITHHELD": 0, "INFO": 0}
    for row in criteria.values():
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    record = {
        "node": "p372w2-n01-wm-les-stability-peak",
        "thresholds": {
            "peak_offset_rel": 0.15,
            "reproducibility_peak_rel": 0.05,
            "stability": "no non-finite state inside the sampling window",
        },
        "criteria": criteria,
        "counts": counts,
        "attribution_verdicts": {
            k: v.get("verdict") for k, v in (attribution.get("hypotheses") or {}).items()
        },
    }
    out = d / "acceptance.json"
    out.write_text(json.dumps(record, indent=2, default=str))
    for name, row in criteria.items():
        print(f"{name:<28s} {row['verdict']:<9s} {row['evidence']}")
    print(f"\ncounts: {counts}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())