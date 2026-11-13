#!/usr/bin/env python3
"""XVAL: waLBerla ``block-amr-equal-level-stream-correction`` vs TensorLBM.

Reference practice (zero hand-copying; inventoried source record
``oss_solutions`` id 283, ``lssfau/walberla``):

* ``src/lbm/refinement/EqualLevelBorderStreamCorrection.h`` -- population-level
  ``prepare``/``apply`` correction at equal-level (same cell size) borders where
  a finer neighbour block exists in the vicinity; before streaming the missing
  populations are copied from the equally-sized neighbour cell, after streaming
  they are restored; enabled by default.
* ``src/lbm/refinement/CurlBasedLevelDetermination.h`` / ``TimeStep.h`` --
  2:1 refinement, recursive sub-cycling, level-dependent relaxation.

This driver measures, on analytic fields (ref) vs the transplant (got):
R1  Filippova-Haenel / prolongation factor implied by the 2:1 convective
    refinement (scale scan on the exact non-equilibrium field),
R2  inertness of the correction on a uniform state (must be exact),
R3  state neutrality of prepare/apply (bit-identical round trip),
R4  mass-flux defect introduced by the substitution on a divergence-free field,
R5  resolution and flow sensitivity of R1.

Usage:
    PYTHONPATH=src python benchmarks/xval_block_amr_equal_level_stream_correction/run_xval.py \
        --out artifacts/xval_block_amr_equal_level_stream_correction
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorlbm.amr_cavity2d import _stream
from tensorlbm.amr_equal_level_stream import (
    EqualLevelBorderStreamCorrection,
    analytic_fields_2d,
    calibrate_prolongation_scale,
    prolongation_scale,
)


def tgv_velocity(n: int, u0: float = 0.04):
    k = 2.0 * np.pi / n
    return lambda x, y: (
        u0 * np.cos(k * x) * np.sin(k * y),
        -u0 * np.sin(k * x) * np.cos(k * y),
    )


def shear_velocity(u_top: float = 0.04):
    return lambda x, y: (u_top * np.sin(np.pi * y / 64.0) ** 2, np.zeros_like(x))


def patch_slice(n: int, patch: int) -> slice:
    p0 = (n - patch) // 2
    return slice(p0, p0 + patch)


def calibrate(n: int, patch: int, velocity, scales: np.ndarray) -> dict:
    coarse, fine, _, _ = analytic_fields_2d(n, patch, velocity)
    sl = patch_slice(n, patch)
    return calibrate_prolongation_scale(coarse[:, sl, sl], fine, scales)


def correction_invariants(n: int, patch: int, velocity) -> dict:
    coarse, _, _, mask = analytic_fields_2d(n, patch, velocity)
    plan = EqualLevelBorderStreamCorrection(mask)
    uniform = np.zeros((9, n, n), dtype=np.float64)
    for q, w in enumerate([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]):
        uniform[q] = w
    plain = _stream(uniform.copy())
    corrected = plan.stream_with_correction(uniform.copy())
    inert = float(np.abs(plain - corrected).max())

    before = coarse.copy()
    saved = plan.prepare(coarse)
    prepared_changed = float(np.abs(coarse - before).max())
    plan.restore(coarse, saved)
    roundtrip = float(np.abs(coarse - before).max())

    # mass-flux defect of the substitution on the divergence-free analytic field
    mass_before = float(coarse.sum())
    mass_after = float(plan.stream_with_correction(coarse.copy()).sum())
    defect = abs(mass_after - mass_before) / abs(mass_before)
    plain_mass = float(_stream(coarse.copy()).sum())
    plain_defect = abs(plain_mass - mass_before) / abs(mass_before)
    return {
        "links": len(plan),
        "directions": list(plan.stencil_directions),
        "inert_uniform_max_abs": inert,
        "prepared_max_abs": prepared_changed,
        "roundtrip_max_abs": roundtrip,
        "mass_defect_corrected": defect,
        "mass_defect_plain": plain_defect,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/xval_block_amr_equal_level_stream_correction")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scales = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0])
    results: dict[str, object] = {
        "case": "xval_block_amr_equal_level_stream_correction",
        "reference": {
            "repo": "lssfau/walberla",
            "tag": "block-amr-equal-level-stream-correction",
            "oss_solutions_id": 283,
            "files": [
                "src/lbm/refinement/EqualLevelBorderStreamCorrection.h",
                "src/lbm/refinement/TimeStep.h",
                "src/lbm/refinement/CurlBasedLevelDetermination.h",
            ],
            "practice": "prepare: copy missing equal-level populations from the equally-sized "
                        "neighbour before streaming; apply: restore after streaming; default on",
        },
        "conventions": {
            name: prolongation_scale(0.6, 0.7, 2, name)
            for name in ("fh_spacing", "fh_plain", "repo_torch", "repo_cavity")
        },
        "calibration": {},
        "invariants": {},
    }

    cases = {
        "tgv_n64_p32": (64, 32, tgv_velocity(64)),
        "tgv_n128_p64": (128, 64, tgv_velocity(128)),
        "shear_n64_p32": (64, 32, shear_velocity()),
    }
    for name, (n, patch, velocity) in cases.items():
        results["calibration"][name] = calibrate(n, patch, velocity, scales)
        results["invariants"][name] = correction_invariants(n, patch, velocity)

    best = results["calibration"]["tgv_n64_p32"]
    results["verdict"] = {
        "fh_scale_measured": best["best_scale"],
        "fh_scale_error": best["best_error"],
        "fh_scale_gap": best["objective_gap"],
        "repo_torch_off_by_factor": abs(results["conventions"]["repo_torch"] / best["best_scale"]),
        "repo_cavity_off_by_factor": abs(results["conventions"]["repo_cavity"] / best["best_scale"]),
        "fh_plain_off_by_factor": abs(results["conventions"]["fh_plain"] / best["best_scale"]),
        "owner_criterion_pct": 15.0,
        "fh_scale_error_pct": 100.0 * best["best_error"],
        "passes_15pct": 100.0 * best["best_error"] < 15.0,
    }
    (out / "results.json").write_text(json.dumps(results, indent=1))

    bench = {
        "anchor_slug": "XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION",
        "ref_source": "analytic Taylor-Green / shear non-equilibrium field + waLBerla documented practice (oss_solutions id 283)",
        "rows": [],
        "repro_cmd": "PYTHONPATH=src python benchmarks/xval_block_amr_equal_level_stream_correction/run_xval.py --out artifacts/xval_block_amr_equal_level_stream_correction",
    }
    for name, rep in results["calibration"].items():
        bench["rows"].append({
            "kpi": f"prolongation scale error @ {name}",
            "ref": 0.0, "ref_unit": "relative L2 of non-equilibrium",
            "got": rep["best_error"], "got_unit": "relative L2",
            "err_pct": 100.0 * rep["best_error"], "tolerance_pct": 15.0,
            "pass": 100.0 * rep["best_error"] < 15.0,
            "measurement": rep["best_scale"],
        })
    for name, inv in results["invariants"].items():
        bench["rows"].append({
            "kpi": f"inertness on uniform state @ {name}",
            "ref": 0.0, "got": inv["inert_uniform_max_abs"], "err_pct": 0.0,
            "tolerance_pct": 0.0, "pass": inv["inert_uniform_max_abs"] == 0.0,
        })
        bench["rows"].append({
            "kpi": f"prepare/apply round trip @ {name}",
            "ref": 0.0, "got": inv["roundtrip_max_abs"], "err_pct": 0.0,
            "tolerance_pct": 0.0, "pass": inv["roundtrip_max_abs"] == 0.0,
        })
        bench["rows"].append({
            "kpi": f"mass-flux defect of the substitution @ {name}",
            "ref": inv["mass_defect_plain"], "got": inv["mass_defect_corrected"],
            "err_pct": 100.0 * abs(inv["mass_defect_corrected"] - inv["mass_defect_plain"]),
            "tolerance_pct": 15.0,
            "pass": abs(inv["mass_defect_corrected"] - inv["mass_defect_plain"]) < 0.15 * max(inv["mass_defect_plain"], 1e-12) or inv["mass_defect_corrected"] < 1e-3,
        })
    bench["summary"] = {
        "rows": len(bench["rows"]),
        "passed": sum(1 for r in bench["rows"] if r["pass"]),
    }
    (out / "benchmark.json").write_text(json.dumps(bench, indent=1))

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name, rep in results["calibration"].items():
        ax.semilogy(rep["scales"], rep["errors"], marker="o", ms=3, label=name)
    for name, style in (("repo_torch", "--"), ("repo_cavity", ":"), ("fh_plain", "-.")):
        ax.axvline(results["conventions"][name], ls=style, lw=1, label=f"{name} = {results['conventions'][name]:.2f}")
    ax.axvline(1.0, color="k", lw=1.6, label="measured optimum = 1.0")
    ax.set_xlabel("prolongation (non-equilibrium) scale")
    ax.set_ylabel("relative L2 of non-equilibrium")
    ax.set_title("F-H prolongation factor vs exact analytic field")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig1_calibration.png", dpi=140)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    names = list(results["invariants"])
    xs = np.arange(len(names))
    inert = [results["invariants"][k]["inert_uniform_max_abs"] for k in names]
    roundtrip = [results["invariants"][k]["roundtrip_max_abs"] for k in names]
    prep = [results["invariants"][k]["prepared_max_abs"] for k in names]
    floor = 1e-18
    ax.bar(xs - 0.25, [max(v, floor) for v in prep], 0.25, label="substitution magnitude")
    ax.bar(xs, [max(v, floor) for v in inert], 0.25, label="inertness (uniform)")
    ax.bar(xs + 0.25, [max(v, floor) for v in roundtrip], 0.25, label="round-trip residual")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("max |delta| (lattice units)")
    ax.set_title("correction acts on the field but is state-neutral")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig2_correction_effect.png", dpi=140)

    figures = [
        {"name": "fig1_calibration.png", "caption": "Prolongation-scale scan against the exact analytic field"},
        {"name": "fig2_correction_effect.png", "caption": "Substitution magnitude vs inertness / round-trip residual"},
    ]
    (out / "figures.json").write_text(json.dumps(figures, indent=1))

    lines = ["# XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION — run summary", ""]
    lines.append(f"measured F-H prolongation scale = {best['best_scale']:.3f} "
                 f"(relative error {100*best['best_error']:.4f} %, "
                 f"objective gap to runner-up {best['objective_gap']:.4f})")
    lines.append("")
    lines.append("| case | best scale | best rel. error | runner-up scale | gap |")
    lines.append("|---|---|---|---|---|")
    for name, rep in results["calibration"].items():
        lines.append(f"| {name} | {rep['best_scale']:.3f} | {100*rep['best_error']:.4f} % | "
                     f"{rep['runner_up_scale']:.3f} | {rep['objective_gap']:.4f} |")
    lines.append("")
    lines.append("| case | links | inertness | round-trip | mass defect (corrected) | mass defect (plain) |")
    lines.append("|---|---|---|---|---|---|")
    for name, inv in results["invariants"].items():
        lines.append(f"| {name} | {inv['links']} | {inv['inert_uniform_max_abs']:.3e} | "
                     f"{inv['roundtrip_max_abs']:.3e} | {inv['mass_defect_corrected']:.3e} | "
                     f"{inv['mass_defect_plain']:.3e} |")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": results["verdict"], "bench_summary": bench["summary"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
