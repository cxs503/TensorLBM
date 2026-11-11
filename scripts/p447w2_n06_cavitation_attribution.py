"""p447w2-N06 driver: L_cav error attribution (criterion caliber x discretisation).

Runs one cavitating-airfoil solution per grid of the ``cells_per_chord`` ladder
``20/25/30/40`` at **fixed physical time** ``T = tU/c`` (so that every grid is
integrated to the same dimensionless time -- the coarser grids must *not* be
given a shorter physical window, which was one of the confounders of the
P41-N02 baseline), then evaluates the three criterion calibers on the
time-averaged fields of the converged window:

===================================  =========================================
criterion                            definition
===================================  =========================================
``isobaric``                         ``p < p_v`` (smooth 1-cell ramp), L from
                                     the chord-frame ``alpha=0.5`` cut of the
                                     indicator field
``alpha_v>=0.5``                     transported vapour fraction cut at 0.5
``L_eq``                             ``int alpha_v dV / (t c)``
                                     (:func:`...attribution.equivalent_length`)
===================================  =========================================

The 4 x 3 = 12-point matrix is then decomposed with a two-way ANOVA (block
replicates of the converged window) and, per criterion, with a generalised
Richardson extrapolation of the two three-grid sequences ``20-25-30`` and
``25-30-40``.

Outputs (all numbers generated here, never transcribed):

* ``artifacts/p447w2_n06/lattice_<cpc>.json``   -- per-grid raw results
* ``artifacts/p447w2_n06/mean_fields_<cpc>.npz``-- time-averaged p / alpha
* ``artifacts/p447w2_n06/attribution.json``     -- matrix, GCI, ANOVA, budget,
  verdicts (a)-(h), determinism check
* ``docs/figures/p447w2_n06_*.png``             -- report figures

Usage::

    python3 scripts/p447w2_n06_cavitation_attribution.py --sigma 1.0 0.8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.cavitation_airfoil import (  # noqa: E402
    CavitationFlowConfig,
    MixtureCavitationSolver,
    airfoil_surface,
    reference_data,
)
from tensorlbm.cavitation_lcav_attribution import (  # noqa: E402
    ALPHA_THRESHOLDS,
    CRITERIA,
    criterion_lengths,
    error_budget,
    relative_error,
    richardson_generalized,
    two_way_anova,
)

ART = ROOT / "artifacts" / "p447w2_n06"
FIG = ROOT / "docs" / "figures"

N_BLOCKS = 6
TRANSIENT_FRACTION = 0.5


def build_config(cells_per_chord: int, sigma: float, physical_time: float) -> CavitationFlowConfig:
    """Fixed-physics configuration; ``n_steps`` is derived from ``dt = cfl*h``.

    ``pressure_anchor`` and ``nuclei_inflow`` are switched **on**: without the
    inlet pressure anchor the Neumann projection leaves the pressure level
    arbitrary and the ``p < p_v`` criterion is meaningless; without the nuclei
    inflow the vapour population condenses to the sub-grid seed and every
    ``alpha_v`` threshold is degenerate.  Both switches are off by default in
    :class:`CavitationFlowConfig` only to keep the historical P41-N02 runs
    bit-for-bit reproducible.
    """
    cfl = 0.2
    h = 1.0 / cells_per_chord
    n_steps = int(round(physical_time / (cfl * h)))
    return CavitationFlowConfig(
        airfoil="NACA0015",
        alpha_deg=5.0,
        sigma=sigma,
        cells_per_chord=cells_per_chord,
        domain_x_chords=6.0,
        domain_y_chords=3.0,
        le_x_chords=1.5,
        le_y_chords=1.5,
        n_steps=n_steps,
        cfl=cfl,
        spinup_steps=0,
        ramp_steps=0,
        pressure_anchor=True,
        nuclei_inflow=True,
        sample_interval=1,
    )


def run_grid(spec: dict) -> dict:
    """Integrate one grid and return the per-grid criterion matrix."""
    torch.set_num_threads(1)
    cpc = int(spec["cells_per_chord"])
    sigma = float(spec["sigma"])
    t_end = float(spec["physical_time"])
    cfg = build_config(cpc, sigma, t_end)
    geom = airfoil_surface(cfg.airfoil)
    t0 = time.time()
    solver = MixtureCavitationSolver(cfg)
    dt = solver.dt
    n_steps = cfg.n_steps
    window_start = int(round(TRANSIENT_FRACTION * n_steps))
    # ~20 samples per unit physical time
    sample_every = max(1, int(round(0.05 / dt)))
    blocks = [{"p": torch.zeros_like(solver.p), "a": torch.zeros_like(solver.alpha), "n": 0}
              for _ in range(N_BLOCKS)]
    # instantaneous criterion lengths, accumulated per block (replicates for the
    # ANOVA): the *mean cavity length* of a threshold criterion on a fluctuating
    # field is <L(t)>, not L(<field>) -- the latter is a systematic
    # underestimate (threshold-averaging bias) and is kept as a diagnostic only.
    per_block = [
        {"isobaric": [], "alpha_v>=0.5": [], "L_eq": [],
         "alpha_absolute": {f"{tau:g}": [] for tau in ALPHA_THRESHOLDS},
         "alpha_relative": {"0.1": [], "0.5": [], "0.9": []},
         "p_min": [], "alpha_max": []}
        for _ in range(N_BLOCKS)
    ]
    series: list[dict] = []
    p_min_hist: list[float] = []
    n_window_samples = max(1, (n_steps - window_start) // sample_every)
    k_window = 0
    for k in range(n_steps):
        solver.step()
        if (k + 1) % sample_every:
            continue
        p_used = solver.p + solver.p_offset
        p_min_hist.append(float(p_used.min()))
        if (k + 1) <= window_start:
            continue
        bi = min(N_BLOCKS - 1, (k_window * N_BLOCKS) // n_window_samples)
        k_window += 1
        blocks[bi]["p"] += p_used
        blocks[bi]["a"] += solver.alpha
        blocks[bi]["n"] += 1
        cl = criterion_lengths(p_used, solver.alpha, cfg, geometry=geom)
        for crit in CRITERIA:
            per_block[bi][crit].append(cl["criterion_lengths"][crit])
        for tau in ALPHA_THRESHOLDS:
            per_block[bi]["alpha_absolute"][f"{tau:g}"].append(
                cl["alpha_absolute_lengths"][f"alpha_v>={tau:g}"]
            )
        rel_keys = [key for key in cl if key.startswith("l_alpha_rel>=")]
        for idx, lvl in enumerate(("0.1", "0.5", "0.9")):
            per_block[bi]["alpha_relative"][lvl].append(cl[rel_keys[idx]])
        per_block[bi]["p_min"].append(float(p_used.min()))
        per_block[bi]["alpha_max"].append(float(solver.alpha.max()))
        series.append(
            {
                "t": (k + 1) * dt,
                "l_isobaric_inst": cl["criterion_lengths"]["isobaric"],
                "l_alpha50_inst": cl["criterion_lengths"]["alpha_v>=0.5"],
                "l_eq_inst": cl["criterion_lengths"]["L_eq"],
                "alpha_max": float(solver.alpha.max()),
                "p_min": float(p_used.min()),
            }
        )

    if any(blk["n"] == 0 for blk in blocks):
        msg = "empty averaging block: increase physical_time"
        raise RuntimeError(msg)

    def _mean(values):
        return float(sum(values)) / max(len(values), 1)

    replicates = {
        crit: [_mean(per_block[b][crit]) for b in range(N_BLOCKS)] for crit in CRITERIA
    }
    replicates["alpha_absolute"] = {
        f"{tau:g}": [_mean(per_block[b]["alpha_absolute"][f"{tau:g}"]) for b in range(N_BLOCKS)]
        for tau in ALPHA_THRESHOLDS
    }
    replicates["alpha_relative"] = {
        lvl: [_mean(per_block[b]["alpha_relative"][lvl]) for b in range(N_BLOCKS)]
        for lvl in ("0.1", "0.5", "0.9")
    }
    matrix_mean = {
        crit: _mean([s_["l_isobaric_inst" if crit == "isobaric" else (
            "l_alpha50_inst" if crit == "alpha_v>=0.5" else "l_eq_inst")] for s_ in series])
        for crit in CRITERIA
    }
    alpha_abs_mean = {
        f"alpha_v>={tau:g}": _mean([v for b in range(N_BLOCKS)
                                    for v in per_block[b]["alpha_absolute"][f"{tau:g}"]])
        for tau in ALPHA_THRESHOLDS
    }

    # field-mean diagnostic: L(<field>) vs <L(t)>
    block_fields = [(blk["p"] / blk["n"], blk["a"] / blk["n"], blk["n"]) for blk in blocks]
    wsum = sum(blk[2] for blk in block_fields)
    p_mean = sum(blk[0] * blk[2] for blk in block_fields) / wsum
    a_mean = sum(blk[1] * blk[2] for blk in block_fields) / wsum
    field_mean = criterion_lengths(p_mean, a_mean, cfg, geometry=geom)
    field_matrix = field_mean["criterion_lengths"]
    matrix_detail = {k: v for k, v in field_mean.items() if not isinstance(v, dict)}

    out = {
        "cells_per_chord": cpc,
        "sigma": sigma,
        "h": cfg.h,
        "dt": dt,
        "n_steps": n_steps,
        "physical_time": n_steps * dt,
        "n_samples_window": len(series),
        "block_counts": [blk[2] for blk in block_fields],
        "matrix": matrix_mean,
        "matrix_field_mean": field_matrix,
        "matrix_detail": matrix_detail,
        "alpha_absolute_lengths": alpha_abs_mean,
        "replicates": replicates,
        "extent_footprint": float(field_mean.get("extent_footprint", 0.0)),
        "residuals": {
            "mass_balance_max": solver.global_balance_max,
            "div_residual_max": solver.div_residual_max,
            "div_residual_final": float(
                ((solver.u[1:, :] - solver.u[:-1, :]) / cfg.h).abs().max()
            ),
        },
        "alpha_max_final": float(solver.alpha.max()),
        "p_min_final": float((solver.p + solver.p_offset).min()),
        "p_min_window": min(p_min_hist) if p_min_hist else float("nan"),
        "p_v": cfg.p_v,
        "series": series,
        "wall_seconds": time.time() - t0,
    }
    npz = ART / f"mean_fields_{cpc}_sigma{sigma:g}.npz"
    import numpy as np

    np.savez_compressed(
        npz,
        p_mean=p_mean.double().numpy(),
        alpha_mean=a_mean.double().numpy(),
        solid=solver.solid.numpy(),
        h=cfg.h,
        sigma=sigma,
    )
    try:
        out["mean_field_file"] = str(npz.relative_to(ROOT))
    except ValueError:  # ART monkeypatched to a tmp dir (tests)
        out["mean_field_file"] = str(npz)
    return out


def _sequences(cells: list[int]) -> list[list[int]]:
    """The two three-grid sequences of the ladder (``20-25-30``, ``25-30-40``)."""
    seqs = [list(cells[i : i + 3]) for i in range(len(cells) - 2)]
    return [s for s in seqs if len(s) == 3]


def determinism_check(cells_per_chord: int, sigma: float, physical_time: float) -> dict:
    """(h) run the same case twice with identical single-thread settings."""
    spec = {"cells_per_chord": cells_per_chord, "sigma": sigma,
            "physical_time": physical_time}
    a = run_grid(spec)
    b = run_grid(spec)
    rel = {
        crit: abs(b["matrix"][crit] - a["matrix"][crit]) / max(abs(a["matrix"][crit]), 1e-300)
        for crit in CRITERIA
    }
    for key in ("mass_balance_max", "div_residual_max"):
        rel[key] = abs(b["residuals"][key] - a["residuals"][key]) / max(
            abs(a["residuals"][key]), 1e-300
        )
    worst = max(rel.values())
    return {
        "cell": cells_per_chord,
        "rel_diffs": rel,
        "worst_rel": worst,
        "values": a["matrix"],
        "pass": bool(worst < 1e-12),
    }


def aggregate(grids: list[dict], sigma: float, reference: dict, tag: str) -> dict:
    """Combine the per-grid results into matrix + GCI + ANOVA + budget + verdicts."""
    grids = sorted(grids, key=lambda d: d["cells_per_chord"])
    cells = [g["cells_per_chord"] for g in grids]
    ref = float(reference["l_cav_exp"][reference["sigma"].index(sigma)])

    # ---- 12-point matrix ---------------------------------------------------
    matrix = {
        c: {crit: floats["matrix"][crit] for crit in CRITERIA}
        for c, floats in ((g["cells_per_chord"], g) for g in grids)
    }
    errors = {
        c: {crit: relative_error(matrix[c][crit], ref) for crit in CRITERIA} for c in cells
    }

    # ---- grid convergence per criterion -----------------------------------
    gci = {}
    for crit in CRITERIA:
        entries = []
        for seq in _sequences(cells):
            h = [1.0 / c for c in seq[::-1]]  # fine -> coarse
            f = [matrix[c][crit] for c in seq[::-1]]
            r = richardson_generalized(h, f)
            r["sequence"] = seq
            entries.append(r)
        gci[crit] = entries

    # ---- optimal criterion -------------------------------------------------
    # selection rule: smallest |L - L_ref| of the h->0 extrapolate of the
    # refined sequence (25-30-40); fall back to the finest-grid value when no
    # extrapolation is admissible.  The rule is a diagnostic, no threshold or
    # coefficient is tuned to reach it.
    scored = {}
    for crit in CRITERIA:
        ext = gci[crit][-1]
        ext_ok = bool(ext["valid"]) and (0.0 <= float(ext["f_extrap"]) <= 1.2)
        candidate = float(ext["f_extrap"]) if ext_ok else matrix[cells[-1]][crit]
        scored[crit] = {
            "value": candidate,
            "from_extrapolation": ext_ok,
            "extrapolation_physical": ext_ok,
            "extrapolation_valid_math": bool(ext["valid"]),
            "extrapolation_value": float(ext["f_extrap"]) if ext["valid"] else None,
            "error": relative_error(candidate, ref),
        }
    optimal = min(scored, key=lambda k: scored[k]["error"])

    # ---- ANOVA on the relative-error response ------------------------------
    table = {}
    for crit in CRITERIA:
        for g in grids:
            table[(g["cells_per_chord"], crit)] = [
                relative_error(v, ref) for v in g["replicates"][crit]
            ]
    anova = two_way_anova(table)

    # ---- additive budget (all terms at the finest grid except the extrap) --
    finest = cells[-1]
    base_crit = "alpha_v>=0.5"
    l_base = matrix[finest][base_crit]
    l_criterion = matrix[finest][optimal]
    ext = gci[optimal][-1]
    l_extrap = scored[optimal]["value"]
    budget = error_budget(
        l_base=l_base,
        l_criterion=l_criterion,
        l_extrap=l_extrap,
        l_reference=ref,
        l_disc_fine=matrix[finest][optimal],
    )
    budget["base_criterion"] = base_crit
    budget["optimal_criterion"] = optimal
    budget["extrapolation_valid"] = bool(scored[optimal]["from_extrapolation"])

    # ---- verdicts (a)-(h) --------------------------------------------------
    col = [matrix[c][optimal] for c in cells]
    diffs = [col[i + 1] - col[i] for i in range(len(col) - 1)]
    monotone_up = all(d >= 0 for d in diffs)
    monotone_down = all(d <= 0 for d in diffs)
    block_std = [
        float(torch.tensor(g["replicates"][optimal]).std(unbiased=True)) for g in grids
    ]
    noise_ratio = [
        abs(diffs[i]) / max(block_std[i], block_std[i + 1], 1e-12)
        for i in range(len(diffs))
    ]
    p_ext = ext["p"] if scored[optimal]["from_extrapolation"] else float("nan")
    verdicts = {
        "a_monotone": {
            "values": col,
            "cells": cells,
            "diffs": diffs,
            "block_std": block_std,
            "diff_over_noise": noise_ratio,
            "monotone": bool(monotone_up or monotone_down),
            "direction": "increasing" if monotone_up else ("decreasing" if monotone_down else "none"),
            "above_noise": bool(all(r > 1.0 for r in noise_ratio)),
            "pass": bool((monotone_up or monotone_down)),
        },
        "b_observed_order": {
            "p": p_ext,
            "in_band": bool(0.5 <= p_ext <= 2.5) if math.isfinite(p_ext) else False,
            "valid_sequence": bool(ext["valid"]),
            "reason": ext.get("reason", ""),
            "ratio_r": ext.get("ratio_r"),
            "pass": bool(math.isfinite(p_ext) and 0.5 <= p_ext <= 2.5),
        },
        "c_criterion_main_effect": {
            "pct_criterion": anova["pct_criterion"],
            "pct_grid": anova["pct_grid"],
            "pct_interaction": anova["pct_interaction"],
            "pct_residual": anova["pct_residual"],
            "pass": bool(anova["pct_criterion"] >= 50.0),
        },
        "d_optimal_error": {
            "criterion": optimal,
            "value": scored[optimal]["value"],
            "reference": ref,
            "error": scored[optimal]["error"],
            "pass": bool(scored[optimal]["error"] <= 0.20),
        },
        "e_gci": {
            "gci_fine": ext["gci_fine"],
            "safety_factor": 1.25,
            "pass": bool(math.isfinite(ext["gci_fine"]) and ext["gci_fine"] <= 0.15),
        },
        "f_conservation": {
            "mass_balance_max": max(g["residuals"]["mass_balance_max"] for g in grids),
            "div_residual_max": max(g["residuals"]["div_residual_max"] for g in grids),
            "threshold": 1e-4,
            "mass_pass": bool(
                max(g["residuals"]["mass_balance_max"] for g in grids) <= 1e-4
            ),
            "div_pass": bool(
                max(g["residuals"]["div_residual_max"] for g in grids) <= 1e-4
            ),
            "pass": bool(
                max(g["residuals"]["mass_balance_max"] for g in grids) <= 1e-4
                and max(g["residuals"]["div_residual_max"] for g in grids) <= 1e-4
            ),
        },
        "g_alpha_sensitivity": {
            "absolute": {
                c: [matrix[c][crit] for crit in CRITERIA] for c in cells
            },
            "absolute_lengths": {
                str(g["cells_per_chord"]): g["alpha_absolute_lengths"] for g in grids
            },
            "relative_lengths": {
                str(g["cells_per_chord"]): {
                    lvl: float(torch.tensor(g["replicates"]["alpha_relative"][lvl]).mean())
                    for lvl in ("0.1", "0.5", "0.9")
                }
                for g in grids
            },
            "degenerate_absolute": bool(
                all(
                    max(g["alpha_absolute_lengths"].values()) == 0.0 for g in grids
                )
            ),
        },
    }
    rel = verdicts["g_alpha_sensitivity"]["relative_lengths"]
    mono_rel = all(
        rel[str(c)]["0.1"] >= rel[str(c)]["0.5"] >= rel[str(c)]["0.9"] for c in cells
    )
    nondeg = any(
        g["matrix_detail"].get("alpha_max", 0.0) > 0.0 for g in grids
    )
    verdicts["g_alpha_sensitivity"]["relative_monotone"] = bool(mono_rel)
    verdicts["g_alpha_sensitivity"]["pass"] = bool(mono_rel and nondeg)
    verdicts["g_alpha_sensitivity"]["note"] = (
        "absolute alpha_v in {0.1,0.5,0.9} family is degenerate (vapour fraction "
        "never reaches 0.1); the scale-free relative family is used to test "
        "monotonicity of the criterion in its own threshold"
        if verdicts["g_alpha_sensitivity"]["degenerate_absolute"]
        else ""
    )

    return {
        "tag": tag,
        "sigma": sigma,
        "reference": {"l_cav_exp": ref, "source": reference["source"]},
        "cells": cells,
        "grid_summary": [
            {
                "cells_per_chord": g["cells_per_chord"],
                "h": g["h"],
                "dt": g["dt"],
                "n_steps": g["n_steps"],
                "physical_time": g["physical_time"],
                "matrix": g["matrix"],
                "matrix_field_mean": g["matrix_field_mean"],
                "alpha_max": g["matrix_detail"]["alpha_max"],
                "p_min": g["matrix_detail"]["p_min"],
                "p_rel_min": g["matrix_detail"]["p_rel_min"],
                "residuals": g["residuals"],
                "wall_seconds": g["wall_seconds"],
                "block_counts": g["block_counts"],
                "mean_field_file": g["mean_field_file"],
                "series": g["series"],
            }
            for g in grids
        ],
        "matrix": matrix,
        "errors": errors,
        "gci": gci,
        "optimal": {"criterion": optimal, "scores": scored},
        "anova": anova,
        "budget": budget,
        "verdicts": verdicts,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, nargs="+", default=[1.0])
    ap.add_argument("--cells", type=int, nargs="+", default=[20, 25, 30, 40])
    ap.add_argument("--physical-time", type=float, default=6.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-determinism", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--determinism-only", action="store_true",
                    help="only run the (h) reproducibility pair and store the verdict")
    ap.add_argument("--from-artifacts", action="store_true",
                    help="re-aggregate the saved lattice_*.json files (no solver runs)")
    args = ap.parse_args(argv)

    ART.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    reference = reference_data()["sheet"]

    all_out: dict = {"runs": {}, "meta": {
        "physical_time": args.physical_time,
        "cells_per_chord": args.cells,
        "n_blocks": N_BLOCKS,
        "transient_fraction": TRANSIENT_FRACTION,
        "alpha_thresholds": list(ALPHA_THRESHOLDS),
        "criteria": list(CRITERIA),
        "reference_source": reference["source"],
    }}

    for sigma in args.sigma:
        if args.from_artifacts:
            grids = [
                json.loads((ART / f"lattice_{c}_sigma{sigma:g}.json").read_text())
                for c in args.cells
            ]
            result = aggregate(grids, sigma, reference, tag=args.tag or f"sigma{sigma:g}")
            print(f"[sigma={sigma}] re-aggregated from saved grids", flush=True)
        else:
            specs = [
                {"cells_per_chord": c, "sigma": sigma, "physical_time": args.physical_time}
                for c in args.cells
            ]
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                grids = list(pool.map(run_grid, specs))
            print(f"[sigma={sigma}] {len(grids)} grids done in {time.time()-t0:.0f}s", flush=True)
            for g in grids:
                (ART / f"lattice_{g['cells_per_chord']}_sigma{sigma:g}.json").write_text(
                    json.dumps(g, indent=1)
                )
            result = aggregate(grids, sigma, reference, tag=args.tag or f"sigma{sigma:g}")
            if not args.skip_determinism:
                t1 = time.time()
                det = determinism_check(args.cells[0], sigma, args.physical_time)
                (ART / f"determinism_sigma{sigma:g}.json").write_text(json.dumps(det, indent=1))
                print(f"[sigma={sigma}] determinism rerun in {time.time()-t1:.0f}s: "
                      f"{det['rel_diffs']}", flush=True)

        det_file = ART / f"determinism_sigma{sigma:g}.json"
        result["verdicts"]["h_determinism"] = (
            json.loads(det_file.read_text()) if det_file.exists() else {"skipped": True, "pass": False}
        )
        all_out["runs"][f"sigma{sigma:g}"] = result
        if args.determinism_only:
            continue
        print(
            f"[sigma={sigma}] matrix={json.dumps(result['matrix'])} "
            f"optimal={result['optimal']['criterion']}", flush=True
        )

    (ART / "attribution.json").write_text(json.dumps(all_out, indent=1))
    print(f"wrote {ART/'attribution.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())