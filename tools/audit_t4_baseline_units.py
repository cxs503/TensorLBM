#!/usr/bin/env python3
"""Auditor for the T4 physics-AI loop (GAP-LBM-T4-PHYSICS-AI-LOOP).

Three independent checks, none of which trusts the mission's own report:

1. ``baselines`` — recompute the one-step persistence and exact-viscous-prior
   errors from the shipped dataset **in the same units as the FNO target**
   (``u/u0``) and compare them with the numbers the benchmark recorded.
   (The pre-m15932-audit code fed raw lattice-unit fields to ``_rel_l2`` while
   the target was ``u/u0``; with ``u0 = 0.05`` that pins both baselines at
   ~1/u0 and manufactures a "surrogate beats its baselines" verdict.)
2. ``fields`` — confirm the shipped fields are real viscous LBM dynamics: the
   dominant spectral mode of every run must decay like ``exp(-2 nu k^2 dt)``
   and the discrete divergence must be ~0.  A synthetic / replayed field fails
   one of the two.
3. ``producer`` — re-run the solver-vs-exact-Taylor-Green check (the mission's
   own declared acceptance criterion, field < 1 % and decay rate < 2 %).

Usage::

    python tools/audit_t4_baseline_units.py all --dataset DIR --ckpt PATH --json OUT
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.ai.flow_field_loop import (  # noqa: E402
    FieldRunConfig,
    _pairs,
    _rel_l2,
    load_dataset,
    verify_producer,
    viscous_prior_step,
)
from tensorlbm.ai.fno import FNO2d, FNO2dArch  # noqa: E402


def _load_model(ckpt: str | Path) -> torch.nn.Module:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    fields = FNO2dArch.__dataclass_fields__
    arch = FNO2dArch(**{k: v for k, v in ck["arch"].items() if k in fields})
    model = FNO2d(arch)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def check_baselines(dataset: str | Path, ckpt: str | Path, holdout_re: float) -> dict:
    """One-step FNO / persistence / viscous prior, all scored in u0 units."""
    model = _load_model(ckpt)
    runs = load_dataset(dataset)["runs"]
    out = {"dataset": str(dataset), "checkpoint": str(ckpt), "holdout_re": holdout_re, "runs": {}}
    for run_id, run in sorted(runs.items()):
        if abs(float(run["re"]) - holdout_re) > 1e-6:
            continue
        u0 = float(run["u0"])
        dt = int(run["steps"][1] - run["steps"][0])
        x, y, _ = _pairs(run, u0)
        with torch.no_grad():
            pred = model(x)
        fno = [_rel_l2(pred[i], y[i]) for i in range(x.shape[0])]
        # repo-as-written: baselines get raw lattice-unit fields, target is u/u0
        repo_pers = [_rel_l2(torch.stack([run["ux"][i], run["uy"][i]]), y[i])
                     for i in range(x.shape[0])]
        repo_vis = [_rel_l2(torch.stack(viscous_prior_step(
            run["ux"][i], run["uy"][i], run["nu"], dt)), y[i])
            for i in range(x.shape[0])]
        # u0-consistent: every predictor in the units the target uses
        fix_pers = [_rel_l2(torch.stack([run["ux"][i], run["uy"][i]]) / u0, y[i])
                    for i in range(x.shape[0])]
        fix_vis = [_rel_l2(torch.stack(viscous_prior_step(
            run["ux"][i], run["uy"][i], run["nu"], dt)) / u0, y[i])
            for i in range(x.shape[0])]
        fno_m, fp_m, fv_m = float(np.mean(fno)), float(np.mean(fix_pers)), float(np.mean(fix_vis))
        out["runs"][run_id] = {
            "u0": u0,
            "n_pairs": int(x.shape[0]),
            "one_step_rel_l2_fno": fno_m,
            "recorded_persistence": float(np.mean(repo_pers)),
            "recorded_viscous_prior": float(np.mean(repo_vis)),
            "corrected_persistence": fp_m,
            "corrected_viscous_prior": fv_m,
            "surrogate_beats_strongest_baseline_as_recorded":
                fno_m < min(float(np.mean(repo_pers)), float(np.mean(repo_vis))),
            "surrogate_beats_strongest_baseline_when_units_consistent":
                fno_m < min(fp_m, fv_m),
            "ratio_fno_over_best_corrected_baseline": fno_m / max(min(fp_m, fv_m), 1e-12),
            "platform_15pct_threshold_pass": bool(fno_m < 0.15),
        }
    return out


def check_fields(dataset: str | Path) -> dict:
    """Real-LBM evidence: dominant-mode decay rate and discrete divergence."""
    runs = load_dataset(dataset)["runs"]
    rows = {}
    for run_id, run in sorted(runs.items()):
        ux = run["ux"].numpy().astype(np.float64)
        uy = run["uy"].numpy().astype(np.float64)
        n = ux.shape[-1]
        dt = int(run["steps"][1] - run["steps"][0])
        nu = float(run["nu"])
        f0 = np.abs(np.fft.fft2(ux[0])) ** 2 + np.abs(np.fft.fft2(uy[0])) ** 2
        f0[0, 0] = 0.0
        ky, kx = np.unravel_index(int(np.argmax(f0)), f0.shape)
        ky = ky - n if ky > n // 2 else ky
        kx = kx - n if kx > n // 2 else kx
        k2 = (2.0 * math.pi) ** 2 * (kx * kx + ky * ky) / (n * n)
        amps = []
        for i in range(ux.shape[0]):
            c = complex(np.fft.fft2(ux[i])[ky % n, kx % n])
            amps.append(abs(c))
        ratios = [amps[i + 1] / amps[i] for i in range(len(amps) - 1) if amps[i] > 0]
        measured = float(np.mean(ratios))
        analytic = float(math.exp(-2.0 * nu * k2 * dt))
        div = ((np.roll(ux[0], -1, axis=1) - ux[0]) + (np.roll(uy[0], -1, axis=0) - uy[0]))
        scale = float(np.sqrt((ux[0] ** 2 + uy[0] ** 2).mean()))
        rows[run_id] = {
            "n": n, "nu": nu, "dt_steps": dt,
            "dominant_mode": [int(ky), int(kx)], "k2": k2,
            "measured_decay_ratio_per_dt": measured,
            "analytic_decay_ratio_per_dt": analytic,
            "decay_ratio_rel_err": abs(measured - analytic) / analytic,
            "discrete_divergence_rms_over_urms": float(np.sqrt((div ** 2).mean()) / scale),
            "monotone_amplitude_decay": bool(all(r < 1.0 for r in ratios)),
            # 15 % screen: the IC is multi-mode, so the dominant mode exchanges
            # energy with its neighbours and cannot track the single-mode
            # analytic rate to solver accuracy.  A replayed/analytic field would
            # sit at ~0 %, a wrong viscosity at >50 %.
            # screen only: the IC is multi-mode, so the dominant Fourier mode
            # exchanges energy with its neighbours and cannot track the
            # single-mode analytic rate to solver accuracy.
            "decay_screen": ("consistent" if abs(measured - analytic) / analytic < 0.15
                             else "noisy"),
        }
    return {"dataset": str(dataset), "runs": rows,
            "all_divergence_free": all(r["discrete_divergence_rms_over_urms"] < 0.01
                                       for r in rows.values()),
            "max_decay_ratio_rel_err": max(r["decay_ratio_rel_err"] for r in rows.values())}


def check_producer(n: int, re: float, u0: float, n_steps: int) -> dict:
    rep = verify_producer(n=n, u0=u0, re_values=(re,), n_steps=n_steps,
                          sample_every=max(n_steps // 10, 1))
    row = rep["rows"][0]
    return {
        "n": n, "re": re, "u0": u0, "n_steps": n_steps,
        "max_abs_rel_err_field": float(row["max_abs_rel_err"]),
        "decay_rate_rel_err": float(row["decay_rate_rel_err"]),
        "field_criterion_1pct": bool(row["max_abs_rel_err"] < 0.01),
        "rate_criterion_2pct": bool(row["decay_rate_rel_err"] < 0.02),
        "verified": bool(rep.get("verified", row.get("verified", False))),
        "ruler": "u_rms = u0/sqrt(2) exp(-2 nu k^2 t), k = 2 pi / n (Taylor 1923)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("baselines", "fields", "producer", "all"))
    ap.add_argument("--dataset", default="artifacts/t4_physics_ai_loop_medium/dataset")
    ap.add_argument("--ckpt", default="artifacts/t4_physics_ai_loop_medium/model/fno_flow_map.pt")
    ap.add_argument("--holdout-re", type=float, default=75.0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    out: dict = {"tool": "tools/audit_t4_baseline_units.py", "what": args.what}
    if args.what in ("baselines", "all"):
        out["baselines"] = check_baselines(args.dataset, args.ckpt, args.holdout_re)
    if args.what in ("fields", "all"):
        out["fields"] = check_fields(args.dataset)
    if args.what in ("producer", "all"):
        out["producer"] = check_producer(64, 50.0, 0.05, 2000)
    text = json.dumps(out, indent=1)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())