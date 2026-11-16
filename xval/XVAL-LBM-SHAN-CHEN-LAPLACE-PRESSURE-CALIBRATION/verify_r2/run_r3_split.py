#!/usr/bin/env python3
"""R3 executor — per-(dim, leg) split runs for the R2 pre-registered plan.

Why this file exists
--------------------
R2 (`run_r2.py --stage all`) was submitted as one monolithic unit: the 2D and 3D
droplet/slab legs had to *all* finish inside one queue window, and none of them
did, so the whole artifact was lost and P1..P5 stayed NOT_RUN.  R3 keeps the
same pre-registered protocol, thresholds and radii, but runs each (dim, leg)
independently and stores it as its own part file.  A part that finishes is
banked even if a sibling part never gets scheduled, and the merger below marks
anything missing as NOT_RUN instead of inventing a number.

Usage
-----
    python run_r3_split.py --dim 2 --leg droplet --jobs 3 --threads 4
    python run_r3_split.py --dim 3 --leg droplet --jobs 3 --threads 8
    python run_r3_split.py --dim 2 --leg flat    --threads 4
    python run_r3_split.py --merge            # → artifacts/r3_summary.json

Part files: ``artifacts/r3_parts/{leg}_{dim}d.json``.  Merge writes
``artifacts/r3_summary.json`` (same schema as ``r2_summary.json`` plus a
``provenance`` block) and prints the verdict table.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_r2 as R2  # noqa: E402  (single source of truth for protocol/thresholds)

PARTS = HERE / "artifacts" / "r3_parts"
SUMMARY = HERE / "artifacts" / "r3_summary.json"
PREDECLARED = HERE / "predeclared.json"


# ---------------------------------------------------------------------------
# runners (thin wrappers around the R2 case functions — no physics duplicated)
# ---------------------------------------------------------------------------


def run_leg(dim: int, leg: str, jobs: int, threads: int, device: str, quick: bool) -> dict[str, Any]:
    torch.set_num_threads(threads)
    radii = R2.QUICK["radii"] if quick else R2.RADII
    nscale = 0.5 if quick else 1.0
    if leg == "droplet":
        steps = R2.QUICK["droplet_steps"] if quick else R2.DROPLET["steps"]
        min_steps = min(R2.DROPLET["min_steps"], steps // 2)
        t0 = time.perf_counter()
        spec = [(dim, R, steps, min_steps, device) for R in radii]
        # jobs <= 1 runs in-process: a forked Pool around a torch/OMP process is a
        # deadlock magnet on some sandboxes, and 2D cases are only ~10 s each.
        if jobs <= 1:
            rows = [R2.run_droplet_case(sp) for sp in spec]
        else:
            with Pool(min(jobs, len(spec))) as pool:
                rows = pool.map(R2.run_droplet_case, spec)
        payload: dict[str, Any] = {"rows": rows}
    elif leg == "flat":
        steps = R2.QUICK["flat_steps"] if quick else R2.FLAT_RUN["steps"]
        half = R2.QUICK["flat_half"] if quick else R2.FLAT[str(dim)]["half"]
        t0 = time.perf_counter()
        out = R2.run_flat_case((dim, steps, device, half, nscale))
        out.pop("profile", None)  # keep part files small; profile re-derivable
        payload = out  # type: ignore[assignment]
    else:  # pragma: no cover - argparse guards this
        raise SystemExit(f"unknown leg {leg!r}")
    payload["meta"] = {
        "dim": dim,
        "leg": leg,
        "quick": quick,
        "radii": radii if leg == "droplet" else None,
        "threads": threads,
        "jobs": jobs,
        "wallclock_s": time.perf_counter() - t0,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
    }
    PARTS.mkdir(parents=True, exist_ok=True)
    dst = PARTS / f"{leg}_{dim}d.json"
    dst.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved → {dst}")
    return payload


def run_field(dim: int, R: float, steps: int, threads: int, device: str) -> dict[str, Any]:
    """One converged droplet + a density-field dump (for the field-cloud figure).

    The R2 artifacts carry no field arrays, so the R3 report had no way to show a
   云图 without re-running a case.  This leg is the smallest thing that fixes
    that: same protocol/initialisation as ``run_r2.run_droplet_case``, with the
    converged ``rho(x)`` written out.
    """
    torch.set_num_threads(threads)
    L = int(R2.L_FACTOR * R)
    rho0, _ = R2._init_droplet(dim, L, R, 1.957, 0.1596, torch.device(device))
    zero = torch.zeros_like(rho0)
    f = (
        R2.equilibrium(rho0, zero, zero)
        if dim == 2
        else R2.equilibrium3d(rho0, zero, zero, zero)
    )
    rr = R2._radius_field(dim, L, torch.device(device))
    min_steps = min(R2.DROPLET["min_steps"], steps // 2)
    hist: list[dict[str, float]] = []
    conv, step = False, 0
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        f = R2._step(dim, f)
        if step % R2.DROPLET["sample_every"] == 0:
            m = R2.measure_droplet(dim, f, rr)
            hist.append(m)
            if len(hist) >= 2 and step >= min_steps:
                d_dp = abs(hist[-1]["dp"] - hist[-2]["dp"]) / abs(hist[-1]["dp"])
                d_r = abs(hist[-1]["R_eq"] - hist[-2]["R_eq"]) / hist[-1]["R_eq"]
                if d_dp < R2.DROPLET["conv_dp"] and d_r < R2.DROPLET["conv_r"]:
                    conv = True
                    break
    rho = f.sum(dim=0)
    payload = {
        "meta": {
            "dim": dim,
            "leg": "field",
            "R_init": R,
            "L": L,
            "converged": conv,
            "steps": int(step),
            "threads": threads,
            "wallclock_s": time.perf_counter() - t0,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
        },
        "rho": rho.tolist(),
        "measure": hist[-1],
    }
    PARTS.mkdir(parents=True, exist_ok=True)
    dst = PARTS / f"field_{dim}d.json"
    dst.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(f"saved → {dst} (steps={step}, converged={conv})")
    return payload


# ---------------------------------------------------------------------------
# merge + verdicts
# ---------------------------------------------------------------------------


def _load(leg: str, dim: int) -> dict[str, Any] | None:
    p = PARTS / f"{leg}_{dim}d.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def merge() -> dict[str, Any]:
    tol = R2.THRESHOLDS["tolerance"]
    tight = R2.THRESHOLDS["tight"]
    pre = json.loads(PREDECLARED.read_text(encoding="utf-8")) if PREDECLARED.exists() else {}

    out: dict[str, Any] = {
        "anchor": "XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION",
        "round": "R3",
        "protocol": {
            "model": "SCMP SC94, psi=1-exp(-rho), D2Q9/D3Q19",
            "G_lib": R2.G_LIB,
            "G_eff": R2.G_EFF,
            "tau": R2.TAU,
            "radii": R2.RADII,
            "L_factor": R2.L_FACTOR,
            "dims": [2, 3],
            "law": "dp = (dims-1) sigma / R",
            "thresholds": R2.THRESHOLDS,
            "predeclared": pre,
        },
        "kappa3": {"2": R2.kappa3(2), "3": R2.kappa3(3)},
        "maxwell": R2.maxwell_coexistence(),
        "flat": {},
        "fit": {},
        "droplets": [],
        "provenance": {},
    }
    out["kappa3_ratio"] = out["kappa3"]["3"]["kappa3"] / out["kappa3"]["2"]["kappa3"]

    for dim in (2, 3):
        flat = _load("flat", dim)
        if flat is None:
            out["flat"][str(dim)] = None
        else:
            out["flat"][str(dim)] = {k: v for k, v in flat.items() if k != "meta"}
            out["provenance"][f"flat_{dim}d"] = flat["meta"]
        drop = _load("droplet", dim)
        if drop is None:
            out["fit"][str(dim)] = None
            continue
        out["provenance"][f"droplet_{dim}d"] = drop["meta"]
        rows = drop["rows"]
        out["droplets"].extend(rows)
        out["fit"][str(dim)] = R2._fit_sigma(rows, dim)

    def num(x: Any) -> bool:
        return isinstance(x, (int, float))

    v: dict[str, Any] = {}
    f2, f3 = out["flat"].get("2"), out["flat"].get("3")
    s2 = (out["fit"].get("2") or {}).get("sigma_mean_of_radii")
    s3 = (out["fit"].get("3") or {}).get("sigma_mean_of_radii")

    if num(s2) and num(s3):
        ratio = s3 / s2
        out["cross_dim"] = {
            "sigma_2d": s2,
            "sigma_3d": s3,
            "ratio": ratio,
            "dev_from_prediction_1": abs(ratio - 1.0),
            "predicted_ratio_from_kappa3": out["kappa3_ratio"],
        }
        v["P1_cross_dim"] = abs(ratio - 1.0) < tol
        v["P1_cross_dim_tight3pct"] = abs(ratio - 1.0) < tight
    else:
        out["cross_dim"] = None
        v["P1_cross_dim"] = "NOT_RUN"

    for dim, flat in (("2", f2), ("3", f3)):
        if flat is None:
            v[f"P2_mech_balance_{dim}d"] = "NOT_RUN"
            v[f"P4_sigmaK_vs_lap_{dim}d"] = "NOT_RUN"
            v[f"P5_coexist_vs_maxwell_{dim}d"] = "NOT_RUN"
            continue
        v[f"P2_mech_balance_{dim}d"] = flat["mech_gap"] < R2.THRESHOLDS["mech_balance"]
        mw = out["maxwell"]
        v[f"P5_coexist_vs_maxwell_{dim}d"] = (
            abs(flat["rho_l"] - mw["rho_l"]) / mw["rho_l"] < tol
            and abs(flat["rho_v"] - mw["rho_v"]) / mw["rho_v"] < tol
        )
        s = s2 if dim == "2" else s3
        if num(s):
            sk = flat["sigma_K_tanh"]
            v[f"P4_sigmaK_vs_lap_{dim}d"] = abs(sk - s) / sk < tol
        else:
            v[f"P4_sigmaK_vs_lap_{dim}d"] = "NOT_RUN"

    for dim in ("2", "3"):
        fit = out["fit"].get(dim)
        v[f"P3_radius_consistency_{dim}d"] = (
            "NOT_RUN" if fit is None else fit["max_dev"] < tol
        )

    out["verdicts"] = v
    SUMMARY.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out["verdicts"], indent=2, sort_keys=True))
    print(f"saved → {SUMMARY}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, choices=[2, 3])
    ap.add_argument("--leg", choices=["droplet", "flat", "field"])
    ap.add_argument("--R", type=float, default=14.0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
        return
    if a.dim is None or a.leg is None:
        raise SystemExit("need --dim and --leg (or --merge)")
    if a.leg == "field":
        steps = R2.QUICK["droplet_steps"] if a.quick else R2.DROPLET["steps"]
        run_field(a.dim, a.R, steps, a.threads, a.device)
        return
    run_leg(a.dim, a.leg, a.jobs, a.threads, a.device, a.quick)


if __name__ == "__main__":
    main()