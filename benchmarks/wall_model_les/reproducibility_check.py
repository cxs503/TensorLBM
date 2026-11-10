#!/usr/bin/env python3
"""Reproducibility check for the Re_tau = 590 WMLES (P372W2-N01).

Two different questions are answered here, and they must not be conflated:

1. **Determinism** -- the same configuration with the same seed must reproduce
   the same state bit for bit.  This is a property of the solver (torch, fixed
   seed, single device) and it is what "reproducible run" means mechanically.
2. **Statistical reproducibility** -- two independent seeds must give the same
   peak position and peak value within the sampling error of the averaging
   window.  This is the property the physical claim needs, and it is bounded
   here by the measured seed spread rather than asserted.

The configuration is deliberately small (a short box, few steps) because the
check is about repeatability, not about the converged statistics.

Usage
-----
    python benchmarks/wall_model_les/reproducibility_check.py --steps 1200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.wall_model_les.channel import (  # noqa: E402
    ChannelWMLESConfig,
    run_channel_wmles,
)
from tensorlbm.wall_model_les.peak_diagnostic import find_stress_peak  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "artifacts" / "wall_model_les_peak"


def _run(*, seed: int, steps: int, re_tau: float, ny: int, nx: int, nz: int) -> dict:
    config = ChannelWMLESConfig(
        nx=nx,
        nz=nz,
        ny=ny,
        re_tau=re_tau,
        steps=steps,
        averaging_start=int(0.5 * steps),
        stat_interval=5,
        log_interval=0,
        seed=seed,
        run_name=f"repro_seed{seed}",
    )
    stats = run_channel_wmles(config)
    prof = stats["profiles"]
    y_plus = np.asarray(prof["y_plus"], dtype=float)
    uv = np.asarray(prof["uv"], dtype=float)
    u_tau = float(prof["u_tau"])
    # inner-layer band of the lower half channel (outer 75 % excluded) and no
    # wall row (y+ <= 0 sits inside the bounce-back wall): see
    # peak_offset_study.analyse for the same band choice
    h_plus = 0.5 * float(np.max(y_plus))
    mask = (y_plus > 0.0) & (y_plus <= 0.25 * h_plus)
    peak = find_stress_peak(y_plus[mask], -uv[mask] / u_tau**2)
    return {
        "seed": seed,
        "y_plus": y_plus,
        "uv_plus": -uv / u_tau**2,
        "u_plus": np.asarray(prof["u_plus"], dtype=float),
        "u_bulk": float(prof["u_bulk"]),
        "peak_y_plus": peak.y_peak,
        "peak_value": peak.value_peak,
        "finite": bool(np.all(np.isfinite(uv))),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--re-tau", type=float, default=590.0)
    ap.add_argument("--ny", type=int, default=40)
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--nz", type=int, default=8)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    common = dict(steps=args.steps, re_tau=args.re_tau, ny=args.ny, nx=args.nx, nz=args.nz)
    first = _run(seed=0, **common)
    repeat = _run(seed=0, **common)
    other_seed = _run(seed=1, **common)

    def _max_abs_diff(a, b) -> float:
        return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

    determinism = {
        "uv_plus_max_abs_diff": _max_abs_diff(first["uv_plus"], repeat["uv_plus"]),
        "u_plus_max_abs_diff": _max_abs_diff(first["u_plus"], repeat["u_plus"]),
        "u_bulk_diff": abs(first["u_bulk"] - repeat["u_bulk"]),
        "bitwise_identical": bool(
            np.array_equal(first["uv_plus"], repeat["uv_plus"])
            and np.array_equal(first["u_plus"], repeat["u_plus"])
        ),
    }
    statistical = {
        "peak_y_plus_seed0": first["peak_y_plus"],
        "peak_y_plus_seed1": other_seed["peak_y_plus"],
        "peak_y_plus_rel_diff": abs(other_seed["peak_y_plus"] - first["peak_y_plus"])
        / first["peak_y_plus"],
        "peak_value_seed0": first["peak_value"],
        "peak_value_seed1": other_seed["peak_value"],
        "peak_value_rel_diff": abs(other_seed["peak_value"] - first["peak_value"])
        / first["peak_value"],
        "u_bulk_rel_diff": abs(other_seed["u_bulk"] - first["u_bulk"]) / abs(first["u_bulk"]),
    }
    # A "reproducible peak" is only meaningful when a peak exists: on a grid
    # and window where the resolved Reynolds stress never develops, two seeds
    # can agree on the (noise) maximum and that agreement says nothing about
    # reproducibility.  The gate records the peak magnitude against the DNS
    # scale (-<u'v'>+ peak ~ 0.87 at Re_tau = 590).
    stress_scale = 0.87
    developed = [
        run["peak_value"] / stress_scale for run in (first, other_seed)
        if math.isfinite(run["peak_value"])
    ]
    record = {
        "config": {**common, "averaging_start": int(0.5 * args.steps)},
        "stress_developed": bool(developed and min(developed) >= 0.2),
        "peak_value_over_dns_scale": developed,
        "determinism": determinism,
        "statistical": statistical,
        "note": (
            "determinism is exact reproducibility of the same seed; statistical is the seed-to-seed "
            "spread of the peak diagnostic over the averaging window of this short run"
        ),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reproducibility.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())