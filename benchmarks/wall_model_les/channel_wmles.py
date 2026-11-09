#!/usr/bin/env python3
"""Wall-modelled LES of a high-Reynolds-number channel (criteria a, b context).

Runs :func:`tensorlbm.wall_model_les.channel.run_channel_wmles` for one wall-model
closure, then post-processes the time-averaged fields:

* velocity profile in wall units against the Spalding law and the Reichardt law
  (log-layer error over ``y+ = 30...300``);
* resolved Reynolds shear stress ``-<u'v'>+``, its peak abscissa and value;
* the total-stress identity ``(nu + nu_sgs) dU/dy - <u'v'> = u_tau^2 (1 - y/h)``;
* the Reynolds-stress budget terms (production, resolved + SGS dissipation);
* the skin-friction coefficient against Dean's channel correlation
  ``C_f = 0.073 Re_b^{-0.25}``.

The output is written to ``artifacts/wall_model_les/channel_<case>.json`` plus a
``profiles_<case>.npz`` with the raw wall-unit profiles.

Usage
-----
    python benchmarks/wall_model_les/channel_wmles.py --case eq --steps 4000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.wall_model_les.budget import (  # noqa: E402
    budget_from_profile,
    log_region_error,
    peak_location,
)
from tensorlbm.wall_model_les.channel import (  # noqa: E402
    ChannelWMLESConfig,
    run_channel_wmles,
)
from tensorlbm.wall_model_les.laws import reichardt_u_plus, spalding_u_plus  # noqa: E402

CASES = {
    "eq": dict(wall_model="equilibrium", wall_law="spalding"),
    "neq": dict(wall_model="nonequilibrium", wall_law="spalding"),
    "eq_log": dict(wall_model="equilibrium", wall_law="log"),
}


def dean_cf(re_bulk: float) -> float:
    """Dean's correlation for plane-channel flow, ``C_f = 0.073 Re_b^-0.25``."""
    return 0.073 * re_bulk ** -0.25


def run(case: str, *, steps: int, averaging_start: int, out_dir: Path, nx: int, nz: int, ny: int, cs: float = 0.1) -> dict:
    extra = CASES[case]
    config = ChannelWMLESConfig(
        nx=nx,
        nz=nz,
        ny=ny,
        re_tau=590.0,
        u_tau=0.02,
        steps=steps,
        averaging_start=averaging_start,
        stat_interval=10,
        log_interval=max(1, steps // 10),
        run_name=f"wmles-{case}",
        smagorinsky_cs=cs,
        **extra,
    )
    t0 = time.time()
    stats = run_channel_wmles(config, output_dir=out_dir / f"runs_channel_{case}")
    profiles = stats.pop("profiles")
    stats["wall_time_s"] = time.time() - t0

    y = np.asarray(profiles["y"], dtype=float)
    y_plus = np.asarray(profiles["y_plus"], dtype=float)
    u_plus = np.asarray(profiles["u_plus"], dtype=float)
    uv = np.asarray(profiles["uv"], dtype=float)  # <u'v'> (negative in the log layer)
    vv = np.asarray(profiles["vv"], dtype=float)
    ww = np.asarray(profiles["ww"], dtype=float)
    uu = np.asarray(profiles["uu"], dtype=float)
    diss_res = np.asarray(profiles["diss_res"], dtype=float)
    diss_sgs = np.asarray(profiles["diss_sgs"], dtype=float)

    u_tau = float(stats["u_tau_model_mean"])
    nu = float(stats["nu"])
    h = float(stats["h"])
    h_plus = h * u_tau / nu

    # wall-unit conversion of the statistics
    u_tau_conv = u_tau if u_tau > 0 else config.u_tau
    scale_stress = u_tau_conv**2
    uv_plus = -uv / scale_stress
    diss_res_plus = diss_res * nu / u_tau_conv**4
    diss_sgs_plus = diss_sgs * nu / u_tau_conv**4
    vvv_plus = np.asarray(profiles["vvv"], dtype=float) / u_tau_conv**3
    pv_plus = np.asarray(profiles["pv"], dtype=float) / u_tau_conv**3

    spalding = np.asarray(spalding_u_plus(torch.tensor(y_plus))).reshape(-1)
    reichardt = np.asarray(reichardt_u_plus(torch.tensor(y_plus))).reshape(-1)
    log_err_spalding = log_region_error(y_plus, u_plus, spalding, band=(30.0, 300.0))
    log_err_reichardt = log_region_error(y_plus, u_plus, reichardt, band=(30.0, 300.0))
    resolved_law = log_region_error(
        y_plus, u_plus, lambda yy: np.log(yy) / 0.41 + 5.2, band=(30.0, 300.0)
    )

    budget = budget_from_profile(
        y_plus=y_plus,
        u_plus=u_plus,
        uv_plus=uv_plus,
        re_tau=config.re_tau,
        h_plus=h_plus,
        diss_res_plus=diss_res_plus,
        diss_sgs_plus=diss_sgs_plus,
        vvv_plus=vvv_plus,
        pv_plus=pv_plus,
    )

    mask = y_plus >= 5.0
    uv_peak = peak_location(y_plus[mask], uv_plus[mask]) if mask.any() else float("nan")

    result = {
        "case": case,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in config.__dict__.items()},
        "run": stats,
        "wall_units": {
            "u_tau_model_mean": u_tau,
            "y_plus_first_node": float(y_plus[0]),
            "y_plus_exchange": float(stats["y_plus_exchange"]),
            "delta_y_plus": float(stats["delta_y_plus"]),
            "h_plus": h_plus,
            "u_bulk": float(profiles["u_bulk"]),
            "re_bulk": float(profiles["re_bulk"]),
        },
        "log_layer_error_vs_spalding": log_err_spalding,
        "log_layer_error_vs_reichardt": log_err_reichardt,
        "log_layer_error_vs_law_of_the_wall": resolved_law,
        "reynolds_stress": {
            "uv_peak_y_plus_resolved": uv_peak,
            "uv_peak_value_plus": float(np.max(uv_plus[mask])) if mask.any() else float("nan"),
            "uu_max_plus": float(np.max(uu / scale_stress)),
            "vv_max_plus": float(np.max(vv / scale_stress)),
            "ww_max_plus": float(np.max(ww / scale_stress)),
            "turbulence_intensity_bulk": float(
                np.sqrt(np.mean(uu[1:-1] + vv[1:-1] + ww[1:-1])) / max(profiles["u_bulk"], 1e-30)
            ),
        },
        "skin_friction": {
            "c_f_here": float(profiles["c_f"]),
            "c_f_dean_correlation": dean_cf(float(profiles["re_bulk"])),
            "relative_error_vs_dean": abs(
                float(profiles["c_f"]) / dean_cf(float(profiles["re_bulk"])) - 1.0
            ),
        },
        "budget": {k: v for k, v in budget.items() if k != "y_plus"},
        "momentum_balance": {
            "tau_w_model": float(stats["tau_w_model"]),
            "tau_w_force_balance": float(stats["tau_w_balance"]),
            "relative_gap": abs(float(stats["tau_w_model"]) / float(stats["tau_w_balance"]) - 1.0),
        },
        "verdicts": {
            "log_layer_vs_spalding_pass": bool(log_err_spalding.get("max_rel", 1.0) <= 0.03),
            "log_layer_vs_reichardt_pass": bool(log_err_reichardt.get("max_rel", 1.0) <= 0.03),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    # A run that blows up leaves complex/NaN entries behind (sqrt of a negative
    # dissipation, for instance); record them as null instead of crashing the
    # benchmark, because the fact that they exist is part of the result.
    def _sanitise(obj):
        if isinstance(obj, dict):
            return {k: _sanitise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitise(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.complexfloating, complex)):
            return None
        if isinstance(obj, np.ndarray):
            return _sanitise(obj.tolist())
        return obj

    (out_dir / f"channel_{case}.json").write_text(
        json.dumps(_sanitise(result), indent=2, allow_nan=False)
    )
    np.savez(
        out_dir / f"profiles_channel_{case}.npz",
        y=y,
        y_plus=y_plus,
        u_plus=u_plus,
        uv_plus=uv_plus,
        uu_plus=uu / scale_stress,
        vv_plus=vv / scale_stress,
        ww_plus=ww / scale_stress,
        spalding=spalding,
        reichardt=reichardt,
        diss_res_plus=diss_res_plus,
        diss_sgs_plus=diss_sgs_plus,
        production_plus=np.asarray(budget["production_plus"], dtype=float),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="eq", choices=sorted(CASES))
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--averaging-start", type=int, default=2000)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--ny", type=int, default=49)
    parser.add_argument("--cs", type=float, default=0.1)
    parser.add_argument("--out", default="artifacts/wall_model_les")
    args = parser.parse_args()
    result = run(
        args.case,
        steps=args.steps,
        averaging_start=args.averaging_start,
        out_dir=Path(args.out),
        nx=args.nx,
        nz=args.nz,
        ny=args.ny,
        cs=args.cs,
    )
    print(json.dumps({k: result[k] for k in ("wall_units", "log_layer_error_vs_spalding",
                                             "reynolds_stress", "skin_friction", "verdicts")},
                     indent=2, default=float))


if __name__ == "__main__":
    main()