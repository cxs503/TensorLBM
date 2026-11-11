#!/usr/bin/env python3
"""Surface-tension *dynamics* verification driver (P152W1-N01, TensorLBM).

Three machine-checkable anchors, all against closed-form references:

* ``wave``        -- standing capillary wave, ``omega(k)``/``gamma`` vs the
  finite-depth capillary dispersion and ``2 nu k^2``; plus the ``sigma = 0``
  negative control.
* ``marangoni``   -- interfacial shear driven channel, whole ``u_x(y)`` profile
  vs the exact Stokes two-layer solution.
* ``coalescence`` -- two merging 2-D drops, neck growth law and the
  dimensionless capillary time ``tau* = t(r_b=R/2)/sqrt(rho R^3/sigma)``.

Usage
-----
    PYTHONPATH=src python3 tools/surface_tension_dynamics_verification.py \\
        run --suite wave --shard 0 --shards 4 --out artifacts/surface_tension_dynamics
    PYTHONPATH=src python3 tools/surface_tension_dynamics_verification.py \\
        aggregate --out artifacts/surface_tension_dynamics

Every number quoted in the report is read back from the JSON written here; no
value is transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorlbm.surface_tension_dynamics.capillary_wave import (  # noqa: E402
    WaveConfig,
    run_standing_wave,
)
from tensorlbm.surface_tension_dynamics.coalescence import (  # noqa: E402
    CoalescenceConfig,
    run_coalescence,
)
from tensorlbm.surface_tension_dynamics.marangoni import (  # noqa: E402
    MarangoniConfig,
    run_marangoni_channel,
)

# --------------------------------------------------------------------------- #
# case tables (the only place where the run matrix is defined)
# --------------------------------------------------------------------------- #

WAVE_CASES: list[tuple[str, WaveConfig]] = [
    # kW sweep at fixed k = 0.0982 (nx/2 cells per wavelength, periodic in x:
    # the x-box only fixes the mode quantisation, so the cheap box 64 x 64 gives
    # the same physics as 128 x 64 with mode=2 -- same k, same kH = 3.14)
    ("w_k0.098_W1.00", WaveConfig(nx=64, ny=64, mode=1, width=1.0, amplitude=0.3, steps=12000, sample=2, label="kW=0.098")),
    ("w_k0.098_W1.50", WaveConfig(nx=64, ny=64, mode=1, width=1.5, amplitude=0.3, steps=12000, sample=2, label="kW=0.147")),
    ("w_k0.098_W2.00", WaveConfig(nx=64, ny=64, mode=1, width=2.0, amplitude=0.3, steps=12000, sample=2, label="kW=0.196")),
    ("w_k0.098_W4.00", WaveConfig(nx=64, ny=64, mode=1, width=4.0, amplitude=0.3, steps=12000, sample=2, label="kW=0.393")),
    # k sweep at fixed W (a shorter wavelength at the same interface width)
    ("w_k0.196_W2.00", WaveConfig(nx=32, ny=64, mode=1, width=2.0, amplitude=0.3, steps=12000, sample=2, label="k=0.196")),
    # amplitude (nonlinearity) and viscosity controls at the reference geometry
    ("w_k0.098_W2.00_A0.15", WaveConfig(nx=64, ny=64, mode=1, width=2.0, amplitude=0.15, steps=12000, sample=2, label="A=0.15")),
    ("w_k0.098_W2.00_tau0.51", WaveConfig(nx=64, ny=64, mode=1, width=2.0, amplitude=0.3, steps=12000, sample=2, tau=0.51, label="nu/2")),
    # lower-viscosity end of the trend: the deficit shrinks with nu, so this pins
    # down whether it is the collision viscosity (numerical dissipation) or the
    # force normalisation that is responsible.
    ("w_k0.098_W2.00_tau0.505", WaveConfig(nx=64, ny=64, mode=1, width=2.0, amplitude=0.3, steps=12000, sample=2, tau=0.505, label="nu/4")),
    # negative control: no surface tension -> no capillary wave
    ("w_neg_sigma0", WaveConfig(nx=64, ny=64, mode=1, width=2.0, amplitude=0.3, steps=3000, sample=5, sigma=0.0, label="sigma=0")),
]

# NOTE on the run length: the Stokes profile fills the channel by viscous
# diffusion, tau_diff ~ h^2/nu ~ 7700 steps for h = 32 and mu = 0.133.  A 6000-step
# run is therefore *not* converged at Ly = 64 (measured: interior slope 13% low,
# drift 1.4%); the production cases run 24000 steps (channel is only 4 cells wide
# in x, so this costs seconds).
MARANGONI_CASES: list[tuple[str, MarangoniConfig]] = [
    ("m24_ds1e-5", MarangoniConfig(dsdx=1.0e-5, Ly=64, Lx=4, steps=24000, sample=500, label="dsigma/dx=1e-5")),
    ("m24_ds2e-5", MarangoniConfig(dsdx=2.0e-5, Ly=64, Lx=4, steps=24000, sample=500, label="dsigma/dx=2e-5")),
    ("m24_ds4e-5", MarangoniConfig(dsdx=4.0e-5, Ly=64, Lx=4, steps=24000, sample=500, label="dsigma/dx=4e-5")),
    ("m24_ds1e-5_asym", MarangoniConfig(dsdx=1.0e-5, Ly=64, Lx=4, y0=20.0, steps=24000, sample=500, label="asymmetric h1/h2")),
    ("m24_ds1e-5_Ly32", MarangoniConfig(dsdx=1.0e-5, Ly=32, Lx=4, steps=8000, sample=300, label="Ly=32 resolution check")),
]

COALESCENCE_CASES: list[tuple[str, CoalescenceConfig]] = [
    ("c_R12_g2", CoalescenceConfig(radius=12.0, gap=2.0, sigma=0.05, steps=2400, sample=2, label="R=12")),
    ("c_R12_g2_s010", CoalescenceConfig(radius=12.0, gap=2.0, sigma=0.10, steps=2400, sample=2, label="R=12, sigma=0.10")),
]


# --------------------------------------------------------------------------- #
# runners
# --------------------------------------------------------------------------- #

def run_suite(suite: str, out_dir: str, shard: int = 0, shards: int = 1) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    todo: list[tuple[str, Any, Any]] = []
    if suite in ("wave", "all"):
        todo += [("wave", name, cfg) for name, cfg in WAVE_CASES]
    if suite in ("marangoni", "all"):
        todo += [("marangoni", name, cfg) for name, cfg in MARANGONI_CASES]
    if suite in ("coalescence", "all"):
        todo += [("coalescence", name, cfg) for name, cfg in COALESCENCE_CASES]
    todo = [t for i, t in enumerate(todo) if i % shards == shard]
    done: list[str] = []
    for kind, name, cfg in todo:
        path = os.path.join(out_dir, f"{kind}__{name}.json")
        if os.path.exists(path):
            done.append(path)
            continue
        t0 = time.time()
        if kind == "wave":
            res = run_standing_wave(cfg)
        elif kind == "marangoni":
            res = run_marangoni_channel(cfg)
        else:
            res = run_coalescence(cfg)
        res["wall_seconds"] = round(time.time() - t0, 1)
        res["case_name"] = name
        with open(path, "w") as fh:
            json.dump(res, fh)
        done.append(path)
        print(f"[shard {shard}] {kind}/{name} done in {res['wall_seconds']}s", flush=True)
    return {"suite": suite, "shard": shard, "shards": shards, "cases": done}


def _load(out_dir: str, prefix: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not os.path.isdir(out_dir):
        return out
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith(prefix) and fn.endswith(".json") and "__" in fn:
            with open(os.path.join(out_dir, fn)) as fh:
                d = json.load(fh)
            out[fn[len(prefix) + 2 : -5]] = d
    return out


def aggregate(out_dir: str) -> dict[str, Any]:
    """Collect all case JSONs into one results file + the criteria verdicts."""
    waves = _load(out_dir, "wave")
    mar = _load(out_dir, "marangoni")
    coa = _load(out_dir, "coalescence")
    summary: dict[str, Any] = {"wave": {}, "marangoni": {}, "coalescence": {}, "criteria": {}}

    for name, r in waves.items():
        if "negative_control" in r:
            summary["wave"][name] = {
                "negative_control": r["negative_control"],
                "sharpness_initial": r["sharpness_initial"],
                "sharpness_final": r["sharpness_final"],
                "params": r["params"],
            }
            continue
        summary["wave"][name] = {
            "k": r["params"]["k"],
            "kW": r["params"]["kW"],
            "width": r["params"].get("width"),
            "amplitude": r["params"].get("amplitude"),
            "tau": r["params"].get("tau"),
            "steps": r["params"].get("steps"),
            "kH": r["params"]["k_half_depth"],
            "omega_theory": r["theory"]["omega"],
            "omega_theory_deep": r.get("omega_deep_theory"),
            "omega_meas": r["metrics"]["omega_meas"],
            "omega_rel_err": r["metrics"]["omega_rel_err"],
            "omega_cross": r.get("omega_cross"),
            "omega_cross_rel_err": r.get("omega_cross_rel_err"),
            "omega_fft": r.get("omega_fft"),
            "omega_fft_rel_err": r.get("omega_fft_rel_err"),
            "sigma_eff_over_sigma": 1.0 + r["metrics"]["sigma_eff_rel_err"],
            "half_period_spread": r["extrema"].get("spread"),
            "n_half_periods": r["extrema"].get("n_half_periods"),
            "fit_windows": r["fit_windows"],
            "sharpness_final": r["sharpness_final"],
            "mass_drift_rel": r["mass_drift_rel"],
            "wall_seconds": r.get("wall_seconds"),
        }

    for name, r in mar.items():
        summary["marangoni"][name] = {
            "dsdx": r["params"]["dsdx"],
            "Ly": r["params"]["Ly"],
            "y0": r["params"]["y0"],
            "h1": r["params"].get("h1"),
            "h2": r["params"].get("h2"),
            "steps": r["params"].get("steps"),
            "u_interface_theory": r["params"]["u_interface_theory"],
            "metrics": r["metrics"],
            "wall_seconds": r.get("wall_seconds"),
        }

    for name, r in coa.items():
        summary["coalescence"][name] = {
            "radius": r["params"]["radius"],
            "gap": r["params"].get("gap"),
            "sigma": r["params"]["sigma"],
            "mu": r["params"].get("mu"),
            "Oh": r["params"]["Oh"],
            "tau_ci": r["params"]["tau_ci"],
            "metrics": r["metrics"],
            "wall_seconds": r.get("wall_seconds"),
        }

    # ---- criteria -----------------------------------------------------------
    ref = summary["wave"].get("w_k0.098_W2.00", {})
    errs = [v["omega_rel_err"] for k, v in summary["wave"].items() if "omega_rel_err" in v]
    neg = [v for k, v in summary["wave"].items() if "negative_control" in v]
    mar_errs = [v["metrics"]["profile_rel_l2"] for v in summary["marangoni"].values()]
    coa_tau = [v["metrics"]["tau_star"] for v in summary["coalescence"].values()]
    coa_exp = [v["metrics"]["growth_exponent"] for v in summary["coalescence"].values()]

    def ok(x: Any, lim: float) -> bool:
        return bool(x is not None and abs(x) <= lim)

    # ---- softening law: |omega error| = a * (kW)^p on the kW sweep at fixed k
    # only the W sweep at fixed k enters the law; the A/tau controls are not part
    # of it, and a case whose amplitude record was not a single damped sinusoid
    # (|error| > 50% -> the extrema detector locked onto numerical jitter) is
    # flagged instead of being silently averaged in.
    kw_pts = []
    unreliable: list[str] = []
    for name, v in summary["wave"].items():
        if "omega_rel_err" not in v:
            continue
        if not name.startswith("w_k0.098_W") or "A0.15" in name or "tau" in name:
            continue
        err = v["omega_rel_err"]
        if err != err:
            continue
        if abs(err) > 0.5:
            unreliable.append(name)
            continue
        kw_pts.append((v["kW"], abs(err)))
    law: dict[str, Any] = {"n_points": len(kw_pts), "unreliable_cases": unreliable}
    if len(kw_pts) >= 3:
        xs = [math.log(p[0]) for p in kw_pts]
        ys = [math.log(p[1]) for p in kw_pts if p[1] > 0]
        if len(ys) == len(xs):
            n = len(xs)
            sx, sy = sum(xs), sum(ys)
            sxx = sum(x * x for x in xs)
            sxy = sum(x * y for x, y in zip(xs, ys))
            den = n * sxx - sx * sx
            p_exp = (n * sxy - sx * sy) / den
            a_coef = math.exp((sy - p_exp * sx) / n)
            ybar = sy / n
            ss_tot = sum((y - ybar) ** 2 for y in ys)
            ss_res = sum((y - (math.log(a_coef) + p_exp * x)) ** 2 for x, y in zip(xs, ys))
            law.update(
                {
                    "a": a_coef,
                    "p": p_exp,
                    "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
                    "kW_at_5pct": (0.05 / a_coef) ** (1.0 / p_exp) if a_coef > 0 else None,
                    "points": kw_pts,
                }
            )
    summary["criteria"] = {
        "C1c_softening_law": law,
        "C1_wave_reference_lt5pct": {
            "case": "w_k0.098_W2.00",
            "value": ref.get("omega_rel_err"),
            "limit": 0.05,
            "pass": ok(ref.get("omega_rel_err"), 0.05),
        },
        "C1b_wave_all_cases": {
            "max_abs_err": max((abs(e) for e in errs), default=None),
            "n_cases": len(errs),
            "n_within_5pct": sum(1 for e in errs if abs(e) <= 0.05),
        },
        "C2_negative_control": {
            "cases": [
                {
                    "name": k,
                    "amp_max_dev": v["negative_control"]["amp_max_dev"],
                    "amp_initial": v["negative_control"]["amp_initial"],
                    "umax": v["negative_control"]["umax"],
                }
                for k, v in summary["wave"].items()
                if "negative_control" in v
            ],
            "pass": all(
                v["negative_control"]["amp_max_dev"] <= 0.05 * max(v["negative_control"]["amp_initial"], 1e-30)
                for v in summary["wave"].values()
                if "negative_control" in v
            ),
        },
        "C3_marangoni_profile_lt10pct": {
            "max_profile_rel_l2": max(mar_errs, default=None),
            "per_case": {k: v["metrics"]["profile_rel_l2"] for k, v in summary["marangoni"].items()},
            "pass": bool(mar_errs) and max(mar_errs) <= 0.10,
        },
        "C4_coalescence": {
            "tau_star": {k: v["metrics"]["tau_star"] for k, v in summary["coalescence"].items()},
            "growth_exponent": {k: v["metrics"]["growth_exponent"] for k, v in summary["coalescence"].items()},
            "note": "tau* is compared with the experimental literature in the report; "
            "the growth exponent is compared with the inertial self-similar prediction 2/3",
        },
    }
    path = os.path.join(out_dir, "results.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"wrote {path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["run", "aggregate"])
    ap.add_argument("--suite", default="all", choices=["wave", "marangoni", "coalescence", "all"])
    ap.add_argument("--out", default="artifacts/surface_tension_dynamics")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args(argv)
    if args.mode == "run":
        print(json.dumps(run_suite(args.suite, args.out, args.shard, args.shards)))
    else:
        s = aggregate(args.out)
        print(json.dumps(s["criteria"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())