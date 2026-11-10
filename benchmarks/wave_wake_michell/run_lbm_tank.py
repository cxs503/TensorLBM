#!/usr/bin/env python3
"""LBM free-surface leg: 2-D numerical wave tank dispersion / mass audit.

The 3-D free-surface ship run is out of budget (see the mission report: the
Körner 3-D kernel moves ~4e5 cells at ~1e1 s/step on the control plane, i.e.
O(10^5 s) for a Wigley case at the campaign's lambda/dx >= 20).  What *is*
affordable -- and what the dispersion/angle criteria actually need -- is a
2-D free-surface tank at resolved wavelength, which measures

* the linear dispersion ``omega^2 = g k tanh(k h)`` recovered by the Körner
  free-surface LBM (celerity from the probe phasors),
* the incident-amplitude error against the Stokes-2 target,
* the reflection coefficient of the sponge (did the domain absorb the wake?),
* the mass drift of the liquid ledger over thousands of steps.

Each of those is an acceptance-relevant number for the wave-resistance
campaign, and none of them is a hard-coded expectation: everything is measured
from the probe records the tank returns.

STATUS: secondary harness, superseded by ``run_lbm_dispersion.py`` for the numbers
quoted in REPORT.md.  Measured with the tank defaults (``eps=0.02`` -> crest
amplitude ``a = eps/k = 0.16`` lattice cells, i.e. below the resolution floor,
and ``n_periods=1.5`` shorter than the reflection fit window) the incident
amplitude error comes out at ~285% and the mass figure is dominated by the
wavemaker's own injection -- both are artefacts of that configuration, not
physics.  Kept because it is the only path that exercises the relaxation-zone
wavemaker + sponge, which a future 3-D route will need; do not quote it.

Run::

    PYTHONPATH=src python benchmarks/wave_wake_michell/run_lbm_tank.py \
        --out runs/wave_wake_michell/lbm_tank.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from tensorlbm.wave_tank2d.tank import TankConfig, run_tank


def _phasor(series: np.ndarray, t: np.ndarray, omega: float) -> complex:
    """Least-squares phasor at ``omega`` (no window, DC removed)."""
    s = series - np.mean(series)
    c = np.cos(omega * t)
    n = np.sin(omega * t)
    a = np.vstack([c, n]).T
    coef, *_ = np.linalg.lstsq(a, s, rcond=None)
    return complex(coef[0], -coef[1])


def analyse(case: dict, res: dict) -> dict:
    """Celerity / wavenumber measured from the probe phasors."""
    wave = res["wave"]
    omega = float(wave["omega"])
    t = np.asarray(res["t"], float)
    xs = np.asarray(res["geometry"]["probe_x"], float)
    mask = t > t[-1] - 2.0 * (2.0 * math.pi / omega)
    tm = t[mask]
    z = np.array([_phasor(np.asarray(res["probes"][str(int(x))], float)[mask], tm, omega)
                  for x in xs])
    # adjacent probes are lambda/4 apart: dphi = k * ds (mod 2 pi), unwrap locally
    ds = np.diff(xs)
    dphi = np.angle(z[1:] / z[:-1])
    k_est = float(np.median(np.abs(dphi) / ds))
    c_est = omega / k_est if k_est > 0 else float("nan")
    k_theory = float(wave["k"])
    c_theory = float(wave["phase_speed"])
    lam_cells = 2.0 * math.pi / k_theory
    return {
        "case": case,
        "h": float(res["config"]["h"]),
        "kh": float(res["config"]["kh"]),
        "wavelength_cells": lam_cells,
        "lambda_over_dx": lam_cells,
        "celerity_measured": c_est,
        "celerity_theory": c_theory,
        "celerity_error_pct": 100.0 * (c_est - c_theory) / c_theory,
        "wavenumber_measured": k_est,
        "wavenumber_theory": k_theory,
        "wavenumber_error_pct": 100.0 * (k_est - k_theory) / k_theory,
        "omega": omega,
        "incident_amplitude_error": float(res["reflection"]["incident_amplitude_error"]),
        "reflection_K_R": float(res["reflection"]["K_R_least_squares"]),
        "fit_rms_residual": float(res["reflection"]["fit_rms_residual"]),
        "mass_relative_drift": float(res["mass"]["relative_drift"]),
        "mass_injected_per_period": float(res["mass"]["injected_per_period"]),
        "n_steps": int(res["config"]["n_steps"]),
        "runtime_s": float(res["config"]["runtime_s"]),
        "grid": {"nx": int(res["geometry"]["nx"]), "ny": int(res["geometry"]["ny"])},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/wave_wake_michell/lbm_tank.json")
    ap.add_argument("--h", type=float, default=32.0)
    ap.add_argument("--periods", type=float, default=3.0)
    ap.add_argument("--kh-list", default="1.5,3.0,6.0")
    ap.add_argument("--eps", type=float, default=0.02)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    for kh in [float(v) for v in args.kh_list.split(",") if v.strip()]:
        t0 = time.time()
        cfg = TankConfig(h=float(args.h), kh=float(kh), eps=float(args.eps),
                         n_periods=float(args.periods), record_every=10)
        print(f"[tank] h={cfg.h} kh={cfg.kh} eps={cfg.eps}", flush=True)
        res = run_tank(cfg, verbose=True)
        rec = analyse({"h": cfg.h, "kh": cfg.kh, "eps": cfg.eps}, res)
        rec["wall_clock_s"] = time.time() - t0
        cases.append(rec)
        print(
            "[tank] kh=%.2f c=%.5f (theory %.5f, %+.2f%%) mass_drift=%.3e KR=%.3f "
            "amp_err=%.3f" % (
                rec["kh"], rec["celerity_measured"], rec["celerity_theory"],
                rec["celerity_error_pct"], rec["mass_relative_drift"],
                rec["reflection_K_R"], rec["incident_amplitude_error"]),
            flush=True,
        )

    ref = next((c for c in cases if c["kh"] == 3.0), cases[len(cases) // 2])
    summary = {
        "n_cases": len(cases),
        "lambda_over_dx": [c["lambda_over_dx"] for c in cases],
        "celerity_error_pct": [c["celerity_error_pct"] for c in cases],
        "max_abs_mass_drift": max(abs(c["mass_relative_drift"]) for c in cases),
        "max_abs_celerity_error_pct": max(abs(c["celerity_error_pct"]) for c in cases),
        "reference_case": ref["case"],
        "note": (
            "2-D free-surface LBM (Korner GALS) dispersion/mass audit; the 3-D "
            "ship-scale wave-resistance run is not affordable in this budget"
        ),
    }
    payload = {"cases": cases, "summary": summary,
               "env": {"torch_threads": os.environ.get("TL_WT_THREADS", "1")}}
    out.write_text(json.dumps(payload, indent=2))
    print(f"[done] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())