#!/usr/bin/env python3
"""Extra P87W2-N01 runs that the report needs but the case sweep does not store.

Three records, each saved as ``.npz`` under ``artifacts/p87w2_n01/extra``:

``waveform``  h = 0.20 m, mode 1, free slip, base grid: the full ``eta(x,t)``
              field (for the free-surface contour figure), the modal series and
              the wall-pressure gauges — the case used for the phase figure.
``friction``  the same case with **no-slip** walls: the viscous detuning
              sensitivity (``f_d`` shift and extra damping ``zeta``).
``impact``    h = 0.40 m (fill ratio 0.8), sway ``X0 = 0.04 L`` at ``omega_1``:
              full wall-pressure time history for the impact figure.

Usage::

    python scripts/run_sloshing_resonance_extras.py --out artifacts/p87w2_n01/extra --job waveform
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("TL_THREADS", "1")))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.sloshing_resonance import (  # noqa: E402
    CampaignSpec,
    LatticeScaling,
    SloshingTank,
    abramson_angular_frequency,
    abramson_frequency,
    build_tank_field,
    identify_mode,
    run_free_decay,
    run_sway,
)


def _save_free_decay(res, path: Path, spec: CampaignSpec, tag: str) -> dict:
    t = res["t"].numpy()
    eta_field = res["eta"].numpy()
    modal = res["eta_modes"][1].numpy()
    press = res["pressure"].numpy()
    h = res["depth_cells"] * res["dx"]
    f_th = abramson_frequency(1, h, spec.length, spec.g)
    fit = identify_mode(t, modal, f_th)
    np.savez_compressed(
        path,
        t=t,
        eta=eta_field,
        modal=modal,
        pressure=press,
        gauge_rows=np.array(res["gauge_rows"]),
        inventory=res["inventory"].numpy(),
        dx=res["dx"],
        depth=h,
        f_theory=f_th,
        f_damped=fit.f_damped,
        f_natural=fit.f_natural,
        zeta=fit.zeta,
        amplitude=0.01,
    )
    return {
        "tag": tag,
        "depth_m": h,
        "slip_walls": bool(spec.slip_walls),
        "n_steps": res["n_steps"],
        "f_theory": f_th,
        "f_damped": fit.f_damped,
        "f_natural": fit.f_natural,
        "zeta": fit.zeta,
        "phase_error_deg": fit.phase_error_deg,
        "max_mass_drift": res["max_mass_drift"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "p87w2_n01" / "extra")
    ap.add_argument("--job", choices=["waveform", "friction", "impact"], required=True)
    ap.add_argument("--periods", type=float, default=4.0)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    tank = SloshingTank()
    base = dict(cells_per_height=40, u_lat=0.002, tau=0.502)
    t0 = time.time()

    if args.job == "waveform":
        spec = CampaignSpec(slip_walls=True, **base)
        sc = LatticeScaling.from_resolution(tank, 40, u_lat=0.002, tau=0.502)
        st = build_tank_field(tank, sc, depth=0.20, mode=1, amplitude=0.01, headroom=0.06)
        res = run_free_decay(
            st, args.periods, record_every=20, modes=(1, 2), gauge_rows=[15, 8, 1], slip_walls=True
        )
        meta = _save_free_decay(res, args.out / "waveform.npz", spec, "waveform_slip")
        # dense eta snapshot at t = 0 for the "initial condition" panel
        (args.out / "waveform.json").write_text(json.dumps(meta, indent=2))

    elif args.job == "friction":
        spec = CampaignSpec(slip_walls=False, **base)
        sc = LatticeScaling.from_resolution(tank, 40, u_lat=0.002, tau=0.502)
        st = build_tank_field(tank, sc, depth=0.20, mode=1, amplitude=0.01, headroom=0.06)
        res = run_free_decay(
            st, args.periods, record_every=20, modes=(1, 2), gauge_rows=[15, 8, 1], slip_walls=False
        )
        meta = _save_free_decay(res, args.out / "friction.npz", spec, "waveform_noslip")
        (args.out / "friction.json").write_text(json.dumps(meta, indent=2))

    else:  # impact
        h, ratio, periods = 0.40, 0.04, 6.0
        sc = LatticeScaling.from_resolution(tank, 40, u_lat=0.002, tau=0.502)
        st = build_tank_field(tank, sc, depth=h, mode=1, amplitude=0.0, headroom=0.10)
        omega = abramson_angular_frequency(1, h, tank.length, tank.g)
        res = run_sway(
            st,
            periods,
            sway_amplitude=ratio * tank.length,
            omega=omega,
            record_every=4,
            modes=(1,),
            gauge_rows=[max(1, int(round(h / sc.dx)) - 1), int(round(0.5 * h / sc.dx)), 1],
            slip_walls=True,
        )
        t = res["t"].numpy()
        press = res["pressure"].numpy()
        modal = res["eta_modes"][1].numpy()
        ok = np.isfinite(press)
        np.savez_compressed(
            args.out / "impact.npz",
            t=t,
            pressure=press,
            modal=modal,
            gauge_rows=np.array(res["gauge_rows"]),
            depth=h,
            omega=omega,
            sway=ratio * tank.length,
        )
        meta = {
            "depth_m": h,
            "fill_ratio": h / tank.height,
            "sway_amplitude_m": ratio * tank.length,
            "omega": omega,
            "n_steps": res["n_steps"],
            "p_peak_pa": float(np.max(press[ok])) if ok.any() else None,
            "eta_max_m": float(modal.max()),
            "max_mass_drift": res["max_mass_drift"],
        }
        (args.out / "impact.json").write_text(json.dumps(meta, indent=2))

    print(f"[{args.job}] done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
