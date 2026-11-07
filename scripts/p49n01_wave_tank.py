#!/usr/bin/env python3
"""P49-N01 — Stokes second-order wave-tank benchmark driver (TensorLBM).

Three sub-commands, each writing self-describing artefacts into ``--out``:

``periodic``
    Run one periodic-domain propagation case (dispersion / harmonics / mass).
``tank``
    Run one relaxation-zone wave-tank case (wavemaker + optional sponge) and
    measure the reflection coefficient with the Goda–Suzuki two-point method.
``report``
    Merge every JSON found in ``--out`` into ``summary.json`` and draw the
    report figures.

Examples
--------
    python scripts/p49n01_wave_tank.py periodic --case kh1.5_eps0.02
    python scripts/p49n01_wave_tank.py tank --case sponge_on
    python scripts/p49n01_wave_tank.py report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.wave_tank2d.periodic import PeriodicWaveConfig, run_periodic  # noqa: E402
from tensorlbm.wave_tank2d.tank import TankConfig, run_tank  # noqa: E402

DEFAULT_OUT = Path("artifacts/P49_N01")

# ---------------------------------------------------------------------------
# case tables (single source of truth for the campaign)
# ---------------------------------------------------------------------------
#: periodic-domain cases: (depth, relative depth, steepness, periods, tag)
PERIODIC_CASES: dict = {
    # dispersion relation, three relative depths (kh = 0.7 / 1.5 / 3.0)
    "kh3.0_eps0.02": dict(h=20.0, kh=3.0, eps=0.02, n_periods=2.0),
    "kh3.0_eps0.08": dict(h=20.0, kh=3.0, eps=0.08, n_periods=2.0),
    "kh1.5_eps0.02": dict(h=20.0, kh=1.5, eps=0.02, n_periods=2.0),
    "kh1.5_eps0.08": dict(h=20.0, kh=1.5, eps=0.08, n_periods=2.0),
    "kh0.7_eps0.02": dict(h=14.0, kh=0.7, eps=0.02, n_periods=2.0),
    "kh0.7_eps0.08": dict(h=14.0, kh=0.7, eps=0.08, n_periods=2.0),
    # kh = 3.0 is under-resolved at h = 20 (42 cells/wavelength): refined family
    "kh3.0_hi_eps0.02": dict(h=32.0, kh=3.0, eps=0.02, n_periods=2.0),
    "kh3.0_hi_eps0.08": dict(h=32.0, kh=3.0, eps=0.08, n_periods=2.0),
    # long run: liquid-mass conservation over five periods
    "mass_5periods": dict(h=20.0, kh=1.5, eps=0.02, n_periods=5.0),
    # grid-convergence family: fixed physical wave, three resolutions
    "grid_coarse": dict(h=13.0, kh=1.5, eps=0.02, n_periods=2.0),
    "grid_medium": dict(h=20.0, kh=1.5, eps=0.02, n_periods=2.0),
    "grid_fine": dict(h=28.0, kh=1.5, eps=0.02, n_periods=2.0),
}

#: wave-tank cases: reflection verification and its negative control
TANK_CASES: dict = {
    "sponge_on": dict(h=20.0, kh=1.5, eps=0.02, sponge=True, n_periods=4.5),
    "sponge_off": dict(h=20.0, kh=1.5, eps=0.02, sponge=False, n_periods=4.5),
    # the reflected train needs ~3.5 T to reach the end wall and ~2 T to come
    # back to the probe row, so a 4.5 T record still under-reads K_R: the long
    # run is the one that carries the negative-control verdict
    "sponge_off_long": dict(h=20.0, kh=1.5, eps=0.02, sponge=False, n_periods=7.5),
}

TAU = float(os.environ.get("P49N01_TAU", "0.502"))


def _save_periodic(tag: str, out: Path) -> dict:
    spec = PERIODIC_CASES[tag]
    cfg = PeriodicWaveConfig(tau=TAU, hydrostatic_init=True, record_every=4, **spec)
    res = run_periodic(cfg, verbose=True)
    fields = res.get("field_matrix", {})
    np.savez(out / f"periodic_{tag}_fields.npz",
             eta_matrix=np.array(fields.get("eta", [])),
             t_matrix=np.array(fields.get("t", [])),
             eta_near_probe=np.array(res["series"]["eta_probe0"]),
             t=np.array(res["series"]["t"]),
             amplitude_profile=np.array(res["series"]["amplitude_profile"]),
             eta_snapshot=np.array(res["series"]["eta_snapshot_final"]),
             eta_analytic=np.array(res["eta_analytic_final"]),
             k=res["wave"]["k"], omega=res["wave"]["omega"], a=res["wave"]["a"],
             period=res["wave"]["period"], h=res["wave"]["h"])
    payload = {"tag": tag, "kind": "periodic", **res}
    payload.pop("series", None)
    payload.pop("field_matrix", None)
    (out / f"periodic_{tag}.json").write_text(json.dumps(payload, indent=1))
    return payload


def _save_tank(tag: str, out: Path) -> dict:
    spec = TANK_CASES[tag]
    cfg = TankConfig(tau=TAU, **spec)
    res = run_tank(cfg, verbose=True)
    fields = res.pop("snapshots", {})
    np.savez(out / f"tank_{tag}_probes.npz",
             snapshot_keys=np.array(list(fields)),
             t=np.array(res["t"]),
             **{f"eta_x{x}": np.array(v) for x, v in res["probes"].items()},
             probe_s=np.array(res["geometry"]["probe_s"]),
             probe_x=np.array(res["geometry"]["probe_x"]),
             **{f"eta_field_{k_}": np.array(v) for k_, v in fields.items()})
    payload = {"tag": tag, "kind": "tank", **res}
    (out / f"tank_{tag}.json").write_text(json.dumps(payload, indent=1))
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["periodic", "tank", "report"])
    ap.add_argument("--case", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.command == "periodic":
        cases = [args.case] if args.case else list(PERIODIC_CASES)
        for tag in cases:
            print(f"[p49n01] periodic case {tag}", flush=True)
            r = _save_periodic(tag, out)
            d, hm = r["dispersion"], r["harmonics"]
            print(f"[p49n01] {tag}: c_err={d['c_rel_error']*100:.3f}% "
                  f"a1_err={hm['a1_rel_error']*100:.2f}% "
                  f"a2_err={hm['a2_rel_error']*100:.1f}%", flush=True)
        return 0

    if args.command == "tank":
        cases = [args.case] if args.case else list(TANK_CASES)
        for tag in cases:
            print(f"[p49n01] tank case {tag}", flush=True)
            r = _save_tank(tag, out)
            print(f"[p49n01] {tag}: K_R={r['reflection']['K_R_least_squares']:.4f}",
                  flush=True)
        return 0

    from tensorlbm.wave_tank2d.report import build_report  # local import: matplotlib

    build_report(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())