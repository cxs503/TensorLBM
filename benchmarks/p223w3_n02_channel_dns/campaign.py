#!/usr/bin/env python3
"""p223w3-n02: wall-model-free turbulent channel DNS/LES verification campaign.

Runs the native D3Q19 TRT + Smagorinsky kernel
(``src/tensorlbm/turbulence/les_chan.c``) for the three target friction
Reynolds numbers, reduces every case to wall units, and stores one JSON per
case under ``benchmarks/p223w3_n02_channel_dns/case_data/``.

Design choices that matter for the numbers (see the report, section 2):

* ``u_tau`` is *not* a free knob for accuracy, it is the time-scale knob:
  one eddy turnover is ``h / u_tau`` lattice steps, so the statistics budget
  scales like ``1 / u_tau``.  It is pushed to the compressibility envelope
  (``u_tau = 0.0125``  ->  ``U_cl ~ 20 u_tau = 0.25``, Ma<sub>lattice</sub>~0.43)
  which is the largest value for which the centreline stays subsonic.
* Domain ``Lx x Lz = 4h x 2h`` (i.e. ``4 pi`` wall units short of the DNS box
  ``6.28h x 3.14h``): a deliberately cheaper box, so very-large-scale motions
  are truncated.  This is recorded as a limitation, not hidden.
* ``dy+`` is uniform; the wall-normal ladders are the grid-convergence study.

Usage::

    python benchmarks/p223w3_n02_channel_dns/campaign.py --case R180_dy3
    python benchmarks/p223w3_n02_channel_dns/campaign.py --list
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_HERE))

from tensorlbm.turbulence.les_channel import LesChannelConfig, run_native  # noqa: E402

import analysis as an  # noqa: E402

RESULTS = _HERE / "case_data"
U_TAU = 0.0125  # lattice friction velocity: fixed time-step scale for all cases


@dataclass(frozen=True)
class Case:
    name: str
    re_tau: float
    dy_plus: float
    warmup_turns: float
    stat_turns: float
    cs: float = 0.1
    lx_over_h: float = 4.0
    lz_over_h: float = 2.0
    dx_over_dy: float = 2.0
    u_tau: float = U_TAU
    allow_high_mach: bool = False
    threads: int = 48
    note: str = ""
    # populated by geometry()
    nx: int = field(init=False, default=0)
    ny: int = field(init=False, default=0)
    nz: int = field(init=False, default=0)
    steps: int = field(init=False, default=0)
    warmup: int = field(init=False, default=0)
    half_height: float = field(init=False, default=0.0)

    def geometry(self) -> "Case":
        h = self.re_tau / self.dy_plus          # half height in lattice units
        ny = int(round(2 * h)) + 2
        h = 0.5 * (ny - 2)                      # exact half height of the grid
        dx_plus = self.dx_over_dy * self.dy_plus
        nx = int(round(self.lx_over_h * self.re_tau / dx_plus))
        nz = int(round(self.lz_over_h * self.re_tau / dx_plus))
        steps_turn = h / self.u_tau
        warmup = int(round(self.warmup_turns * steps_turn))
        stat = int(round(self.stat_turns * steps_turn))
        steps = warmup + stat
        object.__setattr__(self, "nx", nx)
        object.__setattr__(self, "ny", ny)
        object.__setattr__(self, "nz", nz)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "warmup", warmup)
        object.__setattr__(self, "half_height", h)
        return self

    @property
    def cells(self) -> int:
        return self.nx * self.ny * self.nz


def case_matrix() -> dict[str, Case]:
    """The campaign: 3 Reynolds numbers + a Re_tau=180 wall-normal ladder."""
    # Wall-normal spacings were chosen against a hard compute budget (single
    # shared 128-core host, ~16 Mcell-steps/s aggregate): cost ~ cells x steps
    # and steps ~ Re_tau / (u_tau dy+), so Delta+ is the only free knob once the
    # statistics length is fixed.  Re_tau = 180 gets the finest ladder;
    # Re_tau = 395/590 are deliberately coarse (stated as a limitation).
    raw = [
        Case("R180_dy4", 180.0, 4.0, warmup_turns=3.0, stat_turns=9.0,
             note="primary Re_tau=180 case"),
        Case("R180_dy5", 180.0, 5.0, warmup_turns=3.0, stat_turns=7.0,
             note="grid ladder (medium)"),
        Case("R180_dy3", 180.0, 3.0, warmup_turns=3.0, stat_turns=8.0,
             note="primary Re_tau=180 case (finest wall-normal spacing)"),
        Case("R180_dy3c", 180.0, 3.0, warmup_turns=3.0, stat_turns=8.0,
             dx_over_dy=3.3333, note="grid ladder (coarser streamwise/spanwise)"),
        # Relaxation-time margin beats compressibility here: tau0 = 0.5 + 3 u_tau/dy+
        # so u_tau = 0.02 at dy+ = 3 gives tau0 = 0.52 (the only band where long
        # runs survived), at the price of a centreline Ma ~ 0.7.
        Case("R180_u02", 180.0, 3.0, warmup_turns=2.0, stat_turns=6.0,
             u_tau=0.02, allow_high_mach=True, note="high-tau0 stabilised primary case"),
        Case("R180_u02c", 180.0, 3.0, warmup_turns=2.0, stat_turns=6.0,
             u_tau=0.02, allow_high_mach=True, dx_over_dy=3.3333,
             note="high-tau0 grid ladder (coarser x/z)"),
        Case("R395_dy8", 395.0, 8.0, warmup_turns=2.0, stat_turns=8.0,
             cs=0.2, note="Re_tau=395, coarse wall-normal spacing, Cs=0.2"),
        Case("R590_dy12", 590.0, 12.0, warmup_turns=2.0, stat_turns=6.0,
             cs=0.2, note="Re_tau=590, very coarse wall-normal spacing, Cs=0.2"),
    ]
    return {c.name: c.geometry() for c in raw}


#: cases that were run before the final matrix was fixed and are only kept so the
#: archived divergence logs can be parsed into machine-readable records.
RETIRED = {"R180_dy6": Case("R180_dy6", 180.0, 6.0, 3.0, 7.0).geometry()}

LAMINAR = Case("LAM_poiseuille", 10.0, 10.0 / 15.0, warmup_turns=0.0, stat_turns=0.0,
               note="laminar negative control (gate e)").geometry()


# ---------------------------------------------------------------------------
# one case
# ---------------------------------------------------------------------------

def reduce_case(payload: dict, case: Case) -> dict:
    """Turn a raw kernel payload into the reduced, gate-ready record."""
    derived = payload["derived"]
    derived["u_tau"] = payload["configuration"]["u_tau"]
    prof = an.build_profiles(payload["profiles"], derived)
    routes = an.friction_routes(payload["profiles"], derived)
    u_tau = routes["body_force"]

    u_bulk = float(payload["bulk_velocity"])
    height = float(derived["height"])
    nu = float(derived["nu"])
    re_bulk = an.bulk_reynolds(u_bulk, height, nu)
    cf = an.cf_from_bulk(u_tau, u_bulk)
    cf_dean = an.dean_cf(re_bulk)

    log = an.log_law_error(prof, 30.0, 200.0)
    reich = an.reichardt_error(prof, 5.0, 200.0)
    uu_peak = an.peak(prof.uu_plus, prof.y_plus)
    uv_peak = an.peak(-prof.uv_plus, prof.y_plus)

    # streamwise spectrum from the mid-plane slice
    lx = float(case.nx)
    k, e = an.spectrum_from_csv(str(RESULTS / f"{case.name}_xzm.csv"), lx)
    k_plus = k * nu / u_tau
    spec = an.power_law_slope(k_plus, e, 0.05, 0.30)
    spec["k_plus_units"] = True

    record = {
        "case": case.name,
        "note": case.note,
        "geometry": {
            "nx": case.nx, "ny": case.ny, "nz": case.nz, "cells": case.cells,
            "half_height": case.half_height, "re_tau": case.re_tau,
            "dy_plus": case.dy_plus, "u_tau": case.u_tau, "nu": nu,
            "tau0": float(derived["tau0"]),
            "steps": case.steps, "warmup": case.warmup,
            "stat_eddy_turnovers": (case.steps - case.warmup) * case.u_tau
            / case.half_height,
            "lx_over_h": case.nx / case.half_height,
            "lz_over_h": case.nz / case.half_height,
            "smagorinsky_cs": case.cs,
            "regularized": payload["configuration"].get("regularized", 0),
        },
        "mean_flow": {
            "u_bulk": u_bulk,
            "re_bulk": re_bulk,
            "u_tau_routes": routes,
            "cf": cf,
            "cf_dean": cf_dean,
            "cf_error_vs_dean": (cf - cf_dean) / cf_dean,
            "re_tau_achieved": float(derived["half_height"]) * u_tau / nu,
        },
        "profiles": {
            "y_plus": prof.y_plus.tolist(),
            "u_plus": prof.u_plus.tolist(),
            "uu_plus": prof.uu_plus.tolist(),
            "vv_plus": prof.vv_plus.tolist(),
            "ww_plus": prof.ww_plus.tolist(),
            "uv_plus": prof.uv_plus.tolist(),
            "y_over_h": prof.y_over_h.tolist(),
        },
        "diagnostics": {
            "log_law": log,
            "reichardt": reich,
            "uu_peak": uu_peak,
            "uv_minus_peak": uv_peak,
            "mass_drift_fraction": payload["diagnostics"]["mass_drift_fraction"],
            "max_speed": payload["diagnostics"]["max_speed"],
            "bulk_velocity_series": payload["bulk_velocity_series"],
        },
        "spectrum": {
            "k_plus": k_plus.tolist(),
            "e": e.tolist(),
            "fit": spec,
        },
        "provenance": payload.get("provenance", {}),
    }
    return record


def run_case(case: Case, out_dir: Path = RESULTS, overwrite: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{case.name}.json"
    if target.exists() and not overwrite:
        return json.loads(target.read_text())
    cfg = LesChannelConfig(
        nx=case.nx, ny=case.ny, nz=case.nz, steps=case.steps, warmup=case.warmup,
        sample=max(5, int(round(case.steps / 400))), re_tau=case.re_tau,
        u_tau=case.u_tau, cs=case.cs, perturb=0.05, perturb_modes=6,
        seed=20261110, threads=case.threads, regularized=0,
        allow_high_mach=case.allow_high_mach,
    )
    started = time.time()
    try:
        payload = run_native(cfg, out_dir, case.name, overwrite=True)
    except RuntimeError as exc:
        # honest bookkeeping: a diverged case leaves a record, not silence
        (out_dir / f"{case.name}.failed.json").write_text(json.dumps({
            "case": case.name, "note": case.note,
            "geometry": {"nx": case.nx, "ny": case.ny, "nz": case.nz,
                         "cells": case.cells, "re_tau": case.re_tau,
                         "dy_plus": case.dy_plus, "u_tau": case.u_tau,
                         "steps": case.steps, "warmup": case.warmup,
                         "smagorinsky_cs": case.cs},
            "error": str(exc).splitlines()[0] if str(exc) else "unknown",
            "wall_clock_seconds": time.time() - started,
        }, indent=1))
        raise
    record = reduce_case(payload, case)
    record["wall_clock_seconds"] = time.time() - started
    target.write_text(json.dumps(record, indent=1))
    print(f"[{case.name}] ubulk={record['mean_flow']['u_bulk']:.4f} "
          f"cf={record['mean_flow']['cf']:.5f} "
          f"log_rms={record['diagnostics']['log_law']['rms_rel_error']:.4f} "
          f"({record['wall_clock_seconds']:.0f}s)", flush=True)
    return record


# ---------------------------------------------------------------------------
# laminar negative control
# ---------------------------------------------------------------------------

def run_laminar(out_dir: Path = RESULTS) -> dict:
    """Laminar plane-Poiseuille twin: the analytic solution must be recovered."""
    out_dir.mkdir(parents=True, exist_ok=True)
    case = LAMINAR
    h = case.half_height
    ns = 60  # lattice spacing in x/z as well; irrelevant for a 1-D solution
    ny = case.ny
    re_laminar = 10.0
    u_tau = 0.0125
    nu = u_tau * h / re_laminar
    height = float(ny - 2)
    accel = 2.0 * u_tau**2 / height
    # viscous diffusion time of the channel: t_nu = H^2 / (pi^2 nu)
    t_nu = height**2 / (math.pi**2 * nu)
    steps = int(round(8.0 * t_nu))
    cfg = LesChannelConfig(
        nx=ns, ny=ny, nz=ns, steps=steps, warmup=int(0.7 * steps),
        sample=max(5, steps // 100), re_tau=re_laminar, u_tau=u_tau, cs=0.0,
        perturb=0.0, perturb_modes=1, seed=7, threads=16, regularized=0,
    )
    payload = run_native(cfg, out_dir, case.name, overwrite=True)
    y = np.asarray(payload["profiles"]["y"], dtype=float)
    u = np.asarray(payload["profiles"]["mean_ux"], dtype=float)
    exact = an.poiseuille_exact(y, height, accel, nu)
    record = {
        "case": case.name,
        "geometry": {"nx": ns, "ny": ny, "nz": ns, "steps": steps, "nu": nu,
                     "half_height": h, "accel": accel, "u_tau": u_tau,
                     "re_laminar": re_laminar},
        "norm": {
            "l2_rel_error": an.relative_l2(u, exact),
            "max_rel_error": an.relative_max(u, exact),
            "u_max_sim": float(u.max()),
            "u_max_exact": float(exact.max()),
        },
        "profiles": {"y": y.tolist(), "u_mean": u.tolist(), "u_exact": exact.tolist()},
        "provenance": payload.get("provenance", {}),
    }
    (out_dir / f"{case.name}.json").write_text(json.dumps(record, indent=1))
    print(f"[{case.name}] l2={record['norm']['l2_rel_error']:.5f} "
          f"max={record['norm']['max_rel_error']:.5f} ({steps} steps)", flush=True)
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    matrix = case_matrix()
    if args.list:
        for c in matrix.values():
            print(f"{c.name}: cells={c.cells:,} steps={c.steps:,} "
                  f"({c.warmup:,} warmup) dy+={c.dy_plus}")
        print(f"LAM_poiseuille: ny={LAMINAR.ny}")
        return 0
    if args.case in (None, "all"):
        names = list(matrix)
    else:
        names = [args.case]
    for name in names:
        if name == "LAM_poiseuille":
            run_laminar()
            continue
        case = matrix[name]
        if args.threads:
            case = Case(case.name, case.re_tau, case.dy_plus,
                        warmup_turns=case.warmup_turns,
                        stat_turns=case.stat_turns, cs=case.cs,
                        lx_over_h=case.lx_over_h, lz_over_h=case.lz_over_h,
                        dx_over_dy=case.dx_over_dy, u_tau=case.u_tau,
                        allow_high_mach=case.allow_high_mach,
                        threads=args.threads, note=case.note).geometry()
        run_case(case, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())