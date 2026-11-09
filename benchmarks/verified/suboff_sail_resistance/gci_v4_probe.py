#!/usr/bin/env python3
"""gci_v4_probe — stability / executability probe for the anchor-Re ladder.

Why this exists
---------------
``gci_v4_run.py`` executes one tier of the resolution ladder with the lattice
viscosity pinned by the **DTRC anchor Reynolds number**
(Re = V*L_max/nu = 1.16474e7).  At L = 96/112/128 that pins

    nu_lb = u_lb*L/Re  ~  4.1e-7 .. 5.5e-7   =>   tau - 0.5 ~ 1.2e-6 .. 1.6e-6

i.e. the D3Q19 relaxation time sits on the tau -> 1/2 floor.  The first tier
run aborted through the engine's own divergence detector within one step, so
before spending the ladder budget we measure *where* the frontier is: a Re
sweep at fixed resolution with two closures (plain MRT, MRT+Smagorinsky).

Everything here is a diagnostic of the *instrument*, not a drag result; no Ct
from this probe enters any accuracy verdict.

Usage:
  python3 gci_v4_probe.py --resolution 48 --steps 30 \
      --re 1000 10000 100000 1000000 11647400 --closures mrt smagorinsky \
      --out artifacts/gci_v4_probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(_HERE))

import torch  # noqa: E402

from run import SUBOFF_LENGTH_M, SUBOFF_RADIUS_M, SuboffAppendageEngine, run_engine_routed  # noqa: E402
from tensorlbm.general_sim import (  # noqa: E402
    CollisionModel,
    ForceMethod,
    GeneralSimConfig,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    PhysicsConfig,
    SolverConfig,
    WallTreatment,
)
from tensorlbm.suboff_cad import suboff_analytic_wetted_area  # noqa: E402

V_ANCHOR_MS = 5.92 * 0.514444
NU_ANCHOR = 1.139e-6
RE_ANCHOR = V_ANCHOR_MS * SUBOFF_LENGTH_M / NU_ANCHOR


def build_config(L: int, re: float, steps: int, closure: str, device: str, scratch: Path):
    collision = CollisionModel.SMAGORINSKY_MRT if closure == "smagorinsky" else CollisionModel.MRT
    # physical pair expressing Re = V*L/nu with V held at the anchor speed
    nu = V_ANCHOR_MS * SUBOFF_LENGTH_M / re
    return GeneralSimConfig(
        name=f"gci_v4_probe_L{L}_Re{re:.3g}_{closure}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SUBOFF,
            suboff_length=SUBOFF_LENGTH_M,
            suboff_radius=SUBOFF_RADIUS_M,
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=nu,
            inlet_velocity=V_ANCHOR_MS,
            reference_length=SUBOFF_LENGTH_M,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=collision,
            resolution=L,
            domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
            max_steps=steps,
            warmup_steps=None,
            snapshot_interval=10_000_000,
            force_sample_interval=5,
            device=device,
            wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap="none",
            p0_method="near_wall",
            friction_formula="standard",
            mass_correction=True,
            mass_correction_interval=200,
            smagorinsky_cs=0.05,
        ),
        output=OutputConfig(
            directory=str(scratch), formats=[], save_macroscopic=False, save_forces=True
        ),
    )


def probe_one(L: int, re: float, steps: int, closure: str, device: str) -> dict:
    scratch = Path(f"/tmp/gci_v4_probe_L{L}_{closure}_{int(re)}")
    scratch.mkdir(parents=True, exist_ok=True)
    cfg = build_config(L, re, steps, closure, device, scratch)
    t0 = time.time()
    engine = SuboffAppendageEngine(cfg, hull_type="bare_hull")
    setup = engine.setup()
    setup_s = time.time() - t0
    u_lb = float(engine.uc.u_lb)
    nu_lb = float(engine.uc.nu_lb)
    tau = float(engine.uc.tau)
    s_lb = float(suboff_analytic_wetted_area(float(L)))
    dpS_front = 0.5 * u_lb**2 * math.pi * (SUBOFF_RADIUS_M / (SUBOFF_LENGTH_M / L)) ** 2
    rescale = dpS_front / (0.5 * u_lb**2 * s_lb)

    t_run = time.time()
    info, elapsed = run_engine_routed(engine, "eager")
    log = engine.forces_log
    ct_series = [float(e["cd_total"]) * rescale for e in log]
    record = {
        "L": L,
        "Re_target": re,
        "closure": closure,
        "tau": tau,
        "tau_minus_half": tau - 0.5,
        "nu_lb": nu_lb,
        "u_lb": u_lb,
        "domain_lu": setup["domain_lu"],
        "cells": setup["total_cells"],
        "steps_requested": steps,
        "steps_executed": info["steps"],
        "diverged": bool(info.get("diverged")),
        "elapsed_s": elapsed,
        "ms_per_step": 1000.0 * elapsed / max(info["steps"], 1),
        "setup_s": setup_s,
        "ct_samples": len(ct_series),
        "ct_last": ct_series[-1] if ct_series else None,
        "ct_mean": (sum(ct_series) / len(ct_series)) if ct_series else None,
        "ct_series": ct_series,
        "ct_window_last_quarter_mean": (
            sum(ct_series[-max(1, len(ct_series) // 4):]) / max(1, len(ct_series) // 4)
            if ct_series
            else None
        ),
        "ct_window_prev_quarter_mean": (
            sum(ct_series[-max(1, len(ct_series) // 2):-max(1, len(ct_series) // 4)])
            / max(1, len(ct_series) // 4)
            if len(ct_series) >= 4
            else None
        ),
        "device": device,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=48)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--re", type=float, nargs="+", default=[1000.0, 1e4, 1e5, 1e6, RE_ANCHOR])
    ap.add_argument("--closures", nargs="+", default=["mrt", "smagorinsky"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    out = Path(args.out) if args.out else _REPO_ROOT / "artifacts" / "gci_v4_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for closure in args.closures:
        for re in args.re:
            rec = probe_one(args.resolution, float(re), args.steps, closure, args.device)
            rows.append(rec)
            print(json.dumps({k: rec[k] for k in (
                "L", "Re_target", "closure", "tau", "tau_minus_half", "steps_executed",
                "diverged", "ms_per_step", "ct_last")}), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {
        "artifact_kind": "gci_v4_stability_probe",
        "purpose": (
            "locate the tau->1/2 executability frontier of the frozen ladder protocol; "
            "no Ct from this file is used in any accuracy verdict"
        ),
        "protocol_reference": "gci_v4_prereg.md",
        "anchor": {"Re": RE_ANCHOR, "V_ms": V_ANCHOR_MS, "nu": NU_ANCHOR},
        "resolution": args.resolution,
        "steps": args.steps,
        "rows": rows,
        "provenance": {
            "script": "benchmarks/verified/suboff_sail_resistance/gci_v4_probe.py",
            "invocation": (
                f"python3 gci_v4_probe.py --resolution {args.resolution} --steps {args.steps} "
                f"--re {' '.join(str(r) for r in args.re)} --closures {' '.join(args.closures)} "
                f"--device {args.device} --threads {args.threads}"
            ),
        },
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()