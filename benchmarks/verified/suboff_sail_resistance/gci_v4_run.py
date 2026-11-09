#!/usr/bin/env python3
"""gci_v4 single-tier runner — SUBOFF **bare hull** (AFF-1) drag at the DTRC anchor Re.

Purpose
-------
One tier of the resolution-refinement ladder used by ``gci_v4.py`` (research-plan
item 2 of topic ``lbm-suboff-tau``: "resolution convergence and GCI").  Everything
except the lattice resolution ``L`` (cells per hull length) is frozen so that the
three tiers differ in the mesh only.

Frozen protocol (see ``gci_v4_prereg.md``; do not tune per tier)
---------------------------------------------------------------
* geometry    : DARPA SUBOFF parametric bare hull (``hull_type="bare_hull"``),
                L_max = 4.356 m, R_max = 0.254 m (L/D = 8.57)
* Re matching : Re = V*L_max/nu with the DTRC anchor's own assumptions
                V = 5.92 kn = 3.045509 m/s, nu = 1.139e-6 m^2/s
                => Re = 1.16474e7 (identical for all tiers, matched to the
                anchor by construction; verified independently from the
                reported lattice pair ``(u_lb, nu_lb)``)
* domain      : padding (1,4,1,1,1,1) — same formula as the frozen ladder of
                this case directory (blockage scales with L, so it does not
                confound the resolution axis)
* solver      : D3Q19 + MRT (no sub-grid model — the closure is held fixed at
                "none" so that the refinement axis is purely the mesh)
* steps       : 150*L (same flow-through budget as the frozen ladder)
* forces      : every 10 steps; friction estimator ``standard`` (the calibrated
                first-order one-sided instrument frozen by erratum E2);
                p0 = near_wall, pressure extrap = none
* reference   : bare-hull wetted area from the analytic surface-of-revolution
                quadrature ``suboff_analytic_wetted_area(L_lb)`` = 0.31576 L^2
                (matches the experimental WSA 5.998 m^2 to 0.08%), i.e. the
                same caliber the DTRC Ct uses.  NOT the pi*D*L cylinder proxy
                used by the Re=1000 ladder (that proxy is 15.9% larger).

Output: ``artifacts/gci_v4_L{L}.json`` (per-tier raw record; the ladder driver
``gci_v4.py`` is the only writer of the judged artifact ``gci_v4.json``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
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

# --------------------------------------------------------------------------- #
# Frozen anchor block (DTRC / Liu & Huang 1998, Table 14, report p.23)
# --------------------------------------------------------------------------- #
ANCHOR = {
    "case": "DARPA SUBOFF AFF-1 bare hull, tow-tank resistance",
    "source": (
        "H.-L. Liu and T. T. Huang, Summary of DARPA Suboff Experimental Program Data, "
        "CRDKNSWC/HD-1298-11, ADA359226, Table 14, report p.23, DOI 10.21236/ADA359226"
    ),
    # Verbatim row transcribed into the repository (frozen reference table):
    "verbatim_row": "| AFF-1 裸艇 | 5.92 / 10.00 / 11.84 / 13.92 / 16.00 / 17.99 | "
                    "87.40 / 242.2 / 332.9 / 451.5 / 576.9 / 697.0 |",
    "verbatim_provenance": (
        "docs/suboff_experimental_resistance_validation_2026-08-01.md section "
        "'原始实验数据' (repo transcription of Table 14, p.23) and the frozen "
        "dimensional registry src/tensorlbm/suboff_reference_data.py::"
        "SUBOFF_TOW_TANK_RESISTANCE_TABLE14 (bare_hull, config 1)"
    ),
    "V_knots": 5.92,
    "RT_N": 87.40,
    "rho": 1000.0,
    "nu": 1.139e-6,
    "L_max_m": SUBOFF_LENGTH_M,
    "R_max_m": SUBOFF_RADIUS_M,
    "S_wetted_m2": 5.998,  # Table 1 WSA of the DTRC model (refs_lock S1_Table1_verbatim)
    "kn_to_ms": 0.514444,
}


def anchor_numbers() -> dict:
    V = ANCHOR["V_knots"] * ANCHOR["kn_to_ms"]
    S = ANCHOR["S_wetted_m2"]
    CT = ANCHOR["RT_N"] / (0.5 * ANCHOR["rho"] * V**2 * S)
    Re = V * ANCHOR["L_max_m"] / ANCHOR["nu"]
    return {
        "V_ms": V,
        "S_m2": S,
        "CT": CT,
        "Re": Re,
        "formula": "CT = RT / (0.5*rho*V^2*S); Re = V*L_max/nu",
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, required=True, help="cells per hull length L")
    ap.add_argument("--steps", type=int, default=None, help="default 150*L (frozen protocol)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave default)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compile-mode", default="eager", help="eager | default | reduce-overhead")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    L = int(args.resolution)
    steps = int(args.steps) if args.steps else 150 * L
    a = anchor_numbers()

    out_path = Path(args.out) if args.out else _REPO_ROOT / "artifacts" / f"gci_v4_L{L}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ.get("GCI_V4_SCRATCH", f"/tmp/gci_v4_L{L}"))
    scratch.mkdir(parents=True, exist_ok=True)

    config = GeneralSimConfig(
        name=f"gci_v4_suboff_bare_anchorRe_L{L}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SUBOFF,
            suboff_length=SUBOFF_LENGTH_M,
            suboff_radius=SUBOFF_RADIUS_M,
        ),
        physics=PhysicsConfig(
            density=ANCHOR["rho"],
            viscosity=ANCHOR["nu"],
            inlet_velocity=a["V_ms"],
            reference_length=SUBOFF_LENGTH_M,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=CollisionModel.MRT,
            resolution=L,
            domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
            max_steps=steps,
            warmup_steps=None,
            snapshot_interval=10_000_000,
            force_sample_interval=10,
            device=args.device,
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
            directory=str(scratch),
            formats=[],
            save_macroscopic=False,
            save_forces=True,
        ),
    )

    t_setup = time.time()
    engine = SuboffAppendageEngine(config, hull_type="bare_hull")
    setup = engine.setup()
    setup_s = time.time() - t_setup

    u_lb = float(engine.uc.u_lb)
    nu_lb = float(engine.uc.nu_lb)
    tau = float(engine.uc.tau)
    L_lb = float(L)
    R_lb = SUBOFF_RADIUS_M / (SUBOFF_LENGTH_M / L)

    # reference areas in lattice units
    s_analytic = float(suboff_analytic_wetted_area(L_lb))  # bare hull, surface of revolution
    s_cyl_proxy = math.pi * (2.0 * R_lb) * L_lb  # legacy pi*D*L proxy (disclosed, not judged)
    dpS_front = 0.5 * u_lb**2 * math.pi * R_lb**2  # engine force normalization
    rescale = dpS_front / (0.5 * u_lb**2 * s_analytic)

    re_lb = u_lb * L_lb / nu_lb  # independent Re check from the reported lattice pair

    print(
        json.dumps(
            {
                "L": L,
                "steps": steps,
                "domain": setup["domain_lu"],
                "cells": setup["total_cells"],
                "tau": tau,
                "nu_lb": nu_lb,
                "u_lb": u_lb,
                "Re_config": setup["Re"],
                "Re_from_lattice": re_lb,
                "S_analytic_lb2": s_analytic,
                "S_cyl_proxy_lb2": s_cyl_proxy,
                "obstacle_cells": setup.get("obstacle_cells"),
                "near_wall_cells": setup.get("near_wall_cells"),
                "setup_s": round(setup_s, 2),
            },
            indent=1,
        ),
        flush=True,
    )

    run_info, elapsed = run_engine_routed(engine, args.compile_mode)
    ms_per_step = 1000.0 * elapsed / max(run_info["steps"], 1)
    print(
        f"run finished: {run_info['status']} steps={run_info['steps']} "
        f"diverged={run_info.get('diverged')} "
        f"{elapsed:.1f}s ({ms_per_step:.1f} ms/step) compile={run_info.get('compile_mode_effective')}",
        flush=True,
    )

    log = engine.forces_log
    series = [
        {
            "step": int(e["step"]),
            "ct_p": float(e["cd_pressure"]) * rescale,
            "ct_f": float(e["cd_friction"]) * rescale,
            "ct": float(e["cd_total"]) * rescale,
        }
        for e in log
    ]
    n = len(series)
    if n == 0:
        raise SystemExit("no force samples collected")

    def _win(frac_end: float, width: int) -> list[float]:
        k = int(n * frac_end)
        seg = series[max(0, k - width) : k]
        return [s["ct"] for s in seg]

    w_main = _win(1.0, min(1000, n))  # last 1000 samples = last 10000 steps
    mean_ct = sum(w_main) / len(w_main)
    mean_ct_p = sum(s["ct_p"] for s in series[-len(w_main) :]) / len(w_main)
    mean_ct_f = sum(s["ct_f"] for s in series[-len(w_main) :]) / len(w_main)
    std_ct = (sum((x - mean_ct) ** 2 for x in w_main) / max(len(w_main) - 1, 1)) ** 0.5

    # window drift: |mean(last 25%) - mean(previous 25%)| / |mean(previous 25%)|
    q3 = [s["ct"] for s in series[-max(2, n // 4) :]]
    q2 = (
        [s["ct"] for s in series[-2 * max(2, n // 4) : -max(2, n // 4)]]
        if n >= 3 * max(2, n // 4)
        else []
    )
    drift = None
    if q2:
        m3 = sum(q3) / len(q3)
        m2 = sum(q2) / len(q2)
        drift = abs(m3 - m2) / abs(m2) * 100.0 if m2 else None

    err_pct = (mean_ct - a["CT"]) / a["CT"] * 100.0

    record = {
        "tier": L,
        "artifact_kind": "gci_v4_tier_raw",
        "protocol_reference": "gci_v4_prereg.md (frozen before any tier was run)",
        "hull_type": "bare_hull",
        "anchor": {**ANCHOR, **a},
        "setup": {
            "domain_lu": setup["domain_lu"],
            "total_cells": setup["total_cells"],
            "obstacle_cells": setup.get("obstacle_cells"),
            "near_wall_cells": setup.get("near_wall_cells"),
            "domain_padding": [1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
            "lattice": "D3Q19",
            "collision": "mrt",
            "subgrid_model": "none",
            "friction_formula": "standard",
            "p0_method": "near_wall",
            "pressure_extrap": "none",
            "mass_correction_interval": 200,
            "force_sample_interval": 10,
            "steps": run_info["steps"],
            "steps_requested": steps,
            "u_lb": u_lb,
            "nu_lb": nu_lb,
            "tau": tau,
            "Re_config": setup["Re"],
            "Re_from_lattice_pair": re_lb,
            "Re_anchor": a["Re"],
            "Re_rel_dev_pct": (re_lb - a["Re"]) / a["Re"] * 100.0,
            "L_cells": L_lb,
            "R_lb": R_lb,
            "L_over_D": SUBOFF_LENGTH_M / (2.0 * SUBOFF_RADIUS_M),
            "S_analytic_lb2": s_analytic,
            "S_cyl_proxy_lb2": s_cyl_proxy,
            "S_analytic_over_proxy": s_analytic / s_cyl_proxy,
            "cells_per_diameter": 2.0 * R_lb,
        },
        "result": {
            "ct": mean_ct,
            "ct_pressure": mean_ct_p,
            "ct_friction": mean_ct_f,
            "ct_std_window": std_ct,
            "ct_err_vs_anchor_pct": err_pct,
            "window_samples": len(w_main),
            "window_drift_pct_q3_vs_q2": drift,
            "last_quarter_mean_ct": sum(q3) / len(q3),
            "prev_quarter_mean_ct": (sum(q2) / len(q2)) if q2 else None,
        },
        "run": {
            "device": args.device,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "elapsed_s": elapsed,
            "ms_per_step": ms_per_step,
            "setup_s": setup_s,
            "compile_mode": args.compile_mode,
            "diverged": run_info.get("diverged"),
            "compile_status_effective": run_info.get("compile_mode_effective"),
            "status": run_info.get("status"),
            "host": platform.node(),
            "python": platform.python_version(),
        },
        "provenance": {
            "script": "benchmarks/verified/suboff_sail_resistance/gci_v4_run.py",
            "script_sha256": _sha256(Path(__file__)),
            "invocation": f"python3 gci_v4_run.py --resolution {L} --steps {steps} "
                          f"--device {args.device} --threads {args.threads}",
            "engine": "src/tensorlbm/general_sim.py::GeneralSimEngine (via "
                      "benchmarks/verified/suboff_sail_resistance/run.py::SuboffAppendageEngine)",
            "raw_series": series,
        },
    }

    out_path.write_text(json.dumps(record, indent=1))
    print(f"wrote {out_path}  ct={mean_ct:.6e}  err={err_pct:+.3f}%  drift={drift}", flush=True)


if __name__ == "__main__":
    main()