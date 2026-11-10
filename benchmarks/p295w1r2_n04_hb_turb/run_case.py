#!/usr/bin/env python3
"""Run one Herschel-Bulkley channel case (laminar calibration or turbulent).

Campaign node P295W1R2-N04.  Writes a JSON with the derived quantities
(``Re_MR``, Darcy ``f``, TKE, wall stress) plus the wall-normal profiles, so
that every number in the report can be regenerated from the artifacts.

Examples
--------
Turbulent, matched Re_MR, n = 0.8, Y = 0.5::

    python benchmarks/p295w1r2_n04_hb_turb/run_case.py --name turb_re180_n0.8_Y0.5 \
        --mode turbulent --re-tau 180 --n 0.8 --y 0.5 --nx 64 --ny 48 --nz 32

Laminar calibration point::

    python benchmarks/p295w1r2_n04_hb_turb/run_case.py --name lam_n0.8_u0.002 \
        --mode laminar --re-tau 50 --n 0.8 --y 0.3 --u-tau 0.002 \
        --ny 46 --nx 8 --nz 8 --steps 40000 --warmup 25000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from tensorlbm.hb_turbulent_channel3d import (  # noqa: E402
    HBTurbulentChannelConfig,
    run_hb_turbulent_channel,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["turbulent", "laminar"], default="turbulent")
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=48)
    ap.add_argument("--nz", type=int, default=32)
    ap.add_argument("--re-tau", type=float, default=180.0)
    ap.add_argument("--u-tau", type=float, default=0.006)
    ap.add_argument("--n", type=float, default=1.0, dest="flow_index")
    ap.add_argument("--y", type=float, default=0.0, help="yield number Y")
    ap.add_argument("--K", type=float, default=None, help="consistency index")
    ap.add_argument(
        "--nu-ref",
        type=float,
        default=0.05,
        help="laminar mode: apparent viscosity at the nominal wall shear rate (sets K)",
    )
    ap.add_argument(
        "--re-mr-target",
        type=float,
        default=None,
        help="if given, K is set to put a power-law fluid at this Re_MR",
    )
    ap.add_argument(
        "--u-bulk-ref",
        type=float,
        default=None,
        help="bulk velocity used for the Re_MR matching (from the Newtonian reference case)",
    )
    ap.add_argument(
        "--m-pap", type=float, default=0.0, help="0 = automatic scaling with nu_ref/tau_y"
    )
    ap.add_argument("--smagorinsky-cs", type=float, default=0.1)
    ap.add_argument(
        "--nu-max",
        type=float,
        default=0.2,
        help="upper clamp of the effective viscosity (must exceed "
        "the regularised plug viscosity tau_y*m/rho for the "
        "plug to be rigid)",
    )
    ap.add_argument("--lattice", choices=["d3q19", "d3q27"], default="d3q19")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=6000)
    ap.add_argument("--sample-interval", type=int, default=10)
    ap.add_argument("--nu-update-interval", type=int, default=4)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- rheology: either explicit K or matched Re_MR --------------------
    from tensorlbm.hb_turbulent_channel3d import (
        consistency_for_target_remr,
        metzner_reed_reynolds,
    )

    half_height = 0.5 * (args.ny - 2)
    K = args.K
    if K is None:
        if args.mode == "turbulent":
            if args.re_mr_target is None or args.u_bulk_ref is None:
                raise SystemExit("turbulent mode needs --K or (--re-mr-target and --u-bulk-ref)")
            K = consistency_for_target_remr(
                args.re_mr_target, args.u_bulk_ref, 4.0 * half_height, args.flow_index
            )
        else:
            # laminar calibration: fix the apparent viscosity at the nominal
            # wall shear rate; a laminar run needs nu_ref >> 1e-3 so that the
            # Poiseuille state is reached in O(3 h^2 / nu) steps.
            nu_ref = args.nu_ref
            gd_ref = 2.0 * args.u_tau / half_height
            K = nu_ref * gd_ref ** (1.0 - args.flow_index)
            args.re_tau = args.u_tau * half_height / nu_ref
    tau_y = args.y * args.u_tau**2

    cfg = HBTurbulentChannelConfig(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        re_tau=args.re_tau,
        u_tau=args.u_tau,
        tau_y=tau_y,
        K=K,
        n=args.flow_index,
        m_pap=args.m_pap,
        smagorinsky_cs=args.smagorinsky_cs,
        nu_max=args.nu_max,
        lattice=args.lattice,
        steps=args.steps,
        warmup_steps=args.warmup,
        sample_interval=args.sample_interval,
        nu_update_interval=args.nu_update_interval,
        dtype=args.dtype,
        seed=args.seed,
        name=args.name,
        output=out_dir / f"{args.name}.json",
    )

    t0 = time.time()
    result = run_hb_turbulent_channel(cfg, log=lambda m: print(m, flush=True))
    result["mode"] = args.mode
    result["elapsed_s"] = time.time() - t0
    result["host"] = os.uname().nodename
    result["torch"] = torch.__version__
    result["threads"] = args.threads
    result["K_input"] = K
    result["re_mr_target"] = args.re_mr_target
    result["u_bulk_ref"] = args.u_bulk_ref
    result["re_mr_from_formula"] = metzner_reed_reynolds(
        result["derived"]["u_bulk"], 4.0 * half_height, K, args.flow_index
    )
    path = out_dir / f"{args.name}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    print(
        "RESULT "
        + json.dumps(
            {
                "name": args.name,
                "u_bulk": result["derived"]["u_bulk"],
                "re_mr": result["derived"]["re_mr"],
                "f_darcy": result["derived"]["f_darcy"],
                "tke": result["derived"]["tke_volume_mean"],
                "elapsed_s": round(result["elapsed_s"], 1),
                "ms_per_step": round(1000.0 * result["elapsed_s"] / args.steps, 2),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
