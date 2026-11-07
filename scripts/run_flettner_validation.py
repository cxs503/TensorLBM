"""Flettner rotor-sail validation / application runner (TensorLBM).

Subcommands
-----------
``case``    run one D2Q9 rotating-cylinder channel (laminar or LES) and dump a
            JSON summary (coefficients, convergence series, optional fields).
``voyage``  run the voyage-level energy/CO2 assessment from a coefficient table.

Examples
--------
    python scripts/run_flettner_validation.py case --re 200 --diameter-cells 24 \
        --alpha 2.0 --steps 12000 --warmup 6000 --out artifacts/x.json

The runner is deliberately thin: all physics lives in
:mod:`tensorlbm.rotor_sail`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.rotor_sail.lbm2d import RotorChannelConfig, run_rotor_channel  # noqa: E402


def _profile(kind: str, ny: int, params: dict) -> list[float] | None:
    """Inlet velocity profile u(y)/u_in (bottom row first)."""
    from tensorlbm.rotor_sail.aero import deck_boundary_layer_profile

    if kind == "uniform":
        return None
    if kind == "deckbl":
        h = params["height_over_diameter"] * params["diameter_cells"]
        delta = params["delta_over_height"] * h
        z0 = params.get("z0_over_height", 0.0) * h
        return deck_boundary_layer_profile(ny, delta=delta, z0=z0, exponent=params.get("exponent", 1.0 / 7.0))
    raise ValueError(f"unknown profile kind {kind!r}")


def cmd_case(args: argparse.Namespace) -> int:
    import torch

    torch.set_num_threads(args.threads)
    d = args.diameter_cells
    nx = args.nx or int(round((args.upstream_d + args.downstream_d) * d))
    ny = args.ny or int(round(2 * args.height_d * d) + 1)
    cx = args.upstream_d * d
    cy = 0.5 * (ny - 1)
    profile = _profile(args.profile, ny, json.loads(args.profile_params))
    cfg = RotorChannelConfig(
        nx=nx,
        ny=ny,
        radius=0.5 * d,
        u_in=args.u_in,
        re=args.re,
        spin=tuple(args.alpha),
        cx=cx,
        cy=cy,
        les_cs=args.les_cs,
        inlet_profile=None if profile is None else tuple(profile),
        n_steps=args.steps,
        warmup=args.warmup,
        sample_every=args.sample_every,
        device="cpu",
    )
    res = run_rotor_channel(cfg, keep_fields=args.save_fields)
    out = {
        "task": "rotor_channel_case",
        "config": {
            "nx": nx, "ny": ny, "diameter_cells": d, "re": args.re, "u_in": args.u_in,
            "alpha": list(args.alpha), "les_cs": args.les_cs, "steps": args.steps,
            "warmup": args.warmup, "profile": args.profile,
            "profile_params": json.loads(args.profile_params),
            "upstream_d": args.upstream_d, "downstream_d": args.downstream_d,
            "height_d": args.height_d, "blockage": d / ny,
        },
        "metrics": {k: res[k] for k in (
            "spin", "cl", "cd", "cq", "cl_cv", "cd_cv", "cl_std", "steps", "warmup",
            "samples", "nu", "tau", "re", "u_in", "diameter", "nx", "ny", "les_cs", "mach")},
        "series": res["series"],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    m = out["metrics"]
    for i, a in enumerate(m["spin"]):
        print(
            f"[case {out_path.name}] Re={m['re']:g} D={m['diameter']:.0f} alpha={a:+.2f} "
            f"CL={m['cl'][i]:+.4f} (CV {m['cl_cv'][i]:+.4f}) CD={m['cd'][i]:+.4f} "
            f"CQ={m['cq'][i]:+.4f} tau={m['tau']:.4f}",
            flush=True,
        )
    if args.save_fields:
        fld = res["fields"]
        np.savez_compressed(
            str(out_path.with_suffix(".npz")),
            u=fld["u"], v=fld["v"], rho=fld["rho"], solid=fld["solid"],
            cv_box=np.array(fld["cv_box"]), cx=fld["cx"], cy=fld["cy"], radius=fld["radius"],
            alpha=np.array(m["spin"]),
        )
    return 0


def cmd_voyage(args: argparse.Namespace) -> int:
    from tensorlbm.rotor_sail.voyage import VoyageInputs, run_voyage_assessment

    coeffs = json.loads(Path(args.coeffs).read_text())
    inputs = VoyageInputs(**json.loads(Path(args.inputs).read_text()))
    rep = run_voyage_assessment(coeffs, inputs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["summary"], indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("case")
    c.add_argument("--re", type=float, required=True)
    c.add_argument("--u-in", type=float, default=0.1)
    c.add_argument("--diameter-cells", type=float, required=True)
    c.add_argument("--alpha", type=float, nargs="+", required=True)
    c.add_argument("--nx", type=int, default=0)
    c.add_argument("--ny", type=int, default=0)
    c.add_argument("--upstream-d", type=float, default=6.0)
    c.add_argument("--downstream-d", type=float, default=12.0)
    c.add_argument("--height-d", type=float, default=8.0)
    c.add_argument("--steps", type=int, default=12000)
    c.add_argument("--warmup", type=int, default=6000)
    c.add_argument("--sample-every", type=int, default=20)
    c.add_argument("--les-cs", type=float, default=0.0)
    c.add_argument("--profile", default="uniform")
    c.add_argument("--profile-params", default="{}")
    c.add_argument("--threads", type=int, default=16)
    c.add_argument("--out", required=True)
    c.add_argument("--save-fields", action="store_true")
    c.set_defaults(func=cmd_case)

    v = sub.add_parser("voyage")
    v.add_argument("--coeffs", required=True)
    v.add_argument("--inputs", required=True)
    v.add_argument("--out", required=True)
    v.set_defaults(func=cmd_voyage)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
