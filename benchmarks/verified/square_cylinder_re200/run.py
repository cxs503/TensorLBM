#!/usr/bin/env python3
"""Square-cylinder vortex shedding — anchor CASE-LBM-SQUARE-CYL-NORBERG, Re=200.

Case definition (as specified by the anchor):
    Re = U*D/nu = 200, normal incidence (alpha = 0 deg), cross-stream blockage
    B = D/H = 1/20 (5 %), two metrics: St = f*D/U and mean Cd.

Engineering chain (all library, no hand-written collide/stream/equilibrium):
    tensorlbm.solver.collide_mrt        MRT collision (tau_field -> sponge)
    tensorlbm.solver.stream             pull streaming
    tensorlbm.boundaries.far_field_bc_2d  inlet/lateral Dirichlet + zero-grad outlet
                                          + obstacle half-way bounce-back
    tensorlbm.boundaries.compute_obstacle_forces  Ladd momentum exchange (Cd/Cl)
    tensorlbm.boundaries.make_sponge_strength     downstream absorbing layer
    tensorlbm.d2q9.equilibrium / macroscopic
    tensorlbm.square_cylinder           mask, blockage, St/Cd reduction, scoring

Domain layout: nx = (upstream + wake) * D, ny = blockage * D.  The obstacle
centre sits ``--upstream-D`` from the inlet and a 6 D quadratic sponge absorbs
the wake at the outlet.

Seeding: the inlet column is overwritten for two periods of ``--st-seed`` with
a +/-10 % transverse fluctuation.  Seeding only *triggers* the instability; the
reported St is the measured spectral peak, not the seed frequency (that
distinction is checked in tests/test_square_cylinder.py).

Usage (CPU example, 24 threads):
    PYTHONPATH=src python benchmarks/verified/square_cylinder_re200/run.py \
        --D 24 --steps 72000 --threads 24 \
        --out runs/m16410_re200/D24_re200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from tensorlbm.boundaries import (  # noqa: E402
    compute_obstacle_forces,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_mrt, stream  # noqa: E402
from tensorlbm.square_cylinder import (  # noqa: E402
    assess_case,
    blockage_ratio,
    drag_statistics,
    force_coefficients,
    load_reference,
    rect_mask_2d,
    strouhal_from_series,
)

CASE = "CASE-LBM-SQUARE-CYL-NORBERG"
ANCHOR_PARAMS = {"re": 200.0, "blockage": 0.05, "incidence_deg": 0.0}


def run_case(
    D: int,
    re: float,
    u_in: float,
    steps: int,
    *,
    blockage: float = 20.0,
    len_up_d: float = 8.0,
    len_down_d: float = 18.0,
    sponge_d: float = 6.0,
    sponge_alpha: float = 10.0,
    st_seed: float = 0.16,
    seed_periods: float = 2.0,
    seed_ampl: float = 0.10,
    warmup_frac: float = 0.45,
    device: str = "cpu",
    reference: dict | None = None,
) -> dict:
    """Run one grid and return a fully self-describing result record."""
    dev = torch.device(device)
    nx = int(round((len_up_d + len_down_d) * D))
    ny = int(round(blockage * D))
    nu = u_in * D / re  # lattice viscosity that realises the target Re
    tau = 0.5 + 3.0 * nu

    mask = rect_mask_2d(nx, ny, len_up_d * D, ny / 2.0, D, device=dev)
    achieved_blockage = blockage_ratio(mask)
    sigma = make_sponge_strength(
        ny, nx, int(nx - sponge_d * D), int(sponge_d * D), power=2.0, device=dev
    )
    tau_field = tau * (1.0 + sponge_alpha * sigma)

    rho0 = torch.ones((ny, nx), device=dev)
    f = equilibrium(rho0, torch.full_like(rho0, u_in), torch.zeros_like(rho0))

    def _step(f):
        fx, fy = compute_obstacle_forces(f, mask)  # post-stream, pre bounce-back
        f = far_field_bc_2d(f, u_in, mask)
        return f, fx, fy

    def _step_plain(f):
        return _step(stream(collide_mrt(f, tau, tau_field=tau_field)))

    def _step_seeded(f, uy_col):
        f, fx, fy = _step(stream(collide_mrt(f, tau, tau_field=tau_field)))
        f[:, :, 0] = equilibrium(
            torch.ones((ny, 1), device=dev, dtype=f.dtype),
            torch.full((ny, 1), u_in, device=dev, dtype=f.dtype),
            uy_col.view(ny, 1),
        )[:, :, 0]
        return f, fx, fy

    omega_seed = 2.0 * math.pi * st_seed * u_in / D
    seed_steps = int(round(seed_periods * D / (st_seed * u_in)))
    uy_amp = seed_ampl * u_in

    fx_hist = np.zeros(steps, dtype=np.float64)
    fy_hist = np.zeros(steps, dtype=np.float64)
    t0 = time.time()
    for step in range(1, steps + 1):
        if step <= seed_steps:
            uy = uy_amp * math.sin(omega_seed * step)
            uy_col = torch.full((ny,), uy, device=dev, dtype=f.dtype)
            f, fx, fy = _step_seeded(f, uy_col)
        else:
            f, fx, fy = _step_plain(f)
        fx_hist[step - 1] = float(fx)
        fy_hist[step - 1] = float(fy)
        if step % max(1, steps // 10) == 0:
            print(f"  [D={D} re={re:g}] step {step}/{steps} ({time.time() - t0:.0f}s)", flush=True)
    elapsed = time.time() - t0

    if not bool(torch.isfinite(f).all()):
        raise RuntimeError(f"D={D}: non-finite populations after {steps} steps")

    # ---- wake snapshot (vorticity from the library macroscopic() reduction) --
    rho, ux, uy = macroscopic(f)
    vort = (
        torch.gradient(uy, dim=1)[0] - torch.gradient(ux, dim=0)[0]
    ).detach().cpu().numpy()

    cd, cl = force_coefficients(fx_hist, fy_hist, u_in, D)
    w0 = int(warmup_frac * steps)
    drag = drag_statistics(cd, w0)
    st = strouhal_from_series(cl[w0:], u_in, D)
    cl_amp = float(np.abs(cl[w0:] - cl[w0:].mean()).max())

    got = {"cd": drag["mean"], "st": st["st"]}
    scoring = assess_case(got, reference, re) if reference else []

    return {
        "case": CASE,
        "D": D,
        "re": re,
        "u_in": u_in,
        "nu_lb": nu,
        "tau": round(tau, 6),
        "grid": {"nx": nx, "ny": ny, "cells": nx * ny},
        "geometry": {
            "upstream_D": len_up_d,
            "wake_D": len_down_d,
            "blockage_target": 1.0 / blockage,
            "blockage_measured": achieved_blockage,
            "side_effective_lattice": int(round(D)),
            "sponge_D": sponge_d,
            "sponge_alpha": sponge_alpha,
        },
        "seed": {
            "st_seed": st_seed,
            "ampl_frac": seed_ampl,
            "periods": seed_periods,
            "steps": seed_steps,
        },
        "steps": steps,
        "analyze_from": w0,
        "analysis_periods": round((steps - w0) * st["st"] * u_in / D, 2),
        "cd": {"mean": drag["mean"], "rms": drag["rms"], "min": drag["min"], "max": drag["max"],
               "n_samples": drag["n_samples"]},
        "st": {"value": st["st"], "fft": st["fft"], "crossing": st["crossing"]},
        "cl_amplitude": cl_amp,
        "scoring": scoring,
        "integration": {
            "solver": [
                "tensorlbm.solver.collide_mrt",
                "tensorlbm.solver.stream",
                "tensorlbm.d2q9.equilibrium",
                "tensorlbm.d2q9.macroscopic",
            ],
            "boundaries": [
                "tensorlbm.boundaries.far_field_bc_2d",
                "tensorlbm.boundaries.compute_obstacle_forces",
                "tensorlbm.boundaries.make_sponge_strength",
            ],
            "benchmark_support": "tensorlbm.square_cylinder",
        },
        "finite": True,
        "elapsed_s": round(elapsed, 1),
        "threads": torch.get_num_threads(),
        "_series": {"cd": cd, "cl": cl, "vorticity": vort},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--D", type=int, nargs="+", default=[24])
    ap.add_argument("--re", type=float, default=200.0)
    ap.add_argument("--u-in", type=float, default=0.05)
    ap.add_argument("--steps", type=int, nargs="+", default=None,
                    help="per-D step counts; default 24 shedding periods (at St=0.16)")
    ap.add_argument("--periods", type=float, default=24.0)
    ap.add_argument("--blockage", type=float, default=20.0,
                    help="channel height in D (20 -> B = 1/20 = 5%%)")
    ap.add_argument("--warmup-frac", type=float, default=0.45)
    ap.add_argument("--st-seed", type=float, default=0.16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--reference", default=str(_HERE / "reference.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    reference = load_reference(args.reference) if args.reference else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.steps:
        steps_list = args.steps
    else:
        # one shedding period is D/(St*u_in) steps; keep the run integral
        steps_list = [int(round(args.periods * D / (args.st_seed * args.u_in))) for D in args.D]

    grids: dict[str, dict] = {}
    for D, steps in zip(args.D, steps_list):
        rec = run_case(
            D, args.re, args.u_in, steps,
            blockage=args.blockage,
            st_seed=args.st_seed,
            warmup_frac=args.warmup_frac,
            device=args.device,
            reference=reference,
        )
        series = rec.pop("_series")
        np.savez_compressed(out / f"series_D{D}.npz", **series)
        rec["series_file"] = str(out / f"series_D{D}.npz")
        (out / f"case_D{D}.json").write_text(json.dumps(rec, indent=2))
        print(
            f"[D={D}] Cd={rec['cd']['mean']:.4f} St={rec['st']['value']:.4f} "
            f"(cross {rec['st']['crossing']['st']:.4f}) cl_amp={rec['cl_amplitude']:.3f} "
            f"B={rec['geometry']['blockage_measured']:.4f} t={rec['elapsed_s']:.0f}s",
            flush=True,
        )
        grids[str(D)] = rec

    summary = {
        "case": CASE,
        "anchor_parameters": ANCHOR_PARAMS,
        "re": args.re,
        "blockage": 1.0 / args.blockage,
        "reference_file": str(args.reference),
        "reference_versions": (
            None if reference is None else reference.get("versions")
        ),
        "grids": grids,
        "tag": args.tag,
        "repro_cmd": (
            "PYTHONPATH=src python benchmarks/verified/square_cylinder_re200/run.py "
            + " ".join(
                [
                    f"--D {' '.join(str(d) for d in args.D)}",
                    f"--re {args.re:g}",
                    f"--steps {' '.join(str(s) for s in steps_list)}",
                    f"--blockage {args.blockage:g}",
                    f"--threads {torch.get_num_threads()}",
                    f"--out {out}",
                ]
            )
        ),
    }
    (out / "result.json").write_text(json.dumps(summary, indent=2))
    print("DONE ->", out / "result.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())