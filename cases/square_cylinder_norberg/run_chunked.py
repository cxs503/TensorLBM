#!/usr/bin/env python3
"""CASE-LBM-SQUARE-CYL-NORBERG — resumable (chunked) driver.

Why this exists
---------------
The full case needs ~2-4 x 10^4 lattice steps (shedding period in lattice units
is ``T = D/(St*u_in)``, i.e. thousands of steps), which does not fit in one
short-lived shell call.  ``run_chunked.py`` therefore runs the *same* library
chain as ``benchmarks/verified/square_cylinder/run.py`` for a wall-clock budget
(~100 s per invocation), checkpoints ``f`` + the force history to ``.npz``, and
resumes on the next invocation.  Nothing about the physics is changed: same
``collide_mrt`` / ``stream`` / ``far_field_bc_2d`` / ``make_sponge_strength`` /
``compute_obstacle_forces`` calls, same inlet seeding, same post-processing
(``run.analyze_forces``).

Case under study (anchor CASE-LBM-SQUARE-CYL-NORBERG):
    square cylinder, free stream, alpha = 0 deg, Re = U*D/nu = 200,
    cross-stream height = 20 D  ->  blockage ratio 1/20 (5%),
    reference: Norberg (1993) J. Wind Eng. Ind. Aerodyn. 49:187-196 (see
    ``refs/norberg1993.json`` for provenance and tolerances).

Usage (one invocation = one chunk):
    python3 cases/square_cylinder_norberg/run_chunked.py \
        --D 10 --steps 36000 --domain-x 20 --domain-y 20 --budget 95 \
        --state runs/m16321/sqcyl_norberg/D10/state.npz

Repro from scratch:
    rm -rf runs/m16321/sqcyl_norberg/D10
    for i in $(seq 1 40); do python3 cases/square_cylinder_norberg/run_chunked.py \
        --D 10 --steps 36000 --domain-x 20 --domain-y 20 --budget 95 \
        --state runs/m16321/sqcyl_norberg/D10/state.npz || break; done
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

# <repo>/cases/square_cylinder_norberg/ -> <repo>
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))  # compile_route
sys.path.insert(0, str(REPO / "benchmarks" / "verified" / "square_cylinder"))  # run.py
sys.path.insert(0, str(REPO / "src"))

from run import analyze_forces, square_mask  # noqa: E402

from tensorlbm.boundaries import (  # noqa: E402
    compute_obstacle_forces,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.solver import collide_mrt, stream  # noqa: E402


def load_refs(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--D", type=int, default=10, help="cylinder side in lattice cells")
    ap.add_argument("--re", type=float, default=200.0)
    ap.add_argument("--u-in", type=float, default=0.05)
    ap.add_argument("--steps", type=int, required=True, help="total lattice steps")
    ap.add_argument("--domain-x", type=float, default=20.0, help="streamwise length in D")
    ap.add_argument("--domain-y", type=float, default=20.0, help="height in D (=1/blockage)")
    ap.add_argument("--x-center", type=float, default=6.0, help="square centre, x/D")
    ap.add_argument("--sponge-d", type=float, default=6.0, help="downstream sponge width in D")
    ap.add_argument("--sponge-alpha", type=float, default=10.0)
    ap.add_argument("--st-seed", type=float, default=0.13, help="inlet seeding Strouhal")
    ap.add_argument("--seed-ampl", type=float, default=0.10)
    ap.add_argument("--seed-periods", type=float, default=2.0)
    ap.add_argument("--warmup-frac", type=float, default=0.5)
    ap.add_argument("--budget", type=float, default=95.0, help="wall-clock seconds this call")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--state", required=True, help="checkpoint .npz path")
    ap.add_argument(
        "--refs",
        default=str(REPO / "cases" / "square_cylinder_norberg" / "refs" / "norberg1993.json"),
    )
    ap.add_argument("--sample-every", type=int, default=1, help="force sampling interval")
    ap.add_argument(
        "--drop-history",
        action="store_true",
        help="discard force history loaded from the checkpoint (re-sample from now on)",
    )
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cpu")
    refs = load_refs(Path(args.refs))

    D = args.D
    nx, ny = int(args.domain_x * D), int(args.domain_y * D)
    nu = args.u_in * D / args.re
    tau = 0.5 + 3.0 * nu
    state_path = Path(args.state)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    mask = square_mask(nx, ny, args.x_center * D, ny / 2.0, D, device)
    sigma = make_sponge_strength(
        ny, nx, int(nx - args.sponge_d * D), int(args.sponge_d * D), power=2.0, device=device
    )
    tau_field = tau * (1.0 + args.sponge_alpha * sigma)

    omega_seed = 2.0 * math.pi * args.st_seed * args.u_in / D
    seed_steps = int(round(args.seed_periods / (args.st_seed * args.u_in / D)))
    uy_amp = args.seed_ampl * args.u_in

    # ---- resume or cold start ----
    if state_path.exists():
        st = np.load(state_path, allow_pickle=False)
        f = torch.from_numpy(st["f"]).to(device)
        step = int(st["step"])
        fx_hist = [] if args.drop_history else list(st["fx"])
        fy_hist = [] if args.drop_history else list(st["fy"])
    else:
        rho0 = torch.ones((ny, nx), device=device)
        f = equilibrium(rho0, torch.full_like(rho0, args.u_in), torch.zeros_like(rho0))
        step, fx_hist, fy_hist = 0, [], []

    def _step_plain(f):
        # order matters: momentum-exchange force must be sampled on the
        # post-stream / pre-bounce-back populations (same as run.py::
        # _forces_and_farfield).  Sampling the pre-collision state instead
        # flips the sign of Cd (observed 2026-11-12: Cd = -1.615 at D=10).
        f = stream(collide_mrt(f, tau, tau_field=tau_field))
        fx, fy = compute_obstacle_forces(f, mask)
        return far_field_bc_2d(f, args.u_in, mask), fx, fy

    def _step_seeded(f, uy_col):
        f, fx, fy = _step_plain(f)
        f[:, :, 0] = equilibrium(
            torch.ones((ny, 1), device=device),
            torch.full((ny, 1), args.u_in, device=device),
            uy_col.view(ny, 1),
        )[:, :, 0]
        return f, fx, fy

    t0 = time.time()
    n_chunk = 0
    while step < args.steps:
        step += 1
        if step <= seed_steps:
            uy_col = torch.full(
                (ny,), uy_amp * math.sin(omega_seed * step), device=device
            )
            f, fx, fy = _step_seeded(f, uy_col)
        else:
            f, fx, fy = _step_plain(f)
        if step % args.sample_every == 0:
            fx_hist.append(float(fx))
            fy_hist.append(float(fy))
        n_chunk += 1
        if time.time() - t0 > args.budget:
            break
    wall = time.time() - t0
    if not bool(torch.isfinite(f).all().item()):
        print(json.dumps({"status": "blowup", "step": step}), flush=True)
        return 2

    # ---- checkpoint ----
    tmp = state_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        f=f.detach().cpu().numpy().astype(np.float32),
        step=np.int64(step),
        fx=np.asarray(fx_hist, dtype=np.float64),
        fy=np.asarray(fy_hist, dtype=np.float64),
    )
    tmp.replace(state_path)

    info = {
        "step": step,
        "steps_target": args.steps,
        "chunk_steps": n_chunk,
        "ms_per_step": round(1000.0 * wall / max(n_chunk, 1), 2),
        "done": step >= args.steps,
        "seed_steps": seed_steps,
    }

    if step >= args.steps:
        fx_np, fy_np = np.asarray(fx_hist), np.asarray(fy_hist)
        ana = analyze_forces(fx_np, fy_np, D=D, u_in=args.u_in, warmup_frac=args.warmup_frac)
        cd_ref = float(refs["metrics"]["cd"]["value"])
        st_ref = float(refs["metrics"]["st"]["value"])
        tol_cd = float(refs["metrics"]["cd"]["tolerance_pct"])
        tol_st = float(refs["metrics"]["st"]["tolerance_pct"])
        err_cd = (ana["cd_mean"] - cd_ref) / cd_ref * 100.0
        err_st = (ana["st"] - st_ref) / st_ref * 100.0
        result = {
            "case": "CASE-LBM-SQUARE-CYL-NORBERG",
            "D": D,
            "nx": nx,
            "ny": ny,
            "re": args.re,
            "u_in": args.u_in,
            "nu_lb": nu,
            "tau": round(tau, 6),
            "blockage": round(1.0 / args.domain_y, 5),
            "domain_x_d": args.domain_x,
            "domain_y_d": args.domain_y,
            "x_center_d": args.x_center,
            "sponge": {"width_d": args.sponge_d, "alpha": args.sponge_alpha, "power": 2.0},
            "seed": {"st_seed": args.st_seed, "ampl_frac": args.seed_ampl, "steps": seed_steps},
            "steps": args.steps,
            "sample_every": args.sample_every,
            "warmup_frac": args.warmup_frac,
            "analyze_from": ana["analyze_from"],
            "samples": int(fx_np.size),
            "cd": round(ana["cd_mean"], 4),
            "cd_ref": cd_ref,
            "cd_err_pct": round(err_cd, 2),
            "cd_tol_pct": tol_cd,
            "cd_pass": bool(abs(err_cd) <= tol_cd),
            "st": round(ana["st"], 4) if math.isfinite(ana["st"]) else None,
            "st_crossing": round(ana["st_crossing"], 4)
            if math.isfinite(ana["st_crossing"])
            else None,
            "st_ref": st_ref,
            "st_err_pct": round(err_st, 2) if math.isfinite(ana["st"]) else None,
            "st_tol_pct": tol_st,
            "st_pass": bool(math.isfinite(ana["st"]) and abs(err_st) <= tol_st),
            "cl_amp": round(ana["cl_amp"], 4),
            "ref_provenance": refs.get("provenance", ""),
            "elapsed_s_total": round(step * wall / max(n_chunk, 1), 1),
        }
        out = state_path.parent / f"result_D{D}.json"
        out.write_text(json.dumps(result, indent=2))
        np.savez_compressed(
            state_path.parent / f"series_D{D}.npz",
            cd=ana["cd_series"],
            cl=ana["cl_series"],
            warmup_from=ana["analyze_from"],
        )
        info.update({"status": "finished", "result": str(out), "cd": result["cd"], "st": result["st"]})
        print(json.dumps(info), flush=True)
        return 0

    print(json.dumps(info), flush=True)
    return 1  # more chunks needed


if __name__ == "__main__":
    raise SystemExit(main())