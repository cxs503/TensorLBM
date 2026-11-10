"""Single-case runner for the non-Newtonian LBM verification campaign (P304W2-N07).

Each case is launched as its own process so the campaign can run in parallel::

    python -m tensorlbm.non_newtonian.run_case --case pipe     --n 0.5 --ny 65
    python -m tensorlbm.non_newtonian.run_case --case cavity   --n 1.0 --nx 65
    python -m tensorlbm.non_newtonian.run_case --case contraction --n 1.0 --H-out 16

The runner writes a JSON file with the metrics of the case plus an ``.npz``
with the raw profiles/fields, so that the report generator can rebuild every
number from disk (no hand-copied values).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .cavity import CavityConfig, CavityNonNewtonian
from .contraction import Contraction4to1, ContractionConfig
from .pipe import PipeConfig, PipeNonNewtonian


def _count_tau_converged(history: list[float], tol: float) -> int:
    """Number of Picard (rheology) iterations needed to reach ``tol``."""
    for i, v in enumerate(history, start=1):
        if v < tol:
            return i
    return len(history)


def run_pipe(args, out_dir: Path) -> dict:
    cfg = PipeConfig(
        nx=args.nx_pipe, ny=args.ny, n_power=args.n, rheology=args.rheology,
        nu_0=args.nu0, nu_inf=args.nuinf, lambda_cy=args.lam, yasuda_a=args.yasuda,
        u_max_target=0.05, picard_max_iter=args.blocks, picard_tol=args.tau_tol,
        lbm_steps_per_picard=args.per, flow_tol=args.flow_tol,
        max_steps=args.blocks * args.per, tau_max=args.tau_max,
        shear_rate_floor=args.shear_floor,
        u_init_amplitude=args.u_init, u_init_shape=args.u_init_shape,
    )
    t0 = time.time()
    solver = PipeNonNewtonian(cfg)
    info = solver.solve()
    y, u = solver.get_velocity_profile()
    ua = solver.analytical_profile(y)
    l2 = float(np.linalg.norm(u - ua) / np.linalg.norm(ua))
    linf = float(np.max(np.abs(u - ua)) / np.max(np.abs(ua)))
    res = {
        "case": "pipe_channel_power_law",
        "n_power": args.n, "ny": args.ny, "nx": args.nx_pipe,
        "re_generalized": cfg.re, "consistency_K": solver._consistency_K,
        "body_force_G": solver._body_force,
        "converged": info["converged"],
        "picard_iterations": info["picard_iterations"],
        "tau_iterations_to_tol": _count_tau_converged(info["picard_history"], cfg.picard_tol),
        "total_lbm_steps": info["total_lbm_steps"],
        "flow_residual": info["flow_residual"],
        "tau_residual": info["final_residual"],
        "l2_relative_error": l2, "linf_relative_error": linf,
        "u_max_numeric": float(u.max()), "u_max_analytic": float(ua.max()),
        "picard_history": info["picard_history"],
        "flow_history": info["flow_history"],
        "wall_time_s": round(time.time() - t0, 1),
    }
    tag = "pipe_cy" if args.rheology == "carreau_yasuda" else "pipe"
    np.savez(out_dir / f"{tag}_n{args.n}_ny{args.ny}.npz", y=y, u=u, ua=ua,
             tau=solver.tau_field.cpu().numpy(), ux=solver.ux.cpu().numpy(),
             K=solver._consistency_K, G=solver._body_force, n_power=args.n,
             H=solver.H, rheology=args.rheology,
             nu0=args.nu0, nuinf=args.nuinf, lam=args.lam, yasuda=args.yasuda)
    return res


def run_cavity(args, out_dir: Path) -> dict:
    cfg = CavityConfig(
        nx=args.nx_cav, ny=args.nx_cav, re=100.0, u_lid=0.1,
        n_power=args.n, rheology="power_law",
        picard_max_iter=args.blocks, picard_tol=args.tau_tol,
        lbm_steps_per_picard=args.per, flow_tol=args.flow_tol,
        max_steps=args.blocks * args.per, tau_max=args.tau_max,
    )
    t0 = time.time()
    solver = CavityNonNewtonian(cfg)
    info = solver.solve()
    prof = solver.get_centerline_velocities()
    res = {
        "case": "lid_driven_cavity_power_law",
        "n_power": args.n, "nx": args.nx_cav, "re": cfg.re,
        "u_lid": cfg.u_lid, "consistency_K": solver._consistency_K,
        "tau_ref": solver._tau_ref,
        "converged": info["converged"],
        "picard_iterations": info["picard_iterations"],
        "tau_iterations_to_tol": _count_tau_converged(info["picard_history"], cfg.picard_tol),
        "total_lbm_steps": info["total_lbm_steps"],
        "flow_residual": info["flow_residual"],
        "tau_residual": info["final_residual"],
        "wall_time_s": round(time.time() - t0, 1),
    }
    np.savez(out_dir / f"cavity_n{args.n}_N{args.nx_cav}.npz",
             y=prof["y"], u_vertical=prof["u_vertical"],
             x=prof["x"], u_horizontal=prof["u_horizontal"],
             v_horizontal=prof["v_horizontal"],
             tau=solver.tau_field.cpu().numpy(), ux=solver.ux.cpu().numpy(),
             uy=solver.uy.cpu().numpy())
    return res


def run_contraction(args, out_dir: Path) -> dict:
    cfg = ContractionConfig(
        H_out=args.h_out, n_power=args.n, rheology="power_law",
        L_inlet=args.l_inlet, L_outlet=args.l_outlet,
        picard_max_iter=args.blocks, picard_tol=args.tau_tol,
        lbm_steps_per_picard=args.per, flow_tol=args.flow_tol,
        max_steps=args.blocks * args.per, tau_max=args.tau_max,
    )
    t0 = time.time()
    solver = Contraction4to1(cfg)
    info = solver.solve()
    vl = solver.compute_vortex_length()
    x_cl, u_cl = solver.get_centerline_velocity()
    res = {
        "case": "contraction4to1_power_law",
        "n_power": args.n, "H_out": args.h_out, "H_in": solver.H_in,
        "nx": cfg.L_inlet + cfg.L_outlet, "ny": solver.ny,
        "re": cfg.re, "u_inlet": cfg.u_inlet,
        "body_force_G": solver._body_force,
        "converged": info["converged"],
        "picard_iterations": info["picard_iterations"],
        "total_lbm_steps": info["total_lbm_steps"],
        "flow_residual": info["flow_residual"],
        "tau_residual": info["final_residual"],
        "vortex_length_lattice": vl,
        "vortex_length_over_H_out": vl / cfg.H_out,
        "wall_time_s": round(time.time() - t0, 1),
    }
    np.savez(out_dir / f"contraction_n{args.n}_H{args.h_out}.npz",
             x=x_cl, u=u_cl, uy=solver.uy.cpu().numpy(),
             ux=solver.ux.cpu().numpy(),
             wall_mask=solver.wall_mask.cpu().numpy())
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True,
                    choices=["pipe", "cavity", "contraction"])
    ap.add_argument("--n", type=float, default=1.0)
    ap.add_argument("--ny", type=int, default=65)
    ap.add_argument("--nx-pipe", type=int, default=4)
    ap.add_argument("--nx-cav", type=int, default=65)
    ap.add_argument("--h-out", type=int, default=16)
    ap.add_argument("--l-inlet", type=int, default=128)
    ap.add_argument("--l-outlet", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=200)
    ap.add_argument("--per", type=int, default=500)
    ap.add_argument("--flow-tol", type=float, default=1e-6)
    ap.add_argument("--tau-tol", type=float, default=1e-6)
    ap.add_argument("--tau-max", type=float, default=5.0)
    ap.add_argument("--shear-floor", type=float, default=1e-8)
    ap.add_argument("--rheology", default="power_law",
                    choices=["power_law", "carreau_yasuda"])
    ap.add_argument("--nu0", type=float, default=0.1)
    ap.add_argument("--nuinf", type=float, default=0.005)
    ap.add_argument("--lam", type=float, default=1000.0)
    ap.add_argument("--yasuda", type=float, default=2.0)
    ap.add_argument("--u-init", type=float, default=0.0)
    ap.add_argument("--u-init-shape", default="flat")
    ap.add_argument("--out-dir", default="/tmp")
    args = ap.parse_args(argv)

    torch.set_num_threads(1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = {"pipe": run_pipe, "cavity": run_cavity, "contraction": run_contraction}[args.case]
    res = fn(args, out_dir)
    tag = ("cy" if args.rheology == "carreau_yasuda" else "pl")
    out = out_dir / f"result_{args.case}_{tag}_n{args.n}_ny{args.ny}_N{args.nx_cav}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())