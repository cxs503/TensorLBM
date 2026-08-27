#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B20 2D dam break (SCMP — Shan-Chen single-component) vs Martin & Moyce (1952).

Model: SC94 pseudopotential psi(rho) = 1 - exp(-rho), physical coupling
       G_eff = -5.0, tau = 1.0 — the exact configuration verified in
       benchmarks/verified/laplace_droplet (Laplace law 0.8%) and
       droplet_oscillation (+/-2.6%).  The library is called with G = +5.0
       (sign-flip workaround: the library SC neighbour sum gathers psi(x-c)
       backward, so G_lib = +5 realises the standard attractive SC94 model
       with G_eff = -5; see laplace_droplet README/gap file).

Discrete coexistence (measured from the same scheme, laplace_droplet):
  rho_l = 1.957, rho_v = 0.1596  -> density ratio ~12.3
  (continuum Maxwell values differ due to lattice discreteness — init only)

Setup: closed 2D box nx x ny, bounce-back walls on all four sides.
       Water column a x 2a in the bottom-left corner, gas elsewhere.
       Gravity -y as body force F = rho*g on both phases (the natural SC
       form).  For a two-phase fluid this is exactly equivalent to a
       liquid-only dam break with effective gravity
         g_eff = g*(1 - rho_v/rho_l) = 0.9184*g
       (the gas weight cancels in the momentum balance; incompressible
       limit).  Non-dimensional time is reported on BOTH axes:
         T_raw = t*sqrt(g/a)     (literal definition)
         T_eff = t*sqrt(g_eff/a) (physically consistent for this fluid)
       and the Martin & Moyce comparison is quoted on both.

Diagnostics (every --sample-interval steps):
  X_toe  = x_front/a,   x_front = max x with rho > rho_mid in the bottom
           n_toe fluid rows (the M&M "toe" advancing on the floor)
  X_glob = global max x of liquid (any row)
  H      = h_left/(2a), h_left  = max y with rho > rho_mid at x in [1,4]
  max_u, mass drift (stability / conservation guards)

Reference (Martin & Moyce 1952, Phil. Trans. R. Soc. A 244, 312):
  T = 1 -> X ~ 1.5 ;  T = 2 -> X ~ 2.7

Pass criteria (repo standard, real runs only, no extrapolation):
  1. fine grid (a=80): |X_sim - X_ref|/X_ref <= 3% at T=1 and T=2
  2. two-grid convergence: a=40 vs a=80 within 3% at the checkpoints

Usage: python run.py --a 80 --g 2e-4 --steps 10000 --device cuda:2 [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks

import numpy as np
import torch
from compile_route import add_compile_mode_arg, compile_mode_from_args, route_step  # noqa: E402

torch.set_num_threads(32)

from tensorlbm.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import collide_sc_single_component, psi_exp  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

CS2 = 1.0 / 3.0
G_LIB = 5.0  # library argument (sign-flipped convention, see laplace_droplet)
G_EFF = -5.0  # physical standard-convention SC94 coupling
TAU = 1.0
RHO_L, RHO_V = 1.957, 0.1596  # discrete coexistence (measured)
RHO_MID = 0.5 * (RHO_L + RHO_V)
W_INT = 3.0  # interface width (cells) for the tanh initial condition

# Martin & Moyce (1952) reference checkpoints
MM = {1.0: 1.5, 2.0: 2.7}


def init_rho(nx: int, ny: int, a: float, device: torch.device) -> torch.Tensor:
    """tanh water column: liquid rho_l inside x<a, y<2a; gas rho_v outside."""
    ys = torch.arange(ny, dtype=torch.float32, device=device)
    xs = torch.arange(nx, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    inside = (xx < a) & (yy < 2.0 * a)
    dist_in = torch.minimum(a - xx, 2.0 * a - yy)
    dist_out = torch.minimum(xx - a, yy - 2.0 * a)
    dist = torch.where(inside, dist_in, dist_out)
    rho = RHO_V + 0.5 * (RHO_L - RHO_V) * (1.0 + torch.tanh(dist / W_INT))
    return rho.clamp(min=1e-3)


def wall_mask(nx: int, ny: int, device: torch.device) -> torch.Tensor:
    """Closed-box wall ring (bounce-back on all four sides)."""
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    return mask


def measure(f: torch.Tensor, a: float, n_toe: int) -> dict:
    """Front/height/mass/velocity diagnostics (rho-mid threshold)."""
    rho = f.sum(dim=0)
    liq = rho > RHO_MID
    # M&M toe: rightmost liquid cell in the bottom n_toe fluid rows
    toe = liq[1 : n_toe + 1, 1:-1]
    cols = toe.any(dim=0).nonzero(as_tuple=True)[0]
    x_toe = float(cols.max().item()) if cols.numel() > 0 else 0.0
    # global front: rightmost liquid cell anywhere (fluid region)
    cols_g = liq[1:-1, 1:-1].any(dim=0).nonzero(as_tuple=True)[0]
    x_glob = float(cols_g.max().item()) if cols_g.numel() > 0 else 0.0
    # residual column height at the left wall (columns x=1..4)
    left = liq[1:-1, 1:5].any(dim=1).nonzero(as_tuple=True)[0]
    h_left = float(left.max().item()) + 1.0 if left.numel() > 0 else 0.0
    _, ux, uy = macroscopic(f)
    umag = torch.sqrt(ux * ux + uy * uy)
    return {
        "x_toe": x_toe,
        "x_glob": x_glob,
        "h_left": h_left,
        "max_u": float(umag.max().item()),
        "rho_min": float(rho.min().item()),
        "rho_max": float(rho.max().item()),
        "mass": float(rho.sum().item()),
    }


def interp_x(recs: list[dict], Tq: float) -> float:
    """Linear interpolation of X(T) at Tq from the sampled history (real data)."""
    Ts = np.array([r["T_eff"] for r in recs])
    Xs = np.array([r["X_toe"] for r in recs])
    if Tq <= Ts[0]:
        return float(Xs[0])
    if Tq >= Ts[-1]:
        return float(Xs[-1])
    i = int(np.searchsorted(Ts, Tq))
    t0, t1, x0, x1 = Ts[i - 1], Ts[i], Xs[i - 1], Xs[i]
    return float(x0 + (x1 - x0) * (Tq - t0) / (t1 - t0))


def run_case(
    a: float,
    g: float,
    max_steps: int,
    sample_interval: int,
    device: torch.device,
    compile_mode: str | None,
    out: Path,
) -> tuple[dict, list[dict]]:
    t0 = time.perf_counter()
    nx = int(round(6.4 * a))
    ny = int(round(3.2 * a))
    n_toe = max(2, ny // 64)

    rho0 = init_rho(nx, ny, a, device)
    mass0 = float(rho0.sum().item())
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    wall = wall_mask(nx, ny, device)

    g_eff = g * (1.0 - RHO_V / RHO_L)
    sqrt_ga = math.sqrt(g / a)
    sqrt_geff_a = math.sqrt(g_eff / a)

    def _step(f):
        f = collide_sc_single_component(f, G=G_LIB, tau=TAU, psi_fn=psi_exp, gy=-g, solid_mask=wall)
        f = stream(f)
        f = bounce_back_cells(f, wall)
        return f

    step_fn = route_step(_step, compile_mode, name=f"dam_break_sc[a={a:.0f}]")

    hist: list[dict] = []
    for step in range(1, max_steps + 1):
        f = step_fn(f)
        if step % sample_interval == 0:
            m = measure(f, a, n_toe)
            if (
                m["rho_min"] < 0.0
                or not math.isfinite(m["rho_min"])
                or not math.isfinite(m["rho_max"])
            ):
                raise RuntimeError(f"a={a:.0f}: NaN/negative rho at step {step}")
            rec = {
                "step": step,
                "T_raw": float(step) * sqrt_ga,
                "T_eff": float(step) * sqrt_geff_a,
                "X_toe": m["x_toe"] / a,
                "X_glob": m["x_glob"] / a,
                "H": m["h_left"] / (2.0 * a),
                "max_u": m["max_u"],
                "rho_min": m["rho_min"],
                "rho_max": m["rho_max"],
                "mass_drift": abs(m["mass"] - mass0) / mass0,
            }
            hist.append(rec)
            print(
                f"  a={a:6.0f} step={step:6d} T_eff={rec['T_eff']:6.3f} "
                f"X_toe={rec['X_toe']:6.3f} X_glob={rec['X_glob']:6.3f} "
                f"H={rec['H']:5.3f} max_u={rec['max_u']:7.4f} md={rec['mass_drift']:.1e}",
                flush=True,
            )

    # checkpoint errors on the T_eff axis (primary) and T_raw axis (secondary)
    ck = {}
    for Tq, Xref in MM.items():
        x_eff = interp_x(hist, Tq)
        x_raw = interp_x([dict(r, T_eff=r["T_raw"]) for r in hist], Tq)  # same curve, raw axis
        ck[f"T{Tq:g}"] = {
            "X_ref": Xref,
            "X_sim_T_eff": round(x_eff, 6),
            "X_sim_T_raw": round(x_raw, 6),
            "err_pct_T_eff": round(100.0 * (x_eff - Xref) / Xref, 3),
            "err_pct_T_raw": round(100.0 * (x_raw - Xref) / Xref, 3),
        }

    final = {
        "a": a,
        "nx": nx,
        "ny": ny,
        "g": g,
        "g_eff": g_eff,
        "max_steps": max_steps,
        "sample_interval": sample_interval,
        "T_max_eff": float(hist[-1]["T_eff"]),
        "T_max_raw": float(hist[-1]["T_raw"]),
        "max_u_max": float(max(r["max_u"] for r in hist)),
        "mass_drift_max": float(max(r["mass_drift"] for r in hist)),
        "checkpoints": ck,
        "dt_s": time.perf_counter() - t0,
    }
    return final, hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, default=80.0)
    ap.add_argument("--g", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--sample-interval", type=int, default=100)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out", default="")
    add_compile_mode_arg(ap)
    args = ap.parse_args()
    compile_mode = compile_mode_from_args(args)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"WARN: {args.device} unavailable, falling back to cpu")
        device = torch.device("cpu")

    out = Path(args.out) if args.out else Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)

    final, hist = run_case(
        args.a, args.g, args.steps, args.sample_interval, device, compile_mode, out
    )
    print(
        f"\n  RESULT a={final['a']:.0f}: "
        + "  ".join(
            f"T={Tq:g}: X={final['checkpoints'][f'T{Tq:g}']['X_sim_T_eff']:.3f} "
            f"(err_eff={final['checkpoints'][f'T{Tq:g}']['err_pct_T_eff']:+.2f}%, "
            f"err_raw={final['checkpoints'][f'T{Tq:g}']['err_pct_T_raw']:+.2f}%)"
            for Tq in MM
        )
    )

    case_id = f"a{int(args.a)}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"hist_{case_id}.csv").write_text(
        "step,T_raw,T_eff,X_toe,X_glob,H,max_u,rho_min,rho_max,mass_drift\n"
        + "\n".join(
            f"{r['step']},{r['T_raw']:.8f},{r['T_eff']:.8f},{r['X_toe']:.8f},"
            f"{r['X_glob']:.8f},{r['H']:.8f},{r['max_u']:.8e},{r['rho_min']:.8e},"
            f"{r['rho_max']:.8e},{r['mass_drift']:.8e}"
            for r in hist
        )
        + "\n",
        encoding="utf-8",
    )
    (out / f"case_{case_id}.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved -> {out / f'case_{case_id}.json'}  ({final['dt_s']:.0f}s)")


if __name__ == "__main__":
    main()
