#!/usr/bin/env python3
"""Sessile-drop probe for the geometrical-wetting BC (short and long time).

Hypothesis under test
---------------------
Round 1 of ``XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE`` declared the
solver-level sessile-drop comparison an honest-FAIL.  This probe re-examines
that conclusion with a *component-restricted* measurement
(:mod:`wall_drop_measure`): the host solver condenses a small amount of the
wetting phase onto solid walls, so a global contour/circle fit mixes the drop
with that trailing blob (apex -> top wall, Kasa fit -> 180 deg).  With the blob
excluded, the question is whether the drop actually realises the prescribed
contact angle.

Two probe modes
---------------
``direction``  short time: the drop is still the initial cap, so the sign of the
               contact-line velocity is an unambiguous proxy
               (``theta<90`` must spread, ``theta>90`` must contract,
               ``theta=90`` is the neutral fixed point of the BC).
``relax``      long time: footprint ``W(t)``, apex ``H(t)``, cap angle and the
               independent circle-fit angle, plus the wall-blob diagnostic.

Usage
-----
python benchmarks/xval_openlbmpm_wetting/probe_wetting_direction.py \
       --mode direction --thetas 30,60,90,120,150 --steps 600 --out sign_probe.json
python benchmarks/xval_openlbmpm_wetting/probe_wetting_direction.py \
       --mode relax --thetas 60,90,120 --steps 20000 --ny 128 --nx 160 --radius 24 \
       --out relax.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorlbm.cg_wetting_geometric import sessile_drop_initial  # noqa: E402
from tensorlbm.color_gradient2d import color_gradient_two_tau_step  # noqa: E402
from tensorlbm.solver import OPPOSITE_2D, stream  # noqa: E402
from wall_drop_measure import measure_wall_drop  # noqa: E402

OPP = tuple(int(v) for v in OPPOSITE_2D)  # canonical map; see R3 doc


def run_arm(theta: float, *, steps: int, ny: int, nx: int, radius: float, A: float,
            beta: float, tau: float, xi: float, sample_every: int,
            wetting: bool = True, top_theta: float | None = 90.0,
            top: str = "wall") -> dict:
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = True
    if top == "wall":
        solid[ny - 1, :] = True
    solid3 = solid.unsqueeze(0)
    f_r, f_b = sessile_drop_initial(ny, nx, radius, theta, xi=xi, dtype=torch.float64)
    solid_np = solid.numpy()
    t0 = time.time()
    hist = []
    for it in range(1, steps + 1):
        kw = dict(tau_r=tau, tau_b=tau, A=A, beta=beta, solid_mask=solid, visc_mode="mixture")
        if wetting:
            kw["wall_contact_angle"] = float(theta)
            if top_theta is not None and top == "wall":
                kw["wall_contact_angle_by_direction"] = {(0, 1): float(top_theta)}
        f_r, f_b = color_gradient_two_tau_step(f_r, f_b, **kw)
        f_r = stream(f_r)
        f_b = stream(f_b)
        f_r = torch.where(solid3, f_r[list(OPP)], f_r)
        f_b = torch.where(solid3, f_b[list(OPP)], f_b)
        if it % sample_every == 0 or it == 1:
            rr = f_r.sum(dim=0)
            rb = f_b.sum(dim=0)
            rho = rr + rb
            phi = ((rr - rb) / rho.clamp(min=1e-12)).detach().cpu().numpy()
            jx = ((f_r + f_b)[1] - (f_r + f_b)[3]).sum(dim=0).detach().cpu().numpy()
            jy = ((f_r + f_b)[2] - (f_r + f_b)[4]).sum(dim=0).detach().cpu().numpy()
            u = np.sqrt((jx / rho.detach().numpy()) ** 2 + (jy / rho.detach().numpy()) ** 2)
            g = measure_wall_drop(phi, solid_np)
            g.update(
                step=it,
                u_max=float(u[~solid_np].max()),
                volume=float(phi.sum()),
                mass=float(rho.sum()),
                drop_centre_u=float(u[int(round(g["x_centre"])) if math.isfinite(g["x_centre"])
                                       else nx // 2, 1: int(ny / 2)].max()),
            )
            hist.append(g)
    last = hist[-1]
    W0, W1 = hist[0]["W"], last["W"]
    H0, H1 = hist[0]["H"], last["H"]
    tail = [h for h in hist if h["step"] > 0.75 * steps] or [last]
    tail_cap = [h["theta_cap_deg"] for h in tail if math.isfinite(h["theta_cap_deg"])]
    tail_circ = [h["theta_circle_deg"] for h in tail if math.isfinite(h["theta_circle_deg"])]
    out = {
        "theta_cmd": float(theta),
        "wetting": bool(wetting),
        "top": top,
        "params": {"ny": ny, "nx": nx, "radius": radius, "steps": steps, "A": A,
                   "beta": beta, "tau": tau, "xi": xi},
        "W_initial": W0, "W_final": W1, "dW": W1 - W0,
        "dW_rel": (W1 - W0) / W0 if W0 else float("nan"),
        "H_initial": H0, "H_final": H1,
        "theta_cap_initial_deg": hist[0]["theta_cap_deg"],
        "theta_cap_final_deg": last["theta_cap_deg"],
        "theta_cap_tail_mean_deg": float(np.mean(tail_cap)) if tail_cap else float("nan"),
        "theta_circle_final_deg": last["theta_circle_deg"],
        "theta_circle_tail_mean_deg": float(np.mean(tail_circ)) if tail_circ else float("nan"),
        "blob_area_final": last["blob_area"],
        "blob_phi_max_final": last["blob_phi_max"],
        "u_max_final": last["u_max"],
        "volume_initial": hist[0]["volume"], "volume_final": last["volume"],
        "mass_initial": hist[0]["mass"], "mass_final": last["mass"],
        "wall_time_s": time.time() - t0,
        "history": hist,
    }
    for key, val in (("cap", out["theta_cap_tail_mean_deg"]), ("circle", out["theta_circle_tail_mean_deg"])):
        out[f"err_pct_{key}"] = 100.0 * (val - theta) / theta if math.isfinite(val) else float("nan")
        out[f"pass15_{key}"] = bool(math.isfinite(val) and abs(out[f"err_pct_{key}"]) <= 15.0)
    return out


def expected_sign(theta: float) -> str:
    if theta < 90.0 - 1e-9:
        return "spread (dW>0)"
    if theta > 90.0 + 1e-9:
        return "contract (dW<0)"
    return "neutral (dW~0)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["direction", "relax"], default="direction")
    ap.add_argument("--thetas", default="30,60,90,120,150")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--ny", type=int, default=80)
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--radius", type=float, default=20.0)
    ap.add_argument("--A", type=float, default=0.04)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--xi", type=float, default=2.0)
    ap.add_argument("--sample-every", type=int, default=50)
    ap.add_argument("--top", choices=["wall", "open"], default="wall")
    ap.add_argument("--control", action="store_true", help="also run the wetting=False control")
    ap.add_argument("--out", default="sign_probe.json")
    args = ap.parse_args()

    thetas = [float(t) for t in args.thetas.split(",") if t.strip()]
    common = dict(steps=args.steps, ny=args.ny, nx=args.nx, radius=args.radius, A=args.A,
                  beta=args.beta, tau=args.tau, xi=args.xi, sample_every=args.sample_every,
                  top=args.top)
    arms = {}
    for th in thetas:
        r = run_arm(th, **common)
        arms[f"theta{th:g}"] = r
        tag = expected_sign(th) if args.mode == "direction" else ""
        print(f"[wetting] theta={th:5.1f} dW={r['dW']:+.2f}({r['dW_rel']*100:+.1f}%) "
              f"W={r['W_initial']:.2f}->{r['W_final']:.2f} H={r['H_initial']:.2f}->{r['H_final']:.2f} "
              f"cap={r['theta_cap_tail_mean_deg']:7.2f}({r['err_pct_cap']:+.2f}%) "
              f"circ={r['theta_circle_tail_mean_deg']:7.2f}({r['err_pct_circle']:+.2f}%) "
              f"blob={r['blob_area_final']:.0f} |u|max={r['u_max_final']:.2e} {tag}", flush=True)
    if args.control:
        for th in thetas:
            r = run_arm(th, wetting=False, **common)
            arms[f"theta{th:g}_nobc"] = r
            print(f"[control] theta={th:5.1f} dW={r['dW']:+.2f}({r['dW_rel']*100:+.1f}%) "
                  f"cap={r['theta_cap_tail_mean_deg']:7.2f} circ={r['theta_circle_tail_mean_deg']:7.2f} "
                  f"|u|max={r['u_max_final']:.2e}", flush=True)

    out = {"mode": args.mode, "params": vars(args), "arms": arms,
           "note": "component-restricted sessile-drop probe; see module docstring"}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())