#!/usr/bin/env python3
"""XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE :: solver-level head-to-head.

Case: 2-D sessile drop of a color-gradient (Rothman-Keller / Latva-Kokko) LBM
on a flat no-slip plate, periodic in ``x``, neutral (90 deg) top wall, drop
initialised as the circular cap that satisfies the **prescribed** angle.

Arms
----
* ``cot60`` / ``cot90`` -- the shipped condition
  (:func:`tensorlbm.cg_wetting2d.geometrical_wetting_ghost`, closed form,
  fixed sign; ``lanl/MF-LBM`` port).
* ``oss60`` / ``oss90`` -- the openLBMPM-style two-candidate condition
  (:func:`tensorlbm.wetting.geometric_two_candidate.
  geometric_wetting_ghost_candidates`, same stencil).
* ``ctl60`` -- wetting condition **off** (neutral mirror wall) but the cap
  initialised at 60 deg: sensitivity control, must *not* report 60 deg.

Decision metric (predeclared)
-----------------------------
Contact angle measured with the **reference's own recipe**: cap base width
``b`` + apex height ``h`` -> ``R=(b^2+4h^2)/(8h)``, ``theta=atan2(b/2, R-h)``
(``cap_geometry_contact_angle``; openLBMPM ``cap_geometry_contact_angle`` per
``oss_solutions.id=5``).  Pass = ``|theta_measured - theta_prescribed| /
theta_prescribed <= 15 %``.  A Kasa circle fit of the ``phi=0`` level set and
the boundary-condition read-back are reported as independent cross-checks.

Surface-tension amplitude ``A`` is set to ``0.005`` (not the stock ``0.04``):
at ``A >= 0.01`` the host solver develops a liquid jet that reaches the top
wall within 2000 steps (``umax ~ 1e-2``), which is a *host* defect recorded in
``docs/XVAL-LBM-OPENLBMPM-GEOMETRIC-WETTING.md`` §6 -- at ``A = 0.005`` the cap
stays put (``umax ~ 9e-3``, apex unchanged) and the measurement is meaningful.

Usage: ``python run.py --arms cot60,oss60,ctl60 --steps 8000``
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE.parents[0] / "xval_akai2018_wetting"))

import run as akai  # noqa: E402  (the Akai-2018 campaign harness: step/init/measure)
from tensorlbm.wetting.geometric_two_candidate import (  # noqa: E402
    cap_geometry_contact_angle,
    geometric_wetting_ghost_candidates,
)

GATE_PCT = 15.0
A_STABLE = 0.005

ARMS = {
    "cot60": ("closed", 60.0, True),
    "cot90": ("closed", 90.0, True),
    "cot30": ("closed", 30.0, True),
    "oss60": ("candidates", 60.0, True),
    "oss90": ("candidates", 90.0, True),
    "oss30": ("candidates", 30.0, True),
    "ctl60": ("closed", 90.0, False),
}


def make_step(bc: str, theta_bottom: float, solid, phi_wall):
    """Return a step function with the requested wetting-condition realisation."""

    def step(f_r, f_b, solid_, spec, _theta, solid3):
        rho_r = f_r.sum(dim=0)
        rho_b = f_b.sum(dim=0)
        phi = akai.phase_field(rho_r, rho_b)
        if bc == "closed":
            ghost = akai.geometrical_wetting_ghost(
                phi, solid_, theta_bottom, periodic=(True, False), theta_by_direction={(1, 0): 90.0}
            )
        else:
            ghost = geometric_wetting_ghost_candidates(
                phi, solid_, theta_bottom, periodic=(True, False),
                theta_by_direction={(1, 0): 90.0},
            )
        f_r, f_b = akai.color_gradient_two_tau_step(
            f_r, f_b, tau_r=spec.tau, tau_b=spec.tau, A=spec.A, beta=spec.beta,
            solid_mask=solid_, phi_solid=ghost,
        )
        f_r = akai.stream(f_r)
        f_b = akai.stream(f_b)
        f_r = torch.where(solid3, f_r[list(akai.OPP)], f_r)
        f_b = torch.where(solid3, f_b[list(akai.OPP)], f_b)
        return f_r, f_b

    return step


def run_arm(name: str, steps: int, sample_every: int = 500, save_fields: bool = True) -> dict:
    bc, theta, wetting = ARMS[name]
    spec = akai.ArmSpec(
        name=name, theta_target=theta, init_theta=theta, wetting=wetting,
        A=A_STABLE, beta=0.70, tau=1.0, radius=14.0, nx=96, ny=64, xi=2.0, steps=steps,
    )
    torch.set_num_threads(min(8, torch.get_num_threads() or 1))
    dtype = torch.float64
    solid = akai.build_solid(spec.ny, spec.nx)
    solid3 = solid.unsqueeze(0)
    f_r, f_b = akai.init_drop(spec, dtype=dtype)
    theta_bottom = theta if wetting else 90.0
    stepper = make_step(bc, theta_bottom, solid, None)

    t0 = time.time()
    history = []
    prev_phi = None
    mass0 = None
    for it in range(1, steps + 1):
        f_r, f_b = stepper(f_r, f_b, solid, spec, theta_bottom, solid3)
        if it % sample_every == 0 or it == 1:
            rho_r = f_r.sum(dim=0)
            rho_b = f_b.sum(dim=0)
            phi = akai.phase_field(rho_r.cpu(), rho_b.cpu())
            rho = (rho_r + rho_b).cpu()
            jx = ((f_r + f_b)[1] - (f_r + f_b)[3]).sum(dim=0).cpu()
            jy = ((f_r + f_b)[2] - (f_r + f_b)[4]).sum(dim=0).cpu()
            u = torch.sqrt((jx / rho) ** 2 + (jy / rho) ** 2)
            if mass0 is None:
                mass0 = (float(rho_r.sum()), float(rho_b.sum()))
            dphi = float((phi - prev_phi).abs().max()) if prev_phi is not None else float("nan")
            prev_phi = phi
            pn = phi.numpy()
            cap = cap_geometry_contact_angle(pn, y_wall=0.5)
            try:
                circ = akai.contact_angle_circle_fit(pn, 0.5, trim=0.05, level=0.0)["theta_deg"]
            except Exception:
                circ = float("nan")
            history.append(
                {
                    "step": it,
                    "theta_capgeom": cap["theta_deg"],
                    "theta_circle": circ,
                    "b": cap["b"],
                    "h": cap["h"],
                    "u_max": float(u[~solid].max()),
                    "dphi_max": dphi,
                    "mass_drift": (float(rho_r.sum()) - mass0[0]) / mass0[0],
                }
            )
            print(
                f"  [{name}] {it:6d}  capgeom={cap['theta_deg']:7.2f}  circle={circ:7.2f}  "
                f"b={cap['b']:6.2f} h={cap['h']:6.2f}  |u|max={float(u[~solid].max()):.2e}  "
                f"dphi={dphi:.2e}",
                flush=True,
            )

    phi = akai.phase_field(f_r.sum(0).cpu(), f_b.sum(0).cpu()).numpy()
    final_cap = cap_geometry_contact_angle(phi, y_wall=0.5)
    try:
        final_circle = akai.contact_angle_circle_fit(phi, 0.5, trim=0.05, level=0.0)["theta_deg"]
    except Exception:
        final_circle = float("nan")
    phi_t = torch.from_numpy(phi).to(dtype)
    if bc == "closed":
        ghost = akai.geometrical_wetting_ghost(phi_t, solid, theta_bottom, periodic=(True, False),
                                               theta_by_direction={(1, 0): 90.0})
    else:
        ghost = geometric_wetting_ghost_candidates(phi_t, solid, theta_bottom, periodic=(True, False),
                                                  theta_by_direction={(1, 0): 90.0})
    face_bottom = torch.zeros(spec.ny, spec.nx, dtype=torch.bool)
    face_bottom[1, :] = True
    ang = akai.wall_local_contact_angle(phi_t, ghost, solid, face_subset=face_bottom)

    tail = [h for h in history if h["step"] > 0.75 * history[-1]["step"]]
    out = {
        "arm": name,
        "bc": bc,
        "wetting": wetting,
        "theta_prescribed": theta,
        "theta_bottom_applied": theta_bottom,
        "steps_run": steps,
        "wall_time_s": time.time() - t0,
        "A": A_STABLE, "beta": spec.beta, "tau": spec.tau, "radius": spec.radius,
        "nx": spec.nx, "ny": spec.ny, "xi": spec.xi,
        "theta_capgeom": final_cap["theta_deg"],
        "theta_capgeom_status": final_cap["status"],
        "b": final_cap["b"],
        "h": final_cap["h"],
        "theta_circle": final_circle,
        "theta_bc_readback_median": float(ang.median()) if ang.numel() else float("nan"),
        "theta_bc_readback_n": int(ang.numel()),
        "theta_capgeom_tail_mean": float(np.nanmean([h["theta_capgeom"] for h in tail])),
        "theta_capgeom_tail_std": float(np.nanstd([h["theta_capgeom"] for h in tail])),
        "u_max_fluid": history[-1]["u_max"],
        "mass_drift": history[-1]["mass_drift"],
        "dphi_final": history[-1]["dphi_max"],
        "history": history,
        "phi_sha256": __import__("hashlib").sha256(phi.astype(np.float64).tobytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "machine": platform.machine(),
        },
    }
    out["rel_err_pct_capgeom"] = 100.0 * abs(out["theta_capgeom"] - theta) / theta
    out["rel_err_pct_circle"] = (
        100.0 * abs(out["theta_circle"] - theta) / theta if math.isfinite(out["theta_circle"]) else float("nan")
    )
    out["gate_pass_capgeom"] = bool(out["rel_err_pct_capgeom"] <= GATE_PCT)
    out["gate_pass_control"] = bool(not wetting and abs(out["theta_capgeom"] - 60.0) / 60.0 > GATE_PCT)
    if save_fields:
        np.save(HERE / f"phi_{name}.npy", phi)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="cot60,oss60,ctl60")
    ap.add_argument("--steps", type=int, default=8000)
    args = ap.parse_args(argv)
    results = {}
    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        print(f"== arm {name}", flush=True)
        res = run_arm(name, args.steps)
        results[name] = res
        (HERE / f"results_{name}.json").write_text(json.dumps(res, indent=1, default=float))
        (HERE / "results.json").write_text(json.dumps({"results": results}, indent=1, default=float))
    for name, r in results.items():
        print(
            f"{name:7s} prescribed={r['theta_prescribed']:5.1f}  capgeom={r['theta_capgeom']:7.2f} "
            f"({r['rel_err_pct_capgeom']:5.1f}%)  circle={r['theta_circle']:7.2f}  "
            f"readback={r['theta_bc_readback_median']:7.2f}  |u|={r['u_max_fluid']:.1e}  "
            f"pass={r['gate_pass_capgeom']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())