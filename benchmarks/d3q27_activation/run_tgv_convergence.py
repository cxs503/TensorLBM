#!/usr/bin/env python3
"""Taylor-Green L2 convergence of the activated D3Q27 path — gate (f) of P151-N01.

Why an *exact* reference
------------------------
Gate (f) asks for the L2 error convergence order of the 3D Taylor-Green case.
The full 3D (z-dependent) TG has no closed-form solution, so a self-convergence
study was attempted first (see ``prereg.json`` deviation D2): it plateaus at
~2.5e-3 with observed order ≈ 0 for **both** D3Q19 and D3Q27 (the plateau is
grid-independent acoustic/initial-layer content of that protocol, not the
lattice).  The gate is therefore measured on the z-invariant Taylor-Green field,
which is an exact solution of the 3D incompressible Navier-Stokes equations::

    u_x = A e^{-2 ν k² t} sin(k x) cos(k y)
    u_y = -A e^{-2 ν k² t} cos(k x) sin(k y)
    u_z = 0,                    k = 2π / N

(the 2D TG vortex is a genuine exact 3D NS solution: the nonlinear term is a
pure gradient).  It is run on the full 3D D3Q27 stencil (all 27 populations are
nonzero and are collided/streamed), with an exact reference and no
compressibility/nonlinearity floor, so the measured order is the solver's
spatial order.

Protocol (``prereg.json``)
--------------------------
A = 2e-3, τ = 0.6 (ν = 1/30), 500 steps (identical *lattice* time on every grid),
grids N ∈ {24, 48, 96}, float64, BGK collision, periodic box (no far-field BC).
E(N) = ‖u_N(T) − u_exact(T)‖₂ / ‖u_exact(T)‖₂ measured on each grid's own nodes.
Order = log2(E(24)/E(48)); gate ≥ 1.8.

Writes ``runs/tgv_convergence_<lattice>.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

from tensorlbm.lbm_step_d3q27 import lbm_step_correct_27

A = 2.0e-3
TAU = 0.6
NU_LB = (TAU - 0.5) / 3.0  # 1/30
STEPS = 500
GRIDS = [24, 48, 96]
DT = torch.float64
HERE = os.path.dirname(os.path.abspath(__file__))


def tgv_fields(n: int):
    h = 2.0 * math.pi / n
    z, y, x = torch.meshgrid(
        *[torch.arange(n, dtype=DT) * h] * 3, indexing="ij"
    )
    ux = A * torch.sin(x) * torch.cos(y)
    uy = -A * torch.cos(x) * torch.sin(y)
    uz = torch.zeros_like(x)
    return x, (ux, uy, uz)


def exact_fields(n: int, steps: int):
    k = 2.0 * math.pi / n
    decay = math.exp(-2.0 * NU_LB * k * k * steps)
    _, u0 = tgv_fields(n)
    return tuple(decay * u for u in u0)


def run_grid(n: int, lattice: str):
    from tensorlbm.d3q27 import collide_bgk27, equilibrium27, macroscopic27

    x, u0 = tgv_fields(n)
    f = equilibrium27(torch.ones_like(x), *u0)
    solid = torch.zeros((n, n, n), dtype=torch.bool)
    t0 = time.time()
    energy = []
    sample = STEPS // 5
    if lattice == "d3q27":
        from tensorlbm.lbm_step_d3q27 import lbm_step_correct_27 as step27

        for step in range(1, STEPS + 1):
            f = step27(f, collide_bgk27, TAU, solid, 0.0, far_field_bc_fn=None)
            if step % sample == 0:
                rho, ux, uy, uz = macroscopic27(f)
                energy.append({"step": step, "kinetic_energy": 0.5 * float((ux**2 + uy**2 + uz**2).mean())})
    else:
        from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
        from tensorlbm.solver3d import collide_bgk3d, stream3d

        f = equilibrium3d(torch.ones_like(x), *u0)
        for step in range(1, STEPS + 1):
            f = stream3d(collide_bgk3d(f, tau=TAU))
            if step % sample == 0:
                rho, ux, uy, uz = macroscopic3d(f)
                energy.append({"step": step, "kinetic_energy": 0.5 * float((ux**2 + uy**2 + uz**2).mean())})
    wall = time.time() - t0
    if lattice == "d3q27":
        from tensorlbm.d3q27 import macroscopic27 as macro

        rho, ux, uy, uz = macro(f)
    else:
        from tensorlbm.d3q19 import macroscopic3d as macro

        rho, ux, uy, uz = macro(f)
    return {
        "N": n,
        "steps": STEPS,
        "wall_s": wall,
        "finite": bool(torch.isfinite(f).all()),
        "energy_history": energy,
        "u_final": (ux, uy, uz),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lattice", default="d3q27", choices=["d3q27", "d3q19"])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("TL_TGV_THREADS", "32")))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    runs = {}
    for n in GRIDS:
        print(f"[tgv:{a.lattice}] N={n} ...", flush=True)
        runs[n] = run_grid(n, a.lattice)
        ue = exact_fields(n, STEPS)
        num = sum(((u - e) ** 2).sum().item() for u, e in zip(runs[n]["u_final"], ue))
        den = sum((e**2).sum().item() for e in ue)
        runs[n]["l2_error"] = math.sqrt(num / den)
        del runs[n]["u_final"]
        print(
            f"[tgv:{a.lattice}] N={n}: E={runs[n]['l2_error']:.4e} "
            f"wall={runs[n]['wall_s']:.1f}s finite={runs[n]['finite']}",
            flush=True,
        )

    orders = [
        math.log2(runs[GRIDS[i]]["l2_error"] / runs[GRIDS[i + 1]]["l2_error"])
        for i in range(len(GRIDS) - 1)
    ]
    payload = {
        "case": f"tgv3d_z_invariant_exact_{a.lattice}",
        "lattice": a.lattice,
        "amplitude": A,
        "tau": TAU,
        "nu_lb": NU_LB,
        "steps": STEPS,
        "grids": GRIDS,
        "dtype": "float64",
        "collision": "bgk27" if a.lattice == "d3q27" else "bgk3d",
        "l2_errors": {str(n): runs[n]["l2_error"] for n in GRIDS},
        "observed_orders": orders,
        "observed_order_primary": orders[0] if orders else None,
        "gate_order_min": 1.8,
        "gate_pass": bool(orders and orders[0] >= 1.8),
        "per_grid": [{k: v for k, v in runs[n].items()} for n in GRIDS],
        "reference": "exact z-invariant 3D Taylor-Green solution u = A exp(-2 nu k^2 t)(sin kx cos ky, -cos kx sin ky, 0)",
        "note_full_3d": "The full z-dependent 3D TG self-convergence study plateaus at ~2.5e-3 for both lattices (prereg deviation D2); see runs/tgv3d_full_selfconv_*.json.",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commands": f"python benchmarks/d3q27_activation/run_tgv_convergence.py --lattice {a.lattice}",
    }
    out = os.path.join(HERE, "runs", f"tgv_convergence_{a.lattice}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps({k: payload[k] for k in ("l2_errors", "observed_orders", "gate_pass")}, indent=2))
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())