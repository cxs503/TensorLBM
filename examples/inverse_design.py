"""Differentiable geometry inverse design through the soft-solid path.

The solid in ``tensorlbm.autograd_path`` can be a **soft solid**
(``tensorlbm.soft_geometry.SoftGeometry``): an analytic signed distance
function blurred by a temperature into a differentiable weight field, wired
into the step chain as soft NoDynamics collision, soft full-way bounce-back
and the soft momentum-exchange force (all convex homotopies of the hard-mask
operators, degenerating bit-for-bit at saturated weights).  Every geometric
parameter is then a 0-dim tensor in the autograd graph, so an observable
measured after ``--steps`` solver steps can be optimised directly against
the geometry itself — shape inverse design with the solver in the loop.

Problem setup: a uniform inflow ``u = (u_in, 0, 0)`` drives a bounded box
(equilibrium velocity inlet on x = 0, zero-gradient outlet on x = nx-1,
lateral planes periodic) around a soft obstacle.  A *hidden* true geometry
produces the reference drag; gradient descent starting from a wrong guess
must recover it.  The matched observable is the window-mean momentum-exchange
drag (probes ``--probe-start .. --steps``), reported as C_D with the truth
projected area as the fixed normaliser.

Modes
-----
``--param radius`` (default) — learn the sphere radius (truth 2.3, grid
12×16×26, guess 2.9).

``--param semi-axis`` — learn the y semi-axis of a triaxial ellipsoid
(a, b, c) with the other two axes fixed (truth b = 2.2, guess 2.6).  The
window-drag observable is *multi-modal* in b: it has an interior minimum
near b = c (the round-cross-section shape, ~1.6 here) and two geometries
with the same drag on either side (measured: b = 0.8 and b = 2.2 give the
same window drag to 7e-4), so the initial guess must sit on the truth
branch — descending from below the minimum converges to the twin solution
on the slender branch instead.

``--target hard`` — generate the reference drag from the *hard-mask* campaign
of the same parameters instead of the soft one: the optimiser then still
nails the target C_D, but at a shifted radius — the visible signature of the
epsilon bias documented in ``docs/differentiable_path.md``.  The bias is
window-dependent (measured on this grid: the soft campaign overestimates the
drag of the same sphere by ~1.6x in the startup window 30..60, but only by
+0.5% in the steady window 300..400), so the recovered radius sits well
below the truth at the default short window — run with ``--iters 400
--lr 0.05`` (the longer descent needs the budget; measured: the target C_D
is matched to 1.5e-07 at radius 1.690 vs truth 2.3).  Default
``--target soft`` is the self-consistent soft reference, which recovers the
truth radius.

Usage
-----
    python examples/inverse_design.py                          # sphere radius
    python examples/inverse_design.py --param semi-axis        # ellipsoid b
    python examples/inverse_design.py --target hard --iters 400 --lr 0.05  # epsilon bias
    python examples/inverse_design.py --eps 0.125 --iters 160
    python examples/inverse_design.py --device cuda            # any free GPU
"""

from __future__ import annotations

import argparse
import math

import torch

from tensorlbm.autograd_path import (
    InletSpec,
    OutletSpec,
    differentiable_step,
    obstacle_force,
    rollout,
)
from tensorlbm.boundaries3d import sphere_mask
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.soft_geometry import SoftGeometry

NZ, NY, NX = 12, 16, 26
CX, CY, CZ = 7.0, 8.0, 6.0
R_STAR = 2.3
ELL_A, ELL_B_STAR, ELL_C = 2.0, 2.2, 1.6
U_IN = 0.08
TAU = 0.55
STEPS = 60
PROBE_START = 30
EPS_DEFAULT = 0.25
GUESS_RADIUS, GUESS_SEMI = 2.9, 2.6
LR_DEFAULT = 0.02
ITERS_DEFAULT = 120


def uniform_inflow_f0(
    u_in: float, nz: int, ny: int, nx: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """D3Q19 equilibrium initialisation at the uniform inflow state."""
    ones = torch.ones(nz, ny, nx, dtype=dtype, device=device)
    zeros = torch.zeros_like(ones)
    return equilibrium3d(ones, u_in * ones, zeros, zeros)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--param", choices=["radius", "semi-axis"], default="radius", help="learned parameter"
    )
    parser.add_argument(
        "--target",
        choices=["soft", "hard"],
        default="soft",
        help="reference campaign the target drag comes from (hard shows the epsilon bias)",
    )
    parser.add_argument(
        "--eps", type=float, default=EPS_DEFAULT, help="SDF temperature (lattice units)"
    )
    parser.add_argument("--steps", type=int, default=STEPS, help="solver steps per rollout (K)")
    parser.add_argument(
        "--probe-start", type=int, default=PROBE_START, help="first probe of the drag window"
    )
    parser.add_argument("--iters", type=int, default=ITERS_DEFAULT, help="optimisation iterations")
    parser.add_argument("--lr", type=float, default=LR_DEFAULT, help="Adam learning rate")
    parser.add_argument("--init", type=float, default=None, help="initial parameter guess")
    parser.add_argument("--u-in", type=float, default=U_IN, help="inlet velocity")
    parser.add_argument("--device", default="cpu", help="torch device (e.g. cpu, cuda, cuda:3)")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)

    semi = args.param == "semi-axis"
    truth = ELL_B_STAR if semi else R_STAR
    guess = args.init if args.init is not None else (GUESS_SEMI if semi else GUESS_RADIUS)

    def geometry(param: float | torch.Tensor) -> SoftGeometry:
        if semi:
            return SoftGeometry(
                kind="ellipsoid",
                center=(CX, CY, CZ),
                size=(ELL_A, param, ELL_C),
                epsilon=args.eps,
            )
        return SoftGeometry(kind="sphere", center=(CX, CY, CZ), size=(param,), epsilon=args.eps)

    # fixed normaliser: the truth geometry's projected area (frontal to the x flow)
    area = math.pi * (ELL_B_STAR * ELL_C if semi else R_STAR * R_STAR)
    q_dyn = 0.5 * args.u_in * args.u_in
    inlet, outlet = InletSpec(ux=args.u_in), OutletSpec()
    f0 = uniform_inflow_f0(args.u_in, NZ, NY, NX, dtype, device)

    def window_drag(param: float | torch.Tensor) -> torch.Tensor:
        """Mean momentum-exchange drag over the probe window (graph kept)."""
        geom = geometry(param)
        _f, probes = rollout(
            f0,
            args.steps,
            TAU,
            None,
            soft=geom,
            inlet=inlet,
            outlet=outlet,
            return_probes=True,
            probe_start=args.probe_start,
        )
        w = geom.fluid_weight(NZ, NY, NX, dtype=dtype, device=device)
        return sum(obstacle_force(p, 1.0 - w)[0] for p in probes) / len(probes)

    # --- reference ("measured") target: soft truth, or the hard-mask campaign
    with torch.no_grad():
        drag_star = window_drag(torch.tensor(truth, dtype=dtype, device=device))
        if args.target == "hard":
            # hard-mask campaign of the truth geometry (sphere: radius; the
            # ellipsoid hard reference uses a sphere of the learned axis)
            hard_radius = ELL_B_STAR if semi else R_STAR
            mask = sphere_mask(NX, NY, NZ, CX, CY, CZ, hard_radius, device=device)
            f = f0
            hard_drags = []
            for _ in range(args.steps):
                f, probe = differentiable_step(
                    f, TAU, mask, return_probe=True, inlet=inlet, outlet=outlet
                )
                hard_drags.append(obstacle_force(probe, mask)[0])
            drag_star = torch.stack(hard_drags[args.probe_start :]).mean(0)
    cd_star = float(drag_star) / (q_dyn * area)

    # --- solver-in-the-loop optimisation of the geometry parameter
    param = torch.tensor(guess, dtype=dtype, device=device, requires_grad=True)
    optim = torch.optim.Adam([param], lr=args.lr)
    clamp = (0.5, min(NY, NZ) / 2 - 0.5)  # keep the obstacle inside the box
    print(
        f"inverse design: param={args.param}, target={args.target} reference, eps={args.eps}, "
        f"K={args.steps} steps (window from {args.probe_start}), "
        f"{args.iters} Adam iters (lr={args.lr}), {args.dtype}, device={device}"
    )
    print(
        f"grid ({NZ}, {NY}, {NX}), hidden truth = {truth:.4f}, initial guess = {guess:.4f}, "
        f"target C_D = {cd_star:.6f}"
    )

    loss0 = None
    best_loss, best_param = math.inf, guess
    for it in range(args.iters):
        # cosine learning-rate decay with a 5% floor (keeps a landing velocity)
        for group in optim.param_groups:
            group["lr"] = args.lr * (
                0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * it / args.iters))
            )
        drag = window_drag(param)
        loss = ((drag - drag_star) / drag_star) ** 2
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            param.clamp_(*clamp)
        loss_val = float(loss.detach())
        if loss0 is None:
            loss0 = loss_val
        if loss_val < best_loss:
            best_loss, best_param = loss_val, float(param.detach())
        if it == 0 or (it + 1) % args.log_every == 0:
            cd = float(drag.detach()) / (q_dyn * area)
            p_val = float(param.detach())
            print(
                f"  iter {it + 1:4d}  loss={loss_val:.6e}  param={p_val:.6f}  "
                f"C_D={cd:.6f}  rel_err={(cd - cd_star) / cd_star:+.2e}"
            )

    with torch.no_grad():
        cd_end = float(window_drag(param.detach())) / (q_dyn * area)
    print(
        f"result: loss {loss0:.6e} -> {float(loss.detach()):.6e} (best {best_loss:.6e} at "
        f"param={best_param:.6f}); param {float(param.detach()):.6f} vs truth {truth:.4f} "
        f"(abs err {abs(float(param.detach()) - truth):.2e}); "
        f"C_D endpoint error {abs((cd_end - cd_star) / cd_star):.2e}"
    )


if __name__ == "__main__":
    main()
