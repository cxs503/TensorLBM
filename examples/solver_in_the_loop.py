"""Solver-in-the-loop parameter identification through the differentiable path.

The discrete D3Q19 solver step packaged in ``tensorlbm.autograd_path``
(BGK-family collision on fluid cells -> periodic gather streaming ->
full-way bounce-back on a solid mask) is differentiable end to end, so a
scalar observable measured after ``--steps`` solver steps can be optimised
directly against a solver parameter — the time-stepping operator itself is
in the backward graph (the paradigm of Um et al., NeurIPS 2020, and Autodesk
XLB's differentiable-lbm demos, Apache-2.0; no XLB code is used here).

Problem setup (identical for both modes): a periodic box with a centred
sphere obstacle is initialised with a decaying shear flow.  A *hidden*
parameter is used to roll out the reference; gradient descent starting from
a wrong guess must recover it.

Modes
-----
``--mode tau`` — viscosity inverse problem.  Learn the BGK relaxation time
``tau`` (truth 0.85) through the solver.

``--mode cs`` — sub-grid model identification.  Learn the Smagorinsky
constant ``C_s`` (truth 0.12, molecular tau fixed at 0.55) through the
BGK-Smagorinsky collision slot: the eddy viscosity is part of the simulated
observable, so the LES constant is identifiable from the flow itself.

Observables
-----------
``--observable field`` — mean squared error against the target final
``ux`` field on the fluid cells.

``--observable drag`` — the momentum-exchange drag on the obstacle
(``autograd_path.obstacle_force``, Ladd wet-node convention, sampled
post-stream / pre-bounce-back at every step) is accumulated over the rollout
and matched to the reference value.

Usage
-----
    python examples/solver_in_the_loop.py                     # tau recovery, field loss
    python examples/solver_in_the_loop.py --mode cs           # Smagorinsky C_s recovery
    python examples/solver_in_the_loop.py --mode tau --observable drag
    python examples/solver_in_the_loop.py --device cuda       # any free GPU
"""

from __future__ import annotations

import argparse
import functools
import math

import torch

from tensorlbm.autograd_path import obstacle_force, rollout
from tensorlbm.boundaries3d import sphere_mask
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.turbulence import collide_smagorinsky_bgk3d

NZ, NY, NX = 12, 16, 24
RADIUS = 3.5
TAU_STAR = 0.85
TAU0_SMAG = 0.55
CS_STAR = 0.12
TWO_PI = 2.0 * math.pi


def shear_flow_f0(amplitude: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """D3Q19 equilibrium initialisation u = a*(sin(2*pi*y/ny) + 0.3*cos(2*pi*z/nz)) x-hat."""
    zz, yy, _xx = torch.meshgrid(
        torch.arange(NZ, dtype=dtype, device=device),
        torch.arange(NY, dtype=dtype, device=device),
        torch.arange(NX, dtype=dtype, device=device),
        indexing="ij",
    )
    ux = amplitude * (torch.sin(TWO_PI * yy / NY) + 0.3 * torch.cos(TWO_PI * zz / NZ))
    zeros = torch.zeros_like(ux)
    return equilibrium3d(torch.ones_like(ux), ux, zeros, zeros)


def smagorinsky_collide(cs: torch.Tensor):
    """Collision slot bound to a (learnable) Smagorinsky constant."""
    return functools.partial(collide_smagorinsky_bgk3d, C_s=cs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode", choices=["tau", "cs"], default="tau", help="parameter to identify"
    )
    parser.add_argument(
        "--observable", choices=["field", "drag"], default="field", help="matched quantity"
    )
    parser.add_argument("--steps", type=int, default=15, help="solver steps per rollout (K)")
    parser.add_argument("--iters", type=int, default=120, help="optimisation iterations")
    parser.add_argument("--lr", type=float, default=None, help="Adam learning rate")
    parser.add_argument("--amplitude", type=float, default=0.1, help="shear-flow amplitude")
    parser.add_argument("--init", type=float, default=None, help="initial parameter guess")
    parser.add_argument("--device", default="cpu", help="torch device (e.g. cpu, cuda, cuda:3)")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    steps, iters = args.steps, args.iters

    f0 = shear_flow_f0(args.amplitude, dtype, device)
    mask = sphere_mask(NX, NY, NZ, cx=NX / 2, cy=NY / 2, cz=NZ / 2, radius=RADIUS, device=device)
    fluid = ~mask

    def ux_fluid(f: torch.Tensor) -> torch.Tensor:
        _rho, ux, _uy, _uz = macroscopic3d(f)
        return ux[fluid]

    if args.mode == "tau":
        truth, clamp, init_guess = TAU_STAR, (0.5005, 5.0), 0.6
        # the drag observable is ~quadratic in tau with small curvature: it
        # converges to the same optimum but wants a larger learning rate
        default_lr = 0.02 if args.observable == "field" else 0.05

        def rollout_with(param: torch.Tensor, probes: bool = False):
            return rollout(f0, steps, param, mask, return_probes=probes)

    else:
        truth, clamp, init_guess = CS_STAR, (0.0, 0.5), 0.03
        default_lr = 0.01

        def rollout_with(param: torch.Tensor, probes: bool = False):
            return rollout(
                f0,
                steps,
                TAU0_SMAG,
                mask,
                collide=smagorinsky_collide(param),
                return_probes=probes,
            )

    lr = args.lr if args.lr is not None else default_lr
    guess = args.init if args.init is not None else init_guess

    def accumulated_drag(probes: list[torch.Tensor]) -> torch.Tensor:
        return sum(obstacle_force(p, mask)[0] for p in probes)

    # --- reference ("measured") data produced with the hidden true parameter
    with torch.no_grad():
        f_true, probes_true = rollout_with(
            torch.tensor(truth, dtype=dtype, device=device), probes=True
        )
        target_ux = ux_fluid(f_true)
        drag_true = accumulated_drag(probes_true)

    def loss_of(param: torch.Tensor) -> torch.Tensor:
        f_end, probes = rollout_with(param, probes=True)
        if args.observable == "field":
            return ((ux_fluid(f_end) - target_ux) ** 2).mean()
        return (accumulated_drag(probes) - drag_true) ** 2

    # --- solver-in-the-loop optimisation
    param = torch.tensor(guess, dtype=dtype, device=device, requires_grad=True)
    optim = torch.optim.Adam([param], lr=lr)
    print(
        f"solver-in-the-loop identification: mode={args.mode} observable={args.observable} "
        f"K={steps} steps, {iters} Adam iters (lr={lr}), {args.dtype}, device={device}"
    )
    print(f"grid ({NZ}, {NY}, {NX}), sphere r={RADIUS}, fluid cells={int(fluid.sum())}")
    print(f"hidden truth = {truth:.4f}, initial guess = {guess:.4f}")

    loss0 = None
    for it in range(iters):
        # cosine learning-rate decay: fast approach first, smooth landing on
        # the recoverable optimum (the loss is ~quadratic in the parameter)
        for group in optim.param_groups:
            group["lr"] = lr * 0.5 * (1.0 + math.cos(math.pi * it / iters))
        loss = loss_of(param)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            param.clamp_(*clamp)
        loss_val = loss.detach().item()
        if loss0 is None:
            loss0 = loss_val
        if it == 0 or (it + 1) % args.log_every == 0:
            print(
                f"  iter {it + 1:4d}  loss={loss_val:.6e}  "
                f"param={param.detach().item():.6f}  "
                f"err={abs(param.detach().item() - truth):.2e}"
            )

    err = abs(param.detach().item() - truth)
    print(
        f"result: loss {loss0:.6e} -> {loss_val:.6e} "
        f"(reduced x{loss0 / max(loss_val, 1e-300):.1f}), "
        f"param {param.detach().item():.6f} vs truth {truth:.4f} "
        f"(abs err {err:.2e}, rel err {err / truth:.2e})"
    )


if __name__ == "__main__":
    main()
