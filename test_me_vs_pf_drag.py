"""Test: D3Q19 Momentum Exchange drag vs Pressure-Face drag on SUBOFF bare_hull.

Runs D3Q19 MRT+Smag Cs=0.05 bare_hull 160³ for 1000 steps,
computing BOTH pressure-face drag AND momentum exchange drag at each step.
Reports Ct_ME vs Ct_pf at steps 200/400/600/800/1000 and stability comparison.

Usage:
    PYTHONPATH=src python test_me_vs_pf_drag.py [--device cuda]
"""
from __future__ import annotations

import argparse
import math
import sys

import torch

from tensorlbm.d3q19 import C, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.suboff_cad import build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.wall_model import wall_function_3d as wall_fn_main
from tensorlbm.obstacles import compute_obstacle_forces_3d


# ---------------------------------------------------------------------------
# Link-wise D3Q19 momentum exchange (only fluid→solid links)
# ---------------------------------------------------------------------------
def compute_me_linkwise_3d(
    f: torch.Tensor,
    solid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Link-wise momentum exchange force on a stationary 3-D obstacle (D3Q19).

    For each solid cell, for each population direction i:
      If the source neighbour (solid cell - c_i) is fluid:
        f_in = f_i at solid (post-streaming, before bounce-back)
        f_out = f_opposite_i at solid (the bounced population)
        momentum_exchange = (f_in + f_out) * c_i_x

    For wall-function simulations without bounce-back, f_out is the
    pre-collision population at the opposite direction (the one that
    streams back to the fluid in the next step).  When bounce-back IS
    applied (standard BC), f_out = f_in by construction.

    Must be called **after streaming but before the wall function
    modifies near-wall fluid cells**.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)`` after streaming.
        solid: Boolean tensor of shape ``(nz, ny, nx)`` marking solid cells.

    Returns:
        Tuple ``(fx, fy, fz)`` — scalar tensors.
    """
    device = f.device
    fluid = ~solid
    c = C.to(device).float()  # (19, 3)
    opp = OPPOSITE.to(device)  # (19,)

    nz, ny, nx = solid.shape
    fx = torch.tensor(0.0, device=device)
    fy = torch.tensor(0.0, device=device)
    fz = torch.tensor(0.0, device=device)

    for q in range(19):
        cqx, cqy, cqz = int(c[q, 0].item()), int(c[q, 1].item()), int(c[q, 2].item())
        q_opp = int(opp[q].item())

        # f_in: population at solid cell that arrived via streaming from x - c_q
        f_in = f[q][solid]  # shape (n_solid,)

        # Check if the source neighbour is fluid
        # solid cell at (z, y, x) receives from (z-cz, y-cy, x-cx)
        # We need to check: is (z-cz, y-cy, x-cx) a fluid cell?
        # Use torch.roll to align fluid mask with solid cells
        # Roll fluid by (+cz, +cy, +cx) so that fluid_shifted[solid] = fluid[source_pos]
        fluid_shifted = torch.roll(fluid, shifts=(cqz, cqy, cqx), dims=(0, 1, 2))
        is_fluid_link = fluid_shifted[solid]  # bool at solid positions

        if not is_fluid_link.any():
            continue

        # f_out: population at opposite direction (the one that streams back)
        f_out = f[q_opp][solid]  # shape (n_solid,)

        # Contribution: (f_in + f_out) * c_q_x, only counting fluid→solid links
        contrib = (f_in + f_out) * is_fluid_link.to(f.dtype)
        fx = fx + float(cqx) * contrib.sum()
        fy = fy + float(cqy) * contrib.sum()
        fz = fz + float(cqz) * contrib.sum()

    return fx, fy, fz


def compute_me_streaming_3d(
    f: torch.Tensor,
    solid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Streaming-only momentum exchange (fluid→solid pre-streaming check).

    Computes the momentum flux from fluid into solid by checking, BEFORE
    streaming, which fluid populations will stream into solid cells.
    This avoids relying on solid-cell populations that may be polluted
    in wall-function simulations (no bounce-back).

    For each fluid cell at position x, direction q:
      If x + c_q is a solid cell:
        contribution = f[q](x) * c_q

    Equivalent to f_in-only ME on solid cells after streaming, but
    with proper boundary handling.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)`` — BEFORE streaming.
        solid: Boolean tensor of shape ``(nz, ny, nx)`` marking solid cells.

    Returns:
        Tuple ``(fx, fy, fz)`` — scalar tensors.
    """
    device = f.device
    fluid = ~solid
    c = C.to(device).float()

    nz, ny, nx = solid.shape
    fx = torch.tensor(0.0, device=device)
    fy = torch.tensor(0.0, device=device)
    fz = torch.tensor(0.0, device=device)

    # Pad solid mask with zeros so we can safely check x + c_q
    solid_pad = torch.nn.functional.pad(
        solid.unsqueeze(0).unsqueeze(0).float(),
        (1, 1, 1, 1, 1, 1), mode='constant', value=0.0
    ).squeeze(0).squeeze(0).bool()

    for q in range(19):
        cqx, cqy, cqz = int(c[q, 0].item()), int(c[q, 1].item()), int(c[q, 2].item())

        # f[q] at (z, y, x) will stream to (z+cz, y+cy, x+cx)
        # Check if the destination is solid (using padded mask)
        # solid_pad has shape (nz+2, ny+2, nx+2) with padding=1
        # For cell at (z, y, x) in original domain,
        # destination in padded coords is (z+1+cz, y+1+cy, x+1+cx)
        sz = slice(1 + cqz, 1 + cqz + nz)
        sy = slice(1 + cqy, 1 + cqy + ny)
        sx = slice(1 + cqx, 1 + cqx + nx)
        dest_is_solid = solid_pad[sz, sy, sx]  # (nz, ny, nx), True where stream target is solid

        # Only count fluid cells where stream target is solid
        valid = fluid & dest_is_solid
        if not valid.any():
            continue

        contrib = f[q][valid]  # populations at fluid cells streaming INTO solid
        fx = fx + float(cqx) * contrib.sum()
        fy = fy + float(cqy) * contrib.sum()
        fz = fz + float(cqz) * contrib.sum()

    return fx, fy, fz


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def run(
    re: float = 2e6,
    nx: int = 160,
    ny: int = 160,
    nz: int = 160,
    u_in: float = 0.06,
    cs: float = 0.05,
    n_steps: int = 1000,
    warmup: int = 200,
    y_val: float = 0.5,
    device: str = "cuda",
):
    hull_length = nx * 0.6  # default proportion
    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5

    cx = nx * 0.35
    cy = ny / 2.0
    cz = nz / 2.0

    print(f"=== D3Q19 MRT+Smag Cs={cs} bare_hull {nx}³ ===", flush=True)
    print(f"Re={re:.0e} tau_lam={tau:.5f} nu_lat={nu_lat:.2e} hull_L={hull_length:.0f}", flush=True)
    print(f"u_in={u_in} y_val={y_val}\n", flush=True)

    solid, _stats = build_suboff_mask(
        hull_type="bare_hull",
        nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz,
        length=hull_length,
        device=device,
    )
    S_wet = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in * u_in * S_wet
    print(f"wetted area S={S_wet:.0f}  dyn_p_S={dyn_p_S:.6f}\n", flush=True)

    # Initialize
    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    uy0 = torch.zeros_like(ux0)
    uz0 = torch.zeros_like(ux0)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=torch.device(device))
    initial_mass = float(rho0.sum().item())

    # Lists for post-warmup drag history
    pf_fric: list[float] = []
    pf_pres: list[float] = []
    me_link_fx: list[float] = []
    me_stream_fx: list[float] = []

    print(f"{'Step':>6s}  {'Ct_ME_link':>12s}  {'Ct_ME_stream':>14s}  {'Ct_pf_fric':>12s}  {'Ct_pf_pres':>12s}  {'Ct_pf_tot':>12s}  {'max|u|':>10s}")
    print("-" * 95, flush=True)

    for step in range(1, n_steps + 1):
        # 1. Collision (MRT + Smagorinsky)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs)

        # 2a. Compute streaming-only ME drag BEFORE streaming (pre-stream f)
        #     Uses fluid→solid link check on pre-stream populations
        me_stream_fx_val, _, _ = compute_me_streaming_3d(f, solid)
        me_stream_step = float(me_stream_fx_val.item())

        # 2b. Streaming
        f = stream3d(f)

        # 2c. Compute link-wise ME drag AFTER streaming (post-stream f on solid)
        me_link_fx_val, _, _ = compute_me_linkwise_3d(f, solid)
        me_link_step = float(me_link_fx_val.item())

        # 3. Wall function (body force + pressure-face drag)
        f, drag_fric, drag_pres = wall_fn_main(f, solid, nu_lat, y_val=y_val)

        # 4. Far-field BC
        f = far_field_bc_3d(f, u_in=u_in)

        # 5. Mass correction every 100 steps
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        # Record post-warmup
        if step > warmup and math.isfinite(drag_fric):
            pf_fric.append(drag_fric)
            pf_pres.append(drag_pres)
            me_link_fx.append(me_link_step)
            me_stream_fx.append(me_stream_step)

        # Report at specified steps
        if step in (200, 400, 600, 800, 1000) or step == n_steps:
            n_rec = len(pf_fric)
            if n_rec > 0:
                ct_me_link = sum(me_link_fx) / n_rec / dyn_p_S
                ct_me_stream = sum(me_stream_fx) / n_rec / dyn_p_S
                ct_pf_fric = sum(pf_fric) / n_rec / dyn_p_S
                ct_pf_pres = sum(pf_pres) / n_rec / dyn_p_S
                ct_pf_tot = ct_pf_fric + ct_pf_pres
            else:
                ct_me_link = me_link_step / dyn_p_S
                ct_me_stream = me_stream_step / dyn_p_S
                ct_pf_fric = drag_fric / dyn_p_S
                ct_pf_pres = drag_pres / dyn_p_S
                ct_pf_tot = ct_pf_fric + ct_pf_pres

            _, ux, uy, uz = macroscopic3d(f)
            ms = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max().item())

            print(
                f" {step:5d}  {ct_me_link:12.5f}  {ct_me_stream:14.5f}  {ct_pf_fric:12.5f}  {ct_pf_pres:12.5f}  {ct_pf_tot:12.5f}  {ms:10.4f}",
                flush=True,
            )

    # --- Final statistics ---
    n_total = len(me_link_fx)
    print(f"\n=== Final Statistics (post-warmup, n={n_total}) ===", flush=True)

    ct_me_link_arr = [v / dyn_p_S for v in me_link_fx]
    ct_me_stream_arr = [v / dyn_p_S for v in me_stream_fx]
    ct_pf_fric_arr = [v / dyn_p_S for v in pf_fric]
    ct_pf_pres_arr = [v / dyn_p_S for v in pf_pres]
    ct_pf_tot_arr = [f + p for f, p in zip(ct_pf_fric_arr, ct_pf_pres_arr)]

    mean_me_link = sum(ct_me_link_arr) / n_total
    mean_me_stream = sum(ct_me_stream_arr) / n_total
    mean_pf_fric = sum(ct_pf_fric_arr) / n_total
    mean_pf_pres = sum(ct_pf_pres_arr) / n_total
    mean_pf_tot = sum(ct_pf_tot_arr) / n_total

    std_me_link = _std(ct_me_link_arr)
    std_me_stream = _std(ct_me_stream_arr)
    std_pf_fric = _std(ct_pf_fric_arr)
    std_pf_pres = _std(ct_pf_pres_arr)
    std_pf_tot = _std(ct_pf_tot_arr)

    print(f"  Ct_ME_link:   mean={mean_me_link:+.6f}  std={std_me_link:.6f}", flush=True)
    print(f"  Ct_ME_stream: mean={mean_me_stream:+.6f}  std={std_me_stream:.6f}", flush=True)
    print(f"  Ct_pf_fric:   mean={mean_pf_fric:+.6f}  std={std_pf_fric:.6f}", flush=True)
    print(f"  Ct_pf_pres:   mean={mean_pf_pres:+.6f}  std={std_pf_pres:.6f}", flush=True)
    print(f"  Ct_pf_tot:    mean={mean_pf_tot:+.6f}  std={std_pf_tot:.6f}", flush=True)

    # Stability comparison — use the more stable ME variant
    best_me = "stream" if std_me_stream < std_me_link else "link"
    best_me_std = min(std_me_stream, std_me_link)
    std_ratio = best_me_std / (std_pf_tot + 1e-12)
    print(f"\n  Best ME variant: {best_me}", flush=True)
    print(f"  Std ratio (best ME / PF_tot): {std_ratio:.3f}", flush=True)
    if std_ratio < 0.9:
        verdict = "ME is MORE stable (lower variance) than pressure-face drag"
    elif std_ratio > 1.1:
        verdict = "ME is LESS stable (higher variance) than pressure-face drag"
    else:
        verdict = "ME and pressure-face drag have SIMILAR stability"
    print(f"  Verdict: {verdict}", flush=True)

    return {
        "mean_me_link": mean_me_link, "std_me_link": std_me_link,
        "mean_me_stream": mean_me_stream, "std_me_stream": std_me_stream,
        "mean_pf_fric": mean_pf_fric, "std_pf_fric": std_pf_fric,
        "mean_pf_pres": mean_pf_pres, "std_pf_pres": std_pf_pres,
        "mean_pf_tot": mean_pf_tot, "std_pf_tot": std_pf_tot,
        "verdict": verdict,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Momentum Exchange vs Pressure-Face drag on SUBOFF bare_hull"
    )
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    parser.add_argument("--re", type=float, default=2e6, help="Reynolds number")
    parser.add_argument("--nx", type=int, default=160, help="Grid size X")
    parser.add_argument("--ny", type=int, default=160, help="Grid size Y")
    parser.add_argument("--nz", type=int, default=160, help="Grid size Z")
    parser.add_argument("--n-steps", type=int, default=1000, help="Number of steps")
    parser.add_argument("--u-in", type=float, default=0.06, help="Inlet velocity")
    parser.add_argument("--cs", type=float, default=0.05, help="Smagorinsky constant")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup steps")
    args = parser.parse_args()

    run(
        re=args.re,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        u_in=args.u_in,
        cs=args.cs,
        n_steps=args.n_steps,
        warmup=args.warmup,
        device=args.device,
    )
