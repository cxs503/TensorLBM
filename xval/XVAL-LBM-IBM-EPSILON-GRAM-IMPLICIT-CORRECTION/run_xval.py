"""XVAL runner: LIFE epsilon-Gram implicit correction vs uncorrected IBM.

Primary arm (A): force-transfer round-trip + total-force conservation.
  A1 = LIFE 3-point delta + epsilon-Gram correction (transplant)
  A2 = LIFE 3-point delta, epsilon = 1 (uncorrected control)
Secondary arm (B): 2D fixed cylinder in uniform flow, force-based IBM,
  time-averaged drag coefficient vs literature.

Writes results.json + figures into this directory.  No threshold is tuned
after seeing results (see predeclared.json).
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorlbm.ibm_epsilon_gram import (  # noqa: E402
    build_spread_matrix_2d,
    build_support_2d,
    compute_ds_2d,
    compute_epsilon_gram,
    interpolate_force_2d,
    interpolate_force_2d_vec,
    spread_force_2d,
    spread_force_2d_vec,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Part A: round-trip + total-force conservation
# ---------------------------------------------------------------------------


def circle_markers(n: int, cx: float, cy: float, r: float) -> tuple[torch.Tensor, torch.Tensor]:
    th = torch.arange(n, dtype=DTYPE) * (2.0 * math.pi / n)
    return cx + r * torch.cos(th), cy + r * torch.sin(th)


def run_roundtrip(n_markers: int, force_field: str, arm: str,
                  ny: int = 64, nx: int = 64, r: float = 10.0,
                  cx: float = 32.0, cy: float = 32.0) -> dict:
    mx, my = circle_markers(n_markers, cx, cy, r)
    supports = build_support_2d(mx, my, ny, nx)
    ds = compute_ds_2d(mx, my)

    # target marker force field (force per unit length)
    th = torch.arctan2(my - cy, mx - cx)
    if force_field == "constant":
        fx_t = torch.ones(n_markers, dtype=DTYPE)
        fy_t = torch.zeros(n_markers, dtype=DTYPE)
    else:  # radial
        fx_t = torch.cos(th)
        fy_t = torch.sin(th)

    if arm == "A1":
        eps = compute_epsilon_gram(mx, my, supports, ds)
    else:
        eps = torch.ones(n_markers, dtype=DTYPE)

    fx_grid, fy_grid = spread_force_2d(fx_t, fy_t, supports, eps, ds, ny, nx)
    fx_rec, fy_rec = interpolate_force_2d(fx_grid, fy_grid, supports)

    tgt = torch.stack([fx_t, fy_t], dim=1)
    rec = torch.stack([fx_rec, fy_rec], dim=1)
    l2 = float((rec - tgt).norm().item() / tgt.norm().item())
    mx_abs = float((rec - tgt).abs().max().item() / tgt.abs().max().item())

    # total deposited force vs target total force (sum f_m ds_m)
    tgt_tot = (torch.stack([fx_t, fy_t], dim=1) * ds.unsqueeze(1)).sum(dim=0)
    got_tot = torch.stack([fx_grid.sum(), fy_grid.sum()])
    cons = (got_tot - tgt_tot).abs() / tgt_tot.abs().clamp(min=1e-30)
    # radial field has zero net target force: report absolute residual instead
    cons_abs = got_tot.abs()

    return {
        "n_markers": n_markers,
        "force_field": force_field,
        "arm": arm,
        "roundtrip_l2_rel": l2,
        "roundtrip_maxabs_rel": mx_abs,
        "total_force_cons_rel_x": float(cons[0].item()),
        "total_force_cons_rel_y": float(cons[1].item()),
        "total_force_abs_x": float(cons_abs[0].item()),
        "total_force_abs_y": float(cons_abs[1].item()),
        "epsilon_min": float(eps.min().item()),
        "epsilon_max": float(eps.max().item()),
        "epsilon_mean": float(eps.mean().item()),
    }


# ---------------------------------------------------------------------------
# Part B: 2D fixed cylinder drag (force-based IBM)
# ---------------------------------------------------------------------------


def cylinder_drag(n_steps: int, arm: str, nx: int = 128, ny: int = 64,
                   diameter: float = 8.0, u_target: float = 0.1,
                   tau: float = 0.524, n_transient: int = 500) -> dict:
    from tensorlbm.d2q9 import equilibrium, macroscopic
    from tensorlbm.solver import collide_bgk, stream
    from tensorlbm.ibm_force_transfer import (
        build_cylinder_markers_2d,
        body_source_2d,
        interior_mask_force,
    )
    from tensorlbm.ibm_validation_cases import disc_mask_2d

    nu = (tau - 0.5) / 3.0
    re = u_target * diameter / nu
    c = (0.35 * nx, 0.5 * ny)
    markers = build_cylinder_markers_2d(c, 0.5 * diameter, spacing=0.8)
    mx = markers[:, 0]
    my = markers[:, 1]
    supports = build_support_2d(mx, my, ny, nx)
    ds = compute_ds_2d(mx, my)
    if arm == "A1":
        eps = compute_epsilon_gram(mx, my, supports, ds)
    else:
        eps = torch.ones(len(supports), dtype=DTYPE)

    mask = disc_mask_2d((ny, nx), c, max(0.5 * diameter - 0.5, 0.5))
    S, _ = build_spread_matrix_2d(mx, my, ny, nx)
    rho = torch.ones((ny, nx), dtype=DTYPE)
    ux = u_target * torch.ones_like(rho)
    uy = torch.zeros_like(rho)
    f = equilibrium(rho, ux, uy)

    drag_hist: list[float] = []
    for step in range(n_steps):
        rho_f, uxf, uyf = macroscopic(f)
        u_grid = torch.stack([uxf, uyf])
        # interpolate fluid velocity to markers (LIFE 3-point delta, vectorised)
        u_int = torch.stack([S @ uxf.reshape(-1), S @ uyf.reshape(-1)], dim=1)
        # direct forcing: F = (u_fluid - u_body)/dt, u_body = 0, dt = 1,
        # with the two Guo corrector passes (required for stability, as in the
        # in-repo direct_forcing)
        f_m = u_int.clone()
        rho_m = S @ rho_f.reshape(-1)
        for _ in range(2):
            u_phys = u_int + 0.5 * f_m / rho_m.unsqueeze(1)
            f_m = f_m + (torch.zeros_like(u_int) - u_phys)
        fx_grid, fy_grid = spread_force_2d_vec(f_m[:, 0], f_m[:, 1], S, eps, ds, ny, nx)
        f_int = interior_mask_force(u_grid, mask, torch.zeros_like(u_grid))
        fx = fx_grid + f_int[0]
        fy = fy_grid + f_int[1]
        f = collide_bgk(f, tau)
        f = body_source_2d(f, fx, fy)
        f = stream(f)
        if step >= n_transient:
            drag_hist.append(float(-(fx_grid + f_int[0]).sum().item()))

    cd_inst = [d / (0.5 * 1.0 * u_target ** 2 * diameter) for d in drag_hist]
    cd_mean = sum(cd_inst) / len(cd_inst)
    return {"re": re, "cd_mean": cd_mean, "n_avg": len(cd_inst), "arm": arm}


def selftest_vectorisation() -> dict:
    """Prove the vectorised ops are semantics-identical to the loop form."""
    torch.manual_seed(7)
    n = 9
    th = torch.arange(n, dtype=DTYPE) * (2.0 * math.pi / n)
    mx = 20.0 + 6.0 * torch.cos(th)
    my = 32.0 + 6.0 * torch.sin(th)
    ny, nx = 64, 64
    supports = build_support_2d(mx, my, ny, nx)
    ds = compute_ds_2d(mx, my)
    eps = compute_epsilon_gram(mx, my, supports, ds)
    fx = torch.randn(n, dtype=DTYPE)
    fy = torch.randn(n, dtype=DTYPE)
    S, _ = build_spread_matrix_2d(mx, my, ny, nx)
    fxg1, fyg1 = spread_force_2d(fx, fy, supports, eps, ds, ny, nx)
    fxg2, fyg2 = spread_force_2d_vec(fx, fy, S, eps, ds, ny, nx)
    fxr1, fyr1 = interpolate_force_2d(fxg1, fyg1, supports)
    fxr2, fyr2 = interpolate_force_2d_vec(fxg1, fyg1, S)
    return {
        "spread_maxdiff": float((fxg1 - fxg2).abs().max().item()),
        "interp_maxdiff": float((fxr1 - fxr2).abs().max().item()),
        "pass": bool((fxg1 - fxg2).abs().max().item() < 1e-14
                     and (fxr1 - fxr2).abs().max().item() < 1e-14),
    }


def main() -> None:
    out: dict = {"selftest": selftest_vectorisation(), "arms": {}, "physics": {}}

    # Part A: round-trip / conservation
    for n in (16, 32):
        for ff in ("constant", "radial"):
            for arm in ("A1", "A2"):
                key = f"N{n}_{ff}_{arm}"
                out["arms"][key] = run_roundtrip(n, ff, arm)
                print(f"[A] {key}: rt_l2={out['arms'][key]['roundtrip_l2_rel']:.3e} "
                      f"cons_x={out['arms'][key]['total_force_cons_rel_x']:.3e} "
                      f"eps=[{out['arms'][key]['epsilon_min']:.4f},"
                      f"{out['arms'][key]['epsilon_max']:.4f}]")

    # Part B: cylinder drag
    for arm in ("A1", "A2"):
        r = cylinder_drag(n_steps=700, arm=arm)
        out["physics"][arm] = r
        print(f"[B] {arm}: Re={r['re']:.1f} Cd={r['cd_mean']:.4f} (n={r['n_avg']})")

    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote results.json")
    print("selftest:", out["selftest"])


if __name__ == "__main__":
    main()
