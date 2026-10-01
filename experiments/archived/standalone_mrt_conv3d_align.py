#!/usr/bin/env python3
"""Conv3D MRT alignment test: SDAA vs CPU, step-by-step precision comparison.

Same initial field, same collision+stream+BC, compare every step:
  - f distribution (max abs diff, mean abs diff)
  - macroscopic (rho, ux, uy, uz)
  - drag (friction + pressure)

Run:
    PYTHONPATH=src python standalone_mrt_conv3d_align.py --steps 50
"""

import argparse, math, time, torch
import numpy as np

U_IN = 0.06; RE = 2_000_000
NX, NY, NZ, HL = 200, 80, 80, 80.0
NU = U_IN * HL / RE; TAU = 3.0 * NU + 0.5; CS = 0.05


# ── helpers (shared, device-agnostic) ──────────────────────────────────

def build_solid(device):
    from tensorlbm.suboff_cad import build_suboff_mask, SuboffHullType
    cx, cy, cz = NX * 0.35, NY / 2.0, NZ / 2.0
    solid, _ = build_suboff_mask(
        SuboffHullType.BARE_HULL, NX, NY, NZ,
        cx=cx, cy=cy, cz=cz, length=HL, device=device,
    )
    return solid


def collide_mrt_conv3d(f, tau, C_s):
    from tensorlbm.d3q19 import macroscopic3d, equilibrium3d
    from tensorlbm.turbulence import _neq_stress_norm_3d, _smagorinsky_tau, _get_d3q19_mrt_matrices

    M, Mi = _get_d3q19_mrt_matrices(f.device)
    Wm = M.reshape(19, 19, 1, 1, 1)
    Wmi = Mi.reshape(19, 19, 1, 1, 1)

    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq
    pn = _neq_stress_norm_3d(f_neq)
    te = _smagorinsky_tau(tau, pn, rho, C_s)
    sn = 1.0 / te

    f4 = f.unsqueeze(0)
    feq4 = feq.unsqueeze(0)
    conv = torch.nn.functional.conv3d

    m = conv(f4, Wm)
    me = conv(feq4, Wm)
    dm = m - me

    sf = torch.tensor([0.,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                      device=f.device).reshape(1,19,1,1,1)
    ms = m - sf * dm
    sn4 = sn.reshape(1, 1, *sn.shape)
    for k in [9, 11, 13, 14, 15]:
        ms[:, k] = m[:, k] - sn4 * dm[:, k]

    return conv(ms, Wmi).squeeze(0)


def step(f, solid, nu):
    """One full LBM step: collide + stream + wall + farfield."""
    from tensorlbm.wall_model import wall_function_3d
    from tensorlbm.boundaries3d import far_field_bc_3d
    from tensorlbm.solver3d import stream3d

    f = collide_mrt_conv3d(f, TAU, CS)
    f = stream3d(f)
    f, df, dp = wall_function_3d(f, solid, nu)
    f = far_field_bc_3d(f, u_in=U_IN)
    return f, df, dp


def init_field(device):
    """Return (f0, solid) on the given device."""
    solid = build_solid(device)
    from tensorlbm.d3q19 import equilibrium3d
    nz, ny, nx = solid.shape
    r0 = torch.ones(nz, ny, nx, device=device)
    u0 = torch.full((nz, ny, nx), U_IN, device=device)
    u0[solid] = 0.0
    f = equilibrium3d(r0, u0, torch.zeros_like(u0), torch.zeros_like(u0))
    return f, solid


def macroscopic_from_f(f):
    """Return (rho, ux, uy, uz) on the same device."""
    from tensorlbm.d3q19 import macroscopic3d
    rho, ux, uy, uz = macroscopic3d(f)
    return rho, ux, uy, uz


# ── comparison helpers ────────────────────────────────────────────────

def compare_tensors(name, t_cpu, t_sdaa):
    """Return (max_abs, mean_abs, max_rel_pct, rel_pct_at_max)."""
    tc = t_cpu.float().cpu()
    ts = t_sdaa.float().cpu()
    diff = (ts - tc).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    denom = tc.abs()
    denom[denom < 1e-10] = 1.0
    rel = diff / denom * 100.0
    max_rel = rel.max().item()
    rel_at_max = rel.flatten()[diff.flatten().argmax()].item()

    return max_abs, mean_abs, max_rel, rel_at_max


def print_header():
    print(f"{'step':>5s}  {'max|df|':>10s}  {'mean|df|':>10s}  "
          f"{'max|dρ|':>10s}  {'max|dux|':>10s}  {'max|duy|':>10s}  "
          f"{'max|duz|':>10s}  {'drag_CPU':>12s}  {'drag_SDAA':>12s}  "
          f"{'Δdrag':>10s}")
    print("-" * 120)


def print_step(step, f_cpu, f_sdaa, drag_cpu, drag_sdaa):
    max_f, mean_f, _, _ = compare_tensors("f", f_cpu, f_sdaa)

    rho_c, ux_c, uy_c, uz_c = macroscopic_from_f(f_cpu)
    rho_s, ux_s, uy_s, uz_s = macroscopic_from_f(f_sdaa)

    max_rho, _, _, _ = compare_tensors("rho", rho_c, rho_s)
    max_ux, _, _, _ = compare_tensors("ux", ux_c, ux_s)
    max_uy, _, _, _ = compare_tensors("uy", uy_c, uy_s)
    max_uz, _, _, _ = compare_tensors("uz", uz_c, uz_s)

    ddrag = drag_sdaa - drag_cpu
    print(f"{step:5d}  {max_f:10.3e}  {mean_f:10.3e}  "
          f"{max_rho:10.3e}  {max_ux:10.3e}  {max_uy:10.3e}  "
          f"{max_uz:10.3e}  {drag_cpu:12.6f}  {drag_sdaa:12.6f}  "
          f"{ddrag:+.6f}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50,
                   help="number of steps to compare (default: 50)")
    p.add_argument("--print-every", type=int, default=1,
                   help="print every N steps (default: 1)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # ── initialise on CPU ──
    print("Initialising CPU field...")
    f_cpu, solid_cpu = init_field(torch.device("cpu"))
    nu = NU  # float

    # ── clone to SDAA ──
    sd = torch.device("sdaa:0")
    print("Copying to SDAA...")
    f_sdaa = f_cpu.clone().to(sd)
    solid_sdaa = solid_cpu.clone().to(sd)

    # ── print run info ──
    n_solid = int(solid_cpu.sum().item())
    print(f"Grid: {NX}×{NY}×{NZ}  solid cells: {n_solid}  steps: {args.steps}")
    print(f"Re={RE:.0f}  U_in={U_IN}  τ={TAU:.6f}  ν={NU:.3e}")
    print()

    print_header()

    t0 = time.time()
    for step_num in range(1, args.steps + 1):
        # CPU step
        f_cpu, df_cpu, dp_cpu = step(f_cpu, solid_cpu, nu)
        drag_cpu = (df_cpu + dp_cpu) if math.isfinite(df_cpu) else float("nan")

        # SDAA step
        f_sdaa, df_sdaa, dp_sdaa = step(f_sdaa, solid_sdaa, nu)
        drag_sdaa = (df_sdaa + dp_sdaa) if math.isfinite(df_sdaa) else float("nan")

        if step_num % args.print_every == 0:
            print_step(step_num, f_cpu, f_sdaa, drag_cpu, drag_sdaa)

    elapsed = time.time() - t0
    print()
    print(f"Wall time: {elapsed:.1f}s  ({elapsed/args.steps*1000:.1f} ms/step)")

    # ── final deep comparison ──
    print()
    print("═" * 60)
    print("  FINAL-STEP DEEP COMPARISON")
    print("═" * 60)
    _print_deep_compare(f_cpu, f_sdaa, solid_cpu)


def _print_deep_compare(f_cpu, f_sdaa, solid):
    """Print detailed per-channel and per-region breakdown."""
    rho_c, ux_c, uy_c, uz_c = macroscopic_from_f(f_cpu)
    rho_s, ux_s, uy_s, uz_s = macroscopic_from_f(f_sdaa)

    print(f"\n  {'Channel':>8s}  {'max|df|':>10s}  {'mean|df|':>10s}  {'max_rel%':>10s}")
    print("  " + "-" * 45)
    for k in range(19):
        max_a, mean_a, max_r, _ = compare_tensors(
            f"f[{k}]", f_cpu[k].unsqueeze(0), f_sdaa[k].unsqueeze(0),
        )
        print(f"  f[{k:>2d}]     {max_a:10.3e}  {mean_a:10.3e}  {max_r:10.3f}")

    print(f"\n  {'Macro':>8s}  {'max|d|':>10s}  {'mean|d|':>10s}  {'max_rel%':>10s}")
    print("  " + "-" * 45)
    for name, tc, ts in [("rho", rho_c, rho_s), ("ux", ux_c, ux_s),
                          ("uy", uy_c, uy_s), ("uz", uz_c, uz_s)]:
        max_a, mean_a, max_r, _ = compare_tensors(name, tc, ts)
        print(f"  {name:>8s}  {max_a:10.3e}  {mean_a:10.3e}  {max_r:10.3f}")

    # Per-region breakdown: fluid vs solid vs boundary
    solid_mask = solid.cpu().bool()
    fluid_mask = ~solid_mask

    print(f"\n  Region breakdown (final step):")
    for region, mask, label in [
        (fluid_mask, fluid_mask, "fluid"),
        (solid_mask, solid_mask, "solid"),
    ]:
        diff = (f_sdaa.cpu() - f_cpu.cpu()).abs()
        df_region = diff[:, mask].mean().item()
        print(f"    {label:>6s}: mean|f_sdaa - f_cpu| = {df_region:.3e}")


if __name__ == "__main__":
    main()
