"""Single SDAA worker for collision operator comparison (D3Q19/D3Q27).

Handles all collision operators from one unified script.
Usage:
    python _collision_comparison_worker.py <did> <lattice> <collision> <nx> <ny> <nz> <hull_length> <n_steps> <Cs>
"""

import json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import torch

KAPPA = 0.41
B_CONST = 5.0
REF_CT = 0.00405


# ==============================================================================
# D3Q19 helpers
# ==============================================================================
def _setup_d3q19():
    from tensorlbm.d3q19 import C as C19, equilibrium3d, macroscopic3d
    from tensorlbm.solver3d import correct_mass3d, stream3d
    from tensorlbm.boundaries3d import far_field_bc_3d
    return C19, equilibrium3d, macroscopic3d, correct_mass3d, stream3d, far_field_bc_3d


def wallfn19(f, solid, nu, y_val=0.5):
    """Log-law wall function for D3Q19 — Guo body force."""
    from tensorlbm.d3q19 import macroscopic3d, C as C19

    fluid = ~solid
    near = torch.zeros_like(solid)
    for ax, sgn in [(2, 1), (2, -1), (1, 1), (1, -1), (0, 1), (0, -1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid

    rho, ux, uy, uz = macroscopic3d(f)
    um = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
    ut = torch.sqrt(nu * um / y_val).clamp(min=1e-12)
    yp = y_val * ut / nu
    turb = (yp > 11.6) & near
    if turb.any():
        uu = ut[turb].clone()
        vm = um[turb]
        for _ in range(8):
            ly = torch.log(y_val * uu / nu)
            fv = uu * (ly / KAPPA + B_CONST) - vm
            fp = (ly / KAPPA + B_CONST) + 1.0 / KAPPA
            uu = (uu - fv / fp.clamp(min=1e-10)).clamp(min=1e-12)
        ut[turb] = uu
    tw = ut * ut
    ium = 1.0 / um
    coef = -(tw / y_val) * near.to(f.dtype)
    fx = coef * (ux * ium)
    fy = coef * (uy * ium)
    fz = coef * (uz * ium)

    # D3Q19 Guo body force
    device = f.device
    c19 = C19.to(device).float()
    cx19 = c19[:, 0].view(19, 1, 1, 1)
    cy19 = c19[:, 1].view(19, 1, 1, 1)
    cz19 = c19[:, 2].view(19, 1, 1, 1)
    w19 = torch.tensor(
        [1 / 3] + [1 / 18] * 6 + [1 / 36] * 12,
        dtype=f.dtype, device=device,
    ).view(19, 1, 1, 1)
    cs2 = 1.0 / 3.0
    cu = cx19 * ux + cy19 * uy + cz19 * uz
    forcing = w19 * (1.0 + cu / cs2) * (cx19 * fx + cy19 * fy + cz19 * fz) / cs2
    f = f + forcing
    df = (tw * (ux * ium) * near.to(f.dtype)).sum().item()
    p = (rho - 1.0) / 3.0
    sp = torch.roll(solid, 1, dims=2)
    sm = torch.roll(solid, -1, dims=2)
    dp = (p * (sp.to(f.dtype) - sm.to(f.dtype)) * fluid.to(f.dtype)).sum().item()
    return f, df, dp


# ==============================================================================
# D3Q27 helpers
# ==============================================================================
_D3Q27_SHIFTS = [
    (cx, cy, cz) for cz in [-1, 0, 1] for cy in [-1, 0, 1] for cx in [-1, 0, 1]
]


def stream27_roll(f):
    out = torch.empty_like(f)
    for q in range(27):
        sx, sy, sz = _D3Q27_SHIFTS[q]
        out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


def far_field_bc_27(f, u_in=0.06):
    from tensorlbm.d3q27 import equilibrium27
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    rho1 = torch.ones(nz, ny, nx, dtype=f.dtype, device=f.device)
    feq = equilibrium27(
        rho1,
        torch.full_like(rho1, u_in),
        torch.zeros_like(rho1),
        torch.zeros_like(rho1),
    )
    f = f.clone()
    f[:, :, :, 0] = feq[:, :, :, 0]
    f[:, :, :, -1] = f[:, :, :, -2]
    f[:, 0, :, :] = feq[:, 0, :, :]
    f[:, -1, :, :] = feq[:, -1, :, :]
    f[:, :, 0, :] = feq[:, :, 0, :]
    f[:, :, -1, :] = feq[:, :, -1, :]
    return f


def wallfn27(f, solid, nu, y_val=0.5):
    """Log-law wall function for D3Q27 — Guo body force."""
    from tensorlbm.d3q27 import macroscopic27, C as C27

    device = f.device
    c = C27.to(device).float()
    cx = c[:, 0].view(27, 1, 1, 1)
    cy = c[:, 1].view(27, 1, 1, 1)
    cz = c[:, 2].view(27, 1, 1, 1)

    fluid = ~solid
    near = torch.zeros_like(solid)
    for ax, sgn in [(2, 1), (2, -1), (1, 1), (1, -1), (0, 1), (0, -1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid

    rho, ux, uy, uz = macroscopic27(f)
    um = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
    ut = torch.sqrt(nu * um / y_val).clamp(min=1e-12)
    yp = y_val * ut / nu
    turb = (yp > 11.6) & near
    if turb.any():
        uu = ut[turb].clone()
        vm = um[turb]
        for _ in range(8):
            ly = torch.log(y_val * uu / nu)
            fv = uu * (ly / KAPPA + B_CONST) - vm
            fp = (ly / KAPPA + B_CONST) + 1.0 / KAPPA
            uu = (uu - fv / fp.clamp(min=1e-10)).clamp(min=1e-12)
        ut[turb] = uu
    tw = ut * ut
    ium = 1.0 / um
    coef = -(tw / y_val) * near.to(f.dtype)
    fx = coef * (ux * ium)
    fy = coef * (uy * ium)
    fz = coef * (uz * ium)

    # D3Q27 Guo body force
    w27 = torch.tensor(
        [8 / 27] + [2 / 27] * 6 + [1 / 54] * 12 + [1 / 216] * 8,
        dtype=f.dtype, device=device,
    ).view(27, 1, 1, 1)
    cs2 = 1.0 / 3.0
    cu = cx * ux + cy * uy + cz * uz
    forcing = w27 * (1.0 + cu / cs2) * (cx * fx + cy * fy + cz * fz) / cs2
    f = f + forcing
    df = (tw * (ux * ium) * near.to(f.dtype)).sum().item()
    p = (rho - 1.0) / 3.0
    sp = torch.roll(solid, 1, dims=2)
    sm = torch.roll(solid, -1, dims=2)
    dp = (p * (sp.to(f.dtype) - sm.to(f.dtype)) * fluid.to(f.dtype)).sum().item()
    return f, df, dp


# ==============================================================================
# Smagorinsky wrappers for operators without built-in Smag
# ==============================================================================
def _smag_tau_d3q19(f, tau, C_s):
    """Compute domain-averaged Smagorinsky tau_eff for D3Q19."""
    from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
    from tensorlbm.turbulence import _neq_stress_norm_3d, _smagorinsky_tau

    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq
    pi_norm = _neq_stress_norm_3d(f_neq)
    tau_eff_per_cell = _smagorinsky_tau(tau, pi_norm, rho, C_s)
    tau_eff = float(tau_eff_per_cell.mean().item())
    return max(tau, min(tau_eff, tau * 10.0))


def _smag_tau_d3q27(f, tau, C_s):
    """Compute domain-averaged Smagorinsky tau_eff for D3Q27."""
    from tensorlbm.d3q27 import equilibrium27, macroscopic27
    from tensorlbm.turbulence import _neq_stress_norm_27, _smagorinsky_tau

    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq
    pi_norm = _neq_stress_norm_27(f_neq)
    tau_eff_per_cell = _smagorinsky_tau(tau, pi_norm, rho, C_s)
    tau_eff = float(tau_eff_per_cell.mean().item())
    return max(tau, min(tau_eff, tau * 10.0))


# ==============================================================================
# Collision dispatchers
# ==============================================================================
def collide_d3q19(name, f, tau, Cs):
    """Dispatch to D3Q19 collision operator by name."""
    if name == "MRT+Smag":
        from tensorlbm.turbulence import collide_smagorinsky_mrt3d
        return collide_smagorinsky_mrt3d(f, tau=tau, C_s=Cs)
    elif name == "CUMULANT":
        from tensorlbm.cumulant import collide_cumulant_d3q19
        return collide_cumulant_d3q19(f, tau, C_s=Cs)
    elif name == "CASCADED":
        from tensorlbm.cascaded_collision import collide_cascaded_d3q19
        if Cs > 0:
            tau_eff = _smag_tau_d3q19(f, tau, Cs)
            return collide_cascaded_d3q19(f, tau_eff)
        else:
            return collide_cascaded_d3q19(f, tau)
    else:
        raise ValueError(f"Unknown D3Q19 collision: {name}")


def collide_d3q27(name, f, tau, Cs):
    """Dispatch to D3Q27 collision operator by name."""
    if name == "CUMULANT":
        if Cs > 0:
            from tensorlbm.cumulant_smag import collide_cumulant_smag_d3q27
            return collide_cumulant_smag_d3q27(f, tau, C_s=Cs)
        else:
            from tensorlbm.cumulant import collide_cumulant_d3q27
            return collide_cumulant_d3q27(f, tau)
    elif name == "CASCADED":
        from tensorlbm.cascaded_collision import collide_cascaded_d3q27
        if Cs > 0:
            tau_eff = _smag_tau_d3q27(f, tau, Cs)
            return collide_cascaded_d3q27(f, tau_eff)
        else:
            return collide_cascaded_d3q27(f, tau)
    else:
        raise ValueError(f"Unknown D3Q27 collision: {name}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    did = int(sys.argv[1])
    lattice = sys.argv[2]       # "D3Q19" or "D3Q27"
    collision = sys.argv[3]     # "MRT+Smag", "CUMULANT", "CASCADED"
    nx = int(sys.argv[4])
    ny = int(sys.argv[5])
    nz = int(sys.argv[6])
    hull_length = float(sys.argv[7])
    n_steps = int(sys.argv[8])
    Cs = float(sys.argv[9])
    hull_type_str = sys.argv[10] if len(sys.argv) > 10 else "bare_hull"

    u_in, re = 0.06, 2e6
    nu = u_in * hull_length / re
    tau = 3.0 * nu + 0.5
    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    lt_short = "Q19" if lattice == "D3Q19" else "Q27"
    smag_tag = f" Cs={Cs}" if Cs > 0 else ""
    tag = f"[SDAA:{did}] {lt_short} {collision}{smag_tag} {nx}³"
    print(f"{tag} tau={tau:.6f} nu={nu:.2e}", flush=True)

    # Build mask
    from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
    from tensorlbm.suboff_resistance import _voxel_wetted_area

    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    try:
        ht = SuboffHullType(hull_type_str)
    except ValueError:
        ht = SuboffHullType.BARE_HULL
        print(f"{tag} WARNING: unknown hull type '{hull_type_str}', using BARE_HULL", flush=True)

    solid, _ = build_suboff_mask(
        hull_type=ht, nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz, length=hull_length, device=device,
    )
    S = _voxel_wetted_area(solid, 1.0)
    dpS = 0.5 * 1.0 * u_in ** 2 * S
    print(f"{tag} S={S:.0f} init done", flush=True)

    # Initialize distributions
    if lattice == "D3Q19":
        from tensorlbm.d3q19 import equilibrium3d
        from tensorlbm.solver3d import correct_mass3d, stream3d
        from tensorlbm.boundaries3d import far_field_bc_3d
        rho0 = torch.ones(nz, ny, nx, device=device)
        ux0 = torch.full((nz, ny, nx), u_in, device=device)
        ux0[solid] = 0
        f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
        im = float(rho0.sum().item())
        stream_fn = stream3d
        wall_fn = wallfn19
        ff_bc = far_field_bc_3d
        correct_mass = correct_mass3d
        collide_fn = collide_d3q19
    else:  # D3Q27
        from tensorlbm.d3q27 import equilibrium27, correct_mass27
        rho0 = torch.ones(nz, ny, nx, device=device)
        ux0 = torch.full((nz, ny, nx), u_in, device=device)
        ux0[solid] = 0
        f = equilibrium27(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
        im = float(rho0.sum().item())
        stream_fn = stream27_roll
        wall_fn = wallfn27
        ff_bc = far_field_bc_27
        correct_mass = correct_mass27
        collide_fn = collide_d3q27

    warmup = n_steps // 3
    slide = max(1, n_steps // 6)  # sliding window
    fric, pres = [], []
    t0 = time.time()
    final_step = 0
    all_finite = True

    for step in range(1, n_steps + 1):
        try:
            f = collide_fn(collision, f, tau, Cs)
        except Exception as e:
            print(f"{tag} COLLISION ERROR at {step}: {e}", flush=True)
            break

        if lattice == "D3Q19":
            f = stream_fn(f)
        else:
            f = stream_fn(f)

        f, df, dp = wall_fn(f, solid, nu, y_val=0.5)
        f = ff_bc(f, u_in=u_in)
        if step % 100 == 0:
            f = correct_mass(f, im)
        if step > warmup and math.isfinite(df):
            fric.append(df)
            pres.append(dp)
        final_step = step

        if not torch.isfinite(f).all():
            print(f"{tag} DIV at {step}", flush=True)
            all_finite = False
            break

        if step % 1000 == 0 or step == n_steps:
            # Sliding window Ct (last `slide` samples)
            wf = fric[-slide:] if len(fric) >= slide else fric
            wp = pres[-slide:] if len(pres) >= slide else pres
            cf = sum(wf) / max(len(wf), 1) / dpS if wf else 0
            cp = sum(wp) / max(len(wp), 1) / dpS if wp else 0
            ct = cf + cp
            elap = time.time() - t0
            print(f"{tag} step={step} Ct={ct:.5f} ({elap:.0f}s)", flush=True)

    # Final results (full post-warmup average)
    cf = sum(fric) / max(len(fric), 1) / dpS if fric else 0
    cp = sum(pres) / max(len(pres), 1) / dpS if pres else 0
    ct = cf + cp

    elapsed = time.time() - t0

    result = {
        "lattice": lattice,
        "collision": collision,
        "Cs": Cs,
        "hull": hull_type_str,
        "grid": f"{nx}x{ny}x{nz}",
        "tau": tau,
        "nu": nu,
        "Ct_fric": cf,
        "Ct_pres": cp,
        "Ct_total": ct,
        "error_pct": abs(ct - REF_CT) / REF_CT * 100,
        "steps": final_step,
        "n_steps": n_steps,
        "finite": all_finite,
        "elapsed_s": elapsed,
        "did": did,
    }
    print(f"{tag} DONE Ct={ct:.5f} err={result['error_pct']:.1f}% ({elapsed:.0f}s)", flush=True)

    out_dir = Path("/tmp/collision_comparison_long")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"result_{did:02d}.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
