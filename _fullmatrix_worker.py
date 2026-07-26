"""Agent FullMatrix worker — unified 16-config worker (4 collisions × 4 benchmarks).

Maps device ID 0-15 to one of 16 collision×benchmark combos on 16 SDAA cards.
1500 steps each, log-law wall function, farfield BC, sign-fixed.

Collisions (C1-C4):
  C1. D3Q19 MRT+Smag Cs=0.05
  C2. D3Q27 CASCADED
  C3. D3Q27 CUMULANT
  C4. D3Q27 KBC+Smag Cs=0.05

Benchmarks (B1-B4):
  B1. SUBOFF 200×80×80 Re=2e6
  B2. Cylinder 200×80×4 Re=200
  B3. Sphere 200×100×100 Re=1000
  B4. Square prism 200×80×4 Re=22000

Mapping: did=0..15 → combo = (collision_index * 4 + benchmark_index)
  did 0-3   = C1 × [B1,B2,B3,B4]
  did 4-7   = C2 × [B1,B2,B3,B4]
  did 8-11  = C3 × [B1,B2,B3,B4]
  did 12-15 = C4 × [B1,B2,B3,B4]

Usage:
    python _fullmatrix_worker.py <did>
"""
from __future__ import annotations

import json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import torch

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
KAPPA = 0.41
B_CONST = 5.0
SLIDE = 300
N_STEPS = 1500

# Reference values for error computation
REF_CT_SUBOFF = 0.00405
REF_CD_CYLINDER = 1.30
REF_CD_SPHERE = 0.47
REF_CD_SQUARE = 2.05   # Square cylinder at Re=22000 (approximate)

# Collision labels
COLLISIONS = [
    {"name": "MRT+Smag",   "lattice": "D3Q19", "label": "C1_D3Q19_MRT_Smag",    "cs": 0.05},
    {"name": "CASCADED",   "lattice": "D3Q27", "label": "C2_D3Q27_CASCADED",    "cs": 0.0},
    {"name": "CUMULANT",   "lattice": "D3Q27", "label": "C3_D3Q27_CUMULANT",    "cs": 0.0},
    {"name": "KBC+Smag",   "lattice": "D3Q27", "label": "C4_D3Q27_KBC_Smag",    "cs": 0.05},
]

# Benchmark configs: (name, nx, ny, nz, u_in, Re, geom_param, geom_param_label, ref)
BENCHMARKS = [
    {"name": "suboff",     "nx": 200, "ny": 48, "nz": 40,  "u_in": 0.06, "Re": 2e6,   "geom": 80.0,  "geom_label": "hull_length", "ref": REF_CT_SUBOFF},
    {"name": "cylinder",   "nx": 200, "ny": 80, "nz": 4,   "u_in": 0.08, "Re": 200.0, "geom": 24.0,  "geom_label": "D",           "ref": REF_CD_CYLINDER},
    {"name": "sphere",     "nx": 120, "ny": 60, "nz": 60,  "u_in": 0.08, "Re": 1000.0,"geom": 24.0,  "geom_label": "D",           "ref": REF_CD_SPHERE},
    {"name": "square",     "nx": 200, "ny": 80, "nz": 4,   "u_in": 0.08, "Re": 22000.0,"geom": 16.0, "geom_label": "side",        "ref": REF_CD_SQUARE},
]


# ──────────────────────────────────────────────────────────────────
# Geometry mask builders
# ──────────────────────────────────────────────────────────────────
def _build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    return circle.unsqueeze(0).expand(nz, ny, nx).clone()


def _build_sphere_mask(nx, ny, nz, cx, cy, cz, radius, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius ** 2


def _build_square_mask(nx, ny, nz, cx, cy, side, device):
    """Square prism extruded along z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    half = side / 2.0
    square = (xx >= cx - half) & (xx <= cx + half) & (yy >= cy - half) & (yy <= cy + half)
    return square.unsqueeze(0).expand(nz, ny, nx).clone()


# ──────────────────────────────────────────────────────────────────
# Smagorinsky helpers
# ──────────────────────────────────────────────────────────────────
def _smag_tau_d3q19(f, tau, C_s):
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
    from tensorlbm.d3q27 import equilibrium27, macroscopic27
    from tensorlbm.turbulence import _neq_stress_norm_27, _smagorinsky_tau
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq
    pi_norm = _neq_stress_norm_27(f_neq)
    tau_eff_per_cell = _smagorinsky_tau(tau, pi_norm, rho, C_s)
    tau_eff = float(tau_eff_per_cell.mean().item())
    return max(tau, min(tau_eff, tau * 10.0))


# ──────────────────────────────────────────────────────────────────
# D3Q19 collision + setup
# ──────────────────────────────────────────────────────────────────
def _setup_d3q19(f, solid, u_in, nz, ny, nx, nu, device, collision_name, Cs):
    from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
    from tensorlbm.solver3d import correct_mass3d, stream3d
    from tensorlbm.boundaries3d import far_field_bc_3d
    from tensorlbm.wall_model import wall_function_3d

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())

    def collide_fn(f, tau):
        if collision_name == "MRT+Smag":
            from tensorlbm.turbulence import collide_smagorinsky_mrt3d
            return collide_smagorinsky_mrt3d(f, tau=tau, C_s=Cs)
        elif collision_name == "CASCADED":
            from tensorlbm.cascaded_collision import collide_cascaded_d3q19
            if Cs > 0:
                tau_eff = _smag_tau_d3q19(f, tau, Cs)
                return collide_cascaded_d3q19(f, tau_eff)
            else:
                return collide_cascaded_d3q19(f, tau)
        else:
            raise ValueError(f"Unknown D3Q19 collision: {collision_name}")

    def step_fn(f, solid, nu):
        f, df, dp = wall_function_3d(f, solid, nu, y_val=0.5)
        return f, df, dp

    return f, im, stream3d, far_field_bc_3d, correct_mass3d, collide_fn, step_fn


def _setup_d3q27(f_init, solid, u_in, nz, ny, nx, nu, device, collision_name, Cs):
    from tensorlbm.d3q27 import equilibrium27, macroscopic27, correct_mass27, stream27
    from tensorlbm.boundaries_d3q27 import far_field_bc_27
    from tensorlbm.wall_model import wall_function_d3q27

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium27(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    im = float(rho0.sum().item())

    def collide_fn(f, tau):
        if collision_name == "CUMULANT":
            if Cs > 0:
                from tensorlbm.cumulant_smag import collide_cumulant_smag_d3q27
                return collide_cumulant_smag_d3q27(f, tau, C_s=Cs)
            else:
                from tensorlbm.cumulant import collide_cumulant_d3q27
                return collide_cumulant_d3q27(f, tau)
        elif collision_name == "CASCADED":
            from tensorlbm.cascaded_collision import collide_cascaded_d3q27
            if Cs > 0:
                tau_eff = _smag_tau_d3q27(f, tau, Cs)
                return collide_cascaded_d3q27(f, tau_eff)
            else:
                return collide_cascaded_d3q27(f, tau)
        elif collision_name == "KBC+Smag":
            from tensorlbm.advanced_collision import collide_kbc_d3q27
            if Cs > 0:
                tau_eff = _smag_tau_d3q27(f, tau, Cs)
                return collide_kbc_d3q27(f, tau_eff, C_s=0.0, beta=0.99)
            else:
                return collide_kbc_d3q27(f, tau, C_s=0.0, beta=0.99)
        elif collision_name == "MRT+Smag":
            # D3Q27 MRT via KBC with beta=0.5
            from tensorlbm.advanced_collision import collide_kbc_d3q27
            if Cs > 0:
                tau_eff = _smag_tau_d3q27(f, tau, Cs)
                return collide_kbc_d3q27(f, tau_eff, C_s=0.0, beta=0.5)
            else:
                return collide_kbc_d3q27(f, tau, C_s=0.0, beta=0.5)
        else:
            raise ValueError(f"Unknown D3Q27 collision: {collision_name}")

    def step_fn(f, solid, nu):
        f, df, dp = wall_function_d3q27(f, solid, nu, y_val=0.5)
        return f, df, dp

    return f, im, stream27, far_field_bc_27, correct_mass27, collide_fn, step_fn


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    did = int(sys.argv[1])
    ci = did // 4   # collision index 0-3
    bi = did % 4    # benchmark index 0-3

    coll = COLLISIONS[ci]
    bench = BENCHMARKS[bi]

    lattice = coll["lattice"]
    collision_name = coll["name"]
    Cs = coll["cs"]
    coll_label = coll["label"]

    flow = bench["name"]
    nx, ny, nz = bench["nx"], bench["ny"], bench["nz"]
    u_in = bench["u_in"]
    Re = bench["Re"]
    geom_param = bench["geom"]
    ref_value = bench["ref"]
    dp_label = "Ct" if flow == "suboff" else "Cd"

    nu = u_in * geom_param / Re
    tau = 3.0 * nu + 0.5

    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{did}] {coll_label} × {flow} {nx}×{ny}×{nz} Re={Re:.0f}"
    print(f"{tag} u_in={u_in} nu={nu:.3e} tau={tau:.6f} Cs={Cs}",
          flush=True)

    t0 = time.time()

    # ── Build geometry ──────────────────────────────────────────
    if flow == "suboff":
        from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
        from tensorlbm.suboff_resistance import _voxel_wetted_area
        cx_g = nx * 0.35
        cy_g = ny / 2.0
        cz_g = nz / 2.0
        solid, _ = build_suboff_mask(
            hull_type=SuboffHullType.BARE_HULL,
            nx=nx, ny=ny, nz=nz,
            cx=cx_g, cy=cy_g, cz=cz_g,
            length=geom_param, device=str(device),
        )
        A_ref = _voxel_wetted_area(solid, 1.0)
    elif flow == "cylinder":
        radius = geom_param / 2.0
        cx_g = nx * 0.25; cy_g = ny * 0.5
        solid = _build_cylinder_mask(nx, ny, nz, cx_g, cy_g, radius, device)
        A_ref = geom_param * nz
    elif flow == "sphere":
        radius = geom_param / 2.0
        cx_g = nx * 0.25; cy_g = ny * 0.5; cz_g = nz * 0.5
        solid = _build_sphere_mask(nx, ny, nz, cx_g, cy_g, cz_g, radius, device)
        A_ref = math.pi * radius ** 2
    elif flow == "square":
        side = geom_param
        cx_g = nx * 0.25; cy_g = ny * 0.5
        solid = _build_square_mask(nx, ny, nz, cx_g, cy_g, side, device)
        A_ref = side * nz
    else:
        raise ValueError(f"Unknown flow: {flow}")

    dyn_p = 0.5 * 1.0 * u_in ** 2 * A_ref
    print(f"{tag} A_ref={A_ref:.1f} dyn_p={dyn_p:.6e} solid_cells={solid.sum().item()} ({time.time()-t0:.1f}s)",
          flush=True)

    # ── Setup lattice ────────────────────────────────────────────
    if lattice == "D3Q19":
        f, im, stream_fn, ff_bc, cmass_fn, collide_fn, step_fn = _setup_d3q19(
            None, solid, u_in, nz, ny, nx, nu, device, collision_name, Cs)
    else:
        f, im, stream_fn, ff_bc, cmass_fn, collide_fn, step_fn = _setup_d3q27(
            None, solid, u_in, nz, ny, nx, nu, device, collision_name, Cs)

    # ── Simulation loop ─────────────────────────────────────────
    warmup = N_STEPS // 3
    fric, pres = [], []
    all_finite = True
    final_step = 0

    for step in range(1, N_STEPS + 1):
        try:
            f = collide_fn(f, tau)
        except Exception as e:
            print(f"{tag} COLLISION ERROR at {step}: {e}", flush=True)
            all_finite = False
            break

        f = stream_fn(f)
        f, df, dp = step_fn(f, solid, nu)
        f = ff_bc(f, u_in=u_in)

        if step % 100 == 0:
            f = cmass_fn(f, im)

        if step > warmup and math.isfinite(df) and math.isfinite(dp):
            fric.append(df); pres.append(dp)
        final_step = step

        if not torch.isfinite(f).all():
            print(f"{tag} DIV at {step}", flush=True)
            all_finite = False
            break

        if step % 500 == 0 or step == N_STEPS:
            wf = fric[-SLIDE:] if len(fric) >= SLIDE else fric
            wp = pres[-SLIDE:] if len(pres) >= SLIDE else pres
            cf = sum(wf) / max(len(wf), 1) / dyn_p if wf else 0
            cp = sum(wp) / max(len(wp), 1) / dyn_p if wp else 0
            ct = cf + cp
            elap = time.time() - t0
            print(f"{tag} step={step} {dp_label}={ct:.5f} (Cf={cf:.5f} Cp={cp:.5f}) [{elap:.0f}s]",
                  flush=True)

    elapsed = time.time() - t0

    # ── Statistics ──────────────────────────────────────────────
    cf_full = sum(fric) / max(len(fric), 1) / dyn_p if fric else 0
    cp_full = sum(pres) / max(len(pres), 1) / dyn_p if pres else 0
    ct_full = cf_full + cp_full

    wf = fric[-SLIDE:] if len(fric) >= SLIDE else fric
    wp = pres[-SLIDE:] if len(pres) >= SLIDE else pres
    cf_slide = sum(wf) / max(len(wf), 1) / dyn_p if wf else 0
    cp_slide = sum(wp) / max(len(wp), 1) / dyn_p if wp else 0
    ct_slide = cf_slide + cp_slide

    err_pct = abs(ct_full - ref_value) / ref_value * 100 if ref_value and ref_value > 0 and math.isfinite(ct_full) else float("nan")

    if len(wf) > 1:
        ct_std = math.sqrt(sum(((fi + pi) / dyn_p - ct_slide) ** 2
                               for fi, pi in zip(wf, wp)) / (len(wf) - 1))
    else:
        ct_std = 0.0

    result = {
        "config_id": did,
        "collision_index": ci,
        "benchmark_index": bi,
        "flow": flow,
        "collision": collision_name,
        "collision_label": coll_label,
        "lattice": lattice,
        "Cs": Cs,
        "grid": f"{nx}x{ny}x{nz}",
        "geom_param": geom_param,
        "u_in": u_in,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "n_steps": N_STEPS,
        "warmup": warmup,
        "sliding_window": SLIDE,
        "A_ref": A_ref,
        "dyn_p": dyn_p,
        "dp_full": ct_full,
        "dp_fric_full": cf_full,
        "dp_pres_full": cp_full,
        "dp_slide": ct_slide,
        "dp_fric_slide": cf_slide,
        "dp_pres_slide": cp_slide,
        "dp_std_slide": ct_std,
        "ref_value": ref_value,
        "error_pct": err_pct,
        "samples": len(fric),
        "finite": all_finite,
        "elapsed_s": elapsed,
        "device": f"sdaa:{did}",
    }

    emoji = "✓" if all_finite else "✗"
    print(f"{tag} DONE {emoji} {dp_label}={ct_full:.5f} slide={ct_slide:.5f}±{ct_std:.5f} "
          f"err={err_pct:.1f}% ({elapsed:.0f}s)", flush=True)

    out_dir = Path("/tmp/fullmatrix_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"result_{did:02d}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"{tag} wrote result to {out_path}", flush=True)


if __name__ == "__main__":
    main()
