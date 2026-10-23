"""High-Re turbulence model comparison worker.

Runs a single case (cylinder Re=2000, cylinder Re=10000, or square Re=22000)
with all 4 LES turbulence models (Smagorinsky, WALE, Vreman, Dynamic Smag)
on one SDAA device, using the verified main loop (NoDynamics + half-way BB
+ far-field BC + mass correction) and drag_pressure_integration for Cd.

Usage:
    PYTHONPATH=src python high_re_turb_worker.py <device_id> <case> <output_json>
    case: cylinder_re2000 | cylinder_re10000 | square_re22000 | rans_all
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import (
    collide_smagorinsky_mrt3d,
    collide_wale_mrt3d,
    collide_vreman_mrt3d,
    collide_dynamic_smagorinsky_bgk3d,
)
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_pressure_integration,
    get_near_wall_2d,
)


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------
CASES = {
    "cylinder_re2000": {
        "geometry": "cylinder",
        "nx": 400, "ny": 160, "nz": 4,
        "diameter": 48.0,
        "u_in": 0.08,
        "Re": 2000.0,
        "ref_cd": 1.0,
        "n_steps": 5000,
        "warmup": 1000,
    },
    "cylinder_re10000": {
        "geometry": "cylinder",
        "nx": 400, "ny": 160, "nz": 4,
        "diameter": 48.0,
        "u_in": 0.08,
        "Re": 10000.0,
        "ref_cd": 0.8,
        "n_steps": 5000,
        "warmup": 1000,
    },
    "square_re22000": {
        "geometry": "square",
        "nx": 400, "ny": 160, "nz": 4,
        "diameter": 48.0,
        "u_in": 0.08,
        "Re": 22000.0,
        "ref_cd": 2.10,
        "n_steps": 5000,
        "warmup": 1000,
    },
}

# Turbulence models to test (LES)
LES_MODELS = {
    "smagorinsky": {
        "fn": collide_smagorinsky_mrt3d,
        "kwargs": {"C_s": 0.1},
        "label": "MRT+Smag(Cs=0.1)",
    },
    "wale": {
        "fn": collide_wale_mrt3d,
        "kwargs": {"C_w": 0.5},
        "label": "MRT+WALE(Cw=0.5)",
    },
    "vreman": {
        "fn": collide_vreman_mrt3d,
        "kwargs": {"C_V": 0.025},
        "label": "MRT+Vreman(Cv=0.025)",
    },
    "dynamic": {
        "fn": collide_dynamic_smagorinsky_bgk3d,
        "kwargs": {"filter_width": 2},
        "label": "BGK+DynSmag(fw=2)",
    },
}


def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    """Boolean solid mask for a cylinder extruded along z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def build_square_mask(nx, ny, nz, cx, cy, D, device):
    """Boolean solid mask for a square prism extruded along z-axis.

    Square spans x in [cx, cx+D-1], y in [cy-D//2, cy+D//2-1].
    """
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    half = int(D) // 2
    square = (xx >= cx) & (xx < cx + int(D)) & (yy >= cy - half) & (yy < cy + half)
    solid = square.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def run_model(device, case_cfg, model_name, model_cfg, tag):
    """Run one turbulence model for one case. Returns result dict."""
    nx, ny, nz = case_cfg["nx"], case_cfg["ny"], case_cfg["nz"]
    D = case_cfg["diameter"]
    radius = D / 2.0
    u_in = case_cfg["u_in"]
    Re = case_cfg["Re"]
    n_steps = case_cfg["n_steps"]
    warmup = case_cfg["warmup"]
    ref_cd = case_cfg["ref_cd"]

    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5

    cx_c = int(nx * 0.25)
    cy_c = int(ny * 0.5)

    A_frontal = D * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * A_frontal

    collide_fn = model_cfg["fn"]
    collide_kwargs = model_cfg["kwargs"]

    print(f"{tag} [{model_name}] tau={tau:.6f} nu={nu:.6e} "
          f"dpS={dpS:.4f} {model_cfg['label']}", flush=True)

    t0 = time.time()

    # Build geometry
    if case_cfg["geometry"] == "cylinder":
        solid = build_cylinder_mask(nx, ny, nz, cx_c, cy_c, radius, device)
    else:
        solid = build_square_mask(nx, ny, nz, cx_c, cy_c, D, device)

    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Near-wall mask and surface mesh
    near = get_near_wall_2d(solid)
    n_near = int(near.sum().item())

    if case_cfg["geometry"] == "cylinder":
        mesh = SurfaceMesh.from_cylinder(solid, near, cx_c, cy_c, radius)
    else:
        mesh = SurfaceMesh.from_square_prism(solid, near, cx_c, cy_c, int(D))

    print(f"{tag} [{model_name}] solid={n_solid} near={n_near} "
          f"mesh built ({time.time()-t0:.1f}s)", flush=True)

    # Initialize flow field
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} [{model_name}] init done ({time.time()-t0:.1f}s), "
          f"starting loop...", flush=True)

    # Accumulators
    cd_p_hist = []
    cd_tot_hist = []
    cl_hist = []
    diverged = False
    last_step = 0

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision
        f_pre = f.clone()

        # 2. Collision (turbulence model)
        f = collide_fn(f, tau=tau, **collide_kwargs)

        # 3. NoDynamics: restore solid cells to pre-collision
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming)
        f = bounce_back_cells_3d(f, solid)

        # 5. Streaming
        f = stream3d(f)

        # 6. Far-field BC
        f = far_field_bc_3d(f, u_in=u_in)

        # 7. Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # 8. Drag (pressure integration)
        cd_x, cd_y, _ = drag_pressure_integration(f, mesh, dpS)

        # Check divergence
        if not torch.isfinite(f).all():
            print(f"{tag} [{model_name}] DIVERGED at step {step}", flush=True)
            diverged = True
            last_step = step
            break

        last_step = step

        # Record post-warmup
        if step > warmup:
            if math.isfinite(cd_x):
                cd_tot_hist.append(cd_x)
            if math.isfinite(cd_y):
                cl_hist.append(cd_y)

        if step % 500 == 0:
            _, ux, _, _ = macroscopic3d(f)
            ms = float(torch.sqrt(ux * ux).max().item())
            elapsed = time.time() - t0
            cd_avg = sum(cd_tot_hist) / max(len(cd_tot_hist), 1) if cd_tot_hist else float("nan")
            print(f"{tag} [{model_name}] step={step} Cd={cd_avg:.4f} "
                  f"max|ux|={ms:.4f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0

    # Compute statistics
    def _mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    def _std(vals):
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))

    cd_mean = _mean(cd_tot_hist)
    cd_std = _std(cd_tot_hist)
    cl_mean = _mean(cl_hist)
    cl_std = _std(cl_hist)

    err_pct = abs(cd_mean - ref_cd) / ref_cd * 100 if (
        ref_cd > 0 and math.isfinite(cd_mean)) else float("nan")

    result = {
        "case": case_cfg["geometry"] + f"_Re{int(Re)}",
        "model": model_name,
        "model_label": model_cfg["label"],
        "device": str(device),
        "grid": f"{nx}x{ny}x{nz}",
        "diameter": D,
        "u_in": u_in,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "warmup": warmup,
        "Cd_mean": cd_mean,
        "Cd_std": cd_std,
        "Cl_mean": cl_mean,
        "Cl_std": cl_std,
        "Cd_ref": ref_cd,
        "error_pct": err_pct,
        "n_samples": len(cd_tot_hist),
        "finite": not diverged,
        "diverged": diverged,
        "last_step": last_step,
        "elapsed_s": elapsed,
    }

    status = "DIVERGED" if diverged else "OK"
    print(f"{tag} [{model_name}] DONE {status} Cd={cd_mean:.4f}±{cd_std:.4f} "
          f"(ref={ref_cd}, err={err_pct:.1f}%) Cl={cl_mean:.4f} "
          f"time={elapsed:.0f}s", flush=True)

    return result


def run_rans(device, case_cfg, tag):
    """Run RANS k-epsilon model for one case. Returns result dict."""
    from tensorlbm.rans_ke import KESolver, collide_rans_ke

    nx, ny, nz = case_cfg["nx"], case_cfg["ny"], case_cfg["nz"]
    D = case_cfg["diameter"]
    radius = D / 2.0
    u_in = case_cfg["u_in"]
    Re = case_cfg["Re"]
    n_steps = case_cfg["n_steps"]
    warmup = case_cfg["warmup"]
    ref_cd = case_cfg["ref_cd"]

    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5

    cx_c = int(nx * 0.25)
    cy_c = int(ny * 0.5)

    A_frontal = D * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * A_frontal

    print(f"{tag} [k-epsilon] tau={tau:.6f} nu={nu:.6e} "
          f"dpS={dpS:.4f} MRT+k-epsilon", flush=True)

    t0 = time.time()

    # Build geometry
    if case_cfg["geometry"] == "cylinder":
        solid = build_cylinder_mask(nx, ny, nz, cx_c, cy_c, radius, device)
    else:
        solid = build_square_mask(nx, ny, nz, cx_c, cy_c, D, device)

    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    near = get_near_wall_2d(solid)
    n_near = int(near.sum().item())

    if case_cfg["geometry"] == "cylinder":
        mesh = SurfaceMesh.from_cylinder(solid, near, cx_c, cy_c, radius)
    else:
        mesh = SurfaceMesh.from_square_prism(solid, near, cx_c, cy_c, int(D))

    print(f"{tag} [k-epsilon] solid={n_solid} near={n_near} "
          f"mesh built ({time.time()-t0:.1f}s)", flush=True)

    # Initialize flow field
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=device)
    initial_mass = float(rho0.sum().item())

    # Initialize k-epsilon solver
    _, ux_init, uy_init, uz_init = macroscopic3d(f)
    ke_solver = KESolver(nu=nu, dx=1.0)
    ke_solver.initialize(ux_init, uy_init, uz_init)

    print(f"{tag} [k-epsilon] init done ({time.time()-t0:.1f}s), "
          f"starting loop...", flush=True)

    cd_tot_hist = []
    cl_hist = []
    diverged = False
    last_step = 0

    for step in range(1, n_steps + 1):
        f_pre = f.clone()

        # RANS k-epsilon collision (MRT)
        f = collide_rans_ke(f, tau=tau, ke_solver=ke_solver, mask=solid,
                            lattice="D3Q19", collision="MRT")

        # NoDynamics
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # Bounce-back
        f = bounce_back_cells_3d(f, solid)

        # Stream
        f = stream3d(f)

        # Far-field BC
        f = far_field_bc_3d(f, u_in=u_in)

        # Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # Drag
        cd_x, cd_y, _ = drag_pressure_integration(f, mesh, dpS)

        if not torch.isfinite(f).all():
            print(f"{tag} [k-epsilon] DIVERGED at step {step}", flush=True)
            diverged = True
            last_step = step
            break

        last_step = step

        if step > warmup:
            if math.isfinite(cd_x):
                cd_tot_hist.append(cd_x)
            if math.isfinite(cd_y):
                cl_hist.append(cd_y)

        if step % 500 == 0:
            _, ux, _, _ = macroscopic3d(f)
            ms = float(torch.sqrt(ux * ux).max().item())
            elapsed = time.time() - t0
            cd_avg = sum(cd_tot_hist) / max(len(cd_tot_hist), 1) if cd_tot_hist else float("nan")
            print(f"{tag} [k-epsilon] step={step} Cd={cd_avg:.4f} "
                  f"max|ux|={ms:.4f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0

    def _mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    def _std(vals):
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))

    cd_mean = _mean(cd_tot_hist)
    cd_std = _std(cd_tot_hist)
    cl_mean = _mean(cl_hist)
    cl_std = _std(cl_hist)

    err_pct = abs(cd_mean - ref_cd) / ref_cd * 100 if (
        ref_cd > 0 and math.isfinite(cd_mean)) else float("nan")

    result = {
        "case": case_cfg["geometry"] + f"_Re{int(Re)}",
        "model": "k-epsilon",
        "model_label": "MRT+k-epsilon(RANS)",
        "device": str(device),
        "grid": f"{nx}x{ny}x{nz}",
        "diameter": D,
        "u_in": u_in,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "warmup": warmup,
        "Cd_mean": cd_mean,
        "Cd_std": cd_std,
        "Cl_mean": cl_mean,
        "Cl_std": cl_std,
        "Cd_ref": ref_cd,
        "error_pct": err_pct,
        "n_samples": len(cd_tot_hist),
        "finite": not diverged,
        "diverged": diverged,
        "last_step": last_step,
        "elapsed_s": elapsed,
    }

    status = "DIVERGED" if diverged else "OK"
    print(f"{tag} [k-epsilon] DONE {status} Cd={cd_mean:.4f}±{cd_std:.4f} "
          f"(ref={ref_cd}, err={err_pct:.1f}%) Cl={cl_mean:.4f} "
          f"time={elapsed:.0f}s", flush=True)

    return result


def main():
    device_id = int(sys.argv[1])
    case_name = sys.argv[2]
    output_path = sys.argv[3]

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{device_id}]"

    if case_name == "rans_all":
        # Run RANS k-epsilon on all 3 cases
        all_results = []
        for cn in ["cylinder_re2000", "cylinder_re10000", "square_re22000"]:
            cfg = CASES[cn]
            print(f"\n{tag} === RANS k-epsilon: {cn} ===", flush=True)
            r = run_rans(device, cfg, tag)
            all_results.append(r)
            # Save intermediate results
            Path(output_path).write_text(json.dumps(all_results, indent=2))
        print(f"\n{tag} All RANS cases complete.", flush=True)
        return

    cfg = CASES[case_name]
    print(f"\n{tag} === Case: {case_name} ({cfg['geometry']} Re={int(cfg['Re'])}) ===",
          flush=True)
    print(f"{tag} Grid: {cfg['nx']}x{cfg['ny']}x{cfg['nz']} D={cfg['diameter']} "
          f"u_in={cfg['u_in']} steps={cfg['n_steps']}", flush=True)

    all_results = []
    for model_name, model_cfg in LES_MODELS.items():
        print(f"\n{tag} --- Model: {model_name} ({model_cfg['label']}) ---",
              flush=True)
        r = run_model(device, cfg, model_name, model_cfg, tag)
        all_results.append(r)
        # Save intermediate results after each model
        Path(output_path).write_text(json.dumps(all_results, indent=2))

    print(f"\n{tag} All models complete for {case_name}.", flush=True)


if __name__ == "__main__":
    main()
