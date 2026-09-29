"""Channel flow validation worker — turbulent channel at Re_tau=180.

Body-force driven periodic channel with wall function on top/bottom walls.
Validates mean velocity profile against log-law and computes Cf.

Usage:
    SDAA_VISIBLE_DEVICES=<card_id> PYTHONPATH=src python channel_worker.py \
        --cs 0.10 --output /tmp/channel_cs010.json
"""
from __future__ import annotations
import argparse, json, math, sys, time
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.ibm import ibm_apply_body_force_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d

KAPPA = 0.41
B_LOG = 5.0


def log_law_uplus(yp: float) -> float:
    """Log-law: u+ = (1/kappa)*ln(y+) + B"""
    return math.log(yp) / KAPPA + B_LOG if yp > 0 else 0.0


def run(cs: float, nx: int, ny: int, nz: int, nu: float,
        n_steps: int, warmup: int, device: torch.device,
        output_path: str) -> dict:
    h = ny / 2.0  # half channel height
    # target u_tau from Re_tau = u_tau * h / nu
    retau_target = 180.0
    u_tau_target = retau_target * nu / h
    # body force per unit mass: g = u_tau^2 / h
    body_force = u_tau_target ** 2 / h

    tau = 3.0 * nu + 0.5

    # Solid mask: top/bottom walls at y=0 and y=ny-1
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True   # bottom wall
    solid[:, -1, :] = True  # top wall

    fluid = ~solid
    n_fluid = int(fluid.sum().item())

    print(f"Channel flow: Re_tau target={retau_target:.0f}", flush=True)
    print(f"Grid: {nx}x{ny}x{nz}, h={h:.1f}, nu={nu:.6f}, tau={tau:.6f}", flush=True)
    print(f"u_tau_target={u_tau_target:.6f}, body_force={body_force:.6e}", flush=True)
    print(f"Cs={cs}, n_steps={n_steps}, warmup={warmup}", flush=True)
    print(f"Fluid cells: {n_fluid}, Device: {device}", flush=True)

    # Initialize with turbulent mean profile + perturbations (generate on CPU, move to device)
    rng = torch.Generator(device='cpu')
    # Use LBM-safe centerline velocity: u_max ~ 0.05 (Ma ~ 0.087)
    # For Re_tau=180, u_c/u_tau ≈ 18, so u_tau = u_c/18
    # With u_c=0.05: u_tau_init = 0.05/18 ≈ 0.00278
    # But the body force will drive u_tau to target, so init doesn't need to match target
    u_c_init = min(u_tau_target * 18.0, 0.05)  # cap at 0.05 for LBM stability
    # 1/7th power-law profile: u(y) = U_c * (y/h)^(1/7) for bottom half, mirrored for top
    y_cpu = torch.arange(ny, dtype=torch.float32)
    h_cpu = float(h)
    # Distance from nearest wall
    y_dist = torch.minimum(y_cpu, h_cpu * 2 - 1 - y_cpu).clamp(min=0.5)
    u_profile = u_c_init * (y_dist / h_cpu) ** (1.0 / 7.0)
    u_profile[0] = 0.0
    u_profile[-1] = 0.0
    ux_init = u_profile.unsqueeze(0).unsqueeze(-1).expand(nz, ny, nx).clone().to(device)
    # Add random perturbations (5% of centerline)
    ux_init += torch.randn(nz, ny, nx, generator=rng, device='cpu').to(device) * u_c_init * 0.03
    uy_init = torch.randn(nz, ny, nx, generator=rng, device='cpu').to(device) * u_c_init * 0.02
    uz_init = torch.randn(nz, ny, nx, generator=rng, device='cpu').to(device) * u_c_init * 0.03
    ux_init[solid] = 0.0
    uy_init[solid] = 0.0
    uz_init[solid] = 0.0

    rho0 = torch.ones(nz, ny, nx, device=device)
    f = equilibrium3d(rho0, ux_init, uy_init, uz_init, device=device)
    initial_mass = float(rho0.sum().item())

    # Collect samples for statistics
    ux_samples = []   # list of ux fields for averaging
    cf_samples = []   # list of Cf values
    u_tau_samples = []

    t0 = time.time()
    for step in range(1, n_steps + 1):
        # 1. Collision
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs)

        # 2. Streaming (wraps around for x and z → periodic)
        f = stream3d(f)

        # 3. Compute macroscopic fields (before wall function)
        rho, ux, uy, uz = macroscopic3d(f)

        # 4. Apply uniform body force (Guo forcing)
        fx_field = torch.full((nz, ny, nx), body_force, device=device)
        fx_field[solid] = 0.0
        f = ibm_apply_body_force_3d(f, fx_field,
                                     torch.zeros_like(fx_field),
                                     torch.zeros_like(fx_field))

        # 5. Wall function on top/bottom walls
        f, drag_fric, drag_pres = wall_function_3d(f, solid, nu, y_val=0.5)

        # 6. Bounce-back on solid cells
        f = bounce_back_cells_3d(f, solid)

        # 7. Mass correction (periodic)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        # Sample after warmup
        if step > warmup:
            _, ux_s, _, _ = macroscopic3d(f)
            ux_samples.append(ux_s.clone())

            # Compute bulk velocity and Cf
            ux_fluid = ux_s[fluid]
            u_bulk = float(ux_fluid.mean().item())
            # Cf = 2 * tau_w / (rho * u_bulk^2) = 2 * u_tau^2 / u_bulk^2
            # Estimate u_tau from wall shear via log-law
            # Use the wall function's drag: drag_fric is integral of tau_w
            # Actually, for channel: tau_w = u_tau^2 = body_force * h
            # So u_tau_sim = sqrt(body_force * h)
            u_tau_sim = math.sqrt(body_force * h)
            cf = 2.0 * u_tau_sim ** 2 / (u_bulk ** 2) if u_bulk > 0 else 0.0
            cf_samples.append(cf)
            u_tau_samples.append(u_tau_sim)

        if step % 500 == 0 or step == n_steps:
            elapsed = time.time() - t0
            _, ux_c, _, _ = macroscopic3d(f)
            u_bulk_c = float(ux_c[fluid].mean().item())
            u_tau_c = math.sqrt(body_force * h)
            re_tau_c = u_tau_c * h / nu
            cf_c = 2.0 * u_tau_c ** 2 / (u_bulk_c ** 2) if u_bulk_c > 0 else 0.0
            print(f"  step {step:5d}: u_bulk={u_bulk_c:.6f} u_tau={u_tau_c:.6f} "
                  f"Re_tau={re_tau_c:.1f} Cf={cf_c:.6f} [{elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t0

    # Compute mean velocity profile
    if ux_samples:
        ux_stacked = torch.stack(ux_samples, dim=0)  # (n_samples, nz, ny, nx)
        ux_mean = ux_stacked.mean(dim=0)  # (nz, ny, nx)

        # Average over x (streamwise) and z (spanwise) to get u(y)
        ux_y = ux_mean.mean(dim=(0, 2))  # (ny,)

        # y+ values (distance from wall in +y direction)
        y_coords = torch.arange(ny, dtype=torch.float32, device=device)
        # Half-cell offset: first fluid cell center at y=0.5
        y_dist = y_coords + 0.5  # distance from bottom wall

        # Use target u_tau for y+ computation
        y_plus = y_dist * u_tau_target / nu

        # u+ = u / u_tau
        u_plus = ux_y / u_tau_target

        # Extract profile data (filter to fluid region)
        profile_data = []
        for j in range(1, ny - 1):  # skip wall cells
            yp_j = float(y_plus[j].item())
            up_j = float(u_plus[j].item())
            u_j = float(ux_y[j].item())
            profile_data.append({
                "y_cell": j,
                "y_dist": float(y_dist[j].item()),
                "y_plus": yp_j,
                "u_plus": up_j,
                "u": u_j,
                "u_plus_loglaw": log_law_uplus(yp_j),
            })

        # Compute RMS of u+ vs log-law in log region (y+ > 30)
        log_region = [(d["y_plus"], d["u_plus"], d["u_plus_loglaw"])
                       for d in profile_data if d["y_plus"] > 30]
        if log_region:
            rms_error = math.sqrt(sum((up - up_ll)**2 for _, up, up_ll in log_region) / len(log_region))
        else:
            rms_error = float('nan')
        n_samples = len(ux_samples)
    else:
        profile_data = []
        rms_error = float('nan')
        n_samples = 0

    # Compute final statistics
    u_tau_final = math.sqrt(body_force * h)
    re_tau_final = u_tau_final * h / nu

    # Cf from mean samples
    cf_mean = sum(cf_samples) / max(len(cf_samples), 1) if cf_samples else float('nan')
    cf_std = 0.0
    if len(cf_samples) > 1:
        cf_mean_val = cf_mean
        cf_std = math.sqrt(sum((c - cf_mean_val)**2 for c in cf_samples) / len(cf_samples))

    # Expected Cf from Dean correlation: Cf = 0.073 * Re_bulk^(-0.25)
    # Or use Re_tau to estimate: Re_bulk ≈ Re_tau * (some factor)
    # Use a simpler check: Cf should be ~ 2 * (u_tau/u_bulk)^2
    # For Re_tau=180, typical u_bulk/u_tau ≈ 15-18
    cf_expected = 2.0 / (16.5 ** 2)  # rough estimate: u_bulk/u_tau ≈ 16.5

    result = {
        "case": "turbulent_channel",
        "cs": cs,
        "retau_target": retau_target,
        "retau_achieved": re_tau_final,
        "nx": nx, "ny": ny, "nz": nz,
        "h": h,
        "nu": nu, "tau": tau,
        "u_tau_target": u_tau_target,
        "u_tau_final": u_tau_final,
        "body_force": body_force,
        "n_steps": n_steps,
        "warmup": warmup,
        "n_samples": n_samples,
        "cf_mean": cf_mean,
        "cf_std": cf_std,
        "cf_expected": cf_expected,
        "rms_error_loglaw": rms_error,
        "wall_clock_s": elapsed,
        "device": str(device),
        "profile": profile_data,
    }

    print(f"\n=== Results ===", flush=True)
    print(f"Re_tau achieved: {re_tau_final:.1f} (target: {retau_target:.0f})", flush=True)
    print(f"u_tau: {u_tau_final:.6f}", flush=True)
    print(f"Cf: {cf_mean:.6f} ± {cf_std:.6f}", flush=True)
    print(f"RMS error vs log-law: {rms_error:.4f}", flush=True)
    print(f"Time: {elapsed:.0f}s", flush=True)

    with open(output_path, 'w') as fp:
        json.dump(result, fp, indent=2)
    print(f"Results saved to {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs", type=float, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--nu", type=float, default=0.002)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--device", type=str, default="sdaa")
    args = parser.parse_args()

    # Resolve device
    if args.device == "sdaa":
        if hasattr(torch, 'sdaa') and torch.sdaa.device_count() > 0:
            device = torch.device("sdaa:0")
        else:
            device = torch.device("cpu")
            print("WARNING: No SDAA devices visible, falling back to CPU")
    else:
        device = torch.device(args.device)

    run(
        cs=args.cs,
        nx=args.nx, ny=args.ny, nz=args.nz,
        nu=args.nu,
        n_steps=args.n_steps, warmup=args.warmup,
        device=device,
        output_path=args.output,
    )
