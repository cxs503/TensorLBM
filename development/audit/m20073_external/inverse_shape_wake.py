"""Multi-parameter geometry inverse problem: recover center + radius from wake field.

This extends the single-parameter inverse design (radius from drag) to
multi-parameter recovery using spatially-distributed wake velocity information.

Problem setup:
- Same bounded box with uniform inflow as inverse_design.py
- Hidden true sphere geometry produces reference wake velocity field
- Optimize (cy, cz, r) or (cx, cy, r) to match wake field at downstream probe plane
- Observable: velocity field at x = x_probe (wake plane), L2 norm matching

Key insight: drag is a single scalar, so it cannot distinguish between different
geometric configurations that produce the same total force. The wake velocity
field provides spatially-distributed information that breaks this degeneracy.

Usage:
    python examples/inverse_shape_wake.py                    # recover (cy, cz, r)
    python examples/inverse_shape_wake.py --params cx,cy,r   # recover (cx, cy, r)
    python examples/inverse_shape_wake.py --noise 0.01       # 1% noise + bootstrap
"""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path

import torch
import numpy as np

from tensorlbm.autograd_path import (
    InletSpec,
    OutletSpec,
    differentiable_step,
    obstacle_force,
    rollout,
)
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.soft_geometry import SoftGeometry

# Grid and geometry parameters (smaller for faster iteration)
NZ, NY, NX = 12, 16, 26
CX_TRUE, CY_TRUE, CZ_TRUE = 7.0, 8.0, 6.0
R_TRUE = 2.3
U_IN = 0.08
TAU = 0.55
STEPS = 60  # solver steps
PROBE_START = 30  # start measuring after transient
EPS_SOFT = 0.25  # SDF temperature

# Initial guesses (deliberately wrong)
CX_GUESS, CY_GUESS, CZ_GUESS = 8.0, 9.0, 7.0
R_GUESS = 2.9

# Optimization parameters
LR_DEFAULT = 0.05
ITERS_DEFAULT = 150

# Wake probe location (downstream of obstacle)
X_PROBE = 20  # x-coordinate of wake measurement plane


def uniform_inflow_f0(
    u_in: float, nz: int, ny: int, nx: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """D3Q19 equilibrium initialisation at uniform inflow."""
    ones = torch.ones(nz, ny, nx, dtype=dtype, device=device)
    zeros = torch.zeros_like(ones)
    return equilibrium3d(ones, u_in * ones, zeros, zeros)


def compute_macroscopic(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract macroscopic fields (rho, ux, uy, uz) from distribution function."""
    rho = f.sum(dim=0)
    # D3Q19 velocities: c = [(0,0,0), (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), ...]
    # ux = sum_q c_qx * f_q / rho
    ux = (f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[13] - f[14] + f[15] - f[16]) / rho
    uy = (f[3] - f[4] + f[7] - f[8] - f[9] + f[10] + f[11] - f[12] - f[13] + f[14]) / rho
    uz = (f[5] - f[6] + f[11] - f[12] - f[13] + f[14] - f[15] + f[16] + f[17] - f[18]) / rho
    return rho, ux, uy, uz


def extract_wake_field(
    f: torch.Tensor, x_probe: int
) -> torch.Tensor:
    """Extract velocity field at wake probe plane x = x_probe.
    
    Returns:
        Velocity field shape (3, nz, ny) - (ux, uy, uz) at the probe plane
    """
    _, ux, uy, uz = compute_macroscopic(f)
    # Extract plane at x = x_probe
    wake_vel = torch.stack([
        ux[:, :, x_probe],
        uy[:, :, x_probe],
        uz[:, :, x_probe],
    ], dim=0)  # (3, nz, ny)
    return wake_vel


def run_forward(
    cx: float | torch.Tensor,
    cy: float | torch.Tensor,
    cz: float | torch.Tensor,
    r: float | torch.Tensor,
    u_in: float,
    steps: int,
    probe_start: int,
    eps: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run forward simulation and extract wake field.
    
    Returns:
        (final_f, wake_field) where wake_field has shape (3, nz, ny)
    """
    # Create soft geometry
    geom = SoftGeometry(
        kind="sphere",
        center=(cx, cy, cz),
        size=(r,),
        epsilon=eps,
    )
    
    # Initial condition
    f0 = uniform_inflow_f0(u_in, NZ, NY, NX, dtype, device)
    
    # Boundary conditions
    inlet = InletSpec(ux=u_in)
    outlet = OutletSpec()
    
    # Rollout
    f_final, probes = rollout(
        f0,
        steps,
        TAU,
        None,
        soft=geom,
        inlet=inlet,
        outlet=outlet,
        return_probes=True,
        probe_start=probe_start,
    )
    
    # Extract wake field from final state
    wake_field = extract_wake_field(f_final, X_PROBE)
    
    return f_final, wake_field


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--params",
        default="cy,cz,r",
        help="Comma-separated parameters to optimize (cx, cy, cz, r)",
    )
    parser.add_argument("--lr", type=float, default=LR_DEFAULT)
    parser.add_argument("--iters", type=int, default=ITERS_DEFAULT)
    parser.add_argument("--noise", type=float, default=0.0, help="Noise level (fraction)")
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap samples (0=disable)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--out", default=None, help="Output directory for results")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    
    # Parse parameters to optimize
    param_names = [p.strip() for p in args.params.split(",")]
    
    # True values
    true_params = {"cx": CX_TRUE, "cy": CY_TRUE, "cz": CZ_TRUE, "r": R_TRUE}
    guess_params = {"cx": CX_GUESS, "cy": CY_GUESS, "cz": CZ_GUESS, "r": R_GUESS}
    
    print(f"Multi-parameter geometry inverse problem")
    print(f"Grid: ({NZ}, {NY}, {NX}), wake probe at x={X_PROBE}")
    print(f"Parameters to recover: {param_names}")
    print(f"True values: {', '.join(f'{k}={true_params[k]:.2f}' for k in param_names)}")
    print(f"Initial guesses: {', '.join(f'{k}={guess_params[k]:.2f}' for k in param_names)}")
    print(f"Optimization: {args.iters} iters, lr={args.lr}, noise={args.noise}")
    
    # Generate reference wake field (with optional noise)
    with torch.no_grad():
        _, wake_star = run_forward(
            CX_TRUE, CY_TRUE, CZ_TRUE, R_TRUE,
            U_IN, STEPS, PROBE_START, EPS_SOFT, dtype, device
        )
        if args.noise > 0:
            noise_std = args.noise * wake_star.abs().mean()
            wake_star = wake_star + torch.randn_like(wake_star) * noise_std
            print(f"Added noise: std={noise_std:.6f} ({args.noise*100:.1f}% of mean |wake|)")
    
    # Create learnable parameters
    param_tensors = {}
    for name in param_names:
        param_tensors[name] = torch.tensor(
            guess_params[name], dtype=dtype, device=device, requires_grad=True
        )
    
    # Optimizer
    optim = torch.optim.Adam(list(param_tensors.values()), lr=args.lr)
    
    # Optimization loop
    print("\nStarting optimization...")
    loss0 = None
    best_loss = math.inf
    best_params = {k: guess_params[k] for k in param_names}
    history = []
    
    for it in range(args.iters):
        # Cosine learning rate decay
        for group in optim.param_groups:
            group["lr"] = args.lr * (
                0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * it / args.iters))
            )
        
        # Build geometry with current parameters
        cx = param_tensors.get("cx", torch.tensor(CX_TRUE, dtype=dtype, device=device))
        cy = param_tensors.get("cy", torch.tensor(CY_TRUE, dtype=dtype, device=device))
        cz = param_tensors.get("cz", torch.tensor(CZ_TRUE, dtype=dtype, device=device))
        r = param_tensors.get("r", torch.tensor(R_TRUE, dtype=dtype, device=device))
        
        # Forward pass
        _, wake_pred = run_forward(
            cx, cy, cz, r,
            U_IN, STEPS, PROBE_START, EPS_SOFT, dtype, device
        )
        
        # Loss: L2 norm of velocity field difference
        loss = ((wake_pred - wake_star) ** 2).mean()
        
        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        # Clamp parameters to reasonable bounds
        with torch.no_grad():
            for name, param in param_tensors.items():
                if name in ["cx", "cy", "cz"]:
                    param.clamp_(2.0, min(NX, NY, NZ) - 2.0)
                elif name == "r":
                    param.clamp_(0.5, min(NY, NZ) / 2 - 0.5)
        
        loss_val = float(loss.detach())
        if loss0 is None:
            loss0 = loss_val
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = {k: float(param_tensors[k].detach()) for k in param_names}
        
        # Record history
        current_params = {k: float(param_tensors[k].detach()) for k in param_names}
        history.append({"iter": it + 1, "loss": loss_val, "params": current_params})
        
        if it == 0 or (it + 1) % args.log_every == 0:
            param_str = ", ".join(f"{k}={current_params[k]:.4f}" for k in param_names)
            print(f"  iter {it + 1:4d}  loss={loss_val:.6e}  {param_str}")
    
    # Final results
    final_params = {k: float(param_tensors[k].detach()) for k in param_names}
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Initial loss: {loss0:.6e}")
    print(f"Final loss:   {float(loss.detach()):.6e}")
    print(f"Best loss:    {best_loss:.6e}")
    print("\nParameter recovery:")
    errors = {}
    for k in param_names:
        true_val = true_params[k]
        pred_val = final_params[k]
        rel_err = abs(pred_val - true_val) / true_val
        errors[k] = rel_err
        status = "✓" if rel_err < 0.03 else "✗"
        print(f"  {k:3s}: true={true_val:.4f}, pred={pred_val:.4f}, rel_err={rel_err:.2e} {status}")
    
    # Bootstrap confidence intervals
    if args.bootstrap > 0 and args.noise > 0:
        print(f"\nBootstrap confidence intervals ({args.bootstrap} samples):")
        bootstrap_results = {k: [] for k in param_names}
        
        for b in range(args.bootstrap):
            torch.manual_seed(args.seed + b + 1000)
            
            # Regenerate noisy reference
            with torch.no_grad():
                _, wake_star_b = run_forward(
                    CX_TRUE, CY_TRUE, CZ_TRUE, R_TRUE,
                    U_IN, STEPS, PROBE_START, EPS_SOFT, dtype, device
                )
                noise_std = args.noise * wake_star_b.abs().mean()
                wake_star_b = wake_star_b + torch.randn_like(wake_star_b) * noise_std
            
            # Re-optimize
            param_tensors_b = {}
            for name in param_names:
                param_tensors_b[name] = torch.tensor(
                    guess_params[name], dtype=dtype, device=device, requires_grad=True
                )
            optim_b = torch.optim.Adam(list(param_tensors_b.values()), lr=args.lr)
            
            for it in range(args.iters):
                for group in optim_b.param_groups:
                    group["lr"] = args.lr * (
                        0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * it / args.iters))
                    )
                
                cx = param_tensors_b.get("cx", torch.tensor(CX_TRUE, dtype=dtype, device=device))
                cy = param_tensors_b.get("cy", torch.tensor(CY_TRUE, dtype=dtype, device=device))
                cz = param_tensors_b.get("cz", torch.tensor(CZ_TRUE, dtype=dtype, device=device))
                r = param_tensors_b.get("r", torch.tensor(R_TRUE, dtype=dtype, device=device))
                
                _, wake_pred_b = run_forward(
                    cx, cy, cz, r,
                    U_IN, STEPS, PROBE_START, EPS_SOFT, dtype, device
                )
                loss_b = ((wake_pred_b - wake_star_b) ** 2).mean()
                
                optim_b.zero_grad()
                loss_b.backward()
                optim_b.step()
                
                with torch.no_grad():
                    for name, param in param_tensors_b.items():
                        if name in ["cx", "cy", "cz"]:
                            param.clamp_(2.0, min(NX, NY, NZ) - 2.0)
                        elif name == "r":
                            param.clamp_(0.5, min(NY, NZ) / 2 - 0.5)
            
            for k in param_names:
                bootstrap_results[k].append(float(param_tensors_b[k].detach()))
        
        # Compute 95% confidence intervals
        for k in param_names:
            values = np.array(bootstrap_results[k])
            ci_low = np.percentile(values, 2.5)
            ci_high = np.percentile(values, 97.5)
            true_val = true_params[k]
            covers = "✓" if ci_low <= true_val <= ci_high else "✗"
            print(f"  {k:3s}: 95% CI=[{ci_low:.4f}, {ci_high:.4f}], true={true_val:.4f} {covers}")
    
    # Save results
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "config": {
                "grid": [NZ, NY, NX],
                "x_probe": X_PROBE,
                "params": param_names,
                "iters": args.iters,
                "lr": args.lr,
                "noise": args.noise,
                "bootstrap": args.bootstrap,
            },
            "true_params": {k: true_params[k] for k in param_names},
            "initial_guess": {k: guess_params[k] for k in param_names},
            "recovered_params": final_params,
            "relative_errors": errors,
            "loss_history": history,
            "success": all(errors[k] < 0.03 for k in param_names),
        }
        
        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
