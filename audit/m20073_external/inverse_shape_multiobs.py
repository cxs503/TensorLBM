"""Multi-parameter geometry inverse: recover (r, cy) from drag + wake center.

This uses two complementary observables:
1. Drag (integrated force): sensitive to radius, insensitive to center
2. Wake center of mass: sensitive to center position, also sensitive to radius

By combining both observables, we can recover both parameters simultaneously.

Usage:
    python examples/inverse_shape_multiobs.py
    python examples/inverse_shape_multiobs.py --noise 0.01 --bootstrap 20
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
    obstacle_force,
    rollout,
)
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.soft_geometry import SoftGeometry

# Grid and geometry parameters
NZ, NY, NX = 12, 16, 26
CX_TRUE, CY_TRUE, CZ_TRUE = 7.0, 8.0, 6.0
R_TRUE = 2.3
U_IN = 0.08
TAU = 0.55
STEPS = 60
PROBE_START = 30
EPS_SOFT = 0.25

# Initial guesses
CY_GUESS = 9.0
R_GUESS = 2.9

# Optimization parameters
LR_DEFAULT = 0.05
ITERS_DEFAULT = 150

# Wake probe location
X_PROBE = 20


def uniform_inflow_f0(
    u_in: float, nz: int, ny: int, nx: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """D3Q19 equilibrium initialisation at uniform inflow."""
    ones = torch.ones(nz, ny, nx, dtype=dtype, device=device)
    zeros = torch.zeros_like(ones)
    return equilibrium3d(ones, u_in * ones, zeros, zeros)


def compute_observables(
    f_final: torch.Tensor,
    probes: list[torch.Tensor],
    geom: SoftGeometry,
    u_in: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute drag and wake center of mass.
    
    Returns:
        (drag, cy_wake) where drag is scalar and cy_wake is the y-center of wake
    """
    # Drag: integrated momentum-exchange force
    w = geom.fluid_weight(NZ, NY, NX, dtype=f_final.dtype, device=f_final.device)
    drag = sum(obstacle_force(p, 1.0 - w)[0] for p in probes) / len(probes)
    
    # Wake center of mass
    rho = f_final.sum(dim=0)
    ux = (f_final[1] - f_final[2] + f_final[7] - f_final[8] + 
          f_final[9] - f_final[10] + f_final[13] - f_final[14] + 
          f_final[15] - f_final[16]) / rho
    
    # Wake deficit at probe plane
    wake_deficit = u_in - ux[:, :, X_PROBE]
    wake_deficit = torch.clamp(wake_deficit, min=0.0)
    
    # Center of mass in y
    yy = torch.arange(NY, dtype=f_final.dtype, device=f_final.device).reshape(1, -1)
    total_deficit = wake_deficit.sum()
    cy_wake = (wake_deficit * yy).sum() / total_deficit
    
    return drag, cy_wake


def run_forward(
    cy: float | torch.Tensor,
    r: float | torch.Tensor,
    u_in: float,
    steps: int,
    probe_start: int,
    eps: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run forward simulation and compute observables.
    
    Returns:
        (drag, cy_wake, f_final)
    """
    geom = SoftGeometry(
        kind="sphere",
        center=(CX_TRUE, cy, CZ_TRUE),
        size=(r,),
        epsilon=eps,
    )
    
    f0 = uniform_inflow_f0(u_in, NZ, NY, NX, dtype, device)
    inlet = InletSpec(ux=u_in)
    outlet = OutletSpec()
    
    f_final, probes = rollout(
        f0, steps, TAU, None,
        soft=geom, inlet=inlet, outlet=outlet,
        return_probes=True, probe_start=probe_start,
    )
    
    drag, cy_wake = compute_observables(f_final, probes, geom, u_in)
    
    return drag, cy_wake, f_final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lr", type=float, default=LR_DEFAULT)
    parser.add_argument("--iters", type=int, default=ITERS_DEFAULT)
    parser.add_argument("--noise", type=float, default=0.0, help="Noise level (fraction)")
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--weight-drag", type=float, default=1.0, help="Weight for drag loss")
    parser.add_argument("--weight-wake", type=float, default=1.0, help="Weight for wake loss")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    
    print(f"Multi-observable geometry inverse problem")
    print(f"Grid: ({NZ}, {NY}, {NX}), wake probe at x={X_PROBE}")
    print(f"Parameters to recover: (cy, r)")
    print(f"True values: cy={CY_TRUE:.2f}, r={R_TRUE:.2f}")
    print(f"Initial guesses: cy={CY_GUESS:.2f}, r={R_GUESS:.2f}")
    print(f"Optimization: {args.iters} iters, lr={args.lr}")
    print(f"Loss weights: drag={args.weight_drag}, wake={args.weight_wake}")
    
    # Generate reference observables
    with torch.no_grad():
        drag_star, cy_wake_star, _ = run_forward(
            CY_TRUE, R_TRUE, U_IN, STEPS, PROBE_START, EPS_SOFT, dtype, device
        )
        print(f"\nReference observables:")
        print(f"  drag* = {drag_star.item():.6e}")
        print(f"  cy_wake* = {cy_wake_star.item():.4f}")
        
        if args.noise > 0:
            noise_drag = args.noise * abs(drag_star.item())
            noise_cy = args.noise * abs(cy_wake_star.item())
            drag_star = drag_star + torch.randn_like(drag_star) * noise_drag
            cy_wake_star = cy_wake_star + torch.randn_like(cy_wake_star) * noise_cy
            print(f"  Added noise: drag ±{noise_drag:.2e}, cy_wake ±{noise_cy:.2e}")
    
    # Create learnable parameters
    cy = torch.tensor(CY_GUESS, dtype=dtype, device=device, requires_grad=True)
    r = torch.tensor(R_GUESS, dtype=dtype, device=device, requires_grad=True)
    
    optim = torch.optim.Adam([cy, r], lr=args.lr)
    
    # Optimization loop
    print("\nStarting optimization...")
    loss0 = None
    best_loss = math.inf
    best_cy, best_r = CY_GUESS, R_GUESS
    history = []
    
    for it in range(args.iters):
        # Cosine LR decay
        for group in optim.param_groups:
            group["lr"] = args.lr * (
                0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * it / args.iters))
            )
        
        # Forward pass
        drag_pred, cy_wake_pred, _ = run_forward(
            cy, r, U_IN, STEPS, PROBE_START, EPS_SOFT, dtype, device
        )
        
        # Multi-observable loss
        loss_drag = ((drag_pred - drag_star) / drag_star) ** 2
        loss_wake = ((cy_wake_pred - cy_wake_star) / cy_wake_star) ** 2
        loss = args.weight_drag * loss_drag + args.weight_wake * loss_wake
        
        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        # Clamp parameters
        with torch.no_grad():
            cy.clamp_(2.0, NY - 2.0)
            r.clamp_(0.5, min(NY, NZ) / 2 - 0.5)
        
        loss_val = float(loss.detach())
        if loss0 is None:
            loss0 = loss_val
        if loss_val < best_loss:
            best_loss = loss_val
            best_cy, best_r = float(cy.detach()), float(r.detach())
        
        history.append({
            "iter": it + 1,
            "loss": loss_val,
            "loss_drag": float(loss_drag.detach()),
            "loss_wake": float(loss_wake.detach()),
            "cy": float(cy.detach()),
            "r": float(r.detach()),
        })
        
        if it == 0 or (it + 1) % args.log_every == 0:
            print(f"  iter {it + 1:4d}  loss={loss_val:.6e}  "
                  f"cy={float(cy.detach()):.4f}  r={float(r.detach()):.4f}")
    
    # Final results
    final_cy, final_r = float(cy.detach()), float(r.detach())
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Initial loss: {loss0:.6e}")
    print(f"Final loss:   {float(loss.detach()):.6e}")
    print(f"Best loss:    {best_loss:.6e}")
    
    errors = {
        "cy": abs(final_cy - CY_TRUE) / CY_TRUE,
        "r": abs(final_r - R_TRUE) / R_TRUE,
    }
    
    print("\nParameter recovery:")
    for name, true_val, pred_val in [
        ("cy", CY_TRUE, final_cy),
        ("r", R_TRUE, final_r),
    ]:
        rel_err = errors[name]
        status = "✓" if rel_err < 0.03 else "✗"
        print(f"  {name:3s}: true={true_val:.4f}, pred={pred_val:.4f}, "
              f"rel_err={rel_err:.2e} {status}")
    
    # Bootstrap confidence intervals
    if args.bootstrap > 0 and args.noise > 0:
        print(f"\nBootstrap confidence intervals ({args.bootstrap} samples):")
        bootstrap_cy = []
        bootstrap_r = []
        
        for b in range(args.bootstrap):
            torch.manual_seed(args.seed + b + 1000)
            
            # Regenerate noisy reference
            with torch.no_grad():
                drag_star_b, cy_wake_star_b, _ = run_forward(
                    CY_TRUE, R_TRUE, U_IN, STEPS, PROBE_START, EPS_SOFT, dtype, device
                )
                noise_drag = args.noise * abs(drag_star_b.item())
                noise_cy = args.noise * abs(cy_wake_star_b.item())
                drag_star_b = drag_star_b + torch.randn_like(drag_star_b) * noise_drag
                cy_wake_star_b = cy_wake_star_b + torch.randn_like(cy_wake_star_b) * noise_cy
            
            # Re-optimize
            cy_b = torch.tensor(CY_GUESS, dtype=dtype, device=device, requires_grad=True)
            r_b = torch.tensor(R_GUESS, dtype=dtype, device=device, requires_grad=True)
            optim_b = torch.optim.Adam([cy_b, r_b], lr=args.lr)
            
            for it in range(args.iters):
                for group in optim_b.param_groups:
                    group["lr"] = args.lr * (
                        0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * it / args.iters))
                    )
                
                drag_pred_b, cy_wake_pred_b, _ = run_forward(
                    cy_b, r_b, U_IN, STEPS, PROBE_START, EPS_SOFT, dtype, device
                )
                
                loss_drag_b = ((drag_pred_b - drag_star_b) / drag_star_b) ** 2
                loss_wake_b = ((cy_wake_pred_b - cy_wake_star_b) / cy_wake_star_b) ** 2
                loss_b = args.weight_drag * loss_drag_b + args.weight_wake * loss_wake_b
                
                optim_b.zero_grad()
                loss_b.backward()
                optim_b.step()
                
                with torch.no_grad():
                    cy_b.clamp_(2.0, NY - 2.0)
                    r_b.clamp_(0.5, min(NY, NZ) / 2 - 0.5)
            
            bootstrap_cy.append(float(cy_b.detach()))
            bootstrap_r.append(float(r_b.detach()))
        
        # Compute 95% CIs
        for name, values, true_val in [
            ("cy", bootstrap_cy, CY_TRUE),
            ("r", bootstrap_r, R_TRUE),
        ]:
            values_arr = np.array(values)
            ci_low = np.percentile(values_arr, 2.5)
            ci_high = np.percentile(values_arr, 97.5)
            covers = "✓" if ci_low <= true_val <= ci_high else "✗"
            print(f"  {name:3s}: 95% CI=[{ci_low:.4f}, {ci_high:.4f}], "
                  f"true={true_val:.4f} {covers}")
    
    # Save results
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "config": {
                "grid": [NZ, NY, NX],
                "x_probe": X_PROBE,
                "iters": args.iters,
                "lr": args.lr,
                "noise": args.noise,
                "bootstrap": args.bootstrap,
                "weight_drag": args.weight_drag,
                "weight_wake": args.weight_wake,
            },
            "true_params": {"cy": CY_TRUE, "r": R_TRUE},
            "initial_guess": {"cy": CY_GUESS, "r": R_GUESS},
            "recovered_params": {"cy": final_cy, "r": final_r},
            "relative_errors": errors,
            "reference_observables": {
                "drag": float(drag_star.detach()),
                "cy_wake": float(cy_wake_star.detach()),
            },
            "loss_history": history,
            "success": all(errors[k] < 0.03 for k in ["cy", "r"]),
        }
        
        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
