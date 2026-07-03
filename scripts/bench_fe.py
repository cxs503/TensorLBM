"""Free-Energy (FE) phase-field model benchmarks.

Tests the Swift et al. binary-fluid formulation:
1. Laplace: static droplet pressure jump vs radius
2. Two-Phase Poiseuille: velocity profile vs analytical solution
"""
import json
import sys
from pathlib import Path

import torch
import torch_sdaa

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tensorlbm.multiphase import free_energy_step, init_free_energy_g
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import stream

CS2 = 1.0 / 3.0
C = torch.tensor([
    [0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]
], dtype=torch.float32)


def run_fe_laplace(
    nx=128, ny=128, radius=20,
    A=0.1, B=0.1, kappa=0.02, Gamma=0.5,
    tau_f=1.0, tau_g=0.7,
    n_steps=5000, output_interval=1000,
    device="cpu",
):
    """FE Laplace test: static circular droplet."""
    print(f"\n  FE Laplace: nx={nx}, R={radius}, kappa={kappa}, A={A}, B={B}")
    
    device_t = torch.device(device)
    c_dev = C.to(device_t)
    
    # Phase field: +1 inside droplet, -1 outside
    ys = torch.arange(ny, dtype=torch.float32, device=device_t)
    xs = torch.arange(nx, dtype=torch.float32, device=device_t)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cy, cx = ny / 2.0, nx / 2.0
    r_field = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    phi = torch.where(r_field <= radius, 0.9, -0.9)  # not exactly ±1 to avoid sharp interface
    
    # Init distributions
    ux = torch.zeros_like(phi)
    uy = torch.zeros_like(phi)
    rho = torch.ones_like(phi)
    f = equilibrium(rho, ux, uy)
    g = init_free_energy_g(phi, ux, uy)
    
    diagnostics = []
    
    for step in range(1, n_steps + 1):
        f, g = free_energy_step(
            f, g, tau_f=tau_f, tau_g=tau_g,
            A=A, B=B, kappa=kappa, Gamma=Gamma,
        )
        f = stream(f)
        g = stream(g)
        
        if step % output_interval == 0:
            phi_cur = g.sum(dim=0)
            # Measure max velocity (spurious currents)
            rho_cur = f.sum(dim=0).clamp(min=1e-12)
            cx_d = c_dev[:, 0].view(9, 1, 1)
            cy_d = c_dev[:, 1].view(9, 1, 1)
            ux_cur = (f * cx_d).sum(0) / rho_cur
            uy_cur = (f * cy_d).sum(0) / rho_cur
            max_u = float(torch.sqrt(ux_cur ** 2 + uy_cur ** 2).max().item())
            
            # Pressure jump
            inner = r_field <= radius * 0.5
            outer = r_field >= radius * 1.5
            p_in = float((CS2 * rho_cur[inner]).mean().item()) if inner.any() else 0.0
            p_out = float((CS2 * rho_cur[outer]).mean().item()) if outer.any() else 0.0
            dp = p_in - p_out
            
            diagnostics.append({"step": step, "delta_p": dp, "max_u": max_u})
            print(f"    step={step:5d}  ΔP={dp:.6f}  max|u|={max_u:.4e}")
    
    # Final metrics
    final = diagnostics[-1] if diagnostics else {"delta_p": 0, "max_u": 0}
    result = {
        "benchmark": "fe_laplace",
        "config": {"nx": nx, "ny": ny, "radius": radius, "A": A, "B": B,
                    "kappa": kappa, "Gamma": Gamma, "tau_f": tau_f, "tau_g": tau_g,
                    "n_steps": n_steps},
        "delta_p": final["delta_p"],
        "max_spurious_u": final["max_u"],
        "diagnostics": diagnostics,
    }
    return result


def analytical_two_phase_poiseuille(y, y_int, u_max1, u_max2, nu1, nu2, L_y):
    """Piecewise parabolic profile for two-phase channel flow.
    
    Phase 1: bottom, Phase 2: top, interface at y_int.
    """
    u = torch.zeros_like(y)
    
    # Domain center
    half = L_y / 2
    
    # Phase 1: [0, y_int] — quadratic passing through u(y_int)=v_int, u'(half)=0 if y_int < half
    # Phase 2: [y_int, L_y] — likewise
    
    # Let's use the standard solution for body-force-driven two-layer Poiseuille
    # with equal density, different viscosities
    # u1(y) = A1 y^2 + B1 y + C1, u2(y) = A2 y^2 + B2 y + C2
    
    G = u_max1  # body force proxy — we'll scale
    
    # Actually let's use a simpler approach: compute from known params
    # Use the exact solution for two-layer Poiseuille with body force G
    # u'' = -G/nu in each layer
    
    # Boundary: u(0)=u(L_y)=0, u and nu u' continuous at y=y_int
    G_val = 1.0
    A1 = -G_val / (2 * nu1)
    A2 = -G_val / (2 * nu2)
    
    # Match at interface: u1(y_int) = u2(y_int), nu1 u1'(y_int) = nu2 u2'(y_int)
    B2 = (A1 * y_int + A2 * (L_y - y_int)) / (nu1/nu2 * (L_y - y_int) + y_int)
    # Actually, let's solve properly
    # u1 = A1 y^2 + B1 y,  u1(0)=0 → C1=0
    # u2 = A2 y^2 + B2 y + C2,  u2(L_y)=0 → C2 = -A2 L_y^2 - B2 L_y
    
    # u1(y_int) = u2(y_int):
    # A1 y_int^2 + B1 y_int = A2 y_int^2 + B2 y_int + C2
    # = A2 y_int^2 + B2 y_int - A2 L_y^2 - B2 L_y
    # A1 y_int^2 + B1 y_int = A2 (y_int^2 - L_y^2) + B2 (y_int - L_y)  ... (1)
    
    # nu1 u1'(y_int) = nu2 u2'(y_int):
    # nu1 (2 A1 y_int + B1) = nu2 (2 A2 y_int + B2)  ... (2)
    
    B1_factor = y_int
    B2_factor = y_int - L_y
    
    # From (1): B1 = [A2 (y_int^2 - L_y^2) + B2 (y_int - L_y) - A1 y_int^2] / y_int
    # From (2): B1 = [nu2 (2 A2 y_int + B2) / nu1 - 2 A1 y_int]
    
    # Equate and solve for B2
    # [A2 (y_int^2 - L_y^2) + B2 (y_int - L_y) - A1 y_int^2] / y_int = nu2/nu1 (2 A2 y_int + B2) - 2 A1 y_int
    
    # Multiply by y_int:
    # A2 (y_int^2 - L_y^2) + B2 (y_int - L_y) - A1 y_int^2 = y_int * nu2/nu1 (2 A2 y_int + B2) - 2 A1 y_int^2
    # = 2 A2 y_int^2 nu2/nu1 + B2 y_int nu2/nu1 - 2 A1 y_int^2
    
    # B2 [(y_int - L_y) - y_int nu2/nu1] = 2 A2 y_int^2 nu2/nu1 - 2 A1 y_int^2 - A2 (y_int^2 - L_y^2) + A1 y_int^2
    # = 2 A2 y_int^2 nu2/nu1 - A1 y_int^2 - A2 (y_int^2 - L_y^2)
    
    # B2 = [2 A2 y_int^2 nu2/nu1 - A1 y_int^2 - A2 (y_int^2 - L_y^2)] / [(y_int - L_y) - y_int nu2/nu1]
    
    denom_B2 = (y_int - L_y) - y_int * nu2 / nu1
    num_B2 = 2 * A2 * y_int**2 * nu2 / nu1 - A1 * y_int**2 - A2 * (y_int**2 - L_y**2)
    B2 = num_B2 / denom_B2
    B1 = (A2 * (y_int**2 - L_y**2) + B2 * (y_int - L_y) - A1 * y_int**2) / y_int
    
    C2 = -A2 * L_y**2 - B2 * L_y
    
    # Build profile
    mask1 = y <= y_int
    mask2 = y > y_int
    u = torch.where(mask1, A1 * y**2 + B1 * y,
                   A2 * y**2 + B2 * y + C2)
    
    # Normalize
    u_max = u.max()
    if u_max > 0:
        u = u / u_max
    
    return u


def run_fe_poiseuille(
    nx=100, ny=40,
    A=0.1, B=0.1, kappa=0.02, Gamma=0.5,
    tau_f_heavy=0.8, tau_f_light=1.2,
    tau_g=0.7,
    gx=1e-5,  # tiny body force
    n_steps=10000, output_interval=2000,
    device="cpu",
):
    """FE two-phase Poiseuille test."""
    print(f"\n  FE Poiseuille: nx={nx}, ny={ny}, tau_h={tau_f_heavy}, tau_l={tau_f_light}")
    
    device_t = torch.device(device)
    
    # Interface at ny/2
    y_int = ny // 2
    
    # Phase field: +1 bottom, -1 top
    phi = torch.ones((ny, nx), device=device_t)
    phi[y_int:, :] = -1.0
    
    ux = torch.zeros_like(phi)
    uy = torch.zeros_like(phi)
    rho = torch.ones_like(phi)
    
    f = equilibrium(rho, ux, uy)
    g = init_free_energy_g(phi, ux, uy)
    
    diagnostics = []
    final_uy = None
    
    for step in range(1, n_steps + 1):
        # Use different tau_f per row for viscosity contrast
        # Bottom (heavy): tau_f_heavy, Top (light): tau_f_light
        f_bot = f[:, :y_int, :]
        g_bot = g[:, :y_int, :]
        f_top = f[:, y_int:, :]
        g_top = g[:, y_int:, :]
        
        f_bot_new, g_bot_new = free_energy_step(
            f_bot, g_bot, tau_f=tau_f_heavy, tau_g=tau_g,
            A=A, B=B, kappa=kappa, Gamma=Gamma, gx=gx,
        )
        f_top_new, g_top_new = free_energy_step(
            f_top, g_top, tau_f=tau_f_light, tau_g=tau_g,
            A=A, B=B, kappa=kappa, Gamma=Gamma, gx=gx,
        )
        
        f = torch.cat([f_bot_new, f_top_new], dim=1)
        g = torch.cat([g_bot_new, g_top_new], dim=1)
        
        f = stream(f)
        g = stream(g)
        
        if step % output_interval == 0:
            rho_cur = f.sum(dim=0).clamp(min=1e-12)
            # Velocity profile: average over x
            c_dev = C.to(device_t)
            cx_d = c_dev[:, 0].view(9, 1, 1)
            ux_cur = (f * cx_d).sum(0) / rho_cur
            ux_profile = ux_cur.mean(dim=1)  # (ny,)
            
            # Analytical
            y = torch.arange(ny, dtype=torch.float32, device=device_t)
            nu_h = CS2 * (tau_f_heavy - 0.5)
            nu_l = CS2 * (tau_f_light - 0.5)
            u_anal = analytical_two_phase_poiseuille(y, y_int, 1.0, 1.0, nu_h, nu_l, ny)
            
            # Scale numerical to match analytical max
            scale = u_anal.max().item() / (ux_profile.max().item() + 1e-12)
            ux_scaled = ux_profile * scale
            
            l2_err = float(torch.sqrt(torch.mean((ux_scaled - u_anal) ** 2)).item())
            diagnostics.append({"step": step, "l2_error": l2_err})
            print(f"    step={step:5d}  L2_err={l2_err:.6f}")
            final_uy = ux_profile
    
    final = diagnostics[-1] if diagnostics else {"l2_error": float("nan")}
    nu_h = CS2 * (tau_f_heavy - 0.5)
    nu_l = CS2 * (tau_f_light - 0.5)
    
    result = {
        "benchmark": "fe_poiseuille",
        "config": {"nx": nx, "ny": ny, "A": A, "B": B, "kappa": kappa, "Gamma": Gamma,
                    "tau_f_heavy": tau_f_heavy, "tau_f_light": tau_f_light, "tau_g": tau_g,
                    "gx": gx, "n_steps": n_steps},
        "l2_error": final["l2_error"],
        "viscosity_ratio": nu_h / nu_l if nu_l > 0 else 1.0,
        "nu_heavy": nu_h,
        "nu_light": nu_l,
        "diagnostics": diagnostics,
    }
    return result


if __name__ == "__main__":
    device = "sdaa" if torch.sdaa.is_available() else "cpu"
    print(f"Device: {device}")
    
    results = {}
    
    # Laplace test with multiple kappa values
    for kappa in [0.01, 0.02, 0.05]:
        print(f"\n{'='*50}")
        print(f"FE Laplace  kappa={kappa}")
        print(f"{'='*50}")
        r = run_fe_laplace(kappa=kappa, n_steps=5000, device=device)
        results[f"laplace_k{kappa}"] = r
    
    # Poiseuille test
    print(f"\n{'='*50}")
    print(f"FE Poiseuille")
    print(f"{'='*50}")
    r = run_fe_poiseuille(n_steps=10000, device=device)
    results["poiseuille"] = r
    
    # Summary
    print(f"\n{'='*60}")
    print("FE Model Summary")
    print(f"{'='*60}")
    for name, res in results.items():
        if "laplace" in name:
            print(f"  {name}: ΔP={res['delta_p']:.6f}, max_u={res['max_spurious_u']:.4e}")
        else:
            print(f"  {name}: L2_err={res['l2_error']:.6f}, ν_ratio={res['viscosity_ratio']:.2f}")
    
    Path("outputs/fe_benchmarks").mkdir(parents=True, exist_ok=True)
    out_path = Path("outputs/fe_benchmarks/fe_benchmark.json")
    out_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"\nSaved → {out_path}")
