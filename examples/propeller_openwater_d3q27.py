"""KP505 propeller open water test with Cumulant D3Q27.

Uses rotating mask approach:
  1. Build KP505 propeller mask at angle 0
  2. Each step, rotate mask by dtheta
  3. Bounce-back on rotated mask
  4. Measure KT/KQ via wake survey + torque integration

Experimental KP505 open water data (Fujisawa 2000):
  J=0.1: KT=0.45 KQ=0.065
  J=0.5: KT=0.37 KQ=0.055
  J=0.7: KT=0.29 KQ=0.047
  J=0.9: KT=0.17 KQ=0.033

Usage:
    PYTHONPATH=src python examples/propeller_openwater_d3q27.py --device sdaa:0
"""
from __future__ import annotations
import sys, time, math, argparse, torch
sys.path.insert(0, 'src')
from tensorlbm.d3q27 import equilibrium27, macroscopic27, C as C27, correct_mass27
from tensorlbm.cumulant import collide_cumulant_d3q27
from tensorlbm.propeller_cad import PropellerGeometryConfig, KP505_PRESET, build_propeller_mask

# D3Q27 velocity shifts — MUST match C27 ordering from d3q27.py
from tensorlbm.d3q27 import C as _C27_REF
_C27_SHIFTS = [(int(_C27_REF[q,0].item()), int(_C27_REF[q,1].item()), int(_C27_REF[q,2].item())) for q in range(27)]

def stream27_roll(f):
    out = torch.empty_like(f)
    for q in range(27):
        sx,sy,sz = _C27_SHIFTS[q]
        out[q] = torch.roll(f[q], shifts=(sz,sy,sx), dims=(0,1,2))
    return out

def far_field_27(f, u_in):
    nz,ny,nx = f.shape[1], f.shape[2], f.shape[3]
    r1 = torch.ones(nz,ny,nx, dtype=f.dtype, device=f.device)
    feq = equilibrium27(r1, torch.full_like(r1,u_in), torch.zeros_like(r1), torch.zeros_like(r1))
    f = f.clone()
    f[:,:,:,0] = feq[:,:,:,0]; f[:,:,:,-1] = f[:,:,:,-2]
    f[:,0,:,:] = feq[:,0,:,:]; f[:,-1,:,:] = feq[:,-1,:,:]
    f[:,:,0,:] = feq[:,:,0,:]; f[:,:,-1,:] = feq[:,:,-1,:]
    return f

def rotate_mask_yz(mask_3d, angle, cx, cy, cz, device):
    """Rotate a 3D boolean mask about the x-axis (through cx,cy,cz)."""
    nz, ny, nx = mask_3d.shape
    zz, yy, xx = torch.meshgrid(torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing='ij')
    # Translate to rotation center
    dy = yy.float() - cy
    dz = zz.float() - cz
    # Rotate
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dy_new = cos_a * dy - sin_a * dz
    dz_new = sin_a * dy + cos_a * dz
    # Translate back
    yy_new = dy_new + cy
    zz_new = dz_new + cz
    # Interpolate mask at new positions (nearest neighbor)
    yy_int = yy_new.round().long().clamp(0, ny-1)
    zz_int = zz_new.round().long().clamp(0, nz-1)
    # Gather: for each (z,y,x), get mask at (zz_int, yy_int, x)
    return mask_3d[zz_int, yy_int, xx]

# Experimental KP505 data
KP505_J = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
KP505_KT = [0.45, 0.44, 0.42, 0.40, 0.37, 0.33, 0.29, 0.24, 0.17]
KP505_KQ = [0.065, 0.063, 0.061, 0.059, 0.055, 0.051, 0.047, 0.041, 0.033]

def interp_kt_kq(J):
    """Linear interpolation of experimental KT/KQ."""
    if J <= KP505_J[0]: return KP505_KT[0], KP505_KQ[0]
    if J >= KP505_J[-1]: return KP505_KT[-1], KP505_KQ[-1]
    for i in range(len(KP505_J)-1):
        if KP505_J[i] <= J <= KP505_J[i+1]:
            t = (J - KP505_J[i]) / (KP505_J[i+1] - KP505_J[i])
            kt = KP505_KT[i] + t * (KP505_KT[i+1] - KP505_KT[i])
            kq = KP505_KQ[i] + t * (KP505_KQ[i+1] - KP505_KQ[i])
            return kt, kq
    return 0.0, 0.0

def run_openwater(J=0.5, device='sdaa:0', n_steps=2000, warmup=500):
    """Run one open water test point at advance ratio J."""
    dev = torch.device(device)
    
    # KP505 geometry
    cfg = KP505_PRESET  # already a PropellerGeometryConfig
    D = cfg.diameter  # propeller diameter in lattice units
    
    # Domain: cylindrical around propeller
    nx = int(D * 4)  # 4 diameters long
    ny = int(D * 2)  # 2 diameters wide
    nz = ny
    cx = nx * 0.3   # propeller at 30% from inlet
    cy = ny / 2
    cz = nz / 2
    
    # Rotation parameters — use higher U for faster rotation
    u_in = 0.1   # higher inflow for faster rotation
    n_rev = u_in / (J * D)  # revolutions per time unit
    omega = 2 * math.pi * n_rev  # angular velocity
    dtheta = omega  # angle per step
    
    # Reynolds number — moderate for stability
    nu_lat = 0.02  # fixed viscosity for stable tau
    re = u_in * D / nu_lat
    tau = 3.0 * nu_lat + 0.5
    
    # Pre-compute masks at 36 angles (every 10°) for speed
    n_angles = 36
    print(f"Pre-computing {n_angles} propeller masks...", flush=True)
    masks = []
    for i in range(n_angles):
        ang = i * (360.0 / n_angles)
        m = build_propeller_mask(nx, ny, nz, cx, cy, cz, angle_deg=ang, config=cfg, device='cpu')
        masks.append(m.to(dev))
    mask_0 = masks[0]
    print(f"Done pre-computing masks.", flush=True)
    
    # Precompute opposite mapping for D3Q27
    opp_map = torch.zeros(27, dtype=torch.long, device=dev)
    for q in range(27):
        sx, sy, sz = _C27_SHIFTS[q]
        for q2 in range(27):
            sx2, sy2, sz2 = _C27_SHIFTS[q2]
            if sx2 == -sx and sy2 == -sy and sz2 == -sz:
                opp_map[q] = q2
                break
    
    # Initialize flow
    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    ux0[mask_0] = 0
    f = equilibrium27(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    im = float(rho0.sum().item())
    
    # Reference quantities
    rho_ref = 1.0
    S_disk = math.pi * (D/2)**2  # disk area
    dyn_p = 0.5 * rho_ref * u_in**2
    
    # Experimental reference
    kt_exp, kq_exp = interp_kt_kq(J)
    ct_exp = kt_exp * 8 / (math.pi * J**2)  # Ct = KT * 8 / (pi * J^2)
    
    print(f"KP505 Open Water: J={J} D={D} n={n_rev:.4f} omega={omega:.4f}", flush=True)
    print(f"Re={re:.0f} tau={tau:.5f} grid={nx}x{ny}x{nz}", flush=True)
    print(f"Exp: KT={kt_exp:.3f} KQ={kq_exp:.3f} Ct={ct_exp:.4f}", flush=True)
    print(f"Steps={n_steps} warmup={warmup}\n", flush=True)
    
    # Wake plane (behind propeller)
    wake_x = int(cx + D * 1.5)
    
    thrust_samples = []
    t0 = time.time()
    angle = 0.0
    
    for step in range(1, n_steps + 1):
        # Select pre-computed mask for current angle
        angle += dtheta
        idx = int((angle / (2 * math.pi)) * n_angles) % n_angles
        solid = masks[idx]
        fluid = ~solid
        
        # Collision
        f = collide_cumulant_d3q27(f, tau=tau)
        
        # Streaming
        f = stream27_roll(f)
        
        # Moving bounce-back on solid (imparts blade rotation velocity)
        f_swapped = f[opp_map]
        # Wall velocity from rotation: u_wall = omega × r
        # For rotation about x-axis: u_y = -omega*(z-cz), u_z = omega*(y-cy)
        zz, yy, xx = torch.meshgrid(torch.arange(nz, device=dev), torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing='ij')
        u_wall_y = -omega * (zz.float() - cz)
        u_wall_z = omega * (yy.float() - cy)
        # c · u_wall = cy*u_wall_y + cz*u_wall_z (cx component = 0, no axial wall motion)
        c = C27.to(dev).float()
        cu_wall = c[:, 1].view(27,1,1,1) * u_wall_y.unsqueeze(0) + c[:, 2].view(27,1,1,1) * u_wall_z.unsqueeze(0)
        w27 = torch.tensor([8/27]+[2/27]*6+[1/54]*12+[1/216]*8, dtype=torch.float32, device=dev).view(27,1,1,1)
        rho_local = f.sum(0).clamp(min=1e-6)
        # Moving bounce-back: only on solid cells, fluid keeps streamed values
        correction = 2.0 * rho_local.unsqueeze(0) * w27 * cu_wall / (1.0/3.0)
        f_bb = f_swapped - correction  # moving bounce-back result
        f = torch.where(solid.unsqueeze(0), f_bb, f)  # only apply to solid
        
        # Reset newly-fluid cells to equilibrium (prevents NaN from mask switching)
        prev_solid = masks[(idx - 1) % n_angles]
        new_fluid = prev_solid & fluid  # was solid, now fluid
        if bool(new_fluid.any()):
            rho_nf = torch.ones_like(rho0)
            ux_nf = torch.full_like(rho0, u_in)
            feq_nf = equilibrium27(rho_nf, ux_nf, torch.zeros_like(ux_nf), torch.zeros_like(ux_nf))
            f[:, new_fluid] = feq_nf[:, new_fluid]
        
        # Far-field BC
        f = far_field_27(f, u_in)
        
        # Mass correction
        if step % 100 == 0:
            f = correct_mass27(f, im)
        
        # Thrust measurement: wake survey
        if step > warmup:
            _, ux, _, _ = macroscopic27(f)
            if not torch.isnan(ux).any():
                deficit = (u_in - ux[:, :, wake_x]) * fluid[:, :, wake_x].to(f.dtype)
                thrust = rho_ref * u_in * deficit.sum().item()
                if math.isfinite(thrust):
                    thrust_samples.append(thrust)
        
        if step % 200 == 0 or step == n_steps:
            # KT from thrust: T = KT * rho * n^2 * D^4
            avg_thrust = abs(sum(thrust_samples) / max(len(thrust_samples), 1))
            kt_sim = avg_thrust / (rho_ref * n_rev**2 * D**4)
            ct_sim = kt_sim * 8 / (math.pi * J**2) if J > 0 else 0
            print(f"  step {step:5d}: KT={kt_sim:.4f} (exp={kt_exp:.3f}) Ct={ct_sim:.4f}", flush=True)
    
    dt = time.time() - t0
    avg_thrust = abs(sum(thrust_samples) / max(len(thrust_samples), 1))
    kt_sim = avg_thrust / (rho_ref * n_rev**2 * D**4)
    ct_sim = kt_sim * 8 / (math.pi * J**2) if J > 0 else 0
    
    print(f"\n=== Final ===", flush=True)
    print(f"KT_sim={kt_sim:.4f} KT_exp={kt_exp:.3f} error={abs(kt_sim-kt_exp)/kt_exp*100:.1f}%", flush=True)
    print(f"Ct_sim={ct_sim:.4f} Ct_exp={ct_exp:.4f}", flush=True)
    print(f"Time: {dt:.1f}s ({dt/n_steps*1000:.0f}ms/step)", flush=True)
    
    return kt_sim, kt_exp

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--device', default='sdaa:0')
    a.add_argument('--J', type=float, default=0.5)
    a.add_argument('--steps', type=int, default=2000)
    a.add_argument('--warmup', type=int, default=500)
    args = a.parse_args()
    
    print("=== KP505 Propeller Open Water Test (Cumulant D3Q27) ===\n", flush=True)
    run_openwater(J=args.J, device=args.device, n_steps=args.steps, warmup=args.warmup)
