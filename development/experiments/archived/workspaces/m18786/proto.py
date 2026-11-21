import math, time, sys
import torch
from tensorlbm.wetting.wall_order_parameter_2d import (
    WallOrderWettingConfig, free_energy_parameters, initialise_drop_on_wall,
    interface_points, _circle_fit, _zero_crossing_1d, theta_from_cap_area)

def pad_ghost(psi, gl, gt):
    return torch.cat((gl.unsqueeze(0), psi, gt.unsqueeze(0)), dim=0)

def grad_lap_ghost(psi, gl, gt):
    p = pad_ghost(psi, gl, gt)
    north, south = p[2:], p[:-2]
    east, west = torch.roll(psi,-1,1), torch.roll(psi,1,1)
    ne, nw = torch.roll(north,-1,1), torch.roll(north,1,1)
    se, sw = torch.roll(south,-1,1), torch.roll(south,1,1)
    gx = (4.0*(east-west) + (ne+se-nw-sw))/12.0
    gy = (4.0*(north-south) + (ne+nw-se-sw))/12.0
    lap = (4.0*(east+west+north+south) + (ne+nw+sw+se) - 20.0*psi)/6.0
    return gx, gy, lap

def ghosts(psi, arm, theta, kappa, sigma, psi_top=-1.0):
    ny, nx = psi.shape
    if arm == "dirichlet":
        c = 0.5*(3*math.cos(theta) - math.cos(theta)**3)  # inverse of cos th=(3p-p^3)/2
        return torch.full((nx,), c, dtype=psi.dtype), torch.full((nx,), psi_top, dtype=psi.dtype)
    if arm == "neumann":
        # kappa dpsi/dn|_wall = (sigma/2)*(3/2)*(1-psi_s^2) * cos(theta)
        flux = 0.75*sigma*math.cos(theta)
        lo = psi[0]
        for _ in range(3):
            ps = 0.5*(psi[0]+lo)
            lo = psi[0] - flux*(1.0-ps**2)/kappa
        hi = psi[-1] - flux*(1.0-psi_top**2)/kappa
        return lo, hi
    if arm == "geometric":
        dlo = (torch.roll(psi[0],-1)-torch.roll(psi[0],1))/2.0
        dhi = (torch.roll(psi[-1],-1)-torch.roll(psi[-1],1))/2.0
        ct = math.cos(theta)/math.sin(theta)
        return psi[0] + ct*dlo.abs(), psi[-1] + ct*dhi.abs()
    raise ValueError(arm)

def relax(arm, theta, xi=3.0, sigma=0.01, ny=64, nx=96, R=16.0, max_iter=40000,
          check_every=200, tol=1e-9, omega=0.25, psi_top=-1.0):
    cfg = WallOrderWettingConfig(ny=ny, nx=nx, interface_width=xi, surface_tension=sigma)
    a, kappa = free_energy_parameters(sigma, xi)
    psi = initialise_drop_on_wall(cfg, radius=R, centre_y=-0.5 + R*(1-math.cos(theta)) )["g"].sum(0)
    # centre so that initial cap already satisfies theta: yc = wall - R cos(theta)... wall at -0.5
    yc = -0.5 + R*math.cos(theta) if False else None
    prev = psi.clone(); converged=False; it=0; drift=float('inf')
    while it < max_iter:
        for _ in range(check_every):
            gl, gt = ghosts(psi, arm, theta, kappa, sigma, psi_top)
            gx, gy, lap = grad_lap_ghost(psi, gl, gt)
            mu = a*psi*(psi**2-1.0) - kappa*lap
            psi = psi - omega*(mu-mu.mean())
        it += check_every
        drift = float((psi-prev).abs().max()); prev = psi.clone()
        if drift < tol: converged=True; break
    return psi, it, converged, drift, a, kappa

def measure(psi, xi, wall_y=-0.5, nx=96):
    # layer profile from edges
    edge = 6
    film = torch.cat((psi[:, :edge], psi[:, -edge:]), 1).mean(1)
    layer = float(((film+1)*0.5).sum())
    area_tot = float(((psi+1)*0.5).sum())
    area_exc = area_tot - layer*nx
    j = nx//2
    apex = _zero_crossing_1d(psi[:, j], last=True)
    h = apex - wall_y
    th_cap = math.degrees(theta_from_cap_area(area_exc, h))
    pts = interface_points(psi)
    th_circ = float('nan')
    if pts.shape[0] >= 20:
        mask = pts[:,0] > 1.5*xi
        use = pts[mask] if int(mask.sum())>=20 else pts
        yc, xc, R = _circle_fit(use)
        th_circ = math.degrees(math.acos(max(-1.,min(1.,(yc-wall_y)/R))))
    return dict(theta_cap=th_cap, theta_circle=th_circ, apex=h, layer_frac=layer*nx/area_tot)

if __name__ == "__main__":
    for arm in ["dirichlet","neumann","geometric"]:
        for deg in [60.0, 90.0, 120.0]:
            th = math.radians(deg)
            t0=time.time()
            psi, it, conv, drift, a, kappa = relax(arm, th)
            m = measure(psi, 3.0)
            print(f"{arm:10s} th={deg:5.1f} it={it:6d} conv={conv} drift={drift:.1e} "
                  f"cap={m['theta_cap']:7.2f} circ={m['theta_circle']:7.2f} apex={m['apex']:.2f} layer={m['layer_frac']:.3f} t={time.time()-t0:.1f}s")
