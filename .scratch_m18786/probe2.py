import math, time, torch, numpy as np
torch.set_num_threads(2)
from tensorlbm.wetting.two_impl_contact_angle import (TwoImplConfig, wall_ghost_rows,
    grad_laplacian_ghost, _cap_centre_y)
from tensorlbm.wetting.wall_order_parameter_2d import (WallOrderWettingConfig,
    initialise_drop_on_wall, contact_angle_from_field)
cfg = TwoImplConfig(ny=56, nx=80, interface_width=3.0, surface_tension=0.05)
a, kappa = cfg.a, cfg.kappa
wcfg = WallOrderWettingConfig(ny=56, nx=80, interface_width=3.0, surface_tension=0.05)
for arm, deg in [("geometric_ghost",90.0),("geometric_ghost",110.0),("neumann_flux",110.0)]:
    th = math.radians(deg)
    psi = initialise_drop_on_wall(wcfg, radius=16.0, centre_y=_cap_centre_y(th,16.0))["g"].sum(0)
    t0=time.time()
    for blk in range(1,13):
        for _ in range(2000):
            gl, gh = wall_ghost_rows(arm, psi, th, kappa, cfg.surface_tension)
            gx, gy, lap = grad_laplacian_ghost(psi, gl, gh)
            mu = a*psi*(psi**2-1.0) - kappa*lap
            psi = psi - 1.0*(mu-mu.mean())
        m = contact_angle_from_field(psi, psi_wall=0.0, interface_width=3.0)
        print(f"{arm:16s} {deg:5.1f} blk={blk:2d} it={blk*2000:6d} cap={m['theta_cap_deg']:7.2f} circ={m['theta_circle_deg']:7.2f} apex={m['apex_height']:6.2f} Rc={m['R_curv']:6.2f} lay={m['layer_fraction']:.3f} t={time.time()-t0:.0f}", flush=True)
    np.savez(f".scratch_m18786/{arm}_{int(deg)}.npz", psi=psi.numpy())
