import math, time, sys, torch
torch.set_num_threads(2)
from tensorlbm.wetting.two_impl_contact_angle import run_arm, TwoImplConfig
cfg = TwoImplConfig(ny=56, nx=80, interface_width=3.0, surface_tension=0.05)
for arm in ["dirichlet","neumann_flux","geometric_ghost"]:
    for deg in [70.0, 110.0]:
        t0=time.time()
        r = run_arm(arm, math.radians(deg), cfg=cfg, max_iter=8000, omega=0.4, tol=1e-9)
        print(f"{arm:16s} {deg:5.1f} -> cap={r['theta_cap_deg']:7.2f} circ={r['theta_circle_deg']:7.2f} rb={r['wall_angle_readback_deg']:7.2f} it={r['iterations']} drift={r['drift']:.1e} lay={r['layer_fraction']:.3f} meas={r['cap_measurable']} t={time.time()-t0:.1f}", flush=True)
