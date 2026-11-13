import math, time, torch, sys
from multiprocessing import Pool
def work(args):
    arm, deg, xi = args
    import torch; torch.set_num_threads(1)
    from tensorlbm.wetting.two_impl_contact_angle import run_arm, TwoImplConfig
    cfg = TwoImplConfig(ny=56, nx=80, interface_width=xi, surface_tension=0.05)
    t0=time.time()
    r = run_arm(arm, math.radians(deg), cfg=cfg, max_iter=12000, omega=1.0, tol=1e-9)
    ref = deg
    ec = abs(r['theta_cap_deg']-ref)/ref*100 if r['theta_cap_deg']==r['theta_cap_deg'] else float('nan')
    eci = abs(r['theta_circle_deg']-ref)/ref*100 if r['theta_circle_deg']==r['theta_circle_deg'] else float('nan')
    return (f"{arm:16s} th={deg:5.1f} xi={xi} cap={r['theta_cap_deg']:7.2f}({ec:5.1f}%) circ={r['theta_circle_deg']:7.2f}({eci:5.1f}%) gap={r['estimator_gap_deg']:5.2f} rb={r['wall_angle_readback_deg']:6.2f} it={r['iterations']:6d} drift={r['drift']:.0e} gas={r['gas_supersaturation']:+.3f} rms={r['circle_rms']:.3f} meas={r['cap_measurable']} t={time.time()-t0:.0f}s")
if __name__ == "__main__":
    jobs=[(a,d,3.0) for a in ["dirichlet","neumann_flux","geometric_ghost"] for d in [45.0,70.0,90.0,110.0,135.0]]
    with Pool(8) as p:
        for line in p.imap(work, jobs): print(line, flush=True)
