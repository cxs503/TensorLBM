import sys, math, time
sys.path.insert(0, '/root/tp-agent/work/m9483/src')
import numpy as np
from tensorlbm.invariance.geometry import tilted_channel, channel_boundary_cells, poiseuille_profile
from tensorlbm.invariance.solver import D2Q9Solver, VelocityDirichlet

def run(theta, width, u_mean=0.01, nu=0.1, steps_factor=30.0, shift_mode="x", bodyforce=False, trim=0.25):
    th = math.radians(theta)
    nx = int(round(6*width)); ny = int(round(nx*math.tan(th) + width/math.cos(th) + 2*width)) + 2
    geom = tilted_channel(nx, ny, theta, width)
    tau = 0.5 + 3*nu
    s = D2Q9Solver(geom, tau=tau)
    ux_ex, uy_ex = poiseuille_profile(geom, u_mean)
    s.initialise(ux_ex, uy_ex)
    inlet, outlet = channel_boundary_cells(geom)
    ax, ay = geom.axis
    if shift_mode == "x":
        sh_in, sh_out = (0, 1), (0, -1)
    else:
        def nearest(v):
            best = max([(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)],
                       key=lambda c: c[0]*v[0]+c[1]*v[1])
            return best
        ci = nearest((ax, ay)); co = (-ci[0], -ci[1])
        sh_in, sh_out = (ci[1], ci[0]), (co[1], co[0])
    s.set_velocity_dirichlet([VelocityDirichlet(inlet, ux_ex, uy_ex, sh_in),
                              VelocityDirichlet(outlet, ux_ex, uy_ex, sh_out)])
    G = 12.0*nu*u_mean/(width*width)
    fx = np.full((ny, nx), G*ax); fy = np.full((ny, nx), G*ay)
    steps = int(steps_factor*width*width)
    for _ in range(steps):
        if bodyforce:
            s.step(fx, fy)
        else:
            s.step()
    ux, uy = s.velocity()
    ex = (ux-ux_ex); ey = (uy-uy_ex)
    mask = geom.fluid.copy()
    k = max(1, int(trim*nx)); mask[:, :k] = False; mask[:, -k:] = False
    l2 = float(np.sqrt(np.mean((ex**2+ey**2)[mask])))
    sdf = geom.sdf
    near = mask & (sdf < 2.0)
    mid = mask & (sdf >= 2.0)
    def rms(m):
        return float(np.sqrt(np.mean((ex**2+ey**2)[m]))) if m.any() else float('nan')
    return dict(nx=nx, ny=ny, l2=l2, l2_near=rms(near), l2_mid=rms(mid),
                frac_near=float(near.sum()/mask.sum()))

for mode in ["x", "axis"]:
    for bf in [False, True]:
        print("=== shift", mode, "bodyforce", bf, flush=True)
        for theta in [0.0, 15.0, 30.0, 45.0]:
            out = []
            for w in [8, 12, 16]:
                t0 = time.time()
                r = run(theta, w, shift_mode=mode, bodyforce=bf)
                out.append(r)
                print(f"  th={theta:4g} H={w:3d} l2={r['l2']:.3e} near={r['l2_near']:.3e} mid={r['l2_mid']:.3e} fracnear={r['frac_near']:.3f} ({time.time()-t0:.1f}s)", flush=True)
            e = [o['l2'] for o in out]; ws=[8,12,16]
            orders=[math.log(e[i]/e[i+1])/math.log(ws[i+1]/ws[i]) for i in range(len(e)-1)]
            print(f"   orders={['%.2f'%o for o in orders]}", flush=True)