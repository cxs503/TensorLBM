import sys, math, time, os
sys.path.insert(0, '/root/tp-agent/work/m9483/src')
import numpy as np
from multiprocessing import Pool
from tensorlbm.invariance.geometry import tilted_channel, channel_boundary_cells, poiseuille_profile
from tensorlbm.invariance.solver import D2Q9Solver, VelocityDirichlet


def run(cfg):
    theta, width, mode, bf, u_mean, nu, steps_factor, trim = cfg
    th = math.radians(theta)
    nx = int(round(6*width)); ny = int(round(nx*math.tan(th) + width/math.cos(th) + 2*width)) + 2
    geom = tilted_channel(nx, ny, theta, width)
    tau = 0.5 + 3*nu
    s = D2Q9Solver(geom, tau=tau)
    ux_ex, uy_ex = poiseuille_profile(geom, u_mean)
    s.initialise(ux_ex, uy_ex)
    inlet, outlet = channel_boundary_cells(geom)
    ax, ay = geom.axis
    if mode == "x":
        sh_in, sh_out = (0, 1), (0, -1)
    elif mode == "axis":
        def nearest(v):
            return max([(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)],
                       key=lambda c: c[0]*v[0]+c[1]*v[1])
        ci = nearest((ax, ay)); co = (-ci[0], -ci[1])
        sh_in, sh_out = (ci[1], ci[0]), (co[1], co[0])
    elif mode == "none":
        sh_in = sh_out = None
    if sh_in is not None:
        s.set_velocity_dirichlet([VelocityDirichlet(inlet, ux_ex, uy_ex, sh_in),
                                  VelocityDirichlet(outlet, ux_ex, uy_ex, sh_out)])
    G = 12.0*nu*u_mean/(width*width)
    fx = np.full((ny, nx), G*ax); fy = np.full((ny, nx), G*ay)
    steps = int(steps_factor*width*width)
    t0 = time.time()
    for _ in range(steps):
        if bf:
            s.step(fx, fy)
        else:
            s.step()
    ux, uy = s.velocity()
    ex = (ux-ux_ex); ey = (uy-uy_ex)
    mask = geom.fluid.copy()
    k = max(1, int(trim*nx)); mask[:, :k] = False; mask[:, -k:] = False
    l2 = float(np.sqrt(np.mean((ex**2+ey**2)[mask])))
    sdf = geom.sdf
    near = mask & (sdf < 2.0); mid = mask & (sdf >= 2.0)
    rms = lambda m: float(np.sqrt(np.mean((ex**2+ey**2)[m]))) if m.any() else float('nan')
    # streamwise location of the error (which x-band dominates)
    jj = np.arange(nx)[None, :]
    bands = []
    for lo, hi in [(0.25, 0.4), (0.4, 0.6), (0.6, 0.75)]:
        m = mask & (jj >= lo*nx) & (jj < hi*nx)
        bands.append(rms(m))
    return dict(theta=theta, width=width, mode=mode, bf=bf, l2=l2, l2_near=rms(near),
                l2_mid=rms(mid), bands=bands, wall=time.time()-t0)


if __name__ == "__main__":
    cfgs = []
    for mode in ["x", "axis"]:
        for bf in [False, True]:
            for theta in [0.0, 15.0, 30.0, 45.0]:
                for w in [8, 12, 16, 24]:
                    cfgs.append((theta, w, mode, bf, 0.01, 0.1, 30.0, 0.25))
    with Pool(48) as p:
        res = p.map(run, cfgs, chunksize=1)
    from collections import defaultdict
    g = defaultdict(list)
    for r in res:
        g[(r['mode'], r['bf'], r['theta'])].append(r)
    for key in sorted(g):
        rs = sorted(g[key], key=lambda r: r['width'])
        e = [r['l2'] for r in rs]; ws = [r['width'] for r in rs]
        orders = [math.log(e[i]/e[i+1])/math.log(ws[i+1]/ws[i]) for i in range(len(e)-1)]
        print(f"mode={key[0]:5s} bf={key[1]!s:5s} th={key[2]:4g} "
              f"l2={['%.2e'%v for v in e]} orders={['%.2f'%o for o in orders]} "
              f"near={['%.1e'%r['l2_near'] for r in rs]}", flush=True)