"""Does the interpolated BB survive a sub-cell wall offset at theta=0?"""
import sys, math, time
sys.path.insert(0, '/root/tp-agent/work/m9483/src')
import numpy as np
from multiprocessing import Pool
from tensorlbm.invariance.geometry import tilted_channel, channel_boundary_cells, poiseuille_profile
from tensorlbm.invariance.solver import D2Q9Solver, VelocityDirichlet


def run(cfg):
    width, off, bf, theta = cfg
    th = math.radians(theta)
    nx = int(round(6*width)); ny = int(round(nx*math.tan(th) + width/math.cos(th) + 2*width)) + 2
    geom = tilted_channel(nx, ny, theta, width, center=(0.5*nx, 0.5*ny + off))
    s = D2Q9Solver(geom, tau=0.5 + 0.3)
    ux_ex, uy_ex = poiseuille_profile(geom, 0.01)
    s.initialise(ux_ex, uy_ex)
    inlet, outlet = channel_boundary_cells(geom)
    s.set_velocity_dirichlet([VelocityDirichlet(inlet, ux_ex, uy_ex, (0, 1)),
                              VelocityDirichlet(outlet, ux_ex, uy_ex, (0, -1))])
    ax, ay = geom.axis
    G = 12.0*0.1*0.01/(width*width)
    fx = np.full((ny, nx), G*ax); fy = np.full((ny, nx), G*ay)
    for _ in range(int(30*width*width)):
        s.step(fx, fy) if bf else s.step()
    ux, uy = s.velocity()
    err = np.hypot(ux-ux_ex, uy-uy_ex)
    m = geom.fluid.copy(); k = max(1, int(0.25*nx)); m[:, :k]=False; m[:,-k:]=False
    # fraction of near-wall nodes with q far from 0.5
    qs = np.concatenate([w.ravel() for w, mk in zip(geom.fluid.astype(float).ravel()*0+1, [0]) for w in []]) if False else None
    return dict(width=width, off=off, bf=bf, theta=theta,
                l2=float(np.sqrt(np.mean(err[m]**2))), linf=float(err[m].max()))


if __name__ == "__main__":
    cfgs = []
    for off in (0.0, 0.25, -0.3, 0.13):
        for w in (8, 12, 16, 24):
            cfgs.append((w, off, True, 0.0))
    with Pool(16) as p:
        res = p.map(run, cfgs, chunksize=1)
    from collections import defaultdict
    g = defaultdict(list)
    for r in res:
        g[r['off']].append(r)
    for off in sorted(g):
        rs = sorted(g[off], key=lambda r: r['width'])
        e = [r['l2'] for r in rs]; ws = [r['width'] for r in rs]
        o = [math.log(e[i]/e[i+1])/math.log(ws[i+1]/ws[i]) for i in range(len(e)-1)]
        print(f"off={off:+.2f} l2={['%.2e'%v for v in e]} orders={['%.2f'%x for x in o]}", flush=True)