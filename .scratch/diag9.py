"""Hybrid wall closure: quadratic near branch (q<0.5) + linear far branch."""
import sys, math
sys.path.insert(0, '/root/tp-agent/work/m9483/src')
import numpy as np
from multiprocessing import Pool
from tensorlbm.invariance.solver import D2Q9Solver, OPP, CVEL
from tensorlbm.invariance.cases import PeriodicChannel, _body_force, _parabolic_profile


class HybridSolver(D2Q9Solver):
    def stream_interpolated(self, fs):
        f_new = np.empty_like(fs)
        for i in range(9):
            f_new[i] = np.roll(fs[i], shift=(int(CVEL[i, 1]), int(CVEL[i, 0])), axis=(0, 1))
        walls = self._walls
        for i in range(1, 9):
            mask = walls.mask[i]
            if mask is None or not mask.any():
                continue
            cx, cy = int(CVEL[i, 0]), int(CVEL[i, 1])
            q = walls.q[i]
            fi_b = fs[i]
            fi_1 = np.roll(fs[i], shift=(cy, cx), axis=(0, 1))
            fi_2 = np.roll(fs[i], shift=(2*cy, 2*cx), axis=(0, 1))
            fo_1 = np.roll(fs[OPP[i]], shift=(cy, cx), axis=(0, 1))
            near = q < 0.5
            val_near = q*(1+2*q)*fi_b + (1-2*q)*(1+2*q)*fi_1 - q*(1-2*q)*fi_2
            val_far = (1.0/(2*q))*fi_b + (1.0 - 1.0/(2*q))*fo_1
            val = np.where(near, val_near, val_far)
            np.copyto(f_new[OPP[i]], val, where=mask)
        return f_new


def run(cfg):
    width, off, scheme = cfg
    nu = 0.1
    ny = int(round(width)) + 8
    geom = PeriodicChannel(8, ny, width, off)
    if scheme == "hybrid":
        s = HybridSolver(geom, tau=0.5+3*nu)
    else:
        s = D2Q9Solver(geom, tau=0.5+3*nu, bounce_back="halfway")
    ux_ex, _ = _parabolic_profile(geom, 0.01, nu)
    s.initialise(ux_ex, np.zeros_like(ux_ex))
    G = _body_force(width, nu, 0.01)
    for _ in range(int(20*width*width)):
        s.step(G, 0.0)
    ux, uy = s.velocity()
    if not np.all(np.isfinite(ux)):
        return dict(width=width, off=off, scheme=scheme, l2=float('nan'))
    err = np.hypot(ux-ux_ex, uy)
    m = geom.fluid
    return dict(width=width, off=off, scheme=scheme,
                l2=float(np.sqrt(np.mean(err[m]**2))))


if __name__ == "__main__":
    cfgs = [(float(w), 0.25, "hybrid") for w in (8, 12, 16, 24)]
    cfgs += [(float(w), off, "halfway") for off in (0.0, 0.25) for w in (8, 12, 16, 24)]
    with Pool(12) as p:
        res = p.map(run, cfgs, chunksize=1)
    from collections import defaultdict
    g = defaultdict(list)
    for r in res:
        g[(r['scheme'], r['off'])].append(r)
    for k in sorted(g):
        rs = sorted(g[k], key=lambda r: r['width'])
        e = [r['l2'] for r in rs]; ws = [r['width'] for r in rs]
        o = [math.log(e[i]/e[i+1])/math.log(ws[i+1]/ws[i]) for i in range(len(e)-1)] if all(np.isfinite(e)) else [float('nan')]
        print(f"{k[0]:8s} off={k[1]:+.2f} l2={['%.2e'%v for v in e]} orders={['%.2f'%x for x in o]}", flush=True)