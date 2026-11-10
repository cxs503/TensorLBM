"""Is the O(h) wall-position error intrinsic to the BFL scheme or caused by the
inflow/outflow BC?  Test: *force-driven periodic* channel, walls at arbitrary
sub-cell positions, no inflow/outflow at all."""
import sys, math, time
sys.path.insert(0, '/root/tp-agent/work/m9483/src')
import numpy as np
from multiprocessing import Pool
from tensorlbm.invariance.solver import D2Q9Solver


class PeriodicChannel:
    def __init__(self, nx, ny, width, off=0.0):
        self.nx, self.ny = nx, ny
        cx, cy = 0.5*nx, 0.5*ny
        self.y0 = cy - 0.5*width + off
        self.y1 = cy + 0.5*width + off
        Y = (np.arange(ny) + 0.5)[:, None]
        X = (np.arange(nx) + 0.5)[None, :]
        dl = Y - self.y0
        du = self.y1 - Y
        self.sdf = np.broadcast_to(np.minimum(dl, du), (ny, nx)).copy()
        gy = np.where(dl < du, 1.0, -1.0)
        self.gy = np.broadcast_to(gy, (ny, nx)).copy()
        self.gx = np.zeros((ny, nx))
        self.solid = self.sdf <= 0.0
        self.theta_rad = 0.0
        self.axis = (1.0, 0.0)
        self.normal = (0.0, 1.0)
        self.meta = {"kind": "periodic_channel", "width": width, "center": (cx, cy),
                     "y0": self.y0, "y1": self.y1}

    @property
    def fluid(self):
        return ~self.solid


def exact(geom, u_mean, nu):
    W = float(geom.meta["width"])
    G = 12.0*nu*u_mean/(W*W)
    d = geom.sdf
    u = (G/(2*nu))*d*(W - d)
    return np.broadcast_to(u, (geom.ny, geom.nx)).copy()


def run(cfg):
    width, off, nu, nx = cfg
    ny = int(round(width)) + 8
    geom = PeriodicChannel(nx, ny, width, off)
    s = D2Q9Solver(geom, tau=0.5+3*nu)
    ux_ex = exact(geom, 0.01, nu)
    s.initialise(ux_ex, np.zeros_like(ux_ex))
    W = float(geom.meta["width"])
    G = 12.0*nu*0.01/(W*W)
    steps = int(20*width*width)
    for _ in range(steps):
        s.step(G, 0.0)
    ux, uy = s.velocity()
    err = np.hypot(ux-ux_ex, uy)
    m = geom.fluid.copy()
    l2 = float(np.sqrt(np.mean(err[m]**2)))
    # slip: linear fit of ux vs sdf for the lowest band of both walls
    sdf = geom.sdf
    mm = m & (sdf < 1.5)
    A = np.stack([np.ones(mm.sum()), sdf[mm]], axis=1)
    coef, *_ = np.linalg.lstsq(A, ux[mm], rcond=None)
    dudn = G*W/(2*nu)
    return dict(width=width, off=off, l2=l2, slip=float(coef[0]),
                delta=float(-coef[0]/dudn), linf=float(err[m].max()))


if __name__ == "__main__":
    cfgs = []
    for off in (0.0, 0.25, 0.3, 0.13):
        for w in (8, 12, 16, 24):
            cfgs.append((float(w), off, 0.1, 8))
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
        print(f"off={off:+.2f} l2={['%.2e'%v for v in e]} orders={['%.2f'%x for x in o]} "
              f"delta={['%+.3f'%r['delta'] for r in rs]}", flush=True)