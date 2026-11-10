"""Test the quadratic (3-point) BFL interpolation against the linear one in a
force-driven periodic channel with the wall at an arbitrary sub-cell position."""
import sys, math, time
sys.path.insert(0, '/root/tp-agent/work/m9483/src')
import numpy as np
from multiprocessing import Pool
from tensorlbm.invariance.solver import D2Q9Solver, OPP, CVEL

class PeriodicChannel:
    def __init__(self, nx, ny, width, off=0.0):
        self.nx, self.ny = nx, ny
        cx, cy = 0.5*nx, 0.5*ny
        self.y0 = cy - 0.5*width + off
        self.y1 = cy + 0.5*width + off
        Y = (np.arange(ny) + 0.5)[:, None]
        dl = Y - self.y0; du = self.y1 - Y
        self.sdf = np.broadcast_to(np.minimum(dl, du), (ny, nx)).copy()
        gy = np.where(dl < du, 1.0, -1.0)
        self.gy = np.broadcast_to(gy, (ny, nx)).copy()
        self.gx = np.zeros((ny, nx))
        self.solid = self.sdf <= 0.0
        self.theta_rad = 0.0; self.axis = (1.0, 0.0); self.normal = (0.0, 1.0)
        self.meta = {"kind": "periodic_channel", "width": width, "center": (cx, cy),
                     "y0": self.y0, "y1": self.y1}
    @property
    def fluid(self):
        return ~self.solid


class QuadSolver(D2Q9Solver):
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
            fo_2 = np.roll(fs[OPP[i]], shift=(2*cy, 2*cx), axis=(0, 1))
            near = q < 0.5
            val_near = q*(1+2*q)*fi_b + (1-2*q)*(1+2*q)*fi_1 - q*(1-2*q)*fi_2
            val_far = (1.0/(q*(1+2*q)))*fi_b + ((2*q-1)/q)*fo_1 - ((2*q-1)/(1+2*q))*fo_2
            val = np.where(near, val_near, val_far)
            np.copyto(f_new[OPP[i]], val, where=mask)
        return f_new


def run(cfg):
    width, off, nu, scheme = cfg
    nx = 8; ny = int(round(width)) + 8
    geom = PeriodicChannel(nx, ny, width, off)
    cls = QuadSolver if scheme == "quad" else D2Q9Solver
    s = cls(geom, tau=0.5+3*nu)
    W = float(geom.meta["width"])
    G = 12.0*nu*0.01/(W*W)
    d = geom.sdf
    ux_ex = np.broadcast_to((G/(2*nu))*d*(W-d), (ny, nx)).copy()
    s.initialise(ux_ex, np.zeros_like(ux_ex))
    for _ in range(int(20*width*width)):
        s.step(G, 0.0)
    ux, uy = s.velocity()
    if not np.all(np.isfinite(ux)):
        return dict(width=width, off=off, scheme=scheme, l2=float('nan'), slip=float('nan'), delta=float('nan'))
    err = np.hypot(ux-ux_ex, uy)
    m = geom.fluid.copy()
    l2 = float(np.sqrt(np.mean(err[m]**2)))
    # slip error: linear fit of the ERROR field vs sdf near the walls -> intercept
    mm = m & (geom.sdf < 1.5)
    A = np.stack([np.ones(mm.sum()), geom.sdf[mm]], axis=1)
    coef, *_ = np.linalg.lstsq(A, (ux-ux_ex)[mm], rcond=None)
    dudn = G*W/(2*nu)
    return dict(width=width, off=off, scheme=scheme, l2=l2, slip=float(coef[0]),
                delta=float(-coef[0]/dudn))


if __name__ == "__main__":
    cfgs = []
    for scheme in ("linear", "quad"):
        for off in (0.0, 0.25, 0.3):
            for w in (8, 12, 16, 24):
                cfgs.append((float(w), off, 0.1, scheme))
    with Pool(12) as p:
        res = p.map(run, cfgs, chunksize=1)
    from collections import defaultdict
    g = defaultdict(list)
    for r in res:
        g[(r['scheme'], r['off'])].append(r)
    for k in sorted(g):
        rs = sorted(g[k], key=lambda r: r['width'])
        e = [r['l2'] for r in rs]; ws = [r['width'] for r in rs]
        o = [math.log(e[i]/e[i+1])/math.log(ws[i+1]/ws[i]) for i in range(len(e)-1)]
        print(f"{k[0]:7s} off={k[1]:+.2f} l2={['%.2e'%v for v in e]} orders={['%.2f'%x for x in o]} "
              f"delta={['%+.3f'%r['delta'] for r in rs]}", flush=True)