"""Compare wall schemes on the inclined channel (diagnostic, scratch)."""
import sys, math, time
sys.path.insert(0, '/root/tp-agent/work/m9483/src')
import numpy as np
from multiprocessing import Pool
from tensorlbm.invariance.geometry import tilted_channel, channel_boundary_cells, poiseuille_profile
from tensorlbm.invariance.solver import D2Q9Solver, VelocityDirichlet, OPP, CVEL


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
            val_near = q*(2*q+1)*fi_b + (1-2*q)*(1+2*q)*fi_1 - q*(1-2*q)*fi_2
            inv = 1.0/(q*(2*q+1))
            val_far = inv*fi_b + (2*q-1)/q*fo_1 - (2*q-1)/(2*q+1)*fo_2
            val = np.where(near, val_near, val_far)
            np.copyto(f_new[OPP[i]], val, where=mask)
        return f_new


def run(cfg):
    theta, width, scheme, u_mean, nu = cfg
    th = math.radians(theta)
    nx = int(round(6*width)); ny = int(round(nx*math.tan(th) + width/math.cos(th) + 2*width)) + 2
    geom = tilted_channel(nx, ny, theta, width)
    cls = QuadSolver if scheme == "quad" else D2Q9Solver
    kw = {} if scheme in ("quad", "linear") else {"bounce_back": scheme}
    s = cls(geom, tau=0.5+3*nu, **kw)
    ux_ex, uy_ex = poiseuille_profile(geom, u_mean)
    s.initialise(ux_ex, uy_ex)
    inlet, outlet = channel_boundary_cells(geom)
    ax, ay = geom.axis
    s.set_velocity_dirichlet([VelocityDirichlet(inlet, ux_ex, uy_ex, (0, 1)),
                              VelocityDirichlet(outlet, ux_ex, uy_ex, (0, -1))])
    G = 12.0*nu*u_mean/(width*width)
    fx = np.full((ny, nx), G*ax); fy = np.full((ny, nx), G*ay)
    for _ in range(int(30*width*width)):
        s.step(fx, fy)
    ux, uy = s.velocity()
    axial_num = ux*ax + uy*ay
    axial_ex = ux_ex*ax + uy_ex*ay
    err = np.hypot(ux-ux_ex, uy-uy_ex)
    mask = geom.fluid.copy(); k = max(1, int(0.25*nx)); mask[:, :k]=False; mask[:,-k:]=False
    l2 = float(np.sqrt(np.mean(err[mask]**2)))
    # wall slip estimate: two fluid nodes closest to the wall along normal
    sdf = geom.sdf
    # fit parabola: u = A (d)(W-d) with d = H/2 - sdf, allowing wall offset delta:
    d = 0.5*width - sdf
    m = mask & (sdf > 0.4) & (sdf < 0.5*width)
    A = float(np.sum(axial_num[m]*d[m]*(width-d[m]))/np.sum((d[m]*(width-d[m]))**2))
    resid = axial_num[m] - A*d[m]*(width-d[m])
    return dict(theta=theta, width=width, scheme=scheme, l2=l2,
                amp_rel=A/(3*nu*u_mean/(width*width)/2/1) if False else A,
                amp_err=(A*width*width/(2*nu) - G)/G,
                resid_rms=float(np.sqrt(np.mean(resid**2))),
                n_first=int((m & (sdf<1.0)).sum()),
                u_first_ex=float(np.abs(axial_ex[m & (sdf<1.0)]).mean()) if (m & (sdf<1.0)).any() else float('nan'),
                u_first_num=float(np.abs(axial_num[m & (sdf<1.0)]).mean()) if (m & (sdf<1.0)).any() else float('nan'),
                max_err=float(err[mask].max()))


if __name__ == "__main__":
    cfgs = [(45.0, w, sch, 0.01, 0.1) for w in (8, 16) for sch in ("linear", "quad", "halfway")]
    cfgs += [(0.0, w, sch, 0.01, 0.1) for w in (8, 16) for sch in ("linear", "quad")]
    with Pool(8) as p:
        for r in p.map(run, cfgs):
            print(f"th={r['theta']:4g} H={r['width']:3d} {r['scheme']:8s} l2={r['l2']:.3e} "
                  f"max={r['max_err']:.2e} resid={r['resid_rms']:.2e} amp_err={r['amp_err']:+.2e} "
                  f"u1_num={r['u_first_num']:.3e} u1_ex={r['u_first_ex']:.3e}", flush=True)