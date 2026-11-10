import sys, math, time
sys.path.insert(0, '/root/tp-agent/work/m9483/src')
import numpy as np
from tensorlbm.invariance.geometry import tilted_channel, channel_boundary_cells, poiseuille_profile
from tensorlbm.invariance.solver import D2Q9Solver, VelocityDirichlet


def run(theta, width, u_mean=0.01, nu=0.1, steps_factor=30.0, bf=False, trim=0.25):
    th = math.radians(theta)
    nx = int(round(6*width)); ny = int(round(nx*math.tan(th) + width/math.cos(th) + 2*width)) + 2
    geom = tilted_channel(nx, ny, theta, width)
    s = D2Q9Solver(geom, tau=0.5+3*nu)
    ux_ex, uy_ex = poiseuille_profile(geom, u_mean)
    s.initialise(ux_ex, uy_ex)
    inlet, outlet = channel_boundary_cells(geom)
    ax, ay = geom.axis
    s.set_velocity_dirichlet([VelocityDirichlet(inlet, ux_ex, uy_ex, (0, 1)),
                              VelocityDirichlet(outlet, ux_ex, uy_ex, (0, -1))])
    G = 12.0*nu*u_mean/(width*width)
    fx = np.full((ny, nx), G*ax); fy = np.full((ny, nx), G*ay)
    steps = int(steps_factor*width*width)
    t0 = time.time()
    for _ in range(steps):
        s.step(fx, fy) if bf else s.step()
    rho, ux, uy = s.macroscopic()
    ex = ux-ux_ex; ey = uy-uy_ex
    err = np.hypot(ex, ey)
    mask = geom.fluid.copy(); k = max(1, int(trim*nx)); mask[:,:k]=False; mask[:,-k:]=False
    sdf = geom.sdf
    # error vs distance from wall
    prof = []
    for lo, hi in [(0,0.5),(0.5,1),(1,2),(2,4),(4,8),(8,1e9)]:
        m = mask & (sdf>=lo) & (sdf<hi)
        prof.append((lo,hi, float(np.sqrt(np.mean(err[m]**2))) if m.any() else float('nan'), int(m.sum())))
    # error vs streamwise band
    jj = np.tile(np.arange(nx)[None,:], (ny,1))
    bands=[]
    for lo,hi in [(0.25,0.35),(0.35,0.45),(0.45,0.55),(0.55,0.65),(0.65,0.75)]:
        m = mask & (jj>=lo*nx) & (jj<hi*nx)
        bands.append(float(np.sqrt(np.mean(err[m]**2))) if m.any() else float('nan'))
    # flux through inlet/outlet and net mass change
    flux_in = float((ux*ax+uy*ay)[inlet].sum())
    flux_out = float((ux*ax+uy*ay)[outlet].sum())
    flux_ex = float((ux_ex*ax+uy_ex*ay)[inlet].sum())
    # where is max error?
    idx = np.unravel_index(np.argmax(np.where(mask, err, 0)), err.shape)
    return dict(width=width, nx=nx, ny=ny, l2=float(np.sqrt(np.mean(err[mask]**2))),
                l2_flu=float(np.sqrt(np.mean(err[geom.fluid]**2))),
                prof=prof, bands=bands, flux_in=flux_in, flux_out=flux_out, flux_ex=flux_ex,
                rho_min=float(rho[geom.fluid].min()), rho_max=float(rho[geom.fluid].max()),
                maxerr_at=(int(idx[0]), int(idx[1]), float(err[idx]), float(sdf[idx]),
                           float((jj[idx])/nx)),
                wall=time.time()-t0)

if __name__ == "__main__":
    for bf in [False, True]:
        for theta in [0.0, 45.0]:
            for w in [8, 16]:
                r = run(theta, w, bf=bf)
                print(f"th={theta} H={w} bf={bf} l2={r['l2']:.3e} l2all={r['l2_flu']:.3e} "
                      f"fluxin={r['flux_in']:.4e} fluxout={r['flux_out']:.4e} fluex={r['flux_ex']:.4e} "
                      f"rho=[{r['rho_min']:.4f},{r['rho_max']:.4f}] maxerr={r['maxerr_at']} t={r['wall']:.1f}s", flush=True)
                print("   prof(dist,l2,n):", ["(%.1f-%.1f:%.2e,%d)"%(a,b,c,d) for a,b,c,d in r['prof']], flush=True)
                print("   bands:", ["%.2e"%b for b in r['bands']], flush=True)