import sys, math, time
sys.path.insert(0, '/root/tp-agent/work/m9483/src')
import numpy as np
from tensorlbm.invariance.geometry import tilted_channel, channel_boundary_cells, poiseuille_profile
from tensorlbm.invariance.solver import D2Q9Solver, VelocityDirichlet

def run(theta=45.0, width=16, u_mean=0.01, nu=0.1, steps_factor=30.0, tag="H16"):
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
    for _ in range(int(steps_factor*width*width)):
        s.step(fx, fy)
    rho, ux, uy = s.macroscopic()
    np.savez(f".scratch/field_{tag}.npz", rho=rho, ux=ux, uy=uy, sdf=geom.sdf,
             solid=geom.solid, ux_ex=ux_ex, uy_ex=uy_ex, axis=np.array(geom.axis), width=width)
    print("saved", tag, ny, nx)

if __name__ == "__main__":
    run(tag="H16")
    run(width=8, tag="H8")