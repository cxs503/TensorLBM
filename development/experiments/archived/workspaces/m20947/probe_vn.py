import sys, math
sys.path.insert(0, 'src'); sys.path.insert(0, 'benchmarks/xval_job_checkpoint')
import numpy as np
from run_xval import taylor_green, kinetic_energy

# D2Q9
ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], float)
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], float)
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
cs2 = 1/3


def rate_linear(n, tau):
    """Spectral radius of the linearised BGK+streaming operator for the transverse
    (shear) mode with k = 2*pi/n along x."""
    k = 2 * math.pi / n
    omega = 1.0 / tau
    # linearised equilibrium projector: dfeq_q = w_q (drho + 3 (e_q . du))
    P = np.zeros((9, 9))
    for q in range(9):
        # row q: dfeq_q as linear functional of df (via moments)
        # drho = sum df ; du_x = sum e_x df ; du_y = sum e_y df
        P[q, :] = w[q] * (1.0 + 3.0 * ex[q] * ex) + 0.0
        P[q, :] += w[q] * 3.0 * ey[q] * ey
    M = np.eye(9) - omega * (np.eye(9) - P)
    S = np.diag(np.exp(-1j * k * ex))  # streaming phase for k along x (k_y = 0)
    A = S @ M
    return float(np.max(np.abs(np.linalg.eigvals(A))))


for n in (64, 128):
    for tau in (0.55, 0.8):
        nu = (tau - 0.5) / 3
        ana = 4 * nu * (2 * math.pi / n) ** 2
        lam_lin = -math.log(rate_linear(n, tau))
        print(f"n={n} tau={tau}: continuum={ana:.6e} discrete_lin={lam_lin:.6e} ratio={lam_lin/ana:.4f}")

# measured, small amplitude
for u0 in (0.01, 0.05):
    sim = taylor_green(64, 64)
    # rescale initial velocity amplitude
    from tensorlbm.lattice import equilibrium
    import torch
    x = (torch.arange(64, dtype=torch.float32) + 0.5) / 64 * 2 * math.pi
    xx, yy = torch.meshgrid(x, x, indexing="xy")
    ux = u0 * torch.sin(xx) * torch.cos(yy)
    uy = -u0 * torch.cos(xx) * torch.sin(yy)
    sim.f = equilibrium(torch.ones(64, 64), torch.stack([ux, uy], dim=-1))
    ke0 = kinetic_energy(sim)
    ts, kes = [0], [ke0]
    for s in range(1, 401):
        sim.step()
        ts.append(s); kes.append(kinetic_energy(sim))
    t = np.array(ts, float); ke = np.array(kes, float)
    m = (t >= 20) & (t <= 400)
    lam = -np.polyfit(t[m], np.log(ke[m]), 1)[0]
    print(f"u0={u0}: measured={lam:.6e} discrete_lin={-math.log(rate_linear(64,0.55)):.6e} "
          f"err_vs_lin={100*(lam+math.log(rate_linear(64,0.55)))/(-math.log(rate_linear(64,0.55))):+.3f}%")