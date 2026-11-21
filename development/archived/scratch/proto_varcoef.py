"""Prototype: exact variable-coefficient curl-Darcy solve for the spectral ST solver.

Exact relation for the *perturbation* streamfunction psi_tilde (base psi0 = y removed),
with total c = cbar(x) + c_tilde::

    lap(psi) + R (c_x psi_x + c_y psi_y) = -R c_y

(since psi0 = y contributes lap=0, and c_y*(1 + psi_y) -> the "+1" part is the -R c_y RHS).
This is what the module claims to solve; the frozen OSS symbol approximates it with a
pointwise (not convolution) coefficient. Solve it exactly, matrix-free, restarted GMRES
with the lap symbol as right preconditioner.
"""
import sys, time
import numpy as np
import torch

sys.path.insert(0, "/srv/tensor/agent/work/m21247/src")
from tensorlbm.spectral_saffman_taylor import SpectralSaffmanTaylor2D

torch.set_num_threads(4)


def grad(sim, f_hat, axis):
    k = sim._km if axis == "x" else sim._kn
    return torch.fft.ifft2(f_hat * k * 1j).real


def apply_A(sim, psi):
    ph = torch.fft.fft2(psi)
    lap = torch.fft.ifft2(-sim._k2 * ph).real
    return lap + sim.viscosity_ratio * (sim.c_x.real * grad(sim, ph, "x")
                                        + sim.c_y.real * grad(sim, ph, "y"))


def apply_Pinv(sim, r):
    rh = torch.fft.fft2(r)
    rh = torch.where(sim._nonzero, rh / (-sim._k2_safe), torch.zeros_like(rh))
    return torch.fft.ifft2(rh).real


def gmres(sim, b, tol=1e-11, restart=40, maxit=400, verbose=False):
    """Right-preconditioned restarted GMRES on A x = b (A ~ lap + R c.grad)."""
    nb = float(torch.linalg.vector_norm(b))
    x = torch.zeros_like(b)
    if nb == 0.0:
        return x, 0, 0.0
    it = 0
    converged = False
    r = b.clone()
    res = 1.0
    while it < maxit and not converged:
        r = b - apply_A(sim, x)
        beta = float(torch.linalg.vector_norm(r))
        res = beta / nb
        if res < tol:
            converged = True
            break
        # Arnoldi with right preconditioning: w = A P^-1 v
        V = [r / beta]
        H = np.zeros((restart + 1, restart), dtype=np.float64)
        m = 0
        for j in range(restart):
            z = apply_Pinv(sim, V[j])
            w = apply_A(sim, z)
            for i in range(j + 1):
                H[i, j] = float(torch.sum(w * V[i]))
                w = w - H[i, j] * V[i]
            H[j + 1, j] = float(torch.linalg.vector_norm(w))
            if H[j + 1, j] < 1e-14:
                m = j + 1
                break
            V.append(w / H[j + 1, j])
            m = j + 1
        H = H[:m + 1, :m]
        e1 = np.zeros(m + 1)
        e1[0] = beta
        y = np.linalg.lstsq(H, e1, rcond=None)[0]
        # x += P^-1 (V[:m] @ y)
        combo = torch.zeros_like(b)
        for j in range(m):
            combo = combo + float(y[j]) * V[j]
        x = x + apply_Pinv(sim, combo)
        it += m
        r = b - apply_A(sim, x)
        res = float(torch.linalg.vector_norm(r)) / nb
        if verbose:
            print(f"   gmres it={it} res={res:.3e}")
        if res < tol:
            converged = True
    return x, it, res


def main():
    NY, NX, ASPECT, PE, MODE, W, DT = 128, 256, 2.0, 500.0, 4, 4.0, 0.1
    for R in (-1.0, 1.0, 2.0, -2.0):
        sim = SpectralSaffmanTaylor2D(NY, NX, aspect=ASPECT, pe=PE, viscosity_ratio=R, dt=DT,
                                     mode_index=MODE)
        sim.set_concentration(sim.single_mode_ic(x0=0.5 * ASPECT * PE, width=W,
                                                 amplitude=1e-3, n_mode=MODE))
        b = -R * sim.c_y.real
        t0 = time.time()
        psi, it, res = gmres(sim, b)
        dt_solve = time.time() - t0
        r = apply_A(sim, psi) - b
        print(f"R={R:+.1f} gmres it={it} res={res:.2e} |r|/|b|={float(torch.linalg.vector_norm(r))/float(torch.linalg.vector_norm(b)):.2e} t={dt_solve:.2f}s")


if __name__ == "__main__":
    main()