import sys

import numpy as np

sys.path.insert(0, "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/src")
from indep_solver import Grid, make_solver, radial_grid, tri_apply_batched

rng = np.random.default_rng(0)
r = radial_grid(30, 8.0, 3.0)
g = Grid(r, 16)
# reference dense solve
n = g.n
err = 0.0
rhs = rng.standard_normal((g.M, n)) + 1j * rng.standard_normal((g.M, n))
solv = make_solver(g.lo, g.di, g.up)
x = solv.solve(rhs.copy())
for i in range(g.M):
    A = np.zeros((n, n))
    idx = g.idx
    A[idx, idx - 1] = g.lo[i, idx]
    A[idx, idx] = g.di[i, idx]
    A[idx, idx + 1] = g.up[i, idx]
    A[0, 0] = g.di[i, 0]
    A[-1, -1] = g.di[i, -1]
    xd = np.linalg.solve(A, rhs[i])
    err = max(err, np.max(np.abs(x[i] - xd)))
print("batched vs dense max err:", err)

# poisson sanity: omega = 0 -> psi should equal potential-flow perturbation
# psi_pot = (r - R^2/r) sin(th) - r sin(th) = -R^2/r sin(th)
g.poisson_solver = make_solver(g.lo, g.di, g.up)
omega = np.zeros((n, g.n_th))
psi = g.poisson(omega)
psi_exact = -(R2 := 0.25) / r[:, None] * np.sin(g.th)[None, :]
print(
    "poisson(0) vs -R^2/r sin(th): max abs err",
    np.max(np.abs(psi - psi_exact)),
    " max |psi|",
    np.max(np.abs(psi)),
)
# check Laplacian consistency: apply discrete L to psi and compare to -omega for a random omega
om2 = rng.standard_normal((n, g.n_th)) * 0.1
# project out Nyquist mode (odd spectral derivative undefined there; .real
# convention zeroes it) so the physical-space check is well posed
h = np.fft.fft(om2, axis=1)
h[:, g.n_th // 2] = 0.0
om2 = np.fft.ifft(h, axis=1).real
om2[0, :] = 0.0
om2[-1, :] = 0.0
psi2 = g.poisson(om2)
lap = tri_apply_batched(g.lo, g.di, g.up, np.zeros((g.M, n)))  # placeholder
res = np.zeros_like(om2)
idx = g.idx
lap_th = g.d_theta(g.d_theta(psi2)) / g.r[:, None] ** 2
res[idx] = (
    g.d2[:, 0:1] * psi2[idx - 1]
    + g.d1[:, 0:1] / g.ri[:, None] * psi2[idx - 1]
    + (g.d2[:, 1:2] + g.d1[:, 1:2] / g.ri[:, None]) * psi2[idx]
    + (g.d2[:, 2:3] + g.d1[:, 2:3] / g.ri[:, None]) * psi2[idx + 1]
) + lap_th[idx]
print("residual interior (should be -omega):", np.max(np.abs(res[idx] + om2[idx])))
# isolate: matrix application in Fourier space vs -omega_hat
psi_h = np.fft.fft(psi2, axis=1).T
lhs = tri_apply_batched(g.lo, g.di, g.up, psi_h)
om_h = np.fft.fft(om2, axis=1).T
print("matrix-apply residual (fourier):", np.max(np.abs(lhs[:, idx] + om_h[:, idx])))
# radial-only residual
rad = np.zeros_like(om2)
rad[idx] = (
    g.d2[:, 0:1] * psi2[idx - 1]
    + g.d1[:, 0:1] / g.ri[:, None] * psi2[idx - 1]
    + (g.d2[:, 1:2] + g.d1[:, 1:2] / g.ri[:, None]) * psi2[idx]
    + (g.d2[:, 2:3] + g.d1[:, 2:3] / g.ri[:, None]) * psi2[idx + 1]
)
print(
    "radial part vs -omega - lap_th:",
    np.max(np.abs(rad[idx] + om2[idx] - lap_th[idx])),
    " max|lap_th|:",
    np.max(np.abs(lap_th)),
)
