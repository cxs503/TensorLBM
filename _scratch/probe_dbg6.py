import numpy as np
from tensorlbm.corotational_beam import straight_beam

CLAMP = [(0, "x"), (0, "y"), (0, "theta")]
b = straight_beam(1.0, 8, E=1.0, A=1.2e5, I=1.0, rho=1.0, bc=CLAMP)
tip = 3 * (b.n_node - 1) + 1
w1 = b.mode_frequencies(1)[0]; T1 = 2 * np.pi / w1
Om = 0.63 * w1; P = 1e-4
K = b.linear_stiffness(); M = b.M; fr = b.free_dofs
Lc = np.linalg.cholesky(M[np.ix_(fr, fr)]); Li = np.linalg.solve(Lc, np.eye(len(fr)))
Ks = Li @ K[np.ix_(fr, fr)] @ Li.T
vals, vecs = np.linalg.eigh(0.5 * (Ks + Ks.T))
phi = np.zeros((b.n_dof, len(fr))); phi[fr] = Li.T @ vecs
w = np.sqrt(vals)
fvec = np.zeros(b.n_dof); fvec[tip] = P
Fi = phi.T @ fvec

prev = None
print("D4b full analytic (harmonic, zero IC), dt ladder, mode-1 only error")
for frac in (10, 20, 40, 80, 160, 320):
    dt = T1 / frac; n = 2 * frac
    hist = []
    out = b.solve_newmark(dt=dt, n_steps=n, loads=lambda t: {tip // 3: (0.0, P * np.sin(Om * t), 0.0)},
                          sample=lambda s, t, q, qd, qdd: hist.append(q.copy()))
    Q = np.asarray(hist); t = dt * np.arange(1, n + 1)
    cm = Q @ (M @ phi)
    ex = np.zeros((n, len(w)))
    for i, wi in enumerate(w):
        ex[:, i] = Fi[i] * (wi * np.sin(Om * t) - Om * np.sin(wi * t)) / (wi**2 - Om**2)
    err1 = np.linalg.norm(cm[:, 0] - ex[:, 0]) / np.linalg.norm(ex[:, 0])
    rate = float("nan") if prev is None else np.log(prev / err1) / np.log(2)
    print(f"  dt=T1/{frac:<4d} n={n:<4d} err_mode1={err1:.4e} order={rate:.2f} "
          f"maxit={out['max_iterations']} nconv={sum(not r.converged for r in out['newton'])}")
    prev = err1