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


def forced_exact(t):
    # zero IC, g(t)=sin(Om t): q_i = (F_i/w_i) Int_0^t sin(w(t-s)) sin(Om s) ds
    return np.array([Fi[i] / w[i] * (w[i] * np.sin(Om * t) - Om * np.sin(w[i] * t)) / (w[i]**2 - Om**2)
                     for i in range(len(w))]).T


prev = None
print("D4b harmonic forcing, modal mode-1 error vs analytic, dt ladder")
for frac in (10, 20, 40, 80, 160, 320):
    dt = T1 / frac; n = 2 * frac
    hist = []
    out = b.solve_newmark(dt=dt, n_steps=n, loads=lambda t: {b.n_node - 1: (0.0, P * np.sin(Om * t), 0.0)},
                          sample=lambda s, t, q, qd, qdd: hist.append(q.copy()))
    Q = np.asarray(hist); t = dt * np.arange(1, n + 1)
    cm = Q @ (M @ phi); ex = forced_exact(t)
    err = np.linalg.norm(cm[:, 0] - ex[:, 0]) / np.linalg.norm(ex[:, 0])
    rate = float("nan") if prev is None else np.log(prev / err) / np.log(2)
    print(f"  dt=T1/{frac:<4d} err={err:.4e} order={rate:.2f} amp FE {np.abs(cm[:,0]).max():.4e}"
          f" exact {np.abs(ex[:,0]).max():.4e} nconv={sum(not r.converged for r in out['newton'])}/{n}")
    prev = err

# free vibration, mode 1 only (dt ladder fine enough for mode 1)
q0, _ = b.solve_static({b.n_node - 1: (0.0, -0.004, 0.0)}, n_steps=1)
c0 = phi.T @ (M @ q0)
print("\nD4a free vibration, mode-1 error vs analytic, dt ladder")
prev = None
for frac in (10, 20, 40, 80, 160, 320):
    dt = T1 / frac; n = 2 * frac
    hist = []
    out = b.solve_newmark(dt=dt, n_steps=n, q0=q0, sample=lambda s, t, q, qd, qdd: hist.append(q.copy()))
    Q = np.asarray(hist); t = dt * np.arange(1, n + 1)
    cm = Q @ (M @ phi); ex = np.cos(np.outer(t, w)) * c0
    err = np.linalg.norm(cm[:, 0] - ex[:, 0]) / np.linalg.norm(ex[:, 0])
    rate = float("nan") if prev is None else np.log(prev / err) / np.log(2)
    print(f"  dt=T1/{frac:<4d} err={err:.4e} order={rate:.2f} floor={sum(r.floor_limited for r in out['newton'])}/{n}")
    prev = err