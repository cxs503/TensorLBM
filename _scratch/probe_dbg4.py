import numpy as np
from tensorlbm.corotational_beam import straight_beam

CLAMP = [(0, "x"), (0, "y"), (0, "theta")]
b = straight_beam(1.0, 8, E=1.0, A=1.2e5, I=1.0, rho=1.0, bc=CLAMP)
tip = 3 * (b.n_node - 1) + 1
K = b.linear_stiffness(); M = b.M; fr = b.free_dofs
Lc = np.linalg.cholesky(M[np.ix_(fr, fr)]); Li = np.linalg.solve(Lc, np.eye(len(fr)))
Ks = Li @ K[np.ix_(fr, fr)] @ Li.T
vals, vecs = np.linalg.eigh(0.5 * (Ks + Ks.T))
phi = np.zeros((b.n_dof, len(fr))); phi[fr] = Li.T @ vecs
w = np.sqrt(vals)
print("omega 1-4:", w[:4])
fvec = np.zeros(b.n_dof); fvec[tip] = 1e-4
Fi = phi.T @ fvec
print("phi_tip,mode1:", phi[tip, 0], " Fi[0]:", Fi[0])
Om = 0.63 * w[0]
T1 = 2 * np.pi / w[0]
dt = T1 / 80
hist = []
out = b.solve_newmark(dt=dt, n_steps=160, loads=lambda t: {b.n_node - 1: (0.0, 1e-4 * np.sin(Om * t), 0.0)},
                      sample=lambda s, t, q, qd, qdd: hist.append(q.copy()))
Q = np.asarray(hist); t = dt * np.arange(len(Q) + 1)
cm = Q @ (M @ phi)
amp_ex = Fi[:2] / (w[:2] ** 2 - Om ** 2)
print("analytic modal amplitude modes 1-2 (with homogeneous part, zero IC):", amp_ex)
print("FE modal amplitude (max |cm|) modes 1-2:", np.abs(cm[:, :2]).max(axis=0) / (0.5 if False else 1.0))
print("FE cm[0,:3]", cm[0, :3], "ex[0,:3]", (Fi[:3] / (w[:3] ** 2 - Om ** 2)) * (np.sin(Om * dt) - (Om / w[:3]) * np.sin(w[:3] * dt)))
print("cm max per mode:", np.abs(cm).max(axis=0)[:4])
print("tip max:", np.abs(Q[:, tip]).max())
print("iters:", [r.iterations for r in out["newton"][:8]])