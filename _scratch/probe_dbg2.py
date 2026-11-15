import numpy as np
from tensorlbm.corotational_beam import straight_beam

CLAMP = [(0, "x"), (0, "y"), (0, "theta")]
b = straight_beam(1.0, 8, E=1.0, A=1.2e5, I=1.0, rho=1.0, bc=CLAMP)
tip = 3 * (b.n_node - 1) + 1

K = b.linear_stiffness(); M = b.M; fr = b.free_dofs
Lc = np.linalg.cholesky(M[np.ix_(fr, fr)])
Li = np.linalg.solve(Lc, np.eye(Lc.shape[0]))
Ks = Li @ K[np.ix_(fr, fr)] @ Li.T
vals, vecs = np.linalg.eigh(0.5 * (Ks + Ks.T))
phi = np.zeros((b.n_dof, len(fr))); phi[fr] = Li.T @ vecs
w = np.sqrt(np.maximum(vals, 0))
print("orthonormality err:", np.abs(phi.T @ M @ phi - np.eye(len(fr))).max())

q0, _ = b.solve_static({b.n_node - 1: (0.0, -0.004, 0.0)}, n_steps=1)
c0 = phi.T @ (M @ q0)
print("c0 (first 6):", c0[:6], "norm", np.linalg.norm(c0))
T1 = 2 * np.pi / w[0]
dt = T1 / 160
hist = []
b.solve_newmark(dt=dt, n_steps=640, q0=q0, sample=lambda s, t, q, qd, qdd: hist.append(q.copy()))
Q = np.asarray(hist)
cm = Q @ (M @ phi)
tt = dt * np.arange(1, len(Q) + 1)
ex_c1 = c0[0] * np.cos(w[0] * tt)
print("c1: FE  start %.4e  max %.4e   exact max %.4e" % (cm[0, 0], np.abs(cm[:, 0]).max(), abs(c0[0])))
print("c1 ratio FE/exact (max over t):", np.abs(cm[:, 0]).max() / abs(c0[0]))
for i in range(4):
    print(f"mode{i+1}: FE amp {np.abs(cm[:,i]).max():.4e} exact amp {abs(c0[i]):.4e}"
          f" ratio {np.abs(cm[:,i]).max()/abs(c0[i]):.4f}")
# consistency of the FE solution with the linearised equation (residual)
# check whether the nonlinear internal force differs from K q
for k in (0, 100, 300):
    q = Q[k]
    print("step", k, "|g(q)-Kq|/|Kq| =", np.linalg.norm(b.internal_forces(q) - K @ q) / np.linalg.norm(K @ q))