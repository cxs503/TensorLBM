import numpy as np
from scipy.integrate import solve_ivp
from tensorlbm.corotational_beam import straight_beam

CLAMP = [(0, "x"), (0, "y"), (0, "theta")]
b = straight_beam(1.0, 8, E=1.0, A=1.2e5, I=1.0, rho=1.0, bc=CLAMP)
tip = 3 * (b.n_node - 1) + 1
w1 = b.mode_frequencies(1)[0]; T1 = 2 * np.pi / w1
Om = 0.63 * w1
P = 1e-4
K = b.linear_stiffness(); M = b.M; fr = b.free_dofs
Ki = np.linalg.inv(K[np.ix_(fr, fr)]); Mi = np.linalg.inv(M[np.ix_(fr, fr)])
fref = np.zeros(b.n_dof); fref[tip] = P

def rhs(t, y):
    n = len(fr)
    q = y[:n]; v = y[n:]
    f = fref[fr] * np.sin(Om * t)
    return np.concatenate([v, Mi @ (f - K[np.ix_(fr, fr)] @ q)])

dt = T1 / 80
n = 160
sol = solve_ivp(rhs, (0, n * dt), np.zeros(2 * len(fr)), method="Radau", rtol=1e-12, atol=1e-16,
                t_eval=dt * np.arange(1, n + 1), dense_output=False)
ref = np.zeros((n, b.n_dof)); ref[:, fr] = sol.y[:len(fr)].T
hist = []
b.solve_newmark(dt=dt, n_steps=n, loads=lambda t: {b.n_node - 1: (0.0, P * np.sin(Om * t), 0.0)},
                sample=lambda s, t, q, qd, qdd: hist.append(q.copy()))
Q = np.asarray(hist)
print("scipy linear tip max:", np.abs(ref[:, tip]).max(), " final:", ref[-1, tip])
print("newmark     tip max:", np.abs(Q[:, tip]).max(), " final:", Q[-1, tip])
err = np.linalg.norm(Q[:, tip] - ref[:, tip]) / np.linalg.norm(ref[:, tip])
print("rel L2 err vs scipy linear:", err)
print("ratio amplitude:", np.abs(Q[:, tip]).max() / np.abs(ref[:, tip]).max())
print("series (scipy) first 5:", ref[:5, tip])
print("series (newmark) first 5:", Q[:5, tip])
print("series (scipy) t~0.25T1:", ref[19:22, tip])
print("series (newmark) t~0.25T1:", Q[19:22, tip])