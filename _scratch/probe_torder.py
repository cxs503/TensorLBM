import numpy as np
from tensorlbm.corotational_beam import straight_beam

CLAMP = [(0, "x"), (0, "y"), (0, "theta")]
E, A, I, rho, L = 1.0, 1.2e5, 1.0, 1.0, 1.0


def modes(b, nmax=4):
    K = b.linear_stiffness(); M = b.M; fr = b.free_dofs
    Kr = K[np.ix_(fr, fr)]; Mr = M[np.ix_(fr, fr)]
    Lc = np.linalg.cholesky(Mr); Li = np.linalg.solve(Lc, np.eye(Lc.shape[0]))
    Ks = Li @ Kr @ Li.T; Ks = 0.5 * (Ks + Ks.T)
    vals, vecs = np.linalg.eigh(Ks)
    idx = np.argsort(vals)[:nmax]
    phi = np.zeros((b.n_dof, len(idx)))
    phi[fr] = Li.T @ vecs[:, idx]
    return np.sqrt(vals[idx]), phi


def run(b, dt, nsteps, loads=None, q0=None):
    rec = {"t": [], "q": []}
    def sample(step, t, q, qdot, qddot):
        rec["t"].append(t); rec["q"].append(q.copy())
    out = b.solve_newmark(dt=dt, n_steps=nsteps, loads=loads, q0=q0, sample=sample)
    return np.asarray(rec["t"]), np.asarray(rec["q"]), out


b = straight_beam(L, 8, E=E, A=A, I=I, rho=rho, bc=CLAMP)
w, phi = modes(b, 4)
tipdof = 3 * (b.n_node - 1) + 1
print("omega:", w)

# ---------------- D4a: free vibration from a static deflected shape
q0, _ = b.solve_static({b.n_node - 1: (0.0, -0.004, 0.0)}, n_steps=1)
T1 = 2 * np.pi / w[0]
c0 = phi.T @ (b.M @ q0)
print("\nD4a free vibration (modal-coordinate error, first 3 modes)")
prev = None
for frac in (10, 20, 40, 80, 160):
    dt = T1 / frac
    t, Q, out = run(b, dt, 2 * frac, q0=q0)
    cm = Q @ (b.M @ phi)                       # modal coords of the FE solution
    ex = np.cos(np.outer(t, w)) * c0
    err = np.linalg.norm(cm[:, :3] - ex[:, :3]) / np.linalg.norm(ex[:, :3])
    e_tip = np.linalg.norm(Q[:, tipdof] - (ex @ phi.T)[:, tipdof]) / np.abs((ex @ phi.T)[:, tipdof]).max()
    rate = float("nan") if prev is None else np.log(prev / err) / np.log(2)
    print(f"  dt=T1/{frac:<4d} err_modal={err:.4e} order={rate:.2f}  err_tip={e_tip:.3e}"
          f"  newton_it={out['max_iterations']} floor={sum(r.floor_limited for r in out['newton'])}")
    prev = err

# ---------------- D4b: forced harmonic, non-resonant
Om = 0.63 * w[0]
P = 1e-4
print("\nD4b forced harmonic Om=0.63*w1 (modal-coordinate error)")
fvec = np.zeros(b.n_dof); fvec[tipdof] = P
Fi = phi.T @ fvec
prev = None
for frac in (10, 20, 40, 80, 160):
    dt = T1 / frac
    loads = lambda t, P=P: {b.n_node - 1: (0.0, P * np.sin(Om * t), 0.0)}
    t, Q, out = run(b, dt, 2 * frac, loads=loads)
    cm = Q @ (b.M @ phi)
    ex = np.zeros((len(t), len(w)))
    for i, wi in enumerate(w):
        ex[:, i] = Fi[i] * (wi * np.sin(Om * t) - Om * np.sin(wi * t)) / (wi**2 - Om**2)
    err = np.linalg.norm(cm[:, :3] - ex[:, :3]) / np.linalg.norm(ex[:, :3])
    rate = float("nan") if prev is None else np.log(prev / err) / np.log(2)
    print(f"  dt=T1/{frac:<4d} err_modal={err:.4e} order={rate:.2f} newton_it={out['max_iterations']}")
    prev = err