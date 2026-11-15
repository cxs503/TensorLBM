import numpy as np
from tensorlbm.corotational_beam import straight_beam

CLAMP = [(0, "x"), (0, "y"), (0, "theta")]
b = straight_beam(1.0, 8, E=1.0, A=1.2e5, I=1.0, rho=1.0, bc=CLAMP)
tip = 3 * (b.n_node - 1) + 1
q0, _ = b.solve_static({b.n_node - 1: (0.0, -0.004, 0.0)}, n_steps=1)
print("q0 tip:", q0[tip], " tip rot:", q0[3 * (b.n_node - 1) + 2])
K = b.linear_stiffness()
print("linear K q0 - f  (should be 0):", np.linalg.norm(K @ q0 - b.external_load({b.n_node - 1: (0.0, -0.004, 0.0)})))
w1 = b.mode_frequencies(1)[0]
T1 = 2 * np.pi / w1
print("w1", w1, "T1", T1)

dt = T1 / 160
rec = []
out = b.solve_newmark(dt=dt, n_steps=640, q0=q0,
                      sample=lambda s, t, q, qd, qdd: rec.append((t, q[tip], qd[tip])))
t = np.array([r[0] for r in rec]); u = np.array([r[1] for r in rec])
print("tip start %.6e  min %.6e  max %.6e" % (u[0], u.min(), u.max()))
# frequency of the response by zero crossings
z = np.where(np.diff(np.sign(u - u.mean())))[0]
print("half periods (s):", np.diff(t[z])[:6], " expected", T1 / 2)
# energy via solver velocities
E = [0.5 * 0 + b.kinetic_energy(r[2]) for r in [(0, 0, np.zeros(b.n_dof))]]
print("newton converged:", out["converged"], "max it", out["max_iterations"],
      "floor", sum(r.floor_limited for r in out["newton"]))
print("first 3 residuals of step1:", out["newton"][0].residuals[:5])
print("residual of step 300:", out["newton"][299].residuals[:5])