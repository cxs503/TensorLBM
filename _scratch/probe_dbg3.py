import numpy as np
from tensorlbm.corotational_beam import straight_beam

CLAMP = [(0, "x"), (0, "y"), (0, "theta")]
b = straight_beam(1.0, 8, E=1.0, A=1.2e5, I=1.0, rho=1.0, bc=CLAMP)
tip = 3 * (b.n_node - 1) + 1
w1 = b.mode_frequencies(1)[0]
T1 = 2 * np.pi / w1
Om = 0.63 * w1
P = 1e-4

# static check of the load application
f = b.external_load({b.n_node - 1: (0.0, P, 0.0)})
print("f tip:", f[tip], " f norm:", np.linalg.norm(f))
qs, _ = b.solve_static({b.n_node - 1: (0.0, P, 0.0)}, n_steps=1)
print("static tip:", qs[tip], " PL^3/3EI =", P / 3.0)

dt = T1 / 80
hist = []
out = b.solve_newmark(dt=dt, n_steps=160, loads=lambda t: {b.n_node - 1: (0.0, P * np.sin(Om * t), 0.0)},
                      sample=lambda s, t, q, qd, qdd: hist.append((t, q[tip])))
t = np.array([h[0] for h in hist]); u = np.array([h[1] for h in hist])
print("tip series: max %.3e  final %.3e" % (np.abs(u).max(), u[-1]))
print("first 6:", u[:6])
print("expected amplitude ~ %.3e" % (P / 3.0 / (1 - 0.63**2)))
print("newton iters:", [r.iterations for r in out["newton"][:6]], "conv:", out["converged"])
print("residuals step 5:", out["newton"][5].residuals)
# gravity?
print("gravity:", b.gravity, "external_load(None) norm:", np.linalg.norm(b.external_load(None)))