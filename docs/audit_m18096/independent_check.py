#!/usr/bin/env python3
"""Independent re-derivation of the |cc|-proxy closed form (auditor, m18149).

Deliberately does NOT import tensorlbm and does NOT reuse the mission's
``reference.py``: it rebuilds D2Q9 weights/velocities from scratch, forms the
Chapman-Enskog non-equilibrium population by hand, and evaluates

    S2 = sum_i w_i^2 * EEk_i * A_i^2 ,   EEk_i = (|c_ix| + |c_iy|)^2

for random incompressible strain tensors, testing the two statements the
mission's E3 makes:

    (i)  S2 == |S|^2 / 81            with |S| = sqrt(2 S:S)
    (ii) a = Q/(3 rho tau |S|) == sqrt(2)/9   (orientation independent)
"""
import numpy as np

W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
              [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=float)
EEK = np.abs(C).sum(axis=1) ** 2            # [0,1,1,1,1,4,4,4,4]
CS2 = 1.0 / 3.0


def check(S):
    Sxx, Syy, Sxy = S
    A = ((C[:, 0] ** 2 - CS2) * Sxx + (C[:, 1] ** 2 - CS2) * Syy
         + 2.0 * C[:, 0] * C[:, 1] * Sxy)
    S2 = float((W ** 2 * EEK * A ** 2).sum())
    S_mag = np.sqrt(2.0 * (Sxx ** 2 + Syy ** 2 + 2.0 * Sxy ** 2))
    # CE-consistent perturbation: f_neq = -3 w rho tau A  -> Q^2 = 2 sum f_neq^2 EEk
    rho = tau = 1.0
    fneq = -3.0 * W * rho * tau * A
    Q = np.sqrt(2.0 * (fneq ** 2 * EEK).sum())
    a = Q / (3.0 * rho * tau * S_mag)
    return S2, S_mag ** 2 / 81.0, a


rng = np.random.default_rng(20261113)
a_vals, ratio_vals = [], []
for _ in range(2000):
    Sxx, Sxy = rng.normal(size=2) * 0.02
    s2, s2_closed, a = check((Sxx, -Sxx, Sxy))
    a_vals.append(a)
    ratio_vals.append(s2 / s2_closed if s2_closed > 0 else np.nan)
a_vals = np.array(a_vals)
print("trials                       : 2000 random incompressible S")
print("max |S2/(S^2/81) - 1|        : %.3e" % np.nanmax(np.abs(np.array(ratio_vals) - 1)))
print("a min/mean/max               : %.15f / %.15f / %.15f"
      % (a_vals.min(), a_vals.mean(), a_vals.max()))
print("closed form sqrt(2)/9        : %.15f" % (np.sqrt(2) / 9))
print("max rel dev vs sqrt(2)/9     : %.3e"
      % np.max(np.abs(a_vals - np.sqrt(2) / 9) / (np.sqrt(2) / 9)))
# sweep of the 24 orientations used by the mission's E3
phi = np.pi * np.arange(24) / 24
a24 = np.array([check((0.01 * np.cos(p), -0.01 * np.cos(p), 0.01 * np.sin(p)))[2] for p in phi])
print("24-orientation spread (max-min)/mean : %.3e"
      % ((a24.max() - a24.min()) / a24.mean()))
np.save("/tmp/m18149_audit/a24.npy", a24)