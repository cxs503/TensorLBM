"""Prototype (vectorised): exact linear (QSSA) stability of a miscible Hele-Shaw front.

lap(psi) + R (c_x psi_x + c_y psi_y) = 0      (Darcy, mu = exp(R c))
base: cbar(xi), psi0 = y (unit base flow along x), xi = x - t
perturbation ~ exp(i k y + sigma t):
  psi'' - k^2 psi + R cbar' psi' = -i k R c
  sigma c = c'' - k^2 c - i k cbar' psi
Eliminate psi (A = D2 + R diag(cbar') D - k^2 I):
  sigma c = (D2 - k^2 I) c - k^2 R diag(cbar') A^{-1} c
"""
import numpy as np


def sigma_exact(k, R, w, N=401, halfwidth_scale=60.0):
    L = halfwidth_scale * max(w, 1.0 / k)
    xi = np.linspace(-L, L, N)
    h = xi[1] - xi[0]
    dcb = -0.5 / w / np.cosh(np.clip(xi / w, -50, 50)) ** 2
    e = np.ones(N)
    D2 = (np.diag(e[1:], 1) - 2 * np.diag(e) + np.diag(e[1:], -1)) / h**2
    D = (np.diag(e[1:], 1) - np.diag(e[1:], -1)) / (2 * h)
    I = np.eye(N)
    sl = (slice(1, -1), slice(1, -1))
    D2i, Di, Ii = D2[sl], D[sl], I[slice(1, -1), slice(1, -1)]
    dci = dcb[1:-1]
    A = D2i + R * np.diag(dci) @ Di - k * k * Ii
    M = D2i - k * k * Ii - (k * k * R) * (np.diag(dci) @ np.linalg.solve(A, Ii))
    ev = np.linalg.eigvals(M)
    return float(ev.real.max())


if __name__ == "__main__":
    k = 2 * np.pi * 4 / 500.0
    for R in (0.1, 1.0, 2.0):
        st = k * np.tanh(R / 2) - k * k
        print(f"R={R}: ST anchor {st:+.6e}")
        for w in (40.0, 20.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1):
            s = sigma_exact(k, R, w)
            print(f"   w={w:<6} sigma={s:+.6e}  dev_vs_ST={100*(s-st)/st:+7.2f}%  k*w={k*w:.3f}")
    print("R=0 check (sigma should be -k^2 =", -k * k, ")")
    for w in (4.0, 1.0):
        print("  w=", w, sigma_exact(k, 0.0, w))