"""Prototype v2: exact linear (QSSA) stability of a miscible Hele-Shaw front.

lap(psi) + R (c_x psi_x + c_y psi_y) = 0      (Darcy, mu = exp(R c))
base: cbar(xi) = 0.5(1-tanh(xi/w)), psi0 = y (unit base flow along +x), xi = x - t
perturbation ~ exp(i k y + sigma t):
  psi'' - k^2 psi + R cbar' psi' = -i k R c
  sigma c = c'' - k^2 c - i k cbar' psi
Eliminate psi (A = D2 + R diag(cbar') D - k^2 I):
  sigma c = (D2 - k^2 I) c - k^2 R diag(cbar') A^{-1} c
"""
import numpy as np


def sigma_exact(k, R, w, N=1201, L=400.0):
    xi = np.linspace(-L, L, N)
    h = xi[1] - xi[0]
    dcb = -0.5 / w / np.cosh(np.clip(xi / w, -60, 60)) ** 2
    e = np.ones(N)
    D2 = (np.diag(e[1:], 1) - 2 * np.diag(e) + np.diag(e[1:], -1)) / h**2
    D = (np.diag(e[1:], 1) - np.diag(e[1:], -1)) / (2 * h)
    sl = (slice(1, -1), slice(1, -1))
    D2i, Di = D2[sl], D[sl]
    Ii = np.eye(N - 2)
    dci = dcb[1:-1]
    A = D2i + R * np.diag(dci) @ Di - k * k * Ii
    psi_op = np.linalg.solve(A, Ii)          # A^{-1}
    M = D2i - k * k * Ii - (k * k * R) * (np.diag(dci) @ psi_op)
    ev = np.linalg.eigvals(M)
    i = int(np.argmax(ev.real))
    return complex(ev[i]), float(h)


def sigma_step(R, k):
    """Sharp-step analytic dispersion (same linearised system, cbar' = -delta)."""
    roots = np.roots([2.0, 2.0 * k, k * R])          # 2 kappa^2 + 2 k kappa + k R = 0
    out = []
    for kap in roots:
        out.append((kap, kap ** 2 - k * k))
    return out


if __name__ == "__main__":
    k = 2 * np.pi * 4 / 500.0
    print(f"k = {k:.6e}, k^2 = {k*k:.6e}")
    print("--- module-measured sigma (round-1 metrics_growth.json) ---")
    for R, s in [(0.05, -1.5971e-3), (0.2, 1.1931e-3), (0.5, 6.7722e-3),
                 (1.0, 1.6025e-2), (2.0, 3.4213e-2), (4.0, 6.9494e-2)]:
        print(f"  R={R:<5} sigma_module={s:+.6e}")
    print("--- exact variable-coefficient linear theory, tanh front ---")
    for R in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        for w in (4.0, 2.0):
            s, h = sigma_exact(k, R, w)
            print(f"  R={R:<5} w={w:<4} h={h:.3f} sigma={s.real:+.6e} {s.imag:+.3e}i")
    print("--- sharp-step analytic roots ---")
    for R in (-2.0, -1.0, 1.0, 2.0):
        for kap, s in sigma_step(R, k):
            print(f"  R={R:<5} kappa={kap:+.4f} sigma={s:+.6e}")