#!/usr/bin/env python
"""Wave-11 Track C: Rayleigh-Benard linear-stability eigen solver (Phase 0 ref).

First-principles solver for the 2D Boussinesq RBC linear stability problem
(rigid-rigid no-slip isothermal walls), written from scratch for this campaign.
Nondimensional form (layer depth d_v, time in d_v^2/alpha units, theta scaled
by G*d_v with G = |dT_base/dz| = dT/d_T):

    sigma*(D^2-a^2)Phi = Pr*(D^2-a^2)^2 Phi + Pr*Ra_t*a*Theta    (vorticity)
    sigma*Theta        = (D^2-a^2)Theta - a*Phi                   (heat)
    BC: Phi = DPhi = Theta = 0 at z=0,1 ;  a = k*d_v
    Ra_t = g_beta*G*d_v^4/(nu*alpha)   (critical Ra is Pr-independent)

Derivation (real fields, psi = Phi(z) sin(kx), theta = Theta(z) cos(kx)):
    (1/Pr) du/dt = -grad pi + Ra_t theta zhat + lap u
    dtheta/dt = w + lap theta          (base gradient dTbar/dz = -1)
  curl-y with zeta = lap psi:
    (1/Pr) sigma (D^2-k^2) Phi = Ra_t k Theta + (D^2-k^2)^2 Phi
    sigma Theta = -k Phi + (D^2-k^2) Theta
At sigma = 0 both reduce to -(D^2-a^2)^3 Theta = Ra_t a^2 Theta (positive eigs).

Two independent discretizations, both converted to a standard eig problem:
  spectral: value-space Chebyshev collocation on Gauss-Lobatto nodes; Phi is
    carried as Phi = (1-t^2) w with w vanishing at the boundary nodes (this
    enforces Phi = dPhi = 0 exactly), Theta unknowns at interior nodes
    (Dirichlet by deletion). Equations collocated at interior nodes only.
  fd: 2nd-order centered finite differences on a uniform grid; wall BCs enter
    through ghost elimination (Phi' = 0 at walls -> ghost = mirror). O(h^2).

Literature anchor (Chandrasekhar 1961 rigid-rigid): Ra_c = 1707.762, a_c = 3.1163.
"""

import json
import math
from pathlib import Path

import numpy as np


def cheb(N):
    """Trefethen cheb: nodes t_j = cos(pi j/N) (GL, t in [-1,1]) and D = d/dt."""
    if N == 0:
        return np.array([1.0]), np.array([[0.0]])
    j = np.arange(N + 1)
    t = np.cos(np.pi * j / N)
    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** j)
    X = np.tile(t[:, None], (1, N + 1))
    dX = X - X.T
    D = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return t, D


def barycentric_eval(t_nodes, vals, x):
    """Evaluate interpolant through (t_nodes, vals) at points x (vectorized)."""
    n = len(t_nodes)
    j = np.arange(n)
    w = ((-1.0) ** j) * np.where((j == 0) | (j == n - 1), 0.5, 1.0)
    out = np.empty_like(x, dtype=np.float64)
    for i, xi in enumerate(x):
        d = xi - t_nodes
        hit = np.abs(d) < 1e-14
        if hit.any():
            out[i] = vals[np.argmax(hit)]
        else:
            g = w / d
            out[i] = (g @ vals) / g.sum()
    return out


class SpectralRB:
    """Value-space Chebyshev collocation, exact composite operators.

    Phi = s*w with s = 1-t^2 and w vanishing at the boundary GL nodes (this
    enforces Phi = dPhi/dt = 0 at the walls exactly). Because s*w has degree
    N+2 while the collocation D matrices are exact only to degree N, the
    operators (D_z^2 - a^2)Phi and (D_z^2 - a^2)^2 Phi are assembled through
    the analytic product rule (Trefethen Spectral Methods in MATLAB, program
    28 construction), so every matrix application stays on degree-N data:

      LPhi      = 4(s w'' - 4 t w' - 2 w) - a^2 s w                (D_z=2 D_t)
      L2Phi     = 4[4(s w'''' - 4t w''' - 2w'') - 16(t w''' + 2w'')
                     - 8 w'' - a^2 (s w'' - 4t w' - 2 w)] - a^2 LPhi
    (primes = d/dt, evaluated with the exact collocation matrices).
    """

    def __init__(self, a, N):
        self.a = a
        self.N = N
        self.t, D = cheb(N)
        self.D = D
        self.D2 = D @ D
        self.D3 = self.D2 @ D
        self.D4 = self.D2 @ self.D2
        t = self.t
        s = 1.0 - t**2
        S = np.diag(s)
        T = np.diag(t)
        I = np.eye(N + 1)
        self.s = s
        self.M1 = 4.0 * (S @ self.D2 - 4.0 * (T @ self.D) - 2.0 * I) - (a * a) * S
        inner = (
            4.0 * (S @ self.D4 - 4.0 * (T @ self.D3) - 2.0 * self.D2)
            - 16.0 * (T @ self.D3 + 2.0 * self.D2)
            - 8.0 * self.D2
            - (a * a) * (S @ self.D2 - 4.0 * (T @ self.D) - 2.0 * I)
        )
        self.M2 = 4.0 * inner - (a * a) * self.M1
        self.L = 4.0 * self.D2 - (a * a) * I
        self.ThetaOp = self.L[1:N, 1:N]
        self.Sint = S[1:N, 1:N]  # Phi values at interior nodes from w_int

    def pencil(self, Ra, Pr):
        N = self.N
        ni = N - 1
        A11 = Pr * self.M2[1:N, 1:N]
        A12 = Pr * Ra * self.a * np.eye(ni)
        B11 = self.M1[1:N, 1:N]
        A21 = -self.a * self.Sint
        A22 = self.ThetaOp
        B22 = np.eye(ni)
        A = np.block([[A11, A12], [A21, A22]])
        B = np.block([[B11, np.zeros_like(B11)], [np.zeros_like(B22), B22]])
        return A, B

    def sigma_max(self, Ra, Pr):
        A, B = self.pencil(Ra, Pr)
        C = np.linalg.solve(B, A)
        ev = np.linalg.eigvals(C)
        return float(ev[np.argmax(ev.real)].real)

    def leading_mode(self, Ra, Pr):
        A, B = self.pencil(Ra, Pr)
        C = np.linalg.solve(B, A)
        ev, vec = np.linalg.eig(C)
        i = int(np.argmax(ev.real))
        v = vec[:, i]
        j = int(np.argmax(np.abs(v)))
        v = v * (np.conj(v[j]) / abs(v[j]))
        x = v.real
        N = self.N
        ni = N - 1
        w_all = np.zeros(N + 1)
        w_all[1:N] = x[:ni]
        th_all = np.zeros(N + 1)
        th_all[1:N] = x[ni : 2 * ni]
        Phi_all = self.s * w_all
        dPhi_all = 2.0 * (self.s * (self.D @ w_all) - 2.0 * self.t * w_all)
        scale = th_all[int(np.argmax(np.abs(th_all)))]
        return {
            "t": self.t,
            "Phi": Phi_all / scale,
            "Theta": th_all / scale,
            "dPhi": dPhi_all / scale,
            "sigma": float(ev[i].real),
        }


def fd_pencil(a, Ra, Pr, m):
    """Assemble FD generalized problem A x = sigma B x, x = [Phi_int, Theta_int].

    Grid: z_j = j*h, j=0..m+1 (walls at 0, m+1), h = 1/(m+1).
    Phi BCs: Phi_0 = Phi_{m+1} = 0 and Phi'=0 at walls via ghosts
    (Phi_{-1} = Phi_1, Phi_{m+2} = Phi_m). Theta: Dirichlet by deletion.
    Equations collocated at interior nodes 1..m.
    """
    h = 1.0 / (m + 1)
    a2 = a * a

    def d2_row(j):
        """(D2 u)_j on extended index (supports ghosts) -> dict e->coef."""
        return {j - 1: 1.0 / h**2, j: -2.0 / h**2, j + 1: 1.0 / h**2}

    def apply_L(u):  # u: dict j -> value (L = D2 - a^2)
        out = {}
        for j in u:
            r = d2_row(j)
            out[j] = -a2 * u[j] + sum(c * u.get(k, 0.0) for k, c in r.items())
        return out

    def compose(matfun):
        """Numerically build operator columns on interior unknowns."""
        cols = []
        for col in range(m):
            u = {j: 0.0 for j in range(0, m + 2)}
            # unit vector at interior j=col+1
            u[col + 1] = 1.0
            # ghost elimination for Phi BCs
            u[-1] = u[1]
            u[m + 2] = u[m]
            v = matfun(u)
            cols.append([v.get(j, 0.0) for j in range(1, m + 1)])
        return np.array(cols).T

    # Eq1 rows: sigma * (L Phi)_int = Pr (L^2 Phi)_int + Pr Ra a Theta_int
    LPhi = compose(apply_L)
    L2Phi = compose(lambda u: apply_L(apply_L(u)))
    A11 = Pr * L2Phi
    A12 = Pr * Ra * a * np.eye(m)
    B11 = LPhi
    # Eq2 rows: sigma Theta = (L Theta)_int - a Phi_int (Theta Dirichlet)
    A21 = -a * np.eye(m)
    # Theta operator: Dirichlet BC (no ghosts, walls = 0), tridiagonal
    Lth = np.zeros((m, m))
    for i in range(m):
        if i - 1 >= 0:
            Lth[i, i - 1] = 1.0 / h**2
        Lth[i, i] = -2.0 / h**2 - a2
        if i + 1 < m:
            Lth[i, i + 1] = 1.0 / h**2
    A22 = Lth
    A = np.block([[A11, A12], [A21, A22]])
    B = np.block([[B11, np.zeros_like(B11)], [np.zeros((m, m)), np.eye(m)]])
    return A, B


def fd_sigma_max(a, Ra, Pr, m):
    A, B = fd_pencil(a, Ra, Pr, m)
    C = np.linalg.solve(B, A)
    ev = np.linalg.eigvals(C)
    return float(ev[np.argmax(ev.real)].real)


def sigma_spectral(a, Ra, Pr, N):
    return SpectralRB(a, N).sigma_max(Ra, Pr)


def ra_critical(a, Pr, solver, lo=100.0, hi=20000.0, iters=200):
    s_lo = solver(a, lo, Pr)
    s_hi = solver(a, hi, Pr)
    assert s_lo < 0 < s_hi, (a, s_lo, s_hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = solver(a, mid, Pr)
        if s > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-11 * max(1.0, lo):
            break
    return 0.5 * (lo + hi)


def golden_min(f, lo, hi, tol=1e-11):
    gr = (math.sqrt(5) - 1) / 2
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(400):
        if hi - lo < tol:
            break
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = f(d)
    x = 0.5 * (lo + hi)
    return x, f(x)


def _selftest():
    N = 40
    sp = SpectralRB(1.0, N)
    # Theta-only check: eig(ThetaOp) ~ -(n*pi)^2 - a^2 (z units)
    ev = np.sort(np.linalg.eigvals(sp.ThetaOp).real)
    exact = [-((n * math.pi) ** 2) - 1.0 for n in range(1, 6)]
    top5 = ev[-5:][::-1]  # the 5 largest (least negative), descending
    ok = bool(np.allclose(top5, exact, rtol=1e-8, atol=1e-8))
    # FD Dirichlet Laplacian check (O(h^2): tolerance 5e-3 at m=200)
    m = 200
    h = 1.0 / (m + 1)
    Lth = (
        np.diag(-2.0 * np.ones(m)) + np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1)
    ) / h**2 - np.eye(m)
    evf = np.sort(np.linalg.eigvals(Lth).real)
    ok &= bool(np.allclose(evf[-5:][::-1], exact, rtol=5e-3, atol=5e-3))
    # at Ra=0 every mode must decay (no buoyancy -> no growth)
    sp0 = SpectralRB(2.0, N)
    A, B = sp0.pencil(0.0, 1.0)
    C = np.linalg.solve(B, A)
    ev0 = np.linalg.eigvals(C)
    ok &= bool(ev0.real.max() < 1e-6)
    return "SELFTEST_OK" if ok else "SELFTEST_FAIL"


def main():
    print(_selftest())
    Pr = 0.71
    N_sp = 64
    m_fd = 320
    solver_sp = lambda a, Ra, Pr: sigma_spectral(a, Ra, Pr, N_sp)
    solver_fd = lambda a, Ra, Pr: fd_sigma_max(a, Ra, Pr, m_fd)

    out = {"pr": Pr, "N_spectral": N_sp, "m_fd": m_fd}
    anchor = {}
    for name, solver in (("spectral", solver_sp), ("fd", solver_fd)):
        ac, rac = golden_min(lambda a: ra_critical(a, Pr, solver), 2.5, 4.0)
        anchor[name] = {"a_c": ac, "Ra_c": rac}
    rac_pr7 = ra_critical(anchor["spectral"]["a_c"], 7.0, solver_sp)
    anchor["pr_independence"] = {
        "Ra_c_at_Pr0.71": anchor["spectral"]["Ra_c"],
        "Ra_c_at_Pr7.0": rac_pr7,
        "abs_diff": abs(rac_pr7 - anchor["spectral"]["Ra_c"]),
    }
    anchor["literature"] = {
        "Ra_c": 1707.762,
        "a_c": 3.1163,
        "source": "Chandrasekhar 1961 (rigid-rigid, Boussinesq)",
    }
    anchor["abs_dev"] = {
        "spectral_Ra_c": abs(anchor["spectral"]["Ra_c"] - 1707.762),
        "spectral_a_c": abs(anchor["spectral"]["a_c"] - 3.1163),
        "fd_Ra_c": abs(anchor["fd"]["Ra_c"] - 1707.762),
    }
    out["anchor"] = anchor

    ac = anchor["spectral"]["a_c"]
    conv = []
    for N in (16, 24, 32, 48, 64, 96):
        conv.append(
            {
                "method": "spectral",
                "resolution": N,
                "Ra_c_at_ac": ra_critical(ac, Pr, solver_sp)
                if N == N_sp
                else ra_critical(ac, Pr, lambda a, Ra, Pr: sigma_spectral(a, Ra, Pr, N)),
            }
        )
    for m in (80, 160, 320, 640):
        conv.append(
            {
                "method": "fd",
                "resolution": m,
                "Ra_c_at_ac": ra_critical(ac, Pr, lambda a, Ra, Pr: fd_sigma_max(a, Ra, Pr, m)),
            }
        )
    out["convergence"] = conv

    xcheck = []
    for Ra in (1707.762, 1800.0, 2000.0, 2220.0):
        for a in (3.0, math.pi, 3.25):
            sA = sigma_spectral(a, Ra, Pr, 64)
            sB = fd_sigma_max(a, Ra, Pr, 320)
            xcheck.append(
                {
                    "Ra": Ra,
                    "a": a,
                    "sigma_spectral": sA,
                    "sigma_fd": sB,
                    "rel_diff": abs(sA - sB) / max(1e-12, abs(sA)),
                }
            )
    out["sigma_crosscheck"] = xcheck

    p = Path("/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/ref")
    (p / "anchor_and_convergence.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(anchor, indent=1))
    print("crosscheck max rel diff =", max(d["rel_diff"] for d in xcheck))
    print("ANCHOR_DONE")


if __name__ == "__main__":
    main()
