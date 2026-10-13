"""Independent solver: steady 2D NS past a circular cylinder (FD/pseudo-spectral).

Streamfunction-vorticity, polar coordinates. Fourier in theta (FFT), 2nd-order
nonuniform FD in r on a clustered grid. IMEX time stepping: Crank-Nicolson
implicit for the linear diffusion part (batched Thomas tridiagonal solves,
one system per Fourier mode, LU-factorized once at t=0), forward Euler for
convection. Wall vorticity: 2nd-order one-sided formula from wall psi.

Scaling: U=1, D=1, R=0.5, rho=1. Re = U*D/nu. Cd = F_x/(0.5*rho*U^2*D).
Wall pressure from tangential momentum: dp/dtheta = rho*nu*R*(dw/dr)_w.
Friction from tau_w = mu*omega_w. theta measured from downstream axis;
separation angle from upstream stagnation = 180deg - theta_sep.

Usage: indep_solver.py OUT.json [smoke|base|refine]
"""

from __future__ import annotations

import json
import sys

import numpy as np

R = 0.5
TWO_PI = 2.0 * np.pi


def radial_grid(n_r: int, r_max: float, beta: float) -> np.ndarray:
    xi = np.linspace(0.0, 1.0, n_r)
    s = (np.exp(beta * xi) - 1.0) / (np.exp(beta) - 1.0)
    return R + (r_max - R) * s


class BatchedTri:
    """Batch of tridiagonal systems (M modes), real bands, complex rhs.

    Matrices are constant: LU (Thomas) factorization is stored once;
    each solve is forward-elimination + back-substitution, vectorized over M.
    """

    def __init__(self, lo: np.ndarray, di: np.ndarray, up: np.ndarray):
        # lo, di, up: (M, n); lo[i,0], up[i,n-1] unused
        M, n = di.shape
        c = np.zeros_like(up)
        d = np.zeros_like(di)
        c[:, 0] = up[:, 0] / di[:, 0]
        d[:, 0] = di[:, 0]
        for i in range(1, n):
            w = lo[:, i] / d[:, i - 1]
            d[:, i] = di[:, i] - w * up[:, i - 1]
            if i < n - 1:
                c[:, i] = up[:, i] / d[:, i]
        self.c, self.d = c, d

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        c, d = self.c, self.d
        n = rhs.shape[1]
        x = np.empty_like(rhs)
        x[:, 0] = rhs[:, 0] / d[:, 0]
        for i in range(1, n):
            x[:, i] = (rhs[:, i] - self.lo_work(x, i)) / d[:, i]
        # back substitution
        for i in range(n - 2, -1, -1):
            x[:, i] = x[:, i] - c[:, i] * x[:, i + 1]
        return x

    def lo_work(self, x, i):
        return self._lo[:, i] * x[:, i - 1]

    def set_lower(self, lo):
        self._lo = lo


def make_solver(lo, di, up):
    bt = BatchedTri(lo.copy(), di.copy(), up.copy())
    bt.set_lower(lo)
    return bt


def tri_apply_batched(lo, di, up, x):
    y = di * x
    y[:, :-1] += up[:, :-1] * x[:, 1:]
    y[:, 1:] += lo[:, 1:] * x[:, :-1]
    return y


class Grid:
    def __init__(self, r: np.ndarray, n_th: int):
        self.r = r
        self.n = len(r)
        self.n_th = n_th
        h_m = r[1:-1] - r[:-2]
        h_p = r[2:] - r[1:-1]
        idx = np.arange(self.n - 2) + 1
        self.d1 = np.zeros((self.n - 2, 3))
        self.d1[:, 0] = -h_p / (h_m * (h_m + h_p))
        self.d1[:, 1] = (h_p - h_m) / (h_m * h_p)
        self.d1[:, 2] = h_m / (h_p * (h_m + h_p))
        self.d2 = np.zeros((self.n - 2, 3))
        self.d2[:, 0] = 2.0 / (h_m * (h_m + h_p))
        self.d2[:, 1] = -2.0 / (h_m * h_p)
        self.d2[:, 2] = 2.0 / (h_p * (h_m + h_p))
        self.idx = idx
        self.h1 = r[1] - r[0]
        self.h2 = r[2] - r[1]
        self.ri = r[1:-1]
        self.k = np.fft.fftfreq(n_th) * n_th
        self.th = TWO_PI * np.arange(n_th) / n_th
        self.M = n_th
        # exponential theta-filter: sigma_k = exp(-filt_alpha*(|k|/kmax)^8)
        self.filt = np.exp(-0.05 * (np.abs(self.k) / (n_th / 2.0)) ** 8)
        # per-mode tridiagonal bands for L_k = d2 + (1/r) d1 - k^2/r^2
        # shape (M, n): row i of bands corresponds to k[i]
        n = self.n
        lo = np.zeros((self.M, n))
        di = np.ones((self.M, n))
        up = np.zeros((self.M, n))
        lo[:, idx] = self.d2[:, 0] + self.d1[:, 0] / self.ri
        di[:, idx] = self.d2[:, 1] + self.d1[:, 1] / self.ri
        up[:, idx] = self.d2[:, 2] + self.d1[:, 2] / self.ri
        di[:, idx] -= self.k[:, None] ** 2 / self.ri[None, :] ** 2
        di[:, 0] = 1.0
        di[:, -1] = 1.0
        self.lo, self.di, self.up = lo, di, up

    def deriv_r(self, f):
        df = np.empty_like(f)
        df[self.idx] = (
            self.d1[:, 0:1] * f[self.idx - 1]
            + self.d1[:, 1:2] * f[self.idx]
            + self.d1[:, 2:3] * f[self.idx + 1]
        )
        df[0] = (f[1] - f[0]) / self.h1
        df[-1] = (f[-1] - f[-2]) / (self.r[-1] - self.r[-2])
        return df

    def theta_filter(self, f):
        return np.fft.ifft(np.fft.fft(f, axis=1) * self.filt[None, :], axis=1).real

    def d_theta(self, f):
        return np.fft.ifft(np.fft.fft(f, axis=1) * (1j * self.k)[None, :], axis=1).real

    def poisson(self, omega):
        """Lap psi = -omega; psi(R)=-R sin th; psi(rmax)=0."""
        om_h = np.fft.fft(omega, axis=1)  # (n, M)
        rhs = -om_h.T.copy()  # (M, n)
        rhs[:, 0] = 0.0
        rhs[:, -1] = 0.0
        i1 = np.where(self.k == 1)[0]
        if len(i1):
            rhs[i1, 0] = -R * self.M / (2j)
        im1 = np.where(self.k == -1)[0]
        if len(im1):
            rhs[im1, 0] = R * self.M / (2j)
        # outer: psi(rmax) = -R^2/rmax sin(th) (potential perturbation)
        rm = self.r[-1]
        if len(i1):
            rhs[i1, -1] = -R * R / rm * self.M / (2j)
        if len(im1):
            rhs[im1, -1] = R * R / rm * self.M / (2j)
        psi_h = self.poisson_solver.solve(rhs)
        return np.fft.ifft(psi_h.T, axis=1).real

    def wall_omega(self, psi, omega):
        """Mode-wise Woods 2nd-order wall vorticity for PERTURBATION psi.

        psi is the perturbation (total = psi + r sin(th)); wall Dirichlet
        psi(R) = -R sin(th); no-slip gives psi_r(R) = -sin(th).
        Per Fourier mode k:
          omega_0,k = -3 dpsi_k/h1^2 + 3 psi'_0,k/h1
                      - 1.5 psi'_0,k/R + 1.5 k^2 psi_0,k/R^2 - omega_1,k/2
        with psi_0,k and psi'_0,k nonzero only for k = +-1
        (psi_0 = -R sin, psi'_0 = -sin).
        """
        h1 = self.h1
        dpsi_h = np.fft.fft(psi[1, :] - psi[0, :])
        om1_h = np.fft.fft(omega[1, :])
        psi0_h = np.zeros(len(self.k), dtype=complex)
        psip0_h = np.zeros(len(self.k), dtype=complex)
        i1 = np.where(self.k == 1)[0][0]
        im1 = np.where(self.k == -1)[0][0]
        # plain e^{ik th} coefficients of -R sin(th) and -sin(th)
        M = self.M  # ifft carries 1/M: coefficients must be unnormalized
        psi0_h[i1] = -R * M / (2j)
        psi0_h[im1] = R * M / (2j)
        psip0_h[i1] = -M / (2j)
        psip0_h[im1] = M / (2j)
        om_w_h = (
            -3.0 * dpsi_h / (h1 * h1)
            + 3.0 * psip0_h / h1
            - 1.5 * psip0_h / R
            + 1.5 * (self.k**2) * psi0_h / (R * R)
            - 0.5 * om1_h
        )
        return np.fft.ifft(om_w_h).real


def run_steady(
    re,
    n_r,
    n_th,
    r_max,
    beta,
    cfl=0.4,
    tol=1e-8,
    save_field=None,
    max_iter=200000,
    log_every=2000,
    wall_relax=0.5,
    filt_alpha=0.02,
):
    """Steady state via local pseudo-time explicit iteration.

    No time accuracy needed: per-node pseudo dt limited by local diffusion
    (radial FD + conservative spectral theta) and advection CFL. The
    impulsive-sheet phase is stable because the sheet cells take tiny
    local steps. Woods wall update is a fixed-point condition (relaxed).
    """
    nu = 1.0 / re
    r = radial_grid(n_r, r_max, beta)
    g = Grid(r, n_th)
    g.filt = np.exp(-filt_alpha * (np.abs(g.k) / (n_th / 2.0)) ** 8)
    g.poisson_solver = make_solver(g.lo, g.di, g.up)
    idx = g.idx
    # local node spacing (conservative min of backward/forward)
    h_node = np.minimum(r[1:] - r[:-1], 0.0)
    h_node = np.zeros(g.n)
    h_node[1:-1] = np.minimum(r[idx] - r[idx - 1], r[idx + 1] - r[idx])
    h_node[0] = g.h1
    h_node[-1] = r[-1] - r[-2]
    kmax = float(n_th / 2.0)
    omega = np.zeros((g.n, n_th))
    prev = omega.copy()
    hist = []
    cd_hist: list[float] = []
    stop_reason = "max_iter"
    for it in range(1, max_iter + 1):
        psi = g.poisson(omega)
        cos_t = np.cos(g.th)[None, :]
        sin_t = np.sin(g.th)[None, :]
        u_r = g.d_theta(psi) / r[:, None] + cos_t  # + free stream U cos(th)
        u_th = -g.deriv_r(psi) - sin_t  # - free stream U sin(th)
        dom_r = g.deriv_r(omega)
        dom_th = g.d_theta(omega)
        adv = -(u_r * dom_r + u_th / r[:, None] * dom_th)
        # spectral-mode laplacian of omega (fourier, exact for bands)
        om_h = np.fft.fft(omega, axis=1).T
        lap_h = tri_apply_batched(g.lo, g.di, g.up, om_h)
        lap = np.fft.ifft(lap_h.T, axis=1).real
        rate = (
            nu * (4.0 / h_node**2 + kmax**2 / r**2)[:, None]
            + np.abs(u_r) / h_node[:, None]
            + np.abs(u_th) * kmax / r[:, None]
        )
        dt_loc = cfl / rate  # per (node, theta) local pseudo step
        omega = omega + dt_loc * (nu * lap + adv)
        omega[0, :] = (1.0 - wall_relax) * omega[0, :] + wall_relax * g.wall_omega(psi, omega)
        omega[-1, :] = 0.0
        omega = g.theta_filter(omega)
        if it % log_every == 0 and it > 1:
            res = float(np.max(np.abs(nu * lap + adv)[1:-1]))
            m = float(np.max(np.abs(omega)))
            d = diagnose(g, omega, nu)
            hist.append({"iter": it, "res": res, "maxom": m, "cd": d["cd"]})
            print(
                f"  Re={re} it={it} res={res:.3e} max|om|={m:.3e} "
                f"Cd={d['cd']:.5f} (p={d['cd_p']:.4f} f={d['cd_f']:.4f})",
                flush=True,
            )
            cd_hist.append(d["cd"])
            # Cd plateau exit: max-min over last 6 reports < 1e-3 absolute,
            # only after the impulsive transient (it > 20000)
            if res < tol:
                stop_reason = "res_tol"
                break
            if it > 20000 and len(cd_hist) >= 6:
                if max(cd_hist[-6:]) - min(cd_hist[-6:]) < 1e-3:
                    stop_reason = "cd_plateau"
                    break
            prev = omega.copy()
    d = diagnose(g, omega, nu)
    d.update(
        {
            "re": re,
            "n_r": n_r,
            "n_th": n_th,
            "r_max": r_max,
            "beta": beta,
            "cfl": cfl,
            "filt_alpha": filt_alpha,
            "iters": it,
            "stop_reason": stop_reason,
            "res": float(np.max(np.abs(nu * lap + adv)[1:-1])),
            "hist_tail": hist[-5:],
        }
    )
    if save_field:
        np.savez(save_field, omega=omega, psi=psi, r=r, th=g.th)
    return d


def diagnose(g, omega, nu):
    psi = g.poisson(omega)
    om_w = omega[0, :]
    h1, h2 = g.h1, g.h2
    dom_dr_w = (
        -(2 * h1 + h2) / (h1 * (h1 + h2)) * om_w
        + (h1 + h2) / (h1 * h2) * omega[1, :]
        - h1 / (h2 * (h1 + h2)) * omega[2, :]
    )
    dth = g.th[1] - g.th[0]
    p = np.cumsum(nu * R * dom_dr_w) * dth
    p -= p.mean()
    cd_p = float(np.sum(-p * np.cos(g.th)) * dth)
    cd_f = float(np.sum(-nu * om_w * np.sin(g.th)) * dth)
    alpha = None
    th = g.th
    for j in range(len(th) - 1):
        if 0.05 < th[j] < np.pi - 0.05 and om_w[j] * om_w[j + 1] < 0:
            t1, t2 = th[j], th[j + 1]
            w1, w2 = om_w[j], om_w[j + 1]
            t0 = t1 - w1 * (t2 - t1) / (w2 - w1)
            alpha = float(np.degrees(np.pi - t0))
            break
    u_r = g.d_theta(psi) / g.r[:, None] + np.cos(g.th)[None, :]
    ur_c = u_r[:, 0]
    lw = None
    jmax = int(np.argmin(np.abs(g.r - 6.0)))
    for j in range(2, jmax):
        if ur_c[j] * ur_c[j + 1] < 0 and ur_c[j + 1] > 0:
            r1, r2 = g.r[j], g.r[j + 1]
            w1, w2 = ur_c[j], ur_c[j + 1]
            r0 = r1 - w1 * (r2 - r1) / (w2 - w1)
            lw = float(2 * (r0 - R))
            break
    u_th_w = -g.deriv_r(psi)[0, :] - np.sin(g.th)
    u_r_w = g.d_theta(psi)[0, :] / R + np.cos(g.th)
    p_jump = float(p[-1] + nu * R * dom_dr_w[-1] * (g.th[1] - g.th[0]) - p[0])
    return {
        "cd": cd_p + cd_f,
        "cd_p": cd_p,
        "cd_f": cd_f,
        "theta_s_deg": alpha,
        "lw_over_d": lw,
        "u_th_wall_max": float(np.max(np.abs(u_th_w))),
        "u_r_wall_max": float(np.max(np.abs(u_r_w))),
        "p_period_jump": p_jump,
        "om_w_max": float(np.max(np.abs(om_w))),
    }


CASES = {
    "smoke": (40.0, 120, 64, 8.0, 3.0),
    "base0": (10.0, 240, 128, 12.0, 2.5),
    "base1": (20.0, 240, 128, 12.0, 2.5),
    "base2": (30.0, 240, 128, 12.0, 2.5),
    "base3": (40.0, 240, 128, 12.0, 2.5),
    "refine1": (20.0, 360, 192, 24.0, 3.0),
    "refine2": (30.0, 360, 192, 24.0, 3.0),
    "refine3": (40.0, 360, 192, 24.0, 3.0),
    "rmax0": (40.0, 240, 128, 6.0, 2.0),
    "rmax1": (40.0, 240, 128, 24.0, 3.5),
    "th0": (40.0, 240, 96, 12.0, 2.5),
    "th1": (40.0, 240, 192, 12.0, 2.5),
}

if __name__ == "__main__":
    out_path, name = sys.argv[1], sys.argv[2]
    re, n_r, n_th, r_max, beta = CASES[name]
    sf = out_path.replace(".json", "_field.npz")
    d = run_steady(re, n_r, n_th, r_max, beta, save_field=sf)
    with open(out_path, "w") as fh:
        json.dump(d, fh, indent=1)
    print("written", out_path)
