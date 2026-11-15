"""Independent probe: are the anchor's physics references the ones the solver's
own operator implies?

Written from the definitions only (imports the D3Q19 velocity set / weights
constants, nothing else).

    PYTHONPATH=src python tools/xval_host_staged_halo/audit_references_m21930.py
"""
import json
import math
import sys

import numpy as np

from tensorlbm.host_staged_halo import D3Q19_C, D3Q19_W  # constants only

C = np.array(D3Q19_C, dtype=np.float64)
W = np.array(D3Q19_W, dtype=np.float64)


def omega(nu):
    return 1.0 / (3.0 * nu + 0.5)


def step_op(kvec, nu):
    proj = W[:, None] * (1.0 + 3.0 * (C @ C.T))
    phase = np.exp(-1j * (C @ np.asarray(kvec, dtype=np.float64)))
    return phase[:, None] * (np.eye(19) - omega(nu) * (np.eye(19) - proj))


def transverse_candidates(kvec, nu):
    A = step_op(kvec, nu)
    vals, vecs = np.linalg.eig(A)
    k = np.asarray(kvec, float)
    out = []
    for j, (lam, vec) in enumerate(zip(vals, vecs)):
        if abs(lam) >= 1.0:
            continue
        mass = abs(complex(vec.sum()))
        mom = C.T @ vec
        nm = np.linalg.norm(mom)
        if mass > 1e-9 or nm < 1e-12:
            continue
        if abs(np.dot(k, mom)) / nm > 1e-8:
            continue
        out.append({"j": j, "rate": -math.log(abs(lam)), "mass": mass,
                    "mom_dir": (mom / nm).real.tolist()})
    out.sort(key=lambda r: r["rate"])
    return out


def mode_prediction(kvec, nu, dfhat, comp, t1, t2):
    """Exact linear prediction of |momentum component| slope between t1 and t2."""
    A = step_op(kvec, nu)
    vals, vecs = np.linalg.eig(A)
    coeff = np.linalg.solve(vecs, dfhat)
    p1 = C.T @ (vecs @ (coeff * vals ** t1))
    p2 = C.T @ (vecs @ (coeff * vals ** t2))
    return -math.log(abs(p1[comp]) / abs(p2[comp])) / (t2 - t1)


def lbm_step(f, nu):
    rho = f.sum(0)
    u = np.einsum("qa,qzyx->azyx", C, f) / rho
    u2 = (u * u).sum(0)
    cu = np.einsum("qa,azyx->qzyx", C, u)
    feq = W[:, None, None, None] * rho * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * u2)
    f = f - omega(nu) * (f - feq)
    g = np.empty_like(f)
    for q, c in enumerate(D3Q19_C):
        g[q] = np.roll(f[q], shift=(c[2], c[1], c[0]), axis=(0, 1, 2))
    return g


def feq_field(u):
    cu = np.einsum("qa,azyx->qzyx", C, u)
    u2 = (u * u).sum(0)
    return W[:, None, None, None] * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * u2)


def kinetic_energy(f):
    rho = f.sum(0)
    u = np.einsum("qa,qzyx->azyx", C, f) / rho
    return float((0.5 * rho * (u * u).sum(0)).mean())


def tg_run(n, nu, t1, t2):
    x = np.arange(n) * (2 * math.pi / n)
    X, Y = np.meshgrid(x, x, indexing="xy")
    Z = np.zeros((n, n, n))
    ux = 0.05 * np.sin(X) * np.cos(Y) * np.ones((n, 1, 1))
    uy = -0.05 * np.cos(X) * np.sin(Y) * np.ones((n, 1, 1))
    u = np.stack([np.broadcast_to(ux, Z.shape).copy(), np.broadcast_to(uy, Z.shape).copy(), Z])
    f = feq_field(u)
    for _ in range(t1):
        f = lbm_step(f, nu)
    k1 = kinetic_energy(f)
    for _ in range(t2 - t1):
        f = lbm_step(f, nu)
    k2 = kinetic_energy(f)
    return -math.log(k1 / k2) / (t2 - t1)


def shear_run(n, nu, a, t1, t2):
    x = np.arange(n) * (2 * math.pi / n)
    u = np.zeros((3, n, n, n))
    u[1] = a * np.sin(x)[None, None, :] * np.ones((n, n, 1))
    f = feq_field(u)
    k = 2 * math.pi / n
    prof = np.sin(k * np.arange(n))

    def amp(ff):
        rho = ff.sum(0)
        uu = np.einsum("qa,qzyx->azyx", C, ff) / rho
        return float((uu[1].mean(0) * prof).sum() / (prof * prof).sum() / n)

    for _ in range(t1):
        f = lbm_step(f, nu)
    a1 = amp(f)
    for _ in range(t2 - t1):
        f = lbm_step(f, nu)
    a2 = amp(f)
    return -math.log(a1 / a2) / (t2 - t1)


def main():
    rows = []
    for n in (16, 24, 32):
        k = 2 * math.pi / n
        khat2 = 2 * (1 - math.cos(k))
        for nu in (0.02, 0.05, 0.10):
            cands = transverse_candidates((k, 0.0, 0.0), nu)
            rates = sorted(c["rate"] for c in cands)
            median = rates[len(rates) // 2]
            a = 1e-4
            dfhat = W * 3 * (C[:, 1] * a / 2j)  # sin(kx) -> +k Fourier mode, amplitude a/2i
            pred = mode_prediction((k, 0.0, 0.0), nu, dfhat, 1, 20, 220)
            meas = shear_run(n, nu, a, 20, 220)
            rows.append({
                "kind": "shear", "n": n, "nu": nu, "khat2": khat2,
                "n_transverse_modes": len(cands), "rates": rates,
                "median_rate": median, "nu_khat2": nu * khat2,
                "modal_prediction_20_220": pred, "measured_20_220": meas,
                "meas_vs_median_pct": 100 * (meas - median) / median,
                "meas_vs_modal_pct": 100 * (meas - pred) / pred,
                "median_vs_nukhat2_pct": 100 * (median - nu * khat2) / (nu * khat2),
            })
        for nu in (0.02, 0.05):
            g_meas = tg_run(n, nu, 40, 200)
            g_cont = 4 * nu * k * k
            a = 0.05
            pred = mode_prediction((k, k, 0.0), nu, W * 3 * (C[:, 0] * a / 2.0), 0, 40, 200)
            rows.append({
                "kind": "tg", "n": n, "nu": nu, "gamma_measured": g_meas, "gamma_4nuk2": g_cont,
                "tg_mode_rate": pred, "tg_mode_ke_rate": 2 * pred,
                "meas_vs_cont_pct": 100 * (g_meas - g_cont) / g_cont,
                "meas_vs_disc_ke_pct": 100 * (g_meas - 2 * pred) / (2 * pred),
            })
    print(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())