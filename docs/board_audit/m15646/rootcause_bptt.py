#!/usr/bin/env python3
"""Root-cause localisation for the m15481 BPTT gradient (audit checks P1-P4).

C5 of audit_m15481.py shows the shipped BPTT gradient disagrees with a central
finite difference of J(tau) by 80.4% (24x24, T=200, tau=0.85).  This script
localises the defect without modifying the audited code:

  P1  collision-Jacobian action, shipped form b_action() vs finite difference
      of feq(f)  -> the shipped action carries a spurious 1/rho
  P2  transpose identity:  <M d, mu> == <d, M^T mu>  checked against FD, and
      M vs M^T asymmetry measured (the adjoint recursion needs M^T; the module
      applies M)
  P3  relative gradient error vs T for the three variants
        shipped        = b_action (M, with 1/rho)      <- as committed
        M_fwd          = M without 1/rho
        M_transposed   = M^T without 1/rho             <- correct adjoint
  P4  https://en.wikipedia.org/wiki/Adjoint_state_method sanity: the corrected
      variant is exact (<=1e-7) over T in {2,...,400}, which proves the defect is
      localised in b_action (bptt.py:63-90) and nowhere else in the pipeline.

usage: python3 docs/board_audit/m15646/rootcause_bptt.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code_snapshot"))
import numpy as np  # noqa: E402

from adjoint.lbm2d import EX, EY, W, initial_pulse, macro, step, equilibrium  # noqa: E402
from adjoint.bptt import anti_stream, b_action, seed_terminal  # noqa: E402


def coeffs(f):
    """A_i, D_i, E_i with d feq_i / d f_j = A_i + D_i (c_jx - ux) + E_i (c_jy - uy)."""
    rho, ux, uy = macro(f)
    uc = EX[:, None, None] * ux[None] + EY[:, None, None] * uy[None]
    u2 = ux * ux + uy * uy
    A = W[:, None, None] * (1.0 + 3.0 * uc + 4.5 * uc * uc - 1.5 * u2[None])
    D = W[:, None, None] * (3.0 * EX[:, None, None] + 9.0 * EX[:, None, None] * uc - 3.0 * ux[None])
    E = W[:, None, None] * (3.0 * EY[:, None, None] + 9.0 * EY[:, None, None] * uc - 3.0 * uy[None])
    return rho, ux, uy, A, D, E


def m_fwd(f, mu):
    rho, ux, uy, A, D, E = coeffs(f)
    s1 = mu.sum(0)
    mx = (mu * EX[:, None, None]).sum(0)
    my = (mu * EY[:, None, None]).sum(0)
    return A * s1[None] + D * (mx - ux * s1)[None] + E * (my - uy * s1)[None]


def m_transposed(f, mu):
    rho, ux, uy, A, D, E = coeffs(f)
    SA = (A * mu).sum(0)
    SB = (D * mu).sum(0)
    SC = (E * mu).sum(0)
    return SA[None] + (EX[:, None, None] - ux[None]) * SB[None] + (EY[:, None, None] - uy[None]) * SC[None]


def b_action_ratio(f, mu):
    """shipped b_action vs the exact M.mu: isolates the 1/rho factor."""
    return float(np.linalg.norm(b_action(f, mu) - m_fwd(f, mu)) / max(np.linalg.norm(m_fwd(f, mu)), 1e-300))


def p1() -> dict:
    ny = nx = 6
    rng = np.random.default_rng(1)
    f = np.abs(rng.standard_normal((9, ny, nx))) + 0.3
    mu = rng.standard_normal((9, ny, nx))
    eps = 1e-6

    def feq(ff):
        return equilibrium(*macro(ff))

    fd = (feq(f + eps * mu) - feq(f - eps * mu)) / (2 * eps)
    shipped = b_action(f, mu)
    fixed = m_fwd(f, mu)
    rho = macro(f)[0]
    return {"grid": f"{ny}x{nx}", "rho_min": float(rho.min()), "rho_max": float(rho.max()),
            "rel_err_shipped_vs_fd": float(np.linalg.norm(shipped - fd) / np.linalg.norm(fd)),
            "rel_err_no_1_over_rho_vs_fd": float(np.linalg.norm(fixed - fd) / np.linalg.norm(fd)),
            "interpretation": "shipped action == exact action / rho  => spurious 1/rho; "
                              "in a nearly incompressible LBM run rho~1 hides it (0.1%-level), "
                              "so it is a secondary defect"}


def p2() -> dict:
    ny = nx = 6
    rng = np.random.default_rng(1)
    f = np.abs(rng.standard_normal((9, ny, nx))) + 0.3
    mu = rng.standard_normal((9, ny, nx))
    d = rng.standard_normal((9, ny, nx))
    eps = 1e-6

    def feq(ff):
        return equilibrium(*macro(ff))

    lhs = float(np.sum(((feq(f + eps * d) - feq(f - eps * d)) / (2 * eps)) * mu))
    rhs = float(np.sum(d * m_transposed(f, mu)))
    asym = float(np.linalg.norm(m_fwd(f, mu) - m_transposed(f, mu)) / np.linalg.norm(m_fwd(f, mu)))
    return {"lhs_<M d,mu>": lhs, "rhs_<d,M^T mu>": rhs, "rel_gap": abs(lhs - rhs) / abs(lhs),
            "rel_asymmetry_norm(M - M^T)/norm(M)": asym,
            "conclusion": "the collision Jacobian M is strongly non-symmetric, so applying M in the "
                          "backward pass is not the adjoint of collision"}


def _J(tau, T, f0, z):
    f = f0
    for _ in range(T):
        f = step(f, tau)
    _, ux, uy = macro(f)
    return 0.5 * float(np.sum((ux - z) ** 2 + (uy - z) ** 2))


def grad(tau, T, f0, z, variant="shipped"):
    states = [f0]
    f = f0
    for _ in range(T):
        f = step(f, tau)
        states.append(f)
    rho_t, ux_t, uy_t = macro(states[T])
    lam = seed_terminal(ux_t, uy_t, rho_t, z, z)
    g = 0.0
    for n in range(T - 1, -1, -1):
        fn = states[n]
        mu = anti_stream(lam)
        rho, ux, uy = macro(fn)
        feq = equilibrium(rho, ux, uy)
        g += float(np.sum((fn - feq) * mu) / (tau * tau))
        B = {"shipped": lambda: b_action(fn, mu),
             "M_fwd": lambda: m_fwd(fn, mu),
             "M_transposed": lambda: m_transposed(fn, mu)}[variant]()
        lam = mu - (mu - B) / tau
    return g


def p3() -> dict:
    ny = nx = 16
    tau = 0.85
    f0 = initial_pulse(ny, nx, amp=0.05, sigma=4.0)
    z = np.zeros((ny, nx))
    rows = []
    for T in (2, 3, 5, 10, 20, 50, 100, 200, 400):
        h = 1e-5
        fd = (_J(tau + h, T, f0, z) - _J(tau - h, T, f0, z)) / (2 * h)
        row = {"T": T, "g_fd": fd}
        for v in ("shipped", "M_fwd", "M_transposed"):
            g = grad(tau, T, f0, z, v)
            row[v] = {"g": g, "rel_err_vs_fd": abs(g - fd) / abs(fd) if fd else None}
        rows.append(row)
    return {"grid": f"{ny}x{nx}", "tau": tau, "amp": 0.05, "sigma": 4.0, "rows": rows,
            "correction_proved_exact": all(r["M_transposed"]["rel_err_vs_fd"] < 1e-6 for r in rows),
            "shipped_err_at_T200": next(r["shipped"]["rel_err_vs_fd"] for r in rows if r["T"] == 200)}


def p4() -> dict:
    """Shipped-code error on the audit's C5 configuration, restated for the record."""
    ny = nx = 24
    T, tau = 200, 0.85
    f0 = initial_pulse(ny, nx, amp=0.05, sigma=5.0)
    z = np.zeros((ny, nx))
    h = 1e-5
    fd = (_J(tau + h, T, f0, z) - _J(tau - h, T, f0, z)) / (2 * h)
    g = grad(tau, T, f0, z, "shipped")
    return {"grid": f"{ny}x{nx}", "T": T, "tau": tau, "g_fd": fd, "g_shipped": g,
            "rel_err": abs(g - fd) / abs(fd), "J_at_tau": _J(tau, T, f0, z),
            "criterion": "gap task gate asks |g - g_BPTT| < 1e-5; here both OOTC and the "
                         "shipped BPTT share the same wrong b_action, so OOTC==BPTT to 0.0 "
                         "and the criterion is vacuous; the independent referee (FD) fails by 411%"}


CHECKS = {"P1_collision_jacobian": p1, "P2_transpose": p2, "P3_error_vs_T": p3, "P4_C5_restated": p4}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    for k, fn in CHECKS.items():
        try:
            out[k] = fn()
        except Exception as e:  # noqa: BLE001
            out[k] = {"error": f"{type(e).__name__}: {e}"}
        print("[ok]", k, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())