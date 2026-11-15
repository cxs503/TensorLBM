"""Linear stability of the *model that is actually discretised* (verifier probe).

Why this file exists
--------------------
``benchmarks/case_pool/rayleigh_taylor_atwood05`` gates the measured LBM growth
rate ``sigma_sim`` against the **sharp-interface** closed form

    sigma_visc = sqrt(A g k + (nu k^2)^2) - nu k^2                     (S)

but the solver is *not* a sharp-interface solver: ``tensorlbm.rayleigh_taylor``
transports the interface with an Allen-Cahn phase field of finite thickness

    d_t phi + div(phi u) = -M mu,   mu = beta phi (phi^2-1) - kappa lap(phi),

whose equilibrium profile is ``phi0(y) = tanh(y/xi)``, ``xi = sqrt(2 kappa/beta)``
(= 2.0 lattice cells in the accepted case).  Before accepting the conclusion
"sigma_sim is ~10 % below (S), therefore the solver is 10 % wrong" a verifier has
to answer:

    does the *continuum model the code solves* have growth rate (S), or does the
    diffuse Allen-Cahn interface already damp the linear mode by ~10 %?

Method (independent of the case scripts, derived from the update in
``src/tensorlbm/rayleigh_taylor.py``).  With the standing-mode ansatz
``phi = phi0(y) + ph(y) cos(kx) e^{st}``, ``u_y = w(y) cos(kx) e^{st}``,
``u_x = -(w'/k) sin(kx) e^{st}`` and the vorticity ``om = (L w / k) sin(kx)``,
``L = d_y^2 - k^2``, the linearised system is

    d_t om = nu L om + A g k ph                     (vorticity, pressure removed)
    d_t ph = -M [ beta (3 phi0^2 - 1) - kappa L ] ph - phi0' w
    L w    = k om                                   (streamfunction inversion)

which is integrated in time (banded solves) until the dominant normal mode
dominates; its growth rate is the returned ``sigma_model``.

Sharp-interface limit check (``phi0' -> 2 delta(y)``, ``M = 0``): the first two
equations give ``d_t (L w) = nu L^2 w - A g k (phi0' w)``; with ``w = e^{-k|y|}``
the jump condition ``[w'] = -2k`` reproduces ``s^2 = A g k`` and, with viscosity,
``(S)`` exactly -- asserted numerically by tests T1/T2 below.

Only ``M beta`` and ``xi`` enter the Allen-Cahn block
(``M[beta(3phi0^2-1) - kappa L] = M beta [(3phi0^2-1) - (xi^2/2) L]``).

Usage
-----
    python3 verify/linear_stability_model.py --selftest
    python3 verify/linear_stability_model.py --sweep      # writes verify/artifacts/*.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"


# ---------------------------------------------------------------------------
# closed forms, re-derived here on purpose: a verifier must not reuse the
# reference lock it is auditing
# ---------------------------------------------------------------------------
def sigma_inviscid(A: float, g: float, lam: float) -> float:
    k = 2.0 * math.pi / lam
    return math.sqrt(A * g * k)


def sigma_viscous(A: float, g: float, lam: float, nu: float) -> float:
    k = 2.0 * math.pi / lam
    x = nu * k * k
    return math.sqrt(A * g * k + x * x) - x


# ---------------------------------------------------------------------------
# operator assembly
# ---------------------------------------------------------------------------
def _operators(n: int, h: float, k: float):
    """``L = D2 - k^2`` with Dirichlet ends, plus a fast banded inverse.

    ``L`` is symmetric negative definite, so ``-L`` is symmetric positive definite
    and one banded Cholesky solve (``solveh_banded``, tridiagonal storage) replaces
    a sparse LU per step -- the difference between ~15 s and ~1.5 s per case.
    """
    from scipy.linalg import solveh_banded

    main = -2.0 / h**2 * np.ones(n)
    off = 1.0 / h**2 * np.ones(n - 1)
    D2 = sp.diags([off, main, off], offsets=[-1, 0, 1], format="lil")
    D2[0, :] = 0.0
    D2[0, 0] = 1.0
    D2[n - 1, :] = 0.0
    D2[n - 1, n - 1] = 1.0
    L = (D2.tocsr() - k * k * sp.identity(n, format="csr")).tocsr()
    # Dirichlet nodes are eliminated: the interior block of -L is symmetric
    # positive definite, so one banded Cholesky solve per step suffices.
    from scipy.linalg import solveh_banded
    m = n - 2
    diag = 2.0 / h**2 + k * k
    # scipy lower form: ab[0] = diagonal, ab[1] = sub-diagonal (ab[1,0] unused)
    sub = np.concatenate([[0.0], np.full(m - 1, -1.0 / h**2)])
    ab = np.vstack([np.full(m, diag), sub])

    class _Solve:
        def solve(self, b):
            bi = np.asarray(b, dtype=float)[1:-1]
            out = np.zeros_like(np.asarray(b, dtype=float))
            # ab stores -L; L w = b  =>  w = -(-L)^{-1} b
            out[1:-1] = -solveh_banded(ab, bi, lower=True)
            return out

    return L, _Solve()


def sigma_model(
    *,
    lam: float = 64.0,
    g: float = 2.0e-4,
    A: float = 0.5,
    nu: float = 0.005,
    xi: float = 2.0,
    m_beta: float = 4.5e-3,
    H: float = 128.0,
    h: float = 0.125,
    t_end: float | None = None,
    dt: float = 0.5,
    decay: float = 0.0,
) -> dict:
    """Dominant linear growth rate of the Boussinesq + Allen-Cahn model.

    ``m_beta = M * beta`` and ``xi = sqrt(2 kappa / beta)`` are the two
    Allen-Cahn groups that enter the linearised problem (``m_beta = 0`` is the
    sharp-interface model, ``M = 0``); ``decay > 0`` multiplies ``phi0'`` by
    ``exp(-decay |y|)`` and is only used by the diagnostics, not by the physics.
    """
    k = 2.0 * math.pi / lam
    n = int(round(H / h)) + 1
    y = np.linspace(-0.5 * H, 0.5 * H, n)
    L, lu_L = _operators(n, h, k)

    phi0 = np.tanh(y / xi)
    phi0p = (1.0 - phi0 * phi0) / xi          # sech^2(y/xi)/xi, -> 2 delta(y)
    if decay > 0.0:
        phi0p = phi0p * np.exp(-decay * np.abs(y))
    dw = 3.0 * phi0 * phi0 - 1.0              # double-well curvature

    s_inv = sigma_inviscid(A, g, lam)
    if t_end is None:
        t_end = 6.0 / max(s_inv, 1e-12)       # ~6 inviscid e-folding times

    # seed: interface displaced by one unit (ph = -phi0' * eta) and no vorticity
    om = np.zeros(n)
    ph = -phi0p.copy()
    lam_ratio = float(np.sqrt(np.sum(ph * ph)) or 1.0)
    ph = ph / lam_ratio
    scale = math.sqrt(A * g * k) * 0.05

    def rhs(om, ph):
        w = lu_L.solve(k * om)                # L w = k om
        do = nu * (L @ om) + A * g * k * ph
        dphi = -m_beta * (dw * ph - 0.5 * xi * xi * (L @ ph)) - phi0p * w
        return do, dphi

    t = 0.0
    ts, logs = [], []
    nstep = int(t_end / dt)
    sample = max(1, nstep // 400)
    for step in range(nstep):
        k1o, k1p = rhs(om, ph)
        om_h, ph_h = om + 0.5 * dt * k1o, ph + 0.5 * dt * k1p
        k2o, k2p = rhs(om_h, ph_h)
        om = om + dt * k2o
        ph = ph + dt * k2p
        t += dt
        if not (np.isfinite(om).all() and np.isfinite(ph).all()):
            return {"sigma_model": float("nan"), "status": "blow_up", "t": t}
        if step % sample == 0:
            ts.append(t)
            logs.append(float(np.log(max(np.abs(ph).max(), 1e-300))))
    ts = np.asarray(ts)
    logs = np.asarray(logs)
    # fit the late, single-mode part: from t = 0.35 t_end, over >= 10 records
    i0 = int(0.35 * len(ts))
    if len(ts) - i0 < 10:
        i0 = max(0, len(ts) - 10)
    p = np.polyfit(ts[i0:], logs[i0:], 1)
    resid = logs[i0:] - np.polyval(p, ts[i0:])
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((logs[i0:] - logs[i0:].mean()) ** 2)) or 1.0
    return {
        "sigma_model": float(p[0]),
        "r2": 1.0 - ss_res / ss_tot,
        "status": "ok",
        "lam": lam, "g": g, "A": A, "nu": nu, "xi": xi, "m_beta": m_beta,
        "H": H, "h": h, "dt": dt, "n_grid": n, "t_end": t_end,
        "k_xi": k * xi,
        "amp_scale_at_fit": scale,
    }


# ---------------------------------------------------------------------------
# self tests: the machinery must reproduce the classical sharp-interface limits
# ---------------------------------------------------------------------------
def selftest() -> int:
    ok = True
    ref = sigma_inviscid(0.5, 2.0e-4, 64.0)
    print("== T1 inviscid sharp limit: sigma_model(xi->0, M=0) vs sqrt(A g k) ==")
    print(f"   analytic sqrt(A g k) = {ref:.6e}")
    for xi in (1.0, 0.5, 0.25, 0.125, 0.0625):
        r = sigma_model(xi=xi, m_beta=0.0, nu=0.0, h=0.0625, H=64.0)
        e = 100.0 * (r["sigma_model"] / ref - 1.0)
        print(f"   xi={xi:6.4f}  sigma={r['sigma_model']:.6e}  err={e:+7.3f}%  r2={r['r2']:.5f}")
    r = sigma_model(xi=0.0625, m_beta=0.0, nu=0.0, h=0.03125, H=64.0)
    e1 = 100.0 * (r["sigma_model"] / ref - 1.0)
    print(f"   finest: sigma={r['sigma_model']:.6e}  err={e1:+7.3f}%   (T1 needs |e| < 1 %)")
    ok &= abs(e1) < 1.0

    ref2 = sigma_viscous(0.5, 2.0e-4, 64.0, 0.005)
    print("== T2 viscous sharp limit: vs sqrt(Agk+(nu k^2)^2) - nu k^2 ==")
    print(f"   analytic = {ref2:.6e}")
    for nu in (0.005, 0.02):
        r2 = sigma_viscous(0.5, 2.0e-4, 64.0, nu)
        r = sigma_model(xi=0.0625, m_beta=0.0, nu=nu, h=0.03125, H=64.0, dt=0.02)
        e2 = 100.0 * (r["sigma_model"] / r2 - 1.0)
        print(f"   nu={nu:6.4f}  sigma={r['sigma_model']:.6e}  ref={r2:.6e}  err={e2:+7.3f}%")
    ok &= abs(e2) < 1.0

    print("== T3 grid/domain independence at the case parameters (xi=2, M beta=4.5e-3) ==")
    for H, h in ((64.0, 0.25), (128.0, 0.125), (128.0, 0.0625), (256.0, 0.125)):
        r = sigma_model(xi=2.0, m_beta=4.5e-3, H=H, h=h, dt=0.5 * min(0.25, h * h / (4 * 0.02)))
        print(f"   H={H:6.1f} h={h:6.4f} dt={r['dt']:.4f}  sigma={r['sigma_model']:.6e}  "
              f"r2={r['r2']:.5f}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# the audit sweep
# ---------------------------------------------------------------------------
#: parameters of the accepted reference case (run_case.SPECS[0] / ref_lock.json)
CASE = dict(lam=64.0, g=2.0e-4, A=0.5, nu=0.005, xi=2.0,
            m_beta=20.0 * 2.25e-4)     # ac_mobility * beta, beta = 3*ac_sigma/(2*W)


def sweep() -> dict:
    ART.mkdir(parents=True, exist_ok=True)
    s_inv = sigma_inviscid(CASE["A"], CASE["g"], CASE["lam"])
    s_vis = sigma_viscous(CASE["A"], CASE["g"], CASE["lam"], CASE["nu"])
    out: dict = {
        "case_parameters": dict(CASE),
        "sigma_reference": {"inviscid": s_inv, "viscous": s_vis},
        "method": "time marching of the linearised Boussinesq+Allen-Cahn system "
                  "(see module docstring); no LBM involved",
    }

    def run(**kw):
        kw.setdefault("H", 64.0)
        kw.setdefault("h", 0.0625)
        h = kw["h"]
        mb = kw.get("m_beta", CASE["m_beta"])
        xi = kw.get("xi", CASE["xi"])
        # explicit stability: vorticity 4 nu/h^2, Allen-Cahn 4 M kappa/h^2 with kappa = M beta xi^2/2
        lim_nu = h * h / (2.0 * max(CASE["nu"], 1e-12))
        lim_ac = h * h / (4.0 * max(mb * xi * xi / 2.0, 1e-12))
        kw.setdefault("dt", min(1.0, 0.8 * min(lim_nu, lim_ac)))
        return sigma_model(**{**CASE, **kw})

    # (a) sharp model (M = 0, xi -> 0): must land on sigma_visc
    out["sharp_model"] = run(xi=0.0625, m_beta=0.0, h=0.03125)

    # (b) the model the code actually integrates: xi = 2, M beta = 4.5e-3
    out["diffuse_model"] = run()

    # (c) decomposition of the model-form loss
    decomp = []
    for label, kw in (
        ("sharp (xi->0, M=0)", dict(xi=0.0625, m_beta=0.0, h=0.03125)),
        ("thickness only (xi=2, M=0)", dict(xi=2.0, m_beta=0.0)),
        ("mobility only (xi->0, M b=4.5e-3)", dict(xi=0.0625, m_beta=4.5e-3, h=0.03125)),
        ("both = the code's model", dict(xi=2.0, m_beta=4.5e-3)),
    ):
        r = run(**kw)
        print(f"  [decomp] {label:38s} sigma={r['sigma_model']:.4e}", flush=True)
        decomp.append({"label": label, "sigma": r["sigma_model"], "r2": r.get("r2"),
                       "err_vs_visc_pct": 100.0 * (r["sigma_model"] / s_vis - 1.0),
                       "err_vs_inv_pct": 100.0 * (r["sigma_model"] / s_inv - 1.0)})
    out["model_form_decomposition"] = decomp

    # (d) Allen-Cahn mobility ladder in the model (the LBM ladder mirrors this)
    ladder = []
    for M in (20.0, 10.0, 2.0, 0.0):
        r = run(m_beta=M * 2.25e-4)
        print(f"  [mob] M={M:5.1f} sigma={r['sigma_model']:.4e} r2={r.get('r2'):.4f}", flush=True)
        ladder.append({"ac_mobility": M, "m_beta": M * 2.25e-4,
                       "sigma": r["sigma_model"], "r2": r.get("r2"),
                       "err_vs_visc_pct": 100.0 * (r["sigma_model"] / s_vis - 1.0)})
    out["mobility_ladder_model"] = ladder

    # (e) interface-thickness ladder
    thick = []
    for xi in (2.0, 1.0):
        r = run(xi=xi)
        print(f"  [xi]  xi={xi:4.2f} sigma={r['sigma_model']:.4e}", flush=True)
        thick.append({"xi": xi, "k_xi": r["k_xi"], "sigma": r["sigma_model"],
                      "err_vs_visc_pct": 100.0 * (r["sigma_model"] / s_vis - 1.0)})
    out["thickness_ladder_model"] = thick

    p = ART / "linear_stability_model.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2)[:5000])
    print(f"\n[written] {p}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.sweep:
        sweep()
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())