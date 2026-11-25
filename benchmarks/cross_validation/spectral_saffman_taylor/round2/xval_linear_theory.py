#!/usr/bin/env python3
"""XVAL round 2: independent first-principles linear-stability anchor for
TensorLBM ``SpectralSaffmanTaylor2D`` (spectral Saffman--Taylor, integrating factor).

Why a round 2 exists
--------------------
Round 1 (``xval_spectral_st.py``, docs/XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-
INTEGRATING-FACTOR.md) validated the port *against the OSS code it was ported
from* (round-off agreement) and against a constant ``C`` that was itself the
average of the same measurements -- that is an internal-consistency check, not a
validation against independent theory.  This script supplies the missing anchor.

Anchor (derived here from the governing PDEs, no OSS code, no TensorLBM code)
---------------------------------------------------------------------------
Hele-Shaw / porous medium, miscible, ``mu = exp(R c)``, ``D = 1``::

    dc/dt + u.c = lap(c),    u = (psi_y, -psi_x)
    lap(psi) + R (c_x psi_x + c_y psi_y) = 0        (curl of Darcy, exact)

Base state: ``cbar(xi) = 0.5 (1 - tanh(xi/w))``, ``xi = x - t`` (unit base flow
along +x, i.e. ``psi0 = y``; this base satisfies the elliptic equation exactly
for every R), perturbation ~ ``exp(i k y + sigma t)``::

    psi'' - k^2 psi + R cbar' psi' = -i k R c          (exact linearisation)
    sigma c = c'' - k^2 c - i k cbar' psi

Eliminating ``psi`` (``A = D2 + R diag(cbar') D - k^2 I``)::

    sigma c = (D2 - k^2 I) c - k^2 R diag(cbar') A^{-1} c

whose spectrum is the independent growth-rate anchor.  Two independent checks of
the anchor itself are performed below:

  V1 ``R = 0``            -> ``sigma = -k^2`` exactly (pure transverse diffusion);
  V2 ``w -> 0`` (step)    -> must converge to the *hand-derived* sharp-interface
     dispersion relation of the same linearised system, obtained analytically
     from the delta-function jump conditions:
         cbar' = -delta(xi),  kappa^2 = k^2 + sigma,
         2 kappa^2 + 2 k kappa + k R = 0,  sigma = kappa^2 - k^2
     (admissible root: Re kappa > 0).
     Sign check against the classical Saffman--Taylor criterion: the front is
     unstable only when the fluid *ahead* (x > x_front) is more viscous, i.e.
     when R < 0 for ``mu = exp(R c)`` with ``c = 1`` upstream -- the analytic
     relation reproduces exactly that criterion.

Run: ``python benchmarks/cross_validation/spectral_saffman_taylor/round2/xval_linear_theory.py``
Outputs: ``artifacts/linear_theory.json`` (all numbers injected into the report)
and ``artifacts/fig_linear_theory_*.png``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorlbm.spectral_saffman_taylor import SpectralSaffmanTaylor2D  # noqa: E402

ART = HERE / "artifacts"
ART.mkdir(exist_ok=True)

# --- parameters of the round-1 growth-rate study (identical, for comparability)
NY, NX, ASPECT, PE, MODE = 128, 256, 2.0, 500.0, 4
K_MODE = 2.0 * np.pi * MODE / PE  # transverse wavenumber of the fitted mode
W_FRONT = 4.0  # tanh width of the round-1 IC
DT = 0.1
T_FIT = (2.0, 8.0)
LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


# --------------------------------------------------------------- anchor solver
def exact_linear_sigma(k: float, R: float, w: float, N: int = 1201, L: float = 800.0) -> dict:
    """Growth rate of the exact variable-coefficient curl-Darcy linearisation.

    Second-order finite differences in xi on [-L, L] with homogeneous Dirichlet
    data (the perturbation is localised on the scale 1/k = 20 << L).
    Returns the leading eigenvalue and its eigenfunction sampled on the grid.
    """
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
    M = D2i - k * k * Ii - (k * k * R) * (np.diag(dci) @ np.linalg.solve(A, Ii))
    ev, vec = np.linalg.eig(M)
    i = int(np.argmax(ev.real))
    mode = np.zeros(N, dtype=complex)
    mode[1:-1] = vec[:, i]
    return {"sigma": complex(ev[i]), "xi": xi, "mode": mode, "h": h, "n_eig": int(ev.size)}


def step_analytic_sigma(R: float, k: float) -> dict:
    """Hand-derived sharp-step dispersion relation of the same linear system.

    ``2 kappa^2 + 2 k kappa + k R = 0``, ``sigma = kappa^2 - k^2``, Re kappa > 0.
    """
    roots = np.roots([2.0, 2.0 * k, k * R])
    adm = [complex(r) for r in roots if r.real > 0]
    if not adm:
        return {"kappa": None, "sigma": None, "admissible": False}
    kap = adm[0]
    return {"kappa": kap, "sigma": kap**2 - k * k, "admissible": True}


# ------------------------------------------------- module growth-rate measurement
def module_growth_rate(R: float, dt: float = DT, ny: int = NY, nx: int = NX) -> dict:
    """Fit sigma from the amplitude of the fitted transverse mode (round-1 method)."""
    sim = SpectralSaffmanTaylor2D(
        ny, nx, aspect=ASPECT, pe=PE, viscosity_ratio=R, dt=dt, mode_index=MODE
    )
    sim.set_concentration(
        sim.single_mode_ic(x0=0.5 * ASPECT * PE, width=W_FRONT, amplitude=1e-3, n_mode=MODE)
    )
    ts, amps = [], []
    for _ in range(int(round(T_FIT[1] / dt)) + 1):
        d = sim.diagnostics()
        ts.append(d.t)
        amps.append(d.mode_amplitude)
        sim.step()
    ts, amps = np.array(ts), np.array(amps)
    mask = (ts >= T_FIT[0]) & (ts <= T_FIT[1]) & (amps > 0)
    slope = float(np.polyfit(ts[mask], np.log(amps[mask]), 1)[0])
    return {
        "R": R,
        "dt": dt,
        "ny": ny,
        "nx": nx,
        "sigma": slope,
        "n_fit": int(mask.sum()),
        "amp_end": float(amps[-1]),
        "ts": ts.tolist(),
        "amps": amps.tolist(),
    }


# ------------------------------------------------------------------- the study
def main() -> dict:
    t0 = time.time()
    k = K_MODE
    log(f"== XVAL round 2: independent linear-stability anchor ==")
    log(f"   k (mode {MODE}) = {k:.6e}, k^2 = {k * k:.6e}, front width w = {W_FRONT}")

    # ---- V1: R = 0 must reproduce -k^2 (pure transverse diffusion)
    v1 = exact_linear_sigma(k, 0.0, W_FRONT)
    v1_rel = abs(v1["sigma"].real + k * k) / (k * k)
    log(
        f"V1 R=0: sigma_theory = {v1['sigma'].real:+.6e} vs -k^2 = {-k * k:+.6e} "
        f"-> rel.dev {v1_rel:.2e} {'PASS' if v1_rel < 1e-3 else 'FAIL'}"
    )

    # ---- V2: sharp-step limit vs the hand-derived analytic relation
    v2 = {"rows": []}
    for R in (-2.0, -1.0, -0.5):
        step = step_analytic_sigma(R, k)
        for w in (8.0, 4.0, 2.0, 1.0):
            s = exact_linear_sigma(k, R, w)
            rel = abs(s["sigma"].real - step["sigma"].real) / abs(step["sigma"].real)
            v2["rows"].append(
                {
                    "R": R,
                    "w": w,
                    "k_w": k * w,
                    "sigma_num": s["sigma"].real,
                    "sigma_step": step["sigma"].real,
                    "rel_dev": rel,
                }
            )
            log(
                f"V2 R={R:<5} w={w:<4} k*w={k * w:.3f} sigma_num={s['sigma'].real:+.6e} "
                f"sigma_step={step['sigma'].real:+.6e} rel.dev={100 * rel:.2f}%"
            )
        v2[f"step_R{R}"] = {"kappa": str(step["kappa"]), "sigma": step["sigma"]}
    # convergence must be monotone towards the step value as w -> 0
    v2["monotone"] = all(
        abs(r2["rel_dev"]) <= abs(r1["rel_dev"]) + 1e-12
        for r1, r2 in zip(
            [r for r in v2["rows"] if r["R"] == -1.0], [r for r in v2["rows"] if r["R"] == -1.0][1:]
        )
    )
    log(f"   w->0 convergence monotone: {v2['monotone']}")

    # ---- grid / domain sensitivity of the anchor
    sens = []
    for N, L in ((401, 800.0), (801, 800.0), (801, 400.0), (1201, 800.0)):
        s = exact_linear_sigma(k, -1.0, W_FRONT, N=N, L=L)
        sens.append({"N": N, "L": L, "h": s["h"], "sigma": s["sigma"].real})
        log(f"   sensitivity N={N:<5} L={L:<6} h={s['h']:.3f} sigma={s['sigma'].real:+.6e}")

    # ---- module measurement (recomputed here, same protocol as round 1)
    mod = [module_growth_rate(R) for R in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)]
    for g in mod:
        log(f"M  module R={g['R']:<5} sigma={g['sigma']:+.6e} (n_fit={g['n_fit']})")

    # ---- theory vs module table
    rows = []
    for g in mod:
        R = g["R"]
        th = exact_linear_sigma(k, R, W_FRONT)
        st = step_analytic_sigma(R, k)
        st_sig = st["sigma"].real if st["admissible"] else float("nan")
        ref = th["sigma"].real
        scale = abs(ref) if abs(ref) > 1e-6 else 1.0
        rows.append(
            {
                "R": R,
                "sigma_module": g["sigma"],
                "sigma_theory": ref,
                "sigma_step_analytic": st_sig,
                "k2_diffusive": k * k,
                "abs_err_module_vs_theory": abs(g["sigma"] - ref),
                "rel_err_module_vs_theory": abs(g["sigma"] - ref) / abs(ref)
                if abs(ref) > 1e-6
                else float("nan"),
                "sign_agree": bool(np.sign(g["sigma"]) == np.sign(ref)),
                "pass_15pct": bool(abs(g["sigma"] - ref) / scale < 0.15),
            }
        )
        log(
            f"X  R={R:<5} sigma_module={g['sigma']:+.6e} sigma_theory={ref:+.6e} "
            f"sigma_step={st_sig:+.6e} sign_agree={rows[-1]['sign_agree']} "
            f"rel.dev={100 * rows[-1]['rel_err_module_vs_theory']:.1f}%"
        )

    # mirrored comparison: module(R) against theory(-R) (the OSS symbol looks
    # like the exact coupling with the sign of R reversed)
    mirror = []
    for g in mod:
        R = g["R"]
        if R == 0.0:
            continue
        th = exact_linear_sigma(k, -R, W_FRONT)["sigma"].real
        mirror.append(
            {
                "R": R,
                "sigma_module": g["sigma"],
                "sigma_theory_minusR": th,
                "rel_dev": abs(g["sigma"] - th) / abs(th),
                "pass_15pct": bool(abs(g["sigma"] - th) / abs(th) < 0.15),
            }
        )
        log(
            f"Y  R={R:<5} sigma_module={g['sigma']:+.6e} vs sigma_theory(-R)={th:+.6e} "
            f"rel.dev={100 * mirror[-1]['rel_dev']:.1f}%"
        )

    # ---- figures
    fig_paths = make_figures(k, rows, mirror, v2, mod)

    res = {
        "k": k,
        "k2": k * k,
        "mode": MODE,
        "pe": PE,
        "w_front": W_FRONT,
        "dt": DT,
        "ny": NY,
        "nx": NX,
        "t_fit": list(T_FIT),
        "V1_sigma_R0": v1["sigma"].real,
        "V1_rel_dev_vs_minus_k2": v1_rel,
        "V2_sharp_step": v2,
        "anchor_sensitivity": sens,
        "module_runs": [{kk: vv for kk, vv in g.items() if kk not in ("ts", "amps")} for g in mod],
        "comparison": rows,
        "mirror_comparison": mirror,
        "figures": fig_paths,
        "wall_s": time.time() - t0,
        "log": LOG,
    }
    (ART / "linear_theory.json").write_text(json.dumps(res, indent=1))
    log(f"artifacts written: {ART / 'linear_theory.json'} ({time.time() - t0:.1f}s)")
    return res


def make_figures(k, rows, mirror, v2, mod):
    paths = {}
    # fig 1: sigma(R) -- module vs exact theory vs sharp-step analytic
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    Rs = [r["R"] for r in rows]
    ax[0].plot(Rs, [r["sigma_module"] for r in rows], "o-", label="TensorLBM port (frozen closure)")
    ax[0].plot(Rs, [r["sigma_theory"] for r in rows], "s-", label="exact curl-Darcy linear theory")
    ax[0].plot(Rs, [r["sigma_step_analytic"] for r in rows], "^--", label="sharp-step analytic")
    ax[0].axhline(-k * k, color="0.6", lw=0.8, ls=":", label=r"$-k^2$ (diffusive decay)")
    ax[0].axhline(0.0, color="k", lw=0.6)
    ax[0].axvline(0.0, color="k", lw=0.6)
    ax[0].set_xlabel("log-viscosity ratio R  ($\\mu=e^{Rc}$, c=1 upstream)")
    ax[0].set_ylabel(r"growth rate $\sigma$")
    ax[0].set_title("growth rate: module vs first-principles theory")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    # right: mirrored comparison |sigma_module(R)| vs theory(-R)
    Rm = [m["R"] for m in mirror]
    ax[1].plot(Rm, [m["sigma_module"] for m in mirror], "o-", label=r"module $\sigma(R)$")
    ax[1].plot(Rm, [m["sigma_theory_minusR"] for m in mirror], "s-", label=r"theory $\sigma(-R)$")
    ax[1].set_xlabel("R")
    ax[1].set_ylabel(r"growth rate $\sigma$")
    ax[1].set_title("mirror test: module(R) vs theory(-R)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = ART / "fig_linear_theory_sigma_vs_R.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["sigma_vs_R"] = p.name

    # fig 2: eigenmode profiles at R = -1 and R = +1
    fig, ax = plt.subplots(figsize=(7, 4))
    for R, ls in ((-1.0, "-"), (1.0, "--")):
        s = exact_linear_sigma(k, R, W_FRONT)
        xi, mode = s["xi"], s["mode"]
        m = np.abs(mode) / np.abs(mode).max()
        sel = np.abs(xi) <= 120
        ax.plot(xi[sel], m[sel], ls, label=f"R={R:+.0f}, $\\sigma$={s['sigma'].real:+.3e}")
    ax.set_xlabel(r"$\xi = x - t$")
    ax.set_ylabel("normalised leading eigenmode $|\\tilde c|$")
    ax.set_title("leading linear eigenmode of the exact curl-Darcy operator")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = ART / "fig_linear_theory_eigenmode.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["eigenmode"] = p.name

    # fig 3: sharp-step limit convergence
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for R, mk in ((-2.0, "o"), (-1.0, "s"), (-0.5, "^")):
        rr = [r for r in v2["rows"] if r["R"] == R]
        ax.semilogx(
            [r["w"] for r in rr],
            [r["sigma_num"] for r in rr],
            mk + "-",
            label=f"exact theory, R={R}",
        )
        ax.semilogx([r["w"] for r in rr], [r["sigma_step"] for r in rr], "k--", lw=0.8)
    ax.set_xlabel("front width w  (step limit: w $\\to$ 0)")
    ax.set_ylabel(r"$\sigma$")
    ax.set_title("anchor validation: numeric finite-width vs hand-derived step relation")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = ART / "fig_linear_theory_step_limit.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["step_limit"] = p.name
    return paths


if __name__ == "__main__":
    main()
