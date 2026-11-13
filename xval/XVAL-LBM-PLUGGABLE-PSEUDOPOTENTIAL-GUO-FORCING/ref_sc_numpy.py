"""Independent numpy reference for the SCMP pseudopotential model with a
*pluggable* forcing axis, plus the two analytic anchors used by the cross
validation.

Scope
-----
This module is **Arm A** of the cross validation ``XVAL-LBM-PLUGGABLE-
PSEUDOPOTENTIAL-GUO-FORCING``.  It is a clean numpy transcription of the
published discrete equations:

* D2Q9 equilibrium, ``f_i^eq = w_i rho [1 + c_i.u/cs^2 + (c_i.u)^2/(2 cs^4)
  - u^2/(2 cs^2)]``  -- Qian, d'Humieres & Lallemand (1992).
* Shan-Chen pseudopotential interaction force, lattice-sum form
  ``F(x) = -G psi(x) sum_i w_i c_i psi(x + c_i)``  -- Shan & Chen (1993)
  Phys. Rev. E 47, 1815, eq. (5); with ``G < 0`` attractive in the physical
  convention used here (``G_eff = -G_lib`` of the library call).
* Exponential pseudopotential ``psi(rho) = rho0 [1 - exp(-rho/rho0)]``
  -- Shan & Chen (1993); Sukop & Thorne (2006) Sec. 9.
* Guo, Zheng & Shi (2002) Phys. Rev. E 65, 046308, eq. (20) forcing:
  ``S_i = (1 - 1/(2 tau)) w_i [(c_i - u).F / cs^2 + (c_i.u)(c_i.F)/cs^4]``
  with the half-force corrected velocity ``u = v + F/(2 rho)`` in ``f^eq``.
* Exact difference method (EDM), Kupershtokh (2004); Kupershtokh, Medvedev &
  Karpov (2009) Comput. Math. Appl. 58, 965:
  ``f_out = f - (f - f^eq(v))/tau + [f^eq(v + F/rho) - f^eq(v)]``.
* Velocity-shift forcing (the historical Shan-Chen / Qian form, as in
  Shan & Chen (1993) and Sukop & Thorne (2006) Sec. 5):
  ``f_out = f - (f - f^eq(v + tau F/rho))/tau``.
* Mechanical equation of state of the SC model,
  ``p(rho) = rho cs^2 + (G_eff/2) cs^2 psi^2``  (Shan & Chen 1993; Yuan &
  Schaefer (2006) Phys. Fluids 18, 042101), with Maxwell (equal-area)
  construction for the coexistence densities.

Independence disclosure
-----------------------
The *equations* above are the external anchors.  The repo's own SC engine was
read before writing this file (to pick the same lattice stencil/ordering), so
this module is a **clean re-derivation with disclosed contamination**, not a
blind reimplementation.  The two checks that are genuinely independent of any
code are the analytic ones: the Maxwell construction (:func:`maxwell_phase`)
and the per-step momentum-injection identity
``sum_i c_i (f_i^out - f_i) = F``, which every consistent forcing scheme must
satisfy exactly and which is asserted at machine precision.

Conventions
-----------
``(9, ny, nx)`` population layout, ``(ny, nx)`` fields, float64, periodic box.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, fsolve

# --------------------------------------------------------------------------
# lattice
# --------------------------------------------------------------------------

CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
CS2 = 1.0 / 3.0

SCHEMES = ("guo", "edm", "velocity_shift")


def psi_exp(rho: np.ndarray, rho0: float = 1.0) -> np.ndarray:
    """Shan-Chen exponential pseudopotential (Shan & Chen 1993).

    ``rho`` is clipped to ``[-200, 200]`` before the exponential purely to keep
    root-finding trial points finite; the physical branch is ``rho > 0``.
    """
    with np.errstate(over="ignore"):
        return rho0 * (1.0 - np.exp(-np.clip(rho, -200.0, 200.0) / rho0))


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium truncated at second order in ``u``."""
    cx = CX[:, None, None]
    cy = CY[:, None, None]
    cu = cx * ux + cy * uy
    usq = ux * ux + uy * uy
    return W[:, None, None] * rho * (1.0 + cu / CS2 + 0.5 * (cu / CS2) ** 2 - 0.5 * usq / CS2)


def macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = f.sum(axis=0)
    rho_s = np.maximum(rho, 1e-12)
    return rho, (f * CX[:, None, None]).sum(axis=0) / rho_s, (f * CY[:, None, None]).sum(axis=0) / rho_s


def stream(f: np.ndarray) -> np.ndarray:
    """Backward (gather) streaming, periodic: ``f_i(x) <- f_i^post(x - c_i)``."""
    out = np.empty_like(f)
    for i in range(9):
        out[i] = np.roll(f[i], (int(CY[i]), int(CX[i])), axis=(0, 1))
    return out


def sc_force(rho: np.ndarray, G_lib: float, psi_fn=psi_exp) -> tuple[np.ndarray, np.ndarray]:
    """Shan-Chen lattice-sum interaction force (library sign: ``G_lib > 0`` attractive).

    ``F = -G_lib psi(x) sum_i w_i c_i psi(x - c_i)``; because ``sum_i w_i c_i
    psi(x - c_i) = -sum_i w_i c_i psi(x + c_i)`` this equals the classic
    ``-G_eff psi grad psi`` with ``G_eff = -G_lib < 0`` (attractive).
    """
    psi = psi_fn(rho)
    sx = np.zeros_like(rho)
    sy = np.zeros_like(rho)
    for i in range(9):
        sh = np.roll(psi, (int(CY[i]), int(CX[i])), axis=(0, 1))
        sx += W[i] * CX[i] * sh
        sy += W[i] * CY[i] * sh
    return -G_lib * psi * sx, -G_lib * psi * sy


def collide(
    f: np.ndarray,
    G_lib: float = 5.0,
    tau: float = 1.0,
    psi_fn=psi_exp,
    scheme: str = "guo",
    fx_ext: np.ndarray | None = None,
    fy_ext: np.ndarray | None = None,
) -> np.ndarray:
    """BGK collision with one of the three pluggable forcing schemes."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")
    rho, ux, uy = macroscopic(f)
    fx, fy = sc_force(rho, G_lib, psi_fn)
    if fx_ext is not None:
        fx = fx + fx_ext
    if fy_ext is not None:
        fy = fy + fy_ext
    rho_s = np.maximum(rho, 1e-12)

    if scheme == "guo":
        uxp = ux + 0.5 * fx / rho_s
        uyp = uy + 0.5 * fy / rho_s
        feq = equilibrium(rho, uxp, uyp)
        cf = CX[:, None, None] * fx + CY[:, None, None] * fy
        cu = CX[:, None, None] * uxp + CY[:, None, None] * uyp
        uf = uxp * fx + uyp * fy
        src = W[:, None, None] * (1.0 - 0.5 / tau) * (
            (cf - uf[None]) / CS2 + cu * cf / CS2**2
        )
        return f - (f - feq) / tau + src
    if scheme == "edm":
        feq_u = equilibrium(rho, ux, uy)
        feq_d = equilibrium(rho, ux + fx / rho_s, uy + fy / rho_s)
        return f - (f - feq_u) / tau + (feq_d - feq_u)
    # velocity shift
    feq = equilibrium(rho, ux + tau * fx / rho_s, uy + tau * fy / rho_s)
    return f - (f - feq) / tau


def step(
    f: np.ndarray,
    G_lib: float = 5.0,
    tau: float = 1.0,
    scheme: str = "guo",
    psi_fn=psi_exp,
) -> np.ndarray:
    return stream(collide(f, G_lib, tau, psi_fn, scheme))


# --------------------------------------------------------------------------
# analytic anchors
# --------------------------------------------------------------------------

def eos_pressure(rho, G_lib: float = 5.0, psi_fn=psi_exp):
    """SC mechanical EOS ``p = rho cs^2 + (G_eff/2) cs^2 psi^2``, ``G_eff = -G_lib``."""
    psi = psi_fn(np.asarray(rho, dtype=float))
    return np.asarray(rho, dtype=float) * CS2 - (G_lib / 2.0) * CS2 * psi**2


def spinodal(G_lib: float = 5.0) -> tuple[float, float]:
    """Densities where ``dp/drho = 0`` (brackets of the unstable branch)."""
    drho = 1e-6

    def dp(r):
        return (eos_pressure(r + drho, G_lib) - eos_pressure(r - drho, G_lib)) / (2 * drho)

    grid = np.linspace(1e-3, 4.0, 40000)
    d = np.array([dp(r) for r in grid])
    idx = np.where(np.diff(np.sign(d)) != 0)[0]
    if len(idx) < 2:
        raise ValueError(f"no liquid/vapour spinodal at G_lib={G_lib}: single-phase EOS")
    return float(grid[idx[0]]), float(grid[idx[-1]])


def maxwell_phase(G_lib: float = 5.0, psi_fn=psi_exp) -> dict:
    """Maxwell (equal-area) coexistence densities of the SC equation of state.

    Solves ``p(rho_l) = p(rho_v) = p*`` together with the equal-area rule

        integral_{rho_v}^{rho_l} (p(rho) - p*) / rho^2 drho = 0,

    which is the thermodynamic (``mu_l = mu_v``) coexistence condition of the
    van der Waals / SC EOS.  This is a **pure one-dimensional root find** on the
    analytic EOS: it shares no code and no discretisation with the lattice
    solver, so it is a genuine external anchor for the simulated bulk densities.
    """
    lo, hi = spinodal(G_lib)
    if hi - lo < 1e-2:
        raise ValueError(
            f"G_lib={G_lib} is at/below the critical coupling (spinodal roots "
            f"{lo:.4f}/{hi:.4f} coincide): no two-phase coexistence"
        )

    def residual(x):
        rl, rv = x
        pstar = eos_pressure(rl, G_lib, psi_fn)
        eq_p = pstar - eos_pressure(rv, G_lib, psi_fn)
        area, _ = quad(lambda r: (eos_pressure(r, G_lib, psi_fn) - pstar) / r**2, rv, rl, limit=400)
        return [eq_p, area]

    # Parameterise the coexistence by the common pressure p*: for every
    # p* in (p(hi), p(lo)) both branches have exactly one root, so the
    # equal-area residual becomes a scalar function of p* alone.
    p_lo = eos_pressure(lo, G_lib, psi_fn)
    p_hi = eos_pressure(hi, G_lib, psi_fn)

    def branches(pstar):
        rv_ = brentq(lambda r: eos_pressure(r, G_lib, psi_fn) - pstar, 1e-12, lo * (1 - 1e-12), xtol=1e-15)
        rl_ = brentq(lambda r: eos_pressure(r, G_lib, psi_fn) - pstar, hi * (1 + 1e-12), 8.0, xtol=1e-15)
        return rv_, rl_

    def area_of(pstar):
        rv_, rl_ = branches(pstar)
        a, _ = quad(
            lambda r: (eos_pressure(r, G_lib, psi_fn) - pstar) / r**2,
            rv_,
            rl_,
            limit=400,
            epsabs=1e-14,
            epsrel=1e-12,
        )
        return a

    # p(rho -> 0) = 0, so the vapour branch has a root only for 0 < p* < p(lo).
    p_star = brentq(area_of, 1e-9 * p_lo, p_lo * (1 - 1e-9), xtol=1e-15)
    rv, rl = branches(p_star)
    sol = [rl, rv]
    if not (rv < lo < hi < rl):
        raise RuntimeError(f"Maxwell solve left the coexistence bracket: {sol}")
    return {
        "G_lib": G_lib,
        "rho_l": rl,
        "rho_v": rv,
        "p_star": float(p_star),
        "spinodal": (lo, hi),
        "residual": [float(area_of(p_star))],
    }


def density_ratio(G_lib: float = 5.0) -> float:
    m = maxwell_phase(G_lib)
    return m["rho_l"] / m["rho_v"]


# --------------------------------------------------------------------------
# geometry / diagnostics
# --------------------------------------------------------------------------

def droplet_rho(L: int, R: float, rho_l: float, rho_v: float, width: float = 2.0) -> np.ndarray:
    """Tanh liquid cylinder of radius ``R`` in the centre of an ``L x L`` box."""
    y = np.arange(L)[:, None] + 0.5
    x = np.arange(L)[None, :] + 0.5
    r = np.hypot(x - L / 2, y - L / 2)
    return rho_v + 0.5 * (rho_l - rho_v) * (1.0 - np.tanh((r - R) / width))


def equivalent_radius(rho: np.ndarray, mid: float) -> float:
    """Area-equivalent radius of the region ``rho > mid`` (cylinder: ``dp = sigma/R``)."""
    area = float((rho > mid).sum())
    return float(np.sqrt(area / np.pi))


def measure_droplet(
    rho: np.ndarray,
    G_lib: float = 5.0,
    psi_fn=psi_exp,
    ux: np.ndarray | None = None,
    uy: np.ndarray | None = None,
) -> dict:
    """Young-Laplace pressure jump, bulk densities and spurious current level.

    ``dp`` is the mechanical-EOS pressure difference between the droplet centre
    and the far field, both averaged over a stencil that excludes the diffuse
    interface (centre patch of radius ``R_eq/2``, outer annulus ``> 1.5 R_eq``).
    """
    L = rho.shape[0]
    y0 = np.arange(L)[:, None] + 0.5
    x0 = np.arange(L)[None, :] + 0.5
    r0 = np.hypot(x0 - L / 2, y0 - L / 2)
    rho_c = float(rho[L // 2, L // 2])
    rho_l_far = float(np.median(rho[r0 > 0.42 * L]))  # corner far field
    mid = 0.5 * (rho_c + rho_l_far)
    R_eq = equivalent_radius(rho, mid)
    y = np.arange(L)[:, None] + 0.5
    x = np.arange(L)[None, :] + 0.5
    r = np.hypot(x - L / 2, y - L / 2)
    inner = r < 0.5 * R_eq
    outer = r > 1.5 * R_eq
    p = eos_pressure(rho, G_lib, psi_fn)
    p_in = float(p[inner].mean())
    p_out = float(p[outer].mean())
    out = {
        "R_eq": R_eq,
        "r_in": rho_c,
        "r_out": rho_l_far,
        "p_in": p_in,
        "p_out": p_out,
        "dp": p_in - p_out,
        "sigma_laplace": (p_in - p_out) * R_eq,  # 2-D (cylinder): dp = sigma / R
        "rho_field": rho,
    }
    if ux is not None and uy is not None:
        out["u_max"] = float(np.hypot(ux, uy).max())
    return out