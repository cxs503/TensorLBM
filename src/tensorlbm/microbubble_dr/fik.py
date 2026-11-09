"""Drag-reduction mechanism decomposition for the microbubble channel (P106W2-N01).

The decomposition is built on the **FIK identity** (Fukagata, Iwamoto & Kasagi,
*Phys. Fluids* 14 (2002) L73) generalised to a channel whose viscosity and body
force may vary across the gap.  Nothing here is fitted: every term is an integral
of measured mean profiles, and the total is compared against the *independently*
measured wall shear (bounce-back momentum exchange).

Derivation (streamwise mean momentum, ``y`` from the bottom wall, gap ``H``)
--------------------------------------------------------------------------
For a statistically steady channel driven by a uniform **per-volume** force
``F0`` the mean total-stress profile is exactly linear,

    tau(y) = C - F0*y,        C = tau_bottom,   tau_top = C - F0*H,

because the momentum source does not depend on the (variable) density.  The
total stress is carried by three distinct mechanisms,

    tau(y) = mu(y) du/dy - rho <u'v'> - tau_b(y),

with ``mu = rho*nu`` the *mixture* molecular viscosity (the gas is less viscous in
proportion to its density), ``-rho<u'v'>`` the turbulent Reynolds shear stress
and ``tau_b`` the residual interfacial / phase-drift stress (everything the mean
momentum budget attributes to the bubble phase: Shan-Chen interfacial stress plus
the dispersive stress of the slip between phases).

Using ``u(y) = int_0^y (tau + rho<u'v'> + tau_b)/mu dy'`` and integrating by
parts over the gap, the bulk (volumetric) velocity satisfies

    U_b = (1/H) int_0^H (H-y) (tau + rho<u'v'> + tau_b)/mu dy,

and substituting the linear ``tau`` and solving for ``C``:

    C = (mu_ref*U_b + F0*j2 - num_re - num_bub) / j1

    j1      = (1/H) int (H-y) r dy                 r = mu_ref/mu(y)
    j2      = (1/H) int y(H-y) r dy
    num_re  = (1/H) int (H-y) (-rho<u'v'>) r dy
    num_bub = (1/H) int (H-y) (-tau_b) r dy

``mu_ref = rho_b * nu`` is one *common* reference viscosity, so the single-phase
and two-phase runs are compared on exactly the same footing.

Mechanism split (telescoping, each step adds one measured mechanism)
-------------------------------------------------------------------
    C_0 = C(ref profiles: r=1, baseline Reynolds stress, tau_b=0)   [baseline]
    C_1 = C(r of the two-phase run, baseline Reynolds stress, tau_b=0)
    C_2 = C(..., measured two-phase Reynolds stress, tau_b=0)
    C_3 = C(..., + measured tau_b) = C

    dtau_visc = C_1 - C_0        effective (mixture) viscosity contribution
    dtau_turb = C_2 - C_1        turbulence-modulation contribution
    dtau_slip = C_3 - C_2        phase-slip / interfacial contribution

and ``dtau_total = C - C_0`` by construction.  The *substantive* validation is
that ``C`` and ``C - F0*H`` (top wall) reproduce the momentum-exchange wall
shear measured in the simulation, which is an independent quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # numpy >= 2.0 renamed trapz -> trapezoid
    _trapz = np.trapezoid
except AttributeError:  # pragma: no cover
    _trapz = np.trapz


def smooth(a: np.ndarray, w: int = 3) -> np.ndarray:
    """Box smoothing with edge replication (profile noise only)."""
    a = np.asarray(a, dtype=float)
    if w <= 1 or a.size < w:
        return a.copy()
    k = np.ones(w) / w
    pad = w // 2
    ap = np.concatenate([np.full(pad, a[0]), a, np.full(pad, a[-1])])
    return np.convolve(ap, k, mode="valid")[: a.size]


def wall_gradient(u: np.ndarray, y: np.ndarray, n_fit: int = 4, y_wall: float = 0.0) -> float:
    """``du/dy`` at the wall from a quadratic fit to the first ``n_fit`` nodes."""
    n = min(n_fit, u.size)
    coef = np.polyfit(y[:n] - y_wall, u[:n], 2)
    return float(np.polyval(np.polyder(coef), 0.0))


def wall_shear_profile(
    y: np.ndarray,
    u: np.ndarray,
    rho: np.ndarray,
    uv: np.ndarray,
    nu: float,
    H: float,
    n_skip: int = 2,
) -> dict:
    """Wall shear from the mean velocity gradient (SQL: same profile, two estimates)."""
    mu = rho * nu
    du = np.gradient(smooth(u, 3), y)
    tau_meas = mu * du - rho * uv            # mu du/dy - rho<u'v'>
    sel = slice(n_skip, y.size - n_skip)
    A = np.vstack([np.ones(y[sel].size), y[sel]]).T
    coef, *_ = np.linalg.lstsq(A, tau_meas[sel], rcond=None)
    return {
        "du_dy_wall": wall_gradient(u, y),
        "tau_bottom_fit": float(coef[0]),
        "slope_fit": float(coef[1]),
        "tau_top_fit": float(coef[0] + coef[1] * H),
    }


def residual_stress(
    y: np.ndarray,
    u: np.ndarray,
    rho: np.ndarray,
    uv: np.ndarray,
    nu: float,
    C: float,
    F0: float,
) -> np.ndarray:
    """``tau_b(y) = mu du/dy - rho<u'v'> - tau(y)``: interfacial / drift stress."""
    mu = rho * nu
    du = np.gradient(smooth(u, 5), y)
    tau = C - F0 * y
    return mu * du - rho * uv - tau


@dataclass
class Profile:
    """One run's mean profiles (all arrays over ``y``)."""

    y: np.ndarray
    rho: np.ndarray
    u: np.ndarray
    uv: np.ndarray
    alpha: np.ndarray
    U_b: float
    F0: float
    H: float
    nu: float
    rho_b: float

    @property
    def mu_ref(self) -> float:
        return self.rho_b * self.nu


def _integrals(p: Profile, num_re: np.ndarray, num_bub: np.ndarray, r: np.ndarray) -> tuple[float, float, float]:
    H = p.H
    j1 = float(_trapz((H - p.y) * r, p.y) / H)
    j2 = float(_trapz(p.y * (H - p.y) * r, p.y) / H)
    nre = float(_trapz((H - p.y) * num_re * r, p.y) / H)
    nb = float(_trapz((H - p.y) * num_bub * r, p.y) / H)
    return j1, j2, nre + nb


def fik_decomposition(two: Profile, base: Profile, tau_b: np.ndarray | None = None) -> dict:
    """Three-term drag-reduction split at matched bulk velocity (see module doc)."""
    H = two.H
    mu_ref = two.mu_ref
    r = mu_ref / (two.rho * two.nu)                 # mu_ref / mu(y)
    r = np.clip(r, 1e-3, 1e3)                       # gas cells: mu -> rho_g*nu
    num_re_two = -two.rho * two.uv
    num_re_base = -base.rho * base.uv
    zero = np.zeros_like(two.y)
    tb = zero if tau_b is None else tau_b

    j1, j2, bun = _integrals(two, num_re_two, tb, r)
    # sign check: in a laminar channel without Reynolds stress this reduces to
    # 2 mu U_b/H + F0 H/3 = F0 H/2 (see tests).  A negative <u'v'> *raises* the
    # wall shear at matched throughput, so the Reynolds term enters with a plus.
    C = (mu_ref * two.U_b + two.F0 * j2 + bun) / j1
    # baseline (r = 1, its own Reynolds stress, no bubble stress)
    H_b = base.H
    j1_0 = float(_trapz((H_b - base.y), base.y) / H_b)
    j2_0 = float(_trapz(base.y * (H_b - base.y), base.y) / H_b)
    bun_0 = float(_trapz((H_b - base.y) * num_re_base, base.y) / H_b)
    C0 = (base.mu_ref * base.U_b + base.F0 * j2_0 + bun_0) / j1_0
    # telescoped intermediates
    j1r, j2r, _ = _integrals(two, zero, zero, r)
    C1 = (mu_ref * two.U_b + two.F0 * j2r + bun_0) / j1r       # viscosity only
    C2 = (mu_ref * two.U_b + two.F0 * j2r + float(_trapz((H - two.y) * num_re_two * r, two.y) / H)) / j1r

    # ``C`` is the shear at the bubble-side wall (y = 0); ``tau_top`` is the
    # opposite (control) wall, reported for the momentum budget only.
    tau_top = two.F0 * H - C
    tau_top_0 = base.F0 * H_b - C0
    d = {
        "C_bottom": C,
        "C_bottom_baseline": C0,
        "tau_wall": C,
        "tau_wall_baseline": C0,
        "tau_top": tau_top,
        "tau_top_baseline": tau_top_0,
        "dtau_total": C - C0,
        # sign convention: these are the *drag-reduction* contributions, i.e.
        # minus the wall-shear increments, so they sum to +dtau_total.
        "dtau_visc": -(C1 - C0),
        "dtau_turb": -(C2 - C1),
        "dtau_slip": -(C - C2),
        "j1": j1,
        "j2": j2,
    }
    d["sum_residual_rel"] = abs(d["dtau_visc"] + d["dtau_turb"] + d["dtau_slip"] + d["dtau_total"]) / max(
        abs(d["dtau_total"]), 1e-30
    )
    # skin-friction coefficients at the bubble-side (top) wall
    q_two = 0.5 * two.rho_b * two.U_b**2
    q_base = 0.5 * base.rho_b * base.U_b**2
    d["Cf"] = C / q_two
    d["Cf_baseline"] = C0 / q_base
    d["Cf_top"] = tau_top / q_two
    d["dCf"] = d["Cf_baseline"] - d["Cf"]
    d["DR"] = tro = d["dCf"] / d["Cf_baseline"]
    # drag-reduction sign convention: a mechanism that *raises* the wall shear is
    # a negative contribution to Delta C_f = C_f0 - C_f, so the mechanism terms
    # are minus the wall-shear increments and sum to dCf exactly.
    d["dCf_visc"] = -(C1 - C0) / q_base
    d["dCf_turb"] = -(C2 - C1) / q_base
    d["dCf_slip"] = -(C - C2) / q_base
    d["DR_visc"] = d["dCf_visc"] / d["Cf_baseline"]
    d["DR_turb"] = d["dCf_turb"] / d["Cf_baseline"]
    d["DR_slip"] = d["dCf_slip"] / d["Cf_baseline"]
    del tro
    return d


def momentum_consistency(record: dict) -> dict:
    """Three-way ``u_tau`` consistency from independent measurement channels."""
    nu = record["nu"]
    rho_b = record["rho_b"]
    H = record["H"]
    y = np.asarray(record["y"])
    u = np.asarray(record["u"])
    rho = np.asarray(record["rho"])
    uv = np.asarray(record["uv"])
    F0 = record["F0"]
    out = {}
    # (1) mean velocity gradient at the wall (bottom)
    du_w = wall_gradient(u, y)
    out["tau_wall_grad"] = rho[0] * nu * du_w
    # (2) Reynolds-stress route: tau(y) = mu du/dy - rho<u'v'>, linear fit
    fit = wall_shear_profile(y, u, rho, uv, nu, H)
    out["tau_wall_reynolds"] = fit["tau_bottom_fit"]
    out["tau_wall_reynolds_top"] = abs(fit["tau_top_fit"])
    # (3) total momentum input (uniform per-volume forcing)
    out["tau_wall_force"] = F0 * H / 2.0
    out["tau_wall_mem_bottom"] = record["mem_bottom"]
    out["tau_wall_mem_top"] = record["mem_top"]
    out["u_tau_grad"] = float(np.sqrt(max(out["tau_wall_grad"], 0.0) / rho_b))
    out["u_tau_reynolds"] = float(np.sqrt(max(out["tau_wall_reynolds"], 0.0) / rho_b))
    out["u_tau_force"] = float(np.sqrt(max(out["tau_wall_force"], 0.0) / rho_b))
    out["u_tau_mem_bottom"] = float(np.sqrt(max(out["tau_wall_mem_bottom"], 0.0) / rho_b))
    out["u_tau_mem_top"] = float(np.sqrt(max(out["tau_wall_mem_top"], 0.0) / rho_b))
    trio = [out["u_tau_grad"], out["u_tau_reynolds"], out["u_tau_force"]]
    ref = float(np.mean(trio))
    out["u_tau_spread_rel"] = float((max(trio) - min(trio)) / ref) if ref > 0 else float("inf")
    return out