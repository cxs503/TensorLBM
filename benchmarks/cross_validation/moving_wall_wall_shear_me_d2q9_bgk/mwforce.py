"""Moving-wall half-way bounce-back (D2Q9 + BGK) — **wall-force** verification core.

Anchor: ``XVAL-LBM-MOVING-WALL-WALL-SHEAR-ME-D2Q9-BGK``.

This module adds to the merged kinematic study
(``benchmarks/cross_validation/moving_wall_halfway_bb_d2q9_bgk``, #2342) the piece that study
explicitly left out: the **wall force** (momentum exchange) produced by the same
closure, and its momentum-accounting identity.

State update = the adjudicated reference closure ``ladd_pre_local`` of
``mwbb.make_step`` (pre-streaming reflection of the pre-collision populations +
Ladd term ``2 w_q rho_w (c_q.u_w)/cs^2``), reproduced here explicitly so that the
populations crossing the wall link are visible.  Equivalence with
``mwbb.make_step`` is pinned bit-for-bit by
``tests/test_xval_moving_wall_wall_shear_me.py``.

Momentum-exchange (ME) accounting
---------------------------------
For the bottom wall (row 0, normal +y, fluid rows 1..ny-2):

* populations leaving the fluid into the wall: ``q in DOWN = (4,7,8)`` (``c_qy=-1``)
  read at fluid row 1 *after collision*;
* populations entering the fluid from the wall: ``q in UP = (2,5,6)`` (``c_qy=+1``)
  read at wall row 0 *after* the bounce-back (i.e. the reflected + Ladd state);

so the impulse delivered to the wall in one step is

    F_bot = sum_x [ sum_{q in DOWN} c_q f_post[q,1,x] - sum_{q in UP} c_q f_new[q,0,x] ]

(the top wall is the mirror image).  ``F`` is the force *on the wall* in lattice
units (momentum per step) summed over the ``nx`` periodic cells; ``F/nx`` is the
wall shear stress per unit area (unit depth).

Because BGK collision and x-periodicity conserve momentum exactly, the ME count
must satisfy, at machine precision,

    F_bot + F_top + dP_fluid = 0 ,  dP_fluid = sum_{fluid} c_q (f_next - f_post)

which is the invariant the gate test asserts (M6).  A wrong ME formula (missing
factor 2, wrong normal, pre- vs post-collision sampling) breaks this identity,
which is why it is a zero-tolerance pin rather than a physics tolerance.

Analytic references
-------------------
* steady plane Couette: ``tau_w = rho nu U0 / H_eff`` (exact N-S solution).
* Stokes' 2nd problem (plate ``U0 cos(omega t)``): the wall shear is obtained by
  *numerically differentiating the analytic profile* it comes from
  (``tau_fluid = rho nu du/dy`` at ``y = 0.5``), so no sign or phase is copied by
  hand; the closed form ``rho U0 sqrt(nu omega) sin(omega t - pi/4)`` is used only
  as a cross-check of the numerical derivative (``analytic_selfcheck``).
* Rayleigh impulsive start (semi-infinite): ``u = U0 erfc(eta/(2 sqrt(nu t)))``,
  same numerical-derivative treatment; closed form ``-rho U0 sqrt(nu/(pi t))``
  as cross-check.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SIBLING = _HERE.parent / "moving_wall_halfway_bb_d2q9_bgk"
for _p in (str(_SIBLING),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mwbb  # noqa: E402  (also installs repo root + src on sys.path)

from tensorlbm.d2q9 import OPPOSITE, C, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

CS2 = mwbb.CS2
DTYPE = mwbb.DTYPE
DOWN = (4, 7, 8)  # c_qy = -1 : fluid -> bottom wall
UP = (2, 5, 6)  # c_qy = +1 : bottom wall -> fluid

VARIANTS = mwbb.VARIANTS


def inject(rho_w: torch.Tensor, u_wall: torch.Tensor) -> torch.Tensor:
    """Ladd momentum term ``6 w_q rho_w (c_q . u_w)`` for a vector wall velocity.

    ``u_wall`` has shape ``(2, ny, nx)``.  For ``u_wall = (ux, 0)`` this is
    ``mwbb._inject`` exactly (checked in the gate test).
    """
    c = C.to(DTYPE)
    w = mwbb._w_for(DTYPE, torch.device("cpu")).view(9, 1, 1)
    cu = c[:, 0].view(9, 1, 1) * u_wall[0].unsqueeze(0) + c[:, 1].view(9, 1, 1) * u_wall[
        1
    ].unsqueeze(0)
    return 2.0 * w * rho_w.unsqueeze(0) * cu / CS2


def make_step_flux(ny: int, nx: int, tau: float, variant: str = "ladd_pre_local"):
    """Return ``step(f, u_wall) -> (f_next, F_bot, F_top, dP, rho_w)``.

    ``u_wall`` is ``(2, ny, nx)`` float64 (wall x/y velocity, read at the wall
    rows only).  ``F_bot``/``F_top`` are momentum-exchange forces **on the wall**
    summed over x, ``dP`` the fluid momentum change of the step (vector), and
    ``rho_w`` the mean wall-row density used by the Ladd term.
    """
    if variant not in VARIANTS and variant != "ladd_vec":
        raise ValueError(f"unknown variant {variant!r}")
    c = C.to(DTYPE)
    opp = OPPOSITE
    iu = torch.tensor(UP)
    idn = torch.tensor(DOWN)
    wall = torch.zeros((ny, nx), dtype=torch.bool)
    wall[0, :] = True
    wall[-1, :] = True
    wmask = wall.unsqueeze(0)
    ones = torch.ones((ny, nx), dtype=DTYPE)
    fluid = slice(1, ny - 1)

    def step(f: torch.Tensor, u_wall: torch.Tensor):
        f_post = collide_bgk(f, tau)
        if variant == "eq_wall_node":
            feq_w = equilibrium(ones, u_wall[0], u_wall[1])
            f_new = torch.where(wmask, feq_w, f_post)
        elif variant == "ladd_precollision":
            raise NotImplementedError(
                "ablation not re-implemented in the force-instrumented step; "
                "the merged study pins its behaviour (term added before collision "
                "is annihilated by the bounce-back) -- see mwbb.make_step"
            )
        else:
            f_new = torch.where(wmask, f[opp], f_post)
            if variant != "static_bb":
                rho_w_f = ones if variant == "ladd_pre_unitrho" else f.sum(dim=0)
                inj = inject(rho_w_f, u_wall)
                if variant == "ladd_sign_flip":
                    inj = -inj
                f_new = f_new + inj * wmask
        f_next = stream(f_new)

        # ---- momentum exchange across the two walls ----
        f_bot_out = f_post[idn][:, 1, :]  # fluid -> bottom wall
        f_bot_in = f_new[iu][:, 0, :]  # bottom wall -> fluid
        f_top_out = f_post[iu][:, ny - 2, :]  # fluid -> top wall
        f_top_in = f_new[idn][:, ny - 1, :]  # top wall -> fluid
        F_bot = c[idn].T @ f_bot_out.sum(dim=1) - c[iu].T @ f_bot_in.sum(dim=1)
        F_top = c[iu].T @ f_top_out.sum(dim=1) - c[idn].T @ f_top_in.sum(dim=1)
        dP = ((f_next[:, fluid, :] - f_post[:, fluid, :]).unsqueeze(1) * c.view(9, 2, 1, 1)).sum(
            dim=(0, 2, 3)
        )
        rho_w = float(f.sum(dim=0)[0].mean()) if variant != "eq_wall_node" else 1.0
        return f_next, F_bot, F_top, dP, torch.tensor(rho_w, dtype=DTYPE)

    return step


# --------------------------------------------------------------------------
# analytic references
# --------------------------------------------------------------------------
def couette_tau(nu: float, u0: float, h_eff: float, rho: float = 1.0) -> float:
    """Steady plane-Couette wall shear stress ``rho nu U0 / H_eff`` (unit depth)."""
    return rho * nu * u0 / h_eff


def _dudy_at_wall(u_of_y, y_wall: float = 0.5, h: float = 1e-3) -> float:
    """Central difference of an analytic ``u(y)`` at the no-slip surface."""
    return (u_of_y(y_wall + h) - u_of_y(y_wall - h)) / (2.0 * h)


def stokes2_u_of_y(t: float, u0: float, omega: float, nu: float):
    k = math.sqrt(omega / (2.0 * nu))
    return lambda y: u0 * math.exp(-k * (y - 0.5)) * math.cos(omega * t - k * (y - 0.5))


def stokes2_tau(t: float, u0: float, omega: float, nu: float, rho: float = 1.0) -> float:
    """Wall shear stress on the **fluid** from the oscillating plate, analytic.

    ``tau_fluid = rho nu du/dy|_{y=0.5}`` with ``u`` from
    :func:`stokes2_u_of_y`; force on the wall is ``-tau_fluid * A``.
    """
    return rho * nu * _dudy_at_wall(stokes2_u_of_y(t, u0, omega, nu))


def rayleigh_u_of_y(t: float, u0: float, nu: float):
    s = 2.0 * math.sqrt(nu * t) if t > 0 else 1.0
    return lambda y: u0 * math.erfc((y - 0.5) / s)


def rayleigh_tau(t: float, u0: float, nu: float, rho: float = 1.0) -> float:
    """Rayleigh (impulsive plate) wall shear on the fluid, half-space solution."""
    return rho * nu * _dudy_at_wall(rayleigh_u_of_y(t, u0, nu))


def analytic_selfcheck(nu: float, u0: float, omega: float, t: float) -> dict:
    """Numerical-derivative reference vs the textbook closed forms."""
    tau_s2_num = stokes2_tau(t, u0, omega, nu)
    tau_s2_cf = rho_times(u0, nu, omega) * math.sin(omega * t - math.pi / 4.0)
    tau_ray_num = rayleigh_tau(t, u0, nu)
    tau_ray_cf = -u0 * math.sqrt(nu / (math.pi * t))
    return {
        "stokes2_num": tau_s2_num,
        "stokes2_closed_form": tau_s2_cf,
        "stokes2_rel_diff": abs(tau_s2_num - tau_s2_cf) / abs(tau_s2_cf),
        "rayleigh_num": tau_ray_num,
        "rayleigh_closed_form": tau_ray_cf,
        "rayleigh_rel_diff": abs(tau_ray_num - tau_ray_cf) / abs(tau_ray_cf),
    }


def rho_times(u0: float, nu: float, omega: float, rho: float = 1.0) -> float:
    """``rho U0 sqrt(nu omega)`` — Stokes-2nd wall-stress amplitude."""
    return rho * u0 * math.sqrt(nu * omega)


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------
def _wall_field(ny: int, nx: int, ubot: tuple[float, float], utop: tuple[float, float]):
    u = torch.zeros((2, ny, nx), dtype=DTYPE)
    u[0, 0, :] = ubot[0]
    u[1, 0, :] = ubot[1]
    u[0, -1, :] = utop[0]
    u[1, -1, :] = utop[1]
    return u


def run_couette_shear(
    ny: int,
    nx: int,
    tau: float,
    u0: float,
    variant: str = "ladd_pre_local",
    n_steps: int = 20000,
    avg_steps: int = 400,
    tol: float = 1e-9,
) -> dict:
    """Steady plane Couette driven by the bottom wall (u=U0), top wall at rest."""
    nu = (tau - 0.5) / 3.0
    h_eff = ny - 2.0
    step = make_step_flux(ny, nx, tau, variant)
    y = np.arange(ny, dtype=float)
    f = mwbb.init_field(ny, nx, u0 * np.clip((y - 0.5) / h_eff, 0.0, 1.0))
    u_wall = _wall_field(ny, nx, (0.0, 0.0), (u0, 0.0))  # top wall drags, bottom at rest
    m0 = float(f.sum())
    F_acc = torch.zeros(2, dtype=DTYPE)
    F_acc_top = torch.zeros(2, dtype=DTYPE)
    id_res = 0.0
    hist: list[float] = []
    steady = False
    n_done = 0
    # settle, then average the ME force over ``avg_steps`` steady steps
    for k in range(1, n_steps + 1):
        f, Fb, Ft, dP, rho_w = step(f, u_wall)
        n_done = k
        if k % 200 == 0:
            _, ux, _ = macroscopic(f)
            hist.append(float(ux[ny - 2, nx // 2]))
            if len(hist) >= 10:
                rec = hist[-10:]
                if (max(rec) - min(rec)) / max(abs(float(sum(rec)) / len(rec)), 1e-30) < tol:
                    steady = True
                    break
    for _ in range(avg_steps):
        f, Fb, Ft, dP, rho_w = step(f, u_wall)
        F_acc += Fb
        F_acc_top += Ft
        id_res = max(id_res, float((Fb + Ft + dP).abs().max()))
    F_bot = (F_acc / avg_steps).numpy()
    F_top = (F_acc_top / avg_steps).numpy()
    _, ux, _ = macroscopic(f)
    u_num = ux[:, nx // 2].detach().numpy()
    _, u_ana = mwbb.couette_analytic(ny, u0)
    # wall shear *magnitude*: both walls must report +rho nu U0/H_eff
    tau_num_top = -float(F_top[0]) / nx  # moving wall drags fluid +x -> reaction -x
    tau_num_bot = float(F_bot[0]) / nx  # static wall reaction is +x
    tau_ana = couette_tau(nu, u0, h_eff, rho=float(rho_w))
    return {
        "case": "couette_shear",
        "variant": variant,
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "nu": nu,
        "H_eff": h_eff,
        "U0": u0,
        "Ma": u0 / math.sqrt(CS2),
        "Re": u0 * h_eff / nu,
        "n_steps": n_done,
        "steady": steady,
        "avg_steps": avg_steps,
        "rho_w": float(rho_w),
        "F_bot": [float(v) for v in F_bot],
        "F_top": [float(v) for v in F_top],
        "tau_num": tau_num_top,
        "tau_num_bot": tau_num_bot,
        "tau_ana": tau_ana,
        "tau_err_pct": (tau_num_top - tau_ana) / tau_ana * 100.0,
        "tau_bot_err_pct": (tau_num_bot - tau_ana) / tau_ana * 100.0,
        "force_balance_rel": abs(float(F_bot[0] + F_top[0])) / max(abs(float(F_bot[0])), 1e-30),
        "me_identity_res_abs": id_res,
        "me_identity_res_rel": id_res / max(abs(tau_ana * nx), 1e-30),
        "mass_drift_pct": (float(f.sum()) - m0) / m0 * 100.0,
        "u_max_err_pct": float(np.max(np.abs(u_num[1:-1] - u_ana[1:-1])) / u0 * 100.0),
        "u_num": [float(v) for v in u_num],
    }


def run_stokes2_shear(
    ny: int,
    nx: int,
    tau: float,
    u0: float,
    delta_bl: float,
    variant: str = "ladd_pre_local",
    n_periods: float = 6.0,
    nu: float | None = None,
) -> dict:
    """Stokes' 2nd problem: bottom wall oscillates ``U0 cos(omega t)``; ME force history."""
    if nu is None:
        nu = (tau - 0.5) / 3.0
    omega = 2.0 * nu / delta_bl**2
    period = 2.0 * math.pi / omega
    n_steps = int(round(n_periods * period))
    step = make_step_flux(ny, nx, tau, variant)
    f = mwbb.init_field(ny, nx)
    m0 = float(f.sum())
    n_last = int(round(period))
    ts, fxs, fx_ana, id_res, fscale = [], [], [], 0.0, 0.0
    for k in range(1, n_steps + 1):
        t = float(k)
        uw = torch.zeros((2, ny, nx), dtype=DTYPE)
        uw[0, 0, :] = u0 * math.cos(omega * t)
        f, Fb, Ft, dP, rho_w = step(f, uw)
        id_res = max(id_res, float((Fb + Ft + dP).abs().max()))
        fscale = max(fscale, abs(float(Fb[0])))
        if k > n_steps - n_last:
            ts.append(t)
            fxs.append(float(Fb[0]) / nx)  # force on wall / area
            fx_ana.append(stokes2_tau(t, u0, omega, nu, rho=float(rho_w)))
    ts = np.array(ts)
    fxs = np.array(fxs)
    fx_ana = np.array(fx_ana)
    fit_num = harmonic(ts, fxs, omega)
    fit_ana = harmonic(ts, fx_ana, omega)
    return {
        "case": "stokes2_shear",
        "variant": variant,
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "nu": nu,
        "U0": u0,
        "Ma": u0 / math.sqrt(CS2),
        "omega": omega,
        "period_steps": period,
        "delta_bl": delta_bl,
        "n_steps": n_steps,
        "rho_w": float(rho_w),
        "amp_num": fit_num["amp"],
        "amp_ana": fit_ana["amp"],
        "amp_err_pct": (fit_num["amp"] - fit_ana["amp"]) / fit_ana["amp"] * 100.0,
        "phase_num_deg": fit_num["phase_deg"],
        "phase_ana_deg": fit_ana["phase_deg"],
        "phase_err_deg": _wrap_deg(fit_num["phase_deg"] - fit_ana["phase_deg"]),
        "me_identity_res_abs": id_res,
        "me_identity_res_rel": id_res / max(fscale, 1e-30),
        "mass_drift_pct": (float(f.sum()) - m0) / m0 * 100.0,
        "t": [float(v) for v in ts],
        "fx_num": [float(v) for v in fxs],
        "fx_ana": [float(v) for v in fx_ana],
        "amp_closed_form": rho_times(u0, nu, omega),
    }


def run_rayleigh_shear(
    ny: int,
    nx: int,
    tau: float,
    u0: float,
    n_steps: int = 2000,
    win: tuple[int, int] = (400, 1600),
    nu: float | None = None,
) -> dict:
    """Impulsive start of the bottom wall from rest (half-space similarity solution)."""
    if nu is None:
        nu = (tau - 0.5) / 3.0
    step = make_step_flux(ny, nx, tau)
    f = mwbb.init_field(ny, nx)
    m0 = float(f.sum())
    uw = _wall_field(ny, nx, (u0, 0.0), (0.0, 0.0))
    ts, fxs, fx_ana, id_res, fscale = [], [], [], 0.0, 0.0
    for k in range(1, n_steps + 1):
        f, Fb, Ft, dP, rho_w = step(f, uw)
        id_res = max(id_res, float((Fb + Ft + dP).abs().max()))
        fscale = max(fscale, abs(float(Fb[0])))
        ts.append(float(k))
        fxs.append(float(Fb[0]) / nx)
        fx_ana.append(rayleigh_tau(float(k), u0, nu, rho=float(rho_w)))
    ts = np.array(ts)
    fxs = np.array(fxs)
    fx_ana = np.array(fx_ana)
    lo, hi = win
    sel = (ts >= lo) & (ts <= hi)
    rel = np.abs(fxs[sel] - fx_ana[sel]) / np.max(np.abs(fx_ana[sel]))
    return {
        "case": "rayleigh_shear",
        "variant": "ladd_pre_local",
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "nu": nu,
        "U0": u0,
        "Ma": u0 / math.sqrt(CS2),
        "n_steps": n_steps,
        "win": [lo, hi],
        "depth_over_bl_at_hi": float(math.sqrt(nu * hi) / (ny - 2.0)),
        "err_pct_max_in_win": float(np.max(rel) * 100.0),
        "err_pct_mean_in_win": float(np.mean(rel) * 100.0),
        "me_identity_res_abs": id_res,
        "me_identity_res_rel": id_res / max(fscale, 1e-30),
        "mass_drift_pct": (float(f.sum()) - m0) / m0 * 100.0,
        "t": [float(v) for v in ts[sel]],
        "fx_num": [float(v) for v in fxs[sel]],
        "fx_ana": [float(v) for v in fx_ana[sel]],
        "fx_num_full": [float(v) for v in fxs],
        "t_full": [float(v) for v in ts],
    }


def run_normal_wall_drift(ny: int, nx: int, tau: float, uw_n: float, n_steps: int = 400) -> dict:
    """Diagnostic D1: wall with a **normal** velocity component (u_w . n != 0).

    The Ladd term is a no-slip closure; a wall moving along its own normal is a
    permeable/piston configuration it is not derived for.  This run measures the
    resulting fluid-mass source instead of asserting anything about it.

    The wall cell's stored state is not part of the fluid domain, so the mass
    balance is taken over the fluid rows only: ``m_fluid(n) - m_fluid(0)``.
    """
    step = make_step_flux(ny, nx, tau)
    f = mwbb.init_field(ny, nx)
    m0 = float(f[1:-1].sum())
    uw = _wall_field(ny, nx, (0.0, uw_n), (0.0, 0.0))
    hist = []
    for k in range(1, n_steps + 1):
        f, Fb, Ft, dP, rho_w = step(f, uw)
        if k % 50 == 0:
            hist.append((float(k), float(f[1:-1].sum()) - m0))
    dm = hist[-1][1] if hist else 0.0
    rate_meas = dm / n_steps
    return {
        "case": "normal_wall_drift",
        "uw_n": uw_n,
        "n_steps": n_steps,
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "dm_fluid": dm,
        "dm_fluid_rel_pct": dm / m0 * 100.0,
        "rate_per_step": rate_meas,
        "rate_predicted_rho_uw_n_x_nx": uw_n * nx,  # see REPORT: sum_{UP} inj = rho u_w,n
        "rate_ratio": rate_meas / (uw_n * nx) if uw_n else float("nan"),
        "hist_t": [h[0] for h in hist],
        "hist_dm": [h[1] for h in hist],
    }


def impulse_identity(seed: int = 7, ny: int = 9, nx: int = 5) -> dict:
    """Operator-level pin (M7) of the Ladd term's momentum content.

    Half-lattice algebra: with ``inj_q = 6 w_q rho_w (c_q . u_w)`` and
    ``UP = {q : c_qy = +1}``,

        sum_{q in UP} c_q inj_q = rho_w * (cs^2 u_x, u_y) ,   cs^2 = 1/3 ,

    i.e. the **normal** component carries the full mass flux ``rho u_n`` while the
    tangential component carries ``rho cs^2 u_t`` per wall link (the remaining
    tangential momentum exchanges through the returning flux difference, which
    is what the ME count adds).  Summed over all nine directions the term carries
    ``2 rho u`` — the factor that makes the map a Galilean fixed point
    (``tests/test_moving_wall_convention_audit.py``).

    The identity is exact in floating point (it is a weighted sum over three
    directions), so it is a zero-tolerance pin: a wrong weight table, a missing
    ``rho_w``, or a transposed wall normal all break it.
    """
    g = torch.Generator().manual_seed(seed)
    rho = torch.rand((ny, nx), generator=g, dtype=torch.float64) + 0.5
    uw = torch.rand((2, ny, nx), generator=g, dtype=torch.float64) * 0.05
    inj = inject(rho, uw)
    c = C.to(DTYPE)
    iu = torch.tensor(UP)
    lhs = torch.stack(
        [
            (c[iu, 0].view(3, 1, 1) * inj[iu]).sum(dim=0),
            (c[iu, 1].view(3, 1, 1) * inj[iu]).sum(dim=0),
        ]
    )
    rhs = torch.stack([rho * uw[0] * CS2, rho * uw[1]])
    ex = lhs - rhs
    lhs_all = (c.view(9, 2, 1, 1) * inj.unsqueeze(1)).sum(dim=0)  # (2, ny, nx)
    rhs_all = 2.0 * rho.unsqueeze(0) * uw
    return {
        "case": "impulse_identity",
        "rel_res": float(ex.abs().max()) / float(rhs.abs().max()),
        "abs_res": float(ex.abs().max()),
        "identity": "sum_{q in UP} c_q inj_q = rho_w (cs^2 u_x, u_y)",
        "all_dirs_rel_res": float((lhs_all - rhs_all).abs().max()) / float(rhs_all.abs().max()),
        "all_dirs_identity": "sum_{all q} c_q inj_q = 2 rho_w u_w",
    }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def harmonic(ts: np.ndarray, x: np.ndarray, omega: float) -> dict:
    """Fundamental-harmonic amplitude (same units as ``x``) and phase in degrees."""
    a = 2.0 / len(ts) * float(np.sum(x * np.cos(omega * ts)))
    b = 2.0 / len(ts) * float(np.sum(x * np.sin(omega * ts)))
    amp = math.hypot(a, b)
    phase = math.degrees(math.atan2(-b, a))  # x ~ amp*cos(omega t - phase_rad)
    return {"amp": amp, "phase_deg": phase, "a": a, "b": b}


def _wrap_deg(d: float) -> float:
    while d > 180.0:
        d -= 360.0
    while d < -180.0:
        d += 360.0
    return d
