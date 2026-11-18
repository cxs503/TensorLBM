"""Air-lubrication frontier layer (IN45-N07): hull-relative buoyancy, Madavan
1985 void-fraction shape, air-flow optimum, curvature / injection-position
retention, and the Re-scan contract used on the loong64 cluster.

What this module adds over :mod:`tensorlbm.air_lubrication`
----------------------------------------------------------
``air_lubrication`` owns the kernels (stratified exact channel, D2Q9 carrier,
conservative void-fraction transport).  This module owns the *frontier case*
built on them and fixes one modelling defect found there:

**Hull-relative buoyancy sign.**  A ship bottom is a ceiling: it bounds the water
from above, so an air bubble under it is pushed *toward* the hull by buoyancy and
is retained there.  In the transport coordinate used by
:class:`~tensorlbm.air_lubrication.VoidFractionTransport2D` the hull is the
``y = 0`` wall and the free stream is at ``y > 0``, therefore the buoyant drift
that enters the generic kernel is ``-V_b`` (toward the wall), not ``+V_b``.  The
P21-N05 campaign of this repository fed ``+V_b`` and consequently reported a
*non-wall-peaked* void fraction and a WITHHELD Madavan comparison; both are
artefacts of that sign, not physics.  :func:`wall_directed_drift` and
:func:`simulate_curtain` use the hull-relative sign and the test suite pins it
with a negative control (``V_b = 0`` -> no wall peak; ``V_b > 0`` toward the hull
-> wall-peaked profile).

Contents
--------
1. :func:`ittc57_cf` -- ITTC-57 friction line, the ``C_f0`` normalisation.
2. :func:`madavan1985_profile` -- the published wall-peaked similarity shape
   (``alpha = alpha_w exp(-y/delta_b)``, ``delta_b/delta = 0.4-0.6``) with its
   provenance recorded in the returned dict (digitised points are **not**
   available offline; only the published shape/band is used).
3. :func:`profile_deviation` -- shape agreement metrics (band-normalised L2 and
   max deviation, optimum ``delta_b`` fit) used for acceptance criterion C2.
4. :func:`simulate_curtain` -- one drift-flux curtain run under the ship
   operating point: wall-directed buoyancy, wall-blocked turbulent dispersion,
   optional hull-curvature drift and a finite-length injection strip; returns
   ``alpha(y)``, ``delta_b/delta``, capture efficiency and the measured species
   mass-budget residual (criterion C3).
5. :func:`curtain_drag_reduction` -- exact stratified ``Delta C_f`` of the
   simulated ``alpha(y)`` (PAL limiting law), and
   :func:`energy_balance` / :func:`optimize_air_flow` -- the air-flow optimum
   with the isothermal compressor cost (criterion C4).
6. :func:`curvature_injection_study` -- capture efficiency vs injection position
   for several hull radii of curvature, with monotonicity statistics
   (criterion C5).
7. :func:`re_scan_case` -- the single (Re, alpha_wall) case of the 3 x 5 lattice
   scan executed on the loong64 cluster (criterion C7); the driver script
   ``experiments/air_lubrication_in45_n07/re_scan_worker.py`` wraps it.

Evidence grading
----------------
* stratified exact channel, ITTC-57, compressor thermodynamics: REAL.
* drift-flux transport with the corrected buoyancy sign: REAL (solver) --
  the closure constants (``Sc_t``, wall blocking) are stated, not calibrated.
* Madavan 1985 comparison: **shape/band** comparison (qualitative +
  quantitative band fit).  Point-by-point digitised experiment is WITHHELD.
* lattice ``Delta C_f``: REAL (numerical vs analytic reference) at the
  lattice-realisable viscosity ratio, WITHHELD at the true air/water ratio
  (a BGK lattice cannot reach ``lambda = 1.8e-2``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .air_lubrication import (
    LAMBDA_AIR_WATER,
    MU_WATER,
    NU_WATER,
    RHO_WATER,
    ChannelAirLubrication2D,
    VoidFractionTransport2D,
    air_flow_rate_from_alpha,
    compressor_power,
    curvature_drift_velocity,
    mixture_viscosity,
    pal_drag_reduction,
    rise_velocity_schiller_naumann,
    stratified_channel_exact,
    stratified_drag_reduction_from_alpha,
)

# ---------------------------------------------------------------------------
# reference ship operating point (same hull as the P21-N05 case, so the two
# layers of the repository are directly comparable)
# ---------------------------------------------------------------------------
U_SHIP = 5.14  # m/s (10 kn)
DELTA = 0.10  # boundary-layer thickness [m]
RHO = RHO_WATER  # 998 kg/m3
NU = NU_WATER  # 1.0e-6 m2/s
S_WET = 5440.0  # wetted area [m2] of a 100 m x 20 m x 6 m hull
L_HULL = 100.0  # [m]
B_HULL = 20.0  # [m]
INJECTION_DEPTH = 6.0  # [m]
ETA_COMPRESSOR = 0.6  # isothermal efficiency
U_TAU_OVER_U = 0.0289  # sqrt(Cf0/2) at the reference Re (computed below)
KAPPA_VON_KARMAN = 0.41

#: reference Reynolds number and ITTC-57 baseline
RE_SHIP = U_SHIP * L_HULL / NU


def ittc57_cf(re: float) -> float:
    """ITTC-1957 friction coefficient ``0.075 / (log10 Re - 2)**2``."""
    if re <= 0.0:
        raise ValueError("Reynolds number must be positive")
    return 0.075 / (np.log10(re) - 2.0) ** 2


CF0_SHIP = float(ittc57_cf(RE_SHIP))


# ---------------------------------------------------------------------------
# C2 reference: Madavan et al. (1985) void-fraction shape
# ---------------------------------------------------------------------------
#: Published bubble-layer thickness band, ``delta_b/delta``.  Madavan, Deutsch &
#: Merkle (1985, Phys. Fluids 28, 1422) report a microbubble layer occupying the
#: inner O(50%) of the turbulent boundary layer, i.e. ``delta_b/delta ~ 0.4-0.6``.
MADAVAN_DELTA_B_BAND = (0.4, 0.6)


def madavan1985_profile(
    eta: Sequence[float],
    alpha_wall: float,
    delta_b_over_delta: float = 0.5,
) -> Dict[str, object]:
    """Wall-peaked void-fraction reference shape after Madavan et al. (1985).

    ``alpha(y) = alpha_wall exp(-y/delta_b)`` with ``delta_b =
    delta_b_over_delta * delta``; ``eta = y/delta``.  The *shape* (monotone,
    maximum at the hull, exponential decay over the inner half of the boundary
    layer) and the ``delta_b/delta`` band are the published result; the digitised
    point table is not available offline and is therefore not encoded -- the
    returned provenance string says so and the report keeps the corresponding
    evidence grade.
    """
    eta = np.asarray(eta, dtype=np.float64)
    if alpha_wall < 0.0 or alpha_wall > 1.0:
        raise ValueError("alpha_wall must lie in [0, 1]")
    if delta_b_over_delta <= 0.0:
        raise ValueError("delta_b_over_delta must be positive")
    prof = alpha_wall * np.exp(-eta / delta_b_over_delta)
    return {
        "profile": prof,
        "eta": eta,
        "alpha_wall": float(alpha_wall),
        "delta_b_over_delta": float(delta_b_over_delta),
        "provenance": (
            "shape + delta_b/delta band of Madavan et al. 1985; digitised "
            "experimental points WITHHELD (not available offline)"
        ),
    }


def profile_deviation(
    alpha_num: Sequence[float],
    eta: Sequence[float],
    alpha_wall: Optional[float] = None,
) -> Dict[str, float]:
    """Deviation of a computed ``alpha(y)`` from the Madavan reference family.

    The reference is evaluated at the fitted layer thickness (the single free
    parameter the experiment also leaves free) and at the *same* wall void
    fraction, so the comparison is a shape comparison: how well the drift-flux
    transport reproduces the published inner-layer decay once its wall value is
    matched.  ``l2_rel`` is the band-mean relative difference,
    ``max_rel`` the pointwise worst, both normalised by ``alpha_wall``.
    """
    a = np.asarray(alpha_num, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    if a.size != eta.size:
        raise ValueError("alpha and eta must have the same length")
    aw = float(a[0]) if alpha_wall is None else float(alpha_wall)
    if aw <= 0.0:
        raise ValueError("alpha_wall must be positive")
    # fit delta_b/delta by a 1-D least-squares scan on a log grid
    grid = np.exp(np.linspace(np.log(0.05), np.log(2.0), 240))
    # ignore the rows where the numerical profile has fallen into round-off
    mask = a > 1e-12 * aw
    best = None
    for db in grid:
        ref = aw * np.exp(-eta / db)
        err = float(np.mean((a[mask] - ref[mask]) ** 2))
        if best is None or err < best[0]:
            best = (err, db)
    db_fit = float(best[1])
    ref = aw * np.exp(-eta / db_fit)
    rel = np.abs(a[mask] - ref[mask]) / aw
    deta = float(eta[1] - eta[0])
    db_int = float(np.sum(a) * deta / aw)
    return {
        "alpha_wall": aw,
        "delta_b_over_delta_fit": db_fit,
        "delta_b_over_delta_integral": db_int,
        "l2_rel": float(np.sqrt(np.mean((a[mask] - ref[mask]) ** 2)) / aw),
        "max_rel": float(np.max(rel)),
        # the published band is an estimate of the layer thickness; both the
        # integral definition and the exponential-fit definition estimate it,
        # so the case is in band if either one lands inside
        "band_ok": bool(
            MADAVAN_DELTA_B_BAND[0] <= db_fit <= MADAVAN_DELTA_B_BAND[1]
            or MADAVAN_DELTA_B_BAND[0] <= db_int <= MADAVAN_DELTA_B_BAND[1]
        ),
        "wall_peaked": bool(np.all(np.diff(a[: max(4, a.size // 4)]) <= 1e-12)),
    }


# ---------------------------------------------------------------------------
# C2/C3/C5: the curtain itself
# ---------------------------------------------------------------------------
def boundary_layer_velocity(eta: np.ndarray, u_max: float = 1.0) -> np.ndarray:
    """One-seventh-power boundary-layer profile (``eta`` clipped to the edge)."""
    eta = np.asarray(eta, dtype=np.float64)
    return u_max * np.clip(eta, 0.0, 1.0) ** (1.0 / 7.0)


def turbulent_diffusivity(
    eta: np.ndarray,
    u_max: float = 1.0,
    u_tau_over_u: float = U_TAU_OVER_U,
    delta: float = 1.0,
    sc_t: float = 1.0,
) -> np.ndarray:
    """Wall-blocked turbulent diffusivity ``nu_t/Sc_t`` of the boundary layer.

    ``nu_t = kappa u_tau y (1 - eta)^2`` inside the boundary layer and a small
    outer value beyond it; ``Sc_t`` is the turbulent Schmidt number of the
    disperse phase (``1`` = passive scalar; the sensitivity to ``Sc_t`` is
    reported with the results because it is the one closure constant).
    """
    eta = np.asarray(eta, dtype=np.float64)
    inside = np.clip(eta, 0.0, 1.0)
    nu_t = KAPPA_VON_KARMAN * u_tau_over_u * inside * (1.0 - inside) ** 2
    # outer layer / free stream: weak residual mixing, kept small but non-zero
    nu_t = np.where(eta <= 1.0, nu_t, 2.0e-4)
    return u_max * delta * nu_t / max(sc_t, 1e-9)


def wall_directed_drift(v_b: float) -> float:
    """Buoyant drift of a bubble under a ship bottom, in the transport frame.

    The hull is the ``y = 0`` ceiling of the transport domain, so buoyancy
    (toward the hull) enters the generic kernel as ``-V_b``.  See the module
    docstring: this sign is the P21-N05 defect this layer corrects.
    """
    if v_b < 0.0:
        raise ValueError("rise velocity magnitude must be non-negative")
    return -float(v_b)


@dataclass
class CurtainResult:
    """Steady curtain of one (air flow, bubble size, curvature) case."""

    q_air: float  # superficial air velocity over the covered hull [m/s]
    d_b: float  # bubble diameter [m]
    eta: np.ndarray  # y/delta of the profile rows
    alpha_profile: np.ndarray
    alpha_wall: float
    delta_b_over_delta: float
    capture_efficiency: float
    conservation_error_max: float
    injected: float
    outflowed: float
    mass: float
    curvature_radius: float  # +inf = flat bottom
    x_inj: float  # injection strip start [m from the leading edge]
    field: Optional[np.ndarray] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "q_air": self.q_air,
            "d_b_um": self.d_b * 1e6,
            "alpha_wall": self.alpha_wall,
            "delta_b_over_delta": self.delta_b_over_delta,
            "capture_efficiency": self.capture_efficiency,
            "conservation_error_max": self.conservation_error_max,
            "curvature_radius_m": self.curvature_radius,
            "x_inj_m": self.x_inj,
            "alpha_profile": [float(v) for v in self.alpha_profile],
            "eta": [float(v) for v in self.eta],
        }


def simulate_curtain(
    q_air: float,
    d_b: float = 5.0e-4,
    *,
    u_ship: float = U_SHIP,
    delta: float = DELTA,
    curvature_radius: float = float("inf"),
    x_inj: float = 0.0,
    inj_length: Optional[float] = None,
    sc_t: float = 1.0,
    ny: int = 48,
    nx: int = 100,
    h_over_delta: float = 2.0,
    l_over_delta: float = 1000.0,
    dt_steps: int = 16000,
    dt: float = 0.25,
    keep_field: bool = False,
) -> CurtainResult:
    """Drift-flux bubble curtain under the ship bottom (nondimensional solve).

    Lengths are in ``delta``, velocities in ``u_ship``.  The hull is the
    ``y = 0`` wall, the free stream sits at ``y = delta``.  Air is injected over
    the strip ``[x_inj, x_inj + inj_length]`` with superficial velocity
    ``q_air``; the steady state balances injection against dispersive mixing and
    streamwise wash-out, and the budget of that balance is *measured*
    (``conservation_error_max``, criterion C3).

    ``curvature_radius = R`` > 0 is a concave-up bottom (trough, centre of
    curvature on the water side): the centrifugal drift of a light bubble points
    away from the hull, i.e. ``+y``.
    """
    if q_air < 0.0:
        raise ValueError("q_air must be non-negative")
    if d_b <= 0.0:
        raise ValueError("d_b must be positive")
    if delta <= 0.0 or u_ship <= 0.0:
        raise ValueError("delta and u_ship must be positive")

    u_max = 1.0
    # cell size in delta units: h_over_delta over ny rows
    dy = h_over_delta / ny
    # The hull is O(1000 delta) long; the domain is truncated at l_over_delta
    # (default 100) with dx = l_over_delta/nx, which is the resolution of the
    # developing bubble layer rather than of the near-wall sublayer.  A domain
    # only a few delta long would leave the curtain in its injection region and
    # bias both delta_b and the energy balance.
    dx = l_over_delta / nx
    y = (np.arange(ny) + 0.5) * dy
    eta = y / 1.0  # y/delta

    u_prof = boundary_layer_velocity(eta, u_max=u_max)
    # slip of the disperse phase is neglected: drift-flux with the mixture velocity
    disp = turbulent_diffusivity(eta, u_max=u_max, sc_t=sc_t)

    v_b = rise_velocity_schiller_naumann(d_b) / u_ship  # nondim magnitude
    v_drift = wall_directed_drift(v_b)

    # hull curvature: kappa = 1/R, drift away from the hull, bounded by the
    # local speed
    curv = np.zeros(ny)
    if np.isfinite(curvature_radius) and curvature_radius > 0.0:
        kappa = 1.0 / curvature_radius
        u_drift = curvature_drift_velocity(u_ship, kappa, d_b) / u_ship
        curv[:] = u_drift

    inj = np.zeros(nx)
    if q_air > 0.0:
        length = dx * nx if inj_length is None else inj_length
        i0 = int(np.clip(round(x_inj / dx), 0, nx - 1))
        i1 = int(np.clip(round((x_inj + length) / dx), i0 + 1, nx))
        inj[i0:i1] = q_air / u_ship * (nx / (i1 - i0))

    tr = VoidFractionTransport2D(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        u_profile=u_prof,
        rise_velocity=v_drift,
        disp_y=disp,
        disp_x=disp,
        injection=inj,
        curvature=curv,
        dt=dt,
    )
    U = np.repeat(u_prof[:, None], nx, axis=1)
    # Species-mass audit.  The per-step *relative* residual of the kernel is
    # meaningless once the curtain is steady (both sides of the budget vanish),
    # so the audit is the absolute per-step closure normalised by the air volume
    # that has entered the domain, plus the cumulative closure at the end.
    cons_max = 0.0
    m_prev, i_prev, o_prev = tr.mass(), 0.0, 0.0
    for _ in range(int(dt_steps)):
        tr.step(U)
        m, i, o = tr.mass(), tr.injected_total, tr.outflow_total
        scale = max(abs(i), abs(o), abs(m), 1.0e-30)
        cons_max = max(cons_max, abs((m - m_prev) - ((i - i_prev) - (o - o_prev))) / scale)
        m_prev, i_prev, o_prev = m, i, o
    cons_cum = abs(tr.mass() - (tr.injected_total - tr.outflow_total)) / max(
        abs(tr.injected_total), 1.0e-30
    )

    # measurement window: the last fifth of the domain, away from the inlet and
    # from the outflow boundary
    x0 = int(0.80 * nx)
    x1 = int(0.95 * nx)
    prof = tr.wall_normal_profile(x0, x1)
    a_wall = float(prof[0])
    delta_b = float(np.sum(prof) * dy / max(a_wall, 1e-30)) if a_wall > 0 else 0.0

    # capture efficiency: fraction of the air carried in the inner half of the
    # boundary layer (y < delta/2) at the measurement window, i.e. air that is
    # still working on the hull instead of having mixed into the outer flow
    win = tr.alpha[:, x0:x1]
    inner = win[eta <= 0.5, :].sum()  # inner half of the boundary layer
    total = win.sum()
    capture = float(inner / total) if total > 0.0 else 0.0

    return CurtainResult(
        q_air=float(q_air),
        d_b=float(d_b),
        eta=eta,
        alpha_profile=prof,
        alpha_wall=a_wall,
        delta_b_over_delta=delta_b / 1.0,
        capture_efficiency=capture,
        conservation_error_max=float(max(cons_max, cons_cum)),
        injected=float(tr.injected_total),
        outflowed=float(tr.outflow_total),
        mass=float(tr.mass()),
        curvature_radius=float(curvature_radius),
        x_inj=float(x_inj),
        field=tr.alpha.copy() if keep_field else None,
    )


# ---------------------------------------------------------------------------
# C1: drag reduction of the computed void fraction
# ---------------------------------------------------------------------------
def curtain_drag_reduction(
    alpha_profile: Sequence[float],
    *,
    h_over_delta: float = 2.0,
    lam: float = LAMBDA_AIR_WATER,
) -> Dict[str, float]:
    """Exact stratified (PAL limiting) ``Delta C_f`` of an ``alpha(y)`` profile.

    ``H = h_over_delta * delta`` in units of ``delta``; the baseline is the
    uniform-viscosity half channel of the same height at the same flow rate, so
    ``Delta C_f/C_f0 = 1 - G/G0`` exactly.
    """
    alpha = np.asarray(alpha_profile, dtype=np.float64)
    out = stratified_drag_reduction_from_alpha(alpha, h_over_delta, lam=lam)
    return {
        "dr": float(out["dr"]),
        "G": float(out["G"]),
        "tau_w": float(out["tau_w"]),
        "tau_w_reference": float(out["tau_w_reference"]),
        "h_air_over_delta": float(np.sum(alpha) * (h_over_delta / alpha.size)),
    }


def pal_formula_comparison(
    alpha_profile: Sequence[float],
    *,
    h_over_delta: float = 2.0,
    delta_b_over_delta: Optional[float] = None,
) -> Dict[str, float]:
    """``Delta C_f`` of the computed profile vs the PAL empirical formula.

    The PAL limiting law is the sandwich-channel drag reduction evaluated on the
    *exponential similarity* void-fraction profile at the wall value and layer
    thickness of the computed curtain (:func:`air_lubrication.pal_drag_reduction`
    at the true air/water viscosity ratio).  The computed profile is the
    drift-flux solution, so the difference measures how much the shape
    assumption of the empirical formula costs -- criterion C1.
    """
    alpha = np.asarray(alpha_profile, dtype=np.float64)
    dy = h_over_delta / alpha.size
    a_wall = float(alpha[0])
    if delta_b_over_delta is None:
        delta_b_over_delta = float(np.sum(alpha) * dy / max(a_wall, 1e-30))
    exact = stratified_drag_reduction_from_alpha(alpha, h_over_delta, lam=LAMBDA_AIR_WATER)["dr"]
    formula = pal_drag_reduction(
        a_wall, float(delta_b_over_delta) * 1.0, h_over_delta, lam=LAMBDA_AIR_WATER
    )["dr"]
    return {
        "dr_exact": float(exact),
        "dr_pal_formula": float(formula),
        "rel_error": float(abs(exact - formula) / exact) if exact > 1e-12 else 0.0,
        "alpha_wall": a_wall,
        "delta_b_over_delta": float(delta_b_over_delta),
    }


# ---------------------------------------------------------------------------
# C4: air-flow optimum and the net energy balance
# ---------------------------------------------------------------------------
def energy_balance(
    q_air: float,
    dr: float,
    *,
    u_ship: float = U_SHIP,
    wetted_area: float = S_WET,
    cf0: float = CF0_SHIP,
    depth: float = INJECTION_DEPTH,
    efficiency: float = ETA_COMPRESSOR,
    coverage: float = 1.0,
) -> Dict[str, float]:
    """Net power of the air-lubrication system [W].

    ``P_saved = dr Cf0 0.5 rho U^3 S coverage`` against the isothermal compressor
    shaft power for the superficial air flow ``q_air`` over the same area.
    ``coverage`` is the fraction of the wetted area the air is delivered to.
    """
    if q_air < 0.0:
        raise ValueError("q_air must be non-negative")
    p_saved = dr * cf0 * 0.5 * RHO * u_ship**3 * wetted_area * coverage
    q_vol = q_air * wetted_area * coverage
    p_air = compressor_power(q_vol, depth=depth, rho_water=RHO, efficiency=efficiency)
    return {
        "power_saved": float(p_saved),
        "power_air": float(p_air),
        "net_power": float(p_saved - p_air),
        "q_air_volume": float(q_vol),
        "ratio": float(p_saved / p_air) if p_air > 0.0 else float("inf"),
    }


def optimize_air_flow(
    q_values: Sequence[float],
    *,
    d_b: float = 5.0e-4,
    sc_t: float = 1.0,
    curvature_radius: float = float("inf"),
    energy: Optional[Dict[str, float]] = None,
    **curtain_kwargs,
) -> Dict[str, object]:
    """Sweep the delivered air flow and locate the net-energy optimum.

    Each ``q_air`` is one :func:`simulate_curtain` run; the resulting ``alpha(y)``
    gives ``Delta C_f`` (exact stratified law) and hence the net power.  The
    maximum of the net power over the sweep is returned together with the whole
    table, and a parabolic refinement around the discrete maximum.
    """
    rows: List[Dict[str, float]] = []
    for q in q_values:
        c = simulate_curtain(
            q, d_b=d_b, sc_t=sc_t, curvature_radius=curvature_radius, **curtain_kwargs
        )
        dr = curtain_drag_reduction(c.alpha_profile)["dr"]
        e = energy_balance(q, dr, **(energy or {}))
        rows.append(
            {
                "q_air": float(q),
                "alpha_wall": c.alpha_wall,
                "delta_b_over_delta": c.delta_b_over_delta,
                "dr": dr,
                "power_saved": e["power_saved"],
                "power_air": e["power_air"],
                "net_power": e["net_power"],
            }
        )
    net = np.array([r["net_power"] for r in rows])
    i = int(np.argmax(net))
    q_opt = float(rows[i]["q_air"])
    # parabolic refinement on the three points around the discrete optimum
    if 0 < i < len(rows) - 1:
        x = np.log([rows[i - 1]["q_air"], rows[i]["q_air"], rows[i + 1]["q_air"]])
        yv = net[i - 1 : i + 2]
        denom = 2.0 * (yv[0] - 2.0 * yv[1] + yv[2])
        if abs(denom) > 0.0:
            shift = float(np.clip((yv[0] - yv[2]) / denom, -0.5, 0.5))
            q_opt = float(np.exp(x[1] + shift * (x[2] - x[1])))
    return {
        "table": rows,
        "q_air_opt": q_opt,
        "net_power_opt": float(net[i]),
        "net_positive": bool(net[i] > 0.0),
        "argmax_q": float(rows[i]["q_air"]),
        "d_b": d_b,
        "sc_t": sc_t,
    }


# ---------------------------------------------------------------------------
# C5: hull curvature and injection position
# ---------------------------------------------------------------------------
def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    return float(np.sum(rx * ry) / den) if den > 0 else 0.0


def curvature_injection_study(
    q_air: float = 2.0e-5,
    d_b: float = 5.0e-4,
    *,
    radii: Sequence[float] = (float("inf"), 50.0, 20.0, 10.0, 5.0),
    injection_positions: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
    inj_length: float = 0.5,
    **curtain_kwargs,
) -> Dict[str, object]:
    """Capture efficiency vs injection position for several hull curvatures.

    ``radii`` are hull-bottom radii of curvature ``R`` [m] (``inf`` = flat);
    ``injection_positions`` are strip starts in ``delta`` units.  Two
    monotonicity statements are tested as acceptance criterion C5:

    * retention must fall monotonically as the bottom becomes more concave-up
      (smaller ``R`` -> stronger drift away from the hull);
    * for a fixed curvature, retention must improve monotonically as the
      injection strip moves closer to the measurement window (shorter residence
      time for the air to mix into the outer flow).
    """
    grid: List[Dict[str, float]] = []
    for r in radii:
        for x0 in injection_positions:
            c = simulate_curtain(
                q_air,
                d_b=d_b,
                curvature_radius=r,
                x_inj=float(x0),
                inj_length=inj_length,
                **curtain_kwargs,
            )
            grid.append(
                {
                    "radius_m": float(r),
                    "x_inj_over_delta": float(x0),
                    "capture": c.capture_efficiency,
                    "alpha_wall": c.alpha_wall,
                    "delta_b_over_delta": c.delta_b_over_delta,
                }
            )
    # per-radius monotonicity in injection position
    per_radius = {}
    for r in radii:
        pts = sorted(
            [g for g in grid if g["radius_m"] == float(r)],
            key=lambda g: g["x_inj_over_delta"],
        )
        caps = [p["capture"] for p in pts]
        per_radius[float(r)] = {
            "captures": caps,
            "rho": _spearman([p["x_inj_over_delta"] for p in pts], caps),
            "strictly_increasing": bool(np.all(np.diff(caps) > 0.0)),
        }
    # per-position monotonicity in curvature (use 1/R as the curvature axis)
    per_position = {}
    for x0 in injection_positions:
        pts = sorted(
            [g for g in grid if g["x_inj_over_delta"] == float(x0)],
            key=lambda g: 0.0 if g["radius_m"] == float("inf") else 1.0 / g["radius_m"],
        )  # flat bottom first (zero curvature), then increasing curvature
        caps = [p["capture"] for p in pts]
        per_position[float(x0)] = {
            "captures": caps,
            "strictly_decreasing": bool(np.all(np.diff(caps) < 0.0)),
        }
    return {
        "grid": grid,
        "per_radius": per_radius,
        "per_position": per_position,
        "q_air": q_air,
        "d_b": d_b,
    }


# ---------------------------------------------------------------------------
# C7: one (Re, alpha_wall) lattice case -- 3 x 5 = 15 cases on the cluster
# ---------------------------------------------------------------------------
RE_SCAN = {"re": (250.0, 500.0, 1000.0), "alpha_wall": (0.0, 0.05, 0.10, 0.20, 0.40)}
LAMBDA_LBM = 0.05  # lattice-realisable air/water viscosity ratio (BGK bound)


def _similarity_alpha(ny: int, alpha_wall: float, delta_b_over_delta: float) -> np.ndarray:
    eta = (np.arange(ny) + 0.5) / ny
    return alpha_wall * np.exp(-eta / delta_b_over_delta)


def re_scan_case(
    re: float,
    alpha_wall: float,
    *,
    u_mean: float = 0.05,
    ny: int = 32,
    nx: int = 8,
    steps: int = 800,
    delta_b_over_delta: float = 0.5,
    lam: float = LAMBDA_LBM,
) -> Dict[str, float]:
    """One lattice case of the Re scan: numerical vs exact ``Delta C_f``.

    ``Re = u_mean * ny / nu`` on the half channel; ``alpha_wall = 0`` is the
    no-air baseline at the same Re (criterion C6 negative control).  ``tau_w``
    follows the discrete momentum balance ``tau_w = rho G H``; ``dr_num`` is the
    drag reduction relative to the baseline at the same Re and ``dr_ref`` the
    exact stratified law for the seeded ``alpha(y)``.
    """
    nu = u_mean * ny / re
    alpha = _similarity_alpha(ny, alpha_wall, delta_b_over_delta)

    def _run(a):
        sim = ChannelAirLubrication2D(
            nx=nx,
            ny=ny,
            u_mean=u_mean,
            nu_liquid=nu,
            lam=lam,
            alpha_profile=a,
            sgs=False,
        )
        st = sim.run(steps)
        return st, sim

    st, sim = _run(alpha if alpha_wall > 0.0 else None)
    st0, sim0 = _run(None)
    dr_num = 1.0 - st.tau_w / st0.tau_w
    dr_ref = stratified_drag_reduction_from_alpha(alpha, float(ny), lam=lam)["dr"]
    return {
        "re": float(re),
        "alpha_wall": float(alpha_wall),
        "nu": float(nu),
        "delta_b_over_delta": float(delta_b_over_delta),
        "lam": float(lam),
        "tau_w": float(st.tau_w),
        "tau_w0": float(st0.tau_w),
        "dr_num": float(dr_num),
        "dr_ref": float(dr_ref),
        "dr_abs_err": float(abs(dr_num - dr_ref)),
        "dr_rel_err": float(abs(dr_num - dr_ref) / dr_ref) if dr_ref > 1e-12 else 0.0,
        "momentum_residual_max": float(st.momentum_residual_max),
        "u_mean_measured": float(st.u_mean),
    }