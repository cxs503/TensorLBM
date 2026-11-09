"""Free-drift momentum balance: ``C_w`` -> Nansen drift angle (criterion C4).

Model
-----
Steady free drift of an ice sheet of thickness ``h_i`` under wind stress only
(ocean velocity zero), in a frame rotating with ``f``:

    rho_a C_a |U_a| U_a  -  rho_w C_w |U_i| U_i  +  rho_i h_i f (k x U_i)  = 0     (1)

Writing ``T = rho_a C_a U_a^2`` (wind stress magnitude), ``U = |U_i|`` and
``B = rho_i h_i f /(rho_w C_w U)``, (1) becomes, after dividing by
``rho_w C_w U^2``,

    A * w_hat - u_hat + B (k x u_hat) = 0,     A = T/(rho_w C_w U^2)              (2)

whose modulus gives ``A = sqrt(1 + B^2)`` -- i.e. a drift speed
``U = sqrt( T / (rho_w C_w sqrt(1+B^2)) )`` -- and whose direction gives the
deflection angle to the right of the wind (Northern Hemisphere)

    tan(alpha) = B = rho_i h_i f / (rho_w C_w U).                              (3)

``alpha`` therefore *decreases* with increasing ``C_w`` and so does the drift
speed (``U^2 = T cos(alpha)/(rho_w C_w)``): a rougher ice underside drifts
**slower and straighter**, because it is coupled more tightly to the water
column.  (Both limits are checked by the tests in
``tests/test_ice_roughness_three_zone.py``.)  Solving (3) together with the
roughness closure of :mod:`tensorlbm.ice_roughness.regime_law` (which makes
``C_w`` a function of the same relative velocity ``U``, implicitly through
``k_s^+``) is a scalar fixed-point problem, solved by
:func:`free_drift_solution`.

Observation
-----------
Nansen's rule for wind-driven drift in the Arctic: the ice moves 20-40 deg to
the right of the wind, canonically ~28 deg (Nansen 1902; see Lepparanta 2011,
*The Drift of Sea Ice*, ch. 4, for the modern restatement and the parameter
ranges).  Criterion C4 asks that the drift angle computed with the
roughness-dependent ``C_w`` be within 10 deg of that observation at a stated
reference condition; the reference condition (wind speed, ice thickness,
Coriolis parameter, air drag coefficient) is therefore part of the result and
:func:`drift_angle_sweep` reports the whole ``alpha(U_a)`` curve plus the
sensitivity to the poorly known inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from . import regime_law as rl

__all__ = [
    "DriftEnvironment",
    "CW_SMOOTH_REFERENCE",
    "NANSEN_DRIFT_ANGLE_DEG",
    "NANSEN_ANGLE_BAND_DEG",
    "free_drift_solution",
    "drift_angle",
    "drift_angle_sweep",
    "backfill_comparison",
]

#: Drag coefficient of a *hydraulically smooth* ice underside referenced to
#: the ice-water relative velocity, i.e. the constant that drift models such as
#: the one behind m5682 hard-code (Hibler 1979 / McPhee 2008 quote 0.0052-0.0058
#: for the logarithmic-drag form).  The roughness closure multiplies it by the
#: LBM-measured drag amplification ``Cf_rough/Cf_smooth``; keeping the two
#: factors separate is what makes the back-fill auditable: the constant stays
#: the *smooth* reference and only the amplification is new physics.
CW_SMOOTH_REFERENCE: float = 5.5e-3

#: Canonical Nansen free-drift deflection to the right of the wind.
NANSEN_DRIFT_ANGLE_DEG: float = 28.0
#: Observed band quoted in the literature (Nansen 1902 / Lepparanta 2011).
NANSEN_ANGLE_BAND_DEG: tuple[float, float] = (20.0, 40.0)


@dataclass(frozen=True)
class DriftEnvironment:
    """Environmental and material constants of the free-drift balance.

    Defaults are the mid-range Arctic values quoted by Lepparanta (2011) ch. 4:
    ``rho_i = 917``, ``h_i = 2 m``, ``rho_a = 1.3``, ``C_a = 1.2e-3`` (form drag
    of a ridged ice cover), ``f = 1.46e-4 s^-1`` (75 N), ``rho_w = 1025``.
    ``wind_speed`` = 5 m/s is the reference condition: the Nansen/Fram drift
    statistics were collected in moderate breeze, and (3) is strongly
    wind-speed dependent, so the value is reported with every angle.
    """

    wind_speed: float = 5.0
    rho_ice: float = 917.0
    ice_thickness: float = 2.0
    rho_air: float = 1.3
    c_air: float = 1.2e-3
    coriolis: float = 1.46e-4
    rho_water: float = rl.RHO_WATER
    nu_water: float = rl.NU_WATER
    z_ref: float = rl.DEFAULT_REFERENCE_HEIGHT_M

    @property
    def wind_stress(self) -> float:
        return self.rho_air * self.c_air * self.wind_speed**2

    def as_dict(self) -> dict[str, float]:
        d = {
            "wind_speed_m_s": self.wind_speed,
            "rho_ice": self.rho_ice,
            "ice_thickness_m": self.ice_thickness,
            "rho_air": self.rho_air,
            "c_air": self.c_air,
            "coriolis_1_s": self.coriolis,
            "rho_water": self.rho_water,
            "nu_water": self.nu_water,
            "z_ref_m": self.z_ref,
            "wind_stress_Pa": self.wind_stress,
        }
        return d


def drift_angle(env: DriftEnvironment, c_w: float) -> dict[str, float]:
    """Solve the free-drift balance for a *given constant* ``C_w``.

    Returns drift speed, deflection angle (deg, positive to the right of the
    wind) and the balance residual.  Closed form: ``tan(alpha) = B`` with
    ``B`` from (3) and ``U`` from ``A = sqrt(1+B^2)``.
    """
    if c_w <= 0.0:
        raise ValueError("C_w must be positive")
    tau_a = env.wind_stress
    # U^2 = T /(rho_w C_w sqrt(1+B^2)),  B = K/U, K = rho_i h f/(rho_w C_w)
    k_b = env.rho_ice * env.ice_thickness * env.coriolis / (env.rho_water * c_w)
    u2 = tau_a / (env.rho_water * c_w)
    # U^4 = u2^2/(1+B^2) -> U^4(1 + K^2/U^2) = u2^2 -> U^4 + K^2 U^2 - u2^2 = 0
    x = 0.5 * (-(k_b**2) + math.sqrt(k_b**4 + 4.0 * u2**2))  # x = U^2
    u_drift = math.sqrt(x)
    b = k_b / u_drift
    angle = math.degrees(math.atan(b))
    # residual of the vector balance (should be ~0)
    lhs = tau_a - env.rho_water * c_w * u_drift**2 * math.sqrt(1.0 + b**2)
    return {
        "C_w": float(c_w),
        "drift_speed_m_s": float(u_drift),
        "drift_angle_deg": float(angle),
        "B": float(b),
        "balance_residual_Pa": float(lhs),
    }


def free_drift_solution(
    env: DriftEnvironment,
    k_s_m: float,
    lambda_over_ks: float,
    *,
    laws: Mapping[float, rl.ThreeZoneLaw] | None = None,
    law: rl.ThreeZoneLaw | None = None,
    tol: float = 1e-9,
    max_iter: int = 100,
) -> dict[str, Any]:
    """Free drift with the roughness-dependent ``C_w(U)`` closure (fixed point).

    Iterates ``U -> k_s^+ -> C_w(k_s^+) -> U``; the map is a contraction
    (``dlnU/dlnC_w = -1/2``, ``dlnC_w/dlnU`` is weak because the closure enters
    through ``ln z^+``), so a damped iteration converges in a few steps.
    """
    env_ok = env  # kept for readability of the iteration
    sol = rl.solve_friction_velocity(0.06, k_s_m, law or rl.interpolate_law(lambda_over_ks, laws),  # type: ignore[arg-type]
                                     z_ref_m=env_ok.z_ref, nu=env_ok.nu_water)
    u_guess = max(sol["u_tau"] * 12.0, 0.01)
    history = []
    for _ in range(max_iter):
        cw = rl.solve_friction_velocity(u_guess, k_s_m,
                                        law or rl.interpolate_law(lambda_over_ks, laws),  # type: ignore[arg-type]
                                        z_ref_m=env.z_ref, nu=env.nu_water)["C_w"]
        bal = drift_angle(env, cw)
        u_new = bal["drift_speed_m_s"]
        history.append({"U": u_guess, "C_w": cw, "angle_deg": bal["drift_angle_deg"]})
        if abs(u_new - u_guess) <= tol * max(u_new, 1e-30):
            u_guess = u_new
            break
        u_guess = 0.5 * (u_guess + u_new)
    closure = rl.solve_friction_velocity(u_guess, k_s_m,
                                         law or rl.interpolate_law(lambda_over_ks, laws),  # type: ignore[arg-type]
                                         z_ref_m=env.z_ref, nu=env.nu_water)
    bal = drift_angle(env, closure["C_w"])
    out = dict(closure)
    out.update(bal)
    out.update({
        "lambda_over_ks": float(lambda_over_ks),
        "k_s_m": float(k_s_m),
        "environment": env.as_dict(),
        "nansen_angle_deg": NANSEN_DRIFT_ANGLE_DEG,
        "nansen_band_deg": list(NANSEN_ANGLE_BAND_DEG),
        "angle_error_deg": float(abs(bal["drift_angle_deg"] - NANSEN_DRIFT_ANGLE_DEG)),
        "iteration_history": history,
    })
    return out


def drift_with_roughness_amplification(
    env: DriftEnvironment,
    k_s_m: float,
    law,
    *,
    cw_smooth: float = CW_SMOOTH_REFERENCE,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> dict[str, Any]:
    """Back-fill ``C_w = C_w,smooth * Cf_rough/Cf_smooth(k_s^+, lambda/k_s)``.

    This is the form the drift model actually consumes: the constant is kept as
    the smooth reference and the LBM supplies the *amplification*
    ``A(k_s^+) = Cf_rough/Cf_smooth`` (Clauser, matched ``u_tau``), which is
    independent of any ocean-boundary-layer reference height.  ``k_s^+`` is
    implicit in the drift speed (``u_tau = sqrt(C_w) U``), so the three
    relations are iterated:

        U <- free-drift balance with C_w
        u_tau = sqrt(C_w) U,   k_s^+ = k_s u_tau / nu
        C_w <- C_w,smooth * A(k_s^+)
    """
    u_drift = 0.1
    cw = float(cw_smooth)
    history = []
    for _ in range(max_iter):
        bal = drift_angle(env, cw)
        u_drift = bal["drift_speed_m_s"]
        u_tau = math.sqrt(cw) * u_drift
        ks_plus = float(k_s_m) * u_tau / env.nu_water
        amp = law.cw_amplification(ks_plus, env.z_ref * u_tau / env.nu_water)
        cw_new = float(cw_smooth) * amp
        history.append({"U": u_drift, "C_w": cw, "ks_plus": ks_plus,
                        "amplification": amp, "angle_deg": bal["drift_angle_deg"]})
        if abs(cw_new - cw) <= tol * max(cw_new, 1e-30):
            cw = cw_new
            break
        cw = 0.5 * (cw + cw_new)
    bal = drift_angle(env, cw)
    u_tau = math.sqrt(cw) * bal["drift_speed_m_s"]
    ks_plus = float(k_s_m) * u_tau / env.nu_water
    return {
        **bal,
        "C_w_smooth": float(cw_smooth),
        "amplification": float(cw / cw_smooth),
        "ks_plus": float(ks_plus),
        "u_tau": float(u_tau),
        "dU_plus": float(law.delta_u_plus(ks_plus)),
        "regime": law.regime_of(ks_plus),
        "k_s_m": float(k_s_m),
        "lambda_over_ks": float(law.lambda_over_ks),
        "z_ref_m": float(env.z_ref),
        "environment": env.as_dict(),
        "nansen_angle_deg": NANSEN_DRIFT_ANGLE_DEG,
        "angle_error_deg": float(abs(bal["drift_angle_deg"] - NANSEN_DRIFT_ANGLE_DEG)),
        "iteration_history": history,
    }


def drift_angle_sweep(
    env: DriftEnvironment,
    k_s_m: float,
    lambda_over_ks: float,
    *,
    laws: Mapping[float, rl.ThreeZoneLaw] | None = None,
    wind_speeds: Iterable[float] = (2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0),
    constant_cw: float | None = None,
) -> list[dict[str, Any]]:
    """``alpha(U_a)`` for the closure and (optionally) a constant-``C_w`` model."""
    rows = []
    for ua in wind_speeds:
        e = DriftEnvironment(**{**env.__dict__, "wind_speed": float(ua)})
        row: dict[str, Any] = {"wind_speed_m_s": float(ua)}
        sol = free_drift_solution(e, k_s_m, lambda_over_ks, laws=laws)
        row.update({"closure_C_w": sol["C_w"], "closure_angle_deg": sol["drift_angle_deg"],
                    "closure_drift_m_s": sol["drift_speed_m_s"], "closure_ks_plus": sol["ks_plus"],
                    "closure_regime": sol["regime"]})
        if constant_cw is not None:
            b = drift_angle(e, float(constant_cw))
            row.update({"constant_C_w": float(constant_cw),
                        "constant_angle_deg": b["drift_angle_deg"],
                        "constant_drift_m_s": b["drift_speed_m_s"],
                        "angle_delta_deg": sol["drift_angle_deg"] - b["drift_angle_deg"]})
        rows.append(row)
    return rows


def backfill_comparison(
    env: DriftEnvironment,
    k_s_m: float,
    lambda_over_ks: float,
    *,
    laws: Mapping[float, rl.ThreeZoneLaw] | None = None,
    constant_cw: float = 5.5e-3,
) -> dict[str, Any]:
    """The mission's back-fill: constant ``C_w`` (m5682 style) vs the closure.

    Returns both drift solutions at the reference condition and the two angle
    errors against the Nansen observation, so criterion C4 can be adjudicated
    on the *closure* while showing what the constant did.
    """
    closure = free_drift_solution(env, k_s_m, lambda_over_ks, laws=laws)
    const = drift_angle(env, float(constant_cw))
    return {
        "environment": env.as_dict(),
        "constant_C_w": float(constant_cw),
        "constant": {**const, "angle_error_deg": abs(const["drift_angle_deg"] - NANSEN_DRIFT_ANGLE_DEG)},
        "closure": {k: v for k, v in closure.items() if k != "iteration_history"},
        "angle_improvement_deg": abs(const["drift_angle_deg"] - NANSEN_DRIFT_ANGLE_DEG)
                                 - abs(closure["drift_angle_deg"] - NANSEN_DRIFT_ANGLE_DEG),
        "nansen_angle_deg": NANSEN_DRIFT_ANGLE_DEG,
        "nansen_band_deg": list(NANSEN_ANGLE_BAND_DEG),
    }