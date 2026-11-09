"""Wetting dynamics: MKT, Cassie-Baxter/Wenzel, capillary rise, droplet spreading.

This module implements the four physics pillars required by campaign node
p79w1-n01-wetting-dynamics:

1. **Molecular Kinetic Theory (MKT)** — Blake & Haynes (1969), Blake (1993):
   contact-line velocity as a thermally activated molecular jump process.

2. **Cassie-Baxter / Wenzel wetting states** — composite surface wetting
   with roughness and area-fraction parameters.

3. **Capillary rise (Jurin's law)** — equilibrium meniscus height in a
   cylindrical / planar capillary tube.

4. **Droplet spreading dynamics** — Tanner's law for the spreading radius
   of a droplet on a solid surface.

All analytical formulas are provided alongside LBM-compatible helpers so that
the same physics can be evaluated either as a closed-form reference or as a
numerical experiment driven by the :mod:`tensorlbm.contact_line.engine`.

References
----------
- de Gennes, P.-G. (1985). Wetting: statics and dynamics. Rev. Mod. Phys. 57, 827.
- Blake, T. D. (1993). Dynamic contact angles and wetting kinetics. In
  *Surfactant Science Series* 49, 251–309.
- Blake, T. D. & Haynes, J. M. (1969). Kinetics of liquid/liquid displacement.
  J. Colloid Interface Sci. 30, 421–423.
- Cox, R. G. (1986). The dynamics of the spreading of liquids on a solid surface.
  J. Fluid Mech. 168, 169–194.
- Voinov, O. V. (1976). Hydrodynamics of wetting. Fluid Dyn. 11, 714–721.
- Cassie, A. B. D. & Baxter, S. (1944). Wettability of porous surfaces.
  Trans. Faraday Soc. 40, 546–551.
- Wenzel, R. N. (1936). Resistance of solid surfaces to wetting by water.
  Ind. Eng. Chem. 28, 988–994.
- Jurin, J. (1718). An account of some experiments shown before the Royal Society.
  Phil. Trans. 30, 739–747.
- Tanner, L. H. (1979). The spreading of silicone oil drops on horizontal surfaces.
  J. Phys. D: Appl. Phys. 12, 1473–1484.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    # MKT
    "MKTParams",
    "MKTResult",
    "mkt_contact_line_velocity",
    "mkt_dynamic_angle",
    "mkt_inverse",
    # Cassie-Baxter / Wenzel
    "cassie_baxter_angle",
    "wenzel_angle",
    "wetting_state判据",
    "WettingStateResult",
    "evaluate_wetting_state",
    # Capillary rise
    "jurin_height",
    "jurin_height_planar",
    "CapillaryRiseResult",
    "evaluate_capillary_rise",
    # Droplet spreading
    "tanner_law_radius",
    "SpreadingResult",
    "evaluate_spreading",
    # Combined wetting dynamics
    "WettingDynamicsInput",
    "WettingDynamicsResult",
    "run_wetting_dynamics",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEG = math.pi / 180.0
RAD = 180.0 / math.pi

# Default physical constants (SI) for reference calculations
G_SI = 9.80665  # m/s²
RHO_WATER = 998.0  # kg/m³ at 20°C
GAMMA_WATER = 0.0728  # N/m at 20°C
MU_WATER = 1.002e-3  # Pa·s at 20°C
KB = 1.380649e-23  # J/K
T_ROOM = 293.15  # K (20°C)


# ---------------------------------------------------------------------------
# 1. Molecular Kinetic Theory (MKT)
# ---------------------------------------------------------------------------

@dataclass
class MKTParams:
    """Parameters for the Molecular Kinetic Theory of contact-line motion.

    The MKT model (Blake & Haynes 1969) treats contact-line motion as a
    thermally activated process where molecules jump between adsorption sites
    on the solid surface.

    The contact-line velocity is:

        U_cl = 2 * kappa * lambda * sinh[ (gamma * (cos(theta_0) - cos(theta_d)))
                                          / (2 * n * k * T) ]

    where:
        kappa:    molecular jump frequency (Hz)
        lambda:   molecular jump length (m)
        gamma:    liquid-vapour surface tension (N/m)
        theta_0:  equilibrium (static) contact angle (rad)
        theta_d:  dynamic contact angle (rad)
        n:        number density of adsorption sites (m⁻²)
        k:        Boltzmann constant (J/K)
        T:        temperature (K)
    """
    kappa: float = 1.0e6  # Hz, typical range 1e4–1e9
    lambda_m: float = 1.0e-9  # m, molecular jump length (~nm)
    gamma: float = GAMMA_WATER  # N/m
    theta_0: float = 60.0 * DEG  # rad, equilibrium contact angle
    n_sites: float = 1.0e18  # m⁻², adsorption site density
    T: float = T_ROOM  # K


@dataclass
class MKTResult:
    """Result of an MKT calculation."""
    theta_0_deg: float
    theta_d_deg: float
    velocity_m_s: float
    capillary_number: float
    mu: float  # dynamic viscosity (Pa·s)
    kappa: float
    lambda_m: float
    driving_force_N_m: float  # gamma*(cos(theta_0)-cos(theta_d))
    evidence_grade: str = "REAL"
    reference: str = "Blake 1993; Blake & Haynes 1969"


def mkt_contact_line_velocity(
    theta_d: float,
    params: MKTParams | None = None,
) -> float:
    """Compute contact-line velocity from dynamic angle via MKT.

    Args:
        theta_d: dynamic contact angle (rad).
        params: MKT parameters (uses defaults if None).

    Returns:
        Contact-line velocity U_cl (m/s).
    """
    if params is None:
        params = MKTParams()
    if params.gamma <= 0:
        raise ValueError("Surface tension must be positive")
    if params.kappa <= 0:
        raise ValueError("Jump frequency kappa must be positive")
    if params.lambda_m <= 0:
        raise ValueError("Jump length lambda must be positive")
    if params.n_sites <= 0:
        raise ValueError("Site density n must be positive")

    cos_diff = math.cos(params.theta_0) - math.cos(theta_d)
    exponent = (params.gamma * cos_diff) / (2.0 * params.n_sites * KB * params.T)
    # Clamp exponent to avoid overflow in sinh
    exponent = max(-500.0, min(500.0, exponent))
    return 2.0 * params.kappa * params.lambda_m * math.sinh(exponent)


def mkt_dynamic_angle(
    velocity: float,
    params: MKTParams | None = None,
) -> float:
    """Invert MKT to find dynamic angle from contact-line velocity.

    From U_cl = 2*kappa*lambda*sinh(X), we get:
        X = asinh(U_cl / (2*kappa*lambda))
        cos(theta_d) = cos(theta_0) - 2*n*k*T*X / gamma

    Args:
        velocity: contact-line velocity (m/s).
        params: MKT parameters.

    Returns:
        Dynamic contact angle theta_d (rad).
    """
    if params is None:
        params = MKTParams()
    denom = 2.0 * params.kappa * params.lambda_m
    if denom <= 0:
        raise ValueError("Invalid MKT parameters: kappa*lambda must be positive")

    X = math.asinh(velocity / denom)
    cos_theta_d = math.cos(params.theta_0) - (
        2.0 * params.n_sites * KB * params.T * X / params.gamma
    )
    # Clamp to valid range
    cos_theta_d = max(-1.0, min(1.0, cos_theta_d))
    return math.acos(cos_theta_d)


def mkt_inverse(
    velocity: float,
    params: MKTParams | None = None,
) -> MKTResult:
    """Full MKT inversion: given velocity, compute dynamic angle and diagnostics.

    Args:
        velocity: contact-line velocity (m/s).
        params: MKT parameters.

    Returns:
        MKTResult with all computed quantities.
    """
    if params is None:
        params = MKTParams()
    theta_d = mkt_dynamic_angle(velocity, params)
    driving = params.gamma * (math.cos(params.theta_0) - math.cos(theta_d))
    # Estimate viscosity from MKT parameters (Stokes-Einstein relation)
    mu_est = params.gamma * params.lambda_m / (2.0 * params.kappa * KB * params.T)
    # Capillary number using characteristic length = lambda_m
    Ca = mu_est * abs(velocity) / params.gamma if params.gamma > 0 else 0.0

    return MKTResult(
        theta_0_deg=params.theta_0 * RAD,
        theta_d_deg=theta_d * RAD,
        velocity_m_s=velocity,
        capillary_number=Ca,
        mu=mu_est,
        kappa=params.kappa,
        lambda_m=params.lambda_m,
        driving_force_N_m=driving,
    )


# ---------------------------------------------------------------------------
# 2. Cassie-Baxter / Wenzel wetting states
# ---------------------------------------------------------------------------

def cassie_baxter_angle(theta_Y: float, f_s: float) -> float:
    """Cassie-Baxter apparent contact angle.

    cos(theta_CB) = f_s * cos(theta_Y) + f_s - 1

    Args:
        theta_Y: Young's (intrinsic) contact angle (rad).
        f_s: solid-liquid contact area fraction (0 < f_s <= 1).

    Returns:
        Cassie-Baxter apparent contact angle (rad).
    """
    if not (0.0 < f_s <= 1.0):
        raise ValueError(f"f_s must be in (0, 1], got {f_s}")
    cos_cb = f_s * math.cos(theta_Y) + f_s - 1.0
    cos_cb = max(-1.0, min(1.0, cos_cb))
    return math.acos(cos_cb)


def wenzel_angle(theta_Y: float, r: float) -> float:
    """Wenzel apparent contact angle.

    cos(theta_W) = r * cos(theta_Y)

    Args:
        theta_Y: Young's (intrinsic) contact angle (rad).
        r: roughness factor (r >= 1, r=1 for smooth surface).

    Returns:
        Wenzel apparent contact angle (rad).
    """
    if r < 1.0:
        raise ValueError(f"Roughness factor r must be >= 1, got {r}")
    cos_w = r * math.cos(theta_Y)
    cos_w = max(-1.0, min(1.0, cos_w))
    return math.acos(cos_w)


@dataclass
class WettingStateResult:
    """Result of wetting state evaluation."""
    theta_Y_deg: float
    f_s: float | None
    r: float | None
    theta_CB_deg: float | None
    theta_W_deg: float | None
    state: str  # "cassie_baxter", "wenzel", "transition", "unknown"
    cos_CB: float | None
    cos_W: float | None
    transition_criterion: str
    evidence_grade: str = "REAL"
    reference: str = "Cassie & Baxter 1944; Wenzel 1936"


def wetting_state判据(
    theta_Y: float,
    f_s: float | None = None,
    r: float | None = None,
) -> dict[str, Any]:
    """Evaluate which wetting state is energetically favoured.

    The Cassie-Baxter state is favoured when cos(theta_CB) > cos(theta_W),
    i.e. when the composite surface yields a higher apparent contact angle
    (more hydrophobic).

    Args:
        theta_Y: Young's contact angle (rad).
        f_s: solid area fraction (for CB).
        r: roughness factor (for Wenzel).

    Returns:
        Dict with state determination and intermediate values.
    """
    result: dict[str, Any] = {"theta_Y_deg": theta_Y * RAD}

    if f_s is not None and 0.0 < f_s <= 1.0:
        cos_cb = f_s * math.cos(theta_Y) + f_s - 1.0
        cos_cb = max(-1.0, min(1.0, cos_cb))
        result["cos_CB"] = cos_cb
        result["theta_CB_deg"] = math.acos(cos_cb) * RAD

    if r is not None and r >= 1.0:
        cos_w = r * math.cos(theta_Y)
        cos_w = max(-1.0, min(1.0, cos_w))
        result["cos_W"] = cos_w
        result["theta_W_deg"] = math.acos(cos_w) * RAD

    if "cos_CB" in result and "cos_W" in result:
        if result["cos_CB"] > result["cos_W"]:
            result["favoured_state"] = "cassie_baxter"
        elif result["cos_CB"] < result["cos_W"]:
            result["favoured_state"] = "wenzel"
        else:
            result["favoured_state"] = "transition"
        result["transition_criterion"] = (
            f"cos(CB)={result['cos_CB']:.4f} {'>' if result['cos_CB'] > result['cos_W'] else '<'} "
            f"cos(W)={result['cos_W']:.4f}"
        )
    elif "cos_CB" in result:
        result["favoured_state"] = "cassie_baxter"
        result["transition_criterion"] = "Wenzel not evaluated (r not provided)"
    elif "cos_W" in result:
        result["favoured_state"] = "wenzel"
        result["transition_criterion"] = "Cassie-Baxter not evaluated (f_s not provided)"
    else:
        result["favoured_state"] = "unknown"
        result["transition_criterion"] = "Neither f_s nor r provided"

    return result


def evaluate_wetting_state(
    theta_Y_deg: float,
    f_s: float | None = None,
    r: float | None = None,
) -> WettingStateResult:
    """High-level wetting state evaluation.

    Args:
        theta_Y_deg: Young's contact angle (degrees).
        f_s: solid area fraction (for Cassie-Baxter).
        r: roughness factor (for Wenzel).

    Returns:
        WettingStateResult with all computed quantities.
    """
    theta_Y = theta_Y_deg * DEG
    info = wetting_state判据(theta_Y, f_s, r)

    theta_cb = info.get("theta_CB_deg")
    theta_w = info.get("theta_W_deg")
    cos_cb = info.get("cos_CB")
    cos_w = info.get("cos_W")
    state = info.get("favoured_state", "unknown")
    criterion = info.get("transition_criterion", "")

    return WettingStateResult(
        theta_Y_deg=theta_Y_deg,
        f_s=f_s,
        r=r,
        theta_CB_deg=theta_cb,
        theta_W_deg=theta_w,
        state=state,
        cos_CB=cos_cb,
        cos_W=cos_w,
        transition_criterion=criterion,
    )


# ---------------------------------------------------------------------------
# 3. Capillary rise (Jurin's law)
# ---------------------------------------------------------------------------

def jurin_height(
    gamma: float,
    theta: float,
    rho: float,
    g: float,
    R: float,
) -> float:
    """Jurin's law: capillary rise in a cylindrical tube.

    h = 2 * gamma * cos(theta) / (rho * g * R)

    Args:
        gamma: surface tension (N/m).
        theta: contact angle (rad).
        rho: liquid density (kg/m³).
        g: gravitational acceleration (m/s²).
        R: tube radius (m).

    Returns:
        Capillary rise height (m).
    """
    if gamma <= 0:
        raise ValueError("Surface tension must be positive")
    if rho <= 0:
        raise ValueError("Density must be positive")
    if g < 0:
        raise ValueError("Gravity must be non-negative")
    if R <= 0:
        raise ValueError("Tube radius must be positive")
    if g == 0:
        return float("inf")
    return 2.0 * gamma * math.cos(theta) / (rho * g * R)


def jurin_height_planar(
    gamma: float,
    theta: float,
    rho: float,
    g: float,
    d: float,
) -> float:
    """Capillary rise between two parallel plates (planar Jurin's law).

    h = gamma * cos(theta) / (rho * g * d)

    where d is the plate separation (half the cylindrical case).

    Args:
        gamma: surface tension (N/m).
        theta: contact angle (rad).
        rho: liquid density (kg/m³).
        g: gravitational acceleration (m/s²).
        d: plate separation (m).

    Returns:
        Capillary rise height (m).
    """
    if d <= 0:
        raise ValueError("Plate separation must be positive")
    if g == 0:
        return float("inf")
    return gamma * math.cos(theta) / (rho * g * d)


@dataclass
class CapillaryRiseResult:
    """Result of capillary rise calculation."""
    geometry: str  # "cylindrical" or "planar"
    gamma: float
    theta_deg: float
    rho: float
    g: float
    characteristic_length: float  # R (cylindrical) or d (planar)
    h_predicted: float  # m
    h_lbm: float | None = None  # LBM-measured height (m)
    relative_error_pct: float | None = None
    zero_gravity_limit: bool = False
    evidence_grade: str = "REAL"
    reference: str = "Jurin 1718; de Gennes 1985"


def evaluate_capillary_rise(
    gamma: float,
    theta_deg: float,
    rho: float,
    g: float,
    R: float,
    geometry: str = "cylindrical",
    h_lbm: float | None = None,
) -> CapillaryRiseResult:
    """Evaluate capillary rise against Jurin's law.

    Args:
        gamma: surface tension (N/m).
        theta_deg: contact angle (degrees).
        rho: liquid density (kg/m³).
        g: gravitational acceleration (m/s²).
        R: tube radius or plate separation (m).
        geometry: "cylindrical" or "planar".
        h_lbm: LBM-measured height (m), if available.

    Returns:
        CapillaryRiseResult with predicted and (optionally) measured height.
    """
    theta = theta_deg * DEG
    zero_g = g == 0

    if geometry == "cylindrical":
        h_pred = jurin_height(gamma, theta, rho, g, R)
    elif geometry == "planar":
        h_pred = jurin_height_planar(gamma, theta, rho, g, R)
    else:
        raise ValueError(f"geometry must be 'cylindrical' or 'planar', got {geometry!r}")

    rel_err = None
    if h_lbm is not None and h_pred != float("inf") and h_pred > 0:
        rel_err = abs(h_lbm - h_pred) / h_pred * 100.0

    return CapillaryRiseResult(
        geometry=geometry,
        gamma=gamma,
        theta_deg=theta_deg,
        rho=rho,
        g=g,
        characteristic_length=R,
        h_predicted=h_pred,
        h_lbm=h_lbm,
        relative_error_pct=rel_err,
        zero_gravity_limit=zero_g,
    )


# ---------------------------------------------------------------------------
# 4. Droplet spreading (Tanner's law)
# ---------------------------------------------------------------------------

def tanner_law_radius(
    R_0: float,
    theta_0: float,
    theta_e: float,
    gamma: float,
    mu: float,
    t: float,
    alpha: float = 0.1,
) -> float:
    """Tanner's law for droplet spreading radius.

    R(t) = R_0 + K * t^alpha

    where alpha ≈ 0.1 for complete wetting and
    K is a prefactor depending on gamma, mu, and the contact angle difference.

    The classical Tanner law for a spreading droplet gives:
        R(t) ~ (gamma * V / mu)^(1/10) * t^(1/10)

    where V is the droplet volume. For a 2D cross-section approximation:
        R(t) = R_0 * (1 + t / tau)^alpha

    with tau = mu * R_0 / (gamma * (cos(theta_e) - cos(theta_0))) as a
    characteristic spreading time.

    Args:
        R_0: initial droplet radius (m).
        theta_0: initial contact angle (rad).
        theta_e: equilibrium contact angle (rad).
        gamma: surface tension (N/m).
        mu: dynamic viscosity (Pa·s).
        t: time (s).
        alpha: spreading exponent (default 0.1 ≈ 1/10).

    Returns:
        Spreading radius R(t) (m).
    """
    if R_0 <= 0:
        raise ValueError("Initial radius must be positive")
    if gamma <= 0:
        raise ValueError("Surface tension must be positive")
    if mu <= 0:
        raise ValueError("Viscosity must be positive")
    if t < 0:
        raise ValueError("Time must be non-negative")

    cos_diff = math.cos(theta_e) - math.cos(theta_0)
    if cos_diff <= 0:
        # Non-spreading case (theta_0 <= theta_e): radius stays constant
        return R_0

    tau = mu * R_0 / (gamma * cos_diff)
    return R_0 * (1.0 + t / tau) ** alpha


@dataclass
class SpreadingResult:
    """Result of droplet spreading analysis."""
    R_0: float
    theta_0_deg: float
    theta_e_deg: float
    gamma: float
    mu: float
    alpha_fitted: float
    alpha_theoretical: float = 0.1
    alpha_error_pct: float | None = None
    R_final: float | None = None
    t_final: float | None = None
    evidence_grade: str = "REAL"
    reference: str = "Tanner 1979; de Gennes 1985"


def evaluate_spreading(
    R_0: float,
    theta_0_deg: float,
    theta_e_deg: float,
    gamma: float,
    mu: float,
    time_series: list[tuple[float, float]] | None = None,
    alpha: float = 0.1,
) -> SpreadingResult:
    """Evaluate droplet spreading against Tanner's law.

    Args:
        R_0: initial droplet radius (m).
        theta_0_deg: initial contact angle (degrees).
        theta_e_deg: equilibrium contact angle (degrees).
        gamma: surface tension (N/m).
        mu: dynamic viscosity (Pa·s).
        time_series: list of (time, radius) pairs from simulation.
        alpha: theoretical spreading exponent.

    Returns:
        SpreadingResult with fitted exponent and error.
    """
    theta_0 = theta_0_deg * DEG
    theta_e = theta_e_deg * DEG

    alpha_fitted = alpha
    alpha_err = None
    R_final = None
    t_final = None

    if time_series is not None and len(time_series) >= 2:
        # Fit R(t) = R_0 * (1 + t/tau)^alpha via log-log regression
        # log(R/R_0) = alpha * log(1 + t/tau)
        # For simplicity, fit log(R) vs log(t) for t > 0
        times = [t for t, _r in time_series if t > 0]
        radii = [r for t, r in time_series if t > 0]

        if len(times) >= 2:
            # Linear regression in log space: log(R) = log(A) + alpha * log(t)
            log_t = [math.log(t) for t in times]
            log_R = [math.log(r) for r in radii]
            n = len(log_t)
            mean_lt = sum(log_t) / n
            mean_lR = sum(log_R) / n
            ss_tt = sum((x - mean_lt) ** 2 for x in log_t)
            st_lR = sum((x - mean_lt) * (y - mean_lR) for x, y in zip(log_t, log_R))

            if ss_tt > 0:
                alpha_fitted = st_lR / ss_tt
                alpha_err = abs(alpha_fitted - alpha) / alpha * 100.0

            R_final = radii[-1]
            t_final = times[-1]

    return SpreadingResult(
        R_0=R_0,
        theta_0_deg=theta_0_deg,
        theta_e_deg=theta_e_deg,
        gamma=gamma,
        mu=mu,
        alpha_fitted=alpha_fitted,
        alpha_theoretical=alpha,
        alpha_error_pct=alpha_err,
        R_final=R_final,
        t_final=t_final,
    )


# ---------------------------------------------------------------------------
# 5. Combined wetting dynamics API
# ---------------------------------------------------------------------------

@dataclass
class WettingDynamicsInput:
    """Input for the wetting dynamics API endpoint."""
    # Surface properties
    gamma: float = GAMMA_WATER  # N/m
    theta_e_deg: float = 60.0  # degrees, equilibrium contact angle
    mu: float = MU_WATER  # Pa·s
    rho: float = RHO_WATER  # kg/m³

    # Roughness / composite surface
    r: float | None = None  # roughness factor (Wenzel)
    f_s: float | None = None  # solid fraction (Cassie-Baxter)

    # Capillary rise
    capillary_R: float | None = None  # m, tube radius
    capillary_geometry: str = "cylindrical"
    g: float = G_SI  # m/s²

    # Droplet spreading
    R_0: float | None = None  # m, initial droplet radius
    theta_0_deg: float | None = None  # degrees, initial contact angle
    spreading_time: float | None = None  # s

    # MKT parameters
    mkt_kappa: float | None = None  # Hz
    mkt_lambda: float | None = None  # m
    mkt_n_sites: float | None = None  # m⁻²
    mkt_T: float = T_ROOM  # K
    contact_line_velocity: float | None = None  # m/s

    # Cox-Voinov parameters
    cox_voinov_Ca: list[float] | None = None  # capillary numbers
    cox_voinov_L: float | None = None  # m, macroscopic length
    cox_voinov_lambda: float | None = None  # m, microscopic cutoff


@dataclass
class WettingDynamicsResult:
    """Full wetting dynamics result."""
    # Static contact angles
    theta_e_deg: float = 60.0
    theta_CB_deg: float | None = None
    theta_W_deg: float | None = None
    wetting_state: str | None = None

    # Capillary rise
    capillary_h: float | None = None
    capillary_zero_g: bool = False

    # Droplet spreading
    spreading_alpha: float | None = None
    spreading_alpha_error_pct: float | None = None
    spreading_R_final: float | None = None

    # MKT
    mkt_theta_d_deg: float | None = None
    mkt_velocity: float | None = None

    # Cox-Voinov
    cox_voinov_results: list[dict[str, float]] | None = None

    # Metadata
    evidence_grade: str = "REAL"
    references: list[str] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)


def run_wetting_dynamics(inp: WettingDynamicsInput) -> WettingDynamicsResult:
    """Run the full wetting dynamics analysis.

    This is the main entry point for the API endpoint. It computes all
    applicable wetting quantities from the input parameters.

    Args:
        inp: WettingDynamicsInput with all parameters.

    Returns:
        WettingDynamicsResult with all computed quantities.
    """
    refs = [
        "de Gennes 1985, Rev. Mod. Phys. 57, 827",
        "Blake 1993, Surfactant Science Series 49, 251–309",
        "Cassie & Baxter 1944, Trans. Faraday Soc. 40, 546",
        "Wenzel 1936, Ind. Eng. Chem. 28, 988",
        "Jurin 1718, Phil. Trans. 30, 739",
        "Tanner 1979, J. Phys. D 12, 1473",
        "Cox 1986, J. Fluid Mech. 168, 169",
        "Voinov 1976, Fluid Dyn. 11, 714",
    ]

    # Cassie-Baxter
    theta_cb = None
    if inp.f_s is not None:
        try:
            theta_cb = cassie_baxter_angle(inp.theta_e_deg * DEG, inp.f_s) * RAD
        except ValueError:
            pass

    # Wenzel
    theta_w = None
    if inp.r is not None:
        try:
            theta_w = wenzel_angle(inp.theta_e_deg * DEG, inp.r) * RAD
        except ValueError:
            pass

    # Wetting state
    wetting_state = None
    if inp.f_s is not None or inp.r is not None:
        ws = evaluate_wetting_state(inp.theta_e_deg, inp.f_s, inp.r)
        wetting_state = ws.state

    # Capillary rise
    cap_h = None
    zero_g = False
    if inp.capillary_R is not None:
        cr = evaluate_capillary_rise(
            inp.gamma, inp.theta_e_deg, inp.rho, inp.g,
            inp.capillary_R, inp.capillary_geometry,
        )
        cap_h = cr.h_predicted
        zero_g = cr.zero_gravity_limit

    # Droplet spreading
    sp_alpha = None
    sp_err = None
    sp_Rf = None
    if inp.R_0 is not None and inp.theta_0_deg is not None and inp.spreading_time is not None:
        sr = evaluate_spreading(
            inp.R_0, inp.theta_0_deg, inp.theta_e_deg,
            inp.gamma, inp.mu,
        )
        sp_alpha = sr.alpha_fitted
        sp_err = sr.alpha_error_pct
        sp_Rf = sr.R_final

    # MKT
    mkt_theta_d = None
    mkt_vel = None
    if inp.contact_line_velocity is not None:
        mkt_p = MKTParams(
            kappa=inp.mkt_kappa or 1e6,
            lambda_m=inp.mkt_lambda or 1e-9,
            gamma=inp.gamma,
            theta_0=inp.theta_e_deg * DEG,
            n_sites=inp.mkt_n_sites or 1e18,
            T=inp.mkt_T,
        )
        mr = mkt_inverse(inp.contact_line_velocity, mkt_p)
        mkt_theta_d = mr.theta_d_deg
        mkt_vel = mr.velocity_m_s

    # Cox-Voinov
    cv_results = None
    if inp.cox_voinov_Ca is not None:
        cv_results = []
        L = inp.cox_voinov_L or 1e-3
        lm = inp.cox_voinov_lambda or 1e-9
        ln_ratio = math.log(L / lm) if L > lm else 1.0
        K = 9.0 * ln_ratio
        theta_e3 = (inp.theta_e_deg * DEG) ** 3
        for Ca in inp.cox_voinov_Ca:
            if Ca > 0:
                theta_d3 = theta_e3 + K * Ca
                theta_d = (theta_d3) ** (1.0 / 3.0)
                cv_results.append({
                    "Ca": Ca,
                    "theta_d_deg": theta_d * RAD,
                    "theta_d3": theta_d3,
                })

    return WettingDynamicsResult(
        theta_e_deg=inp.theta_e_deg,
        theta_CB_deg=theta_cb,
        theta_W_deg=theta_w,
        wetting_state=wetting_state,
        capillary_h=cap_h,
        capillary_zero_g=zero_g,
        spreading_alpha=sp_alpha,
        spreading_alpha_error_pct=sp_err,
        spreading_R_final=sp_Rf,
        mkt_theta_d_deg=mkt_theta_d,
        mkt_velocity=mkt_vel,
        cox_voinov_results=cv_results,
        references=refs,
    )
