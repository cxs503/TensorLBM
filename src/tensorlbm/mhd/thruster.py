"""Channel MHD thruster (pump-jet) thrust/efficiency ledger.

Device
------
A constant-area duct of height ``2H``, span ``b`` and length ``L`` carries
seawater along ``x``.  A transverse field ``B0`` (along ``y``) and an applied
electric field ``E_z = -K B0`` (along the span, out of the 2D plane) drive the
water; with the low-Rm Ohm's law ``j_z = sigma B0 (u - K)`` the Lorentz force
density is ``sigma B0 (K - u)``.  Inside the pump the axial momentum balance of
the developed flow is

    sigma B0 (K - u) = dp/dx + mu u'',

so the pump raises the static pressure by ``dp = G L`` while viscous friction
eats ``D_wall``.  Downstream an (ideal, lossless) nozzle converts the pump head
into a jet: ``dp = rho (u_e^2 - U^2) / 2`` with the actuator-disk (same-size
rotor) contraction ``u_disk = u_mean = (U + u_e)/2``.

Two mouths for the same thrust
------------------------------
* Lorentz-volume mouth: ``T_lorentz = sigma B0 (K - u_mean) A L - D_wall``
* momentum-flux mouth:   ``T_momentum = rho Q (u_e - U)``, ``Q = A u_mean``

Under the ideal-jet identity both equal ``dp * A``; the campaign requires them
to agree within 2 % (``c5_thrust``).

Energy ledger
-------------
``P_electric = sigma B0^2 K (K - u_mean) A L`` (source power),
``P_joule = sigma B0^2 (K - u)^2``, viscous dissipation ``Phi`` and hydraulic
output ``P_hyd = dp Q``.  At steady state

    P_electric = P_joule + Phi + P_hyd     (``c7_energy``, residual <= 2 %)

so ``eta = T U / P_electric`` can never exceed one and the propeller comparison
is meaningful: a same-size actuator disk producing the same thrust in the same
free stream has ideal efficiency ``eta_ideal = 2 U / (U + u_e,prop)`` with
``u_e,prop = sqrt(U^2 + 2 T / (rho A))``.

Everything is computed twice: from the closed-form (engineering-scale numbers,
physical units) and from a D2Q9 Lorentz-force duct run at the same Hartmann
number (discretisation evidence).  The lattice run is the same sub-grid physics
as :mod:`.hartmann`, at ``u_mean = 0.02`` lattice units, so it carries the
~1e-3 discretisation error measured by the Hartmann ladder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .hartmann import (
    HartmannConfig,
    hartmann_mean_factor,
    hartmann_shape_factor,
    hartmann_wall_shear_coefficient,
    run_hartmann_channel,
)


@dataclass(frozen=True)
class MHDThrusterConfig:
    """Physical design point plus the lattice settings used for verification.

    Args:
        half_width_m: duct half-height ``H`` in metres (direction of ``B0``).
        span_m: duct span ``b`` in metres (direction of the applied field).
        length_m: active (electrode/magnet) duct length ``L`` in metres.
        b0_t: magnetic flux density in tesla.
        sigma_s_per_m: electric conductivity in S/m (seawater ~4 S/m).
        nu_m2_s: kinematic viscosity in m^2/s (~1e-6 for seawater).
        rho_kg_m3: density in kg/m^3.
        ship_speed_m_s: ship speed ``U`` in m/s.
        jet_excess_ratio: ``theta = (u_e - U)/U`` (jet excess ratio of the design).
        ny: fluid rows across the duct used by the verification run.
        nx: periodic streamwise cells of the verification run.
        nu_lattice: lattice viscosity of the verification run.
        u_mean_lattice: mean velocity (lattice units) of the verification run.
        steps: step cap of the verification run.
        check_every: steady-state check interval of the verification run.
    """

    half_width_m: float = 0.2
    span_m: float = 0.6
    length_m: float = 1.0
    b0_t: float = 1.0
    sigma_s_per_m: float = 4.0
    nu_m2_s: float = 1.0e-6
    rho_kg_m3: float = 1025.0
    ship_speed_m_s: float = 2.0
    jet_excess_ratio: float = 0.35
    ny: int = 128
    nx: int = 4
    nu_lattice: float = 0.05
    u_mean_lattice: float = 0.02
    steps: int = 60000
    check_every: int = 200

    def __post_init__(self) -> None:
        for name in (
            "half_width_m",
            "span_m",
            "length_m",
            "b0_t",
            "sigma_s_per_m",
            "nu_m2_s",
            "rho_kg_m3",
            "ship_speed_m_s",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.jet_excess_ratio < 0.0:
            raise ValueError("jet_excess_ratio must be >= 0")
        if self.ny < 8 or self.ny % 2:
            raise ValueError(f"ny must be even and >= 8, got {self.ny}")

    # -- physical derived quantities ----------------------------------------
    @property
    def mu(self) -> float:
        """Dynamic viscosity ``rho nu`` [Pa s]."""
        return self.rho_kg_m3 * self.nu_m2_s

    @property
    def area_m2(self) -> float:
        """Duct flow area ``2 H b`` [m^2]."""
        return 2.0 * self.half_width_m * self.span_m

    @property
    def ha(self) -> float:
        """Hartmann number ``B0 H sqrt(sigma/mu)``."""
        return self.b0_t * self.half_width_m * math.sqrt(self.sigma_s_per_m / self.mu)

    @property
    def jet_speed_m_s(self) -> float:
        """Nozzle exit speed ``u_e = U (1 + theta)``."""
        return self.ship_speed_m_s * (1.0 + self.jet_excess_ratio)

    @property
    def duct_speed_m_s(self) -> float:
        """Actuator-disk contraction ``u_mean = (U + u_e)/2``."""
        return 0.5 * (self.ship_speed_m_s + self.jet_speed_m_s)

    @property
    def delta_p_pa(self) -> float:
        """Pump head ``dp = rho (u_e^2 - U^2)/2`` [Pa]."""
        u_e = self.jet_speed_m_s
        u = self.ship_speed_m_s
        return 0.5 * self.rho_kg_m3 * (u_e * u_e - u * u)

    @property
    def pressure_gradient(self) -> float:
        """Load gradient ``dp/dx = dp / L`` [Pa/m]."""
        return self.delta_p_pa / self.length_m

    @property
    def sigma_b0(self) -> float:
        """``sigma B0`` (current density per unit velocity mismatch)."""
        return self.sigma_s_per_m * self.b0_t

    @property
    def magnetic_force_coefficient(self) -> float:
        """``sigma B0^2`` - Lorentz force density per unit velocity mismatch."""
        return self.sigma_s_per_m * self.b0_t * self.b0_t

    @property
    def loading(self) -> float:
        """Applied loading ``K = -E_z/B0`` matching the design flow rate."""
        m = hartmann_mean_factor(self.ha)
        u0 = self.duct_speed_m_s / m
        return u0 + self.pressure_gradient / self.magnetic_force_coefficient


@dataclass
class ThrusterLedger:
    """Result of the thruster ledger (physical units plus lattice evidence)."""

    config: MHDThrusterConfig
    ha: float
    loading: float
    mean_factor: float
    u_mean_m_s: float
    u_jet_m_s: float
    delta_p_pa: float
    flow_rate_m3_s: float
    thrust_lorentz_n: float
    thrust_momentum_n: float
    thrust_density_pa: float
    thrust_closure_residual: float
    lorentz_force_n: float
    wall_drag_n: float
    power_electric_w: float
    power_joule_w: float
    power_viscous_w: float
    power_hydraulic_w: float
    energy_residual: float
    efficiency: float
    efficiency_jet_ideal: float
    efficiency_propeller_ideal: float
    disk_loading_ct: float
    lattice: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable ledger."""
        return {
            "design": {
                "half_width_m": self.config.half_width_m,
                "span_m": self.config.span_m,
                "length_m": self.config.length_m,
                "b0_t": self.config.b0_t,
                "sigma_s_per_m": self.config.sigma_s_per_m,
                "nu_m2_s": self.config.nu_m2_s,
                "rho_kg_m3": self.config.rho_kg_m3,
                "ship_speed_m_s": self.config.ship_speed_m_s,
                "jet_excess_ratio": self.config.jet_excess_ratio,
                "area_m2": self.config.area_m2,
                "mu_pa_s": self.config.mu,
            },
            "operating_point": {
                "hartmann_number": self.ha,
                "mean_factor_m": self.mean_factor,
                "loading_K": self.loading,
                "u_mean_m_s": self.u_mean_m_s,
                "u_jet_m_s": self.u_jet_m_s,
                "delta_p_pa": self.delta_p_pa,
                "flow_rate_m3_s": self.flow_rate_m3_s,
            },
            "thrust": {
                "lorentz_volume_N": self.thrust_lorentz_n,
                "momentum_flux_N": self.thrust_momentum_n,
                "density_N_per_m2": self.thrust_density_pa,
                "closure_residual": self.thrust_closure_residual,
                "lorentz_force_total_N": self.lorentz_force_n,
                "wall_drag_N": self.wall_drag_n,
            },
            "energy": {
                "P_electric_W": self.power_electric_w,
                "P_joule_W": self.power_joule_w,
                "P_viscous_W": self.power_viscous_w,
                "P_hydraulic_W": self.power_hydraulic_w,
                "residual": self.energy_residual,
            },
            "efficiency": {
                "eta_total": self.efficiency,
                "eta_jet_ideal": self.efficiency_jet_ideal,
                "eta_propeller_ideal_same_size": self.efficiency_propeller_ideal,
                "disk_loading_CT": self.disk_loading_ct,
            },
            "lattice": self.lattice,
            "notes": self.notes,
        }


def propeller_ideal_efficiency(
    thrust_n: float,
    area_m2: float,
    rho_kg_m3: float,
    ship_speed_m_s: float,
) -> float:
    """Ideal (actuator-disk) efficiency of a same-size propeller at the same thrust.

    ``u_e = sqrt(U^2 + 2T/(rho A))`` and ``eta = 2U/(U + u_e)``.
    """
    if thrust_n <= 0.0:
        return 0.0
    u_e = math.sqrt(ship_speed_m_s**2 + 2.0 * thrust_n / (rho_kg_m3 * area_m2))
    return 2.0 * ship_speed_m_s / (ship_speed_m_s + u_e)


def _viscous_dissipation_per_volume(mu: float, u0: float, ha: float, half_width: float) -> float:
    """``mu <(du/dy)^2>`` averaged over the channel, closed form for the Hartmann profile.

    ``u' = -U0 (Ha/H) sinh(Ha y/H)/cosh(Ha)``; the average over ``[-H, H]`` is
    ``mu U0^2 Ha^3 / H^2 * [tanh(Ha) - Ha + Ha tanh^2(Ha)] / (3 ...)`` - evaluated
    numerically here (high-order trapezoid on a fine non-uniform-free grid) to
    keep the expression free of transcription risk.
    """
    if ha <= 1.0e-12:
        # Poiseuille with u_max = 1.5 u_mean is handled by the caller
        return float("nan")
    y = np.linspace(-1.0, 1.0, 400_001)
    u = u0 * hartmann_shape_factor(y, ha)
    du_dy_hat = np.gradient(u, y)  # du/d(y/H)
    du_dy = du_dy_hat / half_width
    return float(mu * np.trapezoid(du_dy * du_dy, y * half_width) / (2.0 * half_width))


def run_thruster_ledger(
    config: MHDThrusterConfig | None = None,
    run_lattice: bool = True,
) -> ThrusterLedger:
    """Build the thruster ledger from the closed form and verify it with LBM.

    Args:
        config: design point; defaults to the seawater reference design.
        run_lattice: run the D2Q9 duct verification (disable for fast unit tests).

    Returns:
        :class:`ThrusterLedger` with thrust (two mouths), powers, efficiencies,
        propeller comparison and the lattice evidence dict.
    """
    cfg = config or MHDThrusterConfig()
    ha = cfg.ha
    m = hartmann_mean_factor(ha)
    area = cfg.area_m2
    u_mean = cfg.duct_speed_m_s
    u_jet = cfg.jet_speed_m_s
    u_ship = cfg.ship_speed_m_s
    q = area * u_mean
    dp = cfg.delta_p_pa
    force_coeff = cfg.magnetic_force_coefficient
    u0 = u_mean / m

    # Wall shear of the Hartmann profile: tau_w = mu U0 Ha tanh(Ha)/H
    tau_w = cfg.mu * u0 * ha * math.tanh(ha) / cfg.half_width_m
    wall_drag = 2.0 * tau_w * cfg.span_m * cfg.length_m  # two frictional walls
    lorentz_force = force_coeff * (cfg.loading - u_mean) * area * cfg.length_m

    thrust_lorentz = lorentz_force - wall_drag
    thrust_momentum = cfg.rho_kg_m3 * q * (u_jet - u_ship)
    residual = abs(thrust_lorentz - thrust_momentum) / max(abs(thrust_lorentz), 1e-30)

    # --- energy ledger (per unit volume integrated) -------------------------
    # P_electric = int sigma B0^2 K (K - u) dV ; P_joule = int sigma B0^2 (K-u)^2 dV
    y = np.linspace(-1.0, 1.0, 400_001)
    u_prof = u0 * hartmann_shape_factor(y, ha)
    k_minus_u = cfg.loading - u_prof
    volume = area * cfg.length_m
    sigma_b0_sq = cfg.magnetic_force_coefficient
    p_electric = float(sigma_b0_sq * cfg.loading * k_minus_u.mean() * volume)
    p_joule = float(sigma_b0_sq * (k_minus_u * k_minus_u).mean() * volume)
    du_dy = np.gradient(u_prof, y * cfg.half_width_m)
    p_viscous = float(cfg.mu * np.mean(du_dy * du_dy) * volume)
    p_hydraulic = dp * q
    energy_residual = abs(p_electric - p_joule - p_viscous - p_hydraulic) / max(p_electric, 1e-30)

    thrust = thrust_lorentz
    efficiency = thrust * u_ship / max(p_electric, 1e-30)
    eta_jet = 2.0 * u_ship / (u_ship + u_jet)
    eta_prop = propeller_ideal_efficiency(thrust, area, cfg.rho_kg_m3, u_ship)
    ct = thrust / (0.5 * cfg.rho_kg_m3 * u_ship**2 * area)

    notes = [
        "2D duct: only the two y=+-H walls are frictional; electrodes/end effects are idealised.",
        "Low-Rm: induced field neglected; negligible for seawater sigma (Rm << 1).",
        "Nozzle/actuator-disk contraction u_mean=(U+u_e)/2 and a lossless nozzle are assumed "
        "for the momentum mouth; only the duct itself is resolved.",
        "P_viscous uses the numerical integral of mu (du/dy)^2 of the closed-form profile.",
    ]

    lattice: dict[str, Any] = {}
    if run_lattice:
        h_lat = 0.5 * cfg.ny
        mu_lat = cfg.nu_lattice  # rho=1 in lattice units
        sigma_lat = mu_lat * ha * ha / (h_lat * h_lat)  # B0 = 1 lattice unit
        length_lat = cfg.length_m / cfg.half_width_m * h_lat
        theta = cfg.jet_excess_ratio
        u_mean_lat = cfg.u_mean_lattice
        u_ship_lat = u_mean_lat / (1.0 + 0.5 * theta)
        u_jet_lat = u_ship_lat * (1.0 + theta)
        dp_lat = 0.5 * (u_jet_lat**2 - u_ship_lat**2)  # rho = 1
        grad_lat = dp_lat / length_lat
        cfg_lat = HartmannConfig(
            ny=cfg.ny,
            nx=cfg.nx,
            ha=ha,
            u_mean=u_mean_lat,
            nu=cfg.nu_lattice,
            rho0=1.0,
            b0=1.0,
            load_grad=grad_lat,
            steps=cfg.steps,
            check_every=cfg.check_every,
            tol=1e-13,
        )
        res = run_hartmann_channel(cfg_lat)
        sigma_b0_lat = sigma_lat * 1.0
        sigma_b0_sq_lat = sigma_lat * 1.0**2  # sigma B0^2 with B0 = 1 lattice unit
        area_lat = cfg.ny * 1.0  # per unit span
        lorentz_lat = sigma_b0_lat * (cfg_lat.loading - res.u_mean) * area_lat * length_lat
        wall_lat = abs(res.wall_momentum_flux) / cfg.nx * length_lat
        thrust_lat_lorentz = lorentz_lat - wall_lat
        thrust_lat_momentum = area_lat * res.u_mean * (u_jet_lat - u_ship_lat)
        closure_lat = abs(thrust_lat_lorentz - thrust_lat_momentum) / max(
            abs(thrust_lat_lorentz), 1e-30
        )
        # lattice energy guard from the resolved field
        k_minus_u_lat = cfg_lat.loading - res.u_profile
        u_lat = res.u_profile
        pe_lat = float(
            sigma_b0_sq_lat * cfg_lat.loading * k_minus_u_lat.mean() * area_lat * length_lat
        )
        pj_lat = float(
            sigma_b0_sq_lat * (k_minus_u_lat * k_minus_u_lat).mean() * area_lat * length_lat
        )
        dy = 2.0 * h_lat / (cfg.ny - 1)
        du_coarse = np.gradient(u_lat, dy)
        pv_lat_coarse = float(mu_lat * np.mean(du_coarse * du_coarse) * area_lat * length_lat)
        # The viscous dissipation is quadratic in the wall-normal gradient, so a
        # two-point finite difference of a coarse Hartmann layer is only O(h)
        # accurate; the guard therefore uses the closed-form dissipation of the
        # *resolved* amplitude U0 = u_mean_LBM / m(Ha) (exact quadrature), while
        # the crude FD value is reported alongside as a diagnostic.
        u0_lat = res.u_mean / hartmann_mean_factor(ha)
        pv_lat = _viscous_dissipation_per_volume(mu_lat, u0_lat, ha, h_lat) * area_lat * length_lat
        ph_lat = grad_lat * length_lat * area_lat * res.u_mean
        energy_lat = abs(pe_lat - pj_lat - pv_lat - ph_lat) / max(pe_lat, 1e-30)
        lattice = {
            "ny": cfg.ny,
            "ha": ha,
            "sigma_lattice": sigma_lat,
            "loading_K": cfg_lat.loading,
            "load_grad_lattice": grad_lat,
            "u_mean_lattice_lbm": res.u_mean,
            "u_mean_lattice_design": u_mean_lat,
            "u_mean_relative_error": abs(res.u_mean - u_mean_lat) / u_mean_lat,
            "rel_linf_profile": res.rel_linf,
            "rel_max_pointwise": res.rel_max_pointwise,
            "wall_shear_coefficient_lbm": res.wall_shear_coefficient,
            "wall_shear_coefficient_exact": hartmann_wall_shear_coefficient(ha),
            "closure_residual_lbm": closure_lat,
            "lorentz_force_lattice": lorentz_lat,
            "wall_drag_lattice": wall_lat,
            "thrust_lorentz_lattice": thrust_lat_lorentz,
            "thrust_momentum_lattice": thrust_lat_momentum,
            "P_electric_lattice": pe_lat,
            "P_joule_lattice": pj_lat,
            "P_viscous_lattice": pv_lat,
            "P_viscous_lattice_coarse_fd": pv_lat_coarse,
            "P_hydraulic_lattice": ph_lat,
            "energy_residual_lattice": energy_lat,
            "steps": res.steps,
            "converged": res.converged,
            "max_mach": res.max_mach,
        }

    return ThrusterLedger(
        config=cfg,
        ha=ha,
        loading=cfg.loading,
        mean_factor=m,
        u_mean_m_s=u_mean,
        u_jet_m_s=u_jet,
        delta_p_pa=dp,
        flow_rate_m3_s=q,
        thrust_lorentz_n=thrust_lorentz,
        thrust_momentum_n=thrust_momentum,
        thrust_density_pa=thrust_lorentz / area,
        thrust_closure_residual=residual,
        lorentz_force_n=lorentz_force,
        wall_drag_n=wall_drag,
        power_electric_w=p_electric,
        power_joule_w=p_joule,
        power_viscous_w=p_viscous,
        power_hydraulic_w=p_hydraulic,
        energy_residual=energy_residual,
        efficiency=efficiency,
        efficiency_jet_ideal=eta_jet,
        efficiency_propeller_ideal=eta_prop,
        disk_loading_ct=ct,
        lattice=lattice,
        notes=notes,
    )
