"""Pancake-ice formation dynamics and ship resistance in a pancake field.

Physical background
-------------------
In a freezing, wave-agitated ocean the first ice is *frazil*: millimetre-scale
disc crystals suspended in supercooled water.  Under a wave field the crystals
collide and flocculate, and once the floes can no longer be dispersed by the
wave orbital motion they consolidate into centimetre-to-metre **pancake ice**:
circular discs of diameter D ~ 0.3-3 m, thickness h ~ 0.05-0.3 m, with a
raised, consolidated rim ridge of h_rim ~ 0.02-0.05 m (Lange et al. 1989;
Martin & Kauffman 1981; Wadhams 2000).  A pancake field is *not* a continuous
ice sheet: ships do not crush it, they slide through and push it aside, so the
resistance mechanisms differ from level-ice or brash-ice resistance.

This module provides the three ingredients needed to close a pancake-ice
resistance model (no lattice units, no ship geometry):

1. **Frazil -> pancake coalescence criterion.**  Orthokinetic Smoluchowski and
   Saffman-Turner kernels give the collision *frequency* of the suspended
   crystals; the sticking efficiency of a collision follows a fragile-aggregate
   ``exp(-(u_rel/u_c)^2)`` law.  The step that converts a floc into a
   *pancake* is retention against the wave orbital motion: a floe smaller than
   the local orbital semi-excursion ``A_orb`` is flushed clean by the wave
   before the collected slush consolidates, while a floe larger than ``A_orb``
   retains it and grows an emergent rim (wave pumping / rim filtration).  At
   the storm-wave steepness ``H/lambda = 0.1`` the threshold is
   ``A_orb = H/2 = lambda/20`` -- exactly the Lange et al. pancake-size rule.

2. **Pancake geometry and buoyancy.**  Disc + rim-ridge parameterisation with
   the measured aspect-ratio window ``D/h in [5, 20]``, Archimedean
   draft/freeboard including the rim mass, and the analytic added mass of a
   thin disc (Lamb 1932: ``m_am = (8/3) rho_w a^3`` face-on), which for
   ``D/h = 10`` is ~4.7x the pancake's own mass.  That is why laterally
   displacing pancakes is expensive.

3. **Resistance model** (:func:`pancake_resistance`): two dissipation channels
   weighted by a jamming/clogging split.

   * *slip* -- pancakes are deflected sideways and dragged along the hull.
     The force is the momentum flux imparted to them, including the exact
     added mass per unit plan area ``(4/3pi) rho_w D``, plus the steady
     face-on drag of the pancake (``cd_pancake``, supplied by the LBM drag
     table of ``tools/run_pancake_resistance.py``);
   * *bulldozing* -- at high concentration the pancakes raft and form a
     contact network that cannot be displaced sideways, so a raft is pushed
     ahead of the bow: form drag ``0.5 rho_w C_d A_front V^2`` on the raft plus
     the work needed to lift the ice over the accumulating pile.

   The split is a jamming function ``f_slip = 1/(1 + (C/C_jam)^p)`` (sideways
   displacement needs free space proportional to ``1-C``) and rafting thickens
   the field: ``h_eff = h (1 + k_raft (C/C_jam)^2)``.

The model returns an *added* ice resistance: ``R(C=0) = 0`` identically, so an
open-water resistance curve can be superposed without double counting.

Evidence status
---------------
* REAL -- collision kernels, sticking law, disc added mass, Archimedean draft,
  power-law fitting and the resistance algebra (closed form, unit checked,
  negative control ``C=0 => R=0``).
* REAL (numerical) -- ``cd_pancake`` / ``cd_raft`` are produced by the D2Q9
  bounce-back kernel of this package (``tools/run_pancake_resistance.py``),
  whose drag monitor is validated on the circular-cylinder benchmark
  ``C_d(Re=100) = 1.34``.
* WITHHELD -- the retention sharpness ``s`` and the jamming parameters are
  phenomenological.  They are declared a priori (``DEFAULT_PARAMS``) and
  carried into the report with a sensitivity study.  No closed-form pancake
  field resistance experiment exists, so absolute resistance values are
  order-of-magnitude only.

References
----------
* Lange, M.A., Ackley, S.F., Wadhams, P., Dieckmann, G.S., Eicken, H. (1989)
  Development of sea ice in the Weddell Sea, Ann. Glaciol. 12, 92-96.
* Martin, S. & Kauffman, P. (1981) A field and laboratory study of wave
  damping by grease ice, J. Glaciol. 27, 283-313.
* Saffman, P.G. & Turner, J.S. (1956) On the collision of drops in turbulent
  clouds, J. Fluid Mech. 1, 16-30.
* Smoluchowski, M. (1917) Versuch einer mathematischen Theorie der
  Koagulationskinetik, Z. Phys. Chem. 92, 129.
* Lamb, H. (1932) Hydrodynamics, 6th ed., section 108 (disc added mass).
* Wadhams, P. (2000) Ice in the Ocean, Gordon & Breach.
* Herman, A. (2013) Numerical modelling of wave-induced floe breakup,
  Ocean Model. 69, 1-11.
* Timco, G.W. & Weeks, W.F. (2010) A review of the engineering properties of
  sea ice, Cold Reg. Sci. Technol. 60, 107-129.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CD_BLUNT_PLATE_BAND",
    "DEFAULT_PARAMS",
    "G",
    "NU_WATER",
    "PancakeGeometry",
    "PancakeIceError",
    "RHO_ICE",
    "RHO_WATER",
    "SIGMA_FLEX_BAND",
    "WaveField",
    "added_mass_coefficient_disc",
    "aggregate_growth_rate",
    "bulldozing_resistance",
    "disc_added_mass",
    "frazil_collision_kernel",
    "frazil_number_density",
    "cluster_diameter",
    "jamming_split",
    "orbital_retention_threshold",
    "pancake_bending_stress",
    "pancake_buoyancy",
    "pancake_field_mask",
    "pancake_formation_probability",
    "pancake_geometry",
    "pancake_resistance",
    "pancake_section_mask",
    "power_law_fit",
    "raft_thickness",
    "retention_probability",
    "simulate_aggregate_growth",
    "slip_resistance",
    "sticking_efficiency",
    "thermodynamic_formation_time",
    "breakup_limited_diameter",
    "LATENT_HEAT_FUSION",
    "DEFAULT_HEAT_FLUX",
    "stokes_settling_velocity",
    "sweep_resistance",
]

# ---------------------------------------------------------------------------
# physical constants (SI, sea water at the freezing point)
# ---------------------------------------------------------------------------
RHO_ICE: float = 917.0  # kg m^-3
RHO_WATER: float = 1025.0  # kg m^-3
G: float = 9.81  # m s^-2
NU_WATER: float = 1.79e-6  # m^2 s^-1, sea water at about -1.8 C

#: flexural strength of granular / pancake ice [Pa] (Timco & Weeks 2010 give
#: 0.1-1.0 MPa for granular ice; the mid band is used below).
SIGMA_FLEX_BAND: tuple[float, float] = (0.1e6, 1.0e6)

#: drag-coefficient band for blunt (thick plate / raft) sections in cross flow;
#: used only as a comparison anchor for the LBM-computed values.
CD_BLUNT_PLATE_BAND: tuple[float, float] = (1.6, 2.4)

#: A-priori declared resistance parameters.  These are *not* tuned against the
#: acceptance numbers; they are carried into the sensitivity study.
DEFAULT_PARAMS: dict[str, float] = {
    "k_slip": 0.35,  # fraction of the ship speed imparted to a deflected pancake
    "c_jam": 0.65,  # jamming / clogging concentration of a rafting pancake field
    "p_jam": 3.0,  # sharpness of the jamming transition
    "k_raft": 0.5,  # rafting thickness amplification at C = C_jam
    "k_cluster": 0.0,  # cluster hydraulic-size growth (declared 0; see sensitivity)
    "pile_factor": 0.5,  # extra thickness of the bulldozed pile at the bow
    "rise_factor": 0.5,  # mean lift height of the pile, in units of h_pile
    "beam_eff": 20.0,  # effective width of the swept pancake corridor [m]
    "cd_pancake": 2.0,  # face-on drag coefficient of one pancake (LBM plateau)
    "cd_raft": 2.0,  # drag coefficient of the bulldozed raft (LBM plateau)
}


class PancakeIceError(ValueError):
    """Raised when the model is asked for a state outside its validity range."""


# ---------------------------------------------------------------------------
# wave field
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WaveField:
    """Linear (Airy) wave field used to set the orbital motion.

    Parameters
    ----------
    wavelength:
        lambda [m].
    amplitude:
        a = H/2 [m], the orbital semi-excursion at the surface.
    depth:
        water depth [m]; ``math.inf`` (default) means deep-water dispersion
        ``omega^2 = g k``.
    """

    wavelength: float
    amplitude: float
    depth: float = math.inf

    def __post_init__(self) -> None:
        if not (self.wavelength > 0.0):
            raise PancakeIceError("wavelength must be > 0")
        if not (self.amplitude > 0.0):
            raise PancakeIceError("amplitude must be > 0")
        if not (self.amplitude <= 0.5 * self.wavelength):
            raise PancakeIceError("amplitude outside the Stokes range a <= lambda/2")

    @property
    def wavenumber(self) -> float:
        return 2.0 * math.pi / self.wavelength

    @property
    def omega(self) -> float:
        k = self.wavenumber
        if math.isinf(self.depth):
            return math.sqrt(G * k)
        return math.sqrt(G * k * math.tanh(k * self.depth))

    @property
    def period(self) -> float:
        return 2.0 * math.pi / self.omega

    @property
    def steepness(self) -> float:
        return 2.0 * self.amplitude / self.wavelength

    def _depth_decay(self, z: float) -> float:
        """Orbital decay exp(k z) (z <= 0 below the mean waterline)."""
        if z > 0.0:
            raise PancakeIceError("z must be <= 0 (below the mean waterline)")
        if math.isinf(self.depth):
            return math.exp(self.wavenumber * z)
        return math.cosh(self.wavenumber * (self.depth + z)) / math.sinh(
            self.wavenumber * self.depth
        )

    def orbital_velocity(self, z: float = 0.0) -> float:
        """Orbital velocity amplitude u_orb = a omega exp(kz) [m/s]."""
        return self.amplitude * self.omega * self._depth_decay(z)

    def orbital_excursion(self, z: float = 0.0) -> float:
        """Orbital semi-excursion A_orb = a exp(kz) [m]."""
        return self.amplitude * self._depth_decay(z)

    def strain_rate(self, z: float = 0.0) -> float:
        """Orbital strain-rate amplitude S = a omega k exp(kz) [1/s]."""
        return self.amplitude * self.omega * self.wavenumber * self._depth_decay(z)

    def at_steepness(self, steepness: float) -> "WaveField":
        """Same wavelength, amplitude rescaled to the given steepness 2a/lambda."""
        if steepness <= 0.0:
            raise PancakeIceError("steepness must be > 0")
        return WaveField(self.wavelength, 0.5 * steepness * self.wavelength, self.depth)

    def to_dict(self) -> dict[str, float]:
        return {
            "wavelength_m": self.wavelength,
            "amplitude_m": self.amplitude,
            "height_m": 2.0 * self.amplitude,
            "steepness": self.steepness,
            "period_s": self.period,
            "wavenumber_1_m": self.wavenumber,
            "orbital_velocity_surface_m_s": self.orbital_velocity(0.0),
            "orbital_excursion_surface_m": self.orbital_excursion(0.0),
            "strain_rate_surface_1_s": self.strain_rate(0.0),
        }# ---------------------------------------------------------------------------
# 1. frazil aggregation
# ---------------------------------------------------------------------------
def stokes_settling_velocity(
    diameter: float,
    *,
    rho_ice: float = RHO_ICE,
    rho_water: float = RHO_WATER,
    nu_water: float = NU_WATER,
) -> float:
    """Stokes rise velocity of an ice crystal [m/s] (positive = upward).

    ``w = (rho_w - rho_i) g d^2 / (18 mu_w)``.  Valid for ``Re_d < 1``, i.e.
    crystal diameters of a few mm -- the frazil range.
    """
    if diameter <= 0.0:
        raise PancakeIceError("diameter must be > 0")
    mu = rho_water * nu_water
    return (rho_water - rho_ice) * G * diameter**2 / (18.0 * mu)


def frazil_number_density(diameter: float, ice_volume_fraction: float) -> float:
    """Number density [m^-3] of disc crystals of the given diameter."""
    if diameter <= 0.0:
        raise PancakeIceError("diameter must be > 0")
    if ice_volume_fraction < 0.0:
        raise PancakeIceError("ice_volume_fraction must be >= 0")
    vol = math.pi * diameter**3 / 6.0
    return ice_volume_fraction / vol


def frazil_collision_kernel(
    d_i: float,
    d_j: float | None = None,
    *,
    wave: WaveField | None = None,
    z: float = 0.0,
    dissipation: float = 0.0,
    settling_spread: float = 0.3,
    nu_water: float = NU_WATER,
) -> dict[str, float]:
    """Bimolecular collision kernel beta_ij [m^3 s^-1].

    Three additive mechanisms, all standard coagulation kernels:

    * wave orthokinetic (Smoluchowski shear kernel with the orbital strain rate
      S): ``beta = (4/3) S (a_i + a_j)^3``;
    * turbulent (Saffman & Turner 1956):
      ``beta = 1.294 (eps/nu)^0.5 (a_i + a_j)^3``;
    * differential settling: ``beta = (pi/4) (a_i + a_j)^2 |w_i - w_j|``, the
      settling spread being represented by a size ratio ``1 + settling_spread``.

    The returned ``relative_velocity`` is the kernel divided by the collision
    cross-section, i.e. the mean relative approach speed used by
    :func:`sticking_efficiency`.
    """
    if d_i <= 0.0 or (d_j is not None and d_j <= 0.0):
        raise PancakeIceError("crystal diameters must be > 0")
    dj = d_i if d_j is None else d_j
    a = 0.5 * (d_i + dj)  # r_i + r_j

    beta_wave = 0.0
    if wave is not None:
        beta_wave = (4.0 / 3.0) * wave.strain_rate(z) * a**3

    beta_turb = 0.0
    if dissipation > 0.0:
        beta_turb = 1.294 * math.sqrt(dissipation / nu_water) * a**3

    w_i = stokes_settling_velocity(d_i, nu_water=nu_water)
    w_j = stokes_settling_velocity(dj * (1.0 + settling_spread), nu_water=nu_water)
    beta_settle = math.pi * a * a * abs(w_i - w_j)

    total = beta_wave + beta_turb + beta_settle
    cross = math.pi * a * a
    return {
        "beta_wave": beta_wave,
        "beta_turbulent": beta_turb,
        "beta_settling": beta_settle,
        "beta_total": total,
        "collision_cross_section": cross,
        "relative_velocity": total / cross if cross > 0.0 else 0.0,
    }


def sticking_efficiency(relative_velocity: float, *, u_crit: float = 0.02) -> float:
    """Probability that a collision produces a stable bond.

    Frazil flocs are fragile: energetic impacts restitute instead of sticking,
    so ``p = exp(-(u_rel/u_c)^2)`` with ``u_c = 0.02 m/s`` (order of the impact
    speed at which mm-scale crystals rebound).  The value is declared a priori
    and covered by the sensitivity study.
    """
    if u_crit <= 0.0:
        raise PancakeIceError("u_crit must be > 0")
    if relative_velocity < 0.0:
        raise PancakeIceError("relative_velocity must be >= 0")
    if relative_velocity / u_crit < 1e-8:
        return 1.0
    return math.exp(-((relative_velocity / u_crit) ** 2))


def orbital_retention_threshold(wave: WaveField, *, z: float = 0.0) -> float:
    """Diameter [m] above which the wave can no longer flush a floe clean.

    The orbital semi-excursion ``A_orb = a exp(kz)`` is the stroke of the water
    parcel that carries suspended frazil past a floe.  A floe with
    ``D < A_orb`` is swept clear before its collected frazil consolidates; a
    floe with ``D > A_orb`` keeps it and grows an emergent rim.  For a storm
    wave of steepness ``H/lambda = 0.1`` this threshold is ``H/2 = lambda/20``,
    i.e. exactly the Lange et al. (1989) pancake-size rule.
    """
    return wave.orbital_excursion(z)


def retention_probability(
    diameter: float,
    wave: WaveField,
    *,
    z: float = 0.0,
    sharpness: float = 30.0,
) -> float:
    """Probability that a floe of ``diameter`` retains its slush (pancake forms).

    Logistic in ``ln(D/D_c)`` with threshold ``D_c = A_orb``::

        p = 1 / (1 + (D_c/D)^sharpness)

    ``p(D_c) = 0.5`` by construction, ``p > 0.5`` for ``D > lambda/20`` and
    ``p < 0.2`` for ``D < 0.97 lambda/20`` at the declared sharpness.  A floe
    does not switch state at an exact diameter, so the transition is a narrow
    sigmoid rather than a step: ``s = 30`` corresponds to a 10%-90% transition
    over ``D/D_c in [0.93, 1.08]``.
    """
    if diameter <= 0.0:
        raise PancakeIceError("diameter must be > 0")
    if sharpness <= 0.0:
        raise PancakeIceError("sharpness must be > 0")
    dc = orbital_retention_threshold(wave, z=z)
    logit = sharpness * math.log(dc / diameter)
    if logit > 700.0:
        return 0.0
    if logit < -700.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(logit))


def pancake_bending_stress(
    diameter: float,
    thickness: float,
    wave: WaveField,
    *,
    added_mass: float = 0.0,
    rho_ice: float = RHO_ICE,
) -> float:
    """Wave-induced bending stress in a floating disc [Pa].

    A circular plate loaded by its own (plus added-mass) vertical acceleration
    ``a omega^2`` carries a maximum bending stress
    ``sigma = 0.75 rho_i (1+km) a omega^2 D^2 / h`` (thin-plate strip
    approximation: ``M = q D^2/8``, ``sigma = 6M/h^2``).  Setting
    ``sigma = sigma_flex`` gives Herman (2013)'s floe-breakup scale; with
    ``sigma_flex = 0.3`` MPa and ``h/D = 0.1`` the survivable diameter is about
    ``lambda/10``, the observed upper bound of pancake sizes.
    """
    if diameter <= 0.0 or thickness <= 0.0:
        raise PancakeIceError("diameter and thickness must be > 0")
    accel = wave.amplitude * wave.omega**2
    return 0.75 * rho_ice * (1.0 + added_mass) * accel * diameter**2 / thickness


LATENT_HEAT_FUSION: float = 3.34e5  # J kg^-1
DEFAULT_HEAT_FLUX: float = 200.0  # W m^-2, winter ocean heat loss (order 100-400)


def thermodynamic_formation_time(
    thickness: float,
    *,
    heat_flux: float = DEFAULT_HEAT_FLUX,
    rho_ice: float = RHO_ICE,
    latent_heat: float = LATENT_HEAT_FUSION,
) -> float:
    """Time [s] to build an ice layer of ``thickness`` at the given heat flux.

    ``t = rho_i h L_f / Q``.  For ``h = 0.1 m`` and ``Q = 200 W/m^2`` this is
    ~42 h -- the observed order of pancake-field formation time (Lange et al.
    1989 report pancake fields developing within one to two days).  This
    thermodynamic scale, not the collision kinetics, is what sets the formation
    time of a pancake field.
    """
    if thickness <= 0.0:
        raise PancakeIceError("thickness must be > 0")
    if heat_flux <= 0.0:
        raise PancakeIceError("heat_flux must be > 0")
    return rho_ice * thickness * latent_heat / heat_flux


def breakup_limited_diameter(
    thickness: float,
    wave: WaveField,
    *,
    sigma_flex: float | None = None,
    rho_ice: float = RHO_ICE,
) -> float:
    """Largest floe diameter that survives the wave bending stress [m]."""
    target = 0.5 * sum(SIGMA_FLEX_BAND) if sigma_flex is None else sigma_flex
    accel = wave.amplitude * wave.omega**2
    if accel <= 0.0:
        raise PancakeIceError("wave field has no acceleration")
    return math.sqrt(target * thickness / (0.75 * rho_ice * accel))


def aggregate_growth_rate(
    diameter: float,
    *,
    wave: WaveField,
    ice_volume_fraction: float = 0.05,
    crystal_diameter: float = 2.0e-3,
    aspect_ratio: float = 10.0,
    z: float = 0.0,
    dissipation: float = 0.0,
    retention_sharpness: float = 30.0,
    following: float = 0.9,
    u_crit_ballistic: float = 0.02,
    u_crit_rim: float = 1.0,
    nu_water: float = NU_WATER,
) -> dict[str, float]:
    """Diameter growth rate dD/dt [m/s] of a floe -- "collision frequency x coalescence".

    Two mechanisms, each the product of a *collision frequency* and a
    *coalescence probability*:

    * **ballistic coagulation** of a still-suspended floe (orthokinetic +
      turbulent + differential-settling kernel, sticking window ``u_crit``
      = 0.02 m/s).  A floe that fully follows the wave has no slip and this
      term dies once ``D`` is large, because the approach speed ``S D`` grows
      beyond the adhesive window.
    * **rim filtration** of a consolidated pancake: water is pumped through
      the rim face at the floe's wave-following deficit
      ``u_slip = (1 - following) u_orb`` and the slush adheres
      (``u_crit`` = 1 m/s, rims are compliant slush piles).

    The two are blended by the retention probability, exactly the criterion of
    :func:`retention_probability`: below ``D = A_orb`` the wave disperses the
    collected slush and only the (ineffective) ballistic channel remains, above
    it the rim-filtration channel takes over.  ``dV/dt`` is converted to
    ``dD/dt`` at fixed aspect ratio (``V = pi D^3/(4 AR)``); a floe whose
    bending stress exceeds the flexural strength gets no growth at all.
    """
    if diameter <= 0.0:
        raise PancakeIceError("diameter must be > 0")
    if aspect_ratio < 1.0:
        raise PancakeIceError("aspect_ratio must be >= 1")
    if not 0.0 <= following < 1.0:
        raise PancakeIceError("following must be in [0, 1)")

    n = frazil_number_density(crystal_diameter, ice_volume_fraction)
    crystal_volume = math.pi * crystal_diameter**3 / 6.0
    kernel = frazil_collision_kernel(
        diameter,
        crystal_diameter,
        wave=wave,
        z=z,
        dissipation=dissipation,
        nu_water=nu_water,
    )
    p_ret = retention_probability(diameter, wave, z=z, sharpness=retention_sharpness)
    thickness = diameter / aspect_ratio

    # (A) ballistic: suspended crystals colliding with the floe
    f_ballistic = kernel["beta_total"] * n
    p_ballistic = sticking_efficiency(kernel["relative_velocity"], u_crit=u_crit_ballistic)
    dv_ballistic = f_ballistic * crystal_volume * p_ballistic

    # (B) rim filtration at the wave-following deficit
    u_slip = (1.0 - following) * wave.orbital_velocity(z)
    f_filtration = n * math.pi * diameter * thickness * u_slip
    p_rim = sticking_efficiency(u_slip, u_crit=u_crit_rim)
    dv_filtration = f_filtration * crystal_volume * p_rim

    dv_dt = (1.0 - p_ret) * dv_ballistic + p_ret * dv_filtration
    dd_dt = dv_dt * (4.0 * aspect_ratio) / (3.0 * math.pi * diameter**2)

    sigma = pancake_bending_stress(diameter, thickness, wave)
    sigma_mid = 0.5 * sum(SIGMA_FLEX_BAND)
    survived = 1.0 if sigma < sigma_mid else 0.0

    return {
        "dD_dt": dd_dt * survived,
        "dV_dt": dv_dt * survived,
        "collision_frequency_ballistic": f_ballistic,
        "collision_frequency_filtration": f_filtration,
        "sticking_ballistic": p_ballistic,
        "sticking_rim": p_rim,
        "retention": p_ret,
        "bending_stress": sigma,
        "breakup_limited": survived < 1.0,
        "slip_velocity": u_slip,
        "threshold_diameter": orbital_retention_threshold(wave, z=z),
    }


def simulate_aggregate_growth(
    *,
    wave: WaveField,
    thickness: float = 0.1,
    aspect_ratio: float = 10.0,
    d0: float | None = None,
    d_max: float | None = None,
    ice_volume_fraction: float = 0.05,
    heat_flux: float = DEFAULT_HEAT_FLUX,
    t_max_hours: float = 6.0,
    dt: float = 1.0,
    z: float = 0.0,
    dissipation: float = 0.0,
    retention_sharpness: float = 30.0,
) -> dict[str, object]:
    """Consolidation trajectory of a pancake, and the two formation timescales.

    * ``t_consolidation_h`` -- explicit-Euler integration of
      :func:`aggregate_growth_rate` from the seed ``d0`` (default: the
      retention threshold ``A_orb``, i.e. the pancake size selected by the
      wave field) up to the breakup-limited diameter.
    * ``t_thermo_h`` -- :func:`thermodynamic_formation_time` for the pancake
      thickness.
    * ``t_form_h = t_thermo_h + t_consolidation_h`` -- the physical formation
      time, dominated by the thermodynamic production of the slush layer.

    The sub-threshold stage (frazil -> grease-ice floe, mm to dm) is *not*
    resolved here: orthokinetic coagulation of a monodisperse suspension is
    unrealistically fast unless the floc breakup equilibrium is modelled, so
    that stage is declared WITHHELD and the pancake diameter is taken as
    wave-selected (``A_orb``) rather than grown.
    """
    if d0 is None:
        d0 = orbital_retention_threshold(wave, z=z)
    if d_max is None:
        d_max = breakup_limited_diameter(thickness, wave)
    if d0 <= 0.0 or d_max <= d0:
        d0 = 0.5 * d_max
    n_steps = int(round(t_max_hours * 3600.0 / dt))
    d = d0
    t = 0.0
    hist_t: list[float] = [0.0]
    hist_d: list[float] = [d]
    t_saturation = math.nan
    for _ in range(n_steps):
        state = aggregate_growth_rate(
            d,
            wave=wave,
            ice_volume_fraction=ice_volume_fraction,
            aspect_ratio=aspect_ratio,
            z=z,
            dissipation=dissipation,
            retention_sharpness=retention_sharpness,
        )
        d_next = min(d + dt * float(state["dD_dt"]), d_max)
        t += dt
        d = d_next
        hist_t.append(t / 3600.0)
        hist_d.append(d)
        if d >= 0.999 * d_max and math.isnan(t_saturation):
            t_saturation = t / 3600.0
            break
    t_thermo = thermodynamic_formation_time(thickness, heat_flux=heat_flux)
    t_cons = 0.0 if math.isnan(t_saturation) else t_saturation
    return {
        "t_hours": hist_t,
        "diameter_m": hist_d,
        "d0_m": d0,
        "d_saturation_m": d_max,
        "t_consolidation_h": t_saturation if not math.isnan(t_saturation) else math.nan,
        "t_thermo_h": t_thermo / 3600.0,
        "t_form_h": t_thermo / 3600.0 + t_cons,
        "thickness_m": thickness,
        "aspect_ratio": aspect_ratio,
        "heat_flux_W_m2": heat_flux,
        "wave": wave.to_dict(),
        "retention_sharpness": retention_sharpness,
        "ice_volume_fraction": ice_volume_fraction,
        "target_diameter_m": orbital_retention_threshold(wave, z=z),
    }


# ---------------------------------------------------------------------------
# 2. pancake geometry, buoyancy and added mass
# ---------------------------------------------------------------------------
D_PANCAKE_BAND: tuple[float, float] = (0.3, 3.0)  # m
H_PANCAKE_BAND: tuple[float, float] = (0.05, 0.3)  # m
H_RIM_BAND: tuple[float, float] = (0.02, 0.05)  # m
ASPECT_BAND: tuple[float, float] = (5.0, 20.0)  # D/h


@dataclass(frozen=True)
class PancakeGeometry:
    """Disc + rim-ridge pancake geometry (plan circular, section trapezoidal).

    Parameters
    ----------
    diameter:
        D [m], the plan-view diameter.
    thickness:
        h [m], the (consolidated) sheet thickness of the disc interior.
    rim_height:
        h_rim [m], height of the consolidated rim ridge above the disc surface.
    rim_width_fraction:
        radial width of the rim ridge as a fraction of the radius (0.02-0.10;
        0.05 => the rim occupies 9.75% of the plan area).
    """

    diameter: float
    thickness: float
    rim_height: float = 0.03
    rim_width_fraction: float = 0.05
    rho_ice: float = RHO_ICE
    rho_water: float = RHO_WATER

    def __post_init__(self) -> None:
        for name in ("diameter", "thickness", "rim_height", "rim_width_fraction"):
            if getattr(self, name) <= 0.0:
                raise PancakeIceError(f"{name} must be > 0")
        if not 0.0 < self.rim_width_fraction < 0.5:
            raise PancakeIceError("rim_width_fraction must be in (0, 0.5)")

    # -- shape ------------------------------------------------------------
    @property
    def aspect_ratio(self) -> float:
        """D/h, the measured pancake window being [5, 20]."""
        return self.diameter / self.thickness

    @property
    def plan_area(self) -> float:
        return math.pi * self.diameter**2 / 4.0

    @property
    def rim_width(self) -> float:
        return self.rim_width_fraction * 0.5 * self.diameter

    @property
    def volume_disc(self) -> float:
        return self.plan_area * self.thickness

    @property
    def volume_rim(self) -> float:
        return math.pi * self.diameter * self.rim_width * self.rim_height

    @property
    def volume_rim_fraction(self) -> float:
        return self.volume_rim / self.volume_disc

    @property
    def volume_total(self) -> float:
        return self.volume_disc + self.volume_rim

    @property
    def mass(self) -> float:
        return self.rho_ice * self.volume_total

    # -- hydrostatics -----------------------------------------------------
    @property
    def equivalent_thickness(self) -> float:
        """Volume per unit plan area [m] (rim mass smeared over the disc)."""
        return self.volume_total / self.plan_area

    @property
    def draft(self) -> float:
        """Submerged depth from Archimedes [m]."""
        return (self.rho_ice / self.rho_water) * self.equivalent_thickness

    @property
    def freeboard_disc(self) -> float:
        """Height of the disc top surface above the waterline [m]."""
        return self.thickness - self.draft

    @property
    def freeboard_rim(self) -> float:
        """Height of the rim crest above the waterline [m]."""
        return self.thickness + self.rim_height - self.draft

    @property
    def floats(self) -> bool:
        """True when the disc supports itself (draft < thickness)."""
        return self.draft < self.thickness

    @property
    def rim_emergent(self) -> bool:
        return self.freeboard_rim >= self.rim_height

    # -- hydrodynamics ----------------------------------------------------
    @property
    def added_mass_face_on(self) -> float:
        """Lamb's added mass of a thin disc moving face-on [kg]."""
        return disc_added_mass(self.diameter, self.rho_water)

    @property
    def added_mass_ratio(self) -> float:
        """Added mass / own mass (face-on).  ~4.7 at D/h = 10."""
        return self.added_mass_face_on / self.mass

    def validate(self, *, strict: bool = True) -> dict[str, object]:
        """Check the geometry against the measured pancake windows.

        With ``strict=True`` any violation raises :class:`PancakeIceError`;
        otherwise the violations are returned in ``issues``.
        """
        issues: list[str] = []
        d_lo, d_hi = D_PANCAKE_BAND
        h_lo, h_hi = H_PANCAKE_BAND
        r_lo, r_hi = H_RIM_BAND
        a_lo, a_hi = ASPECT_BAND
        if not d_lo <= self.diameter <= d_hi:
            issues.append(f"diameter {self.diameter:g} outside [{d_lo}, {d_hi}] m")
        if not h_lo <= self.thickness <= h_hi:
            issues.append(f"thickness {self.thickness:g} outside [{h_lo}, {h_hi}] m")
        if not r_lo <= self.rim_height <= r_hi:
            issues.append(f"rim_height {self.rim_height:g} outside [{r_lo}, {r_hi}] m")
        if not a_lo <= self.aspect_ratio <= a_hi:
            issues.append(f"D/h {self.aspect_ratio:g} outside [{a_lo}, {a_hi}]")
        if not self.floats:
            issues.append("geometry does not float (draft >= thickness)")
        if strict and issues:
            raise PancakeIceError("; ".join(issues))
        return {"ok": not issues, "issues": issues, "aspect_ratio": self.aspect_ratio}

    def to_dict(self) -> dict[str, float]:
        return {
            "diameter_m": self.diameter,
            "thickness_m": self.thickness,
            "rim_height_m": self.rim_height,
            "rim_width_m": self.rim_width,
            "rim_width_fraction": self.rim_width_fraction,
            "aspect_ratio": self.aspect_ratio,
            "volume_disc_m3": self.volume_disc,
            "volume_rim_m3": self.volume_rim,
            "volume_rim_fraction": self.volume_rim_fraction,
            "mass_kg": self.mass,
            "plan_area_m2": self.plan_area,
            "draft_m": self.draft,
            "freeboard_disc_m": self.freeboard_disc,
            "freeboard_rim_m": self.freeboard_rim,
            "added_mass_kg": self.added_mass_face_on,
            "added_mass_ratio": self.added_mass_ratio,
            "floats": float(self.floats),
        }

    @classmethod
    def from_aspect_ratio(
        cls,
        diameter: float,
        aspect_ratio: float,
        **kwargs: float,
    ) -> "PancakeGeometry":
        """Build a geometry at a prescribed D/h (the measured window [5, 20])."""
        if aspect_ratio <= 0.0:
            raise PancakeIceError("aspect_ratio must be > 0")
        return cls(diameter=diameter, thickness=diameter / aspect_ratio, **kwargs)


def disc_added_mass(diameter: float, rho_water: float = RHO_WATER) -> float:
    """Added mass of a thin circular disc moving face-on [kg].

    Lamb (1932) section 108: ``m_am = (8/3) rho_w a^3`` with ``a = D/2``, i.e.
    ``rho_w D^3 / 3``.  For ``D/h = 10`` this is ~4.7x the pancake's own mass,
    which is why lateral deflection dominates the slip resistance.
    """
    if diameter <= 0.0:
        raise PancakeIceError("diameter must be > 0")
    return rho_water * diameter**3 / 3.0


def added_mass_coefficient_disc(
    aspect_ratio: float,
    *,
    rho_ice: float = RHO_ICE,
    rho_water: float = RHO_WATER,
) -> float:
    """Face-on added mass per unit ice mass: ``(4 rho_w AR)/(3 pi rho_i)``."""
    if aspect_ratio <= 0.0:
        raise PancakeIceError("aspect_ratio must be > 0")
    return (4.0 * rho_water * aspect_ratio) / (3.0 * math.pi * rho_ice)


def pancake_geometry(
    diameter: float,
    thickness: float,
    rim_height: float = 0.03,
    rim_width_fraction: float = 0.05,
) -> PancakeGeometry:
    """Convenience constructor (see :class:`PancakeGeometry`)."""
    return PancakeGeometry(
        diameter=diameter,
        thickness=thickness,
        rim_height=rim_height,
        rim_width_fraction=rim_width_fraction,
    )


def pancake_buoyancy(
    diameter: float,
    thickness: float,
    rim_height: float = 0.03,
    rim_width_fraction: float = 0.05,
) -> dict[str, object]:
    """Archimedean draft/freeboard of a pancake, rim mass included."""
    geo = pancake_geometry(diameter, thickness, rim_height, rim_width_fraction)
    out = geo.to_dict()
    out.update(
        {
            "buoyancy_force_N": RHO_WATER * G * geo.plan_area * geo.draft,
            "weight_N": geo.mass * G,
            "rim_emergent": float(geo.rim_emergent),
            "issues": geo.validate(strict=False)["issues"],
        }
    )
    return out


# ---------------------------------------------------------------------------
# 3. resistance model
# ---------------------------------------------------------------------------
def jamming_split(
    concentration: float,
    *,
    c_jam: float = DEFAULT_PARAMS["c_jam"],
    p_jam: float = DEFAULT_PARAMS["p_jam"],
) -> tuple[float, float]:
    """Split the ice interaction between the slip and bulldozing channels.

    ``f_slip = 1/(1 + (C/C_jam)^p)``, ``f_bull = 1 - f_slip``.  Sideways
    displacement of a pancake needs free space proportional to ``1-C``; above
    the jamming/clogging concentration ``C_jam`` the rafter pancake field forms
    a contact network and the whole raft has to be pushed.
    """
    if not 0.0 <= concentration <= 1.0:
        raise PancakeIceError("concentration must be in [0, 1]")
    if c_jam <= 0.0 or p_jam <= 0.0:
        raise PancakeIceError("c_jam and p_jam must be > 0")
    f_slip = 1.0 / (1.0 + (concentration / c_jam) ** p_jam)
    return f_slip, 1.0 - f_slip


def cluster_diameter(
    concentration: float,
    diameter: float,
    *,
    c_jam: float = DEFAULT_PARAMS["c_jam"],
    k_cluster: float = DEFAULT_PARAMS["k_cluster"],
) -> float:
    """Hydraulic size of a rafted cluster ``D_eff = D (1 + k_cluster (C/C_jam)^2)`` [m].

    In a concentrated field pancakes raft into conglomerates of several discs
    (Lange et al. 1989; Wadhams 2000), so the area-per-unit-mass that is dragged
    sideways is larger than a single disc.  This only affects the
    added-mass-dominated slip channel; the concentration itself still enters
    every term.
    """
    if diameter <= 0.0:
        raise PancakeIceError("diameter must be > 0")
    if c_jam <= 0.0:
        raise PancakeIceError("c_jam must be > 0")
    return diameter * (1.0 + k_cluster * (concentration / c_jam) ** 2)


def raft_thickness(
    concentration: float,
    thickness: float,
    *,
    c_jam: float = DEFAULT_PARAMS["c_jam"],
    k_raft: float = DEFAULT_PARAMS["k_raft"],
) -> float:
    """Rafted effective thickness ``h_eff = h (1 + k_raft (C/C_jam)^2)`` [m].

    Overlapping (rafted) pancakes and the ridged contacts between them make the
    field thicker than a single layer; the amplification is quadratic in the
    concentration and reaches ``1 + k_raft`` at ``C = C_jam``.
    """
    if thickness <= 0.0:
        raise PancakeIceError("thickness must be > 0")
    if c_jam <= 0.0:
        raise PancakeIceError("c_jam must be > 0")
    return thickness * (1.0 + k_raft * (concentration / c_jam) ** 2)


def slip_resistance(
    concentration: float,
    diameter: float,
    thickness_eff: float,
    speed: float,
    *,
    beam_eff: float = DEFAULT_PARAMS["beam_eff"],
    k_slip: float = DEFAULT_PARAMS["k_slip"],
    cd_pancake: float = DEFAULT_PARAMS["cd_pancake"],
    rho_ice: float = RHO_ICE,
    rho_water: float = RHO_WATER,
) -> dict[str, float]:
    """Slip channel: momentum flux of laterally deflected pancakes [N].

    Two parts, both second order in speed:

    * ``R_inertia = k_s C B_eff V^2 [rho_i h_eff + (4/(3pi)) rho_w D]`` --
      the momentum flux needed to bring a pancake of the swept corridor up to
      the slip speed ``k_s V``.  The bracket is the mass per unit plan area,
      ice plus the exact face-on disc added mass ``rho_w D/3 / (pi D^2/4)``.
    * ``R_drag = 0.5 k_s^2 C B_eff rho_w C_d D V^2`` -- the steady face-on drag
      acting while a pancake is deflected across its own diameter
      ``(residence time D/V)``.  ``C_d`` comes from the LBM drag table.
    """
    if speed < 0.0:
        raise PancakeIceError("speed must be >= 0")
    if diameter <= 0.0 or thickness_eff <= 0.0:
        raise PancakeIceError("diameter and thickness must be > 0")
    inertia_per_area = rho_ice * thickness_eff + (4.0 / (3.0 * math.pi)) * rho_water * diameter
    r_inertia = k_slip * concentration * beam_eff * speed**2 * inertia_per_area
    r_drag = (
        0.5 * k_slip**2 * concentration * beam_eff * rho_water * cd_pancake * diameter * speed**2
    )
    return {
        "R_inertia": r_inertia,
        "R_drag": r_drag,
        "total": r_inertia + r_drag,
        "inertia_per_area": inertia_per_area,
        "added_mass_per_area": (4.0 / (3.0 * math.pi)) * rho_water * diameter,
    }


def bulldozing_resistance(
    concentration: float,
    thickness_eff: float,
    speed: float,
    *,
    beam_eff: float = DEFAULT_PARAMS["beam_eff"],
    cd_raft: float = DEFAULT_PARAMS["cd_raft"],
    pile_factor: float = DEFAULT_PARAMS["pile_factor"],
    rise_factor: float = DEFAULT_PARAMS["rise_factor"],
    rho_ice: float = RHO_ICE,
    rho_water: float = RHO_WATER,
) -> dict[str, float]:
    """Bulldozing channel: drag on the raft plus lifting the piled ice [N].

    * ``A_front = C B_eff h_pile`` with ``h_pile = (1+pile_factor) h_eff``: the
      raft front spans the corridor and the pile is thicker than the field.
    * ``R_form = 0.5 rho_w C_d,raft A_front V^2`` -- from the LBM raft drag
      coefficient.
    * ``R_lift = C rho_i h_eff B_eff g (rise_factor h_pile)`` -- rate of
      potential-energy gain of the ice climbing the pile, ``mdot g dz / V``
      with ``mdot = C rho_i h_eff B_eff V`` (speed independent).
    """
    if speed < 0.0:
        raise PancakeIceError("speed must be >= 0")
    if thickness_eff <= 0.0:
        raise PancakeIceError("thickness must be > 0")
    h_pile = (1.0 + pile_factor) * thickness_eff
    a_front = concentration * beam_eff * h_pile
    r_form = 0.5 * rho_water * cd_raft * a_front * speed**2
    r_lift = concentration * rho_ice * thickness_eff * beam_eff * G * rise_factor * h_pile
    return {
        "R_form": r_form,
        "R_lift": r_lift,
        "total": r_form + r_lift,
        "frontal_area": a_front,
        "pile_thickness": h_pile,
    }


def pancake_resistance(
    concentration: float,
    diameter: float,
    thickness: float,
    speed: float,
    *,
    params: dict[str, float] | None = None,
) -> dict[str, float]:
    """Added ship resistance in a pancake field [N].

    ``R = f_slip R_slip0 + f_bull R_bull0`` with the jamming split
    (:func:`jamming_split`), the rafted thickness (:func:`raft_thickness`) and
    the two channel models (:func:`slip_resistance`,
    :func:`bulldozing_resistance`).  ``C = 0`` returns exactly zero: this is an
    *added* ice resistance, not a total resistance.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    if not 0.0 <= concentration <= 1.0:
        raise PancakeIceError("concentration must be in [0, 1]")
    if diameter <= 0.0 or thickness <= 0.0:
        raise PancakeIceError("diameter and thickness must be > 0")
    h_eff = raft_thickness(concentration, thickness, c_jam=p["c_jam"], k_raft=p["k_raft"])
    d_eff = cluster_diameter(
        concentration, diameter, c_jam=p["c_jam"], k_cluster=p["k_cluster"]
    )
    f_slip, f_bull = jamming_split(concentration, c_jam=p["c_jam"], p_jam=p["p_jam"])
    slip = slip_resistance(
        concentration,
        d_eff,
        h_eff,
        speed,
        beam_eff=p["beam_eff"],
        k_slip=p["k_slip"],
        cd_pancake=p["cd_pancake"],
    )
    bull = bulldozing_resistance(
        concentration,
        h_eff,
        speed,
        beam_eff=p["beam_eff"],
        cd_raft=p["cd_raft"],
        pile_factor=p["pile_factor"],
        rise_factor=p["rise_factor"],
    )
    r_slip = f_slip * slip["total"]
    r_bull = f_bull * bull["total"]
    total = r_slip + r_bull
    return {
        "resistance_N": total,
        "slip_N": r_slip,
        "bulldozing_N": r_bull,
        "slip_share": (r_slip / total) if total > 0.0 else 0.0,
        "bulldozing_share": (r_bull / total) if total > 0.0 else 0.0,
        "slip_inertia_N": f_slip * slip["R_inertia"],
        "slip_drag_N": f_slip * slip["R_drag"],
        "bulldozing_form_N": f_bull * bull["R_form"],
        "bulldozing_lift_N": f_bull * bull["R_lift"],
        "f_slip": f_slip,
        "f_bull": f_bull,
        "thickness_effective_m": h_eff,
        "diameter_effective_m": d_eff,
        "concentration": concentration,
        "diameter_m": diameter,
        "thickness_m": thickness,
        "speed_m_s": speed,
        "aspect_ratio": diameter / thickness,
    }


def sweep_resistance(
    concentrations: np.ndarray | list[float],
    diameters: np.ndarray | list[float],
    thicknesses: np.ndarray | list[float],
    speeds: np.ndarray | list[float],
    *,
    params: dict[str, float] | None = None,
    pointwise: bool = False,
) -> dict[str, np.ndarray]:
    """Evaluate :func:`pancake_resistance` over a parameter set.

    With ``pointwise=False`` (default) the four inputs are treated as axis
    coordinates and the full tensor product is evaluated (``len(C)*len(D)*
    len(h)*len(V)`` points).  With ``pointwise=True`` the inputs are matched
    element by element, which is what a design matrix (e.g. ``C`` and ``h``
    coupled through a fixed aspect ratio) requires.
    """
    c = np.asarray(concentrations, dtype=float)
    d = np.asarray(diameters, dtype=float)
    h = np.asarray(thicknesses, dtype=float)
    v = np.asarray(speeds, dtype=float)
    rows: list[dict[str, float]] = []
    keys = ("C", "D", "h", "V")
    if pointwise:
        c, d, h, v = np.broadcast_arrays(c, d, h, v)
        triples = zip(c.ravel(), d.ravel(), h.ravel(), v.ravel(), strict=True)
    else:
        triples = (
            (ci, di, hi, vi) for ci in c for di in d for hi in h for vi in v
        )
    for ci, di, hi, vi in triples:
        out = pancake_resistance(float(ci), float(di), float(hi), float(vi), params=params)
        out.update(dict(zip(keys, (float(ci), float(di), float(hi), float(vi)), strict=True)))
        rows.append(out)
    result: dict[str, np.ndarray] = {"shape": np.array([len(c), len(d), len(h), len(v)])}
    for key in rows[0]:
        result[key] = np.array([row[key] for row in rows], dtype=float)
    return result


def power_law_fit(
    concentration: np.ndarray,
    speed: np.ndarray,
    resistance: np.ndarray,
    *,
    min_resistance: float = 1e-12,
) -> dict[str, float]:
    """Least-squares fit ``R = a C^alpha V^beta`` in log-log space.

    Returns ``alpha``, ``beta``, ``ln_a``, ``r2`` and the sample count.  Points
    with ``R <= min_resistance`` (the ``C = 0`` negative control) are excluded;
    the fit is a plain ordinary least squares on ``ln R``, no weighting, no
    thresholds tuned after the fact.
    """
    c = np.asarray(concentration, dtype=float).ravel()
    v = np.asarray(speed, dtype=float).ravel()
    r = np.asarray(resistance, dtype=float).ravel()
    if not (c.size == v.size == r.size):
        raise PancakeIceError("concentration, speed and resistance must have equal length")
    mask = (c > 0.0) & (v > 0.0) & (r > min_resistance)
    if mask.sum() < 3:
        raise PancakeIceError("need at least 3 usable points for a power-law fit")
    x1 = np.log(c[mask])
    x2 = np.log(v[mask])
    y = np.log(r[mask])
    design = np.column_stack([np.ones_like(x1), x1, x2])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {
        "ln_a": float(coef[0]),
        "a": float(math.exp(coef[0])),
        "alpha": float(coef[1]),
        "beta": float(coef[2]),
        "r2": r2,
        "n_points": int(mask.sum()),
        "n_excluded": int((~mask).sum()),
    }


# ---------------------------------------------------------------------------
# 4. geometry masks (plan view field and LBM section)
# ---------------------------------------------------------------------------
def pancake_field_mask(
    nx: int,
    ny: int,
    *,
    concentration: float,
    diameter: float,
    cell_size: float,
    seed: int = 0,
) -> np.ndarray:
    """Plan-view boolean mask of a pancake field at concentration ``C``.

    Discs are dropped on a jittered triangular lattice with spacing
    ``sqrt(pi D^2/(4C))`` until the requested areal concentration is reached
    (measured, not assumed -- the returned coverage is exactly
    ``mask.mean()``).  Used for the plan-view figure and as the geometric
    generator for future IBM runs.
    """
    if nx < 2 or ny < 2:
        raise PancakeIceError("nx, ny must be >= 2")
    if not 0.0 < concentration <= 1.0:
        raise PancakeIceError("concentration must be in (0, 1]")
    if diameter <= 0.0 or cell_size <= 0.0:
        raise PancakeIceError("diameter and cell_size must be > 0")
    rng = np.random.default_rng(seed)
    mask = np.zeros((ny, nx), dtype=bool)
    radius = 0.5 * diameter / cell_size
    spacing = math.sqrt(math.pi * diameter**2 / (4.0 * concentration)) / cell_size
    yy, xx = np.mgrid[0:ny, 0:nx]
    target = concentration
    for j, yc in enumerate(np.arange(-spacing, ny + spacing, spacing * math.sqrt(3) / 2.0)):
        offset = 0.5 * spacing if j % 2 else 0.0
        for xc in np.arange(-spacing, nx + spacing, spacing):
            cx = xc + offset + rng.normal(0.0, 0.05 * spacing)
            cy = yc + rng.normal(0.0, 0.05 * spacing)
            mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
            if mask.mean() >= target:
                return mask
    return mask


def pancake_section_mask(
    nx: int,
    ny: int,
    *,
    diameter_cells: int,
    thickness_cells: int,
    rim_height_cells: int = 0,
    rim_width_cells: int = 0,
    x_center: int | None = None,
    y_center: int | None = None,
) -> np.ndarray:
    """2-D vertical section of a pancake for the LBM drag kernel.

    Transverse (``y``) extent = ``diameter_cells`` (the disc diameter),
    streamwise (``x``) extent = ``thickness_cells`` (the disc thickness), with
    an optional rim ridge thickening both streamwise ends by
    ``rim_height_cells`` over ``rim_width_cells``.  The section is a thick flat
    plate normal to the flow: the geometry the pancake presents when it is
    deflected face-on.
    """
    if diameter_cells < 2 or thickness_cells < 1:
        raise PancakeIceError("diameter_cells >= 2 and thickness_cells >= 1 required")
    if rim_height_cells < 0 or rim_width_cells < 0:
        raise PancakeIceError("rim sizes must be >= 0")
    cx = nx // 3 if x_center is None else x_center
    cy = ny // 2 if y_center is None else y_center
    mask = np.zeros((ny, nx), dtype=bool)
    x0 = cx - thickness_cells // 2
    x1 = x0 + thickness_cells
    y0 = cy - diameter_cells // 2
    y1 = y0 + diameter_cells
    mask[y0:y1, x0:x1] = True
    if rim_height_cells > 0 and rim_width_cells > 0:
        y_r0 = y0 - rim_height_cells
        y_r1 = y1 + rim_height_cells
        mask[max(y_r0, 0) : min(y_r1, ny), x0 : x0 + rim_width_cells] = True
        mask[max(y_r0, 0) : min(y_r1, ny), x1 - rim_width_cells : x1] = True
    return mask# ---------------------------------------------------------------------------
# 5. headline formation criterion (acceptance criterion (a))
# ---------------------------------------------------------------------------
def pancake_formation_probability(
    diameter: float,
    wave: WaveField,
    *,
    thickness: float | None = None,
    aspect_ratio: float = 10.0,
    z: float = 0.0,
    sharpness: float = 30.0,
    sigma_flex: float | None = None,
) -> dict[str, float]:
    """Probability that frazil in ``wave`` consolidates into a pancake of ``D``.

    Product of three gates:

    1. **retention gate** (:func:`retention_probability`): the floe must be
       bigger than the orbital semi-excursion ``A_orb`` to keep its slush.
       ``p = 0.5`` exactly at ``D = A_orb``, which for a storm wave of
       steepness ``H/lambda = 0.1`` is ``lambda/20`` (Lange et al. 1989).
    2. **buoyancy gate**: the geometry must float with a positive freeboard
       (Archimedes, rim mass included); ``h`` defaults to ``D/aspect_ratio``.
    3. **breakup gate**: the wave-induced bending stress must stay below the
       flexural strength (``sigma_flex`` defaults to the mid band value).

    Returns a dict with the total probability and each gate, so a violation can
    be attributed instead of just reported as a small number.
    """
    if diameter <= 0.0:
        raise PancakeIceError("diameter must be > 0")
    h = thickness if thickness is not None else diameter / aspect_ratio
    p_ret = retention_probability(diameter, wave, z=z, sharpness=sharpness)
    geo = PancakeGeometry(diameter=diameter, thickness=h)
    p_buoy = 1.0 if (geo.floats and geo.freeboard_rim > 0.0) else 0.0
    target = 0.5 * sum(SIGMA_FLEX_BAND) if sigma_flex is None else sigma_flex
    stress = pancake_bending_stress(diameter, h, wave)
    p_break = 1.0 if stress < target else 0.0
    return {
        "probability": p_ret * p_buoy * p_break,
        "retention": p_ret,
        "buoyancy": p_buoy,
        "breakup": p_break,
        "bending_stress_Pa": stress,
        "sigma_flex_Pa": target,
        "thickness_m": h,
        "aspect_ratio": diameter / h,
        "threshold_diameter_m": orbital_retention_threshold(wave, z=z),
        "diameter_over_wavelength": diameter / wave.wavelength,
    }