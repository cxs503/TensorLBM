"""Aircraft icing: rime ice accretion on a NACA 0012 airfoil (Phase 2a).

Minimal physically credible rime-icing model ("-10 C, everything freezes
on impact, no thermodynamics"):

* Air flow: D3Q19 BGK on a 2-D slice (``nz = 1``) with equilibrium inlet
  (``x = 0``), zero-gradient outlet (``x = nx-1``), free-stream lateral
  walls and full-way bounce-back on airfoil + accreted ice.
* Supercooled droplets: Lagrangian point particles seeded at the inlet at
  the *physical* LWC/MVD mass flux and integrated with Stokes relaxation
  ``du_d/dt = (u_f - u_d) / tau_d``.  No ad-hoc velocity mixing and no
  impact bias: the collection-efficiency (beta) distribution is produced
  by droplet inertia alone.
* Freezing: a droplet that enters a solid cell deposits its mass ``m_d``
  in the last fluid cell it occupied; that cell converts to ice once it
  holds ``rho_rime * dx_phys**3`` of water.  A full mass audit
  (seeded = frozen + exited + airborne + pending) closes to < 1 %.
* Ice feeds back on the flow through the bounce-back solid mask, so the
  drag/lift history shows the icing drift.

Reference case (Ruff & Wright / NASA Glenn IRT, NACA 0012 rime)
    chord 0.5334 m, V = 67 m/s, Re = 2.5e6, LWC = 0.5 g/m^3, MVD = 20 um,
    T = -10 C, t = 360 s, AoA = 4 deg.  Target: LEWICE / IRT leading-edge
    ice shape (qualitative in Phase 2a; quantitative comparison needs
    digitised reference data — deferred to Phase 2b).

Unit mapping (lattice <-> SI)
-----------------------------
Length   ``dx_phys = chord_phys / chord_lu``  [m per lattice unit]
Velocity ``u_phys = u_lu * dx_phys / dt_phys``  ->  choose ``u_lu = u_in``
         matching ``V_inf``:
         ``dt_phys = dx_phys * u_in / v_inf``  [s per lattice time step]

Droplet seeding calibration (derivation)
----------------------------------------
Single droplet mass::

    m_d = rho_w * pi * MVD**3 / 6
        = 1000 * pi * (20e-6)**3 / 6 = 4.19e-12 kg

(a 20 um droplet cloud carries ~1.2e9 droplets/m^3 — far too many to
track individually, see *Parcels* below).

Inlet area of the 2-D slab (one cell deep in z)::

    A_in = ny * dx_phys * dx_phys      [m^2]

Water mass flux through the inlet (SI)::

    Q = LWC * V_inf * A_in             [kg/s]

Mass seeded per LBM step and droplets per LBM step::

    dm/step = LWC * V_inf * A_in * dt_phys
    N/step  = LWC * V_inf * A_in * dt_phys / m_d

so the local impinging water flux — and therefore the collection
efficiency beta — is dimensionally correct, not a free parameter.

Parcels (representative droplets)
---------------------------------
The simulator tracks Lagrangian *parcels*: each parcel represents
``parcel_multiplier`` identical physical droplets (mass ``m_parcel =
N * m_d``).  Because rime droplets do not interact (dilute: volume
fraction LWC/rho_w ~ 5e-7 even after acceleration), parcel statistics
are exact for trajectories; only the sampling noise of beta improves
with more parcels.  ``parcel_multiplier = None`` picks the smallest N
that keeps the seeding rate below ``parcels_per_step_target`` (default
2000/step).  All mass bookkeeping (audit, freezing threshold, beta) is
in kg of *water*, so the parcel choice never biases the physics.

Cloud initialisation (prefill)
------------------------------
The exposure window opens with the cloud already in place, so at t=0 the
run seeds the steady-state inventory ``parcels_per_step * kill_x / u_in``
parcels uniformly upstream of the kill plane (droplet speed ~ u_in:
tau_d >> leading-edge transit, so the line density is rate/u_in).
Velocities start at the local (warmed-up) flow value and positions inside
the airfoil are rejected.  Without prefill the leading edge would be
starved for the first ~x_le/u_in steps of the window.  Set
``prefill_cloud=False`` for pure inlet-flux seeding.
``disable_droplets=True`` runs the clean-airfoil twin (no droplets, no
freezing) for an A/B measurement of the ice feedback on cd/cl.

Time acceleration (rime only)
-----------------------------
LBM time steps are tiny: with the mapping above the 360 s IRT exposure
corresponds to O(10^7) lattice steps.  Rime accretion is *linear* in LWC
(all collected water freezes where it hits, droplets do not interact, the
quasi-steady flow is independent of concentration), so the run uses an
accelerated effective cloud ``LWC_eff = k * LWC`` and reports the
equivalent physical exposure::

    t_equiv = steps * dt_phys * k

This is exact for rime and deliberately *not* used for glaze (runback and
heat balance are non-linear in LWC) — glaze is Phase 3.

Droplet relaxation
------------------
Stokes relaxation time::

    tau_d = rho_w * MVD**2 / (18 * mu_air)      [s]
    tau_d_lu = tau_d / dt_phys                   [lattice steps]

Inertia (Stokes) number, invariant between unit systems::

    St = tau_d * V_inf / chord_phys = tau_d_lu * u_in / chord_lu

With the default ``mu_air = 1.8e-5 Pa s`` the reference case gives
``St = 0.155`` (IRT-consistent).  The explicit Euler relaxation factor
per LBM step is ``dt/tau_d_lu ~ 1/400``, i.e. far below any stability
limit, so a single sub-step per LBM step is sufficient by default; the
sub-step count is configurable and auto-scales if ``tau_d_lu`` shrinks.

Rime density
------------
Ice volume — and hence horn height — scales as 1/rho_rime, so the
density model matters quantitatively.  Implemented options
(``rime_density_mode``):

* ``"macklin"`` (default) — Macklin (1962) correlation
  ``rho [g/cm^3] = 0.110 * R**0.76`` capped at 0.917, with
  ``R = -(mvd[um] * V_inf[m/s]) / (2 * T_s[C])``.  Without the Phase 3
  energy balance the surface temperature is approximated by the static
  temperature plus optional aerodynamic recovery heating
  ``T_s = T_inf + 0.5 * V^2 / (2 cp)`` (r = 0.5, cp = 1005 J/kg/K;
  ~1.1 K for the IRT case — negligible here, kept for honesty).
* ``"jones"`` — Jones (1990): ``rho = 0.0443 * (mvd V)**0.75 / |T_inf|``,
  avoids the surface-temperature iteration.
* ``"const"`` — fixed ``rho_rime`` (measured rime spans ~100-900 kg/m^3,
  glaze 917).

For the IRT reference case Macklin gives R ~ 67-75, i.e. the cap: the
-10 C / 67 m /s / 20 um point is *hard* rime at essentially solid
density (917 kg/m^3) — consistent with it sitting near the glaze
boundary.  Colder/slower/smaller-droplet conditions give the fluffy
100-300 kg/m^3 values (e.g. 20 um, 10 m/s, -30 C -> ~270 kg/m^3).

Phase 2b: Eulerian droplet field + CUMULANT/LES air flow
--------------------------------------------------------
The Phase 2a Lagrangian machinery above is kept byte-identical (same
code path when ``droplet_phase="lagrangian"``, the default).  Phase 2b
adds, in the same module:

* **Eulerian droplet phase** (``droplet_phase="eulerian" | "both"``):
  the droplet cloud is carried as a volume-fraction field ``alpha(x,t)``
  with momentum ``(mx, my) = alpha * u_d`` on the *same* grid as the
  flow, FENSAP-ICE style.  The discretisation is a donor-cell (first
  order upwind) finite-volume scheme, fully tensorised
  (``RimeIcingSimulation._euler_step``): conservative advection of
  ``alpha`` and ``(mx, my)`` with shared face fluxes, then operator-split
  drag relaxation ``u_d <- u_f + (u_d - u_f) * exp(-1/tau_eff)`` —
  unconditionally stable, which removes the high-St stiffness.  Solid
  cells are perfect absorbers: the outgoing face flux of a fluid cell
  into a solid neighbour *is* the wall-impingement sink, recorded on the
  donating fluid cell — exactly the Phase 2a deposit convention, so both
  phases feed the identical freezer (``_freeze``), unit mapping, rime
  density and beta normalisation.  ``alpha`` is clipped at zero
  (positivity) and cells below ``shadow_alpha_frac * alpha_in`` are
  "shadow region": their velocity is penalized to the local carrier
  velocity (Bourgault/Habashi shadow treatment) so ``m / alpha`` never
  blows up in the droplet wake.
* **Cross-validation**: ``droplet_phase="both"`` runs the Lagrangian and
  Eulerian droplet phases on the *same* flow/ice trajectory (the
  Lagrangian arm drives freezing, the Eulerian arm is diagnostic), so
  beta curves from the two formulations differ by discretisation and
  inertia resolution only — the Bellosta (2023) comparison protocol.
* **Collision upgrade**: ``collision="cumulant"`` swaps the BGK collide
  for ``tensorlbm.cumulant.collide_cumulant_d3q19`` (production-proven
  on the SUBOFF chain) with optional per-cell Smagorinsky LES
  (``c_s > 0``, built into the kernel).  ``re_lu_target`` derives the
  knife-edge relaxation time ``tau_flow = 3 u_in chord_lu / Re + 0.5``
  (BGK stalls around ``tau - 0.5 ~ 1.7e-2``; CUMULANT is measured
  stable down to ``tau - 0.5 ~ 9.6e-6`` on this stack, i.e.
  ``Re_lu ~ 2e6`` at ``chord_lu = 128``).
* **Drag law**: ``drag_law="schiller-naumann"`` upgrades both phases to
  ``f_drag = 1 + 0.15 Re_p^0.687`` with the particle Reynolds number
  built from the physical unit mapping (``re_p_scale``; ``Re_p =
  |u_f - u_d|_lu * re_p_scale``, ~100 near the IRT leading edge, i.e.
  a ~4.6x faster relaxation than Stokes).  The default ``"stokes"``
  keeps Phase 2a trajectories unchanged; both phases always use the
  *same* law so cross-validation stays meaningful.

Phase 3: Messinger glaze thermodynamics + thin-film runback
-----------------------------------------------------------
``thermo_model="messinger"`` + :func:`run_glaze_icing` add the glaze
(wet-ice) regime on top of the unchanged Phase 2a/2b machinery, in the
industry-standard multishot sequence (aero -> impingement beta ->
surface thermodynamics -> geometry update, repeated):

* **Surface energy balance** (:func:`messinger_panel_fluxes`): per
  arc-length panel, the Messinger control volume with convective heat
  transfer (Frossling stagnation value with a cylinder-equivalent
  laminar decay and a turbulent flat-plate envelope, or the
  Reynolds/Chilton-Colburn analogy from the sampled LBM wall shear),
  evaporative cooling (Lewis analogy, saturation over ice below 0 C),
  impingement sensible + kinetic heating and the latent heat of
  freezing; solved in the classical two-regime way for the film
  temperature ``T_s`` and the freezing fraction ``n_f`` (rime:
  ``n_f = 1``, ``T_s < 0``; glaze: ``T_s = 0``, ``n_f < 1``).
* **Runback** (:func:`solve_glaze_surface`): the unfrozen fraction
  leaves each panel as a film and feeds the next panel downstream
  (mass-flux "overflow" model, marching away from the stagnation panel
  whose outflow splits between the two surfaces); a steady
  shear-driven Myers film thickness ``h_f = sqrt(2 mu_w q / tau_t)``
  is reported as a diagnostic.
* **Glaze ice shape** (:func:`deposit_glaze_ice`): the frozen mass is
  credited to the panel's deposit cells (impact-share weights) and
  freezes column-by-column outward from the surface with the local ice
  density (Macklin at the *Messinger* surface temperature -> 917 kg/m^3
  in the glaze limit), so the classic features appear: a thinner water
  cap at the stagnation line with runback ice horns downstream.
* **Regression guarantee**: the rime limit (cold / small LWC) gives
  ``n_f = 1`` with zero runback, and the deposit reproduces the Phase
  2a voxel shape from the same ledger.  The droplet/flow run stays
  LWC-accelerated (exact for beta); the acceleration is pinned per shot
  so the ledger equals ``dt_shot`` of *physical* exposure and the
  thermodynamics always runs at the physical LWC.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tensorlbm.compile_utils import compile_step, validate_compile_mode
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import C, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d

__all__ = [
    "IcingConfig",
    "RimeIcingSimulation",
    "run_rime_icing",
    "naca0012_mask_2d",
    "surface_arc_length",
    "collection_efficiency_curve",
    "ice_shape_metrics",
    "seed_counts_total",
    "rime_density_macklin",
    "rime_density_jones",
    "eulerian_mass_audit_report",
    "saturation_vapor_pressure_pa",
    "analytic_htc_w_m2k",
    "analytic_tau_pa",
    "messinger_panel_fluxes",
    "build_surface_panels",
    "solve_glaze_surface",
    "deposit_glaze_ice",
    "run_glaze_icing",
]


# ---------------------------------------------------------------------------
# Configuration + unit mapping
# ---------------------------------------------------------------------------
@dataclass
class IcingConfig:
    """Parameters for the 2-D rime icing simulation.

    Lattice and physical parameter sets are tied together by the unit
    mapping derived in the module docstring; all derived quantities are
    exposed as properties so every printed number is traceable.
    """

    # --- lattice ---
    nx: int = 320
    ny: int = 160
    chord_frac: float = 0.4
    u_in: float = 0.05
    tau: float = 0.55
    aoa_deg: float = 4.0
    cx_frac: float = 0.3
    cy_frac: float = 0.5
    naca_t: float = 0.12

    # --- physical reference case (NASA Glenn IRT NACA 0012 rime) ---
    chord_phys: float = 0.5334  # m
    v_inf: float = 67.0  # m/s
    lwc: float = 5.0e-4  # kg/m^3 (0.5 g/m^3)
    mvd: float = 20.0e-6  # m
    rho_water: float = 1000.0  # kg/m^3
    mu_air: float = 1.8e-5  # Pa s (gives St = 0.155 for this case)
    rime_density_mode: str = "macklin"  # "const" | "macklin" | "jones"
    rho_rime: float = 917.0  # kg/m^3, used when rime_density_mode == "const"
    t_static_c: float = -10.0  # static air temperature [C]
    recovery_heating: bool = True  # T_s proxy: T_inf + 0.5 V^2/(2 cp)
    t_exposure: float = 360.0  # s of cloud exposure to represent

    # --- run control ---
    steps: int = 3000
    warmup_steps: int = 2000
    substeps: int | None = None  # None -> auto from tau_d_lu
    parcel_multiplier: int | None = None  # None -> auto (target below)
    parcels_per_step_target: float = 2000.0
    kill_frac: float = 0.15  # droplet kill plane: LE + kill_frac*chord
    # Steady-state cloud inventory at t=0 of the exposure window (the real
    # cloud is already everywhere when icing starts; without prefill the
    # first ~x_le/u_in steps of the window would collect nothing and the
    # leading-edge flux would be underestimated).
    prefill_cloud: bool = True
    beta_window_frac: float = 0.5  # trailing mode: trailing fraction of steps
    # Beta measurement window (task #84 fix 1).  "clean" (default) measures
    # beta on the *pre-ice reference geometry*: an early window that opens
    # after the cloud has settled (0.5 * tau_d_lu, capped at steps/4) and
    # closes before the first wall cell can fill with ice, so the window
    # beta is an LWC invariant (the Phase 2a/2b trailing window measured
    # collection on the iced geometry, where a filled cell stops collecting
    # and pins the per-cell beta at the moving-boundary cap
    # ``beta_cap_window`` -- at LWC 1.0 the peak collapsed -55%).
    # "trailing" keeps the legacy Phase 2a/2b behaviour exactly.
    beta_window_mode: str = "clean"  # "clean" | "trailing"
    beta_clean_frac: float = 0.2  # clean mode: window length as fraction of steps
    beta_clean_max_fill: float = 0.5  # clean mode: max expected LE fill [cells]
    disable_droplets: bool = False  # clean-airfoil twin for A/B cd drift
    uniform_flow: bool = False  # True: static uniform field (tests, no LBM)
    device: str = "cpu"
    seed: int = 0
    log_every: int = 250
    accel_override: float | None = None  # explicit k; else derived from t_exposure
    compile_mode: str | None = None  # flow step: None | "default" |
    #                                      "max-autotune-no-cudagraphs"

    # --- Phase 2b additions (defaults keep Phase 2a byte-identical) ---
    droplet_phase: str = "lagrangian"  # "lagrangian" | "eulerian" | "both"
    collision: str = "bgk"  # "bgk" (2a) | "cumulant" (+ optional Smag c_s)
    c_s: float = 0.0  # Smagorinsky constant (cumulant collision only)
    re_lu_target: float | None = None  # if set: tau_flow = 3 u L / Re + 0.5
    rho_air: float = 1.34  # kg/m^3 (Schiller-Naumann Re_p, -10 C air)
    # Calibration multiplier on the Schiller-Naumann particle Reynolds
    # number (task #84 fix 2, from the #79 calibration report): decouples
    # the drag-correction strength from the physical air density.
    # 1.0 = physical Schiller-Naumann (rho_air = 1.34 kg/m^3 IRT air);
    # sn_scale_factor = lambda is exactly the rho_air = lambda*1.34 sweep
    # of report #79 (single knob, no lie about the air density).
    sn_scale_factor: float = 1.0
    drag_law: str = "stokes"  # "stokes" (2a) | "schiller-naumann"
    shadow_alpha_frac: float = 1e-3  # shadow threshold as fraction of alpha_in

    # --- Phase 3 additions (glaze Messinger thermodynamics; defaults keep
    #     the Phase 2a/2b paths byte-identical) ---
    # Surface thermodynamics selector: "instant" is the Phase 2a/2b
    # everything-freezes-on-impact model; "messinger" routes the surface
    # energetics through the Phase 3 balance (run_glaze_icing).
    thermo_model: str = "instant"  # "instant" | "messinger"
    # False: measure impacts only (the freezer never fires in-run) — used by
    # the glaze multishot driver; True is the exact 2a/2b behaviour.
    freeze_in_run: bool = True
    # Convective heat-transfer correlation: "analytic" = Frossling
    # stagnation value with a cylinder-equivalent laminar decay and a
    # turbulent flat-plate envelope; "shear" = Reynolds/Chilton-Colburn
    # analogy from the LBM wall shear (Frossling floor near the
    # stagnation line, where tau_w -> 0 breaks the analogy).
    htc_mode: str = "analytic"  # "analytic" | "shear"
    glaze_panel_cells: float = 1.0  # arc-length panel width for the thermo [cells]
    # Ice density used by the glaze deposit: "macklin-ts" = Macklin with the
    # *Messinger* surface temperature (-> 917 kg/m^3 in the glaze limit,
    # matches 2a in the rime limit); "const" = cfg.rho_rime.
    glaze_rho_mode: str = "macklin-ts"  # "macklin-ts" | "const"
    rh: float = 1.0  # cloud relative humidity (evaporation term)
    evap_enabled: bool = True
    p_static: float = 101325.0  # ambient pressure [Pa]
    recovery_factor: float | None = None  # None -> sqrt(Pr) (laminar)
    le_diameter: float | None = None  # None -> 2*1.1019*t^2*chord (NACA LE)
    # thermophysical constants (SI)
    cp_water: float = 4184.0  # J/kg/K
    cp_air: float = 1005.0  # J/kg/K
    l_fusion: float = 3.34e5  # J/kg
    l_vapor: float = 2.501e6  # J/kg
    prandtl_air: float = 0.71
    k_air: float = 0.0235  # W/m/K (analytic htc only)
    mu_water: float = 1.79e-3  # Pa s at 0 C (runback film diagnostic)

    def __post_init__(self) -> None:
        if self.droplet_phase not in ("lagrangian", "eulerian", "both"):
            raise ValueError(
                f"droplet_phase must be 'lagrangian', 'eulerian' or 'both'; "
                f"got {self.droplet_phase!r}"
            )
        if self.collision not in ("bgk", "cumulant"):
            raise ValueError(f"collision must be 'bgk' or 'cumulant'; got {self.collision!r}")
        if self.drag_law not in ("stokes", "schiller-naumann"):
            raise ValueError(f"drag_law must be 'stokes' or 'schiller-naumann'; got {self.drag_law!r}")
        if self.c_s < 0.0:
            raise ValueError(f"c_s must be >= 0; got {self.c_s}")
        if self.collision == "bgk" and self.c_s > 0.0:
            raise ValueError(
                "c_s > 0 (Smagorinsky LES) requires collision='cumulant'; "
                "the Phase 2a BGK path has no SGS coupling"
            )
        if self.re_lu_target is not None and self.re_lu_target <= 0.0:
            raise ValueError(f"re_lu_target must be > 0; got {self.re_lu_target}")
        if self.shadow_alpha_frac < 0.0:
            raise ValueError(f"shadow_alpha_frac must be >= 0; got {self.shadow_alpha_frac}")
        if self.sn_scale_factor < 0.0:
            raise ValueError(f"sn_scale_factor must be >= 0; got {self.sn_scale_factor}")
        # --- task #84 fix 1 ---
        if self.beta_window_mode not in ("clean", "trailing"):
            raise ValueError(
                f"beta_window_mode must be 'clean' or 'trailing'; "
                f"got {self.beta_window_mode!r}"
            )
        if not 0.0 <= self.beta_window_frac <= 1.0:
            raise ValueError(f"beta_window_frac must be in [0, 1]; got {self.beta_window_frac}")
        if not 0.0 < self.beta_clean_frac <= 1.0:
            raise ValueError(f"beta_clean_frac must be in (0, 1]; got {self.beta_clean_frac}")
        if not 0.0 < self.beta_clean_max_fill <= 1.0:
            raise ValueError(
                f"beta_clean_max_fill must be in (0, 1]; got {self.beta_clean_max_fill}"
            )
        # --- Phase 3 ---
        if self.thermo_model not in ("instant", "messinger"):
            raise ValueError(
                f"thermo_model must be 'instant' or 'messinger'; got {self.thermo_model!r}"
            )
        if self.htc_mode not in ("analytic", "shear"):
            raise ValueError(f"htc_mode must be 'analytic' or 'shear'; got {self.htc_mode!r}")
        if self.glaze_rho_mode not in ("macklin-ts", "const"):
            raise ValueError(
                f"glaze_rho_mode must be 'macklin-ts' or 'const'; got {self.glaze_rho_mode!r}"
            )
        if not 0.0 < self.rh <= 1.0:
            raise ValueError(f"rh must be in (0, 1]; got {self.rh!r}")
        if self.glaze_panel_cells <= 0.0:
            raise ValueError(f"glaze_panel_cells must be > 0; got {self.glaze_panel_cells!r}")

    # ------------------------------------------------------------------
    # lattice-side derived quantities
    @property
    def chord_lu(self) -> float:
        return self.nx * self.chord_frac

    @property
    def tau_flow(self) -> float:
        """Relaxation time actually used by the flow step.

        ``re_lu_target`` derives the knife-edge tau from the target lattice
        Reynolds number (Phase 2b high-Re path); without it this is just
        ``tau`` (Phase 2a behaviour).
        """
        if self.re_lu_target is not None:
            return 3.0 * self.u_in * self.chord_lu / self.re_lu_target + 0.5
        return self.tau

    @property
    def nu_lu(self) -> float:
        return (self.tau_flow - 0.5) / 3.0

    @property
    def re_lu(self) -> float:
        return self.u_in * self.chord_lu / self.nu_lu

    # ------------------------------------------------------------------
    # unit mapping
    @property
    def dx_phys(self) -> float:
        """Metres per lattice unit."""
        return self.chord_phys / self.chord_lu

    @property
    def dt_phys(self) -> float:
        """Seconds per lattice time step (u_lu = u_in <-> v_inf)."""
        return self.dx_phys * self.u_in / self.v_inf

    # ------------------------------------------------------------------
    # droplet physics
    @property
    def m_droplet(self) -> float:
        """Single droplet mass [kg]."""
        return self.rho_water * math.pi * self.mvd**3 / 6.0

    @property
    def tau_d_phys(self) -> float:
        """Stokes relaxation time [s]."""
        return self.rho_water * self.mvd**2 / (18.0 * self.mu_air)

    @property
    def tau_d_lu(self) -> float:
        """Stokes relaxation time in lattice steps."""
        return self.tau_d_phys / self.dt_phys

    @property
    def stokes(self) -> float:
        """Inertia number (unit-system invariant)."""
        return self.tau_d_phys * self.v_inf / self.chord_phys

    @property
    def t_surface_eff_c(self) -> float:
        """Surface-temperature proxy for the rime density correlations.

        Static temperature plus (optionally) aerodynamic recovery heating
        ``0.5 * V^2 / (2 cp)`` with cp = 1005 J/kg/K (r = 0.5).
        """
        if self.recovery_heating:
            return self.t_static_c + 0.5 * self.v_inf**2 / (2.0 * 1005.0)
        return self.t_static_c

    @property
    def rime_R_macklin(self) -> float:
        """Macklin parameter R = -(mvd[um] * V) / (2 T_s)."""
        if self.t_surface_eff_c >= 0.0:
            return float("inf")
        return -(self.mvd * 1e6 * self.v_inf) / (2.0 * self.t_surface_eff_c)

    @property
    def rho_rime_eff(self) -> float:
        """Effective rime density [kg/m^3] under the configured mode."""
        if self.rime_density_mode == "const":
            return self.rho_rime
        if self.rime_density_mode == "macklin":
            return rime_density_macklin(self.mvd, self.v_inf, self.t_surface_eff_c)
        if self.rime_density_mode == "jones":
            return rime_density_jones(self.mvd, self.v_inf, self.t_static_c)
        raise ValueError(
            f"rime_density_mode must be 'const', 'macklin' or 'jones'; "
            f"got {self.rime_density_mode!r}"
        )

    @property
    def m_cell_ice(self) -> float:
        """Ice mass that fills one cell [kg]."""
        return self.rho_rime_eff * self.dx_phys**3

    @property
    def inlet_area(self) -> float:
        """Inlet cross-section of the 2-D slab (one cell deep) [m^2]."""
        return self.ny * self.dx_phys * self.dx_phys

    @property
    def lwc_accel(self) -> float:
        """LWC acceleration factor k (see docstring: exact for rime)."""
        if self.accel_override is not None:
            return self.accel_override
        return self.t_exposure / (self.steps * self.dt_phys)

    @property
    def lwc_eff(self) -> float:
        return self.lwc * self.lwc_accel

    @property
    def droplets_per_step(self) -> float:
        """Analytic seeding rate [physical droplets / LBM step] (docstring)."""
        return self.lwc_eff * self.v_inf * self.inlet_area * self.dt_phys / self.m_droplet

    @property
    def effective_parcel_multiplier(self) -> int:
        """Physical droplets represented by one simulated parcel."""
        if self.parcel_multiplier is not None:
            return max(1, int(self.parcel_multiplier))
        n = math.ceil(self.droplets_per_step / max(self.parcels_per_step_target, 1.0))
        return max(1, int(n))

    @property
    def m_parcel(self) -> float:
        """Mass of one simulated parcel [kg]."""
        return self.effective_parcel_multiplier * self.m_droplet

    @property
    def parcels_per_step(self) -> float:
        """Simulated parcels seeded per LBM step (mass flux is exact)."""
        return self.droplets_per_step / self.effective_parcel_multiplier

    @property
    def t_equiv(self) -> float:
        """Equivalent physical cloud exposure achieved by the run [s]."""
        return self.steps * self.dt_phys * self.lwc_accel

    @property
    def n_substeps(self) -> int:
        if self.substeps is not None:
            return max(1, int(self.substeps))
        # explicit Euler relaxation needs dt/tau << 1; 8 sub-steps margin.
        return max(1, int(math.ceil(8.0 / max(self.tau_d_lu, 1e-12))))

    # ------------------------------------------------------------------
    # Phase 2b: Eulerian droplet field derived quantities
    @property
    def alpha_in(self) -> float:
        """Inlet droplet volume fraction (accelerated cloud, like 2a LWC)."""
        return self.lwc_eff / self.rho_water

    @property
    def shadow_alpha_min(self) -> float:
        """Shadow-region threshold on alpha [volume fraction]."""
        return self.shadow_alpha_frac * self.alpha_in

    @property
    def beta_window_bounds(self) -> tuple[int, int]:
        """Step range ``(w0, w1)`` over which the beta ledger is differenced.

        ``"trailing"`` (Phase 2a/2b legacy): the trailing
        ``beta_window_frac`` fraction of the run, on the *iced* geometry.
        ``"clean"`` (default, task #84 fix 1): an early window on the
        pre-ice reference geometry.  It opens after the cloud has settled
        (``0.5 * tau_d_lu``, capped at a quarter of the run so short test
        runs still get a window) and closes before the first wall cell
        can fill with ice: with the frontal bound ``alpha_in * u_in`` on
        the per-step leading-edge catch, the *cumulative* fill up to
        ``w1`` stays below one cell-ice mass, so no cell turns solid
        before the window closes and the per-cell ledger never saturates
        at the moving-boundary cap (see ``beta_cap_window``).  The window
        length is the smaller of ``beta_clean_frac * steps`` and
        ``beta_clean_max_fill`` cell-ice masses of in-window catch.
        """
        if self.beta_window_mode == "trailing":
            w0 = self.steps - int(self.beta_window_frac * self.steps)
            return w0, self.steps
        settle = max(1, min(int(math.ceil(0.5 * self.tau_d_lu)), self.steps // 4))
        n_nominal = max(1, int(self.beta_clean_frac * self.steps))
        fill_rate = self.alpha_in * self.u_in  # frontal bound on LE catch [lu^3/step]
        m_cell_lu = self.rho_rime_eff / self.rho_water  # cell ice mass [lu^3]
        if fill_rate > 0.0:
            # steps until the LE donor accumulates one cell of ice
            n_freeze = m_cell_lu / fill_rate
            # keep at least a minimum window before the first freeze (the
            # prefill starts the cloud in local equilibrium, so opening
            # earlier only shortens the coupled-flow settling)
            min_len = max(2, self.steps // 50)
            horizon = max(1, int(n_freeze))
            settle = max(1, min(settle, horizon - min_len))
            # in-window catch stays below beta_clean_max_fill cells ...
            n_cap = int(self.beta_clean_max_fill * m_cell_lu / fill_rate)
            n_nominal = min(n_nominal, max(1, n_cap))
            # ... and the cumulative catch up to w1 stays below one cell
            n_nominal = min(n_nominal, max(1, horizon - settle))
        w0 = min(settle, max(1, self.steps - 1))
        w1 = min(self.steps, w0 + n_nominal)
        return w0, w1

    @property
    def mass_per_lu3(self) -> float:
        """kg of water per lattice cell volume (alpha is a volume fraction)."""
        return self.rho_water * self.dx_phys**3

    @property
    def re_p_scale(self) -> float:
        """Schiller-Naumann scale: ``Re_p = |u_f - u_d|_lu * re_p_scale``.

        ``|u|_lu * dx/dt`` converts a lattice velocity difference to m/s
        (the same unit mapping as ``tau_d_lu``), so ``Re_p`` is the
        physical particle Reynolds number and invariant across unit
        systems.  Returns 0 for the Stokes law, which switches the drag
        factor to exactly 1 inside the step (single code path).
        """
        if self.drag_law != "schiller-naumann":
            return 0.0
        return (
            self.sn_scale_factor * self.rho_air
            * (self.dx_phys / self.dt_phys) * self.mvd / self.mu_air
        )

    @property
    def beta_cap_window(self) -> float:
        """Moving-boundary cap on the window beta (Eulerian-driven freezing).

        A fluid cell adjacent to the wall stops collecting once the rime
        freezer has filled it, so the per-cell impact mass recorded over
        the beta window cannot exceed one cell of ice::

            beta_cap = m_cell_ice / (lwc_eff * V * dx^2 * t_win)

        evaluated on the *actual* ledger window (``beta_window_bounds``).
        In ``"trailing"`` mode this reduces to
        ``rho_rime*dx / (LWC*V*(1-beta_window_frac)*t_exposure)`` (task
        #79: at t_exposure=360 s this is 0.634 @ dx=4.17 mm, 0.228 @
        1.50 mm, 0.122 @ 0.80 mm -- the window beta of fine-grid 360 s
        runs saturates at the cap and no longer measures the collection
        efficiency; the standard dx=4.17 mm case ran at 93% of its cap).
        The ``"clean"`` window (#84 fix 1) closes before the first fill,
        so beta_pk/cap stays small there by construction.  ``inf`` when
        the window is empty (glaze whole-shot convention).
        """
        w0, w1 = self.beta_window_bounds
        t_win = (w1 - w0) * self.dt_phys
        if t_win <= 0.0:
            return float("inf")
        return self.m_cell_ice / (
            self.lwc_eff * self.v_inf * self.dx_phys**2 * t_win
        )

    # ------------------------------------------------------------------
    # Phase 3: glaze thermodynamics derived quantities
    @property
    def recovery_factor_eff(self) -> float:
        """Thermal recovery factor r (laminar sqrt(Pr) by default)."""
        return math.sqrt(self.prandtl_air) if self.recovery_factor is None else self.recovery_factor

    @property
    def le_diameter_eff(self) -> float:
        """Effective leading-edge cylinder diameter [m] (Frossling htc).

        NACA 4-digit leading-edge radius ``r_le = 1.1019 * t^2 * chord``.
        """
        if self.le_diameter is not None:
            return self.le_diameter
        return 2.0 * 1.1019 * self.naca_t**2 * self.chord_phys

    def mapping_report(self) -> dict[str, Any]:
        """All mapping/physics numbers in one dictionary (printed + JSON)."""
        return {
            "nx": self.nx,
            "ny": self.ny,
            "chord_lu": self.chord_lu,
            "u_in": self.u_in,
            "tau": self.tau,
            "nu_lu": self.nu_lu,
            "re_lu": self.re_lu,
            "dx_phys": self.dx_phys,
            "dt_phys": self.dt_phys,
            "chord_phys": self.chord_phys,
            "v_inf": self.v_inf,
            "lwc": self.lwc,
            "lwc_eff": self.lwc_eff,
            "lwc_accel": self.lwc_accel,
            "mvd": self.mvd,
            "m_droplet": self.m_droplet,
            "m_cell_ice": self.m_cell_ice,
            "rho_rime_eff": self.rho_rime_eff,
            "rime_density_mode": self.rime_density_mode,
            "t_static_c": self.t_static_c,
            "t_surface_eff_c": self.t_surface_eff_c,
            "rime_R_macklin": self.rime_R_macklin,
            "tau_d_phys": self.tau_d_phys,
            "tau_d_lu": self.tau_d_lu,
            "stokes": self.stokes,
            "droplets_per_step": self.droplets_per_step,
            "parcel_multiplier": self.effective_parcel_multiplier,
            "m_parcel": self.m_parcel,
            "parcels_per_step": self.parcels_per_step,
            "inlet_area": self.inlet_area,
            "t_exposure_target": self.t_exposure,
            "t_equiv": self.t_equiv,
            "beta_window_mode": self.beta_window_mode,
            "beta_window_bounds": list(self.beta_window_bounds),
            "steps_realtime": self.t_exposure / self.dt_phys,
            "n_substeps": self.n_substeps,
            "compile_mode": self.compile_mode,
            "uniform_flow": self.uniform_flow,
            # --- Phase 2b ---
            "droplet_phase": self.droplet_phase,
            "collision": self.collision,
            "c_s": self.c_s,
            "re_lu_target": self.re_lu_target,
            "tau_flow": self.tau_flow,
            "drag_law": self.drag_law,
            "re_p_scale": self.re_p_scale,
            "sn_scale_factor": self.sn_scale_factor,
            "rho_air": self.rho_air,
            "beta_cap_window": self.beta_cap_window,
            "alpha_in": self.alpha_in,
            "shadow_alpha_min": self.shadow_alpha_min,
            # --- Phase 3 ---
            "thermo_model": self.thermo_model,
            "freeze_in_run": self.freeze_in_run,
            "htc_mode": self.htc_mode,
            "glaze_rho_mode": self.glaze_rho_mode,
            "glaze_panel_cells": self.glaze_panel_cells,
            "rh": self.rh,
            "evap_enabled": self.evap_enabled,
            "le_diameter_eff": self.le_diameter_eff,
            "recovery_factor_eff": self.recovery_factor_eff,
        }


def seed_counts_total(rate: float, steps: int) -> int:
    """Deterministic integer seeding over ``steps`` steps.

    Fractional droplet-per-step rates are accumulated in a carry so the
    expected total ``rate * steps`` is reproduced exactly (to one droplet)
    without stochastic rounding.  Used by the simulator and by the
    seeding-calibration unit test.
    """
    total = 0
    carry = 0.0
    for _ in range(steps):
        carry += rate
        n = int(carry)
        carry -= n
        total += n
    return total


# ---------------------------------------------------------------------------
# Rime density correlations (see module docstring)
# ---------------------------------------------------------------------------
RIME_DENSITY_CAP_G_CM3 = 0.917  # solid ice


def rime_density_macklin(mvd: float, v_inf: float, t_surface_c: float) -> float:
    """Macklin (1962) rime density [kg/m^3].

    ``rho [g/cm^3] = min(0.917, 0.110 * R**0.76)`` with the Macklin
    parameter ``R = -(mvd[um] * v_inf[m/s]) / (2 * T_s[C]) >= 0``.
    """
    if t_surface_c >= 0.0:
        return RIME_DENSITY_CAP_G_CM3 * 1000.0
    r_param = -(mvd * 1e6 * v_inf) / (2.0 * t_surface_c)
    rho = min(RIME_DENSITY_CAP_G_CM3, 0.110 * r_param**0.76)
    return rho * 1000.0


def rime_density_jones(mvd: float, v_inf: float, t_air_c: float) -> float:
    """Jones (1990) rime density [kg/m^3] (static temperature, no T_s)."""
    rho = min(RIME_DENSITY_CAP_G_CM3, 0.0443 * (mvd * 1e6 * v_inf) ** 0.75 / abs(t_air_c))
    return rho * 1000.0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def naca0012_mask_2d(
    nx: int,
    ny: int,
    chord: float,
    aoa_deg: float = 4.0,
    cx: float | None = None,
    cy: float | None = None,
    t: float = 0.12,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Vectorised NACA 4-digit symmetric airfoil solid mask (ny, nx).

    Implicit fill: a cell is solid when its airfoil-aligned coordinates
    (xr, yr) satisfy 0 <= xr <= 1 and |yr| <= y_t(xr), with
    ``y_t = 5 t (0.2969 sqrt(x) - 0.126 x - 0.3516 x^2 + 0.2843 x^3 - 0.1015 x^4)``.
    Positive ``aoa_deg`` rotates the airfoil nose-up (LE above the TE).
    """
    if cx is None:
        cx = nx / 3.0
    if cy is None:
        cy = ny / 2.0
    dev = torch.device(device)
    xx = torch.arange(nx, device=dev, dtype=torch.float32)
    yy = torch.arange(ny, device=dev, dtype=torch.float32)
    gx, gy = torch.meshgrid(xx, yy, indexing="xy")  # (ny, nx)
    dx = gx - cx
    dy = gy - cy
    aoa = math.radians(aoa_deg)
    cos_a, sin_a = math.cos(aoa), math.sin(aoa)
    # positive AoA = nose-up (trailing edge below the leading edge),
    # matching the convention of tensorlbm.airfoil_benchmark
    xr = (dx * cos_a - dy * sin_a) / chord
    yr = (dx * sin_a + dy * cos_a) / chord
    xc = xr.clamp(0.0, 1.0)
    yt = (
        5.0
        * t
        * (0.2969 * torch.sqrt(xc) - 0.126 * xc - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1015 * xc**4)
    )
    return (xr >= 0.0) & (xr <= 1.0) & (yr.abs() <= yt)


def _dilate4(mask: torch.Tensor) -> torch.Tensor:
    return (
        torch.roll(mask, 1, 0)
        | torch.roll(mask, -1, 0)
        | torch.roll(mask, 1, 1)
        | torch.roll(mask, -1, 1)
    )


# ---------------------------------------------------------------------------
# Surface arc-length machinery (numpy, CPU, run once at the end)
# ---------------------------------------------------------------------------
def surface_arc_length(airfoil: np.ndarray) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """Signed arc distance from the leading-edge surface cell.

    Returns ``(s_grid, stag_xy, surf)``:

    * ``surf`` — surface cells of the *original* airfoil (solid cells with
      a 4-connected fluid neighbour);
    * ``stag_xy`` — leading-edge surface cell (minimum x; the s = 0 origin);
    * ``s_grid`` — for every grid cell, the signed arc distance (in cells)
      from the LE cell to its nearest surface cell, positive on the upper
      surface (y > y_stag).  Distances between surface cells are graph
      distances on the 8-connected surface-cell graph (1 for orthogonal
      neighbours, sqrt(2) for diagonal).

    Impact cells are mapped onto this coordinate so the beta curve is a
    function of surface position, independent of how thick the ice has
    grown during the run.
    """
    ny, nx = airfoil.shape
    fluid = ~airfoil
    surf = airfoil & _dilate4(torch.from_numpy(fluid)).numpy()
    ys, xs = np.nonzero(surf)
    n_surf = len(ys)
    sid = -np.ones((ny, nx), dtype=np.int64)
    sid[ys, xs] = np.arange(n_surf)

    # leading edge: minimum x, tie-break closest to domain centreline
    y_cy = ny / 2.0
    order = np.lexsort((np.abs(ys - y_cy), xs))
    stag_id = int(order[0])
    stag = (int(ys[stag_id]), int(xs[stag_id]))

    # Dijkstra over the 8-connected surface-cell graph
    dist = np.full(n_surf, np.inf)
    dist[stag_id] = 0.0
    heap: list[tuple[float, int]] = [(0.0, stag_id)]
    nbrs = [
        (dys, dxs, math.hypot(dys, dxs))
        for dys in (-1, 0, 1)
        for dxs in (-1, 0, 1)
        if (dys, dxs) != (0, 0)
    ]
    coord2id = {(int(ys[i]), int(xs[i])): i for i in range(n_surf)}
    while heap:
        d, u = heappop(heap)
        if d > dist[u]:
            continue
        yu, xu = int(ys[u]), int(xs[u])
        for dys, dxs, w in nbrs:
            v = coord2id.get((yu + dys, xu + dxs))
            if v is None:
                continue
            nd = d + w
            if nd < dist[v] - 1e-12:
                dist[v] = nd
                heappush(heap, (nd, v))

    sign = np.where(ys > stag[0], 1.0, np.where(ys < stag[0], -1.0, 0.0))
    s_surf = dist * sign

    # multi-source BFS: nearest surface cell id for every grid cell
    nearest = -np.ones((ny, nx), dtype=np.int64)
    queue: deque[tuple[int, int]] = deque()
    for i in range(n_surf):
        nearest[ys[i], xs[i]] = i
        queue.append((int(ys[i]), int(xs[i])))
    while queue:
        y, x = queue.popleft()
        i0 = nearest[y, x]
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yn, xn = y + dy, x + dx
            if 0 <= yn < ny and 0 <= xn < nx and nearest[yn, xn] < 0:
                nearest[yn, xn] = i0
                queue.append((yn, xn))

    s_grid = s_surf[nearest]
    return s_grid, stag, surf


def collection_efficiency_curve(
    s_grid: np.ndarray,
    impact_mass: np.ndarray,
    lwc_eff: float,
    v_inf: float,
    dx_phys: float,
    chord_phys: float,
    t_window: float,
    bin_width: float = 1.0,
) -> dict[str, np.ndarray]:
    """Local collection efficiency beta versus surface position.

    ``beta_i = dm_i / (LWC_eff * V_inf * A_cell * t_lat)`` with the cell
    face area ``A_cell = dx_phys**2`` (one-cell-deep slab), where
    ``t_lat = n_window_steps * dt_phys`` is the *lattice* time of the
    window (NOT the accelerated physical time ``n*dt*k``): the impact mass
    is measured in the accelerated run, so the streamtube that carries it
    is ``LWC_eff * V * dx^2 * dt * n_steps``.  Equivalently
    ``= LWC * V * dx^2 * t_phys`` in real units — the acceleration factor
    cancels.  Values are averaged in arc-length bins of ``bin_width`` cells.
    """
    ys, xs = np.nonzero(impact_mass > 0)
    if len(ys) == 0:
        return {"s_over_c": np.array([]), "beta": np.array([]), "n_cells": np.array([])}
    s = s_grid[ys, xs]
    m = impact_mass[ys, xs]
    denom = lwc_eff * v_inf * dx_phys**2 * t_window
    beta_cell = m / denom
    lo = math.floor(s.min() - 0.5)
    hi = math.ceil(s.max() + 0.5)
    edges = np.arange(lo, hi + bin_width, bin_width)
    idx = np.digitize(s, edges) - 1
    nb = len(edges) - 1
    s_out, b_out, n_out = [], [], []
    for b in range(nb):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        s_out.append(float(s[sel].mean()) * dx_phys / chord_phys)
        b_out.append(float(beta_cell[sel].mean()))
        n_out.append(n)
    return {"s_over_c": np.array(s_out), "beta": np.array(b_out), "n_cells": np.array(n_out)}


# ---------------------------------------------------------------------------
# Ice shape metrics
# ---------------------------------------------------------------------------
def _layer_depth(target: np.ndarray, base: np.ndarray, max_layers: int = 64) -> np.ndarray:
    """Layer distance (in cells) of ``target`` cells from the ``base`` set."""
    ny, nx = target.shape
    layers = np.zeros((ny, nx), dtype=np.int32)
    cur = base.copy()
    rem = target & ~base
    for layer in range(1, max_layers + 1):
        if not rem.any():
            break
        grown = np.zeros_like(cur)
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            grown |= np.roll(cur, shift, axis=axis)
        hit = rem & grown
        layers[hit] = layer
        cur |= hit
        rem &= ~hit
    return layers


def ice_shape_metrics(
    airfoil: np.ndarray,
    solid: np.ndarray,
    dx_phys: float,
    chord_phys: float,
    chord_lu: float,
    stag: tuple[int, int],
) -> dict[str, Any]:
    """Quantitative ice-shape descriptors.

    * ice area (cells and m^2, % of chord^2)
    * upper/lower horn height: max layer depth of ice cells above/below the
      stagnation row, in cells, % chord and metres
    * ice streamwise extent (where ice actually grows relative to the LE)
    * upper/lower symmetry of the horns
    """
    ice_only = solid & ~airfoil
    n_ice = int(ice_only.sum())
    layers = _layer_depth(ice_only, airfoil)
    y_stag, x_le = stag
    y_idx, x_idx = np.nonzero(ice_only)
    metrics: dict[str, Any] = {
        "n_ice_cells": n_ice,
        "ice_area_m2": n_ice * dx_phys**2,
        "ice_area_pct_chord2": 100.0 * n_ice * dx_phys**2 / chord_phys**2,
        "x_le": int(x_le),
        "y_stag": int(y_stag),
    }
    if n_ice == 0:
        return metrics
    up = y_idx > y_stag
    lo = y_idx < y_stag
    if up.any():
        h = int(layers[y_idx[up], x_idx[up]].max())
        k = int(np.argmax(layers[y_idx[up], x_idx[up]]))
        metrics["upper_horn_cells"] = h
        metrics["upper_horn_pct_chord"] = 100.0 * h / chord_lu
        metrics["upper_horn_m"] = h * dx_phys
        metrics["upper_horn_xy"] = [int(y_idx[up][k]), int(x_idx[up][k])]
    if lo.any():
        h = int(layers[y_idx[lo], x_idx[lo]].max())
        k = int(np.argmax(layers[y_idx[lo], x_idx[lo]]))
        metrics["lower_horn_cells"] = h
        metrics["lower_horn_pct_chord"] = 100.0 * h / chord_lu
        metrics["lower_horn_m"] = h * dx_phys
        metrics["lower_horn_xy"] = [int(y_idx[lo][k]), int(x_idx[lo][k])]
    if "upper_horn_cells" in metrics and "lower_horn_cells" in metrics:
        hu, hl = metrics["upper_horn_cells"], metrics["lower_horn_cells"]
        metrics["horn_symmetry_pct"] = 100.0 * (hu - hl) / (hu + hl)
    x_off = (x_idx - x_le) / chord_lu
    metrics["ice_x_offset_min"] = float(x_off.min())
    metrics["ice_x_offset_max"] = float(x_off.max())
    metrics["ice_max_layer"] = int(layers.max())
    return metrics


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
class RimeIcingSimulation:
    """Coupled BGK flow + Lagrangian droplet + rime accretion loop."""

    def __init__(self, cfg: IcingConfig, log: Any = print):
        self.cfg = cfg
        self.log = log
        self.dev = torch.device(cfg.device)
        self.nx, self.ny = cfg.nx, cfg.ny
        self.chord = cfg.chord_lu

        self.airfoil = naca0012_mask_2d(
            self.nx,
            self.ny,
            self.chord,
            cfg.aoa_deg,
            cx=self.nx * cfg.cx_frac,
            cy=self.ny * cfg.cy_frac,
            t=cfg.naca_t,
            device=self.dev,
        )
        self.solid = self.airfoil.clone()  # airfoil | ice
        self.opp = OPPOSITE.to(self.dev)

        # whole-step compilation (shared module: single source of truth
        # for the mode whitelist; cudagraph-class modes are rejected there)
        validate_compile_mode(cfg.compile_mode)
        hint = (
            "two guarded graphs: plain flow step + flow step with the "
            "post-stream/pre-bounce force probe"
        )
        if cfg.collision == "cumulant":
            self._step_plain = compile_step(
                self._flow_step_plain_cumulant, cfg.compile_mode, warmup_hint=hint
            )
            self._step_probe = compile_step(
                self._flow_step_probe_cumulant, cfg.compile_mode, warmup_hint=hint
            )
        else:
            self._step_plain = compile_step(self._flow_step_plain, cfg.compile_mode, warmup_hint=hint)
            self._step_probe = compile_step(self._flow_step_probe, cfg.compile_mode, warmup_hint=hint)

        # droplet-phase selection (Phase 2b; defaults reproduce Phase 2a)
        self.use_lagr = cfg.droplet_phase in ("lagrangian", "both")
        self.use_euler = cfg.droplet_phase in ("eulerian", "both")
        if self.use_euler:
            # Eulerian droplet field: alpha (volume fraction) + momentum.
            # Pure-tensor step compiled through the same shared wrapper.
            self._step_euler = compile_step(
                self._euler_step,
                cfg.compile_mode,
                warmup_hint="eulerian droplet advection-relaxation step",
            )
            self.alpha = torch.zeros((self.ny, self.nx), device=self.dev)
            self.mx = torch.zeros_like(self.alpha)
            self.my = torch.zeros_like(self.alpha)
            self.impact_mass_e = torch.zeros_like(self.alpha)  # kg (all run)
            self.impact_e_w0: torch.Tensor | None = None
            self.impact_e_w1: torch.Tensor | None = None
            self._bflux_acc = torch.zeros(4, dtype=torch.float64, device=self.dev)
            self._dep_acc = torch.zeros((), dtype=torch.float64, device=self.dev)
            self._enc_acc = torch.zeros((), dtype=torch.float64, device=self.dev)
            self.aud_e: dict[str, float] = {
                "initial_fill": 0.0,
                "inlet_in": 0.0,
                "lat_in": 0.0,
                "deposited": 0.0,
                "encased": 0.0,
                "outlet_out": 0.0,
                "lat_out": 0.0,
                "airborne": 0.0,
                "closure_error": 0.0,
            }

        rho0 = torch.ones((1, self.ny, self.nx), device=self.dev)
        u0 = torch.full((1, self.ny, self.nx), cfg.u_in, device=self.dev)
        z0 = torch.zeros_like(u0)
        self.f = equilibrium3d(rho0, u0, z0.clone(), z0.clone())
        self.feq_in = self.f.clone()

        # droplet state (empty; lazily concatenated torch tensors)
        self.px = torch.zeros(0, device=self.dev)
        self.py = torch.zeros(0, device=self.dev)
        self.vx = torch.zeros(0, device=self.dev)
        self.vy = torch.zeros(0, device=self.dev)
        self.age = torch.zeros(0, device=self.dev, dtype=torch.int64)

        # accretion + measurement fields
        self.m_w = torch.zeros((self.ny, self.nx), device=self.dev)  # kg per cell
        self.impact_mass = torch.zeros((self.ny, self.nx), device=self.dev)  # kg (all run)

        # geometry anchors
        ys, xs = torch.nonzero(self.airfoil, as_tuple=True)
        self.x_le = int(xs.min().item())
        self.y_le = int(ys[xs.argmin()].item())
        self.kill_x = float(self.x_le + cfg.kill_frac * self.chord)
        self.max_age = int(6.0 * self.nx / max(cfg.u_in, 1e-3))

        # mass audit (kg, float64)
        self.aud = {
            "seeded": 0.0,
            "frozen": 0.0,
            "exited": 0.0,
            "trapped": 0.0,
            "airborne": 0.0,
            "pending": 0.0,
        }
        self.n_impacts = 0
        self._seed_carry = 0.0
        self.history: dict[str, list[Any]] = {
            "step": [], "t_phys": [], "cd": [], "cl": [], "ice_cells": [], "drops_alive": []
        }
        self.cd0 = self.cl0 = None
        self.cd_end = self.cl_end = None

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _flow_step_plain(
        f: torch.Tensor,
        solid: torch.Tensor,
        feq_in: torch.Tensor,
        opp: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        """Whole LBM step (collide, stream, bounce-back, BCs) — pure tensors.

        This is the unit of compilation (compile_utils lesson 1: whole
        step, not hot op).  No step index, no host sync, no data-dependent
        branch — those live in the eager driver (lesson 2).
        """
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f = torch.where(solid[None, None], f[opp], f)
        f[:, :, :, 0] = feq_in[:, :, :, 0]  # inlet equilibrium
        f[:, :, :, -1] = f[:, :, :, -2]  # outlet zero-gradient
        f[:, :, 0, :] = feq_in[:, :, 0, :]  # free-stream lateral walls
        f[:, :, -1, :] = feq_in[:, :, -1, :]
        return f

    @staticmethod
    def _flow_step_probe(
        f: torch.Tensor,
        solid: torch.Tensor,
        feq_in: torch.Tensor,
        opp: torch.Tensor,
        tau: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same step, additionally returning post-stream pre-bounce f_pre.

        Compiled as a *separate variant* (compile_utils lesson 2: one
        variant per branch pattern; the driver picks plain vs probe).
        """
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f_pre = f.clone()  # post-stream, pre-bounce-back (force probe)
        f = torch.where(solid[None, None], f[opp], f)
        f[:, :, :, 0] = feq_in[:, :, :, 0]
        f[:, :, :, -1] = f[:, :, :, -2]
        f[:, :, 0, :] = feq_in[:, :, 0, :]
        f[:, :, -1, :] = feq_in[:, :, -1, :]
        return f, f_pre

    @staticmethod
    def _flow_step_plain_cumulant(
        f: torch.Tensor,
        solid: torch.Tensor,
        feq_in: torch.Tensor,
        opp: torch.Tensor,
        tau: float,
        c_s: float,
    ) -> torch.Tensor:
        """Phase 2b high-Re variant: CUMULANT collision + optional Smag LES.

        Identical chain to :meth:`_flow_step_plain` except the collision;
        the production-proven ``collide_cumulant_d3q19`` kernel with the
        built-in per-cell Smagorinsky tau_eff (``c_s > 0``).  A separate
        compiled variant per compile_utils lesson 2.
        """
        f = collide_cumulant_d3q19(f, tau, C_s=c_s)
        f = stream3d(f)
        f = torch.where(solid[None, None], f[opp], f)
        f[:, :, :, 0] = feq_in[:, :, :, 0]
        f[:, :, :, -1] = f[:, :, :, -2]
        f[:, :, 0, :] = feq_in[:, :, 0, :]
        f[:, :, -1, :] = feq_in[:, :, -1, :]
        return f

    @staticmethod
    def _flow_step_probe_cumulant(
        f: torch.Tensor,
        solid: torch.Tensor,
        feq_in: torch.Tensor,
        opp: torch.Tensor,
        tau: float,
        c_s: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CUMULANT variant of the force-probe step (see above)."""
        f = collide_cumulant_d3q19(f, tau, C_s=c_s)
        f = stream3d(f)
        f_pre = f.clone()
        f = torch.where(solid[None, None], f[opp], f)
        f[:, :, :, 0] = feq_in[:, :, :, 0]
        f[:, :, :, -1] = f[:, :, :, -2]
        f[:, :, 0, :] = feq_in[:, :, 0, :]
        f[:, :, -1, :] = feq_in[:, :, -1, :]
        return f, f_pre

    @staticmethod
    def _euler_step(
        alpha: torch.Tensor,
        mx: torch.Tensor,
        my: torch.Tensor,
        ux: torch.Tensor,
        uy: torch.Tensor,
        solid: torch.Tensor,
        tau_d_lu: float,
        sn_scale: float,
        alpha_in: float,
        u_in: float,
        shadow_min: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One Eulerian droplet-field step (pure tensors; compile unit).

        Donor-cell finite-volume advection of the droplet volume fraction
        ``alpha`` and its momentum ``(mx, my) = alpha * u_d`` on the flow
        grid (FENSAP-ICE style continuity + momentum, one-way coupled),
        followed by shadow-region penalization and a semi-implicit
        exponential drag relaxation toward the local carrier velocity
        ``(ux, uy)``::

            u_d <- u_f + (u_d - u_f) * exp(-f_drag / tau_d_lu)
            f_drag = 1 + 0.15 Re_p^0.687,  Re_p = |u_f - u_d| * sn_scale
            (sn_scale = 0 -> Stokes, f_drag = 1 exactly)

        Solid cells are perfect absorbers (rime: everything sticks): the
        face flux leaving a fluid cell into a solid neighbour is the
        wall-impingement sink, returned on the *donating fluid cell* in
        ``impact`` (lattice volume units per step) — the same convention
        as the Phase 2a Lagrangian deposits, so both phases feed the
        identical freezer.  Boundary flux sums (raw, positive along the
        +axis of each face) come back for the mass audit as
        ``[inlet(x=0), outlet(x=nx-1), bottom(y=0), top(y=ny-1)]``.
        """
        eps = 1e-12

        # Droplet-velocity cap: |u_d| <= VEL_CAP * u_in.  Physically the
        # droplet velocity relaxes toward the carrier field and cannot
        # exceed it by orders of magnitude; numerically, wake cells with
        # alpha just above the shadow threshold can carry a wildly wrong
        # mx/alpha ratio, and the resulting |u_face| > 1 breaks the
        # donor-cell CFL bound, driving alpha negative (the positivity
        # clamp then *creates* mass — the Phase 2b production-scale audit
        # leak).  Capping restores |div u_face| < 1 and with it exact
        # positivity/conservation of the scheme.
        u_cap = 4.0 * u_in

        # -- droplet velocity from momentum (shadow-penalized) ----------
        a_safe = alpha.clamp(min=eps)
        ud_x = mx / a_safe
        ud_y = my / a_safe
        shadow = alpha < shadow_min
        ud_x = torch.where(shadow, ux, ud_x)
        ud_y = torch.where(shadow, uy, ud_y)
        ud_x = ud_x.clamp(-u_cap, u_cap)
        ud_y = ud_y.clamp(-u_cap, u_cap)
        mx = alpha * ud_x
        my = alpha * ud_y

        # ---- x-direction faces (between columns j and j+1) ------------
        # face velocity: centred between fluid cells, one-sided (fluid
        # side) at solid faces so the impingement flux is not halved.
        ul, ur = ud_x[:, :-1], ud_x[:, 1:]
        uf = 0.5 * (ul + ur)
        sl, sr = solid[:, :-1], solid[:, 1:]
        uf = torch.where(sr, ul, uf)
        uf = torch.where(sl, ur, uf)
        out = uf >= 0.0
        al, ar = alpha[:, :-1], alpha[:, 1:]
        fa = torch.where(out, al * uf, ar * uf)
        fmx = torch.where(out, mx[:, :-1] * uf, mx[:, 1:] * uf)
        fmy = torch.where(out, my[:, :-1] * uf, my[:, 1:] * uf)
        # wall-impingement sinks: fluid -> solid in +x / -x
        imp_r = fa * ((~sl) & sr)
        imp_l = fa.neg() * (sl & (~sr))

        # ---- y-direction faces (between rows i and i+1) ---------------
        vl, vr = ud_y[:-1, :], ud_y[1:, :]
        vf = 0.5 * (vl + vr)
        tl, tr = solid[:-1, :], solid[1:, :]
        vf = torch.where(tr, vl, vf)
        vf = torch.where(tl, vr, vf)
        vout = vf >= 0.0
        ga = torch.where(vout, alpha[:-1, :] * vf, alpha[1:, :] * vf)
        gmx = torch.where(vout, mx[:-1, :] * vf, mx[1:, :] * vf)
        gmy = torch.where(vout, my[:-1, :] * vf, my[1:, :] * vf)
        imp_d = ga * ((~tl) & tr)  # fluid(i) -> solid(i+1): donor row i
        imp_u = ga.neg() * (tl & (~tr))  # solid(i) <- fluid(i+1): donor row i+1

        impact = (
            F.pad(imp_r, (0, 1))
            + F.pad(imp_l, (1, 0))
            + F.pad(imp_d, (0, 0, 0, 1))
            + F.pad(imp_u, (0, 0, 1, 0))
        )

        # ---- x boundary faces: Dirichlet inlet / zero-gradient outlet -
        in_a = torch.full_like(al[:, :1], alpha_in * u_in)
        in_mx = torch.full_like(mx[:, :1], alpha_in * u_in * u_in)
        in_my = torch.zeros_like(my[:, :1])
        u_end = ud_x[:, -1:]
        o_end = u_end >= 0.0
        out_a = torch.where(o_end, alpha[:, -1:] * u_end, torch.full_like(u_end, alpha_in) * u_end)
        out_mx = torch.where(o_end, mx[:, -1:] * u_end, torch.full_like(u_end, alpha_in * u_in) * u_end)
        out_my = torch.where(o_end, my[:, -1:] * u_end, torch.zeros_like(u_end))

        flux_a = torch.cat([in_a, fa, out_a], dim=1)
        flux_mx = torch.cat([in_mx, fmx, out_mx], dim=1)
        flux_my = torch.cat([in_my, fmy, out_my], dim=1)
        alpha = alpha - (flux_a[:, 1:] - flux_a[:, :-1])
        mx = mx - (flux_mx[:, 1:] - flux_mx[:, :-1])
        my = my - (flux_my[:, 1:] - flux_my[:, :-1])

        # ---- y boundary faces: free-stream cloud ghost (alpha_in, 0) --
        vb = 0.5 * ud_y[0:1, :]
        b_a = torch.where(vb >= 0, alpha[0:1, :] * vb, torch.full_like(vb, alpha_in) * vb)
        b_mx = torch.where(vb >= 0, mx[0:1, :] * vb, torch.full_like(vb, alpha_in * u_in) * vb)
        b_my = torch.where(vb >= 0, my[0:1, :] * vb, torch.zeros_like(vb))
        vt = 0.5 * ud_y[-1:, :]
        t_a = torch.where(vt >= 0, alpha[-1:, :] * vt, torch.full_like(vt, alpha_in) * vt)
        t_mx = torch.where(vt >= 0, mx[-1:, :] * vt, torch.full_like(vt, alpha_in * u_in) * vt)
        t_my = torch.where(vt >= 0, my[-1:, :] * vt, torch.zeros_like(vt))

        flux_ay = torch.cat([b_a, ga, t_a], dim=0)
        flux_mx_y = torch.cat([b_mx, gmx, t_mx], dim=0)
        flux_my_y = torch.cat([b_my, gmy, t_my], dim=0)
        alpha = alpha - (flux_ay[1:, :] - flux_ay[:-1, :])
        mx = mx - (flux_mx_y[1:, :] - flux_mx_y[:-1, :])
        my = my - (flux_my_y[1:, :] - flux_my_y[:-1, :])

        # ---- positivity + solid void (sinks already accounted) --------
        alpha = torch.clamp(alpha, min=0.0)
        fluid = ~solid
        alpha = torch.where(fluid, alpha, torch.zeros_like(alpha))
        mx = torch.where(fluid, mx, torch.zeros_like(mx))
        my = torch.where(fluid, my, torch.zeros_like(my))

        # ---- drag relaxation (exponential integrator, per cell) -------
        ud_x = mx / alpha.clamp(min=eps)
        ud_y = my / alpha.clamp(min=eps)
        shadow = alpha < shadow_min
        ud_x = torch.where(shadow, ux, ud_x)
        ud_y = torch.where(shadow, uy, ud_y)
        du = torch.sqrt((ux - ud_x) ** 2 + (uy - ud_y) ** 2)
        f_drag = 1.0 + 0.15 * (du * sn_scale).pow(0.687)
        decay = torch.exp(-f_drag / tau_d_lu)
        ud_x = ux + (ud_x - ux) * decay
        ud_y = uy + (ud_y - uy) * decay
        ud_x = ud_x.clamp(-u_cap, u_cap)
        ud_y = ud_y.clamp(-u_cap, u_cap)
        mx = alpha * ud_x
        my = alpha * ud_y

        bflux = torch.stack((in_a.sum(), out_a.sum(), b_a.sum(), t_a.sum()))
        return alpha, mx, my, impact, bflux

    @staticmethod
    def _sample_bilinear(field: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        ny, nx = field.shape
        x0 = torch.floor(xs).long().clamp(0, nx - 2)
        y0 = torch.floor(ys).long().clamp(0, ny - 2)
        fx = xs - x0.float()
        fy = ys - y0.float()
        return (
            field[y0, x0] * (1 - fx) * (1 - fy)
            + field[y0, x0 + 1] * fx * (1 - fy)
            + field[y0 + 1, x0] * (1 - fx) * fy
            + field[y0 + 1, x0 + 1] * fx * fy
        )

    def _flow_step(self, want_force: bool) -> torch.Tensor | None:
        """One flow step via the (optionally compiled) whole-step variants.

        Droplet advection/freezing stay eager on purpose: the droplet
        phase has per-step host syncs (mass audit) and dynamic tensor
        shapes (parcel filtering) that structurally reject torch.compile;
        the flow step is the hot path and is the compiled unit.
        (Phase 2b: the Eulerian droplet step *is* pure tensors and runs
        compiled through the same shared wrapper — see ``_euler_step``.)
        """
        tau = self.cfg.tau_flow
        if self.cfg.collision == "cumulant":
            if want_force:
                f, f_pre = self._step_probe(self.f, self.solid, self.feq_in, self.opp, tau, self.cfg.c_s)
                self.f = f
                return f_pre
            self.f = self._step_plain(self.f, self.solid, self.feq_in, self.opp, tau, self.cfg.c_s)
            return None
        if want_force:
            f, f_pre = self._step_probe(self.f, self.solid, self.feq_in, self.opp, tau)
            self.f = f
            return f_pre
        self.f = self._step_plain(self.f, self.solid, self.feq_in, self.opp, tau)
        return None

    def _force_coeffs(self, f_pre: torch.Tensor) -> tuple[float, float]:
        solid = self.solid
        surf = solid & _dilate4(~solid)
        df = (f_pre - self.f) * surf[None, None].float()
        c = C.to(self.dev).float()
        fx = float((df * c[:, 0].view(19, 1, 1, 1)).sum().item())
        fy = float((df * c[:, 1].view(19, 1, 1, 1)).sum().item())
        q = 0.5 * self.cfg.u_in**2
        return fx / (q * self.chord), fy / (q * self.chord)

    def _seed(self) -> None:
        self._seed_carry += self.cfg.parcels_per_step
        n = int(self._seed_carry)
        self._seed_carry -= n
        if n <= 0:
            return
        gen_dev = self.dev
        px = 0.5 + 0.4 * torch.rand(n, device=gen_dev)
        py = 0.5 + (self.ny - 1.0) * torch.rand(n, device=gen_dev)
        vx = torch.full((n,), self.cfg.u_in, device=gen_dev)
        vy = torch.zeros(n, device=gen_dev)
        ag = torch.zeros(n, device=gen_dev, dtype=torch.int64)
        self.px = torch.cat([self.px, px])
        self.py = torch.cat([self.py, py])
        self.vx = torch.cat([self.vx, vx])
        self.vy = torch.cat([self.vy, vy])
        self.age = torch.cat([self.age, ag])
        self.aud["seeded"] += n * self.cfg.m_parcel

    def _prefill(self, ux: torch.Tensor, uy: torch.Tensor) -> None:
        """Seed the steady-state cloud inventory at t=0 of the exposure.

        The real cloud is already everywhere when the exposure window opens;
        seeding only at the inlet would leave the leading edge starved for
        the first ~x_le/u_in steps.  Steady line density is
        ``parcels_per_step / u_d`` per cell of x (droplet speed, which stays
        ~u_in through the LE deceleration because tau_d >> LE transit), so
        the inventory over [inlet, kill plane] is
        ``parcels_per_step * kill_x / u_in``.  Positions are uniform in x,
        velocities are sampled from the warmed-up flow (local equilibrium),
        and positions inside the airfoil are rejected (that water was never
        there).
        """
        cfg = self.cfg
        n_pre = int(cfg.parcels_per_step * self.kill_x / cfg.u_in)
        if n_pre <= 0:
            return
        px = 0.5 + (self.kill_x - 1.0) * torch.rand(n_pre, device=self.dev)
        py = 0.5 + (self.ny - 1.0) * torch.rand(n_pre, device=self.dev)
        ix = px.floor().long().clamp(0, self.nx - 1)
        iy = py.floor().long().clamp(0, self.ny - 1)
        ok = ~self.solid[iy, ix]
        n_kept = int(ok.sum().item())
        if n_kept == 0:
            return
        px, py = px[ok], py[ok]
        vx = self._sample_bilinear(ux, px, py)
        vy = self._sample_bilinear(uy, px, py)
        ag = torch.zeros(n_kept, device=self.dev, dtype=torch.int64)
        self.px = torch.cat([self.px, px])
        self.py = torch.cat([self.py, py])
        self.vx = torch.cat([self.vx, vx])
        self.vy = torch.cat([self.vy, vy])
        self.age = torch.cat([self.age, ag])
        self.aud["seeded"] += n_kept * cfg.m_parcel
        self.log(
            f"  [icing] prefill cloud: {n_kept} parcels "
            f"(~{cfg.parcels_per_step * self.kill_x / cfg.u_in:.0f} analytic, "
            f"{n_kept * cfg.m_parcel:.3e} kg) in x<[{self.kill_x:.0f}]"
        )

    def _droplet_step(self, ux: torch.Tensor, uy: torch.Tensor) -> None:
        cfg = self.cfg
        self._seed()
        n_sub = cfg.n_substeps
        dt_sub = 1.0 / n_sub
        relax = dt_sub / cfg.tau_d_lu
        sn_scale = cfg.re_p_scale
        m_p = cfg.m_parcel
        solid = self.solid
        nx, ny = self.nx, self.ny
        for _ in range(n_sub):
            if self.px.numel() == 0:
                return
            ufx = self._sample_bilinear(ux, self.px, self.py)
            ufy = self._sample_bilinear(uy, self.px, self.py)
            if sn_scale > 0.0:
                # Schiller-Naumann: per-parcel f_drag = 1 + 0.15 Re_p^0.687
                # (same law as the Eulerian phase; Re_p from the physical
                # unit mapping, invariant across unit systems)
                du = torch.sqrt((ufx - self.vx) ** 2 + (ufy - self.vy) ** 2)
                relax_p = (1.0 + 0.15 * (du * sn_scale).pow(0.687)) * dt_sub / cfg.tau_d_lu
                self.vx += (ufx - self.vx) * relax_p
                self.vy += (ufy - self.vy) * relax_p
            else:
                self.vx += (ufx - self.vx) * relax
                self.vy += (ufy - self.vy) * relax
            ix = self.px.floor().long().clamp(0, nx - 1)
            iy = self.py.floor().long().clamp(0, ny - 1)
            new_px = self.px + self.vx * dt_sub
            new_py = self.py + self.vy * dt_sub
            nix = new_px.floor().long().clamp(0, nx - 1)
            niy = new_py.floor().long().clamp(0, ny - 1)

            moved_into_solid = solid[niy, nix]
            encased = solid[iy, ix] & ~moved_into_solid  # frozen under a droplet

            # deposits: impactors freeze in their last fluid cell; droplets
            # encased by fresh ice freeze where they sit.
            dep_iy = torch.cat([iy[moved_into_solid], iy[encased]])
            dep_ix = torch.cat([ix[moved_into_solid], ix[encased]])
            hit = moved_into_solid | encased
            if hit.any():
                n_hit = int(hit.sum().item())
                self.n_impacts += n_hit
                vals = torch.full((n_hit,), m_p, device=self.dev)
                flat = dep_iy * nx + dep_ix
                self.m_w.view(-1).index_add_(0, flat, vals)
                self.impact_mass.view(-1).index_add_(0, flat, vals)

            left = (
                (new_px >= self.kill_x)
                | (new_px > nx - 0.5)
                | (new_px < -0.5)
                | (new_py < 0.5)
                | (new_py > ny - 0.5)
            )
            n_out = int((left & ~hit).sum().item())
            self.aud["exited"] += n_out * m_p

            self.px = new_px
            self.py = new_py
            keep = ~(hit | left)
            self.age = self.age + 1
            too_old = self.age > self.max_age
            self.aud["trapped"] += int((too_old & keep).sum().item()) * m_p
            keep &= ~too_old
            self.px = self.px[keep]
            self.py = self.py[keep]
            self.vx = self.vx[keep]
            self.vy = self.vy[keep]
            self.age = self.age[keep]

    def _freeze(self) -> int:
        cfg = self.cfg
        freeze = (self.m_w >= cfg.m_cell_ice) & (~self.solid)
        n = int(freeze.sum().item())
        if n > 0:
            self.m_w[freeze] -= cfg.m_cell_ice
            self.solid |= freeze
            self.aud["frozen"] += n * cfg.m_cell_ice
        return n

    # -- Eulerian droplet phase (Phase 2b) ------------------------------
    def _init_eulerian(self, ux: torch.Tensor, uy: torch.Tensor) -> None:
        """Initialise the alpha cloud (Phase 2b analog of ``_prefill``).

        With ``prefill_cloud`` the exposure window opens with the cloud
        everywhere at ``alpha_in`` (velocities at the local warmed-up flow
        value, local equilibrium); otherwise the field starts empty and
        fills from the inlet.  The initial inventory enters the audit as
        ``initial_fill``.
        """
        cfg = self.cfg
        if cfg.prefill_cloud:
            self.alpha = torch.where(
                self.solid,
                torch.zeros_like(self.alpha),
                torch.full_like(self.alpha, cfg.alpha_in),
            )
        else:
            self.alpha = torch.zeros_like(self.alpha)
        self.mx = self.alpha * ux
        self.my = self.alpha * uy
        self.aud_e["initial_fill"] = (
            float(self.alpha.double().sum().item()) * cfg.mass_per_lu3
        )

    def _euler_advance(self, ux: torch.Tensor, uy: torch.Tensor) -> None:
        """One Eulerian droplet step + audit/impact accumulation (eager).

        The compiled unit is :meth:`_euler_step`; everything here is
        bookkeeping: device-side fp64 audit accumulation (no host sync),
        impact-mass ledger in kg, and — when the Eulerian phase is the
        *freezing* phase (``droplet_phase="eulerian"``) — the deposit into
        the shared water ledger ``m_w`` that ``_freeze`` consumes.  In
        ``"both"`` mode the Lagrangian arm drives freezing and the
        Eulerian arm is diagnostic-only (its beta comes from
        ``impact_mass_e``).
        """
        cfg = self.cfg
        self.alpha, self.mx, self.my, imp, bflux = self._step_euler(
            self.alpha,
            self.mx,
            self.my,
            ux,
            uy,
            self.solid,
            cfg.tau_d_lu,
            cfg.re_p_scale,
            cfg.alpha_in,
            cfg.u_in,
            cfg.shadow_alpha_min,
        )
        self._bflux_acc += bflux.double()
        self._dep_acc += imp.double().sum()
        dm = imp * cfg.mass_per_lu3
        self.impact_mass_e += dm
        if cfg.droplet_phase == "eulerian":
            self.m_w += dm

    def _void_encased(self, prev_solid: torch.Tensor) -> None:
        """Remove cloud trapped by fresh ice and audit it (encased).

        Parity with Phase 2a: Lagrangian parcels engulfed by fresh ice
        deposit their mass in place, so the trapped Eulerian cloud water
        also enters the shared freezer ledger ``m_w`` (when the Eulerian
        phase is the freezing phase; in ``"both"`` mode the Lagrangian
        arm owns freezing and its own encased parcels already do this).
        """
        newly = self.solid & ~prev_solid
        enc = self.alpha * newly
        self._enc_acc += enc.double().sum()
        if self.cfg.droplet_phase == "eulerian":
            self.m_w += enc * self.cfg.mass_per_lu3
        keep = (~newly).to(self.alpha.dtype)
        self.alpha = self.alpha * keep
        self.mx = self.mx * keep
        self.my = self.my * keep

    # -- Phase 3: surface stress / edge-velocity sampling -----------------
    def sample_surface_stress(self) -> dict[str, np.ndarray]:
        """Per-surface-cell wall shear and outer velocity (physical units).

        Runs one extra force-probe flow step: the post-stream / pre-bounce
        state ``f_pre`` against the post-bounce-back state gives the
        momentum-exchange force on every surface cell (the same convention
        as :meth:`_force_coeffs`, resolved per cell instead of summed).
        Unit mapping for the 2-D slab: a lattice force corresponds to
        ``F_phys = F_lu * rho_air * dx**4 / dt**2`` per cell face area
        ``dx**2``, i.e. ``tau [Pa] = F_lu * rho_air * (dx/dt)**2``; the
        edge velocity is the mean flow speed over the fluid 4-neighbours,
        ``v_e [m/s] = |u|_lu * dx/dt``.

        Returns CPU numpy fields ``v_e`` (0 on solid), ``tau_t`` and
        ``tau_mag`` (Pa, >= 0).  In ``uniform_flow`` runs the momentum
        exchange vanishes (no boundary layer) and only ``v_e`` is
        meaningful — the analytic htc does not need the shear.
        """
        f_pre = self._flow_step(want_force=True)
        df = f_pre - self.f
        surf = self.solid & _dilate4(~self.solid)
        w = surf[None, None].float()
        c = C.to(self.dev).float()
        fx = (df * w * c[:, 0].view(19, 1, 1, 1)).sum(dim=(0, 1))
        fy = (df * w * c[:, 1].view(19, 1, 1, 1)).sum(dim=(0, 1))

        # edge velocity: mean |u| over fluid 4-neighbours of surface cells
        _, ux3, uy3, _ = macroscopic3d(self.f)
        vmag = torch.sqrt(ux3[0] ** 2 + uy3[0] ** 2)
        fluid = (~self.solid).float()
        k3 = torch.ones((1, 1, 3, 3), device=self.dev)
        n_fluid = F.conv2d(fluid[None, None], k3, padding=1)[0, 0]
        v_sum = F.conv2d((fluid * vmag)[None, None], k3, padding=1)[0, 0]
        with torch.no_grad():
            v_e_cell = torch.where(
                n_fluid > 0, v_sum / n_fluid.clamp_min(1.0), torch.zeros_like(v_sum)
            )
        v_e_cell = v_e_cell * surf.float()

        scale_v = self.cfg.dx_phys / self.cfg.dt_phys
        scale_tau = self.cfg.rho_air * (self.cfg.dx_phys / self.cfg.dt_phys) ** 2
        fx_np = fx.cpu().numpy()
        fy_np = fy.cpu().numpy()
        tau_mag = np.hypot(fx_np, fy_np) * scale_tau

        # tangential component: project the force onto the local surface
        # plane using the discrete distance-from-airfoil gradient as the
        # surface normal (voxel faces are axis aligned; per-cell force
        # direction alone cannot distinguish normal from shear)
        dist = _bfs_distance_cells(self.airfoil.cpu().numpy())
        gy, gx = np.gradient(dist.astype(np.float64))
        gn = np.hypot(gy, gx)
        with np.errstate(invalid="ignore", divide="ignore"):
            ny_, nx_ = np.where(gn > 0, gy / np.maximum(gn, 1e-12), 0.0), np.where(
                gn > 0, gx / np.maximum(gn, 1e-12), 0.0
            )
        fn = fx_np * nx_ + fy_np * ny_
        tau_t = np.abs(np.hypot(fx_np - fn * nx_, fy_np - fn * ny_)) * scale_tau
        tau_t = np.where(np.isfinite(tau_t), tau_t, 0.0)
        return {
            "v_e": (v_e_cell.cpu().numpy() * scale_v),
            "tau_t": tau_t,
            "tau_mag": tau_mag,
        }

    # -- main loop -------------------------------------------------------
    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        log = self.log
        torch.manual_seed(cfg.seed)

        log(
            f"  [icing] chord={self.chord:.0f} lu AoA={cfg.aoa_deg} deg "
            f"airfoil={int(self.airfoil.sum().item())} cells LE=({self.x_le},{self.y_le})"
        )
        log(
            f"  [icing] dx={cfg.dx_phys:.4e} m dt={cfg.dt_phys:.4e} s "
            f"St={cfg.stokes:.4f} tau_d={cfg.tau_d_lu:.1f} steps "
            f"Re_lu={cfg.re_lu:.0f}"
        )
        log(
            f"  [icing] k(accel)={cfg.lwc_accel:.3e} "
            f"phys.drops/step={cfg.droplets_per_step:.3e} "
            f"parcels/step={cfg.parcels_per_step:.2f} (x{cfg.effective_parcel_multiplier}) "
            f"t_equiv={cfg.t_equiv:.2f} s (target {cfg.t_exposure:.0f} s, "
            f"realtime would need {cfg.t_exposure / cfg.dt_phys:.2e} steps)"
        )

        # ---- flow warmup on the clean airfoil ----
        beta_w0_log, beta_w1_log = cfg.beta_window_bounds
        log(
            f"  [icing] beta window: mode={cfg.beta_window_mode} "
            f"steps [{beta_w0_log}, {beta_w1_log}) "
            f"({'pre-ice reference geometry' if cfg.beta_window_mode == 'clean' else 'iced geometry (legacy)'})"
        )
        if not cfg.uniform_flow:
            for _ in range(cfg.warmup_steps):
                last = _ == cfg.warmup_steps - 1
                f_pre = self._flow_step(want_force=last)
                if last and f_pre is not None:
                    self.cd0, self.cl0 = self._force_coeffs(f_pre)
            if self.cd0 is not None:
                log(f"  [icing] warmup done: cd0={self.cd0:.5f} cl0={self.cl0:.5f}")

        ux_c = torch.full((self.ny, self.nx), cfg.u_in, device=self.dev)
        uy_c = torch.zeros((self.ny, self.nx), device=self.dev)
        # beta window (task #84 fix 1): "clean" = early pre-ice window on the
        # reference geometry (LWC-invariant beta), "trailing" = Phase 2a/2b
        # legacy window on the iced geometry.
        beta_w0, beta_w1 = cfg.beta_window_bounds
        impact_w0: torch.Tensor | None = None
        impact_w1: torch.Tensor | None = None

        if cfg.prefill_cloud and not cfg.disable_droplets and self.use_lagr:
            if cfg.uniform_flow:
                ux0, uy0 = ux_c, uy_c
            else:
                _, ux3, uy3, _ = macroscopic3d(self.f)
                ux0, uy0 = ux3[0], uy3[0]
            self._prefill(ux0, uy0)
        if self.use_euler and not cfg.disable_droplets:
            if cfg.uniform_flow:
                ux0, uy0 = ux_c, uy_c
            else:
                _, ux3, uy3, _ = macroscopic3d(self.f)
                ux0, uy0 = ux3[0], uy3[0]
            self._init_eulerian(ux0, uy0)
            self.log(
                f"  [icing-e] eulerian cloud: alpha_in={cfg.alpha_in:.3e} "
                f"shadow_min={cfg.shadow_alpha_min:.3e} "
                f"drag={cfg.drag_law} (re_p_scale={cfg.re_p_scale:.3e}) "
                f"init_fill={self.aud_e['initial_fill']:.3e} kg"
            )

        for step in range(1, cfg.steps + 1):
            want_force = (
                step == 1 or step == cfg.steps or step % cfg.log_every == 0
                or step == beta_w0 or step == beta_w1
            )
            if cfg.uniform_flow:
                f_pre = None
            else:
                f_pre = self._flow_step(want_force=want_force)
            if not cfg.disable_droplets:
                if cfg.uniform_flow:
                    ux, uy = ux_c, uy_c
                else:
                    _, ux3, uy3, _ = macroscopic3d(self.f)
                    ux, uy = ux3[0], uy3[0]
                if self.use_lagr:
                    self._droplet_step(ux, uy)
                if self.use_euler:
                    self._euler_advance(ux, uy)
                prev_solid = self.solid.clone() if self.use_euler else None
                if cfg.freeze_in_run:  # Phase 3: False = measure-only shots
                    self._freeze()
                if self.use_euler:
                    self._void_encased(prev_solid)

            if impact_w0 is None and step >= beta_w0:
                impact_w0 = self.impact_mass.clone()
            if self.use_euler and self.impact_e_w0 is None and step >= beta_w0:
                self.impact_e_w0 = self.impact_mass_e.clone()
            # clean mode ends the ledger window before the run does: the
            # run keeps freezing (ice shape) while beta stays pre-ice.
            if impact_w1 is None and step >= beta_w1:
                impact_w1 = self.impact_mass.clone()
            if self.use_euler and self.impact_e_w1 is None and step >= beta_w1:
                self.impact_e_w1 = self.impact_mass_e.clone()

            if want_force and f_pre is not None:
                cd, cl = self._force_coeffs(f_pre)
                self.cd_end, self.cl_end = cd, cl
            n_ice = int(self.solid.sum().item()) - int(self.airfoil.sum().item())
            t_phys = step * cfg.dt_phys * cfg.lwc_accel
            self.history["step"].append(step)
            self.history["t_phys"].append(t_phys)
            self.history["cd"].append(self.cd_end)
            self.history["cl"].append(self.cl_end)
            self.history["ice_cells"].append(n_ice)
            self.history["drops_alive"].append(int(self.px.numel()))
            if step % cfg.log_every == 0 or step == cfg.steps:
                cd_s = "n/a" if self.cd_end is None else f"{self.cd_end:.5f}"
                log(
                    f"  [icing] step {step:5d}/{cfg.steps} t_eq={t_phys:8.2f} s "
                    f"cd={cd_s} ice={n_ice:4d} drops={int(self.px.numel()):7d} "
                    f"impacts={self.n_impacts}"
                )
                if self.use_euler:
                    log(
                        f"  [icing-e] step {step:5d} deposited="
                        f"{float(self._dep_acc.item()) * cfg.mass_per_lu3:.4e} kg"
                    )

        # ---- final audit ----
        self.aud["airborne"] = float(self.px.numel()) * cfg.m_parcel
        self.aud["pending"] = float(self.m_w.double().sum().item())
        accounted = (
            self.aud["frozen"] + self.aud["exited"] + self.aud["trapped"]
            + self.aud["airborne"] + self.aud["pending"]
        )
        err = abs(self.aud["seeded"] - accounted) / self.aud["seeded"] if self.aud["seeded"] else 0.0
        self.aud["closure_error"] = err

        # ---- Eulerian alpha-field audit (Phase 2b) ----
        euler_result: dict[str, Any] | None = None
        if self.use_euler and not cfg.disable_droplets:
            mp_lu3 = cfg.mass_per_lu3
            inlet_raw, outlet_raw, bottom_raw, top_raw = self._bflux_acc.tolist()
            # face fluxes are positive along +axis: bottom(+y)=in, top(+y)=out
            self.aud_e["inlet_in"] = max(inlet_raw, 0.0) * mp_lu3
            outlet_in = max(-outlet_raw, 0.0) * mp_lu3
            self.aud_e["outlet_out"] = max(outlet_raw, 0.0) * mp_lu3
            lat_in = (max(bottom_raw, 0.0) + max(-top_raw, 0.0)) * mp_lu3
            lat_out = (max(-bottom_raw, 0.0) + max(top_raw, 0.0)) * mp_lu3
            self.aud_e["lat_in"] = lat_in
            self.aud_e["lat_out"] = lat_out
            self.aud_e["deposited"] = float(self._dep_acc.item()) * mp_lu3
            self.aud_e["encased"] = float(self._enc_acc.item()) * mp_lu3
            self.aud_e["airborne"] = float(self.alpha.double().sum().item()) * mp_lu3
            inflow = (
                self.aud_e["initial_fill"] + self.aud_e["inlet_in"]
                + outlet_in + self.aud_e["lat_in"]
            )
            outflow = (
                self.aud_e["deposited"] + self.aud_e["encased"]
                + self.aud_e["outlet_out"] + self.aud_e["lat_out"]
                + self.aud_e["airborne"]
            )
            self.aud_e["outlet_in"] = outlet_in
            self.aud_e["closure_error"] = (
                abs(inflow - outflow) / inflow if inflow > 0.0 else 0.0
            )

        # ---- beta + metrics ----
        airfoil_np = self.airfoil.cpu().numpy()
        solid_np = self.solid.cpu().numpy()
        s_grid, stag, _surf = surface_arc_length(airfoil_np)
        dm = self.impact_mass.clone()
        if impact_w1 is not None:
            dm = impact_w1.clone()
        if impact_w0 is not None:
            dm = dm - impact_w0
        # lattice time of the beta window (acceleration cancels, see docstring)
        t_win = (beta_w1 - beta_w0) * cfg.dt_phys
        beta = collection_efficiency_curve(
            s_grid, dm.cpu().numpy(), cfg.lwc_eff, cfg.v_inf, cfg.dx_phys,
            cfg.chord_phys, t_win,
        )
        metrics = ice_shape_metrics(
            airfoil_np, solid_np, cfg.dx_phys, cfg.chord_phys, cfg.chord_lu, stag
        )
        if euler_result is not None or (self.use_euler and not cfg.disable_droplets):
            dm_e = self.impact_mass_e.clone()
            if self.impact_e_w1 is not None:
                dm_e = self.impact_e_w1.clone()
            if self.impact_e_w0 is not None:
                dm_e = dm_e - self.impact_e_w0
            beta_e = collection_efficiency_curve(
                s_grid, dm_e.cpu().numpy(), cfg.lwc_eff, cfg.v_inf, cfg.dx_phys,
                cfg.chord_phys, t_win,
            )
            beta_e_grid = (dm_e / (cfg.lwc_eff * cfg.v_inf * cfg.dx_phys**2 * t_win)).cpu().numpy()
            euler_result = {
                "alpha": self.alpha.cpu().numpy(),
                "impact_mass": self.impact_mass_e.cpu().numpy(),
                "beta": beta_e,
                "beta_grid": beta_e_grid,
                "audit": dict(self.aud_e),
                "alpha_in": cfg.alpha_in,
            }
        impact_np = self.impact_mass.cpu().numpy()
        if impact_np.any():
            metrics["max_impact_x_frac"] = float(
                (np.nonzero(impact_np)[1].max() - self.x_le) / cfg.chord_lu
            )
        cd_drift = None
        if self.cd0 is not None and self.cd_end is not None and self.cd0 != 0:
            cd_drift = 100.0 * (self.cd_end - self.cd0) / self.cd0

        return {
            "config": cfg,
            "mapping": cfg.mapping_report(),
            "airfoil": airfoil_np,
            "solid": solid_np,
            "ice_only": solid_np & ~airfoil_np,
            "m_w": self.m_w.cpu().numpy(),
            "impact_mass": impact_np,
            "s_grid": s_grid,
            "stag": stag,
            "kill_x": self.kill_x,
            "beta": beta,
            "beta_grid": (dm / (cfg.lwc_eff * cfg.v_inf * cfg.dx_phys**2 * t_win)).cpu().numpy(),
            "metrics": metrics,
            "audit": dict(self.aud),
            "history": self.history,
            "cd0": self.cd0,
            "cl0": self.cl0,
            "cd_end": self.cd_end,
            "cl_end": self.cl_end,
            "cd_drift_pct": cd_drift,
            "n_impacts": self.n_impacts,
            "eulerian": euler_result,
        }


def mass_audit_report(result: dict[str, Any]) -> str:
    a = result["audit"]
    lines = [
        f"seeded  = {a['seeded']:.6e} kg",
        f"frozen  = {a['frozen']:.6e} kg",
        f"exited  = {a['exited']:.6e} kg",
        f"trapped = {a['trapped']:.6e} kg",
        f"airborne= {a['airborne']:.6e} kg",
        f"pending = {a['pending']:.6e} kg",
        f"closure error = {a['closure_error'] * 100:.4f} %",
    ]
    return "\n".join(lines)


def eulerian_mass_audit_report(result: dict[str, Any]) -> str:
    """Human-readable alpha-field mass audit (Phase 2b Eulerian phase)."""
    e = result.get("eulerian")
    if e is None:
        return "(eulerian phase not active)"
    a = e["audit"]
    lines = [
        f"initial_fill = {a['initial_fill']:.6e} kg",
        f"inlet_in     = {a['inlet_in']:.6e} kg",
        f"lat_in       = {a['lat_in']:.6e} kg",
        f"deposited    = {a['deposited']:.6e} kg",
        f"encased      = {a['encased']:.6e} kg",
        f"outlet_out   = {a['outlet_out']:.6e} kg",
        f"lat_out      = {a['lat_out']:.6e} kg",
        f"airborne     = {a['airborne']:.6e} kg",
        f"closure error = {a['closure_error'] * 100:.4f} %",
    ]
    return "\n".join(lines)


def run_rime_icing(cfg: IcingConfig, log: Any = print) -> dict[str, Any]:
    """Convenience wrapper: build and run the simulation, return results."""
    return RimeIcingSimulation(cfg, log=log).run()


# ---------------------------------------------------------------------------
# Phase 3: Messinger surface energy balance + thin-film runback (glaze)
# ---------------------------------------------------------------------------
def saturation_vapor_pressure_pa(t_c: float, over_ice: bool | None = None) -> float:
    """Saturation vapour pressure [Pa] (Magnus form, e in Pa, T in C).

    Liquid-water branch (Alduchov & Eskridge 1996, a = 17.625,
    b = 243.04) with an ice branch below 0 C (a = 22.46, b = 272.62);
    both meet at 611 Pa at 0 C.  ``over_ice=None`` auto-selects (ice for
    T < 0).  Checks: e_w(-10) = 286.8 Pa, e_i(-10) = 259.9 Pa,
    e_w(20) = 2334 Pa.
    """
    if over_ice is None:
        over_ice = t_c < 0.0
    if over_ice and t_c < 0.0:
        a, b = 22.46, 272.62
    else:
        a, b = 17.625, 243.04
    return 610.94 * math.exp(a * t_c / (b + t_c))


def analytic_htc_w_m2k(
    s_m: np.ndarray, v_e_m: np.ndarray, cfg: IcingConfig
) -> np.ndarray:
    """Analytic convective heat-transfer coefficient along the arc [W/m^2 K].

    * stagnation line: Frossling cylinder correlation
      ``h = 1.14 k/d Re_d^0.5 Pr^0.4`` with the NACA leading-edge
      effective diameter;
    * laminar decay away from the leading edge via the classic
      cylinder-equivalent form ``h_stag * (d/(d+2s))^0.5`` (asymptotic
      s^-0.5);
    * turbulent flat plate with running length (Chilton-Colburn,
      ``Nu = 0.0287 Re_s^0.8 Pr^(1/3)``), capped by the stagnation value,
      taking over downstream where it exceeds the laminar decay.

    This is the simplified Smith-Spalding-class envelope used by the
    ``htc_mode="analytic"`` default (the shear-based mode is the
    Reynolds-analogy alternative); the envelope decays monotonically
    from the stagnation value, which is what sets the freezing fraction.
    """
    s = np.maximum(np.abs(np.asarray(s_m, dtype=np.float64)), 1e-6)
    d = cfg.le_diameter_eff
    rho, mu, k, pr = cfg.rho_air, cfg.mu_air, cfg.k_air, cfg.prandtl_air
    re_d = rho * cfg.v_inf * d / mu
    h_stag = 1.14 * k / d * re_d**0.5 * pr**0.4
    h_lam = h_stag * np.sqrt(d / (d + 2.0 * s))
    re_s = rho * np.maximum(np.asarray(v_e_m, dtype=np.float64), 1e-3) * s / mu
    h_turb = 0.0287 * k * re_s**0.8 * pr ** (1.0 / 3.0) / s
    return np.maximum(h_lam, np.minimum(h_turb, h_stag))


def analytic_tau_pa(s_m: np.ndarray, v_e_m: np.ndarray, cfg: IcingConfig) -> np.ndarray:
    """Analytic wall shear along the arc [Pa] for the film diagnostic.

    Local turbulent flat-plate value ``tau = 0.0592 Re_s^(-1/5) rho V_e^2/2``
    with a stagnation taper ``(1 - exp(-2|s|/d_le))`` so the shear vanishes
    at the stagnation line (Hiemenz behaviour) and reaches the flat-plate
    value within ~half a leading-edge diameter.  Used with the default
    ``htc_mode="analytic"``: the sampled momentum-exchange shear
    (:meth:`RimeIcingSimulation.sample_surface_stress`) is exact for the
    *lattice* Reynolds number the flow actually runs at, not the physical
    one, and would bias the Myers film thickness.
    """
    s = np.maximum(np.abs(np.asarray(s_m, dtype=np.float64)), 1e-6)
    v = np.maximum(np.asarray(v_e_m, dtype=np.float64), 1e-3)
    rho, mu = cfg.rho_air, cfg.mu_air
    re_s = rho * v * s / mu
    tau_fp = 0.0592 * re_s ** (-0.2) * 0.5 * rho * v**2
    taper = 1.0 - np.exp(-2.0 * s / max(cfg.le_diameter_eff, 1e-9))
    return tau_fp * taper


def messinger_panel_fluxes(
    cfg: IcingConfig,
    m_imp: float,
    m_in: float,
    t_in_c: float,
    h: float,
    area: float,
    v_e: float,
) -> dict[str, Any]:
    """Messinger control volume on one surface panel (all fluxes kg/s).

    Energy balance on the surface film at temperature ``T_s`` (water
    reference 0 C, steady state, per unit time)::

        m_ice L_f = h A (T_s - T_rec) + m_evap L_v
                    - m_imp cp_w (T_inf - T_s) - m_in cp_w (T_in - T_s)
                    - m_imp V_inf^2 / 2

    with ``T_rec = T_inf + r V_e^2/(2 cp_air)`` (r = sqrt(Pr)) and the
    Chilton-Colburn / Lewis evaporation mass flux::

        m_evap = (h A / cp_air) * 0.622 * max(0, e_sat(T_s) - e_inf) / p

    (surface branch over ice below 0 C, ambient over the liquid-water
    curve at the cloud relative humidity).  The balance is solved in the
    classical two-regime way: ``m_ice(T_s)`` is strictly increasing in
    ``T_s`` while the available water ``m_w - m_evap(T_s)`` decreases, so

    * ``m_ice(0) > avail(0)``  -> rime: n_f = 1, all available water
      freezes, T_s < 0 from the bisection root;
    * ``0 <= m_ice(0) <= avail(0)`` -> glaze: T_s = 0 C, n_f < 1, the
      unfrozen remainder runs back downstream;
    * ``m_ice(0) < 0`` -> warm: n_f = 0, T_s > 0, everything available
      runs back.

    Returns a dict with ``t_s_c, m_ice, m_evap, m_out, n_f, regime,
    residual_w``; ``n_f = m_ice / (m_w - m_evap)`` is the freezing
    fraction of the water actually available for freezing/runback.
    """
    r_rec = cfg.recovery_factor_eff
    t_inf = cfg.t_static_c
    t_rec = t_inf + r_rec * v_e**2 / (2.0 * cfg.cp_air)
    e_inf = cfg.rh * saturation_vapor_pressure_pa(t_inf, over_ice=False)

    def m_evap_of(ts: float) -> float:
        if not cfg.evap_enabled:
            return 0.0
        de = saturation_vapor_pressure_pa(ts) - e_inf
        if de <= 0.0:
            return 0.0
        return (h * area / cfg.cp_air) * 0.622 * de / cfg.p_static

    def m_ice_of(ts: float) -> float:
        return (
            h * area * (ts - t_rec)
            + m_evap_of(ts) * cfg.l_vapor
            - m_imp * cfg.cp_water * (t_inf - ts)
            - m_in * cfg.cp_water * (t_in_c - ts)
            - m_imp * cfg.v_inf**2 / 2.0
        ) / cfg.l_fusion

    def bisect(f: Any, lo: float, hi: float, it: int = 100) -> float:
        flo = f(lo)
        for _ in range(it):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if (fm > 0.0) == (flo > 0.0):
                lo, flo = mid, fm
            else:
                hi = mid
        return 0.5 * (lo + hi)

    m_w = m_imp + m_in
    if m_w <= 0.0:
        return {
            "t_s_c": t_rec,
            "m_ice": 0.0,
            "m_evap": 0.0,
            "m_out": 0.0,
            "n_f": 0.0,
            "regime": "dry",
            "residual_w": 0.0,
        }

    m0_ice = m_ice_of(0.0)
    m0_evap = m_evap_of(0.0)
    avail0 = m_w - m0_evap

    if avail0 <= 0.0:
        # evaporation alone consumes the inflow (nothing freezes, nothing
        # runs back); n_f := 0 by the available-water convention
        t_s, m_ice, m_evap, m_out, n_f, regime = 0.0, 0.0, min(m_w, m0_evap), 0.0, 0.0, "evap"
    elif m0_ice > avail0:
        # rime: freeze everything available, surface below 0 C
        t_s = bisect(lambda ts: m_ice_of(ts) - (m_w - m_evap_of(ts)), t_inf - 5.0, 0.0)
        m_evap = m_evap_of(t_s)
        m_ice = max(m_w - m_evap, 0.0)
        m_out, n_f, regime = 0.0, 1.0, "rime"
    elif m0_ice < 0.0:
        # warm: no freezing, film above 0 C, all available water runs back
        hi = max(t_inf, 0.0) + 10.0
        while m_ice_of(hi) <= 0.0 and hi < 400.0:
            hi *= 2.0
        t_s = bisect(m_ice_of, 0.0, hi)
        m_evap = m_evap_of(t_s)
        m_ice, n_f, regime = 0.0, 0.0, "warm"
        m_out = max(m_w - m_evap, 0.0)
    else:
        # glaze: T_s pinned at 0 C, partial freezing, remainder runs back
        t_s, m_ice, m_evap = 0.0, m0_ice, m0_evap
        m_out = avail0 - m_ice
        n_f = m_ice / avail0 if avail0 > 0.0 else 0.0
        regime = "glaze"

    residual = m_ice * cfg.l_fusion - (
        h * area * (t_s - t_rec)
        + m_evap * cfg.l_vapor
        - m_imp * cfg.cp_water * (t_inf - t_s)
        - m_in * cfg.cp_water * (t_in_c - t_s)
        - m_imp * cfg.v_inf**2 / 2.0
    )
    return {
        "t_s_c": t_s,
        "m_ice": m_ice,
        "m_evap": m_evap,
        "m_out": m_out,
        "n_f": n_f,
        "regime": regime,
        "residual_w": residual,
    }


def build_surface_panels(
    cfg: IcingConfig,
    s_grid: np.ndarray,
    impact_mass: np.ndarray,
    solid: np.ndarray,
    stress: dict[str, np.ndarray],
    t_window: float,
    lwc_accel: float,
) -> dict[str, Any]:
    """Bin the surface into arc-length panels with impingement flux + htc.

    Three cell families are mapped onto the arc coordinate ``s_grid``
    (cells, signed from the stagnation cell, positive on the upper
    surface): the *impact ledger* cells (fluid), the *deposit targets*
    (fluid cells adjacent to the current solid, plus the impact cells)
    and the *surface cells* (solid boundary, carrying the sampled edge
    velocity and wall shear).  Panel id = round(s / glaze_panel_cells).

    The physical impingement rate per panel is the ledger mass divided by
    the lattice window time *and* the LWC acceleration — the acceleration
    cancels exactly, so ``m_imp`` is the real-LWC flux.  ``beta`` is the
    diagnostic normalisation against ``LWC V A`` with the panel area from
    its deposit-cell count.
    """
    w = cfg.glaze_panel_cells
    dx = cfg.dx_phys
    ny, nx = solid.shape

    imp_y, imp_x = np.nonzero(impact_mass > 0)
    m_imp_cells = impact_mass[imp_y, imp_x] if len(imp_y) else np.zeros(0)
    p_imp = np.rint(s_grid[imp_y, imp_x] / w).astype(int) if len(imp_y) else np.zeros(0, int)

    bf = ~solid & _dilate4(torch.from_numpy(solid)).numpy()
    sf = solid & _dilate4(torch.from_numpy(~solid)).numpy()

    # deposit targets: boundary fluid cells union impact cells (unique set —
    # a cell that is both an impact cell and boundary-adjacent must appear
    # exactly once or the per-cell credit/density double-counts)
    dep_mask = bf.copy()
    dep_mask[imp_y, imp_x] = True
    dep_y, dep_x = np.nonzero(dep_mask)
    p_dep = np.rint(s_grid[dep_y, dep_x] / w).astype(int) if len(dep_y) else np.zeros(0, int)

    sf_y, sf_x = np.nonzero(sf)
    p_sf = np.rint(s_grid[sf_y, sf_x] / w).astype(int) if len(sf_y) else np.zeros(0, int)

    ids = np.unique(np.concatenate([p_imp, p_dep, p_sf]).astype(int))
    n = len(ids)
    empty: dict[str, Any] = {
        "s_m": np.zeros(0), "A_m2": np.zeros(0), "m_imp_kg_s": np.zeros(0),
        "beta": np.zeros(0), "h": np.zeros(0), "v_e": np.zeros(0),
        "tau_t": np.zeros(0), "dep_y": np.zeros(0, int), "dep_x": np.zeros(0, int),
        "dep_p": np.zeros(0, int), "dep_w": np.zeros(0), "n_panels": 0,
    }
    if n == 0:
        return empty

    ii = np.searchsorted(ids, p_imp)
    di = np.searchsorted(ids, p_dep)
    si = np.searchsorted(ids, p_sf)

    m_ledger = np.zeros(n)
    np.add.at(m_ledger, ii, m_imp_cells)
    n_dep = np.zeros(n, dtype=int)
    np.add.at(n_dep, di, 1)
    s_sum = np.zeros(n)
    np.add.at(s_sum, di, s_grid[dep_y, dep_x] if len(dep_y) else np.zeros(0))
    n_sf = np.zeros(n, dtype=int)
    np.add.at(n_sf, si, 1)
    v_sum = np.zeros(n)
    if len(sf_y):
        np.add.at(v_sum, si, stress["v_e"][sf_y, sf_x])
    tau_sum = np.zeros(n)
    if len(sf_y):
        np.add.at(tau_sum, si, stress["tau_t"][sf_y, sf_x])

    # the deposit set (unique union of boundary-fluid and impact cells)
    # already contains every impact cell, so it is used directly — no
    # second listing of the impact cells, which would double the per-cell
    # ice density in the driver's np.add.at credit
    s_m = np.where(n_dep > 0, s_sum / np.maximum(n_dep, 1), ids * w) * dx
    area = n_dep * dx * dx
    m_imp_kgs = m_ledger / (t_window * lwc_accel) if (t_window * lwc_accel) > 0 else np.zeros(n)
    v_e = np.where(n_sf > 0, v_sum / np.maximum(n_sf, 1), cfg.v_inf)
    tau_sampled = np.where(n_sf > 0, tau_sum / np.maximum(n_sf, 1), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(area > 0, m_imp_kgs / (cfg.lwc * cfg.v_inf * area), 0.0)

    h = analytic_htc_w_m2k(s_m, v_e, cfg)
    tau_t = analytic_tau_pa(s_m, v_e, cfg)
    if cfg.htc_mode == "shear":
        # Reynolds analogy St = (c_f/2) Pr^(-2/3) -> h = tau cp Pr^(-2/3)/V,
        # with the Frossling stagnation value as the floor over the
        # leading-edge region (tau_w -> 0 at the stagnation line breaks
        # the analogy there).  The sampled shear belongs to the lattice
        # Reynolds number, so this mode is a sensitivity check, not the
        # default.
        tau_t = tau_sampled
        h_shear = tau_t * cfg.cp_air * cfg.prandtl_air ** (-2.0 / 3.0) / np.maximum(v_e, 1e-3)
        floor = np.where(np.abs(s_m) <= 0.5 * cfg.le_diameter_eff, h, 0.0)
        h = np.maximum(h_shear, floor)

    # deposit weights: impact-share where the panel collects water,
    # uniform otherwise (runback ice accretes over the whole panel).
    # The deposit set contains every impact cell exactly once, so the
    # per-panel impact mass held by deposit cells equals the ledger and
    # the weights sum to one per panel.
    pj = np.searchsorted(ids, p_dep)
    dep_imp = np.where(
        impact_mass[dep_y, dep_x] > 0.0,
        impact_mass[dep_y, dep_x],
        0.0,
    )
    dep_imp_sum = np.zeros(n)
    np.add.at(dep_imp_sum, pj, dep_imp)
    wts = np.zeros(len(dep_y))
    ok = dep_imp_sum[pj] > 0.0
    wts[ok] = dep_imp[ok] / dep_imp_sum[pj][ok]
    unif = ~ok
    wts[unif] = 1.0 / n_dep[pj][unif]

    return {
        "s_m": s_m,
        "A_m2": area,
        "m_imp_kg_s": m_imp_kgs,
        "beta": beta,
        "h": h,
        "v_e": v_e,
        "tau_t": tau_t,
        "tau_sampled_pa": tau_sampled,
        "dep_y": dep_y.astype(int),
        "dep_x": dep_x.astype(int),
        "dep_p": pj.astype(int),
        "dep_w": wts,
        "n_panels": n,
    }


def solve_glaze_surface(cfg: IcingConfig, panels: dict[str, Any], dt: float) -> dict[str, Any]:
    """March the Messinger balance + runback from the stagnation panel.

    The panels are ordered away from the stagnation panel (minimum |s|);
    each panel sees the runback outflow of its upstream neighbour at the
    upstream film temperature (0 C in the glaze regime).  The stagnation
    panel's unfrozen water splits equally between the upper and lower
    surfaces — the mass-flux (LEWICE-style overflow) runback model.  The
    film thickness diagnostic is the steady shear-driven Myers film
    ``h_f = sqrt(2 mu_w q / tau_t)`` with the per-unit-span flux
    ``q = m_out / (rho_w dx)``.

    Mass audit (quasi-steady film, zero storage)::

        impacted = frozen + evaporated + runback_off_surface
    """
    n = int(panels.get("n_panels", 0))
    if n == 0:
        return {
            "s_m": np.zeros(0), "n_f": np.zeros(0), "t_s_c": np.zeros(0),
            "m_ice_kg": np.zeros(0), "m_evap_kg": np.zeros(0),
            "m_runback_out_kg": np.zeros(0), "m_runback_in_kg": np.zeros(0),
            "thickness_m": np.zeros(0), "film_m": np.zeros(0),
            "rho_ice": np.zeros(0), "regime": [],
            "audit": {"impacted": 0.0, "frozen": 0.0, "evaporated": 0.0,
                      "runback_out": 0.0, "closure_error": 0.0},
        }
    s_m = panels["s_m"]
    m_imp = panels["m_imp_kg_s"]
    h, area, v_e, tau_t = panels["h"], panels["A_m2"], panels["v_e"], panels["tau_t"]

    t_s = np.zeros(n)
    n_f = np.zeros(n)
    m_ice = np.zeros(n)
    m_evap = np.zeros(n)
    m_out = np.zeros(n)
    m_in = np.zeros(n)
    regimes: list[str] = []
    residuals = np.zeros(n)

    i_stag = int(np.argmin(np.abs(s_m)))
    up = np.array([i for i in range(n) if s_m[i] > 0.0], dtype=int)
    lo = np.array([i for i in range(n) if s_m[i] < 0.0], dtype=int)
    up = up[np.argsort(s_m[up])] if len(up) else up
    lo = lo[np.argsort(-s_m[lo])] if len(lo) else lo

    r0 = messinger_panel_fluxes(cfg, m_imp[i_stag], 0.0, cfg.t_static_c, h[i_stag], area[i_stag], v_e[i_stag])
    t_s[i_stag], n_f[i_stag] = r0["t_s_c"], r0["n_f"]
    m_ice[i_stag], m_evap[i_stag], m_out[i_stag] = r0["m_ice"], r0["m_evap"], r0["m_out"]
    regimes.append(r0["regime"])
    residuals[i_stag] = r0["residual_w"]

    carry_up = 0.5 * r0["m_out"] if len(up) else r0["m_out"]
    carry_lo = 0.5 * r0["m_out"] if len(lo) else 0.0
    t_up, t_lo = r0["t_s_c"], r0["t_s_c"]
    for side, order, carry, t_in in ((0, up, carry_up, t_up), (1, lo, carry_lo, t_lo)):
        c, tc = carry, t_in
        for i in order:
            ri = messinger_panel_fluxes(cfg, m_imp[i], c, tc, h[i], area[i], v_e[i])
            t_s[i], n_f[i] = ri["t_s_c"], ri["n_f"]
            m_ice[i], m_evap[i], m_out[i], m_in[i] = ri["m_ice"], ri["m_evap"], ri["m_out"], c
            regimes.append(ri["regime"])
            residuals[i] = ri["residual_w"]
            c, tc = ri["m_out"], ri["t_s_c"]
        if side == 0:
            runback_end_up = c if len(order) else carry
        else:
            runback_end_lo = c if len(order) else carry
    if len(up) == 0 and len(lo) == 0:
        runback_end_up, runback_end_lo = r0["m_out"], 0.0
    elif len(up) == 0:
        runback_end_up = carry_up
    elif len(lo) == 0:
        runback_end_lo = carry_lo

    impacted = float(m_imp.sum() * dt)
    frozen = float(m_ice.sum() * dt)
    evaporated = float(m_evap.sum() * dt)
    runback_out = float((runback_end_up + runback_end_lo) * dt)
    closure = abs(impacted - frozen - evaporated - runback_out) / impacted if impacted > 0 else 0.0

    if cfg.glaze_rho_mode == "const":
        rho_ice = np.full(n, float(cfg.rho_rime))
    else:
        rho_ice = np.array([rime_density_macklin(cfg.mvd, cfg.v_inf, t) for t in t_s])
    with np.errstate(divide="ignore", invalid="ignore"):
        thickness = np.where(area > 0, m_ice * dt / (rho_ice * np.maximum(area, 1e-30)), 0.0)
    q_span = m_out / (cfg.rho_water * cfg.dx_phys)  # m^2/s per unit span
    with np.errstate(divide="ignore", invalid="ignore"):
        film = np.where(
            (m_out > 0.0) & (tau_t > 0.0),
            np.sqrt(np.maximum(2.0 * cfg.mu_water * q_span / np.maximum(tau_t, 1e-30), 0.0)),
            0.0,
        )
    film = np.where(np.isfinite(film), film, 0.0)

    return {
        "s_m": s_m,
        "n_f": n_f,
        "t_s_c": t_s,
        "m_ice_kg": m_ice * dt,
        "m_evap_kg": m_evap * dt,
        "m_runback_out_kg": m_out * dt,
        "m_runback_in_kg": m_in * dt,
        "thickness_m": thickness,
        "film_m": film,
        "rho_ice": rho_ice,
        "regime": regimes,
        "energy_residual_w": residuals,
        "audit": {
            "impacted": impacted,
            "frozen": frozen,
            "evaporated": evaporated,
            "runback_out": runback_out,
            "closure_error": closure,
        },
    }


def _bfs_distance_cells(base: np.ndarray) -> np.ndarray:
    """4-neighbour BFS distance [cells] from the ``base`` set (no wrap).

    Unlike :func:`_layer_depth` (used for interior layer counting) this
    does not use periodic ``np.roll`` growth, so it stays correct at the
    domain boundaries — required for the outward-cascade direction of the
    glaze deposit (the leading edge sits close to the inlet).
    """
    ny, nx = base.shape
    dist = np.full((ny, nx), -1, dtype=np.int32)
    ys, xs = np.nonzero(base)
    dist[ys, xs] = 0
    queue: deque[tuple[int, int]] = deque(zip(ys.tolist(), xs.tolist()))
    while queue:
        y, x = queue.popleft()
        d = dist[y, x] + 1
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yn, xn = y + dy, x + dx
            if 0 <= yn < ny and 0 <= xn < nx and dist[yn, xn] < 0:
                dist[yn, xn] = d
                queue.append((yn, xn))
    return dist


def deposit_glaze_ice(
    airfoil: np.ndarray,
    solid: np.ndarray,
    m_w: np.ndarray,
    cell_mass: np.ndarray,
    cell_rho_ice: np.ndarray,
    dx_phys: float,
    max_passes: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Credit water to cells, then freeze columns growing outward.

    Per panel the frozen mass is credited to the panel's deposit cells
    (impact-share weights); a cell freezes once it holds
    ``rho_ice dx^3`` and is adjacent to the solid, consuming exactly one
    cell's worth of ice; the leftover water cascades to the outward
    4-neighbour (largest distance-from-airfoil), building columns normal
    to the surface — the same per-cell ledger semantics as the Phase 2a
    freezer, so in the rime limit the frozen set matches.  Returns the
    updated ``(solid, m_w)``; the pending sub-cell remainder stays liquid.
    """
    solid = solid.copy()
    cell_rho_ice = cell_rho_ice.copy()
    m_w = m_w + cell_mass
    ny, nx = solid.shape
    dist = _bfs_distance_cells(airfoil)
    m_cell = np.where(cell_rho_ice > 0.0, cell_rho_ice * dx_phys**3, np.inf)

    for _ in range(max_passes):
        bf = ~solid & _dilate4(torch.from_numpy(solid)).numpy()
        cand = bf & (m_w >= m_cell)
        ys, xs = np.nonzero(cand)
        if ys.size == 0:
            break
        leftover = m_w[ys, xs] - m_cell[ys, xs]
        rho_src = cell_rho_ice[ys, xs]
        solid[ys, xs] = True
        m_w[ys, xs] = 0.0
        for y, x, lm, lr in zip(ys.tolist(), xs.tolist(), leftover.tolist(), rho_src.tolist()):
            if lm <= 0.0:
                continue
            best, bd = None, -1
            for dy, dxn in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y + dy, x + dxn
                if 0 <= yy < ny and 0 <= xx < nx and not solid[yy, xx]:
                    d = int(dist[yy, xx])
                    if d > bd:
                        bd, best = d, (yy, xx)
            if best is not None:
                m_w[best] += lm
                # the column keeps its panel ice density
                if cell_rho_ice[best] <= 0.0:
                    cell_rho_ice[best] = lr
                    m_cell[best] = lr * dx_phys**3
    return solid, m_w


def run_glaze_icing(cfg: IcingConfig, shots: int = 5, log: Any = print) -> dict[str, Any]:
    """Multishot glaze run: flow+droplets -> Messinger+runback -> geometry.

    The industry-standard sequence coupling on top of the Phase 2a/2b
    machinery: each shot (i) runs a *measure-only* simulation on the
    current iced geometry (``freeze_in_run=False``), (ii) bins the
    impingement ledger of the configured droplet phase into surface
    arc-length panels together with the sampled wall shear / edge
    velocity, (iii) solves the Messinger balance with thin-film runback
    at the *physical* LWC for ``dt_shot = t_exposure / shots`` seconds,
    and (iv) deposits the frozen mass as voxel ice growing outward from
    the surface.  The droplet run stays LWC-accelerated (exact for the
    trajectory/beta physics); the acceleration is pinned per shot so the
    ledger corresponds to exactly ``dt_shot`` of physical cloud exposure
    — the thermodynamics never sees the acceleration.
    """
    if cfg.thermo_model != "messinger":
        raise ValueError(
            f"run_glaze_icing requires thermo_model='messinger'; got {cfg.thermo_model!r}"
        )
    if shots < 1:
        raise ValueError(f"shots must be >= 1; got {shots}")
    dt_shot = cfg.t_exposure / shots
    accel = dt_shot / (cfg.steps * cfg.dt_phys)
    log(
        f"  [glaze] shots={shots} dt_shot={dt_shot:.2f} s "
        f"accel/shot={accel:.3e} T_inf={cfg.t_static_c:.1f} C LWC={cfg.lwc:.2e} kg/m^3 "
        f"htc={cfg.htc_mode} rho={cfg.glaze_rho_mode}"
    )

    totals = {"impacted": 0.0, "frozen": 0.0, "evaporated": 0.0, "runback_out": 0.0}
    shot_reports: list[dict[str, Any]] = []
    airfoil_np: np.ndarray | None = None
    solid_np: np.ndarray | None = None
    m_w: np.ndarray | None = None
    res: dict[str, Any] | None = None
    sol: dict[str, Any] | None = None
    panels: dict[str, Any] | None = None

    for k in range(shots):
        shot_cfg = replace(
            cfg,
            freeze_in_run=False,
            beta_window_mode="trailing",  # glaze: whole shot is the beta window
            beta_window_frac=0.0,
            accel_override=accel,  # ledger == dt_shot of physical exposure
        )
        sim = RimeIcingSimulation(shot_cfg, log=log)
        if solid_np is not None:
            prev_ice = torch.from_numpy(solid_np & ~airfoil_np).to(sim.dev)
            sim.solid = sim.airfoil | prev_ice
        res = sim.run()
        airfoil_np = res["airfoil"]
        solid_np = res["solid"]
        s_grid = res["s_grid"]
        if cfg.droplet_phase in ("eulerian", "both"):
            ledger = res["eulerian"]["impact_mass"]
        else:
            ledger = res["impact_mass"]

        stress = sim.sample_surface_stress()
        panels = build_surface_panels(
            shot_cfg, s_grid, ledger, solid_np, stress,
            t_window=shot_cfg.steps * shot_cfg.dt_phys, lwc_accel=accel,
        )
        sol = solve_glaze_surface(cfg, panels, dt_shot)

        # per-cell credit from the panel solution
        if m_w is None:
            m_w = np.zeros_like(solid_np, dtype=np.float64)
        cell_mass = np.zeros_like(m_w)
        cell_rho = np.zeros_like(m_w)
        if panels["n_panels"] > 0:
            dm = sol["m_ice_kg"][panels["dep_p"]] * panels["dep_w"]
            np.add.at(cell_mass, (panels["dep_y"], panels["dep_x"]), dm)
            np.add.at(cell_rho, (panels["dep_y"], panels["dep_x"]),
                      sol["rho_ice"][panels["dep_p"]])
        solid_np, m_w = deposit_glaze_ice(
            airfoil_np, solid_np, m_w, cell_mass, cell_rho, cfg.dx_phys
        )

        for key in totals:
            totals[key] += sol["audit"][key]
        n_ice = int((solid_np & ~airfoil_np).sum())
        shot_reports.append({
            "shot": k + 1,
            "ice_cells": n_ice,
            "n_f_max": float(sol["n_f"].max()) if len(sol["n_f"]) else 0.0,
            "t_s_min_c": float(sol["t_s_c"].min()) if len(sol["t_s_c"]) else 0.0,
            "frozen_kg": sol["audit"]["frozen"],
            "runback_out_kg": sol["audit"]["runback_out"],
        })
        log(
            f"  [glaze] shot {k + 1}/{shots}: ice={n_ice} cells "
            f"frozen={sol['audit']['frozen']:.3e} kg "
            f"runback_out={sol['audit']['runback_out']:.3e} kg "
            f"n_f_max={shot_reports[-1]['n_f_max']:.3f} "
            f"T_s_min={shot_reports[-1]['t_s_min_c']:.1f} C"
        )

    assert res is not None and airfoil_np is not None and solid_np is not None
    assert sol is not None and panels is not None and m_w is not None
    impacted = totals["impacted"]
    closure = (
        abs(impacted - totals["frozen"] - totals["evaporated"] - totals["runback_out"])
        / impacted
        if impacted > 0.0
        else 0.0
    )
    audit = dict(totals)
    audit["closure_error"] = closure

    ice_only = solid_np & ~airfoil_np
    metrics = ice_shape_metrics(
        airfoil_np, solid_np, cfg.dx_phys, cfg.chord_phys, cfg.chord_lu, res["stag"]
    )
    if len(sol["thickness_m"]):
        i_stag = int(np.argmin(np.abs(sol["s_m"])))
        metrics["stag_ice_thickness_m"] = float(sol["thickness_m"][i_stag])
        j = int(np.argmax(sol["thickness_m"]))
        metrics["max_thickness_m"] = float(sol["thickness_m"][j])
        metrics["max_thickness_s_over_c"] = float(sol["s_m"][j] / cfg.chord_phys)
        metrics["n_f_stag"] = float(sol["n_f"][i_stag])
    log(
        f"  [glaze] done: {int(ice_only.sum())} ice cells "
        f"({metrics.get('stag_ice_thickness_m', 0.0) * 1e3:.2f} mm at stagnation, "
        f"max {metrics.get('max_thickness_m', 0.0) * 1e3:.2f} mm at "
        f"s/c={metrics.get('max_thickness_s_over_c', 0.0):+.3f}); "
        f"audit closure {closure * 100:.4f} %"
    )
    return {
        "config": cfg,
        "shots": shots,
        "dt_shot": dt_shot,
        "accel_per_shot": accel,
        "airfoil": airfoil_np,
        "solid": solid_np,
        "ice_only": ice_only,
        "m_w": m_w,
        "mapping": cfg.mapping_report(),
        "panels": {
            "s_over_c": sol["s_m"] / cfg.chord_phys,
            "beta": panels["beta"],
            "m_imp_kg_s": panels["m_imp_kg_s"],
            "h_w_m2k": panels["h"],
            "v_e_m_s": panels["v_e"],
            "tau_t_pa": panels["tau_t"],
            "n_f": sol["n_f"],
            "t_s_c": sol["t_s_c"],
            "m_ice_kg": sol["m_ice_kg"],
            "m_evap_kg": sol["m_evap_kg"],
            "m_runback_out_kg": sol["m_runback_out_kg"],
            "thickness_m": sol["thickness_m"],
            "film_m": sol["film_m"],
            "rho_ice": sol["rho_ice"],
            "regime": sol["regime"],
            "energy_residual_w": sol["energy_residual_w"],
        },
        "metrics": metrics,
        "audit": audit,
        "shot_reports": shot_reports,
        "stag": res["stag"],
        "s_grid": res["s_grid"],
    }


# ---------------------------------------------------------------------------
# Artifacts: ice profile, CSV, PNG, JSON, NPZ
# ---------------------------------------------------------------------------
def ice_profile(airfoil: np.ndarray, ice_only: np.ndarray) -> dict[str, np.ndarray]:
    """Per-column (streamwise) ice extent for CSV export."""
    ny, nx = airfoil.shape
    ys_a = np.full(nx, -1, dtype=np.int64)
    for x in range(nx):
        col = np.nonzero(airfoil[:, x])[0]
        if len(col):
            ys_a[x] = col.min()
    up = np.zeros(nx, dtype=np.int64)
    lo = np.zeros(nx, dtype=np.int64)
    for x in range(nx):
        col = np.nonzero(ice_only[:, x])[0]
        if not len(col) or ys_a[x] < 0:
            continue
        up[x] = max(0, int((col < ys_a[x]).sum()))
        lo[x] = max(0, int((col > ys_a[x]).sum()))
    return {"x": np.arange(nx), "upper_cells": up, "lower_cells": lo}


def save_icing_artifacts(result: dict[str, Any], out_dir: str) -> dict[str, str]:
    """Write ice-shape CSV/PNG, beta CSV/PNG, history PNG, JSON and NPZ.

    matplotlib is imported lazily so the physics module stays usable
    without it.
    """
    import json
    import os
    from pathlib import Path

    cfg = result["config"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    euler = result.get("eulerian")

    # --- CSV: beta curve ---
    beta = result["beta"]
    p = out / "beta_curve.csv"
    with open(p, "w") as fh:
        fh.write("s_over_c,beta,n_cells\n")
        for s, b, n in zip(beta["s_over_c"], beta["beta"], beta["n_cells"]):
            fh.write(f"{s:.6f},{b:.6f},{int(n)}\n")
    files["beta_csv"] = str(p)

    # --- CSV: Eulerian beta curve (Phase 2b, when active) ---
    if euler is not None:
        be = euler["beta"]
        p = out / "beta_eulerian.csv"
        with open(p, "w") as fh:
            fh.write("s_over_c,beta,n_cells\n")
            for s, b, n in zip(be["s_over_c"], be["beta"], be["n_cells"]):
                fh.write(f"{s:.6f},{b:.6f},{int(n)}\n")
        files["beta_euler_csv"] = str(p)

    # --- CSV: ice profile ---
    prof = ice_profile(result["airfoil"], result["ice_only"])
    x_le = float(result["metrics"].get("x_le", 0.0))
    p = out / "ice_profile.csv"
    with open(p, "w") as fh:
        fh.write("x_lu,x_over_c,upper_cells,lower_cells\n")
        for x, u, l in zip(prof["x"], prof["upper_cells"], prof["lower_cells"]):
            if u or l:
                fh.write(
                    f"{int(x)},{(x - x_le) / cfg.chord_lu:.6f},{int(u)},{int(l)}\n"
                )
    files["profile_csv"] = str(p)

    # --- JSON: mapping + audit + metrics ---
    p = out / "icing_metrics.json"
    blob = {
        "mapping": result["mapping"],
        "audit": result["audit"],
        "metrics": result["metrics"],
        "cd0": result["cd0"],
        "cl0": result["cl0"],
        "cd_end": result["cd_end"],
        "cl_end": result["cl_end"],
        "cd_drift_pct": result["cd_drift_pct"],
        "beta_max": float(beta["beta"].max()) if len(beta["beta"]) else 0.0,
        "n_impacts": result["n_impacts"],
    }
    if euler is not None:
        be = euler["beta"]
        blob["eulerian_audit"] = euler["audit"]
        blob["beta_euler_max"] = float(be["beta"].max()) if len(be["beta"]) else 0.0
    with open(p, "w") as fh:
        json.dump(blob, fh, indent=2, default=float)
    files["json"] = str(p)

    # --- NPZ: everything extract_ice_shape.py needs ---
    npz_blob = {
        "airfoil": result["airfoil"],
        "solid": result["solid"],
        "m_w": result["m_w"],
        "impact_mass": result["impact_mass"],
        "s_grid": result["s_grid"],
        "beta_grid": result["beta_grid"],
        "stag": np.array(result["stag"]),
        "hist_step": np.array(result["history"]["step"], dtype=np.float64),
        "hist_t": np.array(result["history"]["t_phys"], dtype=np.float64),
        "hist_cd": np.array([np.nan if v is None else v for v in result["history"]["cd"]]),
        "hist_cl": np.array([np.nan if v is None else v for v in result["history"]["cl"]]),
        "hist_ice": np.array(result["history"]["ice_cells"], dtype=np.float64),
    }
    if euler is not None:
        npz_blob["alpha_e"] = euler["alpha"]
        npz_blob["impact_mass_e"] = euler["impact_mass"]
        npz_blob["beta_e_grid"] = euler["beta_grid"]
    p = out / "result.npz"
    np.savez_compressed(p, **npz_blob)
    files["npz"] = str(p)

    # --- PNGs ---
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return files

    x_le = result["metrics"].get("x_le", 0)
    chord = cfg.chord_lu
    ny, nx = result["airfoil"].shape

    # 1. ice shape over airfoil
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    yy, xx = np.mgrid[0:ny, 0:nx]
    ax.pcolormesh(xx, yy, result["airfoil"].astype(float), cmap="gray", vmin=0, vmax=2)
    ax.pcolormesh(xx, yy, result["ice_only"].astype(float) * 2.0, cmap="cool", vmin=0, vmax=2)
    m = result["metrics"]
    ax.set_xlim(x_le - 0.15 * chord, x_le + 0.55 * chord)
    ax.set_aspect("equal")
    ax.set_title(
        f"Rime ice, t_eq={cfg.t_equiv:.0f} s (LWC_eff={cfg.lwc_eff * 1e3:.2f} g/m3, "
        f"St={cfg.stokes:.3f}, AoA={cfg.aoa_deg:.0f} deg)\n"
        f"ice cells={m.get('n_ice_cells', 0)}, "
        f"horns U/L={m.get('upper_horn_pct_chord', 0):.1f}/{m.get('lower_horn_pct_chord', 0):.1f}%c"
    )
    ax.set_xlabel("x [lu]")
    ax.set_ylabel("y [lu]")
    p = out / "ice_shape.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files["ice_png"] = str(p)

    # 2. beta curve
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if len(beta["s_over_c"]):
        up = beta["s_over_c"] >= 0
        ax.plot(beta["s_over_c"][up] * 100, beta["beta"][up], "o-", label="upper", ms=3)
        ax.plot(beta["s_over_c"][~up] * 100, beta["beta"][~up], "s-", label="lower", ms=3)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("surface arc from LE s/c [% chord] (+ upper)")
    ax.set_ylabel("collection efficiency beta [-]")
    bmax = float(beta["beta"].max()) if len(beta["beta"]) else 0.0
    ax.set_title(
        f"Collection efficiency (beta_max={bmax:.3f}, St={cfg.stokes:.3f})\n"
        "rime: inertia-driven impingement, no bias terms"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    p = out / "beta_curve.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files["beta_png"] = str(p)

    # 2b. Eulerian vs Lagrangian beta overlay (cross-validation)
    if euler is not None and len(euler["beta"]["beta"]):
        be = euler["beta"]
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        if len(beta["s_over_c"]):
            ax.plot(beta["s_over_c"] * 100, beta["beta"], "o-", ms=3,
                    label=f"Lagrangian (max {bmax:.3f})")
        emax = float(be["beta"].max())
        ax.plot(be["s_over_c"] * 100, be["beta"], "s--", ms=3,
                label=f"Eulerian (max {emax:.3f})")
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_xlabel("surface arc from LE s/c [% chord] (+ upper)")
        ax.set_ylabel("collection efficiency beta [-]")
        ax.set_title(
            f"beta cross-validation, same flow trajectory "
            f"(St={cfg.stokes:.3f}, drag={cfg.drag_law})"
        )
        ax.legend()
        ax.grid(alpha=0.3)
        p = out / "beta_compare.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        files["beta_compare_png"] = str(p)

    # 3. history (cd drift + ice growth)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    t = np.array(result["history"]["t_phys"])
    cd = np.array([np.nan if v is None else v for v in result["history"]["cd"]])
    cl = np.array([np.nan if v is None else v for v in result["history"]["cl"]])
    axes[0].plot(t, cd, label="cd")
    axes[0].plot(t, cl, label="cl")
    axes[0].set_xlabel("equivalent physical time [s]")
    axes[0].set_title(
        f"aero response (cd drift {result['cd_drift_pct']:+.1f}%)"
        if result["cd_drift_pct"] is not None
        else "aero response (uniform-flow mode)"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(t, np.array(result["history"]["ice_cells"]))
    axes[1].set_xlabel("equivalent physical time [s]")
    axes[1].set_ylabel("ice cells")
    axes[1].set_title("ice growth")
    axes[1].grid(alpha=0.3)
    p = out / "history.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files["history_png"] = str(p)

    return files
