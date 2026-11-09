"""LNG cargo-tank sloshing: resonance ladder, phase and impact-pressure metrics.

New frontier capability for campaign node **P87W2-N01** (2-D rectangular tank,
``L x H = 1.0 m x 0.5 m``, partial fill ``h/H = 0.2 .. 0.9``, free decay from
``eta(x,0) = A cos(k_n x)`` plus a resonant-sway impact case).

The physics kernel is the audited Körner free-surface LBM already shipped in
:mod:`tensorlbm.free_surface_lbm_2d`; this package adds

``theory``    Abramson (1966) eigen-frequencies, the *linear* free-decay field,
              the linear wall-pressure amplitude (Faltinsen & Timokha) and the
              stagnation / bore-reflection impact bounds;
``tank2d``    the 1 m x 0.5 m tank with a Froude/Mach-consistent lattice
              scaling, the single-mode release initial condition, sub-cell
              surface elevation from the Körner mass ledger, wall pressure
              gauges and the liquid-inventory ledger;
``identify``  damped-sinusoid identification (period, decay rate, natural
              frequency) and the phase-error metric of criterion (c);
``campaign``  the acceptance campaign: fill sweep (modes 1 and 2), grid family
              ``dx = H/40, H/80, H/160``, ``g -> 0`` negative control and the
              high-fill resonant impact case, plus the criterion judge.

Relation to :mod:`tensorlbm.sloshing_tank` and :mod:`tensorlbm.lng_sloshing`
---------------------------------------------------------------------------
``sloshing_tank`` is a two-phase colour-gradient model driven by harmonic
forcing; ``lng_sloshing`` (campaign IN143/IN144-N07) runs the Körner model with
1 : 1 lattice scaling and *forced sway* free-surface elevation extraction.  This
package is deliberately different where it matters for verification:

* geometry and scaling are tied to **SI units** through a fixed lattice velocity
  (so the Froude number and the lattice Mach number are grid-independent, which
  is what a GCI study requires),
* the excitation is a **free decay from a pure single mode** (mode 1 *and* mode
  2), never a forced amplitude-response sweep,
* the metrics are *damping-aware* (``f_d``, ``zeta``, ``f_n``) and include the
  phase error against the linear theory,
* the walls are free-slip by default: Abramson's anchor is inviscid, and the
  no-slip lattice boundary layer detunes the sloshing frequency by several per
  cent (measured: -2.8 % at ``tau = 0.502``, -6.6 % at ``tau = 0.55``); the
  friction runs are reported as a sensitivity, not used for the criterion.

Honest limitation: no digitised sloshing-impact experiment ships with this
repository, so criterion (d) is reported as WITHHELD against experiment and
evaluated against the linear wall-pressure amplitude plus the inviscid
stagnation / bore-reflection bounds.
"""

from __future__ import annotations

from .campaign import (
    CampaignSpec,
    case_list,
    judge_campaign,
    run_case,
    summarise_free_decay,
    write_json,
)
from .identify import (
    ModeFit,
    envelope_decay_rate,
    fft_peak,
    harmonic_amplitude,
    identify_mode,
    period_from_crossings,
    phase_error_deg,
    pressure_peak,
    zero_crossings,
)
from .tank2d import (
    LatticeScaling,
    SloshingTank,
    TankField,
    build_tank_field,
    column_liquid_height,
    modal_amplitude,
    run_free_decay,
    run_sway,
    surface_elevation,
    wall_pressure_profile,
)
from .theory import (
    AbramsonAnchor,
    abramson_angular_frequency,
    abramson_brief_formula,
    abramson_frequency,
    abramson_wave_number,
    bore_reflection_pressure,
    deep_water_frequency,
    hydrostatic_runup_pressure,
    linear_decay_elevation,
    linear_decay_velocity_amplitude,
    linear_wall_pressure_amplitude,
    linear_wall_pressure_bottom_ratio,
    shallow_water_frequency,
    stagnation_impact_pressure,
)

__all__ = [
    "AbramsonAnchor",
    "CampaignSpec",
    "LatticeScaling",
    "ModeFit",
    "SloshingTank",
    "TankField",
    "abramson_angular_frequency",
    "abramson_brief_formula",
    "abramson_frequency",
    "abramson_wave_number",
    "bore_reflection_pressure",
    "build_tank_field",
    "case_list",
    "column_liquid_height",
    "deep_water_frequency",
    "envelope_decay_rate",
    "fft_peak",
    "harmonic_amplitude",
    "hydrostatic_runup_pressure",
    "identify_mode",
    "judge_campaign",
    "linear_decay_elevation",
    "linear_decay_velocity_amplitude",
    "linear_wall_pressure_amplitude",
    "linear_wall_pressure_bottom_ratio",
    "modal_amplitude",
    "period_from_crossings",
    "phase_error_deg",
    "pressure_peak",
    "run_case",
    "run_free_decay",
    "run_sway",
    "shallow_water_frequency",
    "stagnation_impact_pressure",
    "summarise_free_decay",
    "surface_elevation",
    "wall_pressure_profile",
    "write_json",
    "zero_crossings",
]
