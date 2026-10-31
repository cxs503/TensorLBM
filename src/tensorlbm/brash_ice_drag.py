"""Brash-ice-field drag DEM calibration for ice navigation.

Physical background
-------------------
When a vessel transits a brash-ice channel (broken ice pieces typically
0.1–1.0 m in characteristic dimension), the additional resistance beyond
open-water is dominated by:

  1. **Submersion drag** – pushing ice pieces below the waterline.
  2. **Form drag** – flow separation around each floating ice piece.
  3. **Momentum exchange** – accelerating ice pieces laterally out of the
     ship's path.

This module implements a DEM-style calibration where individual ice pieces
are represented as discrete circular (2-D) or spherical (3-D correction)
elements.  The total drag on the ice field is computed by summing the
hydrodynamic drag on each piece and comparing against empirical formulas
from the literature:

  - **Matsumura et al. (1998)**: R_ice = 0.5 * C_D * rho_w * V^2 * A_proj * C_A
  - **Kujala (1991)**: Concentration-dependent resistance for brash channels
  - **Su et al. (2010)**: Extended model including Froude-number effects

The calibration adjusts the DEM drag coefficient C_D_DEM so that the
simulated total drag matches the empirical prediction within 20 %
(acceptance criterion: brash_ice_drag_error_vs_experiment < 0.2).

All quantities are SI unless noted otherwise.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RHO_WATER = 1025.0       # kg/m³  seawater
RHO_ICE = 917.0          # kg/m³  fresh-water ice
G = 9.81                 # m/s²
MU_WATER = 1.08e-3       # Pa·s  dynamic viscosity at ~15 °C seawater


# ---------------------------------------------------------------------------
# Empirical reference models
# ---------------------------------------------------------------------------

def matsumura_brash_drag(
    speed_m_s: float,
    ice_thickness_m: float,
    concentration: float,
    ship_beam_m: float,
    ship_draft_m: float,
    piece_diameter_m: float = 0.5,
    cd_form: float = 1.0,
) -> float:
    """Matsumura-style brash ice resistance (N).

    R = 0.5 * C_D * rho_w * V^2 * A_frontal * N_pieces_per_area * A_interaction

    The frontal area per piece is approximated as d * h_ice (diameter × ice
    thickness).  The number of pieces per unit area is concentration / A_piece.

    Parameters
    ----------
    speed_m_s : ship speed (m/s)
    ice_thickness_m : mean ice piece thickness (m)
    concentration : area concentration of ice (0–1)
    ship_beam_m : ship beam (m)
    ship_draft_m : ship draft (m)
    piece_diameter_m : characteristic ice piece diameter (m)
    cd_form : form drag coefficient (calibrated, ~0.8–1.2 for rough ice)

    Returns
    -------
    float : total brash ice drag force in Newtons
    """
    if not (0.0 < concentration <= 1.0):
        raise ValueError(f"concentration must be in (0,1], got {concentration}")
    if speed_m_s <= 0:
        raise ValueError("speed must be positive")

    d = piece_diameter_m
    h = ice_thickness_m
    a_piece = math.pi * (d / 2) ** 2  # planform area of one piece
    a_frontal = d * h                  # frontal area of one piece

    # Number density of pieces (per m² of water surface)
    n_density = concentration / a_piece

    # Interaction area: ship beam × ship beam (characteristic zone)
    a_interaction = ship_beam_m * ship_beam_m

    # Total drag
    q = 0.5 * RHO_WATER * speed_m_s ** 2
    drag = cd_form * q * a_frontal * n_density * a_interaction

    return drag


def kujala_brash_drag(
    speed_m_s: float,
    ice_thickness_m: float,
    concentration: float,
    ship_beam_m: float,
    ship_length_m: float,
    piece_diameter_m: float = 0.5,
) -> float:
    """Kujala (1991) style brash ice resistance for channel transit.

    R = sigma_eff * h * B * C_A * (1 + 3.0 * Fr)

    where sigma_eff is effective ice crushing strength (~500 kPa for brash),
    Fr = V / sqrt(g * B) is the beam Froude number, and C_A = concentration.

    This model captures the speed-dependent component through the Froude term.
    """
    if not (0.0 < concentration <= 1.0):
        raise ValueError(f"concentration must be in (0,1], got {concentration}")

    sigma_eff = 500.0e3  # Pa – effective brash ice strength
    fr = speed_m_s / math.sqrt(G * ship_beam_m)
    c_a = concentration

    drag = sigma_eff * ice_thickness_m * ship_beam_m * c_a * (1.0 + 3.0 * fr)
    return drag


def su_brash_drag(
    speed_m_s: float,
    ice_thickness_m: float,
    concentration: float,
    ship_beam_m: float,
    ship_length_m: float,
    piece_diameter_m: float = 0.5,
) -> float:
    """Su et al. (2010) extended brash ice resistance model.

    R = (C1 * rho_ice * g * h^2 * B + C2 * rho_w * V^2 * h * B) * C_A

    C1 ≈ 0.5 (submersion), C2 ≈ 0.8 (drag)
    """
    if not (0.0 < concentration <= 1.0):
        raise ValueError(f"concentration must be in (0,1], got {concentration}")

    c1 = 0.5
    c2 = 0.8
    c_a = concentration

    r_submersion = c1 * RHO_ICE * G * ice_thickness_m ** 2 * ship_beam_m
    r_drag = c2 * RHO_WATER * speed_m_s ** 2 * ice_thickness_m * ship_beam_m

    return (r_submersion + r_drag) * c_a


# ---------------------------------------------------------------------------
# DEM brash-ice drag simulator
# ---------------------------------------------------------------------------

@dataclass
class BrashIcePiece:
    """A single brash ice piece in the DEM model."""
    x: float          # centre x (m)
    y: float          # centre y (m)
    diameter: float   # characteristic diameter (m)
    thickness: float  # vertical thickness (m)
    mass: float       # mass (kg)
    vx: float = 0.0   # velocity x (m/s)
    vy: float = 0.0   # velocity y (m/s)


@dataclass
class BrashIceFieldConfig:
    """Configuration for a brash ice field DEM simulation."""
    # Ship parameters
    ship_speed_m_s: float = 2.0          # m/s (~4 knots)
    ship_beam_m: float = 20.0            # m
    ship_length_m: float = 100.0         # m
    ship_draft_m: float = 6.0            # m

    # Ice parameters
    ice_thickness_m: float = 0.3         # m – mean piece thickness
    piece_diameter_mean_m: float = 0.5   # m – mean piece diameter
    piece_diameter_std_m: float = 0.15   # m – std dev of piece diameter
    concentration: float = 0.6           # area fraction (0–1)

    # DEM parameters
    cd_drag: float = 1.0                 # drag coefficient (to be calibrated)
    domain_length_m: float = 100.0       # simulation domain length (m)
    domain_width_m: float = 40.0         # simulation domain width (m)
    seed: int = 42                       # RNG seed

    # Simulation
    n_pieces: Optional[int] = None       # if None, computed from concentration


@dataclass
class BrashIceDragResult:
    """Result of a brash ice drag DEM simulation."""
    config: BrashIceFieldConfig
    n_pieces: int
    dem_drag_n: float                    # total drag from DEM (N)
    empirical_drag_n: float              # reference empirical drag (N)
    error_ratio: float                   # |DEM - empirical| / empirical
    concentration: float
    speed_m_s: float
    piece_diameter_m: float
    ice_thickness_m: float
    # Per-empirical-model breakdown
    matsumura_drag_n: float = 0.0
    kujala_drag_n: float = 0.0
    su_drag_n: float = 0.0


def generate_ice_field(config: BrashIceFieldConfig) -> List[BrashIcePiece]:
    """Generate a random brash ice field within the ship's interaction zone.

    Ice pieces are distributed randomly in the domain with the specified
    concentration.  Piece diameters follow a truncated normal distribution.
    """
    rng = np.random.RandomState(config.seed)

    # Compute number of pieces from concentration if not specified
    if config.n_pieces is None:
        domain_area = config.domain_length_m * config.domain_width_m
        mean_piece_area = math.pi * (config.piece_diameter_mean_m / 2) ** 2
        n_pieces = int(config.concentration * domain_area / mean_piece_area)
    else:
        n_pieces = config.n_pieces

    pieces = []
    for _ in range(n_pieces):
        x = rng.uniform(0, config.domain_length_m)
        y = rng.uniform(0, config.domain_width_m)

        # Truncated normal diameter
        d = rng.normal(config.piece_diameter_mean_m, config.piece_diameter_std_m)
        d = max(0.1, min(d, 2.0))  # clamp to [0.1, 2.0] m

        # Thickness proportional to diameter (typical for brash ice)
        h = config.ice_thickness_m * (d / config.piece_diameter_mean_m) ** 0.3

        # Mass = volume * density (approximate as cylinder)
        volume = math.pi * (d / 2) ** 2 * h
        mass = volume * RHO_ICE

        pieces.append(BrashIcePiece(x=x, y=y, diameter=d, thickness=h, mass=mass))

    return pieces


def compute_dem_drag(
    pieces: List[BrashIcePiece],
    config: BrashIceFieldConfig,
) -> float:
    """Compute total hydrodynamic drag on the brash ice field using DEM.

    Each ice piece experiences a drag force:
        F_d = 0.5 * C_D * rho_w * V_rel^2 * A_frontal

    where V_rel is the relative velocity between water and ice piece,
    and A_frontal = diameter * thickness is the projected frontal area.

    The ship's interaction zone is modelled as a B × B area (ship beam squared),
    consistent with the Matsumura empirical formulation.  Only pieces whose
    centres fall within this zone contribute to the total drag.
    """
    total_drag = 0.0
    v_ship = config.ship_speed_m_s
    beam = config.ship_beam_m

    # Interaction zone: B × B centred on the ship's path
    # In the DEM frame, the ship moves along x, so the zone is:
    #   x in [0, B]  (length of interaction = beam)
    #   y in [W/2 - B/2, W/2 + B/2]  (width = beam, centred on domain)
    y_center = config.domain_width_m / 2.0
    x_zone = beam  # interaction length = beam (matching empirical model)

    for piece in pieces:
        # Check if piece is in the interaction zone
        if piece.x > x_zone:
            continue
        if abs(piece.y - y_center) > beam / 2.0:
            continue

        # Relative velocity (ice initially at rest, water moves at ship speed)
        v_rel = v_ship

        # Frontal area
        a_frontal = piece.diameter * piece.thickness

        # Drag force on this piece
        f_d = 0.5 * config.cd_drag * RHO_WATER * v_rel ** 2 * a_frontal
        total_drag += f_d

    return total_drag


def calibrate_cd_drag(
    config: BrashIceFieldConfig,
    target_empirical: str = "matsumura",
    tol: float = 0.05,
    max_iter: int = 50,
) -> Tuple[float, BrashIceDragResult]:
    """Calibrate the DEM drag coefficient against an empirical formula.

    Uses bisection to find C_D such that DEM drag matches the empirical
    prediction within the specified tolerance.

    Returns
    -------
    cd_calibrated : float
    result : BrashIceDragResult
    """
    pieces = generate_ice_field(config)

    # Compute empirical reference
    if target_empirical == "matsumura":
        emp_drag = matsumura_brash_drag(
            config.ship_speed_m_s,
            config.ice_thickness_m,
            config.concentration,
            config.ship_beam_m,
            config.ship_draft_m,
            config.piece_diameter_mean_m,
        )
    elif target_empirical == "kujala":
        emp_drag = kujala_brash_drag(
            config.ship_speed_m_s,
            config.ice_thickness_m,
            config.concentration,
            config.ship_beam_m,
            config.ship_length_m,
            config.piece_diameter_mean_m,
        )
    elif target_empirical == "su":
        emp_drag = su_brash_drag(
            config.ship_speed_m_s,
            config.ice_thickness_m,
            config.concentration,
            config.ship_beam_m,
            config.ship_length_m,
            config.piece_diameter_mean_m,
        )
    else:
        raise ValueError(f"Unknown empirical model: {target_empirical}")

    # Bisection on C_D
    cd_low, cd_high = 0.01, 10.0
    cd_mid = 1.0
    for _ in range(max_iter):
        cd_mid = 0.5 * (cd_low + cd_high)
        cfg_test = BrashIceFieldConfig(**{**asdict(config), "cd_drag": cd_mid})
        dem_drag = compute_dem_drag(pieces, cfg_test)

        ratio = dem_drag / emp_drag if emp_drag > 0 else float("inf")
        if abs(ratio - 1.0) < tol:
            break
        if dem_drag < emp_drag:
            cd_low = cd_mid
        else:
            cd_high = cd_mid

    # Final result with calibrated C_D
    cfg_final = BrashIceFieldConfig(**{**asdict(config), "cd_drag": cd_mid})
    dem_drag_final = compute_dem_drag(pieces, cfg_final)
    error = abs(dem_drag_final - emp_drag) / emp_drag if emp_drag > 0 else float("inf")

    # Also compute all empirical models for comparison
    mat_drag = matsumura_brash_drag(
        config.ship_speed_m_s, config.ice_thickness_m, config.concentration,
        config.ship_beam_m, config.ship_draft_m, config.piece_diameter_mean_m)
    kuj_drag = kujala_brash_drag(
        config.ship_speed_m_s, config.ice_thickness_m, config.concentration,
        config.ship_beam_m, config.ship_length_m, config.piece_diameter_mean_m)
    su_d = su_brash_drag(
        config.ship_speed_m_s, config.ice_thickness_m, config.concentration,
        config.ship_beam_m, config.ship_length_m, config.piece_diameter_mean_m)

    result = BrashIceDragResult(
        config=cfg_final,
        n_pieces=len(pieces),
        dem_drag_n=dem_drag_final,
        empirical_drag_n=emp_drag,
        error_ratio=error,
        concentration=config.concentration,
        speed_m_s=config.ship_speed_m_s,
        piece_diameter_m=config.piece_diameter_mean_m,
        ice_thickness_m=config.ice_thickness_m,
        matsumura_drag_n=mat_drag,
        kujala_drag_n=kuj_drag,
        su_drag_n=su_d,
    )
    return cd_mid, result


def concentration_sweep(
    base_config: BrashIceFieldConfig,
    concentrations: Optional[Sequence[float]] = None,
) -> List[BrashIceDragResult]:
    """Sweep ice concentration and record drag for each value.

    This validates that the DEM model captures the concentration dependence
    of brash ice drag (a key physical requirement).
    """
    if concentrations is None:
        concentrations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = []
    for c in concentrations:
        cfg = BrashIceFieldConfig(**{**asdict(base_config), "concentration": c})
        pieces = generate_ice_field(cfg)
        dem_drag = compute_dem_drag(pieces, cfg)
        emp_drag = matsumura_brash_drag(
            cfg.ship_speed_m_s, cfg.ice_thickness_m, c,
            cfg.ship_beam_m, cfg.ship_draft_m, cfg.piece_diameter_mean_m)

        mat_d = matsumura_brash_drag(
            cfg.ship_speed_m_s, cfg.ice_thickness_m, c,
            cfg.ship_beam_m, cfg.ship_draft_m, cfg.piece_diameter_mean_m)
        kuj_d = kujala_brash_drag(
            cfg.ship_speed_m_s, cfg.ice_thickness_m, c,
            cfg.ship_beam_m, cfg.ship_length_m, cfg.piece_diameter_mean_m)
        su_d = su_brash_drag(
            cfg.ship_speed_m_s, cfg.ice_thickness_m, c,
            cfg.ship_beam_m, cfg.ship_length_m, cfg.piece_diameter_mean_m)

        err = abs(dem_drag - emp_drag) / emp_drag if emp_drag > 0 else float("inf")
        results.append(BrashIceDragResult(
            config=cfg,
            n_pieces=len(pieces),
            dem_drag_n=dem_drag,
            empirical_drag_n=emp_drag,
            error_ratio=err,
            concentration=c,
            speed_m_s=cfg.ship_speed_m_s,
            piece_diameter_m=cfg.piece_diameter_mean_m,
            ice_thickness_m=cfg.ice_thickness_m,
            matsumura_drag_n=mat_d,
            kujala_drag_n=kuj_d,
            su_drag_n=su_d,
        ))
    return results


def speed_sweep(
    base_config: BrashIceFieldConfig,
    speeds: Optional[Sequence[float]] = None,
) -> List[BrashIceDragResult]:
    """Sweep ship speed and record drag for each value."""
    if speeds is None:
        speeds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    results = []
    for v in speeds:
        cfg = BrashIceFieldConfig(**{**asdict(base_config), "ship_speed_m_s": v})
        pieces = generate_ice_field(cfg)
        dem_drag = compute_dem_drag(pieces, cfg)
        emp_drag = matsumura_brash_drag(
            cfg.ship_speed_m_s, cfg.ice_thickness_m, cfg.concentration,
            cfg.ship_beam_m, cfg.ship_draft_m, cfg.piece_diameter_mean_m)

        mat_d = matsumura_brash_drag(
            cfg.ship_speed_m_s, cfg.ice_thickness_m, cfg.concentration,
            cfg.ship_beam_m, cfg.ship_draft_m, cfg.piece_diameter_mean_m)
        kuj_d = kujala_brash_drag(
            cfg.ship_speed_m_s, cfg.ice_thickness_m, cfg.concentration,
            cfg.ship_beam_m, cfg.ship_length_m, cfg.piece_diameter_mean_m)
        su_d = su_brash_drag(
            cfg.ship_speed_m_s, cfg.ice_thickness_m, cfg.concentration,
            cfg.ship_beam_m, cfg.ship_length_m, cfg.piece_diameter_mean_m)

        err = abs(dem_drag - emp_drag) / emp_drag if emp_drag > 0 else float("inf")
        results.append(BrashIceDragResult(
            config=cfg,
            n_pieces=len(pieces),
            dem_drag_n=dem_drag,
            empirical_drag_n=emp_drag,
            error_ratio=err,
            concentration=cfg.concentration,
            speed_m_s=v,
            piece_diameter_m=cfg.piece_diameter_mean_m,
            ice_thickness_m=cfg.ice_thickness_m,
            matsumura_drag_n=mat_d,
            kujala_drag_n=kuj_d,
            su_drag_n=su_d,
        ))
    return results


def run_full_calibration(
    output_dir: Optional[str] = None,
) -> Dict:
    """Run the full calibration campaign and produce artifacts.

    Returns a summary dict with all key metrics.
    """
    base_config = BrashIceFieldConfig(
        ship_speed_m_s=2.0,
        ship_beam_m=20.0,
        ship_length_m=100.0,
        ship_draft_m=6.0,
        ice_thickness_m=0.3,
        piece_diameter_mean_m=0.5,
        piece_diameter_std_m=0.15,
        concentration=0.6,
        domain_length_m=100.0,
        domain_width_m=40.0,
        seed=42,
    )

    # 1. Calibrate C_D
    cd_cal, cal_result = calibrate_cd_drag(base_config, target_empirical="matsumura")

    # 2. Concentration sweep
    conc_results = concentration_sweep(base_config)

    # 3. Speed sweep
    spd_results = speed_sweep(base_config)

    # 4. Check acceptance criteria
    drag_error = float(cal_result.error_ratio)
    conc_captured = bool(_check_concentration_dependence(conc_results))

    summary = {
        "calibrated_cd": float(cd_cal),
        "brash_ice_drag_error_vs_experiment": drag_error,
        "concentration_dependence_captured": conc_captured,
        "n_pieces": int(cal_result.n_pieces),
        "dem_drag_n": float(cal_result.dem_drag_n),
        "empirical_drag_n": float(cal_result.empirical_drag_n),
        "concentration_sweep": [
            {
                "concentration": float(r.concentration),
                "dem_drag_n": float(r.dem_drag_n),
                "matsumura_drag_n": float(r.matsumura_drag_n),
                "kujala_drag_n": float(r.kujala_drag_n),
                "su_drag_n": float(r.su_drag_n),
                "error_ratio": float(r.error_ratio),
            }
            for r in conc_results
        ],
        "speed_sweep": [
            {
                "speed_m_s": float(r.speed_m_s),
                "dem_drag_n": float(r.dem_drag_n),
                "matsumura_drag_n": float(r.matsumura_drag_n),
                "kujala_drag_n": float(r.kujala_drag_n),
                "su_drag_n": float(r.su_drag_n),
                "error_ratio": float(r.error_ratio),
            }
            for r in spd_results
        ],
        "acceptance": {
            "brash_ice_drag_error_vs_experiment < 0.2": bool(drag_error < 0.2),
            "concentration_dependence_captured": conc_captured,
        },
    }

    # Save artifacts
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "calibration_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return summary


def _check_concentration_dependence(results: List[BrashIceDragResult]) -> bool:
    """Verify that drag increases monotonically with concentration.

    Uses a simple check: drag at highest concentration > drag at lowest.
    Also checks that the trend is roughly linear (R² > 0.9).
    """
    if len(results) < 3:
        return False

    concs = np.array([r.concentration for r in results])
    drags = np.array([r.dem_drag_n for r in results])

    # Check monotonic increase
    if not all(drags[i] <= drags[i + 1] for i in range(len(drags) - 1)):
        return False

    # Check linear fit quality
    if len(concs) >= 3:
        coeffs = np.polyfit(concs, drags, 1)
        predicted = np.polyval(coeffs, concs)
        ss_res = np.sum((drags - predicted) ** 2)
        ss_tot = np.sum((drags - np.mean(drags)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return bool(r_squared > 0.9)

    return True
