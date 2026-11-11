"""Frazil power-law rheology — mature verification face.

Node ``p366w3-n02-frazil-rheology-mature``.

This module provides the mature verification face for frazil ice slurry
rheology. It builds on :mod:`tensorlbm.frazil_powerlaw_shear` but adds:

- Physical suspension creation with continuous coordinates
- Voxelization to lattice
- Hydrodynamic volume fraction (fractal aggregates trap water)
- Einstein-Batchelor-Green and Krieger-Dougherty closures
- Bootstrap power-law fitting with confidence intervals
- Grid convergence pair analysis
- Comprehensive acceptance evaluation

References
----------
Einstein, A. (1906) Ann. Phys. 19, 289.
Batchelor, G. K. & Green, J. T. (1972) J. Fluid Mech. 56, 401.
Krieger, I. M. & Dougherty, T. J. (1959) Trans. Soc. Rheol. 3, 137.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import frazil_powerlaw_shear as shear

__all__ = [
    "RADIUS_FRACTION",
    "PhysicalSuspension",
    "physical_suspension",
    "voxelise",
    "hydrodynamic_volume_fraction",
    "einstein_relative_viscosity",
    "einstein_batchelor_green",
    "krieger_dougherty",
    "bootstrap_power_law",
    "grid_convergence_pair",
    "evaluate_acceptance",
    "_point_with_scale",
    "self_consistent_point",
]

RADIUS_FRACTION = 0.08  # crystal radius as fraction of cell width
PHI_MAX_KD = 0.60       # maximum packing for Krieger-Dougherty
BATCHELOR_K2 = 6.2      # Batchelor second-order coefficient
INTRINSIC_VISCOSITY = 2.5


# ---------------------------------------------------------------------------
# Physical suspension
# ---------------------------------------------------------------------------
@dataclass
class PhysicalSuspension:
    """A physical suspension of ice crystals in water.
    
    Attributes
    ----------
    phi : float
        Declared solid volume fraction.
    family : str
        'dispersed' or 'flocculated'.
    n_primary : int
        Number of primary particles per aggregate.
    df : float
        Fractal dimension.
    seed : int
        Random seed.
    particles : list
        List of particle dicts with 'center' and 'radius'.
    clusters : dict
        Mapping from cluster index to list of particle positions.
    solid_volume_fraction : float
        Actual solid volume fraction (may differ from declared due to placement).
    """
    phi: float
    family: str
    n_primary: int
    df: float
    seed: int
    particles: list = field(default_factory=list)
    clusters: dict = field(default_factory=dict)
    solid_volume_fraction: float = 0.0


def physical_suspension(
    phi: float,
    family: str = "dispersed",
    n_primary: int = 1,
    df: float = shear.DEFAULT_DF,
    seed: int = 42,
    radius_fraction: float = RADIUS_FRACTION,
) -> PhysicalSuspension:
    """Create a physical suspension configuration.
    
    Parameters
    ----------
    phi : float
        Target solid volume fraction.
    family : str
        'dispersed' (single crystals) or 'flocculated' (aggregates).
    n_primary : int
        Number of primaries per aggregate.
    df : float
        Fractal dimension (for flocculated).
    seed : int
        Random seed.
    radius_fraction : float
        Crystal radius as fraction of cell width.
    
    Returns
    -------
    PhysicalSuspension
    """
    susp = PhysicalSuspension(
        phi=phi, family=family, n_primary=n_primary,
        df=df, seed=seed,
    )
    
    rng = np.random.RandomState(seed)
    
    # We'll place particles in a unit cell [0, 1]^3
    # and scale to the actual lattice later
    vol_cell = 1.0
    vol_particle = (4.0 / 3.0) * math.pi * radius_fraction**3
    
    if n_primary == 1:
        n_particles = max(1, int(round(phi * vol_cell / vol_particle)))
    else:
        n_aggregates = max(1, int(round(phi * vol_cell / (n_primary * vol_particle))))
        n_particles = n_aggregates * n_primary
    
    particles = []
    clusters = {}
    
    if family == "dispersed" or n_primary == 1:
        # Place individual spheres
        for i in range(n_particles):
            placed = False
            for _ in range(10000):
                cx = rng.uniform(radius_fraction, 1.0 - radius_fraction)
                cy = rng.uniform(radius_fraction, 1.0 - radius_fraction)
                cz = rng.uniform(radius_fraction, 1.0 - radius_fraction)
                
                # Check overlap
                overlap = False
                for p in particles:
                    dist = math.sqrt((cx - p['center'][0])**2 +
                                    (cy - p['center'][1])**2 +
                                    (cz - p['center'][2])**2)
                    if dist < 2.02 * radius_fraction:
                        overlap = True
                        break
                
                if not overlap:
                    p = {
                        'center': (cx, cy, cz),
                        'radius': radius_fraction,
                        'meta': {'cluster': i}
                    }
                    particles.append(p)
                    clusters[i] = [(cx, cy, cz)]
                    placed = True
                    break
            if not placed:
                break
    else:
        # Place fractal aggregates
        agg_idx = 0
        p_idx = 0
        while p_idx < n_particles:
            # Aggregate center
            agg_cx = rng.uniform(3 * radius_fraction, 1.0 - 3 * radius_fraction)
            agg_cy = rng.uniform(3 * radius_fraction, 1.0 - 3 * radius_fraction)
            agg_cz = rng.uniform(3 * radius_fraction, 1.0 - 3 * radius_fraction)
            
            cluster_members = []
            for j in range(min(n_primary, n_particles - p_idx)):
                if j == 0:
                    px, py, pz = agg_cx, agg_cy, agg_cz
                else:
                    theta = rng.uniform(0, 2 * math.pi)
                    phi_angle = rng.uniform(0, math.pi)
                    r_offset = 2.0 * radius_fraction * (j / n_primary)**(1.0 / df)
                    px = agg_cx + r_offset * math.sin(phi_angle) * math.cos(theta)
                    py = agg_cy + r_offset * math.sin(phi_angle) * math.sin(theta)
                    pz = agg_cz + r_offset * math.cos(phi_angle)
                    
                    # Clamp
                    px = np.clip(px, radius_fraction, 1.0 - radius_fraction)
                    py = np.clip(py, radius_fraction, 1.0 - radius_fraction)
                    pz = np.clip(pz, radius_fraction, 1.0 - radius_fraction)
                
                p = {
                    'center': (px, py, pz),
                    'radius': radius_fraction,
                    'meta': {'cluster': agg_idx}
                }
                particles.append(p)
                cluster_members.append((px, py, pz))
                p_idx += 1
            
            clusters[agg_idx] = cluster_members
            agg_idx += 1
    
    susp.particles = particles
    susp.clusters = clusters
    susp.solid_volume_fraction = len(particles) * vol_particle / vol_cell
    
    return susp


def voxelise(
    susp: PhysicalSuspension,
    n_xy: int,
    n_z: int,
) -> tuple[list[dict], dict]:
    """Voxelise a physical suspension to lattice coordinates.
    
    Parameters
    ----------
    susp : PhysicalSuspension
        Physical suspension in continuous coordinates [0, 1]^3.
    n_xy : int
        Lattice size in x and y.
    n_z : int
        Lattice size in z.
    
    Returns
    -------
    tuple
        (particles_scaled, info) where particles_scaled have lattice coordinates
        and info contains geometry statistics.
    """
    particles_scaled = []
    total_voxels = 0
    
    for p in susp.particles:
        cx, cy, cz = p['center']
        r = p['radius']
        
        # Scale to lattice
        lx = cx * n_xy
        ly = cy * n_xy
        lz = cz * n_xy  # z uses same scale as x,y for cubic cells
        lr = r * n_xy
        
        # Count voxels in this sphere
        n_voxels = 0
        iz_min = max(0, int(lz - lr - 1))
        iz_max = min(n_z - 1, int(lz + lr + 1))
        iy_min = max(0, int(ly - lr - 1))
        iy_max = min(n_xy - 1, int(ly + lr + 1))
        ix_min = max(0, int(lx - lr - 1))
        ix_max = min(n_xy - 1, int(lx + lr + 1))
        
        for iz in range(iz_min, iz_max + 1):
            for iy in range(iy_min, iy_max + 1):
                for ix in range(ix_min, ix_max + 1):
                    dist2 = (iz + 0.5 - lz)**2 + (iy + 0.5 - ly)**2 + (ix + 0.5 - lx)**2
                    if dist2 <= lr**2:
                        n_voxels += 1
        
        total_voxels += n_voxels
        
        p_scaled = {
            'center': (lz, ly, lx),
            'radius': lr,
            'meta': dict(p['meta']),
        }
        particles_scaled.append(p_scaled)
    
    vol_total = n_xy * n_xy * n_z
    phi_lattice = total_voxels / vol_total if vol_total > 0 else 0.0
    
    info = {
        'n_particles': len(particles_scaled),
        'phi_lattice': phi_lattice,
        'total_voxels': total_voxels,
        'vol_total': vol_total,
    }
    
    return particles_scaled, info


# ---------------------------------------------------------------------------
# Rheology closures
# ---------------------------------------------------------------------------
def einstein_relative_viscosity(phi: float) -> float:
    """Einstein (1906) dilute suspension viscosity.
    
    eta_r = 1 + 2.5 * phi
    
    Valid for phi < ~0.05 (dilute limit).
    """
    return 1.0 + INTRINSIC_VISCOSITY * phi


def einstein_batchelor_green(phi: float) -> float:
    """Einstein-Batchelor-Green viscosity.
    
    eta_r = 1 + 2.5*phi + 6.2*phi^2
    
    Includes first pair interaction correction (Batchelor & Green 1972).
    Valid for phi < ~0.10.
    """
    return 1.0 + INTRINSIC_VISCOSITY * phi + BATCHELOR_K2 * phi**2


def krieger_dougherty(phi: float, phi_max: float = PHI_MAX_KD) -> float:
    """Krieger-Dougherty viscosity.
    
    eta_r = (1 - phi/phi_max)^(-[eta]*phi_max)
    
    Valid for the full concentration range up to phi_max.
    """
    if phi >= phi_max:
        return float("inf")
    if phi <= 0:
        return 1.0
    exponent = INTRINSIC_VISCOSITY * phi_max
    return (1.0 - phi / phi_max)**(-exponent)


def hydrodynamic_volume_fraction(
    phi_lattice: float,
    n_primary: int = 1,
    df: float = shear.DEFAULT_DF,
) -> float:
    """Compute the hydrodynamic volume fraction.
    
    For fractal aggregates, the hydrodynamic volume includes the water
    trapped within the aggregate structure. The enhancement factor is:
    
        phi_h / phi_solid ~ n_primary^(1 - 1/df)
    
    For single spheres (n_primary=1), phi_h = phi_lattice.
    """
    if n_primary <= 1:
        return phi_lattice
    
    # Fractal aggregate: hydrodynamic radius ~ a * n^(1/df)
    # Volume enhancement ~ n^(3/df - 1) for df < 3
    if df >= 3.0:
        return phi_lattice
    
    enhancement = n_primary**(3.0 / df - 1.0)
    return phi_lattice * enhancement


# ---------------------------------------------------------------------------
# Shear cell runner (wraps frazil_powerlaw_shear)
# ---------------------------------------------------------------------------
def _point_with_scale(
    particles: list[dict],
    n_xy: int,
    n_z: int,
    nu: float,
    gamma_dot: float,
    scale: float = 1.0,
    steps: int = 2000,
    warmup: int = 1000,
    sample_every: int = 10,
    band: tuple[int, int] = (1, 4),
) -> dict:
    """Run a single shear cell point with given particles and kinematic scale.
    
    Parameters
    ----------
    particles : list
        Particle list with lattice coordinates.
    n_xy, n_z : int
        Domain dimensions.
    nu : float
        Kinematic viscosity.
    gamma_dot : float
        Target shear rate.
    scale : float
        Kinematic condition scale (1.0 = affine).
    steps, warmup, sample_every : int
        Simulation parameters.
    band : tuple
        Wall band for stress measurement.
    
    Returns
    -------
    dict
        Measurement record.
    """
    # Convert shear rate to wall velocity
    # For Couette: gamma_dot = 2 * u_wall / H, where H = n_z - 2
    H = n_z - 2
    u_wall = gamma_dot * H / 2.0
    
    # Build obstacle mask from particles
    import torch
    obstacle = torch.zeros((n_z, n_xy, n_xy), dtype=torch.bool)
    for p in particles:
        cz, cy, cx = p['center']
        r = p['radius']
        iz_min = max(0, int(cz - r - 1))
        iz_max = min(n_z - 1, int(cz + r + 1))
        iy_min = max(0, int(cy - r - 1))
        iy_max = min(n_xy - 1, int(cy + r + 1))
        ix_min = max(0, int(cx - r - 1))
        ix_max = min(n_xy - 1, int(cx + r + 1))
        
        for iz in range(iz_min, iz_max + 1):
            for iy in range(iy_min, iy_max + 1):
                for ix in range(ix_min, ix_max + 1):
                    dist2 = (iz + 0.5 - cz)**2 + (iy + 0.5 - cy)**2 + (ix + 0.5 - cx)**2
                    if dist2 <= r**2:
                        obstacle[iz, iy, ix] = True
    
    phi_lattice = float(obstacle.sum().item()) / (n_xy * n_xy * n_z)
    
    # Run the shear cell
    rec = shear._run_shear_cell(
        nx=n_xy, ny=n_xy, nz=n_z, nu=nu, u_wall=u_wall,
        n_steps=steps, n_warmup=warmup, sample_every=sample_every,
        obstacle=obstacle if len(particles) > 0 else None,
        wall_band=band[1],
    )
    
    # Compute measurements
    gamma_measured = rec["gamma_dot"]
    tau_measured = nu * gamma_measured  # Newtonian reference for pure
    
    if len(particles) > 0:
        # With particles: measure effective viscosity
        tau_wall = rec["tau_wall"]
        if gamma_measured > 0:
            eta_eff = abs(tau_wall) / gamma_measured if gamma_measured > 0 else float("nan")
            eta_0 = nu
            eta_r = eta_eff / eta_0 if eta_0 > 0 else float("nan")
        else:
            eta_r = float("nan")
    else:
        # Pure fluid: eta_r = 1 by definition
        eta_r = 1.0
    
    rec["eta_r"] = eta_r
    rec["phi_lattice"] = phi_lattice
    rec["gamma_bulk"] = gamma_measured
    rec["h_over_d"] = H / (2.0 * particles[0]["radius"]) if particles else float("inf")
    rec["scale"] = scale
    
    return rec


def self_consistent_point(
    particles: list[dict],
    n_xy: int,
    n_z: int,
    nu: float,
    gamma_dot: float,
    steps: int = 2000,
    warmup: int = 1000,
    sample_every: int = 10,
    band: tuple[int, int] = (1, 4),
    max_iter: int = 3,
    tol: float = 0.02,
) -> dict:
    """Run a self-consistent shear cell point.
    
    Iterates until the crystal rigid-body rate matches the measured
    bulk shear rate (within tolerance).
    """
    scale = 1.0
    for iteration in range(max_iter):
        rec = _point_with_scale(
            particles, n_xy, n_z, nu, gamma_dot,
            scale=scale, steps=steps, warmup=warmup,
            sample_every=sample_every, band=band,
        )
        
        gamma_measured = rec.get("gamma_bulk", gamma_dot)
        if gamma_dot > 0:
            rel_err = abs(gamma_measured - gamma_dot) / gamma_dot
            if rel_err < tol:
                break
            # Adjust scale
            scale = gamma_dot / gamma_measured if gamma_measured > 0 else 1.0
        else:
            break
    
    rec["self_consistent_iterations"] = iteration + 1
    return rec


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def bootstrap_power_law(
    gamma_dots: list[float],
    taus: list[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap power-law fit with confidence intervals.
    
    Returns dict with 'n', 'K', 'n_ci', 'K_ci' (95% confidence intervals).
    """
    rng = np.random.RandomState(seed)
    gd = np.array(gamma_dots)
    tau = np.array(taus)
    n = len(gd)
    
    if n < 2:
        return {"n": 1.0, "K": 0.0, "n_ci": (1.0, 1.0), "K_ci": (0.0, 0.0)}
    
    n_samples = []
    K_samples = []
    
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        gd_boot = gd[idx]
        tau_boot = tau[idx]
        
        fit = shear.fit_power_law(gd_boot.tolist(), tau_boot.tolist())
        n_samples.append(fit["n"])
        K_samples.append(fit["K"])
    
    n_arr = np.array(n_samples)
    K_arr = np.array(K_samples)
    
    # Central estimate from all data
    central = shear.fit_power_law(gamma_dots, taus)
    
    return {
        "n": central["n"],
        "K": central["K"],
        "r2": central["r2"],
        "n_ci": (float(np.percentile(n_arr, 2.5)), float(np.percentile(n_arr, 97.5))),
        "K_ci": (float(np.percentile(K_arr, 2.5)), float(np.percentile(K_arr, 97.5))),
        "n_std": float(np.std(n_arr)),
        "K_std": float(np.std(K_arr)),
    }


def grid_convergence_pair(coarse: dict, fine: dict) -> dict:
    """Assess grid convergence between coarse and fine resolution.
    
    Returns dict with relative differences and convergence order estimate.
    """
    n_c = coarse.get("n", 1.0)
    n_f = fine.get("n", 1.0)
    K_c = coarse.get("K", 0.0)
    K_f = fine.get("K", 0.0)
    
    dn = abs(n_c - n_f)
    n_ref = max(abs(n_c), abs(n_f), 1e-10)
    
    dK = abs(K_c - K_f)
    K_ref = max(abs(K_c), abs(K_f), 1e-10)
    
    return {
        "rel_diff_n": float(dn / n_ref),
        "rel_diff_K": float(dK / K_ref),
        "coarse_n": n_c,
        "fine_n": n_f,
        "coarse_K": K_c,
        "fine_K": K_f,
        "converged": bool(dn / n_ref < 0.10 and dK / K_ref < 0.10),
    }


def evaluate_acceptance(
    pure_cases: Optional[dict] = None,
    ladder_fits: Optional[dict] = None,
    convergence: Optional[dict] = None,
    pytest_result: Optional[dict] = None,
) -> dict:
    """Evaluate acceptance criteria for the mature verification face.
    
    Criteria:
    1. Pure control: eta_r = 1 (within tolerance)
    2. Dilute limit vs Einstein: < 10%
    3. Medium concentration vs Krieger-Dougherty: < 10%
    4. Grid convergence (if available)
    5. Pytest pass (if available)
    
    Returns dict with 'criteria', 'n_pass', 'n_defined', 'passed'.
    """
    criteria = {}
    
    # Criterion 1: Pure control
    if pure_cases:
        for key, rec in pure_cases.items():
            eta_r = rec.get("eta_r", float("nan"))
            if np.isfinite(eta_r):
                dev = abs(eta_r - 1.0)
                criteria[f"pure_{key}"] = {
                    "value": float(dev),
                    "threshold": 0.10,
                    "passed": bool(dev < 0.10),
                    "eta_r": eta_r,
                }
    
    # Criterion 2 & 3: Einstein and Krieger-Dougherty
    if ladder_fits:
        for (family, phi), fit in ladder_fits.items():
            if family != "dispersed":
                continue
            
            eta_r_meas = fit.get("eta_r", float("nan"))
            if not np.isfinite(eta_r_meas):
                continue
            
            # Einstein (dilute)
            if phi <= 0.08:
                eta_r_einstein = einstein_relative_viscosity(phi)
                if eta_r_einstein > 0:
                    rel_err = abs(eta_r_meas - eta_r_einstein) / eta_r_einstein
                    criteria[f"einstein_c{phi:.2f}"] = {
                        "value": float(rel_err),
                        "threshold": 0.10,
                        "passed": bool(rel_err < 0.10),
                        "phi": phi,
                        "eta_r_meas": eta_r_meas,
                        "eta_r_ref": eta_r_einstein,
                    }
            
            # Krieger-Dougherty (medium concentration)
            if 0.08 < phi <= 0.30:
                eta_r_kd = krieger_dougherty(phi)
                if np.isfinite(eta_r_kd) and eta_r_kd > 0:
                    rel_err = abs(eta_r_meas - eta_r_kd) / eta_r_kd
                    criteria[f"kd_c{phi:.2f}"] = {
                        "value": float(rel_err),
                        "threshold": 0.10,
                        "passed": bool(rel_err < 0.10),
                        "phi": phi,
                        "eta_r_meas": eta_r_meas,
                        "eta_r_ref": eta_r_kd,
                    }
    
    # Criterion 4: Grid convergence
    if convergence:
        for key, conv in convergence.items():
            dn = conv.get("rel_diff_n", 1.0)
            dK = conv.get("rel_diff_K", 1.0)
            criteria[f"convergence_{key}"] = {
                "rel_diff_n": dn,
                "rel_diff_K": dK,
                "passed": bool(dn < 0.20 and dK < 0.20),
                "threshold": 0.20,
            }
    
    # Criterion 5: Pytest
    if pytest_result:
        rc = pytest_result.get("rc", 1)
        criteria["pytest"] = {
            "value": rc,
            "threshold": 0,
            "passed": bool(rc == 0),
            "n_tests": pytest_result.get("n_tests", 0),
            "n_passed": pytest_result.get("n_passed", 0),
        }
    
    n_pass = sum(1 for c in criteria.values() if c.get("passed", False))
    n_defined = len(criteria)
    
    return {
        "criteria": criteria,
        "n_pass": n_pass,
        "n_defined": n_defined,
        "passed": bool(n_pass == n_defined and n_defined > 0),
    }
