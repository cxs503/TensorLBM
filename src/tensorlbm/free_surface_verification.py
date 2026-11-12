"""LBM Free Surface Multiphase Flow Verification Suite.

This module implements three verification cases for LBM free-surface flows:

1. Dam Break (Ritter 1892): Wave front position vs analytical solution
2. Wigley Hull Wave Pattern: Wave height vs Michell thin-ship theory
3. Water Entry Slamming: Pressure vs Wagner (1932) theory

Plus negative controls:
- Single-phase limit: should match single-phase LBM
- Zero surface tension: interface should diffuse

References
----------
- Ritter (1892) "Die Zerstörung eines Damms"
- Wagner (1932) "Über Stoß- und Gleitvorgänge an der Oberfläche von Flüssigkeiten"
- Michell (1898) "The wave resistance of a ship"
- Martin & Moyce (1952) "Part IV. An experimental study of the collapse of liquid
  columns on a rigid horizontal plane", Phil. Trans. R. Soc. A 244, 312
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np

from .d2q9 import equilibrium, macroscopic
from .solver import stream
from .boundaries import bounce_back_cells
from .color_gradient2d import color_gradient_two_tau_step, cg_phase_field
from .vof_interface_capture import (
    compute_volume_fraction,
    compute_interface_normal,
    compute_curvature,
    advect_vof,
    height_function_curvature,
)
from .csf_force_model import (
    csf_force,
    guo_force_term,
    surface_tension_step,
    laplace_pressure_test,
)

__all__ = [
    "DamBreakVerification",
    "WigleyWaveVerification",
    "WaterEntrySlamming",
    "NegativeControls",
    "run_all_verifications",
]


# =============================================================================
# Dam Break Verification (Ritter 1892)
# =============================================================================

@dataclass
class DamBreakConfig:
    """Configuration for dam break verification."""
    nx: int = 400
    ny: int = 200
    dam_width: int = 100
    rho_heavy: float = 1.0
    rho_light: float = 0.001
    tau: float = 0.8
    g: float = 1e-4
    n_steps: int = 5000
    output_interval: int = 100
    sigma: float = 0.0  # Surface tension (0 for free-surface limit)
    device: str = "cpu"


class DamBreakVerification:
    """Dam break verification: wave front position vs Ritter (1892) solution.

    The Ritter solution for an ideal dam break (no viscosity, no surface tension)
    gives the wave front position as:

        x_front(t) = L + t * sqrt(2*g*h0)

    where L is the initial dam width, h0 is the initial water depth, and g is
    gravity. In non-dimensional form:

        X* = x_front / L
        t* = t * sqrt(2*g/L)

        X*(t*) = 1 + t*  (for t* < 2)

    The actual front position is tracked from the volume fraction field.
    """

    def __init__(self, config: DamBreakConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.results: Dict = {}

    def ritter_solution(self, t_star: np.ndarray) -> np.ndarray:
        """Analytical Ritter solution for dam break front position.

        Args:
            t_star: Non-dimensional time t* = t * sqrt(2g/L)

        Returns:
            Non-dimensional front position X* = x_front / L
        """
        # Ritter (1892): X* = 1 + t* for t* < 2
        # After t* = 2, the front decelerates
        X_star = np.where(t_star < 2.0, 1.0 + t_star, 1.0 + 2.0 - 0.5 * (t_star - 2.0)**2 / 2.0)
        return X_star

    def initialize(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize dam break simulation.

        Returns:
            (f_r, f_b, alpha): Red/blue distributions and volume fraction
        """
        cfg = self.config
        ny, nx = cfg.ny, cfg.nx

        # Volume fraction: 1 inside dam, 0 outside
        alpha = torch.zeros((ny, nx), device=self.device)
        alpha[:, :cfg.dam_width] = 1.0

        # Smooth interface
        x = torch.arange(nx, device=self.device, dtype=torch.float32)
        interface = 0.5 * (1.0 - torch.tanh((x - cfg.dam_width) / 3.0))
        alpha = interface.unsqueeze(0).expand(ny, nx).clone()

        # Initialize color gradient distributions
        rho_r = cfg.rho_heavy * alpha + cfg.rho_heavy * 0.01 * (1 - alpha)
        rho_b = cfg.rho_light * 0.01 * alpha + cfg.rho_light * (1 - alpha)

        zero = torch.zeros((ny, nx), device=self.device)
        f_r = equilibrium(rho_r, zero, zero)
        f_b = equilibrium(rho_b, zero, zero)

        return f_r, f_b, alpha

    def track_front(self, alpha: torch.Tensor, threshold: float = 0.5) -> float:
        """Track the wave front position from volume fraction field.

        Args:
            alpha: Volume fraction field
            threshold: Front detection threshold

        Returns:
            Front x-position (column index)
        """
        # Average over interior rows (exclude walls)
        alpha_avg = alpha[1:-1, :].mean(dim=0)
        # Find rightmost column where alpha > threshold
        cols = (alpha_avg > threshold).nonzero(as_tuple=True)[0]
        if len(cols) == 0:
            return 0.0
        return float(cols.max().item())

    def run(self) -> Dict:
        """Run dam break verification.

        Returns:
            Dictionary with results and comparison to Ritter solution
        """
        cfg = self.config
        f_r, f_b, alpha = self.initialize()

        # Wall mask
        wall = torch.zeros((cfg.ny, cfg.nx), dtype=torch.bool, device=self.device)
        wall[0, :] = True
        wall[-1, :] = True
        wall[:, 0] = True
        wall[:, -1] = True

        # Gravity
        gy = -cfg.g

        # Tracking
        front_positions = []
        times = []
        t_stars = []

        L = float(cfg.dam_width)
        sqrt_coeff = math.sqrt(2.0 * cfg.g / L) if cfg.g > 0 else 1.0

        for step in range(1, cfg.n_steps + 1):
            # Color gradient step with gravity
            f_r, f_b = color_gradient_two_tau_step(
                f_r, f_b,
                tau_r=cfg.tau, tau_b=cfg.tau,
                A=cfg.sigma * 0.04 if cfg.sigma > 0 else 0.0,
                beta=0.7,
                gx=0.0, gy=gy,
                solid_mask=wall,
            )

            # Streaming
            f_r = stream(f_r)
            f_b = stream(f_b)

            # Bounce-back at walls
            f_r = bounce_back_cells(f_r, wall)
            f_b = bounce_back_cells(f_b, wall)

            # Update volume fraction
            rho_r, rho_b, rho, phi = cg_phase_field(f_r, f_b)
            alpha = 0.5 * (1.0 + phi)

            # Track front
            if step % cfg.output_interval == 0:
                x_front = self.track_front(alpha)
                t = step
                t_star = t * sqrt_coeff
                X_star = x_front / L

                front_positions.append(x_front)
                times.append(t)
                t_stars.append(t_star)

        # Compare with Ritter solution
        t_star_arr = np.array(t_stars)
        X_star_sim = np.array(front_positions) / L
        X_star_ritter = self.ritter_solution(t_star_arr)

        # Compute error
        valid = t_star_arr < 2.0  # Only compare in Ritter-valid region
        if valid.sum() > 0:
            error = np.abs(X_star_sim[valid] - X_star_ritter[valid])
            mean_error = float(error.mean())
            max_error = float(error.max())
        else:
            mean_error = float('nan')
            max_error = float('nan')

        self.results = {
            "case": "dam_break_ritter",
            "config": asdict(cfg),
            "front_positions": front_positions,
            "times": times,
            "t_star": t_stars,
            "X_star_sim": X_star_sim.tolist(),
            "X_star_ritter": X_star_ritter.tolist(),
            "mean_error": mean_error,
            "max_error": max_error,
            "verdict": "PASS" if mean_error < 0.15 else "FAIL",
        }

        return self.results


# =============================================================================
# Wigley Hull Wave Pattern Verification
# =============================================================================

@dataclass
class WigleyWaveConfig:
    """Configuration for the Wigley hull geometry / thin-ship identity case."""
    L: float = 100.0  # Ship length (cells)
    B: float = 10.0   # Beam
    T: float = 6.25   # Draft
    Fr: float = 0.3   # Froude number
    nx: int = 400
    ny: int = 100
    nz: int = 50
    n_steps: int = 10000
    output_interval: int = 500
    device: str = "cpu"


class WigleyWaveVerification:
    """Wigley hull: analytic geometry check + thin-ship dispersion **identity**.

    The Wigley hull is a mathematically defined ship form

        y(x, z) = (B/2) (1 - (2x/L)^2) (1 - (z/T)^2),   x in [-L/2, L/2], z in [-T, 0]

    **What is verified here.**  Only the geometry: the block coefficient implied
    by :meth:`wigley_geometry` integrates to the analytic ``Cb = 4/9``
    (``V = 4 L B T / 9``) with second-order grid convergence.  That check is
    falsifiable and is reported as ``geometry_check``.

    **What is *not* verified here.**  No wave field is simulated.  ``speed_ratio``
    is an algebraic identity, not a test: ``lambda = 2 pi U^2 / g`` (assumed),
    ``g = 1``, ``U = Fr sqrt(L)`` give ``k = 1/U^2``, ``omega = sqrt(g k) = 1/U``,
    ``c = omega/k = U``, hence ``U/c ~ 1`` to machine precision for *any* ``Fr``,
    ``L``.  The old revision called this ``dispersion_check`` and returned
    ``verdict="PASS"``; it is now reported as an identity residual and the
    wave-pattern claim is ``WITHHELD`` (no Michell/thin-ship simulation exists in
    this module).  See ``docs/AUDIT_m16285_FREE_SURFACE_VERIFICATION.md``.
    """

    #: Analytic block coefficient of the Wigley form, ``V/(L B T) = 4/9``.
    CB_ANALYTIC = 4.0 / 9.0

    def __init__(self, config: WigleyWaveConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.results: Dict = {}

    def wigley_geometry(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute Wigley hull half-breadth.

        Args:
            x: Longitudinal position (centered at midship)
            z: Vertical position (z=0 at waterline, z=-T at keel)

        Returns:
            Half-breadth y(x,z)
        """
        cfg = self.config
        x_norm = 2.0 * x / cfg.L
        z_norm = z / cfg.T
        y = (cfg.B / 2.0) * (1.0 - x_norm**2) * (1.0 - z_norm**2)
        # Zero outside hull bounds
        mask = (x_norm.abs() <= 1.0) & (z_norm >= -1.0) & (z_norm <= 0.0)
        return torch.where(mask, y, torch.zeros_like(y))

    def block_coefficient(self, nx: int, nz: int) -> float:
        """Displacement block coefficient of the *discretised* hull.

        Midpoint quadrature of ``2 y(x, z)`` over ``x in [-L/2, L/2]``,
        ``z in [-T, 0]`` normalised by ``L B T``.  The integrand is a product of
        two quadratics, so this converges at second order to ``CB_ANALYTIC``.
        """
        cfg = self.config
        dx = cfg.L / nx
        dz = cfg.T / nz
        x = torch.linspace(-cfg.L / 2 + dx / 2, cfg.L / 2 - dx / 2, nx, dtype=torch.float64)
        z = torch.linspace(-cfg.T + dz / 2, -dz / 2, nz, dtype=torch.float64)
        zz, xx = torch.meshgrid(z, x, indexing="ij")
        y = self.wigley_geometry(xx, zz)
        volume = float(2.0 * y.sum() * dx * dz)
        return volume / (cfg.L * cfg.B * cfg.T)

    def michell_wave_wavelength(self) -> float:
        """Transverse wavelength ``lambda = 2 pi U^2 / g`` with ``g = 1``."""
        cfg = self.config
        U = cfg.Fr * math.sqrt(cfg.L)
        lambda_wave = 2.0 * math.pi * U**2
        return lambda_wave

    def michell_wave_amplitude(self) -> float:
        """Order-of-magnitude wave amplitude ``H ~ 0.5 (B^2/L) Fr^2`` (not a benchmark)."""
        cfg = self.config
        H = (cfg.B**2 / cfg.L) * cfg.Fr**2 * 0.5
        return H

    def geometry_check(self, grids=((200, 25), (400, 50), (800, 100))) -> Dict:
        """Block-coefficient convergence of :meth:`block_coefficient` to ``4/9``."""
        rows = []
        for nx, nz in grids:
            cb = self.block_coefficient(nx, nz)
            rows.append(
                {
                    "nx": nx,
                    "nz": nz,
                    "cb": cb,
                    "rel_error": abs(cb - self.CB_ANALYTIC) / self.CB_ANALYTIC,
                }
            )
        errs = [r["rel_error"] for r in rows]
        monotone = all(b <= a for a, b in zip(errs, errs[1:]))
        # observed order between the two finest grids (h halves in both directions)
        order = float("nan")
        if len(errs) >= 2 and errs[-1] > 0 and errs[-2] > 0:
            order = math.log(errs[-2] / errs[-1]) / math.log(2.0)
        finest = errs[-1]
        return {
            "quantity": "Cb(discretised Wigley geometry) vs analytic 4/9",
            "cb_analytic": self.CB_ANALYTIC,
            "rows": rows,
            "monotone_convergence": monotone,
            "observed_order": order,
            "finest_rel_error": finest,
            "tolerance": 5e-3,
            "pass": bool(monotone and finest < 5e-3),
        }

    def run(self) -> Dict:
        """Run the geometry check and report the dispersion **identity** honestly."""
        cfg = self.config

        lambda_theory = self.michell_wave_wavelength()
        H_theory = self.michell_wave_amplitude()
        k = 2.0 * math.pi / lambda_theory
        U = cfg.Fr * math.sqrt(cfg.L)
        g = 1.0  # Normalized gravity
        omega = math.sqrt(g * k)
        c_phase = omega / k
        speed_ratio = U / c_phase if c_phase > 0 else float("nan")

        geometry = self.geometry_check()

        self.results = {
            "case": "wigley_wave_pattern",
            "config": asdict(cfg),
            "evidence_class": "algebraic identity (dispersion) + geometry quadrature",
            "solver_invoked": False,
            "simulation_verified": False,
            "theoretical_wavelength": lambda_theory,
            "theoretical_amplitude": H_theory,
            "wave_number": k,
            "ship_speed": U,
            "phase_speed": c_phase,
            "speed_ratio": speed_ratio,
            # kept for API compatibility; NOT a verification (see class docstring)
            "dispersion_check": abs(speed_ratio - 1.0) < 1e-12,
            "dispersion_check_kind": "algebraic_identity",
            "dispersion_identity_residual": abs(speed_ratio - 1.0),
            "geometry_check": geometry,
            "geometry_verdict": "PASS" if geometry["pass"] else "FAIL",
            "verdict": "WITHHELD",
            "verdict_reason": (
                "no wave field is simulated: Michell/thin-ship wave-pattern "
                "comparison is WITHHELD; only the geometry (Cb -> 4/9) and the "
                "dispersion identity are evaluated"
            ),
            "note": "Full 3D simulation required for quantitative wave height comparison",
        }

        return self.results


# =============================================================================
# Water Entry Slamming Verification (Wagner 1932)
# =============================================================================

@dataclass
class WaterEntryConfig:
    """Configuration for the Wagner (1932) wedge-entry closed forms.

    .. warning::
       ``nx``, ``ny``, ``n_steps``, ``output_interval`` and ``device`` are kept
       for signature compatibility with the other cases of this module only.
       :class:`WaterEntrySlamming` runs **no flow solver** (this module has no
       LBM water-entry simulation), so the grid/device fields are inert;
       ``n_steps`` and ``output_interval`` are reused as the *sampling grid* of
       the analytic time series in :meth:`WaterEntrySlamming.run`.
    """

    wedge_angle: float = 4.0  # Deadrise angle (degrees)
    V_entry: float = 0.01     # Entry velocity
    rho_water: float = 1.0
    rho_air: float = 0.001
    nx: int = 200
    ny: int = 150
    n_steps: int = 2000
    output_interval: int = 50
    device: str = "cpu"
    #: spray-root regularisation of the pointwise pressure (the inviscid Wagner
    #: pressure is singular at ``xi -> +-1``); the *integral* quantities are
    #: evaluated without it (see :meth:`WaterEntrySlamming.wagner_force_by_quadrature`).
    xi_clip: float = 1.0 - 1e-6


class WaterEntrySlamming:
    """Wagner (1932) wedge water entry — closed forms plus analytic self-consistency.

    Closed-form set (2-D, per unit span, constant entry velocity ``V``, deadrise
    ``beta``) — the same set implemented independently in
    :mod:`tensorlbm.lng_membrane_sloshing.impact` (eqs. (1)-(4) of that module):

    ===========================  ==================================================
    wetted half-width           ``c(t) = (pi/2) V t / tan(beta)``
    pressure (``xi = x/c``)      ``p = rho V^2 (pi / (2 tan beta)) / sqrt(1-xi^2)``
                                 ``  = rho V cdot(t) / sqrt(1-xi^2)``
    slam force                   ``F(t) = int_{-c}^{c} p dx = rho V^2 pi^2 c / (2 tan beta)``
    force slope                  ``dF/dt = rho V^3 pi^3 / (4 tan^2(beta))``
    ===========================  ==================================================

    **What this class does and does not certify.**  It evaluates the closed forms
    and cross-checks them against three mutually independent paths (pressure
    quadrature, the `-rho dphi/dt` finite difference of the plate potential, and
    the separate :mod:`~tensorlbm.lng_membrane_sloshing.impact` implementation).
    It does *not* compare an LBM simulation with Wagner theory: no such
    simulation is run here, so ``simulation_verified`` is ``False`` and the
    LBM-vs-theory calibration stays ``WITHHELD``.

    History (see ``docs/AUDIT_m16285_FREE_SURFACE_VERIFICATION.md``): the earlier
    revision of this class (a) reported ``measured_force_slope`` computed *from*
    ``wagner_force_slope`` (a tautology: ``relative_error`` was 1.4e-16 by
    construction, so the case could never fail) and (b) carried two mutually
    inconsistent closed forms — ``p = rho V^2 / sqrt(1-xi^2)`` (missing the
    ``cdot/V = pi/(2 tan beta)`` factor) and ``dF/dt = rho V^3 pi^3/(8 tan^2 beta)``
    (a factor 2 too small: the module's own pressure integral came out
    ``(4 tan beta)/pi`` times its own force, 8.9 % at ``beta = 4 deg``).
    """

    def __init__(self, config: WaterEntryConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.results: Dict = {}

    # -- kinematics ---------------------------------------------------------

    def wagner_wetted_halfwidth(self, t: float) -> float:
        """Wagner wetted half-width ``c(t) = (pi/2) V t / tan(beta)``."""
        if t <= 0.0:
            return 0.0
        tan_beta = math.tan(math.radians(self.config.wedge_angle))
        return 0.5 * math.pi * self.config.V_entry * t / tan_beta

    def wagner_force(self, t: float) -> float:
        """Wagner slam force per unit span ``F(t) = rho V^2 pi^2 c(t)/(2 tan beta)``."""
        tan_beta = math.tan(math.radians(self.config.wedge_angle))
        c = self.wagner_wetted_halfwidth(t)
        return self.config.rho_water * self.config.V_entry**2 * math.pi**2 * c / (
            2.0 * tan_beta
        )

    def wagner_force_slope(self) -> float:
        """Wagner force slope ``dF/dt = rho V^3 pi^3 / (4 tan^2(beta))``.

        Follows from ``F = rho V^2 pi^2 c/(2 tan beta)`` with
        ``cdot = pi V/(2 tan beta)``.  Independent implementation to compare
        against: :func:`tensorlbm.lng_membrane_sloshing.impact.wagner_slam_force_per_length`.
        """
        tan_beta = math.tan(math.radians(self.config.wedge_angle))
        rho = self.config.rho_water
        return rho * self.config.V_entry**3 * math.pi**3 / (4.0 * tan_beta**2)

    # -- pressure -----------------------------------------------------------

    def wagner_pressure_distribution(self, x: np.ndarray, t: float) -> np.ndarray:
        """Wagner pressure on the wedge, ``p(x, t) [Pa]``.

        ``p = rho V^2 (pi / (2 tan beta)) / sqrt(1 - (x/c)^2)`` for ``|x| < c(t)``
        and ``0`` outside; the spray-root singularity is regularised at
        ``|x/c| = xi_clip``.
        """
        cfg = self.config
        x = np.asarray(x, dtype=float)
        c = self.wagner_wetted_halfwidth(t)
        if c <= 0.0:
            return np.zeros_like(x)
        tan_beta = math.tan(math.radians(cfg.wedge_angle))
        xi = np.clip(x / c, -cfg.xi_clip, cfg.xi_clip)
        p = cfg.rho_water * cfg.V_entry**2 * (0.5 * math.pi / tan_beta) / np.sqrt(
            1.0 - xi * xi
        )
        return np.where(np.abs(x) < c, p, 0.0)

    def wagner_plate_potential(self, x: np.ndarray, t: float) -> np.ndarray:
        """Potential on the wetted plate, ``phi = -V sqrt(c(t)^2 - x^2)`` (``|x| < c``).

        This is the boundary value of the mixed-problem solution
        ``w = -i V (z - sqrt(z^2 - c^2))`` (see
        :func:`tensorlbm.lng_membrane_sloshing.impact.wagner_complex_potential`).
        """
        x = np.asarray(x, dtype=float)
        c = self.wagner_wetted_halfwidth(t)
        if c <= 0.0:
            return np.zeros_like(x)
        inner = np.clip(c * c - x * x, 0.0, None)
        phi = -self.config.V_entry * np.sqrt(inner)
        return np.where(np.abs(x) < c, phi, 0.0)

    def pressure_from_potential(self, x: np.ndarray, t: float, dtau_rel: float = 1e-4):
        """Pressure from linearised unsteady Bernoulli, ``p = -rho d phi/dt``.

        Central difference in ``t`` at fixed ``x``: this path uses *only* the
        plate potential, never the closed-form pressure, so it is an independent
        check of the ``cdot`` factor.
        """
        dt = dtau_rel * t
        phi_p = self.wagner_plate_potential(x, t + dt)
        phi_m = self.wagner_plate_potential(x, t - dt)
        return -self.config.rho_water * (phi_p - phi_m) / (2.0 * dt)

    # -- integrals ----------------------------------------------------------

    QUADRATURE_N = 1000001

    def wagner_force_by_quadrature(self, t: float, n: int = 0) -> float:
        """Slam force by midpoint quadrature of the pressure profile.

        Midpoint sampling in ``xi`` never evaluates the spray-root singularity,
        so the integrable ``1/sqrt(1-xi^2)`` endpoint behaviour is integrated
        rather than clipped: this is an independent numerical path to ``F(t)``
        (it would have exposed the old factor-2 defect).
        """
        n = int(n or self.QUADRATURE_N)
        c = self.wagner_wetted_halfwidth(t)
        if c <= 0.0:
            return 0.0
        tan_beta = math.tan(math.radians(self.config.wedge_angle))
        h = 2.0 / n
        xi = -1.0 + (np.arange(n) + 0.5) * h
        p = (
            self.config.rho_water
            * self.config.V_entry**2
            * (0.5 * math.pi / tan_beta)
            / np.sqrt(1.0 - xi * xi)
        )
        return float(p.sum() * h * c)

    # -- driver -------------------------------------------------------------

    def run(self) -> Dict:
        """Evaluate the closed forms and run the three self-consistency checks.

        Returns:
            Dictionary with the analytic series, the quadrature-measured force
            series, the check residuals and an honest ``verdict`` scope.
        """
        cfg = self.config
        tan_beta = math.tan(math.radians(cfg.wedge_angle))

        # Sample grid of the analytic time series (n_steps/output_interval are
        # reuse-as-sampling, not solver steps: see WaterEntryConfig).
        stride = max(int(cfg.output_interval), 1)
        times = [float(s) for s in range(stride, cfg.n_steps + 1, stride)]
        forces_theory = [self.wagner_force(t) for t in times]
        forces_quad = [self.wagner_force_by_quadrature(t) for t in times]
        penetrations = [cfg.V_entry * t for t in times]

        # (1) measurement: least-squares slope of the *quadrature* series
        if len(times) > 2:
            t_arr = np.array(times)
            F_arr = np.array(forces_quad)
            n = len(t_arr)
            slope = float(
                (n * (t_arr * F_arr).sum() - t_arr.sum() * F_arr.sum())
                / (n * (t_arr**2).sum() - t_arr.sum() ** 2)
            )
            intercept = float((F_arr.sum() - slope * t_arr.sum()) / n)
            F_pred = slope * t_arr + intercept
            ss_res = float(((F_arr - F_pred) ** 2).sum())
            ss_tot = float(((F_arr - F_arr.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            dF_dt_theory = self.wagner_force_slope()
            closure_err = abs(slope - dF_dt_theory) / dF_dt_theory
        else:
            slope = float("nan")
            r2 = float("nan")
            dF_dt_theory = self.wagner_force_slope()
            closure_err = float("nan")

        # (2) cross-implementation check against lng_membrane_sloshing.impact
        cross = self._cross_check_impact()

        # (3) pressure from -rho dphi/dt at a mid-time window
        t_probe = times[len(times) // 2] if times else 0.0
        p_fd = self.pressure_from_potential(self._probe_x(t_probe), t_probe)
        p_cf = self.wagner_pressure_distribution(self._probe_x(t_probe), t_probe)
        pot_err = float(np.max(np.abs(p_fd - p_cf) / p_cf))

        checks = {
            "pressure_integral_closure": {
                "quantity": "dF/dt from quadrature of p(x,t) vs closed form",
                "relative_error": closure_err,
                "quadrature_n": int(self.QUADRATURE_N),
                "tolerance": 2e-3,
                "pass": bool(np.isfinite(closure_err) and closure_err < 2e-3),
            },
            "cross_implementation": cross,
            "potential_time_derivative": {
                "quantity": "max_x |p(-rho dphi/dt) - p_closed| / p_closed, |x| < 0.9 c",
                "relative_error": pot_err,
                "tolerance": 1e-5,
                "pass": bool(np.isfinite(pot_err) and pot_err < 1e-5),
            },
        }
        analytic_ok = all(c["pass"] for c in checks.values())

        self.results = {
            "case": "water_entry_slamming_wagner",
            "config": asdict(cfg),
            "evidence_class": "analytic_closure (no flow solver)",
            "solver_invoked": False,
            "simulation_verified": False,
            "calibration_status": "WITHHELD",
            "calibration_status_reason": (
                "no LBM water-entry simulation is run by this class: the "
                "Wagner-vs-simulation calibration stays WITHHELD"
            ),
            "wagner_force_slope": dF_dt_theory,
            "wagner_force_slope_formula": "rho V^3 pi^3 / (4 tan^2 beta)",
            "tan_beta": tan_beta,
            "measured_force_slope": slope,
            "measured_from": "quadrature of p(x,t) (independent of the closed form)",
            "relative_error": closure_err,
            "fit_r2": r2,
            "checks": checks,
            "times": times,
            "forces": forces_quad,
            "forces_closed_form": forces_theory,
            "penetrations": penetrations,
            "verdict": "PASS" if analytic_ok else "FAIL",
            "verdict_scope": "analytic self-consistency only",
        }
        return self.results

    # -- internals ----------------------------------------------------------

    def _probe_x(self, t: float) -> np.ndarray:
        c = self.wagner_wetted_halfwidth(t)
        return np.linspace(-0.9 * c, 0.9 * c, 51)

    def _cross_check_impact(self) -> Dict:
        """Compare pressure/force against the independent Wagner implementation."""
        key = {
            "quantity": "p(x,t) and F(t) vs lng_membrane_sloshing.impact",
            "tolerance": 1e-9,
        }
        try:
            from .lng_membrane_sloshing.impact import (
                WagnerImpact,
                wagner_pressure_profile,
                wagner_slam_force_per_length,
                wagner_wetted_halfwidth,
            )
        except Exception as exc:  # pragma: no cover - import guard
            return {**key, "available": False, "reason": repr(exc), "pass": False}

        cfg = self.config
        imp = WagnerImpact(
            v_impact=cfg.V_entry, beta_deg=cfg.wedge_angle, rho=cfg.rho_water
        )
        t = max(float(cfg.n_steps), 1.0)
        c_ours = self.wagner_wetted_halfwidth(t)
        c_ref = wagner_wetted_halfwidth(t, imp)
        x = self._probe_x(t)
        p_ours = self.wagner_pressure_distribution(x, t)
        p_ref = wagner_pressure_profile(x, t, imp)
        F_ours = self.wagner_force(t)
        F_ref = wagner_slam_force_per_length(t, imp)
        p_err = float(np.max(np.abs(p_ours - p_ref) / p_ref))
        F_err = abs(F_ours - F_ref) / F_ref
        c_err = abs(c_ours - c_ref) / c_ref
        err = max(p_err, F_err, c_err)
        return {
            **key,
            "available": True,
            "wetted_halfwidth_rel_error": c_err,
            "pressure_rel_error": p_err,
            "force_rel_error": F_err,
            "relative_error": err,
            "pass": bool(err < key["tolerance"]),
        }


# =============================================================================
# Negative Controls
# =============================================================================

class NegativeControls:
    """Negative controls for free surface verification.

    1. Single-phase limit: With alpha=1 everywhere, should match single-phase LBM
    2. Zero surface tension: Interface should diffuse without surface tension
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.results: Dict = {}

    def single_phase_limit(self) -> Dict:
        """Test single-phase limit: alpha=1 everywhere should match single-phase LBM.

        Returns:
            Dictionary with comparison metrics
        """
        ny, nx = 50, 100
        tau = 0.7

        # Initialize with uniform liquid
        alpha = torch.ones((ny, nx), device=self.device)
        rho = torch.ones((ny, nx), device=self.device)
        u = torch.zeros((ny, nx), device=self.device)
        v = torch.zeros((ny, nx), device=self.device)

        # Add small perturbation
        u[ny//2, :] = 0.01

        f = equilibrium(rho, u, v)

        # Run a few steps with color gradient (should behave like single-phase)
        f_r = f.clone()
        f_b = torch.zeros_like(f)

        for _ in range(100):
            f_r, _ = color_gradient_two_tau_step(
                f_r, f_b, tau_r=tau, tau_b=tau,
                A=0.0, beta=0.0,  # No surface tension, no segregation
            )
            f_r = stream(f_r)

        # Check mass conservation
        mass_initial = float(f.sum())
        mass_final = float(f_r.sum())
        mass_drift = abs(mass_final - mass_initial) / mass_initial

        # Check velocity field (should be preserved)
        _, u_final, v_final = macroscopic(f_r)
        u_max_diff = float((u_final - u).abs().max())

        self.results["single_phase_limit"] = {
            "mass_drift_rel": mass_drift,
            "velocity_preservation": u_max_diff,
            "verdict": "PASS" if mass_drift < 1e-6 and u_max_diff < 0.01 else "FAIL",
        }

        return self.results["single_phase_limit"]

    def zero_surface_tension(self) -> Dict:
        """Test zero surface tension: interface should diffuse.

        Returns:
            Dictionary with interface diffusion metrics
        """
        ny, nx = 50, 100
        tau = 0.7

        # Initialize with sharp interface
        alpha = torch.zeros((ny, nx), device=self.device)
        alpha[:, :nx//2] = 1.0

        rho_r = alpha.clone()
        rho_b = (1.0 - alpha).clone() + 0.01

        f_r = equilibrium(rho_r, torch.zeros_like(rho_r), torch.zeros_like(rho_r))
        f_b = equilibrium(rho_b, torch.zeros_like(rho_b), torch.zeros_like(rho_b))

        # Measure initial interface width
        alpha_initial = rho_r / (rho_r + rho_b)
        width_initial = self._measure_interface_width(alpha_initial)

        # Run with zero surface tension
        for _ in range(500):
            f_r, f_b = color_gradient_two_tau_step(
                f_r, f_b, tau_r=tau, tau_b=tau,
                A=0.0, beta=0.0,  # Zero surface tension
            )
            f_r = stream(f_r)
            f_b = stream(f_b)

        # Measure final interface width
        rho_r_final, rho_b_final, _, _ = cg_phase_field(f_r, f_b)
        alpha_final = 0.5 * (1.0 + (rho_r_final - rho_b_final) / (rho_r_final + rho_b_final + 1e-12))
        width_final = self._measure_interface_width(alpha_final)

        # Interface should have diffused
        diffusion_ratio = width_final / width_initial

        self.results["zero_surface_tension"] = {
            "initial_width": width_initial,
            "final_width": width_final,
            "diffusion_ratio": diffusion_ratio,
            "verdict": "PASS" if diffusion_ratio > 1.5 else "FAIL",
        }

        return self.results["zero_surface_tension"]

    def _measure_interface_width(self, alpha: torch.Tensor) -> float:
        """Measure interface width (distance between alpha=0.1 and alpha=0.9)."""
        # Average over y
        alpha_avg = alpha.mean(dim=0)
        # Find positions where alpha crosses 0.1 and 0.9
        x = torch.arange(len(alpha_avg), device=alpha.device)
        idx_01 = (alpha_avg > 0.1).nonzero(as_tuple=True)[0]
        idx_09 = (alpha_avg > 0.9).nonzero(as_tuple=True)[0]
        if len(idx_01) > 0 and len(idx_09) > 0:
            return float(idx_09[0] - idx_01[-1])
        return 0.0


# =============================================================================
# Main Verification Runner
# =============================================================================

def summarise_verifications(results: Dict) -> Dict:
    """Aggregate case verdicts without inflating certification.

    Rules (see ``docs/AUDIT_m16285_FREE_SURFACE_VERIFICATION.md``):

    * a case is *certified* only when it ran a solver and returned ``PASS``;
    * ``WITHHELD`` cases stay ``WITHHELD`` (never folded into ``all_pass``);
    * analytic-only cases are listed separately (they are self-consistency
      checks of closed forms, not simulation validations);
    * the negative controls report their own verdicts instead of a hard-coded
      ``"PASS"``.
    """
    def _verdict(name: str) -> str:
        entry = results.get(name) or {}
        return str(entry.get("verdict", "MISSING"))

    negative = results.get("negative_controls") or {}
    negative_verdicts = {
        key: (val or {}).get("verdict", "MISSING") for key, val in negative.items()
    }
    negative_ok = bool(negative_verdicts) and all(
        v == "PASS" for v in negative_verdicts.values()
    )

    analytic_only = [
        name
        for name, entry in results.items()
        if isinstance(entry, dict) and entry.get("solver_invoked") is False
    ]
    withheld = [
        name for name in results if _verdict(name) == "WITHHELD"
    ]
    solver_cases = {
        "dam_break": _verdict("dam_break"),
        "negative_controls": "PASS" if negative_ok else "FAIL",
    }
    all_pass = all(v == "PASS" for v in solver_cases.values())

    return {
        "all_pass": all_pass,
        "scope": "solver-backed cases only (dam_break, negative_controls)",
        "solver_cases": solver_cases,
        "negative_control_cases": negative_verdicts,
        "analytic_only": analytic_only,
        "withheld": withheld,
        "case_verdicts": {
            "dam_break": _verdict("dam_break"),
            "wigley_wave": _verdict("wigley_wave"),
            "water_entry": _verdict("water_entry"),
        },
    }


def run_all_verifications(output_dir: Path = Path("runs/free_surface_verification")) -> Dict:
    """Run all free surface verification cases.

    Args:
        output_dir: Directory to save results

    Returns:
        Dictionary with all verification results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. Dam break verification (solver-backed: color-gradient LBM vs Ritter)
    print("Running dam break verification...")
    dam_cfg = DamBreakConfig(nx=200, ny=100, dam_width=50, n_steps=2000, output_interval=50)
    dam = DamBreakVerification(dam_cfg)
    results["dam_break"] = dam.run()

    # 2. Wigley hull geometry + dispersion identity (NO solver -> WITHHELD)
    print("Running Wigley hull wave verification...")
    wig_cfg = WigleyWaveConfig(L=80, B=8, T=5, Fr=0.3)
    wig = WigleyWaveVerification(wig_cfg)
    results["wigley_wave"] = wig.run()

    # 3. Water entry slamming (analytic closure only, NO solver)
    print("Running water entry slamming verification...")
    we_cfg = WaterEntryConfig(wedge_angle=4.0, V_entry=0.01, n_steps=1000)
    we = WaterEntrySlamming(we_cfg)
    results["water_entry"] = we.run()

    # 4. Negative controls (solver-backed)
    print("Running negative controls...")
    neg = NegativeControls()
    results["negative_controls"] = {
        "single_phase": neg.single_phase_limit(),
        "zero_sigma": neg.zero_surface_tension(),
    }

    results["summary"] = summarise_verifications(results)

    # Save results
    with open(output_dir / "verification_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nVerification complete. Results saved to {output_dir}")
    print(f"Summary: {results['summary']}")

    return results
