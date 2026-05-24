"""Physical-to-lattice unit converter for TensorLBM.

The standard LBM uses dimensionless (lattice) units.  This module provides
:class:`LBMUnitConverter`, which derives all lattice-unit quantities from the
physical problem parameters and the chosen grid resolution, following the
standard similarity approach:

.. math::

    \\text{Re}_{lb} = \\frac{u_{lb}\\,N_x}{\\nu_{lb}}
                   = \\frac{u_{phys}\\,L_{phys}}{\\nu_{phys}} = \\text{Re}_{phys}

where :math:`N_x` is the number of lattice cells along the reference length.

Usage example
-------------
.. code-block:: python

    from tensorlbm.unit_converter import LBMUnitConverter

    uc = LBMUnitConverter(re=1000.0, l_phys=1.0, u_phys=1.0, nu_phys=1e-3, nx=256)
    print(f"τ = {uc.tau:.4f},  u_lb = {uc.u_lb:.4f},  Ma = {uc.ma:.4f}")

    # Convert physical inlet velocity to lattice units
    u_inlet_lb = uc.phys_to_lb(1.0)

    # Convert a result (lattice velocity) back to physical units
    u_phys_result = uc.lb_to_phys(u_inlet_lb)
"""
from __future__ import annotations

import math
import warnings

__all__ = ["LBMUnitConverter"]

# Speed of sound in lattice units: cs = 1 / sqrt(3)
_CS_LB: float = 1.0 / math.sqrt(3.0)


class LBMUnitConverter:
    """Convert between physical and lattice (dimensionless) LBM units.

    The converter is constructed from the four physical parameters that
    fully define the flow problem plus the grid resolution.  A target
    lattice velocity *u_lb* is chosen so that compressibility errors remain
    small (Ma ≪ 1).

    Parameters
    ----------
    re:
        Reynolds number ``Re = u_phys * L_phys / nu_phys``.
    l_phys:
        Reference length [m] (e.g. cylinder diameter, channel height).
    u_phys:
        Reference velocity [m/s] (e.g. free-stream or inlet speed).
    nu_phys:
        Kinematic viscosity [m² s⁻¹] of the fluid.
    nx:
        Number of lattice cells along the reference length ``L_phys``.
    u_lb:
        Target lattice velocity (dimensionless).  Defaults to 0.05, giving
        Ma ≈ 0.087, well within the low-Mach regime.  Increase carefully;
        values above ~0.15 lead to visible compressibility artefacts.
    ma_warn:
        Issue a :class:`UserWarning` when Ma ≥ this threshold (default 0.1).

    Attributes
    ----------
    dx : float
        Physical grid spacing [m]: ``L_phys / nx``.
    dt : float
        Physical time step [s]: ``u_lb * dx / u_phys``.
    u_lb : float
        Lattice velocity (dimensionless).
    nu_lb : float
        Lattice kinematic viscosity: ``u_lb * nx / Re``.
    tau : float
        BGK relaxation time: ``0.5 + nu_lb / cs_lb²``.
    ma : float
        Mach number: ``u_lb / cs_lb`` where ``cs_lb = 1/√3``.
    """

    def __init__(
        self,
        re: float,
        l_phys: float,
        u_phys: float,
        nu_phys: float,
        nx: int,
        u_lb: float = 0.05,
        ma_warn: float = 0.1,
    ) -> None:
        if re <= 0.0:
            raise ValueError(f"Reynolds number must be positive, got {re}")
        if l_phys <= 0.0:
            raise ValueError(f"l_phys must be positive, got {l_phys}")
        if u_phys <= 0.0:
            raise ValueError(f"u_phys must be positive, got {u_phys}")
        if nu_phys <= 0.0:
            raise ValueError(f"nu_phys must be positive, got {nu_phys}")
        if nx <= 0:
            raise ValueError(f"nx must be positive, got {nx}")
        if not 0.0 < u_lb < _CS_LB:
            raise ValueError(
                f"u_lb must satisfy 0 < u_lb < cs_lb ({_CS_LB:.4f}), got {u_lb}"
            )

        self.re: float = float(re)
        self.l_phys: float = float(l_phys)
        self.u_phys: float = float(u_phys)
        self.nu_phys: float = float(nu_phys)
        self.nx: int = int(nx)

        # --- Grid spacing ---
        self.dx: float = l_phys / nx

        # --- Lattice velocity and derived quantities ---
        self.u_lb: float = float(u_lb)
        self.nu_lb: float = u_lb * nx / re          # Re = u_lb * nx / nu_lb
        self.tau: float = 0.5 + self.nu_lb / (_CS_LB ** 2)  # BGK relaxation time
        self.dt: float = u_lb * self.dx / u_phys    # physical time per lattice step
        self.ma: float = u_lb / _CS_LB              # Mach number

        # --- Consistency warning ---
        re_check = u_phys * l_phys / nu_phys
        if abs(re_check - re) / re > 0.01:
            warnings.warn(
                f"The provided Re={re} differs from u_phys*l_phys/nu_phys={re_check:.4g} "
                "by more than 1 %.  Check your input parameters.",
                stacklevel=2,
            )

        if self.ma >= ma_warn:
            warnings.warn(
                f"Mach number Ma = {self.ma:.4f} ≥ {ma_warn}.  "
                "Compressibility errors may be significant.  "
                "Consider reducing u_lb.",
                stacklevel=2,
            )

        if self.tau < 0.5:  # should be impossible given u_lb > 0, but guard
            raise ValueError(f"Computed τ = {self.tau:.4f} < 0.5.  Check inputs.")

        if self.tau > 2.0:
            warnings.warn(
                f"τ = {self.tau:.4f} > 2.0.  The simulation may be unstable.  "
                "Increase nx or reduce u_lb.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def phys_to_lb(self, v_phys: float) -> float:
        """Convert a physical velocity [m/s] to lattice velocity.

        Uses the relation ``v_lb = v_phys * dt / dx``.

        Args:
            v_phys: Physical velocity in the same units as *u_phys*.

        Returns:
            Equivalent lattice velocity (dimensionless).
        """
        return v_phys * self.dt / self.dx

    def lb_to_phys(self, v_lb: float) -> float:
        """Convert a lattice velocity to physical units [m/s].

        Uses the relation ``v_phys = v_lb * dx / dt``.

        Args:
            v_lb: Lattice velocity (dimensionless).

        Returns:
            Physical velocity in the same units as *u_phys*.
        """
        return v_lb * self.dx / self.dt

    def phys_time_to_steps(self, t_phys: float) -> int:
        """Convert a physical time interval to an integer number of LBM steps.

        Args:
            t_phys: Physical time [s].

        Returns:
            Number of lattice time steps (rounded to nearest integer).
        """
        return round(t_phys / self.dt)

    def steps_to_phys_time(self, n_steps: int) -> float:
        """Convert a number of LBM steps to physical time [s].

        Args:
            n_steps: Number of lattice time steps.

        Returns:
            Physical time in seconds.
        """
        return n_steps * self.dt

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LBMUnitConverter("
            f"Re={self.re}, nx={self.nx}, "
            f"dx={self.dx:.4g}, dt={self.dt:.4g}, "
            f"u_lb={self.u_lb:.4g}, nu_lb={self.nu_lb:.4g}, "
            f"τ={self.tau:.4f}, Ma={self.ma:.4f})"
        )

    def summary(self) -> str:
        """Return a human-readable summary of the unit conversion parameters.

        Returns:
            Multi-line string with all key conversion parameters.
        """
        lines = [
            "LBM Unit Converter",
            "==================",
            f"  Re          = {self.re}",
            f"  L_phys      = {self.l_phys} m",
            f"  u_phys      = {self.u_phys} m/s",
            f"  nu_phys     = {self.nu_phys} m²/s",
            f"  nx          = {self.nx} cells",
            "  ---- lattice units ----",
            f"  dx          = {self.dx:.6g} m/cell",
            f"  dt          = {self.dt:.6g} s/step",
            f"  u_lb        = {self.u_lb:.6g}",
            f"  nu_lb       = {self.nu_lb:.6g}",
            f"  tau         = {self.tau:.6f}",
            f"  Ma          = {self.ma:.6f}",
        ]
        return "\n".join(lines)
