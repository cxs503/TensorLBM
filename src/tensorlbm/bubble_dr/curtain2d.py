"""Gas-curtain / air-layer plane channel with buoyancy: coverage -> drag reduction.

Campaign node IN57-N02, criteria (2)(3)(5).

Set-up (lattice units, ``x`` streamwise, ``y`` wall-normal)
-----------------------------------------------------------
* the **hull plate is the ceiling** (row ``ny-1``, half-way bounce-back) and row
  ``0`` is the outer no-slip boundary; ``x`` is periodic (closed recirculating
  channel).  This orientation is what the physics demands: a real hull bottom
  faces downwards, so the buoyancy of the gas presses it *against* the hull.
  (Putting the plate on the floor of the domain -- the obvious first guess --
  makes the layer Rayleigh-Taylor unstable and the coverage then has nothing to
  do with the gas supply.)
* water is a Shan-Chen single-component fluid with a Peng-Robinson EOS (the
  coexistence densities and the Laplace law of this very kernel set were gated in
  :mod:`tensorlbm.gas_layer.laplace`);
* gas enters through ``n_holes`` injection strips of width ``d`` cells at plate
  positions ``x_h``, at a **booked** volume rate ``Q`` per unit width -- the
  mass source is the only place where mass is created and it is logged;
* the plate is non-wetting (``wall_psi`` low) so the gas is retained;
* a **net-zero buoyancy** force ``F_y = (rho - rho_ref) g_b`` acts on the mixture.
  In a closed channel the uniform part of gravity is balanced by the pressure
  gradient (it only sets the hydrostatic stratification), so the buoyancy that
  actually moves the gas relative to the liquid is the net-zero part.  This term
  is what makes the coverage non-monotone in Froude number: gas is lost off the
  plate by buoyant rise (dominant at low speed) and by shear thinning of the
  layer (dominant at high speed).

Drag reduction is defined at **matched throughput**: an integral controller holds
``<u_x> = U_target`` in both the single-phase reference and the two-phase run, and
the same momentum-exchange estimator is used on both, ``DR = 1 - tau_w/tau_w0``.

Coverage is ``C_f = A_gas / A_bottom`` evaluated on the plate-adjacent row,
``C_f = <alpha(x)>_x`` with ``alpha = (rho_l - rho)/(rho_l - rho_g)`` clamped to
``[0, 1]``; the gas-layer thickness ``delta = sum_y alpha`` is reported alongside,
because a "covered" plate with a sub-cell film is not the same engineering state
as a thick layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..gas_layer import kernels
from ..multiphase import make_psi_eos


@dataclass
class CurtainConfig:
    """One gas-curtain run (all lattice units)."""

    nx: int = 96
    ny: int = 48
    tau: float = 1.0
    eos_temperature: float = 0.04010547044515562
    G: float = 1.0
    rho_liquid: float = 7.997835999814006
    rho_gas: float = 0.0195
    U_target: float = 0.05
    #: net-zero buoyancy acceleration (upward, i.e. opposite to gravity)
    g_buoy: float = 2.0e-3
    #: gas volume rate per unit width injected at the plate
    Q_gas: float = 0.30
    n_holes: int = 3
    hole_width: int = 3
    hole_x_fraction: tuple[float, ...] = (0.1, 0.5, 0.9)
    source_rows: int = 2
    #: pseudopotential of the solid plate, as a fraction of the liquid value
    #: ``psi(rho_liquid)``.  ``1.0`` is the fully wetting plate (no spontaneous gas
    #: at zero supply, which is what makes the negative control exact); values
    #: below 1 model a gas-retaining (non-wetting) coating.
    wall_psi_ratio: float = 1.0
    #: number of plate-adjacent rows over which the gas area is integrated
    band: int = 8
    #: fraction of the liquid excess in the source cells displaced per step (the
    #: knob that turns injected gas mass into gas volume)
    displace_fraction: float = 0.1
    #: per-step fraction of a source cell's liquid content that the injector may
    #: displace (rate limit that keeps the Shan-Chen force stable)
    displacement_rate: float = 0.05
    #: prescribed initial gas-layer thickness (cells) on the ceiling plate.  The
    #: campaign uses this as the *gas inventory*: with a closed channel the layer
    #: volume is conserved, so different Fr redistribute the same gas between a
    #: thick lens (low Fr) and a thin spread film (high Fr).
    initial_thickness: float = 0.0
    control_gain: float = 4.0e-3
    ctrl_every: int = 10
    n_steps: int = 4000
    sample_every: int = 100
    warmup: int = 1200
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    @property
    def height(self) -> float:
        return float(self.ny - 1)

    @property
    def kinematic_viscosity(self) -> float:
        return kernels.CS2 * (self.tau - 0.5)

    @property
    def gas_threshold(self) -> float:
        return 0.5 * (self.rho_liquid + self.rho_gas)

    @property
    def C_q(self) -> float:
        """Dimensionless gas flow ``C_q = Q / (U L)`` (per unit width)."""
        return self.Q_gas / (self.U_target * self.nx)

    @property
    def Fr(self) -> float:
        """Buoyant Froude number ``Fr = U / sqrt(g_b H)``."""
        if self.g_buoy <= 0.0:
            return float("inf")
        return self.U_target / math.sqrt(self.g_buoy * self.height)

    @property
    def rough_wall(self) -> bool:
        return False

    def validate(self) -> None:
        if self.ny < 12 or self.nx < 24:
            raise ValueError("need ny >= 12 and nx >= 24")
        if self.tau <= 0.55:
            raise ValueError("tau <= 0.55 is unstable for this kernel")
        if len(self.hole_x_fraction) != self.n_holes:
            raise ValueError("hole_x_fraction must list every hole")
        if self.Q_gas < 0.0:
            raise ValueError("Q_gas must be non-negative")
        if self.source_rows + 2 >= self.ny:
            raise ValueError("source rows must fit in the channel")


class CurtainChannel2D:
    """Two-phase plane channel with a buoyant gas curtain and drag diagnostics."""

    def __init__(self, config: CurtainConfig | None = None):
        cfg = config or CurtainConfig()
        cfg.validate()
        self.cfg = cfg
        self.band = cfg.band
        self.device = torch.device(cfg.device)
        self.psi = make_psi_eos("pr", cfg.eos_temperature)
        self.solid = torch.zeros((cfg.ny, cfg.nx), dtype=torch.bool, device=self.device)
        self.solid[0, :] = True
        self.solid[cfg.ny - 1, :] = True
        self.fluid = ~self.solid
        self.psi_liquid = float(
            self.psi(torch.tensor([cfg.rho_liquid], dtype=torch.float64))
        )
        self.wall_psi = cfg.wall_psi_ratio * self.psi_liquid
        self.gx = 0.0
        self.t = 0
        self.mass_injected = 0.0
        self.mass_initial = 0.0
        self._wall_hist: list[float] = []
        self._u_hist: list[float] = []
        self._theta_hist: list[float] = []
        self._delta_hist: list[float] = []
        self._driver_hist: list[float] = []
        self._snapshots: list[dict[str, Any]] = []
        self.f = self._initial_field()
        self.holes = self._hole_slices()

    # -- construction --------------------------------------------------------
    def _hole_slices(self) -> list[slice]:
        cfg = self.cfg
        out = []
        for frac in cfg.hole_x_fraction:
            xc = int(round(frac * cfg.nx))
            x0 = max(1, xc - cfg.hole_width // 2)
            out.append(slice(x0, min(cfg.nx - 1, x0 + cfg.hole_width)))
        return out

    def _initial_field(self) -> torch.Tensor:
        cfg = self.cfg
        ny, nx = cfg.ny, cfg.nx
        rho = torch.full((ny, nx), cfg.rho_liquid, dtype=cfg.dtype, device=self.device)
        d0 = cfg.initial_thickness
        if d0 > 0.0:
            ys = torch.arange(ny, dtype=cfg.dtype, device=self.device).view(ny, 1)
            # tanh gas layer against the ceiling: alpha = 1 well inside the layer
            mid = (ny - 1) - d0
            prof = 0.5 * (1.0 - torch.tanh((mid - ys) / 1.0))
            rho = (cfg.rho_liquid + (cfg.rho_gas - cfg.rho_liquid) * prof).expand(ny, nx).contiguous()
        y = np.arange(1, ny - 1, dtype=np.float64)
        h = float(ny - 1)
        shape = 4.0 * y * (h - y) / (h * h)
        ux = np.zeros((ny, nx), dtype=np.float64)
        ux[1:-1, :] = np.repeat((1.5 * cfg.U_target * shape).reshape(-1, 1), nx, axis=1)
        ux_t = torch.tensor(ux, dtype=cfg.dtype, device=self.device)
        zero = torch.zeros_like(ux_t)
        f = kernels.equilibrium(rho, ux_t, zero)
        f[:, self.solid] = 0.0
        self.mass_initial = float(rho[self.fluid].sum())
        return f

    # -- diagnostics ---------------------------------------------------------
    def densities(self) -> tuple[torch.Tensor, torch.Tensor]:
        rho, ux, _ = kernels.macroscopic(self.f)
        return rho, ux

    def gas_fraction(self, rho: torch.Tensor) -> torch.Tensor:
        """Smooth gas-phase indicator.

        A linear density deficit ``(rho_l - rho)/(rho_l - rho_g)`` is *unusable*
        here: the liquid density in a driven channel fluctuates by +-0.5 lattice
        units (compressibility) and the wall rows carry a Shan-Chen density
        offset, so integrating that deficit over the channel returns a spurious
        "coverage" of 0.67 with **no gas in the domain at all** (measured).  The
        indicator is therefore a ``tanh`` step about the mid-coexistence density,
        which is insensitive to those offsets and to the diffuse interface.
        """
        cfg = self.cfg
        mid = cfg.gas_threshold
        width = 0.15 * mid
        return 0.5 * (1.0 - torch.tanh((rho - mid) / width))

    def coverage(self, rho: torch.Tensor) -> float:
        """``C_f = A_gas / A_bottom``: plate-area fraction carrying a gas layer.

        ``A_gas`` is the gas area of one streamwise column measured in units of
        one lattice cell of thickness, saturated at 1: a column whose gas
        thickness ``A(x) = sum_y alpha`` reaches one cell counts as fully covered,
        a half-cell film counts as half.  ``C_f = <min(1, A(x))>_x`` is therefore
        a genuine area fraction in ``[0, 1]``.
        """
        return float(self.gas_fraction(rho)[-1 - self.band: -1, :].sum(dim=0).clamp(max=1.0).mean())

    def layer_thickness(self, rho: torch.Tensor) -> float:
        """Mean gas-layer thickness ``<sum_y alpha>_x`` in cells (unclamped)."""
        return float(self.gas_fraction(rho)[-1 - self.band: -1, :].sum(dim=0).mean())

    def bulk_velocity(self) -> float:
        _, ux = self.densities()
        return float(ux[self.fluid].mean())

    # -- one step ------------------------------------------------------------
    def _collide(self) -> torch.Tensor:
        cfg = self.cfg
        rho, ux, uy = kernels.macroscopic(self.f)
        fx, fy = kernels.sc_force(
            rho,
            self.psi,
            G=cfg.G,
            gx=self.gx,
            gy=0.0,
            solid=self.solid,
            wall_psi=self.wall_psi,
            periodic_x=True,
        )
        # Buoyancy is applied to the *gas* only, upward (+y): a gas parcel in
        # liquid feels (rho_liquid - rho) g_buoy.  Applying the net-zero force
        # (rho_ref - rho) to the whole mixture instead lets the +-0.5 density
        # fluctuations of the liquid drive a spurious convection current, which
        # showed up as a 1.9 % "drag reduction" with **no gas at all** in the
        # negative-control run.
        fy = fy + self.gas_fraction(rho) * (cfg.rho_liquid - rho) * cfg.g_buoy
        rho_s = torch.clamp(rho, min=1e-12)
        feq_u = kernels.equilibrium(rho, ux, uy)
        feq_f = kernels.equilibrium(rho, ux + fx / rho_s, uy + fy / rho_s)
        f_out = self.f - (self.f - feq_u) / cfg.tau + (feq_f - feq_u)
        f_out = torch.where(self.solid.unsqueeze(0), torch.zeros_like(f_out), f_out)
        return f_out

    @property
    def rho_ref(self) -> float:
        """Instantaneous mean fluid density (buoyancy reference)."""
        return float(self.f[2:-2].sum() / ((self.cfg.ny - 4) * self.cfg.nx))

    def step(self, n: int = 1) -> None:
        cfg = self.cfg
        nx = cfg.nx
        for _ in range(n):
            f_post = self._collide()
            out = kernels.stream(f_post, solid=self.solid, periodic_x=True)
            self.f = out["f"]  # type: ignore[assignment]
            if cfg.Q_gas > 0.0:
                self._inject()
            self.t += 1
            self._wall_hist.append(float(out["wall_force_x"][1].sum()) / nx)
            if self.t % cfg.ctrl_every == 0:
                self._flow_control()
            if self.t % cfg.sample_every == 0:
                self.sample()

    def _inject(self) -> None:
        """Hole injection of gas: mass booked, liquid displaced downwards.

        ``dm = rho_gas * Q / n_cells`` per source cell per step is added and
        booked; a fraction ``displace_fraction`` of the liquid excess is moved one
        row downwards so that the injected gas *occupies volume* instead of merely
        dissolving into the liquid.  (Injecting mass alone does not work in this
        model, and emptying the source cells in a few steps destabilises the
        Shan-Chen force -- both were measured, see the report's trap list.)

        The displaced liquid that reaches the floor row is zeroed with the solid
        mask, so the mass ledger closes to ``O(displace_fraction * excess)`` rather
        than to machine precision; the residual is reported as ``mass_drift``.
        """
        cfg = self.cfg
        rows = slice(cfg.ny - 1 - cfg.source_rows, cfg.ny - 1)
        n_cells = cfg.source_rows * cfg.hole_width * cfg.n_holes
        dm = cfg.rho_gas * cfg.Q_gas / n_cells
        weights = kernels._W.to(dtype=cfg.dtype, device=self.device).view(9, 1, 1)
        for sl in self.holes:
            self.f[:, rows, sl] += weights * dm
        self.mass_injected += dm * n_cells
        if cfg.displace_fraction <= 0.0:
            return
        rho, _, _ = kernels.macroscopic(self.f)
        excess = torch.clamp(rho - 1.5 * cfg.rho_gas, min=0.0)
        move = torch.zeros_like(rho)
        for sl in self.holes:
            move[rows, sl] = cfg.displace_fraction * excess[rows, sl]
        transfer = weights * move.unsqueeze(0)
        self.f -= transfer
        shifted = torch.zeros_like(transfer)
        shifted[:, :-1, :] = transfer[:, 1:, :]
        self.f += shifted
        self.f[:, self.solid] = 0.0

    def _flow_control(self) -> None:
        cfg = self.cfg
        u_bulk = self.bulk_velocity()
        err = cfg.U_target - u_bulk
        self.gx += cfg.control_gain * err
        self.gx = float(np.clip(self.gx, -5e-3, 5e-3))
        self._u_hist.append(u_bulk)
        self._driver_hist.append(self.gx)

    def sample(self) -> None:
        rho, _ = self.densities()
        self._theta_hist.append(self.coverage(rho))
        self._delta_hist.append(self.layer_thickness(rho))
        if len(self._snapshots) < 3 and self.t > 0.6 * self.cfg.n_steps:
            self._snapshots.append(
                {
                    "t": int(self.t),
                    "alpha": self.gas_fraction(rho).cpu().numpy().tolist(),
                }
            )

    # -- run -----------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        self.step(self.cfg.n_steps)
        cfg = self.cfg
        tail = slice(len(self._wall_hist) // 2, None)
        tau_w = float(np.mean(self._wall_hist[tail]))
        theta = float(np.mean(self._theta_hist[-max(3, len(self._theta_hist) // 3):]))
        delta = float(np.mean(self._delta_hist[-max(3, len(self._delta_hist) // 3):]))
        u_bulk = float(np.mean(self._u_hist[-max(3, len(self._u_hist) // 3):]))
        rho, _ = self.densities()
        mass = float(rho[self.fluid].sum())
        mu = cfg.rho_liquid * cfg.kinematic_viscosity
        tau_w_analytic = 6.0 * mu * cfg.U_target / cfg.height
        return {
            "config": {
                "nx": cfg.nx,
                "ny": cfg.ny,
                "U_target": cfg.U_target,
                "Q_gas": cfg.Q_gas,
                "C_q": cfg.C_q,
                "Fr": cfg.Fr if math.isfinite(cfg.Fr) else None,
                "g_buoy": cfg.g_buoy,
                "n_holes": cfg.n_holes,
                "hole_width": cfg.hole_width,
                "wall_psi_ratio": cfg.wall_psi_ratio,
                "wall_psi": self.wall_psi,
                "n_steps": cfg.n_steps,
            },
            "tau_w": tau_w,
            "tau_w_analytic_1ph": tau_w_analytic,
            "u_bulk": u_bulk,
            "coverage": theta,
            "layer_thickness": delta,
            "mass_drift": abs(mass + self.mass_injected - self.mass_initial) / self.mass_initial,
            "mass_injected": self.mass_injected,
            "gx": self.gx,
            "theta_history": self._theta_hist,
            "snapshot": self._snapshots[-1] if self._snapshots else None,
        }


def drag_reduction(tau_w: float, tau_w_0: float) -> float:
    """``DR = (tau_w0 - tau_w)/tau_w0``."""
    if tau_w_0 <= 0.0:
        raise ValueError("reference wall shear must be positive")
    return (tau_w_0 - tau_w) / tau_w_0


def run_pair(cfg: CurtainConfig) -> dict[str, Any]:
    """Run the gas curtain and the matched single-phase reference at the same U."""
    two_phase = CurtainChannel2D(cfg).run()
    ref_cfg = CurtainConfig(
        **{
            **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
            "Q_gas": 0.0,
            "g_buoy": 0.0,
            # The reference is the *same hull without air lubrication*: identical
            # geometry, speed, mesh and gas-free water, but a fully wetting plate.
            # Using the gas-retaining coating as its own reference is wrong -- a
            # non-wetting plate spontaneously cavitates, so that reference already
            # carries a gas film (measured tau_w = 0.0085 instead of 0.0126 at
            # U = 0.06) and hides most of the drag reduction.
            "wall_psi_ratio": 1.0,
            "displace_fraction": 0.0,
        }
    )
    single = CurtainChannel2D(ref_cfg).run()
    dr = drag_reduction(two_phase["tau_w"], single["tau_w"])
    out = dict(two_phase)
    out["tau_w_1ph"] = single["tau_w"]
    out["u_bulk_1ph"] = single["u_bulk"]
    out["DR"] = dr
    out["snapshot"] = two_phase["snapshot"]
    return out