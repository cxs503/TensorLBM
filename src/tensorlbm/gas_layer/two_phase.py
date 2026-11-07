"""Shan-Chen two-phase plane channel with a hull gas layer and gas supply.

Physical set-up
---------------
A plane channel represents the hull-bottom flow: row ``0`` is the hull plate
(solid, half-way bounce-back), row ``ny-1`` is the outer no-slip boundary, and
``x`` is the streamwise direction.  Water fills the channel; gas is either

* **initialised as a layer of prescribed thickness on the plate** (the "air
  layer" configuration), and/or
* **supplied through a mass-conserving wall volume source** at an injection
  strip: gas mass ``dm = rho_gas * q_g * dt`` is *added and booked* every step,
  and the liquid occupying the source cells is displaced upwards
  (``excess -> row above``, an exactly conservative redistribution).  No mass is
  created anywhere: the source term is the only mass source and it is logged.

The streamwise boundary is either periodic (closed recirculating channel --
exactly conservative, used for the mass-drift gate) or open (Dirichlet inflow,
zero-gradient outlet and a velocity sponge in the last ``sponge_cells`` columns
-- used to let the gas leave at the "stern").

Drag reduction is defined at **matched throughput**: an integral controller
adjusts the body force so that the bulk velocity equals ``U_target`` in both the
single-phase and the two-phase run, and

    ``DR = 1 - tau_w,2ph / tau_w,1ph``   (same estimator, same U)

with ``tau_w`` from Ladd momentum exchange on the plate (the estimator calibrated
in :mod:`tensorlbm.gas_layer.single_phase`).  Gas coverage is the near-wall gas
volume fraction, ``theta = <alpha(x)>_x`` in the plate-adjacent row.

All diagnostics (coverage, drag, mass ledger, bubble statistics) are computed
from the same density field, so they are mutually consistent by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..multiphase import make_psi_eos
from . import kernels


@dataclass
class TwoPhaseConfig:
    """Configuration of one gas-layer channel run (all lattice units)."""

    ny: int = 48
    nx: int = 96
    tau: float = 1.0
    eos_temperature: float = 0.04010547044515562
    G: float = 1.0
    rho_liquid: float = 7.997835999814006
    rho_gas: float = 0.0195
    U_target: float = 0.06
    enable_sc: bool = True
    x_boundary: str = "periodic"
    initial_layer_thickness: float = 0.0
    source: bool = False
    source_x_fraction: tuple[float, float] = (0.05, 0.20)
    source_rows: int = 2
    source_rate: float = 0.0  # gas volume per unit time per unit width
    displace_fraction: float = 0.01
    wall_psi: float | None = None
    sponge_cells: int = 0
    control_gain: float = 4.0e-3
    ctrl_every: int = 10
    n_steps: int = 6000
    sample_every: int = 200
    warmup: int = 1500
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    # -- derived -------------------------------------------------------------
    @property
    def height(self) -> float:
        """Channel height between the two half-way walls."""
        return float(self.ny - 1)

    @property
    def gas_threshold(self) -> float:
        """Density threshold separating gas from liquid (mid-coexistence)."""
        return 0.5 * (self.rho_liquid + self.rho_gas)

    @property
    def kinematic_viscosity(self) -> float:
        """Liquid kinematic viscosity implied by ``tau``."""
        return kernels.CS2 * (self.tau - 0.5)

    def validate(self) -> None:
        if self.ny < 12 or self.nx < 16:
            raise ValueError("need ny >= 12 and nx >= 16")
        if self.x_boundary not in {"periodic", "open"}:
            raise ValueError("x_boundary must be 'periodic' or 'open'")
        if self.tau <= 0.55:
            raise ValueError("tau <= 0.55 is unstable for this kernel")
        if not 0.0 <= self.initial_layer_thickness < self.ny - 4:
            raise ValueError("initial layer must fit inside the channel")
        if self.source and self.source_rate <= 0.0:
            raise ValueError("source=True needs source_rate > 0")
        if self.x_boundary == "open" and self.sponge_cells == 0:
            raise ValueError("open outlet requires a sponge zone (sponge_cells > 0)")
        if self.x_boundary == "open":
            nx = self.nx
            x0, x1 = self.source_x_fraction
            if self.source and x1 * nx > nx - self.sponge_cells - 2:
                raise ValueError("gas source must lie upstream of the sponge zone")


class GasLayerChannel2D:
    """Two-phase plane channel with a gas layer, gas source and drag diagnostics."""

    def __init__(self, config: TwoPhaseConfig | None = None):
        cfg = config or TwoPhaseConfig()
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.psi = make_psi_eos("pr", cfg.eos_temperature)
        self.solid = torch.zeros((cfg.ny, cfg.nx), dtype=torch.bool, device=self.device)
        self.solid[0, :] = True
        self.solid[cfg.ny - 1, :] = True
        self.fluid = ~self.solid
        self.gx = 0.0
        self.t = 0
        # ledger
        self.mass_injected = 0.0
        self.mass_outflow = 0.0
        self.mass_inflow = 0.0
        self.mass_initial = 0.0
        self.energy_outflow = 0.0
        self._wall_hist: list[float] = []
        self._u_hist: list[float] = []
        self._theta_hist: list[float] = []
        self._gas_mass_hist: list[float] = []
        self._time_hist: list[int] = []
        self._snapshots: list[dict[str, Any]] = []
        self.f = self._initial_field()

    # -- construction -------------------------------------------------------
    def _initial_field(self) -> torch.Tensor:
        cfg = self.cfg
        ny, nx = cfg.ny, cfg.nx
        rho = torch.full((ny, nx), cfg.rho_liquid, dtype=cfg.dtype, device=self.device)
        hg = cfg.initial_layer_thickness
        if hg > 0.0:
            rows = int(math.floor(hg))
            frac = hg - rows
            if rows > 0:
                rho[1 : 1 + rows, :] = cfg.rho_gas
            if frac > 0.0 and 1 + rows < ny - 1:
                rho[1 + rows, :] = cfg.rho_gas + (1.0 - frac) * (cfg.rho_liquid - cfg.rho_gas)
        # start from a parabolic profile scaled to the target throughput
        y = np.arange(1, ny - 1, dtype=np.float64)
        h = float(ny - 1)
        shape = 4.0 * y * (h - y) / (h * h)  # 0 at walls, 1 at centre: <u> = 2/3 max
        ux = np.zeros((ny, nx), dtype=np.float64)
        ux[1:-1, :] = np.repeat(
            (1.5 * cfg.U_target * shape).reshape(-1, 1), nx, axis=1
        )
        ux_t = torch.tensor(ux, dtype=cfg.dtype, device=self.device)
        zero = torch.zeros_like(ux_t)
        f = kernels.equilibrium(rho, ux_t, zero)
        f[:, self.solid] = 0.0
        self.mass_initial = float(rho[self.fluid].sum())
        return f

    # -- diagnostics helpers ------------------------------------------------
    def densities(self) -> tuple[torch.Tensor, torch.Tensor]:
        rho, ux, uy = kernels.macroscopic(self.f)
        return rho, ux

    def gas_fraction(self, rho: torch.Tensor) -> torch.Tensor:
        """Smooth gas volume fraction ``alpha in [0, 1]`` from the density."""
        cfg = self.cfg
        alpha = (cfg.rho_liquid - rho) / (cfg.rho_liquid - cfg.rho_gas)
        return torch.clamp(alpha, 0.0, 1.0)

    def coverage(self, rho: torch.Tensor) -> float:
        """Near-wall gas coverage ``theta`` (plate-adjacent row, x-averaged)."""
        alpha = self.gas_fraction(rho)
        return float(alpha[1, :].mean())

    def layer_thickness(self, rho: torch.Tensor) -> float:
        """Column-averaged gas layer thickness ``delta = sum_y alpha / nx``."""
        alpha = self.gas_fraction(rho)
        return float(alpha[1:-1, :].sum() / alpha.shape[1])

    def bulk_velocity(self) -> float:
        _, ux = self.densities()
        return float(ux[self.fluid].mean())

    # -- one lattice step ---------------------------------------------------
    def step(self, n: int = 1) -> None:
        cfg = self.cfg
        nx = cfg.nx
        for _ in range(n):
            f_post, _, _ = kernels.collide_scp(
                self.f,
                cfg.tau,
                self.psi,
                G=cfg.G if cfg.enable_sc else 0.0,
                gx=self.gx,
                solid=self.solid,
                wall_psi=cfg.wall_psi,
                periodic_x=(cfg.x_boundary == "periodic"),
                forcing="edm",
                form="standard",
            )
            out = kernels.stream(
                f_post, solid=self.solid, periodic_x=(cfg.x_boundary == "periodic")
            )
            self.f = out["f"]  # type: ignore[assignment]
            self.mass_outflow += float(out["mass_out"])
            self.energy_outflow += float(out["momentum_out"])
            if cfg.x_boundary == "open":
                self._apply_open_boundaries()
            if cfg.source:
                self._inject()
            self.t += 1
            self._wall_hist.append(float(out["wall_force_x"][0].sum()) / nx)
            if self.t % cfg.ctrl_every == 0:
                self._flow_control()

    def _apply_open_boundaries(self) -> None:
        """Dirichlet inflow at ``x = 0`` and a velocity sponge near the outlet."""
        cfg = self.cfg
        ny, nx = cfg.ny, cfg.nx
        # inflow: liquid at the target streamwise velocity
        rho_in = torch.full((ny, nx), cfg.rho_liquid, dtype=cfg.dtype, device=self.device)
        ux_in = torch.full_like(rho_in, cfg.U_target)
        feq = kernels.equilibrium(rho_in, ux_in, torch.zeros_like(rho_in))
        f_new = self.f.clone()
        f_new[:, :, 0] = feq[:, :, 0]
        f_new[:, self.solid] = 0.0
        self.f = f_new
        # inflowing *mass flux* rho u_x through the inlet plane
        flux = 0.0
        for i in range(9):
            cx = int(kernels._C[i, 0])
            if cx > 0:
                flux += float(cx) * float(feq[i, 1:-1, 0].sum())
        self.mass_inflow += flux
        # velocity sponge over the last `sponge_cells` columns
        n_sp = cfg.sponge_cells
        if n_sp > 0:
            rho, _, _ = kernels.macroscopic(self.f)
            target = kernels.equilibrium(
                rho, torch.full_like(rho, cfg.U_target), torch.zeros_like(rho)
            )
            for k in range(n_sp):
                col = nx - n_sp + k
                sigma = 0.35 * ((k + 1) / n_sp) ** 2
                self.f[:, :, col] = (1.0 - sigma) * self.f[:, :, col] + sigma * target[:, :, col]
            self.f[:, self.solid] = 0.0

    def _inject(self) -> None:
        """Mass-conserving wall gas source with conservative liquid displacement."""
        cfg = self.cfg
        nx = cfg.nx
        x0 = int(round(cfg.source_x_fraction[0] * nx))
        x1 = int(round(cfg.source_x_fraction[1] * nx))
        x1 = max(x1, x0 + 1)
        rows = slice(1, 1 + cfg.source_rows)
        cols = slice(x0, x1)
        n_cells = (1 + cfg.source_rows - 1) * (x1 - x0)
        # 1) add exactly the injected gas mass (booked)
        dm = cfg.rho_gas * cfg.source_rate / n_cells
        weights = kernels._W.to(dtype=cfg.dtype, device=self.device).view(9, 1, 1)
        self.f[:, rows, cols] += weights * dm
        self.mass_injected += dm * n_cells
        # 2) displace the liquid out of the source cells (mass conserving)
        rho, _, _ = kernels.macroscopic(self.f)
        target = 1.2 * cfg.rho_gas
        excess = torch.clamp(rho - target, min=0.0)
        move = torch.zeros_like(rho)
        move[1 : 1 + cfg.source_rows, cols] = cfg.displace_fraction * excess[
            1 : 1 + cfg.source_rows, cols
        ]
        transfer = weights * move.unsqueeze(0)
        self.f -= transfer
        shifted = torch.zeros_like(transfer)
        shifted[:, 1:, :] = transfer[:, :-1, :]
        self.f += shifted
        self.f[:, self.solid] = 0.0

    def _flow_control(self) -> None:
        """Integral control of the body force to hold the bulk velocity."""
        err = self.cfg.U_target - self.bulk_velocity()
        self.gx = float(np.clip(self.gx + self.cfg.control_gain * err, 0.0, 5.0e-2))

    # -- measurement --------------------------------------------------------
    def bubble_statistics(self, rho: torch.Tensor | None = None) -> dict[str, float]:
        """Connected-component statistics of the gas phase and interface density."""
        if rho is None:
            rho, _ = self.densities()
        alpha = self.gas_fraction(rho)
        mask = (alpha > 0.5).cpu().numpy()
        mask[0, :] = False
        mask[-1, :] = False
        try:
            from scipy import ndimage

            labels, n = ndimage.label(mask)
            areas = ndimage.sum(mask, labels, range(1, n + 1))
        except Exception:  # pragma: no cover - scipy is a hard dep in practice
            labels, n, areas = None, 0, np.array([])
        areas = np.asarray(areas, dtype=float)
        areas = areas[areas >= 4.0]
        if areas.size:
            d_eq = 2.0 * np.sqrt(areas / math.pi)
            mean_d = float(d_eq.mean())
            max_d = float(d_eq.max())
            n_bubbles = int(areas.size)
            total_volume = float(areas.sum())
        else:
            mean_d = max_d = 0.0
            n_bubbles = 0
            total_volume = 0.0
        # interfacial area concentration a_i = interface length / domain area,
        # estimated from the diffused-interface gradient (2-D analogue).
        da_dy = torch.zeros_like(alpha)
        da_dx = torch.zeros_like(alpha)
        da_dy[1:-1, :] = 0.5 * (alpha[2:, :] - alpha[:-2, :])
        da_dx[:, 1:-1] = 0.5 * (alpha[:, 2:] - alpha[:, :-2])
        grad = torch.sqrt(da_dx * da_dx + da_dy * da_dy)
        a_i = float(grad[self.fluid].sum()) / float(self.fluid.sum())
        return {
            "n_bubbles": float(n_bubbles),
            "mean_diameter": mean_d,
            "max_diameter": max_d,
            "gas_area": total_volume,
            "interfacial_area_concentration": a_i,
        }

    def sample(self) -> None:
        rho, ux = self.densities()
        self._time_hist.append(self.t)
        self._u_hist.append(float(ux[self.fluid].mean()))
        self._theta_hist.append(self.coverage(rho))
        alpha = self.gas_fraction(rho)
        self._gas_mass_hist.append(float((alpha * rho)[self.fluid].sum()))
        if self.t % (self.cfg.sample_every * 10) == 0:
            self._snapshots.append(
                {
                    "t": self.t,
                    "rho": rho.cpu().numpy().copy(),
                    "ux": ux.cpu().numpy().copy(),
                    "alpha": alpha.cpu().numpy().copy(),
                }
            )

    def run(self) -> dict[str, Any]:
        """Run the configured number of steps and return a diagnostic record."""
        cfg = self.cfg
        for _ in range(cfg.n_steps):
            self.step(1)
            if self.t % cfg.sample_every == 0:
                self.sample()
        return self.report()

    # -- reporting ----------------------------------------------------------
    def report(self) -> dict[str, Any]:
        cfg = self.cfg
        rho, ux = self.densities()
        mass_now = float(rho[self.fluid].sum())
        n = len(self._wall_hist)
        window = max(1, n // 3)
        wall_series = np.asarray(self._wall_hist[-window:], dtype=float)
        wall_mean = float(wall_series.mean())
        wall_std = float(wall_series.std())
        theta_series = np.asarray(
            self._theta_hist[-max(1, len(self._theta_hist) // 3):], dtype=float
        )
        stats = self.bubble_statistics(rho)
        C_q = cfg.source_rate / (cfg.U_target * cfg.nx)
        ledger_error = mass_now - self.mass_initial - self.mass_injected + self.mass_outflow - self.mass_inflow
        return {
            "config": {
                "ny": cfg.ny,
                "nx": cfg.nx,
                "tau": cfg.tau,
                "eos_temperature": cfg.eos_temperature,
                "G": cfg.G,
                "U_target": cfg.U_target,
                "x_boundary": cfg.x_boundary,
                "initial_layer_thickness": cfg.initial_layer_thickness,
                "source": cfg.source,
                "source_rate": cfg.source_rate,
                "sponge_cells": cfg.sponge_cells,
                "n_steps": cfg.n_steps,
            },
            "steps": self.t,
            "gx_final": self.gx,
            "U_measured": float(ux[self.fluid].mean()),
            "U_target": cfg.U_target,
            "wall_shear": wall_mean,
            "wall_shear_std": wall_std,
            "theta_measured": float(theta_series.mean()) if theta_series.size else float("nan"),
            "layer_thickness": self.layer_thickness(rho),
            "C_q": C_q,
            "mass_initial": self.mass_initial,
            "mass_final": mass_now,
            "mass_injected": self.mass_injected,
            "mass_outflow": self.mass_outflow,
            "mass_inflow": self.mass_inflow,
            "mass_relative_drift": (mass_now - self.mass_initial) / self.mass_initial,
            "ledger_relative_error": ledger_error / self.mass_initial,
            "bubbles": stats,
            "history": {
                "t": self._time_hist,
                "theta": self._theta_hist,
                "U": self._u_hist,
            },
            "snapshot": self._snapshots[-1] if self._snapshots else None,
        }


def run_case(config: TwoPhaseConfig) -> dict[str, Any]:
    """Run one channel case and return its diagnostic record."""
    sim = GasLayerChannel2D(config)
    return sim.run()


def static_layer_drag(
    thicknesses: list[float],
    U_target: float = 0.06,
    ny: int = 48,
    nx: int = 64,
    n_steps: int = 4000,
    tau: float = 0.9,
    eos_temperature: float = 0.075,
    device: str = "cpu",
) -> dict[str, Any]:
    """DR(theta) for prescribed initial gas-layer thicknesses (no gas source).

    Each run holds the bulk velocity fixed by the body-force controller, so the
    measured wall shear is directly comparable with the single-phase reference
    and ``DR = 1 - tau_w/tau_w,1ph``.  The layer erodes slowly (entrainment), so
    the coverage and drag are window-averaged over the second half of the run;
    the entrainment rate is returned as the sustaining gas flow
    ``C_q,required``.
    """
    base = TwoPhaseConfig(
        ny=ny, nx=nx, tau=tau, eos_temperature=eos_temperature, U_target=U_target,
        n_steps=n_steps, x_boundary="periodic", device=device,
    )
    ref = run_case(base)
    tau0 = ref["wall_shear"]
    out = []
    for hg in thicknesses:
        cfg = TwoPhaseConfig(
            ny=ny, nx=nx, tau=tau, eos_temperature=eos_temperature, U_target=U_target,
            n_steps=n_steps, x_boundary="periodic", initial_layer_thickness=hg,
            device=device,
        )
        rec = run_case(cfg)
        theta = rec["theta_measured"]
        dr = 1.0 - rec["wall_shear"] / tau0
        gas_volume_start = hg * nx
        gas_volume_end = max(rec["bubbles"]["gas_area"], 0.0)
        entrainment = max(gas_volume_start - gas_volume_end, 0.0) / max(rec["steps"], 1)
        c_q_req = entrainment / (U_target * nx)
        out.append(
            {
                "initial_layer_thickness": hg,
                "theta": theta,
                "layer_thickness": rec["layer_thickness"],
                "drag_reduction": dr,
                "wall_shear": rec["wall_shear"],
                "wall_shear_1ph": tau0,
                "C_q_required": c_q_req,
                "bubbles": rec["bubbles"],
                "record": rec,
            }
        )
    return {"single_phase": ref, "cases": out, "tau_w_1ph": tau0}