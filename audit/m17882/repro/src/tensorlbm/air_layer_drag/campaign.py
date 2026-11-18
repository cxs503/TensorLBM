"""DR - theta - Qg closure campaign for air-layer / microbubble drag reduction.

Physics set-up (2-D, lattice units)
-----------------------------------
* plane channel, no-slip walls at ``y = -0.5`` and ``y = ny - 0.5``, periodic in
  ``x``, constant body force density ``gx`` (a pressure gradient; the force per
  unit volume acts equally on both phases, so ``tau_w * 2 = gx * H``);
* gas (phi = -1) is seeded as a layer of ``thickness`` nodes covering a fraction
  ``theta`` of the wall; gas can be injected by an exactly accounted volumetric
  source strip and removed by a downstream vent sink strip (the vent models
  gas leaving the layer: dissolution / venting, so the periodic channel reaches
  a steady gas inventory);
* drag reduction is measured at a common driving force,

      DR = 1 - Q_2ph / Q_1ph,      Q = sum(rho u_x)

  which for the (linear, creeping-to-laminar) regime of these runs is identical
  to the matched-flow-rate definition ``DR = 1 - gx_2ph / gx_1ph``.

``Cq = Q_g / (U_b * Lx)`` is the dimensionless gas flow rate (gas volumetric
flux normalised by the bulk velocity and wetted length), i.e. the same group
used by flat-plate micro-bubble experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .core import CS2, PhaseFieldChannel, PhaseFieldParams, SinglePhaseChannel
from .economics import CompressionCase, net_power_gain
from .validations import flat_interface_phi  # noqa: F401  (re-exported for tests)


@dataclass(frozen=True)
class ChannelConfig:
    nx: int = 96
    ny: int = 40
    nu_l: float = 0.05
    nu_g: float = 0.35
    rho_l: float = 1.0
    rho_g: float = 0.01
    sigma: float = 0.01
    interface_width: float = 6.0
    mobility: float = 0.05
    body_force_x: float = 1.0e-5
    spinup_steps: int = 4000
    measure_steps: int = 4000
    measure_window: int = 4          # number of flux samples in the averaging window

    @property
    def params(self) -> PhaseFieldParams:
        return PhaseFieldParams(
            rho_l=self.rho_l, rho_g=self.rho_g, nu_l=self.nu_l, nu_g=self.nu_g,
            sigma=self.sigma, interface_width=self.interface_width,
            mobility=self.mobility)

    @property
    def height(self) -> float:
        return float(self.ny)


# ---------------------------------------------------------------------------
# single-phase reference
# ---------------------------------------------------------------------------
def single_phase_flux(cfg: ChannelConfig) -> dict:
    ch = SinglePhaseChannel(cfg.nx, cfg.ny, rho=cfg.rho_l, nu=cfg.nu_l,
                            body_force_x=cfg.body_force_x)
    ch.step(cfg.spinup_steps + cfg.measure_steps)
    q = float(np.sum(cfg.rho_l * ch.ux))
    ub = ch.bulk_velocity()
    return {
        "mass_flux": q,
        "bulk_velocity": ub,
        "wall_shear": ch.wall_shear_mean(),
        "cf": ch.simulated_cf(),
        "reynolds": cfg.rho_l * ub * cfg.height / cfg.nu_l,
    }


# ---------------------------------------------------------------------------
# two-phase air-layer case
# ---------------------------------------------------------------------------
def signed_distance_patch(ny: int, nx: int, *, x0: float, x1: float,
                          y0: float, y1: float) -> np.ndarray:
    """Signed distance (positive inside) to a wall patch, periodic in x."""
    x = np.arange(nx)[None, :] + 0.5
    y = np.arange(ny)[:, None] + 0.5
    # periodic signed distance to the x-band [x0, x1] (positive inside)
    u = np.mod(x - x0, nx)
    width = x1 - x0
    dx = np.where(u <= width,
                  np.minimum(u, width - u),
                  -np.minimum(u - width, nx - u)) * np.ones((ny, 1))
    dy = np.minimum(y - y0, y1 - y) * np.ones((1, nx))
    return np.minimum(dx, dy)


def _gas_layer_phi(cfg: ChannelConfig, coverage: float, thickness: int,
                   bilayer: bool = False) -> np.ndarray:
    """Gas layer of ``thickness`` nodes covering ``coverage`` of the wall.

    The initial mask is converted to a tanh profile through the signed distance,
    so the phase field starts on (not next to) the Allen-Cahn equilibrium
    profile; a box mask smoothed with a few averaging passes is *not* stable.
    """
    ny, nx = cfg.ny, cfg.nx
    delta = 0.25 * cfg.interface_width
    n = int(round(coverage * nx))
    phi = np.ones((ny, nx))
    if n <= 0:
        return phi
    d = signed_distance_patch(ny, nx, x0=0.0, x1=float(n), y0=-1.0,
                              y1=float(thickness) + 0.5)
    phi = -np.tanh(d / delta)
    if bilayer:
        d2 = signed_distance_patch(ny, nx, x0=float(nx - n), x1=float(nx),
                                   y0=float(ny - thickness) - 0.5, y1=float(ny) + 1.0)
        phi = np.minimum(phi, -np.tanh(d2 / delta))
    return np.clip(phi, -1.0, 1.0)


def air_layer_case(cfg: ChannelConfig, *, coverage: float, thickness: int,
                   injection: float = 0.0, vent: float = 0.0,
                   baseline_flux: float | None = None,
                   bilayer: bool = False) -> dict:
    """Run one two-phase case and return DR, gas inventory and gas flow rate."""
    params = cfg.params
    phi0 = _gas_layer_phi(cfg, coverage, thickness, bilayer=bilayer)
    ch = PhaseFieldChannel(cfg.nx, cfg.ny, params, phi0, body_force_x=cfg.body_force_x)
    inj_mask = np.zeros((cfg.ny, cfg.nx), dtype=bool)
    vent_mask = np.zeros((cfg.ny, cfg.nx), dtype=bool)
    if injection > 0.0:
        inj_mask[1:1 + thickness, : max(1, cfg.nx // 8)] = True
    if vent > 0.0:
        vent_mask[1:1 + thickness, -max(1, cfg.nx // 8):] = True

    gas_vol0 = ch.total_gas_volume()
    ch.step(cfg.spinup_steps, inject=(inj_mask, injection) if injection > 0 else None,
            vent=(vent_mask, vent) if vent > 0 else None)

    fluxes, gas_fluxes, gammas = [], [], []
    n_each = max(1, cfg.measure_steps // cfg.measure_window)
    for _ in range(cfg.measure_window):
        ch.step(n_each, inject=(inj_mask, injection) if injection > 0 else None,
                vent=(vent_mask, vent) if vent > 0 else None)
        rho = ch._density(ch.phi)
        fluxes.append(float(np.sum(rho * ch.ux)))
        alpha_g = ch.gas_fraction()
        gas_fluxes.append(float(np.sum(alpha_g * ch.ux)))
        gammas.append(float(np.sum(alpha_g > 0.5)))

    q2 = float(np.mean(fluxes))
    ub = q2 / (cfg.rho_l * cfg.nx * cfg.ny)
    gas_flux = float(np.mean(gas_fluxes))
    coverage_meas = float(np.mean(gammas)) / (max(thickness, 1) * cfg.nx)
    gas_vol = ch.total_gas_volume()
    dr = None if baseline_flux is None else 1.0 - q2 / baseline_flux
    cq = gas_flux / (abs(ub) * cfg.nx) if ub else 0.0
    # injected gas flow rate (lattice volume per step -> per time unit), the
    # quantity an operator actually controls (and that the compressor pays for)
    q_gas_inj = 0.5 * injection * float(inj_mask.sum())
    cq_inj = q_gas_inj / (abs(ub) * cfg.nx) if ub else 0.0
    return {
        "coverage_set": coverage,
        "coverage_measured": coverage_meas,
        "thickness": thickness,
        "injection_per_step": injection,
        "vent_per_step": vent,
        "mass_flux": q2,
        "bulk_velocity": ub,
        "drag_reduction": dr,
        "gas_flux": gas_flux,
        "cq": cq,
        "q_gas_inj": q_gas_inj,
        "cq_inj": cq_inj,
        "gas_volume": gas_vol,
        "gas_volume_initial": gas_vol0,
        "mean_wall_shear": ch.wall_shear_mean(),
        "u_max": float(np.max(np.abs(ch.ux))),
        "ledger": {
            "injected_volume": ch.injected_volume,
            "vented_volume": ch.vented_volume,
            "gas_volume_gain": gas_vol - gas_vol0,
            "residual": (gas_vol - gas_vol0) - (ch.injected_volume - ch.vented_volume),
        },
        "phi_drift_rel": float(abs(ch.phi.sum() - phi0.sum()) / max(abs(phi0.sum()), 1.0)),
    }


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------
def coverage_sweep(cfg: ChannelConfig, *, coverages=(0.2, 0.4, 0.6, 0.8, 1.0),
                   thickness: int = 8, baseline_flux: float | None = None) -> list[dict]:
    base = single_phase_flux(cfg)["mass_flux"] if baseline_flux is None else baseline_flux
    out = []
    for c in coverages:
        out.append(air_layer_case(cfg, coverage=c, thickness=thickness,
                                  baseline_flux=base))
    return out


def gas_flow_sweep(cfg: ChannelConfig, *, coverage: float = 1.0, thickness: int = 8,
                   injections=(0.0, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3),
                   vent_ratio: float = 1.0, baseline_flux: float | None = None) -> list[dict]:
    base = single_phase_flux(cfg)["mass_flux"] if baseline_flux is None else baseline_flux
    out = []
    for q in injections:
        out.append(air_layer_case(cfg, coverage=coverage, thickness=thickness,
                                  injection=q, vent=q * vent_ratio,
                                  baseline_flux=base))
    return out


# ---------------------------------------------------------------------------
# economics: net power domain
# ---------------------------------------------------------------------------
def net_power_table(dr_curve, case_kwargs: dict | None = None,
                    etas=(0.5, 0.6, 0.7)) -> dict:
    """dP_net on the (Cq, eta_c) grid using the simulated DR(Cq) relation."""
    kw = dict(case_kwargs or {})
    out = {"etas": list(etas), "rows": {}}
    for eta in etas:
        case = CompressionCase(eta_c=eta, **kw)
        rows = []
        for cq, dr in dr_curve:
            res = net_power_gain(max(dr, 0.0), cq, case)
            rows.append(res)
        out["rows"][str(eta)] = rows
    return out


def positive_boundary(rows: list[dict]) -> dict:
    """Largest Cq with dP_net > 0 (rows sorted by Cq)."""
    max_pos = None
    cross = None
    for r in rows:
        if r["dP_net_W"] > 0.0:
            max_pos = r["gas_flow_ratio"]
        elif max_pos is not None and cross is None:
            cross = r["gas_flow_ratio"]
    return {"cq_max_positive": max_pos, "cq_zero_cross": cross,
            "eta_c": rows[0]["eta_c"] if rows else None}