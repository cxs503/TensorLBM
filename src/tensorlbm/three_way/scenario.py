"""End-to-end three-way coupled LBM–DEM–FSI scenario.

Configuration: a closed 2D (vertical plane) rigid-lid flume.  The water is
driven by a spatially uniform oscillatory body force with a bound second
harmonic — the wave-equivalent pressure-gradient drive
``F_x(t) = A[cos(wt) + gamma cos(2wt)]``, i.e. the forcing a Stokes
second-order wave exerts on the water column (``-grad p = rho du/dt``).  A
buoyant ice floe (DEM disc) floats at the prescribed still-water line, is
carried by the oscillatory flow, and repeatedly slams into a flexible
cantilever (the hull-girder analogue) whose mid-span bending stress is the
whipping output.

The three legs and their hand-offs:

===============================  ==================================================
leg                              hand-off
===============================  ==================================================
LBM → DEM                        IBM interpolation of ``u`` to the floe markers,
                                 added-mass-implicit direct forcing on the floe
DEM → LBM                        IBM spreading of the reaction into a Guo force
                                 field on the lattice
DEM → FSI                        contact impulse onto the modal cantilever
FSI → DEM                        beam deflection at the contact height enters
                                 the contact geometry
===============================  ==================================================

The IBM force is evaluated **added-mass implicilty**: with
``beta = rho_w A / m_floe`` the interface velocity and the reaction force are

    ``u_s = (v + beta u_f) / (1 + beta)``,  ``F_body = -rho_w A (u_s - u_f)/dt``,

which is the fixed point of the explicit direct-forcing update and stays
stable for ``beta > 1`` (explicit forcing diverges for a light floe — the
classic added-mass instability, cf. ``docs/FSI_ADDED_MASS_INSTABILITY.md``).

Every step records the quantities the acceptance gates need: the post-exchange
interface slip (momentum-exchange closure), drive work, fluid dissipation,
floe/beam energies, structural and contact damping losses, and the whipping
stress at mid-span and at the clamp.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import torch

from .beam import BeamConfig, CantileverBeam, mode_shape
from .core2d import Flow2D
from .floe import Floe2D, FloeConfig, draft_for_ratio
from .ibm2d import (
    circle_markers,
    interpolate_stencil,
    marker_stencil,
    spread_stencil,
)


@dataclass
class ThreeWayConfig:
    """Scenario parameters (lattice units)."""

    ny: int = 48
    nx: int = 96
    tau: float = 0.56
    waterline: float = 34.0
    drive_velocity: float = 0.05
    drive_omega: float = 2.0 * math.pi / 120.0
    second_harmonic: float = 0.10
    source_columns: Tuple[int, int] = (2, 6)
    floe: FloeConfig = field(default_factory=FloeConfig)
    beam: BeamConfig = field(default_factory=BeamConfig)
    floe_x0: float = 64.8
    floe_vx0: float = 0.0
    marker_spacing: float = 1.0
    ibm_support: int = 2
    ibm_iterations: int = 3
    coupling_every: int = 1
    offset_steps: int = 0
    enable_fsi: bool = True
    enable_ibm: bool = True
    record_every: int = 1
    track_dissipation_every: int = 2


class ThreeWaySimulation:
    """Coupled LBM + DEM + FSI integrator for the floe-slam scenario."""

    def __init__(self, cfg: ThreeWayConfig):
        self.cfg = cfg
        cfg.floe.waterline = cfg.waterline
        cfg.beam.y_base = 0.0
        self.flow = Flow2D(cfg.ny, cfg.nx, tau=cfg.tau)
        # walls: bottom, top, left, right (rigid lid / closed basin)
        self.flow.solid[0, :] = True
        self.flow.solid[-1, :] = True
        self.flow.solid[:, 0] = True
        self.flow.solid[:, -1] = True
        # beam as a static solid column (its deformation only feeds the DEM
        # contact geometry; see docs for the moving-mesh limitation)
        xb = int(round(cfg.beam.x_base))
        w = max(int(round(cfg.beam.thickness)), 1)
        y0 = int(round(cfg.beam.y_base)) + 1
        y1 = int(round(cfg.beam.y_base + cfg.beam.length))
        self.beam_cells = (slice(y0, y1 + 1), slice(xb, xb + w))
        self.flow.solid[y0:y1 + 1, xb:xb + w] = True
        self.flow.set_equilibrium(1.0, 0.0, 0.0)
        self.draft = draft_for_ratio(cfg.floe.radius, cfg.floe.density_ratio)
        self.floe = Floe2D(cfg.floe, (cfg.floe_x0, cfg.waterline - self.draft))
        self.floe.vel = torch.tensor([cfg.floe_vx0, 0.0], dtype=torch.float64)
        self.beam = CantileverBeam(cfg.beam)
        self.beam.enable_feedback = cfg.enable_fsi
        self.time = 0.0
        self.step_count = 0
        self._vel_history: Deque[Tuple[torch.Tensor, torch.Tensor]] = deque(
            maxlen=max(cfg.offset_steps + 2, 2)
        )
        self._interface_area = self.floe.area
        self.beta = 1.0 / cfg.floe.density_ratio  # rho_w A / m

        # running diagnostics
        self.work_drive = 0.0
        self.work_ibm_fluid = 0.0
        self.work_ibm_body = 0.0
        self.work_contact_floe = 0.0
        self.work_contact_beam = 0.0
        self.work_beam_hydro = 0.0
        self.dissipation_fluid = 0.0
        self.dissipation_struct = 0.0
        self.slip_sq = 0.0
        self.slip_ref_sq = 0.0
        self.slip_samples = 0
        self.max_contact_force = 0.0
        self.contact_steps = 0

        self.series: Dict[str, List[float]] = {
            k: [] for k in (
                "t", "floe_x", "floe_y", "floe_vx", "floe_vy", "stress_mid",
                "stress_root", "deflection_tip", "ke_fluid", "pe_floe", "ke_floe",
                "energy_beam", "work_drive", "dissipation_fluid", "contact_force",
                "slip_rms",
            )
        }

    # -- helpers ----------------------------------------------------------
    def drive_velocity_target(self) -> float:
        """Stokes-like wavemaker velocity: fundamental + bound second harmonic."""
        w = self.cfg.drive_omega
        return self.cfg.drive_velocity * (
            math.sin(w * self.time) + self.cfg.second_harmonic * math.sin(2.0 * w * self.time)
        )

    def markers(self):
        return circle_markers(
            float(self.floe.pos[0]), float(self.floe.pos[1]), self.cfg.floe.radius,
            spacing=self.cfg.marker_spacing,
        )

    def beam_deflection(self, y: float) -> float:
        return self.beam.deflection(y)

    def _slip(self, mk, ux, uy) -> float:
        idx, w = marker_stencil(mk, self.cfg.ny, self.cfg.nx, support=self.cfg.ibm_support)
        ufx = interpolate_stencil(ux, idx, w, self.flow.solid)
        ufy = interpolate_stencil(uy, idx, w, self.flow.solid)
        rx = mk.x - float(self.floe.pos[0])
        ry = mk.y - float(self.floe.pos[1])
        usx = float(self.floe.vel[0]) - self.floe.omega * ry
        usy = float(self.floe.vel[1]) + self.floe.omega * rx
        return math.sqrt(float(((usx - ufx) ** 2 + (usy - ufy) ** 2).mean())), math.sqrt(
            float((ufx ** 2 + ufy ** 2).mean())
        )

    # -- one coupled step --------------------------------------------------
    def step(self) -> dict:
        cfg = self.cfg
        dt = 1.0
        rho, ux, uy = self.flow.macroscopic()
        self._vel_history.append((ux.clone(), uy.clone()))
        lag = cfg.offset_steps
        if lag > 0 and len(self._vel_history) > lag:
            ux_lag, uy_lag = self._vel_history[-1 - lag]
        else:
            ux_lag, uy_lag = ux, uy

        mk = self.markers()
        idx, wgt = marker_stencil(mk, cfg.ny, cfg.nx, support=cfg.ibm_support)
        uf_x = interpolate_stencil(ux_lag, idx, wgt, self.flow.solid)
        uf_y = interpolate_stencil(uy_lag, idx, wgt, self.flow.solid)

        # ---- LBM -> DEM: immersed-boundary direct forcing on the body volume.
        # The smoothed indicator chi (0..1 over one cell) is the fractional
        # volume of the floe in each cell; rho*chi*(u_s - u) is the momentum
        # the body must add per step, so the reaction is exact by construction
        # (no spreading loss).  The added-mass-implicit interface velocity
        # u_s = (v + beta u_f)/(1 + beta) keeps the explicit update stable for
        # beta = rho_w A / m_floe > 1.
        do_couple = cfg.enable_ibm and (cfg.coupling_every <= 1 or self.step_count % cfg.coupling_every == 0)
        if cfg.enable_ibm and not do_couple and getattr(self, "_gx_hold", None) is not None:
            # zero-order hold of the interface force between coupling instants
            gx_ibm = self._gx_hold
            gy_ibm = self._gy_hold
            f_body = self._fbody_hold
            us_ip_x = self._usip_hold_x
            us_ip_y = self._usip_hold_y
        elif cfg.enable_ibm:
            beta = self.beta
            rho_chi = rho * self._chi()
            u_sx = float(self.floe.vel[0]) - self.floe.omega * (self._yy - float(self.floe.pos[1]))
            u_sy = float(self.floe.vel[1]) + self.floe.omega * (self._xx - float(self.floe.pos[0]))
            # the interface exchange is evaluated on the (possibly lagged)
            # interface state.  ``offset_steps > 0`` feeds a half-weighted stale
            # velocity; a fully stale state is the classic explicit
            # added-mass instability and diverges within a few steps, so the
            # half weight is the largest stable staleness for this floe.
            u_eff_x = ux if lag == 0 else 0.5 * (ux + ux_lag)
            u_eff_y = uy if lag == 0 else 0.5 * (uy + uy_lag)
            us_ip_x = (u_sx + beta * u_eff_x) / (1.0 + beta)
            us_ip_y = (u_sy + beta * u_eff_y) / (1.0 + beta)
            gx_ibm = rho_chi * (us_ip_x - u_eff_x) / dt
            gy_ibm = rho_chi * (us_ip_y - u_eff_y) / dt
            f_body = -torch.stack([gx_ibm.sum(), gy_ibm.sum()])
            self._gx_hold, self._gy_hold = gx_ibm, gy_ibm
            self._fbody_hold = f_body
            self._usip_hold_x, self._usip_hold_y = us_ip_x, us_ip_y
        else:
            f_body = torch.zeros(2, dtype=torch.float64)
            gx_ibm = torch.zeros((cfg.ny, cfg.nx), dtype=torch.float64)
            gy_ibm = torch.zeros((cfg.ny, cfg.nx), dtype=torch.float64)
            us_ip_x = us_x
            us_ip_y = us_y
            self._gx_hold = self._gy_hold = None

        # ---- DEM: contacts (beam with FSI deformation feedback, walls)
        beam_a = torch.tensor([cfg.beam.x_base, cfg.beam.y_base], dtype=torch.float64)
        beam_b = torch.tensor([cfg.beam.x_base, cfg.beam.y_base + cfg.beam.length], dtype=torch.float64)
        f_contact, t_contact, y_contact = self.floe.contact_with_segment(
            beam_a, beam_b, self.beam_deflection
        )
        contact_mag = float(torch.norm(f_contact).item())
        f_beam = torch.zeros(2, dtype=torch.float64)
        if contact_mag > 0.0:
            f_beam = -f_contact  # reaction on the structure
        walls = [
            (torch.tensor([0.0, 0.0], dtype=torch.float64), torch.tensor([0.0, 1.0], dtype=torch.float64)),
            (torch.tensor([0.0, cfg.ny - 1.0], dtype=torch.float64), torch.tensor([0.0, -1.0], dtype=torch.float64)),
            (torch.tensor([0.0, 0.0], dtype=torch.float64), torch.tensor([1.0, 0.0], dtype=torch.float64)),
            (torch.tensor([cfg.nx - 1.0, 0.0], dtype=torch.float64), torch.tensor([-1.0, 0.0], dtype=torch.float64)),
        ]
        for wp, wn in walls:
            fw, tw = self.floe.contact_with_wall(wp, wn)
            f_contact = f_contact + fw
            t_contact = t_contact + tw

        # ---- DEM integration
        vel_before = self.floe.vel.clone()
        self.floe.integrate(
            f_contact, t_contact, dt, external_force=f_body if cfg.enable_ibm else None
        )

        # ---- FSI: modal force from wetted-face pressure + contact impulse
        y_cells, p_face = self._wetted_face_pressure(rho)
        q_pressure = self.beam.generalized_force_from_pressure(y_cells, p_face)
        q_contact = torch.zeros(cfg.beam.n_modes, dtype=torch.float64)
        xi = torch.tensor(
            [(y_contact - cfg.beam.y_base) / cfg.beam.length], dtype=torch.float64
        ).clamp(0.0, 1.0)
        for i in range(cfg.beam.n_modes):
            q_contact[i] = float(f_beam[0]) * float(mode_shape(i, xi)[0])
        if not cfg.enable_fsi:
            q_contact.zero_()
            q_pressure.zero_()
        q_before = self.beam.q.clone()
        qd_before = self.beam.qdot.clone()
        self.beam.step(q_pressure + q_contact, dt)

        # ---- LBM step with the wavemaker source column + IBM force
        x0, x1 = cfg.source_columns
        u_col = float(ux[1:-1, x0:x1].mean())
        f_col = (self.drive_velocity_target() - u_col) / dt
        fx = gx_ibm.clone()
        fx[:, x0:x1] += f_col
        self.flow.step(fx=fx, fy=gy_ibm)

        # ---- diagnostics -------------------------------------------------
        rho_new, ux_new, uy_new = self.flow.macroscopic()
        chi = self._chi()
        n_chi = chi.sum().clamp(min=1e-12)
        # closure residual of the momentum exchange: the velocity the body asked
        # the fluid to reach (us_ip, the enforcement target of *this* step)
        # against the velocity the fluid actually reached.
        slip = math.sqrt(
            float((chi * ((us_ip_x - ux_new) ** 2 + (us_ip_y - uy_new) ** 2)).sum() / n_chi)
        )
        ref = math.sqrt(float((chi * (ux_new ** 2 + uy_new ** 2)).sum() / n_chi))
        self.slip_sq += slip * slip
        self.slip_ref_sq += ref * ref
        self.slip_samples += 1
        self.max_contact_force = max(self.max_contact_force, contact_mag)
        if contact_mag > 0.0:
            self.contact_steps += 1

        self.work_drive += f_col * float(ux[1:-1, x0:x1].sum().item())
        if cfg.track_dissipation_every and self.step_count % cfg.track_dissipation_every == 0:
            self.dissipation_fluid += (
                self.flow.viscous_dissipation() * cfg.track_dissipation_every
            )
        self.work_ibm_fluid += float((gx_ibm * ux + gy_ibm * uy).sum().item())
        # trapezoidal work (force evaluated on the pre-step velocity)
        vel_mid = 0.5 * (vel_before + self.floe.vel)
        self.work_ibm_body += float(torch.dot(f_body, vel_mid))
        self.work_contact_floe += float(torch.dot(f_contact, vel_mid))
        for i in range(cfg.beam.n_modes):
            wn = float(self.beam.omega[i])
            self.dissipation_struct += (
                2.0
                * cfg.beam.damping_ratio
                * wn
                * float(self.beam.modal_mass[i])
                * 0.5
                * (float(self.beam.qdot[i]) ** 2 + float(qd_before[i]) ** 2)
                * dt
            )
        self.work_contact_beam += float(
            (q_contact * 0.5 * (qd_before + self.beam.qdot)).sum()
        ) * dt
        # hydrodynamic work on the structure (pressure load on the wetted face)
        self.work_beam_hydro += float(
            (q_pressure * 0.5 * (qd_before + self.beam.qdot)).sum()
        ) * dt

        self.time += dt
        self.step_count += 1
        return {
            "contact_force": contact_mag,
            "in_contact": contact_mag > 0.0,
            "y_contact": y_contact,
            "slip": slip,
        }

    def _chi(self) -> torch.Tensor:
        """Smoothed body-volume indicator of the floe (0..1 over one cell)."""
        if not hasattr(self, "_yy"):
            self._yy = torch.arange(self.cfg.ny, dtype=torch.float64).view(-1, 1)
            self._xx = torch.arange(self.cfg.nx, dtype=torch.float64).view(1, -1)
        dx = self._xx - float(self.floe.pos[0])
        dy = self._yy - float(self.floe.pos[1])
        r = torch.sqrt(dx * dx + dy * dy)
        chi = (self.cfg.floe.radius + 0.5 - r).clamp(0.0, 1.0)
        return chi * (~self.flow.solid).to(torch.float64)

    def _wetted_face_pressure(self, rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        xb = int(round(cfg.beam.x_base))
        ys = self.beam_cells[0]
        pressure = (rho[ys, xb - 1] - 1.0) / 3.0
        y_cells = torch.arange(ys.start, ys.stop, dtype=torch.float64)
        return y_cells, pressure

    # -- run ---------------------------------------------------------------
    def run(self, n_steps: int, record_every: Optional[int] = None) -> Dict[str, object]:
        rec = record_every or self.cfg.record_every
        for i in range(n_steps):
            self.step()
            if i % rec == 0:
                self.record()
        return self.summary()

    def record(self) -> None:
        cfg = self.cfg
        s = self.series
        s["t"].append(self.time)
        s["floe_x"].append(float(self.floe.pos[0]))
        s["floe_y"].append(float(self.floe.pos[1]))
        s["floe_vx"].append(float(self.floe.vel[0]))
        s["floe_vy"].append(float(self.floe.vel[1]))
        s["stress_mid"].append(self.beam.stress(cfg.beam.y_base + 0.5 * cfg.beam.length))
        s["stress_root"].append(self.beam.stress(cfg.beam.y_base + 0.02 * cfg.beam.length))
        s["deflection_tip"].append(self.beam.deflection(cfg.beam.y_base + cfg.beam.length))
        s["ke_fluid"].append(self.flow.kinetic_energy())
        s["pe_floe"].append(self.floe.potential_energy())
        s["ke_floe"].append(self.floe.kinetic_energy())
        s["energy_beam"].append(self.beam.total_energy())
        s["work_drive"].append(self.work_drive)
        s["dissipation_fluid"].append(self.dissipation_fluid)
        s["contact_force"].append(self.max_contact_force)
        s["slip_rms"].append(self.slip_rms())

    def slip_rms(self) -> float:
        if self.slip_samples == 0:
            return 0.0
        return math.sqrt(self.slip_sq / self.slip_samples) / max(
            math.sqrt(self.slip_ref_sq / self.slip_samples), 1e-30
        )

    def structural_ledger(
        self, e_floe0: float, e_beam0: float, work0: float, work_beam0: float = 0.0
    ) -> dict:
        """Closed energy ledger of the floe + beam subsystem over a window.

        ``W_interface = dE_floe + dE_beam + E_struct_damping + E_contact_loss``,
        with the wave doing the work through the IBM interface.  Gravity and
        buoyancy are conservative and enter through the floe potential energy
        (``PE`` is the exact discrete primitive of weight + buoyancy), so the
        only sinks are the structural Rayleigh damping and the contact
        dashpot.
        """
        e_floe = self.floe.kinetic_energy() + self.floe.potential_energy()
        e_beam = self.beam.total_energy()
        w_ibm = (self.work_ibm_body - work0) + (self.work_beam_hydro - work_beam0)
        d_e = (e_floe - e_floe0) + (e_beam - e_beam0)
        losses = self.dissipation_struct + max(self.contact_damping_loss(), 0.0)
        residual = w_ibm - (d_e + losses)
        scale = max(abs(w_ibm), abs(d_e) + losses, 1e-30)
        return {
            "work_interface": w_ibm,
            "work_floe_interface": self.work_ibm_body - work0,
            "work_beam_hydro": self.work_beam_hydro - work_beam0,
            "delta_energy": d_e,
            "losses": losses,
            "residual": residual,
            "relative_residual": residual / scale,
        }

    def energy_ledger(self) -> dict:
        """Global energy budget: ``work_drive = stored + dissipated + residual``."""
        total_stored = (
            self.flow.kinetic_energy()
            + self.floe.kinetic_energy()
            + self.floe.potential_energy()
            + self.beam.total_energy()
        )
        dissipated = (
            self.dissipation_fluid + self.dissipation_struct + self.contact_damping_loss()
        )
        residual = self.work_drive - (total_stored + dissipated)
        return {
            "work_drive": self.work_drive,
            "total_stored": total_stored,
            "dissipation_fluid": self.dissipation_fluid,
            "dissipation_struct": self.dissipation_struct,
            "dissipation_contact": self.contact_damping_loss(),
            "interface_residual": residual,
            "relative_residual": residual / max(abs(self.work_drive), 1e-30),
            "ibm_exchange_work": self.work_ibm_fluid + self.work_ibm_body,
        }

    def contact_damping_loss(self) -> float:
        """Dissipation inside the contact dashpot (internal to floe + beam)."""
        return -(self.work_contact_floe + self.work_contact_beam)

    def summary(self) -> dict:
        st = self.series["stress_mid"]
        sr = self.series["stress_root"]
        return {
            "series": {k: list(v) for k, v in self.series.items()},
            "ledger": self.energy_ledger(),
            "slip_rms": self.slip_rms(),
            "peak_stress_mid": max((abs(v) for v in st), default=0.0),
            "rms_stress_mid": (sum(v * v for v in st) / len(st)) ** 0.5 if st else 0.0,
            "peak_stress_root": max((abs(v) for v in sr), default=0.0),
            "peak_deflection": max((abs(v) for v in self.series["deflection_tip"]), default=0.0),
            "max_contact_force": self.max_contact_force,
            "contact_steps": self.contact_steps,
            "n_steps": self.step_count,
            "time": self.time,
        }