"""Ice–propeller interaction with a two-way IBM ⇄ DEM coupling.

The propeller is a two-blade, zero-thickness (1-cell thick) rotor prescribed to
rotate at a constant angular velocity Ω about the z axis; its blade surface is
represented by Lagrangian IBM markers that are re-generated every step, so the
boundary is *moving and rotating* at once.  The ice floe is a spherical
``tensorlbm.dem.IceParticle`` whose surface markers are advected with the
rigid-body velocity ``V + ω × (x − x_c)``; the reaction of the IBM marker forces
is the hydrodynamic force/torque on the ice.

The DEM side reuses the repository contact law
(:class:`tensorlbm.dem.contact.LinearSpringDashpot`) for the ice–blade contact
(normal spring-dashpot + Coulomb-limited tangential friction) and the
repository particle integrator (:class:`tensorlbm.dem.DEMSolver`, with gravity
and its own drag model disabled because the hydrodynamic load comes from IBM).

Coupling sequence per lattice step
----------------------------------
1. interpolate the fluid velocity to the ice markers, compute the marker force
   that drives them to the rigid-body velocity, spread it back to the grid;
2. same for the propeller blade markers (prescribed Ω), accumulate the
   hydrodynamic torque on the rotor;
3. detect ice–blade contact, evaluate the DEM contact force and the equal and
   opposite reaction at the contact point (rotor torque);
4. ``DEMSolver.set_external_forces(F_IBM + F_contact, T_IBM + T_contact)`` and
   advance the particle one step (semi-implicit Euler inside the DEM solver);
5. advance the rotor angle and apply the combined Eulerian body force with the
   *exact* momentum source (:func:`tensorlbm.ibm_force_transfer.body_source_3d`),
   which keeps the fluid-momentum audit a machine-precision identity.

Everything is in lattice units (``dx = dt = ρ = 1``); the ice density is
therefore a ratio ``ρ_ice/ρ_water = 0.9``.

Limitations (see the campaign report): the blade is a flat plate geometry with
no twist/pitch, the domain is periodic (no free surface, no gravity/buoyancy),
and the ice floe is a rigid sphere — real ice floes are irregular and can
fracture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import torch

from .d3q19 import equilibrium3d, macroscopic3d
from .dem import DEMSolver, DEMConfig, IceParticle
from .dem.contact import LinearSpringDashpot
from .ibm_force_transfer import (
    body_source_3d,
    build_stencil,
    direct_forcing,
    interior_mask_force,
    momentum_audit,
    sphere_markers,
)
from .solver3d import collide_bgk3d, stream3d

__all__ = ["IcePropellerConfig", "IcePropellerIBM", "run_ice_propeller_case"]


@dataclass
class IcePropellerConfig:
    """Configuration (lattice units unless stated otherwise)."""

    n: int = 40
    n_steps: int = 1200
    tau: float = 0.7
    kernel: str = "4pt"
    # --- rotor geometry -----------------------------------------------------
    r_hub: float = 4.0
    r_tip: float = 14.0
    chord: float = 5.0
    thickness: float = 1.2
    omega: float = 0.004
    blade_pitch_z: float = 0.0
    blades: int = 2
    n_radial: int = 7
    n_chord: int = 4
    # --- ice floe -----------------------------------------------------------
    ice_radius: float = 4.0
    ice_density: float = 0.9
    ice_start_offset: float = 5.0  # start height above the rotor plane (cells)
    ice_marker_spacing: float = 1.1
    # --- flow ---------------------------------------------------------------
    inflow: float = 0.012  # mean axial (z) speed, negative = downwards
    drive_gain: float = 0.02
    # --- DEM contact --------------------------------------------------------
    normal_stiffness: float = 8.0
    normal_damping: float = 65.0
    tangential_stiffness: float = 4.0
    tangential_damping: float = 6.0
    friction_coefficient: float = 0.3
    # --- numerics -----------------------------------------------------------
    # ``n_correctors > 0`` amplifies the direct-forcing gain (F ≈ 2ρΔu at
    # convergence) which is *unstable* for the explicit IBM ⇄ 6-DOF coupling of
    # a near-neutrally-buoyant floe (added-mass instability, density ratio 0.9).
    # Freely moving bodies therefore run with the single-pass (gain 1) forcing;
    # prescribed fixed/rotating boundaries use 2 correctors elsewhere.
    n_correctors: int = 0
    added_mass_stabilization: bool = True
    sample_every: int = 5


class IcePropellerIBM:
    """Two-way IBM (fluid) ⇄ DEM (ice–blade contact) ice–propeller simulator."""

    def __init__(self, config: IcePropellerConfig = IcePropellerConfig()):
        self.cfg = c = config
        self.n = n = c.n
        self.center = torch.tensor([0.5 * n, 0.5 * n, 0.5 * n], dtype=torch.float64)
        self.rho = torch.ones((n, n, n), dtype=torch.float64)
        zeros = torch.zeros_like(self.rho)
        self.f = equilibrium3d(
            self.rho, zeros, zeros, torch.full_like(self.rho, -c.inflow)
        )
        self.angle = 0.0
        self.drive = 0.0

        # --- ice particle (repository DEM type) ------------------------------
        pos = self.center.clone()
        pos[2] += c.ice_start_offset + c.ice_radius
        self.particle = IceParticle(
            position=pos,
            velocity=torch.tensor([0.0, 0.0, -c.inflow], dtype=torch.float64),
            angular_velocity=torch.zeros(3, dtype=torch.float64),
            radius=c.ice_radius,
            density=c.ice_density,
        )
        self.contact_model = LinearSpringDashpot(
            normal_stiffness=c.normal_stiffness,
            tangential_stiffness=c.tangential_stiffness,
            normal_damping=c.normal_damping,
            tangential_damping=c.tangential_damping,
            friction_coefficient=c.friction_coefficient,
        )
        self.dem = DEMSolver(
            DEMConfig(
                time_step=1.0,
                gravity=torch.zeros(3, dtype=torch.float64),
                contact_model=self.contact_model,
                drag_model="none",
                enable_wall_contacts=False,
            )
        )
        self.dem.initialize([self.particle])

        # --- cached ice marker directions ------------------------------------
        self._ice_local = sphere_markers(
            c.ice_radius, spacing=c.ice_marker_spacing, center=(0.0, 0.0, 0.0)
        )

        # --- diagnostics -----------------------------------------------------
        self.history: List[dict] = []
        self.audit: dict = {}
        self._p0: torch.Tensor | None = None
        self._injected = torch.zeros(3, dtype=torch.float64)
        self.contact_active_steps = 0
        self.max_overlap = 0.0
        self.diverged = False

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #
    def blade_markers(self, angle: float | None = None) -> torch.Tensor:
        """Markers on both faces of every blade at the given rotor angle."""
        c = self.cfg
        ang = self.angle if angle is None else angle
        pts = []
        r_vals = torch.linspace(c.r_hub, c.r_tip, c.n_radial, dtype=torch.float64)
        chord_vals = torch.linspace(
            -0.5 * c.chord, 0.5 * c.chord, c.n_chord, dtype=torch.float64
        )
        for b in range(c.blades):
            base = ang + b * 2.0 * math.pi / c.blades
            for dz in (-0.5 * c.thickness, 0.5 * c.thickness):
                for r in r_vals:
                    for s in chord_vals:
                        # chord offset is applied tangentially
                        phi = base + s / max(float(r.item()), 1e-9)
                        pts.append(
                            [
                                self.center[0] + r * math.cos(phi),
                                self.center[1] + r * math.sin(phi),
                                self.center[2] + dz + c.blade_pitch_z,
                            ]
                        )
        return torch.tensor(pts, dtype=torch.float64)

    def blade_tip_speed(self) -> float:
        return self.cfg.omega * self.cfg.r_tip

    def _rotor_surface_velocity(self, pts: torch.Tensor) -> torch.Tensor:
        rel = pts - self.center
        w = torch.tensor([0.0, 0.0, self.cfg.omega], dtype=torch.float64)
        return torch.cross(w.expand(rel.shape[0], 3), rel, dim=1)

    # ------------------------------------------------------------------ #
    # contacts
    # ------------------------------------------------------------------ #
    def ice_blade_contact(self) -> dict:
        """Detect ice–blade contact and evaluate the DEM contact force.

        The blade is a flat plate of thickness ``t`` in the plane ``z = z_c``:
        contact requires the sphere centre to be within the radial span
        ``[r_hub, r_tip]``, within the chord window around the current blade
        azimuth, and ``|z − z_c| < R + t/2``.
        """
        c = self.cfg
        p = self.particle
        rel = p.position - self.center
        r = float(torch.sqrt(rel[0] ** 2 + rel[1] ** 2).item())
        theta = math.atan2(float(rel[1].item()), float(rel[0].item()))
        dz = float(rel[2].item())
        info = {
            "active": False,
            "overlap": 0.0,
            "force": [0.0, 0.0, 0.0],
            "torque_on_ice": [0.0, 0.0, 0.0],
            "rotor_torque_z": 0.0,
            "contact_point": [0.0, 0.0, 0.0],
        }
        if not (c.r_hub - p.radius <= r <= c.r_tip + p.radius):
            return info
        if abs(dz) >= p.radius + 0.5 * c.thickness:
            return info
        # chord window of the closest blade
        for b in range(c.blades):
            base = self.angle + b * 2.0 * math.pi / c.blades
            dphi = (theta - base + math.pi) % (2.0 * math.pi) - math.pi
            s = dphi * r  # tangential offset from the blade centreline
            if abs(s) > 0.5 * c.chord + p.radius:
                continue
            if r > c.r_tip or r < c.r_hub:
                continue
            normal = torch.tensor([0.0, 0.0, math.copysign(1.0, dz)], dtype=torch.float64)
            overlap = p.radius + 0.5 * c.thickness - abs(dz)
            if overlap <= 0.0:
                return info
            # contact point on the sphere surface
            contact_point = p.position - normal * p.radius
            v_ice = p.velocity + torch.cross(p.angular_velocity, contact_point - p.position, dim=0)
            v_blade = torch.cross(
                torch.tensor([0.0, 0.0, c.omega], dtype=torch.float64),
                contact_point - self.center,
                dim=0,
            )
            v_rel = v_ice - v_blade
            ci = self.contact_model.compute_contact_force(
                p, None, normal, overlap, v_rel
            )
            force = ci.total_force
            r_on_ice = contact_point - p.position
            torque_ice = torch.cross(r_on_ice, force, dim=0)
            # rotor torque from the equal-and-opposite reaction
            rotor_torque = -float(
                torch.cross(
                    contact_point - self.center, force, dim=0
                )[2].item()
            )
            info.update(
                {
                    "active": True,
                    "overlap": overlap,
                    "force": [float(v) for v in force.tolist()],
                    "torque_on_ice": [float(v) for v in torque_ice.tolist()],
                    "rotor_torque_z": rotor_torque,
                    "contact_point": [float(v) for v in contact_point.tolist()],
                    "tangential_force": [float(v) for v in ci.tangential_force.tolist()],
                    "normal_force": [float(v) for v in ci.normal_force.tolist()],
                }
            )
            return info
        return info

    # ------------------------------------------------------------------ #
    # one lattice step
    # ------------------------------------------------------------------ #
    def _interior_force(self, u_grid: torch.Tensor):
        """Eulerian direct forcing inside the floe (rigid-body velocity)."""
        c = self.cfg
        n = self.n
        p = self.particle
        z, y, x = torch.meshgrid(
            torch.arange(n, dtype=torch.float64),
            torch.arange(n, dtype=torch.float64),
            torch.arange(n, dtype=torch.float64),
            indexing="ij",
        )
        mask = (
            (x - p.position[0]) ** 2
            + (y - p.position[1]) ** 2
            + (z - p.position[2]) ** 2
        ) < max(c.ice_radius - 0.6, 0.5) ** 2
        rel = torch.stack([x - p.position[0], y - p.position[1], z - p.position[2]])
        u_solid = (
            p.velocity.view(3, 1, 1, 1)
            + torch.cross(
                p.angular_velocity.view(3, 1, 1, 1).expand(3, n, n, n),
                rel,
                dim=0,
            )
        )
        force = interior_mask_force(u_grid, mask, u_solid)
        # reaction torque about the particle centre (marker-frame consistent)
        fx, fy, fz = force[0], force[1], force[2]
        tqx = (rel[1] * fz - rel[2] * fy).sum()
        tqy = (rel[2] * fx - rel[0] * fz).sum()
        tqz = (rel[0] * fy - rel[1] * fx).sum()
        return force, torch.stack([tqx, tqy, tqz])

    def step(self) -> dict:
        c = self.cfg
        n = self.n
        # 1) ice markers + IBM force
        ice_pts = self._ice_local + self.particle.position.unsqueeze(0)
        st_ice = build_stencil(ice_pts, (n, n, n), c.kernel)
        u_ice = self.particle.velocity.unsqueeze(0) + torch.cross(
            self.particle.angular_velocity.expand(ice_pts.shape[0], 3),
            ice_pts - self.particle.position.unsqueeze(0),
            dim=1,
        )
        # 2) blade markers + prescribed rigid rotation
        blade_pts = self.blade_markers()
        st_blade = build_stencil(blade_pts, (n, n, n), c.kernel)
        u_blade = self._rotor_surface_velocity(blade_pts)

        rho_f, ux, uy, uz = macroscopic3d(self.f)
        u_grid = torch.stack([ux, uy, uz])
        f_m_ice, g_ice, bf_ice, tq_ice = direct_forcing(
            st_ice, u_grid, u_ice, rho_f, c.n_correctors
        )
        # Interior solid mask: the marker shell alone would leave the floe
        # hollow (the trapped fluid is dragged along and inflates the apparent
        # hydrodynamic load).
        f_int, tq_int = self._interior_force(u_grid)
        bf_ice = bf_ice - f_int.sum(dim=(1, 2, 3))
        tq_ice = tq_ice - tq_int
        f_m_blade, g_blade, bf_blade, tq_blade = direct_forcing(
            st_blade, u_grid, u_blade, rho_f, c.n_correctors
        )
        if not self.audit:
            self.audit = {
                "ice": momentum_audit(st_ice, f_m_ice),
                "blade": momentum_audit(st_blade, f_m_blade),
                "marker_force_ice": [float(v) for v in bf_ice.tolist()],
                "marker_force_blade": [float(v) for v in bf_blade.tolist()],
                "rotor_torque_ibm": float(tq_blade[2]),
                "rotor_torque_ibm_vector": [float(v) for v in tq_blade.tolist()],
            }

        # 3) DEM contact
        contact = self.ice_blade_contact()
        if contact["active"]:
            self.contact_active_steps += 1
            self.max_overlap = max(self.max_overlap, contact["overlap"])

        # 4) DEM integration with IBM + contact loads
        # ``direct_forcing`` returns the force *on the body* (=-sum of marker
        # forces), so the hydrodynamic load adds directly to the contact load.
        # Explicit IBM ⇄ 6-DOF coupling of a floe with density ratio 0.9 is at
        # the added-mass stability limit; the hydrodynamic load is therefore
        # scaled by m/(m + C_A m_displaced) (C_A = 0.5, potential-flow value),
        # the standard first-order explicit added-mass correction.  The scale
        # factor is reported so the bias stays auditable.
        m_p = float(self.particle.mass)
        m_disp = float(self.particle.volume)  # rho_fluid = 1
        scale = m_p / (m_p + 0.5 * m_disp) if c.added_mass_stabilization else 1.0
        self.added_mass_scale = scale
        f_ext = (scale * bf_ice).unsqueeze(0) + torch.tensor(
            contact["force"], dtype=torch.float64
        ).unsqueeze(0)
        t_ext = (scale * tq_ice).unsqueeze(0) + torch.tensor(
            contact["torque_on_ice"], dtype=torch.float64
        ).unsqueeze(0)
        self.dem.set_external_forces(f_ext, t_ext)
        self.dem.step()

        # 5) adaptive axial drive + body force on the fluid
        mean_uz = float(uz.mean().item())
        self.drive += c.drive_gain * (-c.inflow - mean_uz)
        F = torch.stack(
            [
                g_ice[0] + g_blade[0] + f_int[0],
                g_ice[1] + g_blade[1] + f_int[1],
                g_ice[2] + g_blade[2] + f_int[2] + self.drive,
            ]
        )
        self._injected += F.sum(dim=(1, 2, 3))
        self.f = collide_bgk3d(self.f, c.tau)
        self.f = body_source_3d(self.f, F[0], F[1], F[2])
        self.f = stream3d(self.f)
        self.angle += c.omega

        rho_n, uxn, uyn, uzn = macroscopic3d(self.f)
        finite = bool(
            torch.isfinite(rho_n).all()
            and torch.isfinite(self.particle.velocity).all()
            and abs(float(uxn.abs().max())) < 1.0
        )
        if not finite:
            self.diverged = True
        return {
            "t": self.dem.time,
            "angle": self.angle,
            "ice_pos": [float(v) for v in self.particle.position.tolist()],
            "ice_vel": [float(v) for v in self.particle.velocity.tolist()],
            "ice_pos_z_rel": float(self.particle.position[2].item() - self.center[2].item()),
            "ice_omega": [float(v) for v in self.particle.angular_velocity.tolist()],
            "f_ibm_ice": [float(v) for v in bf_ice.tolist()],
            "f_contact": contact["force"],
            "contact_active": contact["active"],
            "overlap": contact["overlap"],
            "rotor_torque_ibm": float(tq_blade[2]),
            "rotor_torque_contact": contact["rotor_torque_z"],
            "rotor_torque_total": float(tq_blade[2]) + contact["rotor_torque_z"],
            "fluid_momentum_z": float((rho_n * uzn).sum().item()),
            "injected_momentum_z": float(self._injected[2].item()),
            "ke_ice": self.particle.kinetic_energy(),
        }

    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        c = self.cfg
        n = self.n
        # initial fluid momentum (before the loop) for the conservation audit
        rho0, ux0, uy0, uz0 = macroscopic3d(self.f)
        p0_vec = torch.stack([(rho0 * u).sum() for u in (ux0, uy0, uz0)])
        steps_done = 0
        for step in range(c.n_steps):
            rec = self.step()
            steps_done = step + 1
            if step % c.sample_every == 0:
                self.history.append(rec)
            if self.diverged:
                break
        rho1, ux1, uy1, uz1 = macroscopic3d(self.f)
        p1_vec = torch.stack([(rho1 * u).sum() for u in (ux1, uy1, uz1)])
        dp = p1_vec - p0_vec
        residual = (dp - self._injected).abs().max().item()
        scale = float(dp.abs().max().item()) or 1.0
        pos = torch.tensor([h["ice_pos"] for h in self.history])
        vz = torch.tensor([h["ice_vel"][2] for h in self.history])
        f_contact = torch.tensor([h["f_contact"] for h in self.history])
        f_ibm = torch.tensor([h["f_ibm_ice"] for h in self.history])
        contact_steps = sum(1 for h in self.history if h["contact_active"])
        # capture/expulsion classification: contact episodes and the departure
        # speed of the floe after the last contact
        captured = contact_steps > 0
        last_contact = max(
            (i for i, h in enumerate(self.history) if h["contact_active"]), default=None
        )
        departed = False
        departure_speed = 0.0
        if last_contact is not None and last_contact + 3 < len(self.history):
            departure_speed = float(vz[last_contact:].abs().max().item())
            departed = departure_speed > 0.3 * self.blade_tip_speed()
        return {
            "case": "ice_propeller_ibm_dem_3d",
            "params": {
                "n": n,
                "n_steps": c.n_steps,
                "tau": c.tau,
                "nu": (c.tau - 0.5) / 3.0,
                "omega": c.omega,
                "blade_tip_speed": self.blade_tip_speed(),
                "r_hub": c.r_hub,
                "r_tip": c.r_tip,
                "chord": c.chord,
                "thickness": c.thickness,
                "ice_radius": c.ice_radius,
                "ice_density_ratio": c.ice_density,
                "ice_mass": self.particle.mass,
                "inflow": c.inflow,
                "kernel": c.kernel,
                "n_correctors": c.n_correctors,
                "contact_model": "LinearSpringDashpot",
                "added_mass_stabilization": c.added_mass_stabilization,
                "added_mass_force_scale": getattr(self, "added_mass_scale", 1.0),
                "normal_stiffness": c.normal_stiffness,
                "reynolds_blade_tip": self.blade_tip_speed() * c.r_tip / ((c.tau - 0.5) / 3.0),
            },
            "metrics": {
                "stable": not self.diverged,
                "steps_completed": steps_done,
                "fluid_momentum_audit_residual": residual,
                "fluid_momentum_audit_rel_residual": residual / scale,
                "injected_momentum_z": float(self._injected[2].item()),
                "fluid_momentum_change_z": float(dp[2].item()),
                "contact_steps": contact_steps,
                "max_overlap": self.max_overlap,
                "captured": captured,
                "expelled": departed,
                "departure_speed": departure_speed,
                "peak_contact_force": float(f_contact.abs().max().item()),
                "peak_ibm_force_on_ice": float(f_ibm.abs().max().item()),
                "mean_ibm_force_on_ice_z": float(f_ibm[:, 2].mean().item()),
                "max_ice_z_rel": float(pos[:, 2].max().item() - self.center[2].item()),
                "min_ice_z_rel": float(pos[:, 2].min().item() - self.center[2].item()),
                "final_ice_speed": float(vz[-1].item()),
                "final_ice_z_rel": float(pos[-1, 2].item() - self.center[2].item()),
                "rotor_torque_ibm_mean": float(
                    torch.tensor([h["rotor_torque_ibm"] for h in self.history]).mean().item()
                ),
                "rotor_torque_contact_mean": float(
                    torch.tensor(
                        [h["rotor_torque_contact"] for h in self.history]
                    ).mean().item()
                ),
                "ke_ice_initial": float(self.history[0]["ke_ice"]),
                "ke_ice_max": float(max(h["ke_ice"] for h in self.history)),
            },
            "force_audit": self.audit,
            "series": {
                "t": [h["t"] for h in self.history],
                "ice_x": [h["ice_pos"][0] for h in self.history],
                "ice_y": [h["ice_pos"][1] for h in self.history],
                "ice_z": [h["ice_pos"][2] for h in self.history],
                "ice_z_rel": [h["ice_pos_z_rel"] for h in self.history],
                "ice_vz": [h["ice_vel"][2] for h in self.history],
                "f_contact_z": [h["f_contact"][2] for h in self.history],
                "f_ibm_z": [h["f_ibm_ice"][2] for h in self.history],
                "rotor_torque_total": [h["rotor_torque_total"] for h in self.history],
                "contact_active": [h["contact_active"] for h in self.history],
                "overlap": [h["overlap"] for h in self.history],
            },
        }


def _last_marker_forces(stencil, u_grid, u_target) -> torch.Tensor:
    """Marker forces for the audit (single-pass, no corrector) — cheap."""
    u_int = stencil.interpolate(u_grid).T
    return (u_target.T - u_int).T


def run_ice_propeller_case(config: IcePropellerConfig | None = None) -> dict:
    """Convenience wrapper: build the simulator, run it, return the result dict."""
    sim = IcePropellerIBM(config or IcePropellerConfig())
    return sim.run()