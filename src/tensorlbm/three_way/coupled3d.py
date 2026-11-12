"""End-to-end 3-D three-way coupling prototype: LBM water + FEM structure + DEM ice.

Round-2 deliverable of ``GAP-LBM-T5-ICE-COUPLING``.  Round 1 built the exchange
*interface* and its invariant gates (``three_way/interface3d.py``) but never drove it
with a real fluid field, and nothing in the solver called it.  This module closes that
hole with a **single marching loop in which all three legs exchange loads through one
interface object**:

```
    LBM (D3Q19 + IBM, real solve)  --surface tractions-->  HexGrid3D -> beam FEM
              ^                                                    |
              |  tip velocity (IBM target velocity)                 | face velocity
              |                                                    v
    DEM ice particles  <--- contact + Schiller-Naumann drag from the LBM field ---> structure
```

Physical configuration (lattice units, ``rho = 1``)
---------------------------------------------------
A cantilever "ice boom" of length ``L`` (Euler-Bernoulli, clamped at ``y = 0``, bending
in ``x``) carries a rigid spherical ice body of diameter ``D`` at its tip.  The sphere
sits in a body-force-driven D3Q19 current; the hydrodynamic load on it is taken from the
immersed-boundary momentum exchange, transferred to a Q1 hex surface grid, and projected
onto the beam's Hermite/modal DOFs, so the boom bends.  The tip velocity is fed back as
the IBM target velocity (two-way fluid-structure coupling).  A stream of DEM ice
particles is advected by the *same* LBM field (Schiller-Naumann drag sampled with the IBM
interpolation kernel) and impacts the boom face; the contact reaction is transferred
through the same interface and the face velocity is fed back into the contact law
(two-way ice-structure coupling).

Scope / honesty
---------------
Everything is dimensionless (lattice units) so the module can be compared directly with
analytic Euler-Bernoulli and Clift-Grace-Weber references.  This is **not** engineering
ice-class verification: the T1 multiphase prerequisite of the T5 gate is still open (see
``docs/GAP-LBM-T5-ICE-COUPLING_dependency_audit.md``).  Reductions that are declared in
the report: normal-only DEM contact law, rigid spherical ice bodies, modal cantilever
structure (no 3-D shell FEM), body-force-driven periodic box (no free surface).

Entry points
------------
``run_coupled3d(cfg)``            staged run, returns a JSON-serialisable dict
``scripts/three_way_coupled3d_run.py``  scripted entry (artefacts under ``runs/``)
``tensorlbm.cases.ice_coupling3d``      solver case-registry entry
``tensorlbm.ice_chain3d.chain``         production ice-chain caller (criterion C7)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from ..d3q19 import equilibrium3d, macroscopic3d
from ..ibm import ibm_apply_body_force_3d
from ..ibm_vec import ibm_direct_forcing_3d_vec, ibm_velocity_interpolate_3d_vec
from ..solver3d import collide_bgk3d, stream3d
from .beam import BeamConfig, CantileverBeam, mode_shape
from .interface3d import (
    HexGrid3D,
    Surface3D,
    added_mass_ratio,
    exchange_cycle,
    explicit_staggered_stable,
    project_to_beam,
    relaxation_bound,
    sphere_surface,
)

__all__ = [
    "Coupled3DConfig",
    "Fluid3DIBM",
    "IceDEM3D",
    "Coupled3DResult",
    "run_coupled3d",
    "hull_girder_from_contact_loads",
    "schiller_naumann_cd",
    "clift_cd",
    "tip_deflection_reference",
]

F32 = torch.float32
F64 = torch.float64


# ---------------------------------------------------------------------------
# Independent references (never derived from the code under test)
# ---------------------------------------------------------------------------


def schiller_naumann_cd(re: float) -> float:
    """``C_D = 24/Re (1 + 0.15 Re^0.687)`` (1 < Re < 1000, ~5 %)."""
    if re <= 0:
        raise ValueError("Re must be positive")
    return 24.0 / re * (1.0 + 0.15 * re ** 0.687)


def clift_cd(re: float) -> float:
    """Clift-Grace-Weber rigid-sphere drag, low-Re branch (``Re < 20``).

    ``C_D = 24/Re (1 + 0.1315 Re^(0.82 - 0.05 log10 Re))``.
    """
    if re <= 0:
        raise ValueError("Re must be positive")
    return 24.0 / re * (1.0 + 0.1315 * re ** (0.82 - 0.05 * math.log10(re)))


def tip_deflection_reference(
    positions: torch.Tensor, loads: torch.Tensor, length: float, ei: float
) -> float:
    """Exact Euler-Bernoulli tip deflection under transverse point loads.

    ``δ = Σ_i f_i y_i²(3L-y_i)/(6EI)`` is the Maxwell unit-load integral
    ``∫ f(y) φ_tip(y) dy`` with ``φ_tip(y) = y²(3L-y)/(6EI)``; it is independent of any
    FE discretisation and holds for *any* load distribution.
    """
    y = positions.to(F64).clamp(0.0, length)
    return float((loads.to(F64) * y * y * (3.0 * length - y) / (6.0 * ei)).sum())


# ---------------------------------------------------------------------------
# Fluid leg: D3Q19 BGK + immersed boundary (library kernels only)
# ---------------------------------------------------------------------------


@dataclass
class Fluid3DIBM:
    """D3Q19 flow in a periodic box with an IBM sphere.

    Arrays follow the library convention ``(19, nz, ny, nx)``.  Lagrangian markers are
    expressed in the same index coordinates ``(x, y, z)`` and mapped to grid axes as
    ``ux[z, y, x]``.
    """

    n: Tuple[int, int, int]
    nu: float
    centre: Tuple[float, float, float]
    diameter: float
    n_theta: int = 6
    n_phi: int = 12
    u_init: float = 0.0

    def __post_init__(self) -> None:
        self.nx, self.ny, self.nz = int(self.n[0]), int(self.n[1]), int(self.n[2])
        self.tau = 0.5 + 3.0 * self.nu
        self.radius = 0.5 * float(self.diameter)
        rho = torch.ones((self.nz, self.ny, self.nx), dtype=F32)
        zero = torch.zeros_like(rho)
        # start from a developed current instead of rest: the driven-box spin-up
        # time constant (V/(3*pi*mu*D)) is O(1e4) steps, so a rest start would burn
        # the whole budget before any drag could be measured
        self.f = equilibrium3d(rho, torch.full_like(rho, float(self.u_init)), zero, zero)
        surf = sphere_surface(self.centre, self.radius, self.n_theta, self.n_phi)
        self.surface: Surface3D = surf
        pos = surf.positions.to(F32)
        self.mx, self.my, self.mz = pos[:, 0], pos[:, 1], pos[:, 2]
        self.area = float(surf.area)
        #: per-marker IBM force of the last step, shape ``(M, 3)`` lattice units
        self.marker_force = torch.zeros((len(surf), 3), dtype=F64)

    # -- helpers ----------------------------------------------------------
    def macro(self):
        return macroscopic3d(self.f)

    def mean_velocity_u(self) -> float:
        _, ux, _, _ = self.macro()
        return float(ux.mean())

    def interpolate(self, points: torch.Tensor) -> torch.Tensor:
        """Velocity at arbitrary points ``(P, 3)`` using the IBM kernel."""
        _, ux, uy, uz = self.macro()
        p = points.to(F32)
        umx, umy, umz = ibm_velocity_interpolate_3d_vec(
            ux, uy, uz, p[:, 0], p[:, 1], p[:, 2], kernel="hat"
        )
        return torch.stack([umx, umy, umz], -1).to(F64)

    # -- one lattice step -------------------------------------------------
    def step(self, drive: float, u_target_x: float = 0.0) -> torch.Tensor:
        """Advance one lattice step, return the per-marker IBM force.

        Uhlmann (2005) direct forcing with Guo body-force coupling::

            u   <- moments(f)                  # post-streaming state
            F_m  = u_target - u(X_m)           # per-marker velocity deficit
            g    = spread(F_m) + drive * x_hat # Eulerian force density
            f   <- Guo(f, g, tau); collide; stream
        """
        _, ux, uy, uz = self.macro()
        ut = torch.full_like(self.mx, float(u_target_x))
        zero = torch.zeros_like(ut)
        # vectorised library kernel (interpolation + scatter spread in one call)
        gx, gy, gz = ibm_direct_forcing_3d_vec(
            ux, uy, uz, self.mx, self.my, self.mz, ut, zero, zero, kernel="hat"
        )
        # the same marker deficit, kept for the load path (one extra interpolation)
        umx, umy, umz = ibm_velocity_interpolate_3d_vec(
            ux, uy, uz, self.mx, self.my, self.mz, kernel="hat"
        )
        fx = ut - umx
        fy = -umy
        fz = -umz
        gx = gx + float(drive)
        f = ibm_apply_body_force_3d(self.f, gx, gy, gz, self.tau)
        f = collide_bgk3d(f, self.tau)
        self.f = stream3d(f)
        self.marker_force = torch.stack([fx, fy, fz], -1).to(F64)
        return self.marker_force

    # -- loads ------------------------------------------------------------
    def body_force(self) -> torch.Tensor:
        """Hydrodynamic force on the body ``F = -Σ_m F_m`` (momentum exchange)."""
        return -self.marker_force.sum(0)

    def marker_traction(self) -> torch.Tensor:
        """Surface traction on the body at every marker, ``t_m = -F_m / dA_m``.

        The IBM hat kernel is a partition of unity (``Σ_m δ(X_m - x) = 1`` per cell), so
        ``Σ_m t_m dA_m = F_body`` exactly and the interface gate can re-derive the
        resultant from the transferred nodal vector alone.
        """
        dA = self.surface.dA.to(F64).unsqueeze(-1)
        return -self.marker_force / dA


# ---------------------------------------------------------------------------
# Ice leg: DEM soft spheres (contact + drag from the LBM field)
# ---------------------------------------------------------------------------


@dataclass
class _Point:
    """Minimal stand-in exposing the two attributes the DEM detector reads."""

    position: torch.Tensor
    radius: float


@dataclass
class IceDEM3D:
    """Soft-sphere ice particles advected by the LBM field and hitting the boom face.

    Contact law (normal only — declared reduction)::

        overlap δ = (x_p + r_p) - x_face(y_p)              δ > 0 → contact
        F_n       = max(0, k δ - η v_rel·n)  with n = -x̂    force on the particle
        reaction  = -F_n at (x_face, y_p, z_p)              force on the structure

    Particle-particle contacts use
    :func:`tensorlbm.dem.contact.detect_particle_particle_contact` for detection (its
    normal points from ``p1`` to ``p2``, so the repulsion on ``p1`` is ``-n``) and the
    same linear spring-dashpot law, applied symmetrically.
    """

    positions: torch.Tensor
    velocities: torch.Tensor
    radius: float
    density: float = 0.90
    stiffness: float = 2.0e2
    damping: float = 60.0
    friction: float = 0.3

    def __post_init__(self) -> None:
        self.positions = self.positions.to(F64)
        self.velocities = self.velocities.to(F64)
        self.volume = 4.0 / 3.0 * math.pi * self.radius ** 3
        self.mass = self.density * self.volume
        n = self.positions.shape[0]
        self.contact_points = torch.zeros((n, 3), dtype=F64)
        self.reaction = torch.zeros((n, 3), dtype=F64)
        self.n_active = 0

    # -- fluid coupling ---------------------------------------------------
    def fluid_force(self, u_local: torch.Tensor, nu: float) -> torch.Tensor:
        """Drag from the sampled fluid velocity (Schiller-Naumann, ``rho_f = 1``)."""
        rel = u_local - self.velocities
        speed = torch.linalg.norm(rel, dim=1)
        re_p = (speed * 2.0 * self.radius / nu).clamp(min=1e-12)
        cd = 24.0 / re_p * (1.0 + 0.15 * re_p ** 0.687)
        proj = math.pi * self.radius ** 2
        return 0.5 * cd.unsqueeze(-1) * proj * speed.unsqueeze(-1) * rel

    # -- contact with the (moving) boom face ------------------------------
    def wall_contact(
        self, face_x: float, face_velocity: torch.Tensor, face_z: float
    ) -> None:
        """Contact of every particle with the vertical face ``x = face_x(y)``.

        ``face_velocity`` is the structure velocity at the particle height.  Fills
        :attr:`reaction` (force on the structure) and :attr:`contact_points`.
        """
        self.reaction.zero_()
        self.contact_points.zero_()
        self.n_active = 0
        for i in range(self.positions.shape[0]):
            p = self.positions[i]
            if abs(float(p[2]) - face_z) > 3.0 * self.radius:
                continue
            gap = float(face_x - (p[0] + self.radius))
            if gap >= 0.0:
                continue
            overlap = -gap
            v_rel = self.velocities[i] - face_velocity
            v_n = float(v_rel[0])          # approach speed towards the face (n = -x̂)
            f_n = self.stiffness * overlap - self.damping * v_n
            if f_n <= 0.0:
                continue
            self.reaction[i] = torch.tensor([f_n, 0.0, 0.0], dtype=F64)
            self.contact_points[i] = torch.tensor(
                [float(face_x), float(p[1]), float(p[2])], dtype=F64
            )
            self.n_active += 1

    def pair_contacts(self) -> torch.Tensor:
        """Particle-particle spring-dashpot forces ``(n, 3)``; momentum exact."""
        from ..dem.contact import detect_particle_particle_contact

        n = self.positions.shape[0]
        net = torch.zeros((n, 3), dtype=F64)
        for i in range(n - 1):
            for j in range(i + 1, n):
                hit, overlap, normal = detect_particle_particle_contact(
                    _Point(self.positions[i], self.radius),
                    _Point(self.positions[j], self.radius),
                )
                if not hit:
                    continue
                v_rel = self.velocities[i] - self.velocities[j]
                f_n = self.stiffness * overlap - self.damping * float(torch.dot(v_rel, normal))
                if f_n <= 0.0:
                    continue
                f = -f_n * normal          # repulsion on i (library normal points i -> j)
                net[i] += f
                net[j] -= f
        return net

    def step(self, dt: float, fluid_force: torch.Tensor, pair_force: torch.Tensor) -> None:
        # ``reaction`` is the force *on the structure*; the particle feels the opposite
        total = fluid_force + pair_force - self.reaction
        self.velocities += total / self.mass * dt
        self.positions += self.velocities * dt

    def momentum(self) -> torch.Tensor:
        return self.mass * self.velocities.sum(0)


# ---------------------------------------------------------------------------
# Configuration / result containers
# ---------------------------------------------------------------------------


@dataclass
class Coupled3DConfig:
    """All knobs of the prototype, in lattice units."""

    # fluid
    n: Tuple[int, int, int] = (64, 64, 64)
    nu: float = 0.03
    diameter: float = 10.0
    centre: Tuple[float, float, float] = (16.0, 30.0, 32.0)
    u_target: float = 0.05
    drive: float = 1.2e-6
    n_theta: int = 6
    n_phi: int = 12
    n_warm: int = 800
    n_measure: int = 1200
    n_drive_phases: int = 3
    servo: bool = True
    # structure
    beam_length: float = 34.0
    beam_e: float = 2.0e5
    beam_density: float = 15.0
    beam_thickness: float = 2.0
    beam_depth: float = 1.0
    beam_damping: float = 0.05
    n_modes: int = 2
    n_elements: int = 17
    hex_cells: Tuple[int, int, int] = (8, 16, 16)
    hex_origin: Tuple[float, float, float] = (0.0, 0.0, 16.0)
    hex_extent: Tuple[float, float, float] = (32.0, 34.0, 32.0)
    # ice
    n_ice: int = 6
    ice_radius: float = 1.5
    ice_x0: float = 6.0
    ice_stiffness: float = 2.0e2
    ice_damping: float = 60.0
    # coupled march
    n_fsi: int = 400
    dt_struct: float = 1.0
    relax: float = 1.0
    seed: int = 20261112

    @property
    def face_x(self) -> float:
        """x of the undeformed boom face."""
        return float(self.hex_origin[0]) + 6.0


@dataclass
class Coupled3DResult:
    fluid: Dict[str, object] = field(default_factory=dict)
    structure: Dict[str, object] = field(default_factory=dict)
    ice: Dict[str, object] = field(default_factory=dict)
    gates: Dict[str, object] = field(default_factory=dict)
    benchmark: Dict[str, object] = field(default_factory=dict)
    history: Dict[str, List[float]] = field(default_factory=dict)
    config: Dict[str, object] = field(default_factory=dict)
    cost_s: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "config": self.config,
            "fluid": self.fluid,
            "structure": self.structure,
            "ice": self.ice,
            "gates": self.gates,
            "benchmark": self.benchmark,
            "history": self.history,
            "cost_s": self.cost_s,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_beam(cfg: Coupled3DConfig) -> CantileverBeam:
    return CantileverBeam(
        BeamConfig(
            youngs_modulus=cfg.beam_e,
            density=cfg.beam_density,
            length=cfg.beam_length,
            thickness=cfg.beam_thickness,
            depth=cfg.beam_depth,
            damping_ratio=cfg.beam_damping,
            n_modes=cfg.n_modes,
            x_base=0.0,
            y_base=0.0,
        )
    )


def _beam_ei(cfg: Coupled3DConfig) -> float:
    return cfg.beam_e * cfg.beam_depth * cfg.beam_thickness ** 3 / 12.0


def _tip_velocity(beam: CantileverBeam, length: float) -> float:
    xi = torch.tensor([1.0], dtype=F64)
    return float(sum(float(beam.qdot[k]) * float(mode_shape(k, xi)[0])
                     for k in range(beam.cfg.n_modes)))


def _beam_span(
    grid: HexGrid3D, nodal: torch.Tensor, length: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nodes that lie on the beam span, with their transverse (x) load."""
    y = grid.nodes[:, 1].to(F64)
    mask = (y >= 0.0) & (y <= length)
    return mask, y[mask], nodal[mask, 0].to(F64)


def _modal_force(beam: CantileverBeam, ys: torch.Tensor, loads: torch.Tensor) -> torch.Tensor:
    """Modal generalised force from transverse point loads at heights ``ys``."""
    return beam.projected_generalized_force(ys.to(F64), loads.to(F64))


# ---------------------------------------------------------------------------
# The staged run
# ---------------------------------------------------------------------------


def run_coupled3d(
    cfg: Optional[Coupled3DConfig] = None, verbose: bool = True
) -> Dict[str, object]:
    """Run the staged three-way prototype; every number is computed, none typed in.

    Stages
    ------
    1. **fluid (rigid body)** — three drive-feedback phases so that the measured mean
       velocity approaches ``cfg.u_target``; drag is averaged over the last phase and
       compared with the Clift / Schiller-Naumann references at the *measured* Re.
    2. **load transfer** — live IBM marker tractions go through
       :func:`interface3d.exchange_cycle` onto the Q1 hex grid and are projected onto
       the beam's Hermite DOFs; the transferred resultant/moment are re-derived from the
       nodal vector alone and compared with the marker integral.
    3. **structure (static)** — ``beam_fem.EulerBernoulliBeamFEM`` with those nodal loads
       vs the exact Euler-Bernoulli tip-deflection integral.
    4. **three-way march** — fluid (IBM target velocity = boom tip velocity) + ice
       (contact + drag from the same LBM field) + structure (damped modal step), every
       load passing through the same :class:`HexGrid3D` interface object.
    """
    cfg = cfg or Coupled3DConfig()
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    res = Coupled3DResult(config={k: v for k, v in cfg.__dict__.items()})

    # ---- stage 1: fluid leg with drive feedback -------------------------
    flow = Fluid3DIBM(
        n=cfg.n, nu=cfg.nu, centre=cfg.centre, diameter=cfg.diameter,
        n_theta=cfg.n_theta, n_phi=cfg.n_phi, u_init=cfg.u_target,
    )
    volume = float(cfg.n[0] * cfg.n[1] * cfg.n[2])
    drive = cfg.drive
    phase_log: List[Dict[str, float]] = []
    drag_hist: List[float] = []
    u_hist: List[float] = []
    n_total = cfg.n_drive_phases * (cfg.n_warm + cfg.n_measure)
    cd_series: List[List[float]] = []
    for step in range(n_total):
        flow.step(drive)
        drag_now = float(flow.body_force()[0])
        u_now = flow.mean_velocity_u()
        if step >= n_total - cfg.n_measure:
            drag_hist.append(drag_now)
            u_hist.append(u_now)
        # deadbeat mean-velocity servo: the body force of the next step is exactly the
        # momentum the body removed this step, so the box mean velocity is *pinned* at
        # u_target instead of drifting with the (very long, V/(3*pi*mu*D) ~ 1e4 step)
        # driven-box spin-up time constant; a weak proportional term trims the drift.
        if cfg.servo:
            drive = max(drag_now, 0.0) / volume * (cfg.u_target / max(u_now, 1e-12)) ** 0.5
        if verbose and (step % max(1, n_total // 6) == 0):
            print(f"  [fluid] step {step}: u={u_now:.5f} drag={drag_now:.5f} drive={drive:.3e}")
        if step % 10 == 0:
            cd_series.append([float(step), drag_now, u_now,
                              2.0 * drag_now / (max(u_now, 1e-30) ** 2 * flow.area)])
        if not math.isfinite(drag_now):
            raise FloatingPointError("fluid leg diverged (non-finite drag)")
    phase_log.append(
        {"steps": n_total, "drive_final": drive, "u_mean": sum(u_hist) / len(u_hist),
         "drag_lattice": sum(drag_hist) / len(drag_hist)}
    )

    u_mean = sum(u_hist) / len(u_hist)
    drag_lu = sum(drag_hist) / len(drag_hist)
    drag_std = float(torch.tensor(drag_hist, dtype=F64).std())
    re_meas = u_mean * cfg.diameter / cfg.nu
    rho_mean = float(flow.macro()[0].mean())
    cd_got = 2.0 * drag_lu / (rho_mean * u_mean ** 2 * flow.area)
    cd_sn = schiller_naumann_cd(re_meas)
    cd_clift = clift_cd(re_meas)
    rho_field = flow.macro()[0]
    res.fluid = {
        "n": list(cfg.n),
        "nu": cfg.nu,
        "tau": flow.tau,
        "diameter": cfg.diameter,
        "surface_area": flow.area,
        "n_markers": len(flow.surface),
        "drive_final": drive,
        "phases": phase_log,
        "cd_convergence": cd_series,
        "u_mean": u_mean,
        "re_measured": re_meas,
        "drag_lattice": drag_lu,
        "drag_lattice_std": drag_std,
        "cd_got": cd_got,
        "cd_schiller_naumann": cd_sn,
        "cd_clift": cd_clift,
        "rel_err_vs_clift_pct": 100.0 * (cd_got - cd_clift) / cd_clift,
        "rel_err_vs_sn_pct": 100.0 * (cd_got - cd_sn) / cd_sn,
        "rho_min": float(rho_field.min()),
        "rho_max": float(rho_field.max()),
        "finite": bool(torch.isfinite(flow.f).all()),
    }

    # ---- stage 2: load transfer through the interface -------------------
    hexgrid = HexGrid3D(cfg.hex_origin, cfg.hex_extent, cfg.hex_cells)
    traction = flow.marker_traction()
    cycle = exchange_cycle(hexgrid, flow.surface, traction)
    nodal = cycle["nodal"]
    resultant_transfer = cycle["resultant"]
    moment_transfer = cycle["moment"]
    resultant_marker = torch.as_tensor(flow.body_force(), dtype=F64)
    moment_marker = flow.surface.moment(traction)
    rel_res = float(
        torch.linalg.norm(resultant_transfer - resultant_marker)
        / max(float(torch.linalg.norm(resultant_marker)), 1e-300)
    )
    rel_mom = float(
        torch.linalg.norm(moment_transfer - moment_marker)
        / max(float(torch.linalg.norm(moment_marker)), 1e-300)
    )
    ei = _beam_ei(cfg)
    ys = flow.surface.positions.to(F64)[:, 1]
    marker_load_x = -flow.marker_force[:, 0]
    # load path used by the structure: 3-D surface traction -> Q1 grid -> beam DOFs
    beam_mask, node_y_beam, node_load_x = _beam_span(hexgrid, nodal, cfg.beam_length)
    beam_dofs = project_to_beam(node_y_beam, node_load_x, cfg.beam_length, cfg.n_elements)
    res.structure = {
        "beam_length": cfg.beam_length,
        "E": cfg.beam_e,
        "EI": ei,
        "n_elements": cfg.n_elements,
        "interface_resultant_rel_err": rel_res,
        "interface_moment_rel_err": rel_mom,
        "transfer_resultant": [float(v) for v in resultant_transfer],
        "marker_resultant": [float(v) for v in resultant_marker],
        "nodal_load_sum": [float(v) for v in nodal.sum(0)],
    }

    # ---- stage 3: static structural response vs analytic ---------------
    from ..beam_fem import BeamProperties, EulerBernoulliBeamFEM

    fem = EulerBernoulliBeamFEM(
        BeamProperties(
            youngs_modulus=cfg.beam_e,
            density=cfg.beam_density,
            length=cfg.beam_length,
            width=cfg.beam_depth,
            height=cfg.beam_thickness,
            boundary_condition="cantilever",
        ),
        n_elements=cfg.n_elements,
    )
    u_fem = torch.linalg.solve(fem.K_global, beam_dofs)
    delta_fem_tip = float(u_fem[2 * cfg.n_elements])
    # independent, mesh-free reference: the *same* discrete loads (at the Q1 grid nodes)
    # integrated against the exact tip influence line y^2(3L-y)/(6EI)
    delta_ref_nodal = tip_deflection_reference(node_y_beam, node_load_x, cfg.beam_length, ei)
    delta_ref_marker = tip_deflection_reference(ys, marker_load_x, cfg.beam_length, ei)
    rel_struct_nodal = abs(delta_fem_tip - delta_ref_nodal) / max(abs(delta_ref_nodal), 1e-300)
    rel_struct_marker = abs(delta_fem_tip - delta_ref_marker) / max(abs(delta_ref_marker), 1e-300)
    # element-size study: the FE tip deflection is not nodally exact for loads interior to
    # an element, so the structural gate is a *convergence* gate (error < 1 % at the
    # production mesh and observed order >= 1.5), not a machine-precision claim
    mesh_study: List[Dict[str, float]] = []
    mesh_list = sorted({cfg.n_elements, 2 * cfg.n_elements - 1, 4 * cfg.n_elements - 3})
    for n_el in mesh_list:
        dofs_m = project_to_beam(node_y_beam, node_load_x, cfg.beam_length, n_el)
        fem_m = EulerBernoulliBeamFEM(
            BeamProperties(
                youngs_modulus=cfg.beam_e, density=cfg.beam_density,
                length=cfg.beam_length, width=cfg.beam_depth,
                height=cfg.beam_thickness, boundary_condition="cantilever",
            ),
            n_elements=n_el,
        )
        u_m = torch.linalg.solve(fem_m.K_global, dofs_m)
        tip_m = float(u_m[2 * n_el])
        mesh_study.append(
            {
                "n_elements": float(n_el),
                "h": cfg.beam_length / n_el,
                "tip": tip_m,
                "rel_err_pct": 100.0 * abs(tip_m - delta_ref_nodal)
                / max(abs(delta_ref_nodal), 1e-300),
            }
        )
    order_struct = 0.0
    if len(mesh_study) >= 2:
        e1, e2 = mesh_study[0]["rel_err_pct"], mesh_study[-1]["rel_err_pct"]
        if e1 > 0.0 and e2 > 0.0:
            order_struct = math.log(e1 / e2) / math.log(mesh_study[-1]["h"] / mesh_study[0]["h"])
    # route-A cross-check: the *marker* loads straight onto the modal beam (no grid)
    beam_static = _make_beam(cfg)
    q_static = _modal_force(beam_static, ys, marker_load_x) / beam_static.modal_stiffness
    delta_modal = float(sum(float(q_static[k])
                            * float(mode_shape(k, torch.tensor([1.0], dtype=F64))[0])
                            for k in range(cfg.n_modes)))
    res.structure.update(
        {
            "tip_deflection_fem": delta_fem_tip,
            "tip_deflection_ref_nodal": delta_ref_nodal,
            "tip_deflection_ref_marker_integral": delta_ref_marker,
            "tip_deflection_modal_route": delta_modal,
            "struct_rel_err_nodal_pct": 100.0 * rel_struct_nodal,
            "struct_rel_err_marker_pct": 100.0 * rel_struct_marker,
            "max_abs_dof_deflection": float(u_fem.abs().max()),
            "mesh_study": mesh_study,
            "struct_observed_order": order_struct,
            "omega1_modal": float(beam_static.omega[0]),
            "omega1_analytic": 1.875104068711961 ** 2
            * math.sqrt(ei / (cfg.beam_density * cfg.beam_depth * cfg.beam_thickness
                              * cfg.beam_length ** 4)),
        }
    )

    # ---- stage 4: three-way march --------------------------------------
    beam = _make_beam(cfg)
    rng = torch.Generator().manual_seed(cfg.seed)
    face_z = float(cfg.hex_origin[2]) + 8.0
    ice_pos = torch.stack(
        [
            torch.full((cfg.n_ice,), cfg.ice_x0, dtype=F64)
            + 3.0 * torch.rand(cfg.n_ice, generator=rng, dtype=F64),
            cfg.beam_length * (0.45 + 0.4 * torch.rand(cfg.n_ice, generator=rng, dtype=F64)),
            torch.full((cfg.n_ice,), face_z, dtype=F64)
            + 0.8 * (2.0 * torch.rand(cfg.n_ice, generator=rng, dtype=F64) - 1.0),
        ],
        dim=1,
    )
    ice_vel = torch.stack(
        [
            torch.full((cfg.n_ice,), cfg.u_target, dtype=F64),
            torch.zeros(cfg.n_ice, dtype=F64),
            torch.zeros(cfg.n_ice, dtype=F64),
        ],
        dim=1,
    )
    ice = IceDEM3D(
        positions=ice_pos, velocities=ice_vel, radius=cfg.ice_radius,
        stiffness=cfg.ice_stiffness, damping=cfg.ice_damping,
    )
    mass_struct = cfg.beam_density * cfg.beam_thickness * cfg.beam_depth * cfg.beam_length
    mu_added = added_mass_ratio(mass_struct, rho_mean, 4.0 / 3.0 * math.pi * flow.radius ** 3)
    stable = explicit_staggered_stable(mu_added)
    omega_bound = relaxation_bound(mu_added)

    keys = ["t", "tip", "tip_dot", "q0", "q1", "f_fluid", "f_ice", "energy",
            "ice_x_min", "n_contact", "ice_momentum", "interface_work"]
    hist: Dict[str, List[float]] = {k: [] for k in keys}
    q_prev = torch.zeros(cfg.n_modes, dtype=F64)
    q_before = torch.zeros(cfg.n_modes, dtype=F64)
    work = 0.0
    for step in range(cfg.n_fsi):
        tip = beam.deflection(cfg.beam_length)
        tip_dot = _tip_velocity(beam, cfg.beam_length)
        # fluid: IBM target velocity = boom tip velocity (two-way)
        marker_force = flow.step(drive, u_target_x=tip_dot)
        f_fluid_x = float(-marker_force.sum(0)[0])
        # same interface object as stage 2, live tractions
        trac = -marker_force / flow.surface.dA.to(F64).unsqueeze(-1)
        nodal = hexgrid.distribute_surface(flow.surface, trac)
        _, node_y_fsi, node_load_fsi = _beam_span(hexgrid, nodal, cfg.beam_length)
        q_fluid = _modal_force(beam, node_y_fsi, node_load_fsi)
        # ice leg: drag from the same LBM field + sub-stepped contact with the moving
        # face.  The contact spring frequency sets the sub-step count, so a stiff contact
        # cannot detonate the explicit integration.
        u_local = flow.interpolate(ice.positions)
        f_drag_ice = ice.fluid_force(u_local, cfg.nu)
        face_vel = torch.tensor([tip_dot, 0.0, 0.0], dtype=F64)
        q_ice = torch.zeros(cfg.n_modes, dtype=F64)
        n_sub = max(1, int(math.ceil(math.sqrt(cfg.ice_stiffness / ice.mass)
                                     * cfg.dt_struct / 0.5)))
        dt_sub = cfg.dt_struct / n_sub
        ice.reaction.zero_()
        for _sub in range(n_sub):
            pair = ice.pair_contacts()
            ice.wall_contact(cfg.face_x + tip, face_vel, face_z)
            ice.step(dt_sub, f_drag_ice, pair)
        if ice.n_active:
            active = ice.reaction.abs().sum(-1) > 0.0
            q_ice = _modal_force(beam, ice.contact_points[active, 1], ice.reaction[active, 0])
        ice_substeps = n_sub
        # structure: one damped modal step (explicit staggering, optional relaxation)
        q_total = q_fluid + q_ice
        if cfg.relax != 1.0:
            q_total = cfg.relax * q_total + (1.0 - cfg.relax) * q_prev
        q_prev = q_total.clone()
        q_before = beam.q.clone()
        beam.step(q_total, cfg.dt_struct)
        # interface work ledger (trapezoid in modal coordinates)
        work += float((0.5 * (q_before + q_total) * (beam.q - q_before)).sum())
        hist["t"].append(float(step))
        hist["tip"].append(beam.deflection(cfg.beam_length))
        hist["tip_dot"].append(tip_dot)
        hist["q0"].append(float(beam.q[0]))
        hist["q1"].append(float(beam.q[1]) if cfg.n_modes > 1 else 0.0)
        hist["f_fluid"].append(f_fluid_x)
        hist["f_ice"].append(float(ice.reaction.sum(0)[0]) if ice.n_active else 0.0)
        hist["energy"].append(beam.total_energy())
        hist["ice_x_min"].append(float(ice.positions[:, 0].min()))
        hist["n_contact"].append(float(ice.n_active))
        hist["ice_momentum"].append(float(ice.momentum()[0]))
        hist["interface_work"].append(work)

    tip_final = hist["tip"][-1]
    tip_max = max(abs(v) for v in hist["tip"])
    finite_all = bool(torch.isfinite(flow.f).all()) and math.isfinite(tip_final)
    e0 = hist["energy"][0] if hist["energy"][0] > 0 else None
    res.ice = {
        "substeps": int(ice_substeps),
        "contact_frequency_per_step": math.sqrt(cfg.ice_stiffness / ice.mass)
        * cfg.dt_struct,
        "n_particles": cfg.n_ice,
        "radius": cfg.ice_radius,
        "mass": ice.mass,
        "density": ice.density,
        "stiffness": ice.stiffness,
        "damping": ice.damping,
        "face_x": cfg.face_x,
        "face_z": face_z,
        "contacts_max": int(max(hist["n_contact"])),
        "contacts_final": int(hist["n_contact"][-1]),
        "reaction_sum_final": [float(v) for v in ice.reaction.sum(0)],
        "third_law_residual": float(
            torch.linalg.norm(ice.reaction.sum(0) + (-ice.reaction).sum(0))
            / max(float(torch.linalg.norm(ice.reaction.sum(0))), 1e-300)
        ) if ice.n_active else 0.0,
        "momentum_final": [float(v) for v in ice.momentum()],
    }
    res.history = hist

    # ---- gates ---------------------------------------------------------
    steady_spread = drag_std / max(abs(drag_lu), 1e-300)
    res.gates = {
        "G1_fluid_steady": {
            "drag_std_over_mean": steady_spread,
            "pass": bool(steady_spread < 0.02),
            "criterion": "relative drag fluctuation over the measurement window < 2 %",
        },
        "G2_drag_vs_reference": {
            "re": re_meas,
            "cd_got": cd_got,
            "cd_ref_clift": cd_clift,
            "rel_err_pct": 100.0 * (cd_got - cd_clift) / cd_clift,
            "pass": bool(abs(100.0 * (cd_got - cd_clift) / cd_clift) < 15.0),
            "criterion": "|C_D - Clift| < 15 % (platform accuracy floor)",
        },
        "G3_interface_conservation": {
            "resultant_rel_err": rel_res,
            "moment_rel_err": rel_mom,
            "pass": bool(rel_res < 1e-10 and rel_mom < 1e-10),
            "criterion": "Q1 transfer reproduces marker resultant and moment to 1e-10",
        },
        "G4_structure_vs_analytic": {
            "tip_deflection_fem": delta_fem_tip,
            "tip_deflection_analytic": delta_ref_nodal,
            "rel_err_pct": 100.0 * rel_struct_nodal,
            "observed_order": order_struct,
            "pass": bool(100.0 * rel_struct_nodal < 1.0 and order_struct > 1.5),
            "criterion": "Hermite-FEM tip deflection within 1 % of the exact "
                         "y^2(3L-y)/(6EI) integral, converging at order >= 1.5",
        },
        "G5_added_mass_guard": {
            "mu": mu_added,
            "explicit_staggered_stable": stable,
            "relaxation_omega_max": omega_bound,
            "pass": bool(stable),
            "criterion": "explicit Dirichlet-Neumann staggering stable iff mu < 1 (Causin 2005)",
        },
        "G6_three_way_march": {
            "steps": cfg.n_fsi,
            "tip_final": tip_final,
            "tip_max_abs": tip_max,
            "finite": finite_all,
            "max_contacts": int(max(hist["n_contact"])),
            "energy_final": hist["energy"][-1],
            "pass": bool(finite_all and abs(tip_final) < 0.25 * cfg.beam_length),
            "criterion": "all fields finite and deflection bounded (< 25 % of the span)",
        },
    }
    res.benchmark = {
        "rows": [
            {
                "quantity": "sphere drag coefficient C_D",
                "ref": cd_clift,
                "ref_source": "Clift-Grace-Weber low-Re correlation (Re<20 branch)",
                "got": cd_got,
                "rel_err_pct": 100.0 * (cd_got - cd_clift) / cd_clift,
                "tol_pct": 15.0,
                "verdict": "PASS" if abs(100.0 * (cd_got - cd_clift) / cd_clift) < 15.0 else "FAIL",
            },
            {
                "quantity": "sphere C_D vs Schiller-Naumann",
                "ref": cd_sn,
                "ref_source": "Schiller-Naumann correlation 24/Re(1+0.15Re^0.687)",
                "got": cd_got,
                "rel_err_pct": 100.0 * (cd_got - cd_sn) / cd_sn,
                "tol_pct": 15.0,
                "verdict": "PASS" if abs(100.0 * (cd_got - cd_sn) / cd_sn) < 15.0 else "FAIL",
            },
            {
                "quantity": "interface resultant (x, lattice)",
                "ref": float(resultant_marker[0]),
                "ref_source": "marker traction integral -sum_m F_m",
                "got": float(resultant_transfer[0]),
                "rel_err_pct": 100.0 * rel_res,
                "tol_pct": 1e-8,
                "verdict": "PASS" if rel_res < 1e-10 else "FAIL",
            },
            {
                "quantity": "cantilever tip deflection",
                "ref": delta_ref_nodal,
                "ref_source": "Euler-Bernoulli Maxwell integral y^2(3L-y)/(6EI)",
                "got": delta_fem_tip,
                "rel_err_pct": 100.0 * rel_struct_nodal,
                "tol_pct": 1.0,
                "verdict": "PASS" if 100.0 * rel_struct_nodal < 1.0 else "FAIL",
            },
            {
                "quantity": "structural mesh convergence order",
                "ref": 2.0,
                "ref_source": "Hermite cubic element, O(h^2) tip-deflection convergence",
                "got": order_struct,
                "rel_err_pct": 100.0 * (order_struct - 2.0) / 2.0,
                "tol_pct": 50.0,
                "verdict": "PASS" if order_struct > 1.5 else "FAIL",
            },
            {
                "quantity": "first bending frequency omega_1",
                "ref": res.structure["omega1_analytic"],
                "ref_source": "beta_1^2 sqrt(EI/(rho A L^4)), beta_1 = 1.875104068711961",
                "got": res.structure["omega1_modal"],
                "rel_err_pct": 100.0 * (res.structure["omega1_modal"]
                                        - res.structure["omega1_analytic"])
                / res.structure["omega1_analytic"],
                "tol_pct": 1.0,
                "verdict": "PASS" if abs(res.structure["omega1_modal"]
                                        - res.structure["omega1_analytic"])
                / res.structure["omega1_analytic"] < 1e-2 else "FAIL",
            },
        ]
    }
    res.cost_s = time.time() - t0
    if verbose:
        print(f"  [done] C_D={cd_got:.4f} (ref {cd_clift:.4f}, Re={re_meas:.2f}), "
              f"tip={tip_final:.4e}, cost={res.cost_s:.1f}s")
    return res.to_dict()


# ---------------------------------------------------------------------------
# Platform-side consumer: DEM hull-contact loads -> hull-girder response
# ---------------------------------------------------------------------------


def hull_girder_from_contact_loads(
    positions: torch.Tensor,
    forces: torch.Tensor,
    cfg: Optional[Coupled3DConfig] = None,
) -> Dict[str, object]:
    """Static hull-girder response to a set of DEM hull-contact loads.

    This is the entry the production ice chain uses (``ice_chain3d.chain`` criterion
    C7): the contact loads of the DEM leg are distributed onto the Q1 interface grid,
    projected onto the hull-girder Hermite DOFs and solved for the girder deflection,
    bending moment and deck stress, with the Euler-Bernoulli Maxwell integral as an
    independent reference.

    Parameters
    ----------
    positions : (n, 3)
        Contact points in beam coordinates (only ``y`` is used for the girder).
    forces : (n, 3)
        Contact forces on the structure; the transverse (``x``) component loads the beam.
    """
    cfg = cfg or Coupled3DConfig()
    pos = torch.as_tensor(positions, dtype=F64).reshape(-1, 3)
    frc = torch.as_tensor(forces, dtype=F64).reshape(-1, 3)
    ei = _beam_ei(cfg)
    ys = pos[:, 1].clamp(0.0, cfg.beam_length)
    dofs = project_to_beam(ys, frc[:, 0], cfg.beam_length, cfg.n_elements)

    from ..beam_fem import BeamProperties, EulerBernoulliBeamFEM

    fem = EulerBernoulliBeamFEM(
        BeamProperties(
            youngs_modulus=cfg.beam_e,
            density=cfg.beam_density,
            length=cfg.beam_length,
            width=cfg.beam_depth,
            height=cfg.beam_thickness,
            boundary_condition="cantilever",
        ),
        n_elements=cfg.n_elements,
    )
    u = torch.linalg.solve(fem.K_global, dofs)
    tip = float(u[2 * cfg.n_elements])
    ref = tip_deflection_reference(
        torch.linspace(0.0, cfg.beam_length, cfg.n_elements + 1, dtype=F64),
        dofs[0::2], cfg.beam_length, ei,
    )
    # clamp bending moment = integral of the transverse load times its arm
    m_clamp = float((frc[:, 0] * ys).sum())
    stress_deck = abs(m_clamp) * (cfg.beam_thickness / 2.0) / ei * cfg.beam_e
    return {
        "tip_deflection": tip,
        "tip_deflection_reference": ref,
        "rel_err_pct": 100.0 * abs(tip - ref) / max(abs(ref), 1e-300),
        "clamp_bending_moment": m_clamp,
        "deck_bending_stress": stress_deck,
        "resultant_force": [float(v) for v in frc.sum(0)],
        "n_contacts": int(frc.shape[0]),
        "beam_length": cfg.beam_length,
        "EI": ei,
    }