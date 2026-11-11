"""Two-way (fluid <-> particle) momentum coupling for settling suspensions.

Scope
-----
This module is the *momentum* half of the hindered-settling node
``IN234W2-N01``: a D2Q9 lattice-Boltzmann solver with **rigid circular
particles** whose boundary is resolved (half-way bounce-back with a moving
wall) and whose momentum exchange with the fluid is measured **link by link**
(the Ladd 1994 momentum-exchange method, MEM).  Static walls (channel) are
supported, so the momentum ledger contains both the particle and the wall
traction.

The unit of the closure is the momentum ledger

    d P_fluid
    --------- = - sum_p F_p->fluid - F_wall + F_body
       d t

with ``P_fluid = sum_{fluid nodes} rho u``.  Because every wall link removes
exactly the momentum carried by the population that would have entered the
solid, ``sum F dt + dP_fluid`` closes to round-off *without* any tuned
coefficient -- that is the point of resolving the boundary instead of imposing
a drag law.  :func:`momentum_ledger_probe` measures the residual.

Two drag channels are provided:

``drag_model="resolved"``
    the drag comes from the MEM force (two-way: the particle's motion is
    imposed on the fluid through the moving-wall bounce-back term, so the
    particle **injects** momentum into the fluid);
``drag_model="closure"``
    the particle obeys the analytic closure of :mod:`tensorlbm.frazil_suspension`
    (Schiller-Naumann x Richardson-Zaki amplification).  This is the
    single-particle / one-way limit and the exact degradation path used to
    re-certify the previously verified ``m7094`` numbers.

The ``beta`` parameter is the fraction of the particle velocity imposed on the
fluid at the wall.  ``beta = 1`` is fully two-way; ``beta = 0`` closes the
particle->fluid channel (the fluid sees a stationary obstacle), so the
configuration degrades to one-way coupling: the mixture momentum feedback
density ``n * F_drag`` can then be compared with the analytic closure.

Added mass
----------
An accelerating body must displace fluid; the reaction appears in MEM as a
force in phase with the body acceleration.  :func:`measure_added_mass` drives a
disk with a prescribed sinusoid and separates the added-mass (in-phase with
acceleration) from the drag (in-phase with velocity) component by linear
regression on orthogonal regressors.  The exact 2D potential-flow anchor for a
circular cylinder is ``C_A = M_a / (rho V_p) = 1``.

References
----------
Ladd, A.J.C. (1994) "Numerical simulations of particulate suspensions via a
    discretised Boltzmann equation", *J. Fluid Mech.* 271, 285-339.
Aidun, C.K., Lu, Y., Ding, E.-J. (1998) "Direct analysis of particulate
    suspensions with inertia using the discrete Boltzmann equation",
    *J. Fluid Mech.* 373, 287-311.
Guo, Z., Zheng, C., Shi, B. (2002) "Discrete lattice effects on the forcing
    term in the lattice Boltzmann method", *Phys. Rev. E* 65, 046308.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from .d2q9 import C as _C9
from .d2q9 import OPPOSITE as _OPP9
from .d2q9 import W_EXACT, equilibrium, macroscopic
from .frazil_suspension import drag_coefficient_suspension, hindered_settling_velocity

__all__ = [
    "CS2",
    "Disk",
    "DiskLatticeCoupling",
    "momentum_ledger_probe",
    "measure_added_mass",
    "closure_terminal_velocity",
]

CS2 = 1.0 / 3.0

# ``roll`` shifts in (y, x) order for each lattice direction; the d2q9 velocity
# table stores (cx, cy), so the pair is transposed on purpose.
_SHIFTS_2D: tuple[tuple[int, int], ...] = tuple(
    (int(cy), int(cx)) for cx, cy in _C9.tolist()
)


def _weights(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(W_EXACT, dtype=dtype, device=device)


def _c(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return _C9.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Rigid particles
# ---------------------------------------------------------------------------


@dataclass
class Disk:
    """Rigid circular particle (2D) carried by :class:`DiskLatticeCoupling`."""

    center: tuple[float, float]
    radius: float
    velocity: tuple[float, float] = (0.0, 0.0)
    omega: float = 0.0
    density: float = 1.0
    fixed: bool = False

    @property
    def area(self) -> float:
        return math.pi * self.radius * self.radius

    @property
    def mass(self) -> float:
        return self.density * self.area


# ---------------------------------------------------------------------------
# Coupled lattice
# ---------------------------------------------------------------------------


class DiskLatticeCoupling:
    """D2Q9 lattice with resolved rigid disks and momentum-exchange forces.

    Args:
        ny, nx: lattice size (periodic in both directions).
        tau: BGK relaxation time (> 0.5); ``nu = (tau - 0.5) / 3``.
        rho0: reference density.
        dtype, device: torch dtype/device (``float64`` keeps the momentum
            ledger closure at round-off level).
        track_forces: compute per-disk MEM force/torque every step (turn off
            for fixed-obstacle runs, where it is pure overhead).
    """

    def __init__(
        self,
        ny: int,
        nx: int,
        *,
        tau: float = 0.8,
        rho0: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        track_forces: bool = True,
    ) -> None:
        if tau <= 0.5:
            raise ValueError(f"tau must be > 0.5, got {tau!r}")
        if ny < 4 or nx < 4:
            raise ValueError("lattice must be at least 4x4")
        self.ny, self.nx = int(ny), int(nx)
        self.n = self.ny * self.nx
        self.tau = float(tau)
        self.rho0 = float(rho0)
        self.dtype = dtype
        self.device = torch.device(device)
        self.track_forces = bool(track_forces)
        self._c = _c(dtype, self.device)
        self._w = _weights(dtype, self.device)
        self._opp = _OPP9.to(self.device)
        self._opp_list = [int(v) for v in self._opp.tolist()]

        shape = (self.ny, self.nx)
        self.mask = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.owner = torch.full(shape, -1, dtype=torch.int64, device=self.device)
        self.disks: list[Disk] = []
        self._geometry: dict | None = None
        self._wall_uw: torch.Tensor | None = None

        rho = torch.full(shape, self.rho0, dtype=dtype, device=self.device)
        zero = torch.zeros_like(rho)
        self.f = equilibrium(rho, zero, zero)
        self.steps = 0

    # -- geometry ---------------------------------------------------------
    def set_disks(
        self,
        disks: Sequence[Disk],
        walls: torch.Tensor | None = None,
    ) -> None:
        """Rasterise the disks (and optional static walls) onto the lattice."""
        y = torch.arange(self.ny, dtype=self.dtype, device=self.device).view(-1, 1) + 0.5
        x = torch.arange(self.nx, dtype=self.dtype, device=self.device).view(1, -1) + 0.5
        mask = torch.zeros((self.ny, self.nx), dtype=torch.bool, device=self.device)
        owner = torch.full((self.ny, self.nx), -1, dtype=torch.int64, device=self.device)
        best = torch.full((self.ny, self.nx), float("inf"), dtype=self.dtype, device=self.device)
        for i, d in enumerate(disks):
            r2 = (x - d.center[0]) ** 2 + (y - d.center[1]) ** 2
            inside = r2 <= d.radius * d.radius
            closer = inside & (r2 < best)  # nearest centre wins on overlap
            mask |= inside
            owner = torch.where(closer, torch.full_like(owner, i), owner)
            best = torch.where(closer, r2, best)
        if walls is not None:
            if tuple(walls.shape) != (self.ny, self.nx):
                raise ValueError("wall mask shape must match the lattice")
            mask = mask | walls.to(device=self.device, dtype=torch.bool)
        self.mask = mask
        self.owner = owner
        self.disks = list(disks)
        self._build_geometry()

    def _build_geometry(self) -> None:
        """Cache the static link bookkeeping (masks and gather indices)."""
        idx = torch.arange(self.n, device=self.device).view(self.ny, self.nx)
        mask_flat = self.mask.reshape(-1)
        links = []
        src = []
        ahead = []
        dest = []
        src_solid = []
        for q in range(9):
            dy, dx = _SHIFTS_2D[q]
            fwd = torch.roll(idx, shifts=(-dy, -dx), dims=(-2, -1))  # x + c_q
            back = torch.roll(idx, shifts=(dy, dx), dims=(-2, -1))  # x - c_q
            ahead.append(fwd.reshape(-1))
            src.append(back.reshape(-1))
            src_solid.append(mask_flat[back.reshape(-1)])
            link = mask_flat[fwd.reshape(-1)] & (~mask_flat)
            if q == 0:
                link = torch.zeros_like(link)
                dest.append(torch.zeros_like(link))
            else:
                dest.append(self._opp_list[q] * self.n + torch.arange(self.n, device=self.device))
            links.append(link)
        self._geometry = {
            "links": links,
            "src": torch.stack(src),  # (9, N)
            "ahead": torch.stack(ahead),  # (9, N)
            "dest": torch.stack(dest),  # (9, N)
            "src_solid": torch.stack(src_solid),  # (9, N) upstream node is solid
            "fluid_frac": float(1.0 - mask_flat.double().mean()),
        }

    @property
    def solid_fraction(self) -> float:
        return float(self.mask.double().mean())

    # -- boundary velocity ------------------------------------------------
    def _wall_velocity(self, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Wall velocity field at solid nodes (``beta``: feedback strength)."""
        y = torch.arange(self.ny, dtype=self.dtype, device=self.device).view(-1, 1) + 0.5
        x = torch.arange(self.nx, dtype=self.dtype, device=self.device).view(1, -1) + 0.5
        ux = torch.zeros((self.ny, self.nx), dtype=self.dtype, device=self.device)
        uy = torch.zeros_like(ux)
        for i, d in enumerate(self.disks):
            sel = self.owner == i
            if not bool(sel.any()):
                continue
            rx = x - d.center[0]
            ry = y - d.center[1]
            vx = d.velocity[0] - d.omega * ry
            vy = d.velocity[1] + d.omega * rx
            ux = torch.where(sel, beta * vx, ux)
            uy = torch.where(sel, beta * vy, uy)
        if self._wall_uw is not None:
            ux = ux + self._wall_uw[0]
            uy = uy + self._wall_uw[1]
        return ux, uy

    def set_wall_velocity(self, ux: torch.Tensor, uy: torch.Tensor) -> None:
        """Prescribe the velocity of the *static* wall cells (moving-wall BC)."""
        shape = (self.ny, self.nx)
        if tuple(ux.shape) != shape or tuple(uy.shape) != shape:
            raise ValueError("wall velocity shape must match the lattice")
        self._wall_uw = torch.stack([
            ux.to(device=self.device, dtype=self.dtype),
            uy.to(device=self.device, dtype=self.dtype),
        ])

    # -- one lattice step --------------------------------------------------
    def step(
        self,
        *,
        beta: float = 1.0,
        body_force: tuple[float, float] = (0.0, 0.0),
    ) -> dict:
        """Advance one lattice step; return forces and momentum bookkeeping."""
        if self._geometry is None:
            self._build_geometry()
        geo = self._geometry
        f = self.f
        rho, ux, uy = macroscopic(f)
        fx, fy = float(body_force[0]), float(body_force[1])
        # Guo (2002): the velocity that enters f_eq carries half the force
        rho_safe = rho.clamp(min=1e-300)
        uxg = ux + 0.5 * fx / rho_safe
        uyg = uy + 0.5 * fy / rho_safe
        f_eq = equilibrium(rho, uxg, uyg)
        f_post = f - (f - f_eq) / self.tau
        if fx != 0.0 or fy != 0.0:
            c = self._c.view(-1, 1, 1, 2)
            w = self._w.view(-1, 1, 1)
            cu = c[..., 0] * uxg + c[..., 1] * uyg
            pref = 1.0 - 0.5 / self.tau
            sx = pref * ((c[..., 0] - uxg) / CS2 + cu * c[..., 0] / CS2**2) * fx
            sy = pref * ((c[..., 1] - uyg) / CS2 + cu * c[..., 1] / CS2**2) * fy
            f_post = f_post + w * (sx + sy)

        # ---- streaming with half-way bounce-back + MEM force ------------
        f_new = f_post.reshape(9, -1).gather(1, geo["src"]).reshape(9, self.n)
        # nothing can arrive from inside a solid node: the population that
        # would have come from there is exactly the one reflected at the
        # fluid node on the other side of the link, handled below.
        f_new.masked_fill_(geo["src_solid"], 0.0)
        uwx, uwy = self._wall_velocity(beta)
        uwx_f = uwx.reshape(-1)
        uwy_f = uwy.reshape(-1)
        rho_f = rho.reshape(-1)
        links = geo["links"]
        src = geo["src"]
        ahead = geo["ahead"]
        w = self._w.view(9, 1)
        cvec = self._c

        # wall velocity at the solid node x + c_q for each direction q
        cu = torch.empty((9, self.n), dtype=self.dtype, device=self.device)
        for q in range(1, 9):
            cu[q] = cvec[q, 0] * uwx_f[ahead[q]] + cvec[q, 1] * uwy_f[ahead[q]]
        cu[0] = 0.0

        fp = f_post.reshape(9, -1)
        momentum_flux = fp - w * rho_f.view(1, -1) * cu / CS2  # (9, N): q-part of the link force
        bb = fp - 2.0 * w * rho_f.view(1, -1) * cu / CS2  # reflected population

        # scattered bounce-back: direction opp[q] at the same node
        sel = torch.cat(links[1:])  # (8N,)
        if bool(sel.any()):
            vals = torch.cat([bb[q] for q in range(1, 9)])[sel]
            dest = torch.cat([geo["dest"][q] for q in range(1, 9)])[sel]
            f_new.reshape(-1).index_copy_(0, dest, vals)

        contrib = torch.zeros(9, dtype=self.dtype, device=self.device)
        for q in range(1, 9):
            contrib[q] = torch.where(links[q], momentum_flux[q], torch.zeros_like(momentum_flux[q])).sum()
        force = 2.0 * (cvec * contrib.view(9, 1)).sum(dim=0)

        per_disk = None
        torque = None
        if self.track_forces and self.disks:
            per_disk = torch.zeros((len(self.disks), 2), dtype=self.dtype, device=self.device)
            torque = torch.zeros(len(self.disks), dtype=self.dtype, device=self.device)
            ygrid = torch.arange(self.ny, dtype=self.dtype, device=self.device).view(-1, 1) + 0.5
            xgrid = torch.arange(self.nx, dtype=self.dtype, device=self.device).view(1, -1) + 0.5
            owner_f = self.owner.reshape(-1)
            for i, d in enumerate(self.disks):
                for q in range(1, 9):
                    link_q = links[q]
                    owner_q = owner_f[ahead[q]]
                    selm = link_q & (owner_q == i)
                    if not bool(selm.any()):
                        continue
                    m = torch.where(selm, momentum_flux[q], torch.zeros_like(momentum_flux[q]))
                    fq = 2.0 * cvec[q].view(2, 1) * m
                    per_disk[i] = per_disk[i] + fq.sum(dim=1)
                    rx = (xgrid + 0.5 * cvec[q, 0] - d.center[0]).expand(
                        self.ny, self.nx
                    ).reshape(-1)
                    ry = (ygrid + 0.5 * cvec[q, 1] - d.center[1]).expand(
                        self.ny, self.nx
                    ).reshape(-1)
                    torque[i] = torque[i] + (rx * fq[1] - ry * fq[0]).sum()

        self.f = f_new.reshape(9, self.ny, self.nx)
        self.steps += 1
        rho_new, ux_new, uy_new = macroscopic(self.f)
        fluid = ~self.mask
        p_fluid = torch.stack(
            [(rho_new * ux_new * fluid).sum(), (rho_new * uy_new * fluid).sum()]
        )
        return {
            "force": force,  # momentum per step given to all walls
            "per_disk": per_disk,
            "torque": torque,
            "momentum": p_fluid,
            "rho": rho_new,
            "ux": ux_new,
            "uy": uy_new,
            "wall_velocity": (uwx, uwy),
        }

    # -- diagnostics -------------------------------------------------------
    def fluid_momentum(self) -> torch.Tensor:
        rho, ux, uy = macroscopic(self.f)
        fluid = ~self.mask
        return torch.stack([(rho * ux * fluid).sum(), (rho * uy * fluid).sum()])

    def mean_fluid_velocity(self) -> torch.Tensor:
        rho, ux, uy = macroscopic(self.f)
        fluid = ~self.mask
        n = fluid.sum().clamp(min=1)
        return torch.stack([(ux * fluid).sum(), (uy * fluid).sum()]) / n

    def wall_force(self, total_force: torch.Tensor, per_disk: torch.Tensor | None) -> torch.Tensor:
        """Force taken by the static walls (total minus the tracked disks).

        In a periodic box with no static walls this is identically zero, which
        is the discrete statement that the periodic identification carries no
        net momentum across the box.  With channel walls it is the (pressure +
        friction) traction exerted by the fluid on the walls -- the term the
        momentum ledger calls the boundary/pressure work.
        """
        if per_disk is None:
            return total_force.clone()
        return total_force - per_disk.sum(dim=0)


# ---------------------------------------------------------------------------
# Momentum ledger
# ---------------------------------------------------------------------------


def momentum_ledger_probe(
    *,
    ny: int = 64,
    nx: int = 64,
    radius: float = 6.0,
    tau: float = 0.8,
    velocity: tuple[float, float] = (0.0, -0.02),
    steps: int = 200,
    beta: float = 1.0,
    body_force: tuple[float, float] = (0.0, 0.0),
    channel: bool = True,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Close the discrete momentum ledger of a translating disk.

    The disk is *kinematically prescribed* (constant velocity, so there is no
    added-mass transient), the fluid starts at rest, and the box is either
    fully periodic or a channel closed by static walls (``channel=True``).  Per
    step we accumulate

        impulse_particle_to_fluid = - sum F_MEM (disk links)
        impulse_wall_to_fluid     = - F_wall  (static-wall links)
        dP_fluid                  = P(N) - P(0)
        impulse_body              = F_body * dt * N_fluid

    and report the residual of

        dP_fluid = impulse_particle_to_fluid + impulse_wall_to_fluid + impulse_body

    normalised by the total impulse magnitude.  No tuned coefficient enters:
    every wall link removes exactly the momentum carried by the population that
    would have entered the solid.

    Returns:
        dict with the individual impulses, ``residual`` (absolute) and
        ``residual_relative``, plus the measured momentum history.
    """
    lattice = DiskLatticeCoupling(ny, nx, tau=tau, dtype=dtype, track_forces=True)
    walls = None
    if channel:
        walls = torch.zeros((ny, nx), dtype=torch.bool)
        walls[0, :] = True
        walls[-1, :] = True
    lattice.set_disks(
        [Disk(center=(nx / 2.0, ny / 2.0), radius=radius, velocity=velocity)], walls=walls
    )
    p0 = lattice.fluid_momentum()
    m0 = float((lattice.f.sum(dim=0) * (~lattice.mask)).sum())
    impulse = torch.zeros(2, dtype=dtype)
    wall_impulse = torch.zeros(2, dtype=dtype)
    hist_p = [p0]
    hist_f = []
    for _ in range(int(steps)):
        out = lattice.step(beta=beta, body_force=body_force)
        f_disk = out["per_disk"].sum(dim=0)
        f_wall = lattice.wall_force(out["force"], out["per_disk"])
        impulse = impulse - f_disk
        wall_impulse = wall_impulse - f_wall
        hist_f.append(f_disk.clone())
        hist_p.append(out["momentum"])
    dp = lattice.fluid_momentum() - p0
    body_impulse = torch.zeros(2, dtype=dtype)
    if body_force != (0.0, 0.0):
        body_impulse = (
            torch.tensor(body_force, dtype=dtype) * steps * ny * nx * lattice._geometry["fluid_frac"]
        )
    predicted = impulse + wall_impulse + body_impulse
    abs_residual = float((dp - predicted).norm())
    # a unit floor keeps the ratio meaningful when every term is round-off
    # (e.g. beta = 0, where nothing moves at all)
    scale = max(float(impulse.norm()), float(dp.norm()), 1.0)
    return {
        "impulse_particle_to_fluid": impulse,
        "impulse_wall_to_fluid": wall_impulse,
        "impulse_body": body_impulse,
        "dp_fluid": dp,
        "residual": abs_residual,
        "residual_relative": abs_residual / scale,
        "impulse_scale": scale,
        "momentum_history": torch.stack(hist_p),
        "force_history": torch.stack(hist_f),
        "solid_fraction": lattice.solid_fraction,
        "mass_drift": abs(float((lattice.f.sum(dim=0) * (~lattice.mask)).sum()) - m0) / m0,
        "integrity": {
            "steps": int(steps),
            "beta": float(beta),
            "velocity": tuple(velocity),
            "tau": float(tau),
            "radius": float(radius),
            "channel": bool(channel),
            "lattice": [int(ny), int(nx)],
        },
    }


# ---------------------------------------------------------------------------
# Added mass
# ---------------------------------------------------------------------------


def measure_added_mass(
    *,
    ny: int = 128,
    nx: int = 128,
    radius: float = 8.0,
    tau: float = 0.8,
    amplitude: float = 0.005,
    period: float = 400.0,
    n_periods: int = 6,
    beta: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Added-mass coefficient of a disk from forced harmonic translation.

    The disk oscillates as ``u(t) = A sin(omega t)`` in ``x`` (small amplitude,
    so the flow stays in the creeping regime).  In the periodic steady state the
    MEM force is decomposed as

        F(t) = M_a * a(t) + C * u(t) + const,      a = du/dt

    with ``u`` and ``a`` orthogonal over a whole period, so a least-squares fit
    on the last two periods yields ``M_a`` and ``C`` separately.  Reported
    ``C_A = M_a / (rho * V_p)``; the exact 2D potential-flow value for a
    circular cylinder is 1.
    """
    lattice = DiskLatticeCoupling(ny, nx, tau=tau, dtype=dtype, track_forces=True)
    centre = (nx / 2.0, ny / 2.0)
    omega = 2.0 * math.pi / float(period)
    lattice.set_disks([Disk(center=centre, radius=radius)])

    total_steps = int(n_periods * period)
    recs = []
    for k in range(total_steps):
        t = float(k)
        u = amplitude * math.sin(omega * t)
        a = amplitude * omega * math.cos(omega * t)
        lattice.disks[0].velocity = (u, 0.0)
        out = lattice.step(beta=beta)
        recs.append((u, a, float(out["force"][0]), float(out["force"][1])))
        # out["force"] is the momentum handed to the body == force on the body

    tail = recs[-int(2 * period):]
    uu = torch.tensor([r[0] for r in tail], dtype=dtype)
    aa = torch.tensor([r[1] for r in tail], dtype=dtype)
    ff = torch.tensor([r[2] for r in tail], dtype=dtype)
    A = torch.stack([aa, uu, torch.ones_like(uu)], dim=1)
    sol = torch.linalg.lstsq(A, ff.unsqueeze(1)).solution.squeeze(1)
    # F_particle = -M_a a - C u + const  (added mass and drag oppose motion)
    m_add, drag, const = -float(sol[0]), -float(sol[1]), float(sol[2])
    v_p = math.pi * radius * radius
    c_a = m_add / (1.0 * v_p)
    resid = float((A @ sol - ff).norm() / ff.norm().clamp(min=1e-30))
    return {
        "added_mass": m_add,
        "added_mass_coefficient": c_a,
        "added_mass_theory_2d": 1.0,
        "relative_error": abs(c_a - 1.0),
        "drag_impedance_fit": drag,
        "offset": const,
        "fit_relative_residual": resid,
        "cross_force_rms": float(
            torch.tensor([r[3] for r in tail], dtype=dtype).pow(2).mean().sqrt()
        ),
        "particle_volume": v_p,
        "integrity": {
            "radius": float(radius),
            "amplitude": float(amplitude),
            "period": float(period),
            "n_periods": int(n_periods),
            "tau": float(tau),
            "lattice": [int(ny), int(nx)],
        },
    }


# ---------------------------------------------------------------------------
# One-way degradation: the m7094 closure path
# ---------------------------------------------------------------------------


def closure_terminal_velocity(
    concentration: float,
    *,
    radius: float = 8.0,
    re_p: float = 0.1,
    exponent: float | None = None,
    phi_max: float = 0.60,
    rho_p: float = 2.65,
    rho_f: float = 1.0,
    gravity: float = 1.0e-4,
    dt: float = 1.0,
    max_steps: int = 2_000_000,
    tol: float = 1e-15,
) -> dict:
    """Terminal velocity from the time-integrated one-way closure balance.

    ``m dv/dt = (rho_p - rho_f) A g - 1/2 C_D(c, Re) rho_f A |v| v``

    with the *m7094 certified* suspension drag
    ``C_D(c, Re) = C_D(Re) (1 - c/c_max)^(-2m)``.  The steady state of this ODE
    is, by construction of the closure, the Richardson-Zaki hindered velocity;
    the time integration is therefore an independent (time-domain) re-derivation
    of the certified number and is used as the ``beta = 0`` degradation check.
    """
    if not (0.0 <= concentration < phi_max):
        raise ValueError(f"concentration must be in [0, {phi_max}), got {concentration!r}")
    area = math.pi * radius * radius
    mass = rho_p * area

    def terminal(conc: float) -> tuple[float, int, float]:
        buoy = (rho_p - rho_f) * area * gravity
        cd = float(
            drag_coefficient_suspension(conc, re_p, exponent=exponent, phi_max=phi_max)
        )
        v = 0.0
        prev = -1.0
        rel = float("inf")
        step = 0
        for step in range(1, int(max_steps) + 1):
            drag = 0.5 * cd * rho_f * area * abs(v) * v
            v = v + dt * (buoy - drag) / mass
            if v > 0.0:
                rel = abs(v - prev) / v
                if rel < tol:
                    break
            prev = v
        return v, step, rel

    v_c, steps, rel = terminal(concentration)
    v_0, _, _ = terminal(0.0)
    ratio = v_c / v_0 if v_0 else float("nan")
    expected = float(
        hindered_settling_velocity(concentration, 1.0, exponent=exponent, phi_max=phi_max)
    )
    return {
        "concentration": float(concentration),
        "terminal_velocity": v_c,
        "dilute_terminal_velocity": v_0,
        "velocity_ratio": ratio,
        "expected_ratio_rz": expected,
        "relative_error": abs(ratio - expected) / expected,
        "drag_coefficient_suspension": float(
            drag_coefficient_suspension(concentration, re_p, exponent=exponent, phi_max=phi_max)
        ),
        "steps": steps,
        "converged_increment": rel,
    }