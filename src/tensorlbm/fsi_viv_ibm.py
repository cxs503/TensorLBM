"""Two-dimensional vortex-induced vibration (VIV) of an elastically mounted
cylinder: D2Q9 lattice Boltzmann fluid + direct-forcing immersed boundary
method + partitioned strong coupling with interface accelerators.

This module provides the *flow-side* validation of the partitioned coupling
developed in :mod:`tensorlbm.fsi_partitioned`: the interface accelerators
(:class:`~tensorlbm.fsi_partitioned.IQNILS`,
:class:`~tensorlbm.fsi_partitioned.AitkenRelaxation`) are used to solve the
fluid-structure sub-iterations of a genuine moving-boundary LBM, so that the
lock-in response can be compared with the classical VIV experiments.

Numerics
--------
* Fluid: D2Q9 BGK, Guo forcing, ``tau = 3 nu + 0.5``.
* Body: direct-forcing IBM (Uhlmann 2005) with the Peskin 4-point kernel on
  Lagrangian markers covering the cylinder surface and interior, so that the
  body motion is rigid to first order.
* Boundaries: equilibrium velocity inlet, zero-gradient outlet, periodic in
  the transverse direction.
* Structure: 1-DOF transverse spring-mass-damper, implicit Newmark
  (``gamma = 1/2``, ``beta = 1/4``), mass ratio ``m* = m_s / (rho D^2)``.
* Coupling: partitioned sub-iterations per time step on the interface
  displacement, with selectable accelerator (``none`` / ``aitken`` /
  ``iqn_ils``).

.. warning::
   **VALIDATION STATUS: WITHHELD.**  The direct-forcing IBM implemented here
   inflates the hydrodynamic load on a coarse grid (measured ``Cd = 32.3``
   against the reference ``1.34`` at ``Re = 100``, ``D = 16``) and diverges at
   the relaxation time required by ``Re = 100`` (``tau = 0.53``) once the
   wake develops, see ``artifacts/fsi_partitioned/viv_validation_status.json``.
   The module is kept as a research probe for the partitioned coupling; do not
   use it for quantitative loads.  See :mod:`tensorlbm.fsi_viv_bb` for the
   bounce-back variant (also unvalidated, momentum-exchange estimator shows
   checkerboard aliasing).

Reference data used for the comparison (see ``docs/`` report):
* Fixed cylinder at Re = 100: ``Cd = 1.34``, ``St = 0.164``
  (ROSCOE/DFG benchmark, Tritton-type drag measurements).
* VIV at Re = 100, ``m* = 10``, ``zeta = 0``: maximum response
  ``A/D ~ 0.5`` and lock-in range ``U* ~ 3.5-7``
  (Leontini, Thompson & Hourigan 2006; Placzek et al. 2009).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .d2q9 import C, W, equilibrium, macroscopic
from .fsi_partitioned import AitkenRelaxation, IQNILS, NoRelaxation

VALIDATION_STATUS = "WITHHELD"  # immersed-boundary direct forcing, load inflated

__all__ = [
    "VIVConfig",
    "VIVResult",
    "run_fixed_cylinder",
    "run_viv_case",
    "lbm_step_ibm",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VIVConfig:
    """Lattice-Boltzmann VIV configuration (all quantities in lattice units)."""

    diameter: int = 20
    length_x: float = 30.0          # domain length in diameters
    height_y: float = 16.0          # domain height in diameters
    inflow: float = 0.05            # lattice inflow velocity
    reynolds: float = 100.0
    mass_ratio: float = 10.0        # m_s / (rho D^2)
    zeta: float = 0.0               # structural damping ratio
    reduced_velocity: float = 5.0   # U* = U / (f_n D)
    n_steps: int = 60000
    warmup: int = 20000             # steps discarded before statistics
    max_sub_iter: int = 3
    sub_iter_tol: float = 1e-6
    coupling: str = "iqn_ils"
    marker_spacing: float = 0.5
    interior_spacing: float = 0.8
    dtype: str = "float32"
    seed: int = 0

    # -- derived quantities ------------------------------------------------
    @property
    def nx(self) -> int:
        return int(round(self.length_x * self.diameter))

    @property
    def ny(self) -> int:
        return int(round(self.height_y * self.diameter))

    @property
    def nu(self) -> float:
        return self.inflow * self.diameter / self.reynolds

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def natural_frequency(self) -> float:
        """Structural natural frequency ``f_n = U / (U* D)`` (lattice units)."""
        return self.inflow / (self.reduced_velocity * self.diameter)

    @property
    def omega_n(self) -> float:
        return 2.0 * math.pi * self.natural_frequency

    @property
    def mass(self) -> float:
        """Structural mass per unit depth (rho = 1)."""
        return self.mass_ratio * self.diameter * self.diameter

    @property
    def stiffness(self) -> float:
        return self.mass * self.omega_n ** 2

    @property
    def damping(self) -> float:
        return 2.0 * self.zeta * math.sqrt(self.mass * self.stiffness)

    @property
    def x_center(self) -> float:
        return 0.35 * self.nx

    @property
    def y_center(self) -> float:
        return 0.5 * self.ny

    def __post_init__(self) -> None:
        if self.tau <= 0.5:
            raise ValueError("tau must exceed 0.5 (positive viscosity)")
        if self.mass_ratio <= 0.0:
            raise ValueError("mass_ratio must be positive")
        if self.max_sub_iter < 1:
            raise ValueError("max_sub_iter must be >= 1")


@dataclass
class VIVResult:
    config: VIVConfig
    time: np.ndarray
    displacement: np.ndarray
    lift: np.ndarray
    drag: np.ndarray
    amplitude_ratio: float
    strouhal: float
    mean_drag_coefficient: float
    rms_lift_coefficient: float
    iterations: np.ndarray = field(default_factory=lambda: np.zeros(0))
    converged: bool = True
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Immersed boundary machinery
# ---------------------------------------------------------------------------


def _delta4(r: np.ndarray) -> np.ndarray:
    """Peskin 4-point kernel (vectorised)."""
    a = np.abs(r)
    out = np.zeros_like(a)
    m1 = a < 1.0
    out[m1] = (3.0 - 2.0 * a[m1] + np.sqrt(1.0 + 4.0 * a[m1] - 4.0 * a[m1] ** 2)) / 8.0
    m2 = (a >= 1.0) & (a < 2.0)
    out[m2] = (5.0 - 2.0 * a[m2] - np.sqrt(-7.0 + 12.0 * a[m2] - 4.0 * a[m2] ** 2)) / 8.0
    return out


_OFFSETS = np.array([[i, j] for i in (-1, 0, 1, 2) for j in (-1, 0, 1, 2)], dtype=float)


def lagrangian_markers(
    xc: float, yc: float, radius: float, *, spacing: float = 0.5, interior: float = 0.8
) -> np.ndarray:
    """Return Lagrangian marker positions on and inside a circle.

    The surface ring is sampled at ``spacing`` (in lattice units) and the
    interior is filled on a square lattice of pitch ``interior`` so that the
    direct-forcing IBM drives the whole body as a rigid region.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    n_ring = max(12, int(round(2.0 * math.pi * radius / spacing)))
    theta = np.linspace(0.0, 2.0 * math.pi, n_ring, endpoint=False)
    # slightly inside the geometric radius so that the 4-point kernel covers
    # the boundary nodes symmetrically
    r_ring = radius - 0.25
    xs = xc + r_ring * np.cos(theta)
    ys = yc + r_ring * np.sin(theta)
    ring = np.stack([xs, ys], axis=1)

    if interior > 0.0:
        coords = np.arange(-radius + 0.5, radius - 0.5 + 1e-9, interior)
        gx, gy = np.meshgrid(coords, coords)
        gx = gx.ravel()
        gy = gy.ravel()
        keep = (gx ** 2 + gy ** 2) < (radius - 0.75) ** 2
        inner = np.stack([xc + gx[keep], yc + gy[keep]], axis=1)
        return np.concatenate([ring, inner], axis=0)
    return ring


class _IBMOps:
    """Pre-computed gather/scatter maps for the 4-point IBM kernel."""

    def __init__(self, nx: int, ny: int) -> None:
        self.nx = nx
        self.ny = ny
        self.n_off = _OFFSETS.shape[0]

    def weights(
        self, markers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(flat_index, weight)`` arrays of shape ``(M, 16)``.

        Transverse (y) periodicity is folded with a modulo, while markers must
        stay away from the x boundaries (enforced by the caller).
        """
        base_x = np.floor(markers[:, 0:1] + _OFFSETS[None, :, 0])  # noqa: N806
        base_y = np.floor(markers[:, 1:2] + _OFFSETS[None, :, 1])  # noqa: N806
        wx = _delta4(markers[:, 0:1] - base_x)
        wy = _delta4(markers[:, 1:2] - base_y)
        w = wx * wy
        ix = base_x.astype(np.int64) % self.nx
        iy = base_y.astype(np.int64) % self.ny
        flat = iy * self.nx + ix
        return flat, w


# ---------------------------------------------------------------------------
# One LBM step with a moving immersed body
# ---------------------------------------------------------------------------


def _macroscopic_forced(
    f: torch.Tensor, fx: torch.Tensor | None, fy: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Guo-corrected macroscopic variables ``u = (sum c f + F/2) / rho``."""
    rho, ux, uy = macroscopic(f)
    if fx is not None:
        rho_safe = torch.clamp(rho, min=1e-12)
        ux = ux + 0.5 * fx / rho_safe
        uy = uy + 0.5 * fy / rho_safe
    return rho, ux, uy


def lbm_step_ibm(
    f: torch.Tensor,
    cfg: VIVConfig,
    markers: np.ndarray,
    body_velocity: tuple[float, float],
    ops: _IBMOps,
    c_dev: torch.Tensor,
    w_dev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the D2Q9 field one step with a rigid immersed body.

    Returns ``(f_new, body_force_x, body_force_y)`` where the forces are the
    hydrodynamic loads acting **on the body** in lattice units.
    """
    nx, ny = cfg.nx, cfg.ny
    rho, ux, uy = _macroscopic_forced(f, None, None)

    # --- direct forcing at the Lagrangian markers ------------------------
    flat, w = ops.weights(markers)
    idx = torch.as_tensor(flat.reshape(-1), dtype=torch.long)
    wt = torch.as_tensor(w.reshape(-1), dtype=f.dtype)
    ux_flat = ux.reshape(-1)
    uy_flat = uy.reshape(-1)
    n_mark = markers.shape[0]
    w_sum = wt.view(n_mark, -1).sum(dim=1).clamp(min=1e-12)
    ux_m = (wt * ux_flat[idx]).view(n_mark, -1).sum(dim=1) / w_sum
    uy_m = (wt * uy_flat[idx]).view(n_mark, -1).sum(dim=1) / w_sum
    fx_m = (body_velocity[0] - ux_m)  # dt = 1
    fy_m = (body_velocity[1] - uy_m)
    fx_grid = torch.zeros(nx * ny, dtype=f.dtype).index_add_(
        0, idx, (wt * fx_m.repeat_interleave(ops.n_off)).view(-1)
    )
    fy_grid = torch.zeros(nx * ny, dtype=f.dtype).index_add_(
        0, idx, (wt * fy_m.repeat_interleave(ops.n_off)).view(-1)
    )
    fx_grid = fx_grid.view(ny, nx)
    fy_grid = fy_grid.view(ny, nx)

    # momentum source applied to the fluid -> reaction on the body
    body_fx = float(-fx_grid.sum())
    body_fy = float(-fy_grid.sum())

    # --- Guo forcing + BGK collision + streaming -------------------------
    guo = 1.0 - 1.0 / (2.0 * cfg.tau)
    cu = (
        c_dev[:, 0].view(9, 1, 1) * ux.unsqueeze(0)
        + c_dev[:, 1].view(9, 1, 1) * uy.unsqueeze(0)
    )
    feq = w_dev.view(9, 1, 1) * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux * ux + uy * uy).unsqueeze(0))
    force_term = w_dev.view(9, 1, 1) * 3.0 * guo * (
        c_dev[:, 0].view(9, 1, 1) * fx_grid.unsqueeze(0)
        + c_dev[:, 1].view(9, 1, 1) * fy_grid.unsqueeze(0)
    )
    f_post = f - (f - feq) / cfg.tau + force_term

    f_new = torch.empty_like(f_post)
    for i in range(9):
        f_new[i] = torch.roll(f_post[i], shifts=(int(C[i, 1]), int(C[i, 0])), dims=(0, 1))

    # --- boundary conditions --------------------------------------------
    f_new[:, :, 0:1] = equilibrium(
        torch.ones(ny, 1, dtype=f.dtype),
        torch.full((ny, 1), cfg.inflow, dtype=f.dtype),
        torch.zeros(ny, 1, dtype=f.dtype),
    )
    f_new[:, :, -1] = f_new[:, :, -2]
    return f_new, torch.tensor(body_fx, dtype=f.dtype), torch.tensor(body_fy, dtype=f.dtype)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def _tensors(cfg: VIVConfig):
    dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
    c_dev = C.to(dtype)
    w_dev = W.to(dtype)
    f = equilibrium(
        torch.ones(cfg.ny, cfg.nx, dtype=dtype),
        torch.full((cfg.ny, cfg.nx), cfg.inflow, dtype=dtype),
        torch.zeros(cfg.ny, cfg.nx, dtype=dtype),
    )
    return f, c_dev, w_dev, _IBMOps(cfg.nx, cfg.ny)


def _make_accelerator(name: str):
    table = {"none": NoRelaxation, "aitken": AitkenRelaxation, "iqn_ils": IQNILS}
    if name not in table:
        raise ValueError(f"unknown coupling {name!r}")
    return table[name]()


def run_fixed_cylinder(cfg: VIVConfig, *, n_steps: int = 20000, warmup: int = 8000) -> dict:
    """Fixed-cylinder drag/Strouhal validation (no structural motion)."""
    torch.set_grad_enabled(False)
    f, c_dev, w_dev, ops = _tensors(cfg)
    markers = lagrangian_markers(
        cfg.x_center, cfg.y_center, 0.5 * cfg.diameter,
        spacing=cfg.marker_spacing, interior=cfg.interior_spacing,
    )
    drag, lift = [], []
    for _ in range(n_steps):
        f, fx, fy = lbm_step_ibm(f, cfg, markers, (0.0, 0.0), ops, c_dev, w_dev)
        drag.append(float(fx))
        lift.append(float(fy))
    drag = np.asarray(drag)
    lift = np.asarray(lift)
    tail = slice(warmup, None)
    q = 0.5 * cfg.inflow ** 2 * cfg.diameter
    cd = float(np.mean(drag[tail]) / q)
    cl_rms = float(np.sqrt(np.mean((lift[tail] - lift[tail].mean()) ** 2)) / q)
    st = _strouhal(lift[tail], cfg.inflow, cfg.diameter)
    return {"cd": cd, "cl_rms": cl_rms, "strouhal": st, "drag": drag, "lift": lift}


def _strouhal(signal: np.ndarray, inflow: float = 0.05, diameter: int = 20) -> float:
    """Dominant non-dimensional frequency from the FFT of the lift signal."""
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    if x.size < 64:
        return float("nan")
    window = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * window))
    freqs = np.fft.rfftfreq(x.size, d=1.0)
    if spec.size < 3:
        return float("nan")
    k = int(np.argmax(spec[1:]) + 1)
    f_shed = freqs[k]
    return float(f_shed * diameter / inflow)


def _amplitude_and_frequency(signal: np.ndarray, inflow: float, diameter: int) -> tuple[float, float]:
    """Amplitude (half peak-to-peak) and dominant non-dimensional frequency."""
    x = np.asarray(signal, dtype=float)
    if x.size < 16:
        return float("nan"), float("nan")
    spec = np.abs(np.fft.rfft(x - x.mean()))
    freqs = np.fft.rfftfreq(x.size, d=1.0)
    k = int(np.argmax(spec[1:]) + 1)
    return 0.5 * float(x.max() - x.min()), float(freqs[k] * diameter / inflow)


def run_viv_case(cfg: VIVConfig) -> VIVResult:
    """Run one elastically mounted cylinder case with partitioned coupling.

    Each time step re-solves the fluid from the state at ``t^n`` for every
    interface iterate (Dirichlet-Neumann fixed point) and accepts the iterate
    whose interface residual falls below ``sub_iter_tol``; the accepted fluid
    state is advanced.  This keeps the time accuracy of the strong coupling
    while the cost stays at ``n_iter <= max_sub_iter`` LBM steps per step.
    """
    torch.set_grad_enabled(False)
    f, c_dev, w_dev, ops = _tensors(cfg)

    mass, stiffness, damping = cfg.mass, cfg.stiffness, cfg.damping
    y, v, a = 0.0, 0.0, 0.0
    dt = 1.0
    gamma, beta_n = 0.5, 0.25
    m_eff = mass + damping * gamma * dt + stiffness * beta_n * dt * dt

    times, disps, lifts, drags, iters = [], [], [], [], []
    converged = True
    acc = _make_accelerator(cfg.coupling)

    for step in range(cfg.n_steps):
        c_known = -(
            damping * (v + dt * (1.0 - gamma) * a)
            + stiffness * (y + dt * v + dt * dt * (0.5 - beta_n) * a)
        )
        if hasattr(acc, "reset"):
            acc.reset()
        x = np.array([y], dtype=float)
        f_state = f
        n_iter = 0
        y_accepted = y
        load_y = 0.0
        load_x = 0.0
        for k in range(cfg.max_sub_iter):
            n_iter = k + 1
            markers = lagrangian_markers(
                cfg.x_center, cfg.y_center + float(x[0]), 0.5 * cfg.diameter,
                spacing=cfg.marker_spacing, interior=cfg.interior_spacing,
            )
            f_trial, fx, fy = lbm_step_ibm(f_state, cfg, markers, (0.0, 0.0), ops, c_dev, w_dev)
            load_y, load_x = float(fy), float(fx)
            y_new = (load_y + c_known) / m_eff
            r = np.array([y_new - float(x[0])], dtype=float)
            if abs(float(r[0])) < cfg.sub_iter_tol:
                y_accepted = y_new
                f = f_trial
                break
            x_new, _ = acc.step(x, r)
            if not np.isfinite(float(x_new[0])) or abs(float(x_new[0])) > 2.0 * cfg.diameter:
                converged = False
                x_new = np.array([min(max(float(x_new[0]), -cfg.diameter), cfg.diameter)])
            x = x_new
            y_accepted = float(x[0])
            f = f_trial
        else:
            # sub-iteration budget exhausted: accept the last iterate (the
            # fluid state f already corresponds to it)
            pass

        # Newmark kinematics consistent with the interface fixed point
        a_next = (y_accepted - y - dt * v - dt * dt * (0.5 - beta_n) * a) / (beta_n * dt * dt)
        v_next = v + dt * ((1.0 - gamma) * a + gamma * a_next)

        times.append(step + 1.0)
        disps.append(y_accepted)
        lifts.append(load_y)
        drags.append(load_x)
        iters.append(n_iter)
        y, v, a = y_accepted, v_next, a_next
        if not np.isfinite(y) or abs(y) > 5.0 * cfg.diameter:
            converged = False
            break

    disps = np.asarray(disps)
    lifts = np.asarray(lifts)
    drags = np.asarray(drags)
    start = min(cfg.warmup, max(len(disps) - 200, 0))
    tail = slice(start, None)
    amp, f_resp = _amplitude_and_frequency(disps[tail], cfg.inflow, cfg.diameter)
    q = 0.5 * cfg.inflow ** 2 * cfg.diameter
    return VIVResult(
        config=cfg,
        time=np.asarray(times),
        displacement=disps,
        lift=lifts,
        drag=drags,
        amplitude_ratio=amp / cfg.diameter,
        strouhal=_strouhal(lifts[tail], cfg.inflow, cfg.diameter),
        mean_drag_coefficient=float(drags[tail].mean() / q),
        rms_lift_coefficient=float(
            np.sqrt(np.mean((lifts[tail] - lifts[tail].mean()) ** 2)) / q
        ),
        iterations=np.asarray(iters),
        converged=converged,
        extra={"response_frequency": f_resp, "f_n": cfg.natural_frequency},
    )
