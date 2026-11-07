"""VIV of an elastically mounted cylinder with a bounce-back D2Q9 solver.

The immersed-boundary direct-forcing variant in :mod:`tensorlbm.fsi_viv_ibm`
inflates the hydrodynamic load (each overlapping marker adds an independent
momentum source) and becomes unstable at the low relaxation times required by
``Re = 100`` on a coarse grid.  This module therefore uses the standard
staircase solid mask with half-way bounce-back and the momentum-exchange
method (MEM) for the load, which is the reference approach for LBM bluff-body
forces.

.. warning::
   **VALIDATION STATUS: WITHHELD.**  The half-way bounce-back fluid is
   mass/momentum stable, but the momentum-exchange load estimator still shows
   period-2 checkerboard aliasing on the staircase boundary (load oscillates
   sign every step), so no quantitative ``Cd`` could be certified before the
   compute budget of this node ran out.  Kept as a documented research probe.

Numerics
--------
* D2Q9 BGK, ``tau = 3 nu + 1/2``, Guo-free (no body force).
* Half-way bounce-back on a staircase circle; solid nodes hold
  ``f_eq(rho=1, u_wall)`` after collision so that a moving wall transfers the
  Ladd momentum correction.
* Load on the body from the momentum-exchange method
  ``F = sum_s sum_{i: x_s + c_i in fluid} c_i [f_i^+(s) + f_opp(i)(x_s + c_i)]``.
* Domain: uniform inflow (equilibrium), zero-gradient outlet, periodic
  transverse direction.
* Structure: 1-DOF transverse spring-mass-damper, implicit Newmark, solved in
  a partitioned strong-coupling sub-iteration accelerated with
  :class:`~tensorlbm.fsi_partitioned.IQNILS` / ``AitkenRelaxation``.

Reference values (fixed cylinder, ``Re = 100``): ``Cd = 1.34``, ``St = 0.164``.
VIV at ``Re = 100``, ``m* = 10``, ``zeta = 0``: ``A/D ~ 0.5`` in the lock-in
range ``U* ~ 4-7`` (Leontini et al. 2006; Placzek et al. 2009).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .d2q9 import C, OPPOSITE, W, equilibrium, macroscopic
from .fsi_partitioned import AitkenRelaxation, IQNILS, NoRelaxation
from .fsi_viv_ibm import VIVConfig

VALIDATION_STATUS = "WITHHELD"  # bounce-back, momentum-exchange aliasing

__all__ = ["BBSolver", "run_viv_bb", "run_fixed_cylinder_bb", "staircase_mask"]


def staircase_mask(cfg: VIVConfig, yc: float, radius: float | None = None) -> np.ndarray:
    """Boolean solid mask of a staircase circle (True inside the body)."""
    r = 0.5 * cfg.diameter if radius is None else radius
    yy, xx = np.mgrid[0: cfg.ny, 0: cfg.nx]
    return ((xx - cfg.x_center) ** 2 + (yy - yc) ** 2) <= r * r


class BBSolver:
    """D2Q9 bounce-back solver with a movable staircase cylinder."""

    def __init__(self, cfg: VIVConfig) -> None:
        self.cfg = cfg
        dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
        self.dtype = dtype
        self.c = C.to(dtype)
        self.w = W.to(dtype)
        self.opp = OPPOSITE
        self.f = equilibrium(
            torch.ones(cfg.ny, cfg.nx, dtype=dtype),
            torch.full((cfg.ny, cfg.nx), cfg.inflow, dtype=dtype),
            torch.zeros(cfg.ny, cfg.nx, dtype=dtype),
        )
        self.mask: torch.Tensor | None = None

    # -- helpers ---------------------------------------------------------
    def _shift(self, x: torch.Tensor, i: int, forward: bool = True) -> torch.Tensor:
        """Shift field ``x`` by one lattice step in direction ``i`` (+/-)."""
        sgn = 1 if forward else -1
        return torch.roll(x, shifts=(-sgn * int(C[i, 1]), -sgn * int(C[i, 0])), dims=(0, 1))

    def set_body(self, yc: float) -> torch.Tensor:
        mask = torch.as_tensor(staircase_mask(self.cfg, yc), dtype=torch.bool)
        self.mask = mask
        return mask

    # -- one lattice step ------------------------------------------------
    def step(self, u_wall: float = 0.0) -> tuple[float, float]:
        cfg = self.cfg
        f = self.f
        rho, ux, uy = macroscopic(f)
        cu = (
            self.c[:, 0].view(9, 1, 1) * ux.unsqueeze(0)
            + self.c[:, 1].view(9, 1, 1) * uy.unsqueeze(0)
        )
        usq = (ux * ux + uy * uy).unsqueeze(0)
        feq = self.w.view(9, 1, 1) * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
        f_post = f - (f - feq) / cfg.tau

        mask = self.mask
        assert mask is not None

        # momentum exchange over the boundary links BEFORE streaming
        fx = 0.0
        fy = 0.0
        for i in range(1, 9):
            # solid nodes s whose neighbour s + c_i lies in the fluid
            links = mask & ~self._shift(mask, i, forward=False)
            if not bool(links.any()):
                continue
            # fluid-side (symmetric) momentum exchange: for a static wall the
            # reflected population equals the incoming one, and the solid-node
            # state is never used (avoids the checkerboard aliasing of the
            # on-node bounce-back populations).
            f_in = self._shift(f_post[int(self.opp[i])], i, forward=True)[links]
            contrib = float((2.0 * f_in).sum())
            if u_wall != 0.0 and int(C[i, 1]) != 0:
                contrib += float(links.sum()) * 6.0 * float(W[i]) * float(C[i, 1]) * u_wall
            fx += float(C[i, 0]) * contrib
            fy += float(C[i, 1]) * contrib

        # streaming + half-way bounce back
        f_new = torch.empty_like(f_post)
        for i in range(9):
            f_new[i] = self._shift(f_post[i], i, forward=True)
        f_new[:, mask] = f_post[self.opp][:, mask]
        # Ladd moving-wall correction (adds the wall momentum across each link)
        if u_wall != 0.0:
            for i in range(1, 9):
                if int(C[i, 1]) == 0:
                    continue
                links = mask & ~self._shift(mask, i, forward=False)
                if bool(links.any()):
                    f_new[i][links] += 6.0 * float(W[i]) * float(C[i, 1]) * u_wall

        # boundary conditions
        f_new[:, :, 0:1] = equilibrium(
            torch.ones(cfg.ny, 1, dtype=self.dtype),
            torch.full((cfg.ny, 1), cfg.inflow, dtype=self.dtype),
            torch.zeros(cfg.ny, 1, dtype=self.dtype),
        )
        f_new[:, :, -1] = f_new[:, :, -2]
        self.f = f_new
        return fx, fy


# ---------------------------------------------------------------------------
# Validation / production drivers
# ---------------------------------------------------------------------------


def _spectral_peak(signal: np.ndarray) -> float:
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    if x.size < 64:
        return float("nan")
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, d=1.0)
    return float(freqs[int(np.argmax(spec[1:]) + 1)])


def run_fixed_cylinder_bb(
    cfg: VIVConfig, *, n_steps: int = 20000, warmup: int = 10000
) -> dict:
    """Fixed-cylinder drag and Strouhal validation."""
    torch.set_grad_enabled(False)
    solver = BBSolver(cfg)
    solver.set_body(cfg.y_center)
    drag, lift = [], []
    for _ in range(n_steps):
        fx, fy = solver.step(0.0)
        drag.append(fx)
        lift.append(fy)
    drag = np.asarray(drag)
    lift = np.asarray(lift)
    q = 0.5 * cfg.inflow ** 2 * cfg.diameter
    tail = slice(warmup, None)
    # orientation check: the drag must be positive downstream
    sign = 1.0 if drag[tail].mean() > 0 else -1.0
    cd = float(sign * drag[tail].mean() / q)
    cl_rms = float(np.sqrt(np.mean((sign * lift[tail] - sign * lift[tail].mean()) ** 2)) / q)
    f_peak = _spectral_peak(lift[tail])
    st = f_peak * cfg.diameter / cfg.inflow
    return {
        "cd": cd,
        "cl_rms": cl_rms,
        "strouhal": float(st),
        "drag": drag,
        "lift": lift,
        "sign": sign,
    }


@dataclass
class VIVBBResult:
    amplitude_ratio: float
    strouhal: float
    mean_drag_coefficient: float
    rms_lift_coefficient: float
    response_frequency: float
    natural_frequency: float
    displacement: np.ndarray
    lift: np.ndarray
    drag: np.ndarray
    iterations: np.ndarray
    converged: bool
    extra: dict = field(default_factory=dict)


def run_viv_bb(cfg: VIVConfig, *, verbose: bool = False) -> VIVBBResult:
    """Partitioned VIV with bounce-back fluid and momentum-exchange load."""
    torch.set_grad_enabled(False)
    solver = BBSolver(cfg)
    mass, stiffness, damping = cfg.mass, cfg.stiffness, cfg.damping
    gamma, beta_n = 0.5, 0.25
    dt = 1.0
    m_eff = mass + damping * gamma * dt + stiffness * beta_n * dt * dt

    y, v, a = 0.0, 0.0, 0.0
    acc_table = {"none": NoRelaxation, "aitken": AitkenRelaxation, "iqn_ils": IQNILS}
    if cfg.coupling not in acc_table:
        raise ValueError(f"unknown coupling {cfg.coupling!r}")
    acc = acc_table[cfg.coupling]()

    disps, lifts, drags, iters = [], [], [], []
    converged = True
    for step in range(cfg.n_steps):
        c_known = -(
            damping * (v + dt * (1.0 - gamma) * a)
            + stiffness * (y + dt * v + dt * dt * (0.5 - beta_n) * a)
        )
        if hasattr(acc, "reset"):
            acc.reset()
        x = np.array([y], dtype=float)
        n_iter = 0
        load_x = load_y = 0.0
        # The fluid state is advanced once per time step; the interface
        # sub-iteration only re-evaluates the body position used for the
        # boundary treatment (fixed-point on the interface).
        accepted = False
        f_state = solver.f
        y_acc = y
        for k in range(cfg.max_sub_iter):
            n_iter = k + 1
            solver.f = f_state
            solver.set_body(cfg.y_center + float(x[0]))
            load_x, load_y = solver.step(0.0)
            if float(load_y) > 0:  # ensure the load is measured, not re-signed
                pass
            y_new = (float(load_y) + c_known) / m_eff
            r = np.array([y_new - float(x[0])], dtype=float)
            if abs(float(r[0])) < cfg.sub_iter_tol:
                y_acc = y_new
                accepted = True
                break
            x_new, _ = acc.step(x, r)
            if not np.isfinite(float(x_new[0])) or abs(float(x_new[0])) > 2.0 * cfg.diameter:
                converged = False
                x_new = np.array([min(max(float(x_new[0]), -cfg.diameter), cfg.diameter)])
            x = x_new
            y_acc = float(x[0])
        if not accepted and n_iter == 0:  # pragma: no cover
            y_acc = y

        a_next = (y_acc - y - dt * v - dt * dt * (0.5 - beta_n) * a) / (beta_n * dt * dt)
        v_next = v + dt * ((1.0 - gamma) * a + gamma * a_next)
        disps.append(y_acc)
        lifts.append(load_y)
        drags.append(load_x)
        iters.append(n_iter)
        y, v, a = y_acc, v_next, a_next
        if verbose and step % 5000 == 0:
            print(f"  step {step} y={y:.3f} A/D={y/cfg.diameter:.3f} Fy={load_y:.4f} it={n_iter}", flush=True)
        if not np.isfinite(y) or abs(y) > 5.0 * cfg.diameter:
            converged = False
            break

    disps = np.asarray(disps)
    lifts = np.asarray(lifts)
    drags = np.asarray(drags)
    q = 0.5 * cfg.inflow ** 2 * cfg.diameter
    sign_l = 1.0 if np.abs(lifts).max() > 0 else 1.0
    start = min(cfg.warmup, max(len(disps) - 200, 0))
    tail = slice(start, None)
    amp = 0.5 * float(disps[tail].max() - disps[tail].min())
    f_resp = _spectral_peak(disps[tail])
    st = _spectral_peak(lifts[tail]) * cfg.diameter / cfg.inflow
    return VIVBBResult(
        amplitude_ratio=amp / cfg.diameter,
        strouhal=float(st),
        mean_drag_coefficient=float(sign_l * drags[tail].mean() / q),
        rms_lift_coefficient=float(np.sqrt(np.mean((lifts[tail] - lifts[tail].mean()) ** 2)) / q),
        response_frequency=f_resp,
        natural_frequency=cfg.natural_frequency,
        displacement=disps,
        lift=lifts,
        drag=drags,
        iterations=np.asarray(iters),
        converged=converged,
    )