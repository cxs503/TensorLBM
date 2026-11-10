"""Resumable, wall-resolved ELBM DNS driver for the plane turbulent channel.

Why this module exists
----------------------
:func:`tensorlbm.elbm_channel_native.native_channel_run` runs one channel case
in a single C call: it initialises, spins up, samples and returns -- there is no
way to save the state mid-run.  A *wall-resolved* DNS at ``Re_tau = 180/390``
needs ``O(10^5)`` time steps on ``O(10^7)`` cells, i.e. several hours of wall
clock that must be split across job-slot limits.  This module therefore drives
the same fused kernel step by step from Python and adds exactly the three
things a production DNS needs:

* **checkpoint / restart** -- ``f``, the entropy warm-start ``alpha``, the
  statistics accumulators and the step counter are written to a single ``.npz``
  every ``ckpt_every`` steps and read back on resume, so a killed or timed-out
  job loses at most ``ckpt_every`` steps;
* **statistics accumulated in float64** -- plane-averaged mean velocity and the
  four independent Reynolds stresses per wall-normal row, averaged over the two
  homogeneous directions and over time (a restart continues the same time
  average, it does not start a new one);
* **the raw artefact format consumed by** ``tools/elbm_channel_report.py``
  (``prof_*`` / ``scalars`` / ``history`` plus a ``<stem>_meta.json``), so the
  DNS verdict is produced by the same gate code as every other channel run.

The physics (entropic D3Q27 collision, Guo forcing with the half-force velocity
shift, half-way bounce-back walls at ``y = 0.5`` and ``y = ny - 1.5``) is *not*
re-implemented here: each step is the C kernel ``elbm_channel_step``.  What is
re-implemented is the *initial condition* and the *statistics*, both of which
are pinned to the C driver by ``tests/test_elbm_channel_dns.py`` (one step of
the Python loop and the C single-cell hook agree; the profile statistics agree
with the C accumulation formulas to round-off).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .d3q27 import W_EXACT64 as W27
from .d3q27 import C as C27
from .elbm_channel_native import ELBM_PROFILES, native_channel_step_inplace

__all__ = [
    "DnsChannelConfig",
    "DnsRunState",
    "entropic_equilibrium_np",
    "initial_field",
    "load_checkpoint",
    "run_dns_channel",
    "sample_profiles",
    "save_checkpoint",
    "write_raw_npz",
]

# Lattice data come from :mod:`tensorlbm.d3q27` (single source of truth).  An
# early draft hand-wrote the 27 weights here and got corners/edges/faces
# swapped, which showed up as sum(f) = 0.30 instead of 1 -- exactly the kind of
# silent default-plane corruption the import prevents.
_C = np.asarray(C27.numpy(), dtype=np.float64)
_CX = np.ascontiguousarray(_C[:, 0], dtype=np.float32)
_CY = np.ascontiguousarray(_C[:, 1], dtype=np.float32)
_CZ = np.ascontiguousarray(_C[:, 2], dtype=np.float32)
_W = np.asarray(W27.numpy(), dtype=np.float64)
if abs(float(_W.sum()) - 1.0) > 1e-12:  # pragma: no cover - lattice data guard
    raise RuntimeError(f"D3Q27 weights do not sum to 1: {_W.sum()}")


@dataclass(frozen=True)
class DnsChannelConfig:
    """Grid, physics and sampling parameters of one ELBM channel DNS.

    All lengths are lattice units.  ``re_tau`` and ``u_tau`` fix the viscosity
    through the momentum balance of the forced channel::

        h    = (ny - 2) / 2                # half height, walls at 0.5 / ny-1.5
        nu   = u_tau * h / re_tau          # => tau = 3 nu + 1/2
        g    = u_tau^2 / h                 # body force per unit mass

    so the wall friction velocity is *given* by the forcing (``u_tau = sqrt(g h)``)
    and does not have to be measured, which is what makes ``C_f`` a clean gate.

    ``roll_amplitude`` adds ``roll_pairs`` pairs of counter-rotating streamwise
    vortices (the standard bypass-transition trigger for a periodic channel).
    It matters in practice: with white noise alone (``noise_amplitude``) and a
    coarse first cell (``Delta y+ >~ 5``) the perturbation decays and the flow
    relaminarises -- measured over 3500 steps at ``Delta y+ = 5.6``: the
    streamwise rms stayed at ``u_rms+ ~ 0.01`` against ``2.7`` in the DNS, and
    the mean profile relaxed towards the laminar parabola (``kappa_eff`` fell
    from 0.41 to 0.24).  The rolls inject the O(u_tau) vorticity that the
    near-wall cycle amplifies.
    """

    nx: int = 384
    ny: int = 194
    nz: int = 192
    re_tau: float = 180.0
    u_tau: float = 0.012
    steps: int = 60_000
    warmup: int = 15_000
    sample_every: int = 50
    n_iter: int = 5
    alpha_cap: float = 1.0
    seed: int = 20261110
    noise_amplitude: float = 0.06
    roll_amplitude: float = 0.0
    roll_pairs: int = 2
    log_every: int = 500
    ckpt_every: int = 5_000
    label: str = "elbm_dns"

    # ---- derived quantities -------------------------------------------------
    @property
    def fluid_height(self) -> float:
        """Channel height between the two half-way wall planes."""
        return float(self.ny - 2)

    @property
    def half_height(self) -> float:
        """Half channel height ``h`` (the wall distance of the centreline)."""
        return 0.5 * self.fluid_height

    @property
    def nu(self) -> float:
        return self.u_tau * self.half_height / self.re_tau

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def body_force(self) -> float:
        return self.u_tau**2 / self.half_height

    @property
    def delta_plus(self) -> float:
        """First fluid row in wall units, ``Delta y+ = re_tau / h``."""
        return self.re_tau / self.half_height

    @property
    def cells(self) -> int:
        return self.nx * self.ny * self.nz

    def validate(self) -> None:
        if self.ny < 8:
            raise ValueError(f"ny={self.ny} too small for a two-wall channel")
        if self.nx < 4 or self.nz < 4:
            raise ValueError("nx and nz must be >= 4 (periodic directions)")
        if not 0.0 < self.u_tau < 0.05:
            raise ValueError(
                f"u_tau={self.u_tau} outside (0, 0.05): above it the lattice "
                "Mach number leaves the incompressible regime"
            )
        if self.steps <= 0 or self.warmup < 0:
            raise ValueError("steps must be > 0 and warmup >= 0")
        if self.sample_every <= 0 or self.log_every <= 0:
            raise ValueError("sample_every and log_every must be > 0")
        if self.tau <= 0.5:
            raise ValueError(f"tau={self.tau:.6f} <= 0.5: negative viscosity")
        if self.warmup >= self.steps:
            raise ValueError("warmup must be < steps (no samples otherwise)")


@dataclass
class DnsRunState:
    """Everything that has to survive a checkpoint/restart."""

    f: np.ndarray                    # (27, nz, ny, nx) float32
    alpha: np.ndarray                # (nz, ny, nx) float32
    acc: np.ndarray                  # (6, ny) float64 statistics accumulators
    step: int = 0
    samples: int = 0
    min_population: float = math.inf
    max_abs_u: float = 0.0
    mean_rho: float = 1.0
    alpha_mean: float = 1.0
    alpha_min: float = 1.0
    alpha_max: float = 1.0
    entropy_drop_max: float = -math.inf
    entropy_excess_max: float = -math.inf
    finite: bool = True
    wall_clock: float = 0.0
    history: list[list[float]] | None = None

    def new_history(self, nx: int, ny: int, nz: int, cfg: DnsChannelConfig) -> None:
        f = np.full((27, nz, ny, nx), _W.astype(np.float32).reshape(27, 1, 1, 1))
        self.f = np.ascontiguousarray(f)
        self.alpha = np.ones((nz, ny, nx), dtype=np.float32)
        self.acc = np.zeros((6, ny), dtype=np.float64)
        self.history = []


def _law_of_wall_plus(yp: np.ndarray) -> np.ndarray:
    """Inner-scaled mean velocity used only as an initial condition."""
    yp = np.maximum(np.asarray(yp, dtype=np.float64), 1e-6)
    return np.where(yp < 11.4, yp, np.log(yp) / 0.41 + 5.2)


def _axis_factors(u: np.ndarray) -> np.ndarray:
    """Per-axis factors of the entropic D3Q27 equilibrium, shape ``(3, ..., 3)``.

    ``S = sqrt(1+3u^2)``, ``r = (S-2u)/(1+u)``, ``P = 6/(4+r+1/r)``,
    ``g(-1) = P r``, ``g(0) = P``, ``g(+1) = P/r``.  This mirrors ``g[][]`` of
    ``elbm_channel_kernel.c`` / ``_axis_factors`` of :mod:`tensorlbm.elbm`.
    """
    u = np.clip(np.asarray(u, dtype=np.float64), -0.9, 0.9)
    s = np.sqrt(1.0 + 3.0 * u * u)
    r = (s - 2.0 * u) / (1.0 + u)
    p_ = 6.0 / (4.0 + r + 1.0 / r)
    return np.stack([p_ * r, p_, p_ / r], axis=0)


def entropic_equilibrium_np(
    rho: np.ndarray, ux: np.ndarray, uy: np.ndarray, uz: np.ndarray
) -> np.ndarray:
    """Entropic (Gauss) D3Q27 equilibrium as ``(27,) + shape(ux)`` float64.

    Exact closed form, i.e. the *same* state the kernel collides towards -- an
    important detail: initialising with the second-order polynomial equilibrium
    instead leaves an O(u^3) mismatch that is harmless for the mean flow but
    not for the positivity bound of the entropy solve (small-weight directions
    differ by a factor of a few at u ~ 0.2, so ``alpha`` collapses to ~0.3 and
    the run starts with a spurious burst of numerical dissipation).
    """
    gx = _axis_factors(ux)
    gy = _axis_factors(uy)
    gz = _axis_factors(uz)
    out = np.empty((27,) + np.shape(ux), dtype=np.float64)
    for q in range(27):
        out[q] = (_W[q] * rho * gx[int(_CX[q]) + 1] * gy[int(_CY[q]) + 1]
                  * gz[int(_CZ[q]) + 1])
    return out


def initial_field(cfg: DnsChannelConfig) -> tuple[np.ndarray, np.ndarray]:
    """Law-of-the-wall mean flow + noise, mapped through the exact entropic
    equilibrium, in the kernel's ``(27, nz, ny, nx)`` float32 layout.

    Returns ``(f, alpha)`` with ``alpha`` warm start at 1.
    """
    rng = np.random.default_rng(cfg.seed)
    ny, half = cfg.ny, cfg.half_height
    y_wall = np.arange(ny) - 0.5                 # wall-normal distance from wall
    y_mirror = np.minimum(y_wall, cfg.fluid_height - y_wall)
    ux_row = cfg.u_tau * _law_of_wall_plus(y_mirror * cfg.re_tau / half)
    ux_row[0] = 0.0
    ux_row[-1] = 0.0

    shape = (cfg.nz, ny, cfg.nx)
    ux = np.broadcast_to(ux_row.reshape(1, ny, 1), shape).astype(np.float64).copy()
    uy = np.zeros(shape)
    uz = np.zeros(shape)
    rho = np.ones(shape)

    amp = cfg.noise_amplitude * cfg.u_tau
    fluid = (slice(None), slice(1, ny - 1), slice(None))
    for target in (ux, uy, uz):
        target[fluid] += amp * rng.standard_normal((cfg.nz, ny - 2, cfg.nx))
    if cfg.roll_amplitude > 0.0:
        # Counter-rotating streamwise vortices (the standard bypass-transition
        # trigger for a periodic channel): u_y and u_z are the curl of a
        # streamwise vector potential, tapered to zero at both walls, so the
        # perturbation is divergence free and survives the first collisions.
        taper = np.sin(np.clip(np.pi * y_mirror / half, 0.0, np.pi)).reshape(1, ny, 1)
        kx = 2.0 * np.pi * cfg.roll_pairs
        kz = 2.0 * np.pi * cfg.roll_pairs
        xs = np.arange(cfg.nx) / cfg.nx
        zs = np.arange(cfg.nz) / cfg.nz
        fx_sin = np.sin(kx * xs).reshape(1, 1, cfg.nx)
        fx_cos = np.cos(kx * xs).reshape(1, 1, cfg.nx)
        fz_sin = np.sin(kz * zs).reshape(cfg.nz, 1, 1)
        fz_cos = np.cos(kz * zs).reshape(cfg.nz, 1, 1)
        roll = cfg.roll_amplitude * cfg.u_tau
        uy += roll * fz_cos * fx_cos * taper
        uz -= roll * fz_sin * fx_sin * taper
    for target in (ux, uy, uz):     # no flow at the wall rows either
        target[:, 0, :] = 0.0
        target[:, ny - 1, :] = 0.0

    f = entropic_equilibrium_np(rho, ux, uy, uz)
    f[:, :, 0, :] = 0.0
    f[:, :, ny - 1, :] = 0.0
    alpha = np.ones(shape, dtype=np.float32)
    return np.ascontiguousarray(f, dtype=np.float32), alpha


def _macroscopics(f: np.ndarray, force: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Density and velocity from populations (single pass over the 27 planes)."""
    rho = np.zeros(f.shape[1:], dtype=np.float32)
    mx = np.zeros_like(rho)
    my = np.zeros_like(rho)
    mz = np.zeros_like(rho)
    for q in range(27):
        fq = f[q]
        rho += fq
        mx += _CX[q] * fq
        my += _CY[q] * fq
        mz += _CZ[q] * fq
    inv = np.reciprocal(np.maximum(rho, np.float32(1e-12)))
    ux = mx * inv + np.float32(0.5 * force)     # Guo half-force shift
    return rho, ux, my * inv, mz * inv


def sample_profiles(f: np.ndarray, cfg: DnsChannelConfig, acc: np.ndarray) -> np.ndarray:
    """Add one time sample of the plane-averaged statistics into ``acc``.

    ``acc`` rows are ``u, uu, vv, ww, uv, rho`` per wall-normal row, i.e. the
    same six profiles the C driver returns.  The Reynolds stresses are built
    from the *fluctuating* components, ``<u'u'> = <u^2> - <u>^2`` (the C driver
    uses ``<u_z^2>`` for ``ww``, identical here because ``<u_z> = 0`` in this
    statistically homogeneous problem; the Python path is the consistent one).
    """
    rho, ux, uy, uz = _macroscopics(f, cfg.body_force)
    fluid = slice(1, cfg.ny - 1)
    ux_f = ux[:, fluid].astype(np.float64)
    uy_f = uy[:, fluid].astype(np.float64)
    uz_f = uz[:, fluid].astype(np.float64)
    rho_f = rho[:, fluid].astype(np.float64)
    u_bar = ux_f.mean(axis=(0, 2), dtype=np.float64)
    v_bar = uy_f.mean(axis=(0, 2), dtype=np.float64)
    w_bar = uz_f.mean(axis=(0, 2), dtype=np.float64)
    m_uu = np.square(ux_f).mean(axis=(0, 2), dtype=np.float64)
    m_vv = np.square(uy_f).mean(axis=(0, 2), dtype=np.float64)
    m_ww = np.square(uz_f).mean(axis=(0, 2), dtype=np.float64)
    m_uv = (ux_f * uy_f).mean(axis=(0, 2), dtype=np.float64)
    m_rho = rho_f.mean(axis=(0, 2), dtype=np.float64)
    acc[0, 1:-1] += u_bar
    acc[1, 1:-1] += m_uu - u_bar**2
    acc[2, 1:-1] += m_vv - v_bar**2
    acc[3, 1:-1] += m_ww - w_bar**2
    acc[4, 1:-1] += m_uv - u_bar * v_bar
    acc[5, 1:-1] += m_rho
    return acc


def save_checkpoint(path: Path | str, cfg: DnsChannelConfig, state: DnsRunState) -> Path:
    """Write the full restart state (fields + accumulators + counters)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hist = np.asarray(state.history or [], dtype=np.float64)
    np.savez(
        path,
        f=state.f,
        alpha=state.alpha,
        acc=state.acc,
        history=hist,
        scalars=np.array(
            [state.step, state.samples, state.min_population, state.max_abs_u,
             state.mean_rho, state.alpha_mean, state.alpha_min, state.alpha_max,
             state.entropy_drop_max, float(state.finite), state.wall_clock,
             state.entropy_excess_max],
            dtype=np.float64,
        ),
    )
    (path.parent / f"{path.stem}_config.json").write_text(json.dumps(asdict(cfg), indent=1))
    return path


def load_checkpoint(path: Path | str, cfg: DnsChannelConfig) -> DnsRunState:
    """Read a restart state written by :func:`save_checkpoint`."""
    data = np.load(Path(path))
    s = np.asarray(data["scalars"]).tolist()
    state = DnsRunState(
        f=np.ascontiguousarray(data["f"], dtype=np.float32),
        alpha=np.ascontiguousarray(data["alpha"], dtype=np.float32),
        acc=np.ascontiguousarray(data["acc"], dtype=np.float64),
        step=int(s[0]),
        samples=int(s[1]),
        min_population=float(s[2]),
        max_abs_u=float(s[3]),
        mean_rho=float(s[4]),
        alpha_mean=float(s[5]),
        alpha_min=float(s[6]),
        alpha_max=float(s[7]),
        entropy_drop_max=float(s[8]),
        finite=bool(s[9] > 0.5),
        entropy_excess_max=float(s[11]) if len(s) > 11 else float("nan"),
        wall_clock=float(s[10]),
        history=np.asarray(data["history"]).tolist(),
    )
    expect = (27, cfg.nz, cfg.ny, cfg.nx)
    if state.f.shape != expect:
        raise ValueError(f"checkpoint field shape {state.f.shape} != {expect}: config mismatch")
    return state


def write_raw_npz(
    path: Path | str,
    cfg: DnsChannelConfig,
    state: DnsRunState,
    *,
    extra_scalars: dict[str, float] | None = None,
) -> Path:
    """Write the artefact consumed by ``tools/elbm_channel_report.py``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if state.samples <= 0:
        raise RuntimeError("no statistics samples collected: increase steps")
    prof = state.acc / float(state.samples)
    np.savez(
        path,
        **{f"prof_{name}": prof[k] for k, name in enumerate(ELBM_PROFILES)},
        scalars=np.array(
            [state.samples, state.min_population, state.max_abs_u, state.alpha_mean,
             state.alpha_min, state.alpha_max, state.entropy_drop_max,
             float(state.finite), cfg.nu, state.mean_rho],
            dtype=np.float64,
        ),
        history=np.asarray(state.history or [], dtype=np.float64),
    )
    meta: dict[str, Any] = {
        "steps": state.step,
        "warmup": cfg.warmup,
        "sample_every": cfg.sample_every,
        "samples": state.samples,
        "grid": [cfg.nx, cfg.ny, cfg.nz],
        "delta_plus": cfg.delta_plus,
        "tau": cfg.tau,
        "nu": cfg.nu,
        "u_tau": cfg.u_tau,
        "body_force": cfg.body_force,
        "re_tau": cfg.re_tau,
        "wall_clock_s": state.wall_clock,
        "cells": cfg.cells,
        # H-theorem residual net of the float32 rounding floor (<= 0 means the
        # unforced collision did not increase the discrete entropy)
        "entropy_drop_max_collision": state.entropy_drop_max,
        "entropy_excess_max": state.entropy_excess_max,
        "alpha_mean": state.alpha_mean,
        "alpha_min": state.alpha_min,
        "min_population": state.min_population,
        "throughput_Mcell_s": (
            cfg.cells * state.step / state.wall_clock / 1e6 if state.wall_clock > 0 else None
        ),
        "config": asdict(cfg),
    }
    meta.update(extra_scalars or {})
    (path.parent / f"{path.stem}_meta.json").write_text(json.dumps(meta, indent=1))
    return path


def run_dns_channel(
    cfg: DnsChannelConfig,
    *,
    out_raw: Path | str,
    ckpt: Path | str | None = None,
    resume: bool = False,
    progress: bool = False,
    max_wall_clock: float | None = None,
) -> tuple[DnsRunState, dict[str, Any]]:
    """Run (or continue) one channel DNS and write the raw artefact.

    Args:
        cfg: grid/physics/sampling configuration.
        out_raw: path of the ``.npz`` the report tool reads.
        ckpt: checkpoint path (defaults to ``<out_raw dir>/<stem>_ckpt.npz``).
        resume: continue from ``ckpt`` if it exists (the config must match --
            a mismatch raises instead of silently mixing two grids).
        progress: print a log line every ``cfg.log_every`` steps.
        max_wall_clock: stop (and checkpoint) after this many seconds of this
            call, so a long DNS can be resumed from a later job slot.

    Returns:
        ``(state, summary)``; ``summary["complete"]`` is False when the run
        stopped early on ``max_wall_clock`` and must be resumed.
    """
    cfg.validate()
    out_raw = Path(out_raw)
    ckpt = Path(ckpt) if ckpt is not None else out_raw.parent / f"{out_raw.stem}_ckpt.npz"

    resumed_from: int | None = None
    if resume and ckpt.exists():
        state = load_checkpoint(ckpt, cfg)
        resumed_from = state.step
        if state.step >= cfg.steps:
            write_raw_npz(out_raw, cfg, state)
            return state, {"complete": True, "resumed_from": state.step, "steps": state.step}
    else:
        state = DnsRunState(f=None, alpha=None, acc=None)  # type: ignore[arg-type]
        f, alpha = initial_field(cfg)
        state.f, state.alpha = f, alpha
        state.acc = np.zeros((6, cfg.ny), dtype=np.float64)
        state.history = []

    work = state.f
    scratch = np.empty_like(work)
    alpha = state.alpha
    t0 = time.time()
    steps_done = 0
    while state.step < cfg.steps and state.finite:
        # ---- one ELBM step: collision + Guo source + streaming (C kernel)
        stats = np.zeros(6, dtype=np.float64)
        rc = native_channel_step_inplace(
            work, scratch, alpha, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, tau=cfg.tau,
            force=cfg.body_force, n_iter=cfg.n_iter, alpha_cap=cfg.alpha_cap, stats=stats,
        )
        if rc != 0:  # pragma: no cover
            state.finite = False
            break
        state.step += 1
        steps_done += 1
        state.alpha_mean = float(stats[0])
        state.alpha_min = float(stats[1])
        state.alpha_max = float(stats[2])
        state.entropy_drop_max = max(state.entropy_drop_max, float(stats[3]))
        state.entropy_excess_max = max(state.entropy_excess_max, float(stats[5]))

        if state.step % cfg.log_every == 0 or state.step == 1:
            rho, ux, _, _ = _macroscopics(work, cfg.body_force)
            fluid = slice(1, cfg.ny - 1)
            min_pop = float(work[:, :, fluid, :].min())
            state.min_population = min(state.min_population, min_pop)
            state.max_abs_u = max(state.max_abs_u, float(np.abs(ux[:, fluid, :]).max()))
            state.mean_rho = float(rho[:, fluid, :].mean())
            state.wall_clock += time.time() - t0
            t0 = time.time()
            state.history = list(state.history or [])
            state.history.append(
                [state.step, state.max_abs_u, state.mean_rho, state.min_population,
                 state.alpha_mean, state.alpha_min, state.alpha_max, state.entropy_drop_max]
            )
            if not np.isfinite(work[:, :, fluid, :]).all():
                state.finite = False
            if progress:
                print(
                    f"[{cfg.label}] step {state.step}/{cfg.steps} "
                    f"max|u|={state.max_abs_u:.4f} min f={state.min_population:.3e} "
                    f"<alpha>={state.alpha_mean:.4f} dH_max={state.entropy_drop_max:.2e} "
                    f"({state.wall_clock:.1f}s)",
                    flush=True,
                )

        if state.step > cfg.warmup and (state.step - cfg.warmup) % cfg.sample_every == 0:
            sample_profiles(work, cfg, state.acc)
            state.samples += 1

        if ckpt is not None and steps_done % cfg.ckpt_every == 0:
            state.wall_clock += time.time() - t0
            t0 = time.time()
            save_checkpoint(ckpt, cfg, state)
        if max_wall_clock is not None and time.time() - t0 > max_wall_clock:
            break

    state.wall_clock += time.time() - t0
    state.f = work
    state.alpha = alpha
    if ckpt is not None:
        save_checkpoint(ckpt, cfg, state)
    complete = state.finite and state.step >= cfg.steps
    summary = {
        "complete": bool(complete),
        "resumed_from": resumed_from,
        "steps": state.step,
        "samples": state.samples,
        "finite": state.finite,
        "wall_clock_s": state.wall_clock,
        "throughput_Mcell_s": cfg.cells * state.step / state.wall_clock / 1e6 if state.wall_clock else None,
        "min_population": state.min_population,
        "alpha_mean": state.alpha_mean,
        "alpha_min": state.alpha_min,
        "alpha_max": state.alpha_max,
        "entropy_drop_max": state.entropy_drop_max,
        "entropy_excess_max": state.entropy_excess_max,
        "delta_plus": cfg.delta_plus,
    }
    if complete:
        write_raw_npz(out_raw, cfg, state)
    return state, summary


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description="Resumable ELBM channel DNS (D3Q27)")
    ap.add_argument("--nx", type=int, default=384)
    ap.add_argument("--ny", type=int, default=194)
    ap.add_argument("--nz", type=int, default=192)
    ap.add_argument("--re-tau", type=float, default=180.0)
    ap.add_argument("--u-tau", type=float, default=0.012)
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--warmup", type=int, default=15_000)
    ap.add_argument("--sample-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=5_000)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--roll-amplitude", type=float, default=0.0)
    ap.add_argument("--roll-pairs", type=int, default=2)
    ap.add_argument("--label", default="elbm_dns")
    ap.add_argument("--out", default="artifacts/elbm_dns/raw/elbm_dns.npz")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-wall-clock", type=float, default=None)
    ap.add_argument("--export-only", action="store_true",
                    help="write the raw artefact from an existing checkpoint (partial run)")
    args = ap.parse_args(argv)
    cfg = DnsChannelConfig(
        nx=args.nx, ny=args.ny, nz=args.nz, re_tau=args.re_tau, u_tau=args.u_tau,
        steps=args.steps, warmup=args.warmup, sample_every=args.sample_every,
        ckpt_every=args.ckpt_every, log_every=args.log_every, label=args.label,
        roll_amplitude=args.roll_amplitude, roll_pairs=args.roll_pairs,
    )
    if args.export_only:
        ckpt = Path(args.ckpt) if args.ckpt else Path(args.out).parent / f"{Path(args.out).stem}_ckpt.npz"
        state = load_checkpoint(ckpt, cfg)
        write_raw_npz(args.out, cfg, state)
        print(json.dumps({"exported": args.out, "steps": state.step,
                          "samples": state.samples}, indent=1))
        return 0
    _, summary = run_dns_channel(
        cfg, out_raw=args.out, ckpt=args.ckpt, resume=args.resume, progress=True,
        max_wall_clock=args.max_wall_clock,
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())