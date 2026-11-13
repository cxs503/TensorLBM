"""Quasi-static capillary-pressure ladder with per-rung checkpointing.

Protocol transplant
-------------------
The protocol is the ``quasi-static-pc-ladder-with-checkpoint`` practice of
``je-santos/MPLBM-UT`` (OSS inventory entry, tag
``quasi-static-pc-ladder-with-checkpoint``, files
``src/2-phase_LBM/ShanChen.cpp:526-537,570-583,820-849,875-903``):

* the driving capillary pressure is ramped in ``num_pc_steps`` equal steps
  (upstream: ``runnum = num_pc_steps + 1``, one run number per rung);
* the step size is tied to the *smallest resolvable throat*,
  ``dRho = 6 sigma cos(theta) / r_min`` in the upstream D3Q19 two-component
  convention where ``p = rho/3``, i.e. one step equals the Young-Laplace entry
  pressure of that throat, ``dP = 2 sigma cos(theta) / r_min`` (3D);
* every rung is advanced until the *relative change of the mean velocity*
  drops below ``convergence`` (``% / iteration``), with ``it_conv`` as the
  sampling interval and ``it_max`` as the hard cap;
* the run number and the full lattice state are written to disk between rungs
  so a ladder can be interrupted and resumed.

This module re-derives the protocol against *this* repository's conventions
instead of copying it:

* 2D (D2Q9) geometry: one principal curvature, so the entry pressure of a
  throat of half-width ``r_min`` is ``sigma cos(theta) / r_min`` -- the
  dimension transfer is explicit in :func:`pc_stair_schedule`;
* the driving is a body force (``gravity_mode="rho"`` in
  :mod:`tensorlbm.cac_lbm2d`), which is momentum-equivalent to the upstream
  pressure boundaries but is stable on a periodic loop; the imposed pressure
  difference across a plug is ``Delta rho * g * L`` (exact force balance, not a
  capillary assumption);
* the convergence test is re-expressed as a *rate* (change per iteration) so it
  is independent of ``it_conv``;
* checkpointing reuses :mod:`tensorlbm.checkpoint` (tensor + JSON + SHA-256)
  and stores the *complete* solver state ``(f, g, phi*u_prev)``.

Measured quantities
-------------------
* ``kappa``    interface curvature from ``kappa = -div(grad(phi)/|grad(phi)|)``
  (positive when the liquid side is concave, so ``p_liquid - p_gas = sigma kappa``);
* meniscus positions ``x_lead`` / ``x_trail`` from the ``phi = 1/2`` crossings;
* the static Young-Laplace residual ``sigma (kappa_lead - kappa_trail)`` versus
  the imposed ``Delta rho g L`` -- the rung-by-rung acceptance number.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from . import cac_lbm2d as cac
from .checkpoint import load_checkpoint, save_checkpoint

__all__ = [
    "LadderParams",
    "RungRecord",
    "pc_stair_schedule",
    "tapered_channel",
    "half_width_profile",
    "plug_phi",
    "curvature_field",
    "meniscus_state",
    "settle",
    "save_ladder_state",
    "load_ladder_state",
    "run_ladder",
]


# ---------------------------------------------------------------------------
# Ladder schedule (upstream parity)
# ---------------------------------------------------------------------------


def pc_stair_schedule(
    *,
    sigma: float,
    cos_theta: float,
    r_min: float,
    num_pc_steps: int,
    dim: int = 2,
    span: str = "full",
) -> list[float]:
    """Capillary-pressure rungs of the quasi-static ladder.

    Args:
        sigma: surface tension in lattice units.
        cos_theta: ``cos`` of the contact angle measured through the wetting
            phase (upstream keeps the sign; the inventory's ``abs()`` warning is
            why this argument is not taken as ``|cos|``).
        r_min: half-width of the smallest resolvable throat.
        num_pc_steps: number of rungs (``runnum = num_pc_steps + 1`` upstream,
            rung 0 being the pre-ladder state).
        dim: ``3`` reproduces the upstream ``2 sigma cos(theta) / r_min`` step,
            ``2`` the D2Q9 single-curvature step ``sigma cos(theta) / r_min``.
        span: ``"oss"`` -- upstream step size: every rung adds one full entry
            pressure of ``r_min`` (the upstream inventory text: the step is
            ``dRho = 6 sigma cos(theta)/r_min``).  ``"full"`` -- the same total
            range ``num_pc_steps * dP_step`` is subdivided into
            ``num_pc_steps`` equal rungs, i.e. the resolved ladder used here to
            locate a *single* throat (see the xval report, difference D3).

    Returns:
        List of ``num_pc_steps`` pressures ``[dP, 2 dP, ..., num_pc_steps dP]``
        in increasing magnitude (``0.0`` rungs are dropped from the list; the
        caller enumerates rung 0 itself).
    """
    if num_pc_steps < 1:
        raise ValueError("num_pc_steps must be >= 1")
    if r_min <= 0.0:
        raise ValueError("r_min must be > 0")
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    if span not in ("oss", "full"):
        raise ValueError("span must be 'oss' or 'full'")
    factor = 2.0 if dim == 3 else 1.0
    step = factor * sigma * cos_theta / r_min
    if span == "oss":
        return [step * (k + 1) for k in range(num_pc_steps)]
    return [step * (k + 1) / num_pc_steps for k in range(num_pc_steps)]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def half_width_profile(
    ny: int,
    nx: int,
    *,
    r_wide: float,
    r_min: float,
    throat_center: float,
    throat_half_length: float,
) -> torch.Tensor:
    """Smooth taper ``R(x)`` (cosine bump), quantised to whole lattice rows."""
    x = torch.arange(nx, dtype=torch.float64)
    a = max(float(throat_half_length), 1.0)
    s = ((x - (throat_center - a)) / (2.0 * a)).clamp(0.0, 1.0)
    bump = 0.5 * (1.0 - torch.cos(2.0 * math.pi * s))
    r = r_wide - (r_wide - r_min) * bump
    return r


def tapered_channel(
    ny: int,
    nx: int,
    *,
    wall_t: int = 2,
    r_wide: float,
    r_min: float,
    throat_center: float,
    throat_half_length: float,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Venturi-like channel whose walls are *only* perpendicular to ``y``.

    Keeping every solid/fluid face horizontal is deliberate: this repository's
    contact-angle wall model is derived for the plate normal along ``y``
    (``cac_lbm2d.refresh_solid_ghosts``); walls with an ``x`` normal were found
    to destabilise the run (see the xval report, friction F2).  Every column
    therefore carries its own top and bottom wall row.

    Returns ``(wall_mask, half_width_rows)`` where ``half_width_rows[x]`` is the
    quantised half-width actually realised in lattice rows.
    """
    r = half_width_profile(
        ny,
        nx,
        r_wide=r_wide,
        r_min=r_min,
        throat_center=throat_center,
        throat_half_length=throat_half_length,
    )
    r_rows = torch.round(r).to(torch.int64)
    y = torch.arange(ny, dtype=torch.int64).view(-1, 1)
    y_c = (ny - 1) / 2.0
    solid = (y - y_c).abs() > r_rows.view(1, -1)
    mask = torch.zeros((ny, nx), dtype=torch.bool)
    mask[:wall_t, :] = True
    mask[ny - wall_t :, :] = True
    mask |= solid
    return mask.to(device), r_rows.to(device)


def plug_phi(
    shape: tuple[int, int],
    *,
    width: float,
    x_lo: float,
    x_hi: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Liquid plug ``x_lo < x < x_hi`` (two diffuse menisci) on a periodic x axis."""
    nx = shape[1]
    x = torch.arange(nx, device=device, dtype=dtype).view(1, -1)
    lo = 0.5 * (1.0 + torch.tanh(2.0 * (x - x_lo) / width))
    hi = 0.5 * (1.0 + torch.tanh(2.0 * (x - x_hi) / width))
    return (lo * (1.0 - hi)).expand(*shape).contiguous()


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def _smooth(field: torch.Tensor, passes: int = 2) -> torch.Tensor:
    out = field
    for _ in range(max(0, passes)):
        out = 0.25 * (
            torch.roll(out, 1, dims=-2)
            + torch.roll(out, -1, dims=-2)
            + torch.roll(out, 1, dims=-1)
            + torch.roll(out, -1, dims=-1)
        )
    return out


def curvature_field(phi: torch.Tensor, *, passes: int = 2) -> torch.Tensor:
    """Interface curvature ``kappa = -div(grad phi / |grad phi|)``.

    Sign convention: ``kappa > 0`` where the liquid (``phi = 1``) side is
    concave, so that Young-Laplace reads ``p_liquid - p_gas = sigma * kappa``
    (a liquid drop in gas has ``kappa = +1/R``).
    """
    p = _smooth(phi, passes)
    gx, gy = cac.iso_gradient_2d(p)
    gnorm = (gx * gx + gy * gy).sqrt().clamp(min=1e-12)
    nx, ny_ = gx / gnorm, gy / gnorm
    dnx = 0.5 * (torch.roll(nx, -1, dims=-1) - torch.roll(nx, 1, dims=-1))
    dny = 0.5 * (torch.roll(ny_, -1, dims=-2) - torch.roll(ny_, 1, dims=-2))
    return -(dnx + dny)


def meniscus_state(
    phi: torch.Tensor,
    *,
    fluid_mask: torch.Tensor | None = None,
    band: tuple[float, float] = (0.3, 0.7),
    passes: int = 2,
) -> dict[str, Any]:
    """Mid-channel meniscus positions plus band-averaged curvatures.

    ``x_lead`` / ``x_trail`` are the ``phi = 1/2`` crossings of the centre row of
    the wetted channel (the plug is the ``phi > 1/2`` interval); ``kappa_lead`` /
    ``kappa_trail`` average :func:`curvature_field` over the diffuse interface
    band on either side of the plug mid-point.
    """
    ny, nx = phi.shape
    if fluid_mask is None:
        fluid_mask = torch.ones_like(phi, dtype=torch.bool)
    rows = torch.nonzero(fluid_mask.any(dim=1)).flatten()
    if rows.numel() == 0:
        nan = float("nan")
        return {
            "x_lead": nan,
            "x_trail": nan,
            "x_mid": nan,
            "plug_length": nan,
            "kappa_lead": nan,
            "kappa_trail": nan,
            "ok": False,
        }
    row = int(rows[rows.numel() // 2].item())
    line = phi[row]
    liq = torch.nonzero(line > 0.5).flatten()
    if liq.numel() == 0:
        nan = float("nan")
        return {
            "x_lead": nan,
            "x_trail": nan,
            "x_mid": nan,
            "plug_length": nan,
            "kappa_lead": nan,
            "kappa_trail": nan,
            "ok": False,
        }
    x_trail = float(liq.min().item())
    x_lead = float(liq.max().item())
    x_mid = 0.5 * (x_trail + x_lead)

    kap = curvature_field(phi, passes=passes)
    bandmask = (phi > band[0]) & (phi < band[1]) & fluid_mask
    xx = torch.arange(nx, dtype=kap.dtype).view(1, -1).expand_as(kap)

    def _mean(mask: torch.Tensor) -> float:
        m = mask & bandmask
        n = int(m.sum().item())
        if n < 8:
            return float("nan")
        return float(kap[m].mean().item())

    return {
        "row": row,
        "x_lead": x_lead,
        "x_trail": x_trail,
        "x_mid": x_mid,
        "plug_length": x_lead - x_trail,
        "kappa_lead": _mean(xx > x_mid),
        "kappa_trail": _mean(xx <= x_mid),
        "ok": True,
    }


def centreline_velocity(field_u: torch.Tensor, row: int) -> float:
    """Plug displacement rate proxy: mean streamwise velocity on the centre row."""
    return float(field_u[row].mean().item())


# ---------------------------------------------------------------------------
# Quasi-static settling (upstream convergence rule)
# ---------------------------------------------------------------------------


@dataclass
class LadderParams:
    """Settling controls, named after the upstream XML fields where possible."""

    conv: float = 1e-4
    """upstream ``convergence``: relative velocity change **per iteration**."""

    it_conv: int = 200
    """upstream ``it_conv``: sampling interval of the velocity average."""

    it_max: int = 200_000
    """upstream ``it_max``: hard cap of iterations per rung."""

    min_iters: int = 0
    """Do not accept convergence before this many iterations."""

    u_floor: float = 1e-12
    """Velocity floor used in the relative change (avoids 0/0 at rest)."""


@dataclass
class RungRecord:
    rung: int
    pressure: float
    g: float
    steps: int
    converged: bool
    rel_rate: float
    u_prev: float
    u_rate: float
    u_max: float
    meas: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "pressure": self.pressure,
            "g": self.g,
            "steps": self.steps,
            "converged": self.converged,
            "rel_rate": self.rel_rate,
            "u_prev": self.u_prev,
            "u_rate": self.u_rate,
            "u_max": self.u_max,
            "meas": self.meas,
            "checkpoint": self.checkpoint,
        }


def _u_rate(sim: cac.CAC2D) -> tuple[float, float]:
    m = cac.macro(sim)
    u = m["u"]
    fm = sim.fluid_mask
    sp = (u[0] * u[0] + u[1] * u[1]).sqrt()
    return float(sp[fm].mean().item()), float(sp[fm].max().item())


def settle(
    sim: cac.CAC2D,
    *,
    g: float,
    params: LadderParams,
    on_sample: Callable[[cac.CAC2D, int, float], None] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Advance one rung until the upstream quasi-static criterion is met.

    ``rel_rate = |u1 - u0| / (max(u0, u_floor) * it_conv)`` is the mean-velocity
    change expressed **per iteration**, matching the upstream
    ``convergence`` (``% / iteration``) semantics while being independent of the
    sampling interval.
    """
    sim.gravity = (g, 0.0)
    steps = 0
    u0, u_max = _u_rate(sim)
    rel_rate = float("inf")
    converged = False
    while steps < params.it_max:
        n = min(params.it_conv, params.it_max - steps)
        cac.run(sim, n, sample_every=0)
        steps += n
        u1, u_max = _u_rate(sim)
        rel_rate = abs(u1 - u0) / (max(u0, params.u_floor) * params.it_conv)
        if on_sample is not None:
            on_sample(sim, steps, rel_rate)
        if progress is not None:
            progress(
                {
                    "steps": steps,
                    "u_prev": u0,
                    "u_rate": u1,
                    "rel_rate": rel_rate,
                    "u_max": u_max,
                }
            )
        u0 = u1
        if steps >= params.min_iters and rel_rate < params.conv:
            converged = True
            break
    return {
        "steps": steps,
        "converged": converged,
        "rel_rate": rel_rate,
        "u_prev": u0,
        "u_rate": u1,
        "u_max": u_max,
    }


# ---------------------------------------------------------------------------
# Checkpointing (complete solver state, SHA-256 digested)
# ---------------------------------------------------------------------------


def _state_tensor(sim: cac.CAC2D) -> torch.Tensor:
    ny, nx = sim.shape
    rows = [sim.f.reshape(-1, ny, nx), sim.g.reshape(-1, ny, nx)]
    phi_u = getattr(sim, "phi_u_prev", None)
    if phi_u is not None:
        for t in phi_u:
            rows.append(t.reshape(-1, ny, nx))
    return torch.cat(rows, dim=0).detach().cpu().contiguous()


def _digest(t: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def save_ladder_state(
    sim: cac.CAC2D,
    directory: str | Path,
    *,
    rung: int,
    g: float,
    pressure: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the complete CAC state for rung ``rung``; returns provenance."""
    directory = Path(directory)
    state = _state_tensor(sim)
    meta = {
        "rung": int(rung),
        "g": float(g),
        "pressure": float(pressure),
        "shape": list(sim.shape),
        "state_sha256": _digest(state),
        "gravity": [float(v) for v in sim.gravity],
        "policy": sim.wall_phi_policy,
        "contact_angle_deg": sim.contact_angle_deg,
    }
    if extra:
        meta.update(extra)
    save_checkpoint(state, int(sim.step_count), directory, extra=meta)
    return {"path": str(directory), **meta}


def load_ladder_state(
    path: str | Path,
    sim: cac.CAC2D,
    *,
    verify_digest: bool = True,
) -> dict[str, Any]:
    """Restore a :func:`save_ladder_state` checkpoint into ``sim``."""
    state, step, meta = load_checkpoint(Path(path), expected_shape=None)
    want = meta.get("state_sha256")
    got = _digest(state)
    if verify_digest and want is not None and want != got:
        raise ValueError(
            f"ladder checkpoint digest mismatch at {path}: metadata={want} recomputed={got}"
        )
    state = state.to(dtype=sim.dtype, device=sim.device)
    n_q = state.shape[0]
    if n_q not in (18, 20):
        raise ValueError(f"unexpected stacked state leading dim {n_q}")
    sim.f = state[0:9].clone()
    sim.g = state[9:18].clone()
    if n_q == 20:
        sim.phi_u_prev = (state[18].squeeze(0).clone(), state[19].squeeze(0).clone())
    else:  # pragma: no cover - only for the deliberately incomplete checkpoint
        sim.phi_u_prev = tuple(torch.zeros_like(state[0]) for _ in range(2))
    sim.step_count = int(step)
    return {"step": step, "digest": got, **{k: v for k, v in meta.items() if k != "state_sha256"}}


# ---------------------------------------------------------------------------
# Ladder driver
# ---------------------------------------------------------------------------


def run_ladder(
    sim_factory: Callable[[], cac.CAC2D],
    *,
    pressures: list[float],
    delta_rho: float,
    plug_length: float,
    params: LadderParams,
    checkpoint_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    rung0_pressure: float = 0.0,
    log: Callable[[str], None] | None = None,
) -> list[RungRecord]:
    """Run (or resume) the quasi-static ladder.

    ``pressures`` are the capillary-pressure rungs; the actuator is the body
    force ``g = dP / (delta_rho * plug_length)`` (force balance of the plug, not
    a capillary assumption).  rung 0 is the pre-ladder state settled at
    ``rung0_pressure`` -- so a resumed ladder walks the *same* schedule.

    ``resume_from`` points at a checkpoint directory written by
    :func:`save_ladder_state`; the ladder restarts at ``rung + 1``.
    """
    records: list[RungRecord] = []
    sim = sim_factory()
    start_rung = 1
    if resume_from is not None:
        meta = load_ladder_state(resume_from, sim)
        start_rung = int(meta["rung"]) + 1
        if log:
            log(f"resumed from {resume_from}: rung={meta['rung']} step={meta['step']}")

    if start_rung == 1:
        s0 = settle(sim, g=rung0_pressure / (delta_rho * plug_length), params=params)
        m0 = meniscus_state(sim.f.sum(dim=0), fluid_mask=sim.fluid_mask)
        rec0 = RungRecord(
            rung=0,
            pressure=float(rung0_pressure),
            g=rung0_pressure / (delta_rho * plug_length),
            steps=s0["steps"],
            converged=s0["converged"],
            rel_rate=s0["rel_rate"],
            u_prev=s0["u_prev"],
            u_rate=s0["u_rate"],
            u_max=s0["u_max"],
            meas=m0,
        )
        if checkpoint_dir is not None:
            rec0.checkpoint = save_ladder_state(
                sim,
                Path(checkpoint_dir) / "rung_000",
                rung=0,
                g=rec0.g,
                pressure=rec0.pressure,
            )
        records.append(rec0)
        if log:
            log(f"rung 0 p={rec0.pressure:.4e} steps={rec0.steps} conv={rec0.converged}")

    for k, p in enumerate(pressures, start=1):
        if k < start_rung:
            continue
        g = p / (delta_rho * plug_length)
        s = settle(sim, g=g, params=params)
        meas = meniscus_state(sim.f.sum(dim=0), fluid_mask=sim.fluid_mask)
        rec = RungRecord(
            rung=k,
            pressure=float(p),
            g=g,
            steps=s["steps"],
            converged=s["converged"],
            rel_rate=s["rel_rate"],
            u_prev=s["u_prev"],
            u_rate=s["u_rate"],
            u_max=s["u_max"],
            meas=meas,
        )
        if checkpoint_dir is not None:
            rec.checkpoint = save_ladder_state(
                sim,
                Path(checkpoint_dir) / f"rung_{k:03d}",
                rung=k,
                g=g,
                pressure=float(p),
            )
        records.append(rec)
        if log:
            log(
                f"rung {k} p={p:.4e} g={g:.3e} steps={s['steps']} conv={s['converged']} "
                f"x_lead={meas.get('x_lead')} kap=({meas.get('kappa_trail')}, {meas.get('kappa_lead')})"
            )
    return records


def young_laplace_residual(
    *,
    sim_sigma: float,
    kappa_lead: float,
    kappa_trail: float,
    delta_rho: float,
    g: float,
    plug_length: float,
) -> dict[str, float]:
    """Static Young-Laplace balance: ``sigma (kappa_lead - kappa_trail) = drho g L``."""
    lhs = sim_sigma * (kappa_lead - kappa_trail)
    rhs = delta_rho * g * plug_length
    denom = max(abs(rhs), 1e-30)
    return {
        "sigma_dkappa": lhs,
        "drho_g_L": rhs,
        "relerr": abs(lhs - rhs) / denom,
        "ratio": lhs / rhs if rhs != 0.0 else float("nan"),
    }