"""Declarative example runtime -- the replacement for the *example-per-main* anti-pattern.

Why this module exists
----------------------
``examples/`` in TensorLBM grew to 100+ stand-alone scripts, each of which
parses its own CLI, re-creates the lattice, re-writes the collision/forcing
loop and re-derives the post-processing.  That is the ``example-per-main``
anti-pattern (sorush-khajepor/listLBM style): every script is a full
copy of the driver, so

* a change to the solver (or to a post-processing convention) has to be
  applied N times and silently drifts,
* the *result* of an example is not a first-class artifact -- it is whatever
  the script printed, so it cannot be refereed by a machine,
* there is no provenance: two runs of the "same" example are not comparable.

This module provides one library-level case runtime:

    spec = get_case("channel_poiseuille", ny=48, steps=4000)
    record = run_case(spec)                     # dict, JSON-serialisable
    print(record["kpis"]["u_max"], record["analytic"]["u_max"])

The example script then shrinks to a thin ``__main__`` that only selects a
case and prints/writes the record.  Every arm is described by a
:class:`CaseSpec`, whose ``digest()`` is part of the run record, so a result
is reproducible and auditable from the record alone.

Physics implemented here (D2Q9, BGK, Guo forcing, half-way bounce-back):

* ``channel_poiseuille``  -- force-driven plane channel; reference
  ``u_max = g H^2 / (8 nu)`` with ``H = ny - 1`` (no-slip planes half a
  lattice inside the two wall rows), ``nu = (tau - 1/2)/3``.
* ``taylor_green_2d``     -- periodic Taylor-Green vortex
  ``u = U0 (cos kx sin ky, -sin kx cos ky)``.  Its Laplacian is ``-2k^2 u``,
  so the velocity decays as ``exp(-2 nu k^2 t)`` and the kinetic energy as
  ``exp(-4 nu k^2 t)`` with ``k = 2 pi / nx``.

Both references are closed-form, so the runtime itself can be refereed
without any external table.

Deliberate non-goals: this is not a new solver.  It composes the existing
library primitives (``collide_bgk``, ``stream``, ``equilibrium``,
``macroscopic``, ``bounce_back``) so that an example cannot drift from the
library.  Any improvement lands in the primitives and reaches every case.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Mapping

import torch

from .d2q9 import C, OPPOSITE, W_EXACT, equilibrium, macroscopic
from .solver import collide_bgk, stream

__all__ = [
    "CaseSpec",
    "CASE_REGISTRY",
    "available_cases",
    "get_case",
    "run_case",
    "analytic_reference",
    "record_digest",
]

# ---------------------------------------------------------------------------
# case description
# ---------------------------------------------------------------------------

_KINDS = ("channel_poiseuille", "taylor_green_2d")


@dataclass(frozen=True)
class CaseSpec:
    """Everything needed to reproduce one example run.

    ``digest`` is a hash of the canonical JSON of the spec, so it can be
    quoted in a report as the identity of the arm that was run.
    """

    kind: str
    ny: int
    nx: int
    tau: float
    steps: int
    body_force: float = 0.0
    u0: float = 0.0
    dtype: str = "float64"
    steady_tol: float = 1e-9
    steady_window: int = 200
    sample_every: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown case kind {self.kind!r}; known: {_KINDS}")
        if self.ny < 4 or self.nx < 4:
            raise ValueError(f"lattice too small: ny={self.ny} nx={self.nx}")
        if not (self.tau > 0.5):
            raise ValueError(f"tau must be > 0.5 (stability/finite-nu), got {self.tau}")
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.kind == "channel_poiseuille" and self.body_force <= 0.0:
            raise ValueError("channel_poiseuille needs body_force > 0")
        if self.kind == "taylor_green_2d" and not (0.0 < self.u0 < 0.2):
            raise ValueError(
                f"taylor_green_2d needs 0 < u0 < 0.2 (compressibility), got {self.u0}"
            )

    # -- derived lattice quantities ------------------------------------
    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def wall_height(self) -> int:
        """Distance between the two no-slip planes, in lattice units.

        Half-way bounce-back puts the no-slip planes at ``y = 1/2`` and
        ``y = ny - 3/2`` (rows 0 and ``ny-1`` are the wall rows), so the
        channel width is ``ny - 2``, not ``ny - 1``.
        """
        return self.ny - 2

    @property
    def k(self) -> float:
        """Fundamental wave number of the Taylor-Green mode."""
        return 2.0 * math.pi / self.nx

    @property
    def dtype_(self) -> torch.dtype:
        return {"float64": torch.float64, "float32": torch.float32}[self.dtype]

    def canonical(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("name", None)  # the label is not part of the physical identity
        return d

    def digest(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


def _spec(kind: str, name: str, **kw: Any) -> CaseSpec:
    return CaseSpec(kind=kind, name=name, **kw)


CASE_REGISTRY: dict[str, Callable[..., CaseSpec]] = {
    # Force-driven plane channel.  ny=48 puts u_max ~ 0.05 at g=1e-5
    # (nu=0.1) which keeps Ma = u_max/cs ~ 0.09 -- inside the incompressible
    # band, while H^2/nu = 2209 sets the viscous spin-up time.
    "channel_poiseuille": lambda **kw: replace(
        _spec(
            "channel_poiseuille",
            "channel_poiseuille",
            ny=48,
            nx=8,
            tau=0.8,
            steps=12000,
            body_force=1e-5,
            steady_window=200,
            sample_every=500,
        ),
        **kw,
    ),
    # Periodic Taylor-Green vortex: periodic in both directions, so the
    # closed-form decay rate is directly observable.
    "taylor_green_2d": lambda **kw: replace(
        _spec(
            "taylor_green_2d",
            "taylor_green_2d",
            ny=64,
            nx=64,
            tau=0.8,
            steps=2000,
            u0=0.05,
            sample_every=50,
        ),
        **kw,
    ),
}


def available_cases() -> list[str]:
    return sorted(CASE_REGISTRY)


def get_case(name: str, **overrides: Any) -> CaseSpec:
    """Look up a registered case and apply ``overrides`` (frozen -> replace)."""
    if name not in CASE_REGISTRY:
        raise KeyError(f"unknown example case {name!r}; available: {available_cases()}")
    return CASE_REGISTRY[name](**overrides)


# ---------------------------------------------------------------------------
# analytic references
# ---------------------------------------------------------------------------


def analytic_reference(spec: CaseSpec) -> dict[str, float]:
    """Closed-form reference quantities for ``spec`` (no simulation)."""
    if spec.kind == "channel_poiseuille":
        H = float(spec.wall_height)
        u_max = spec.body_force * H * H / (8.0 * spec.nu)
        return {
            "u_max": u_max,
            "nu": spec.nu,
            "H": H,
            "ma": u_max / (1.0 / math.sqrt(3.0)),
            "re_H": u_max * H / spec.nu,
        }
    k = spec.k
    return {
        "decay_rate": 4.0 * spec.nu * k * k,
        "velocity_decay_rate": 2.0 * spec.nu * k * k,
        "lap_factor": 2.0,
        "k": k,
        "nu": spec.nu,
        "ma": spec.u0 / (1.0 / math.sqrt(3.0)),
        "re_lambda": spec.u0 / (spec.nu * k),
    }


# ---------------------------------------------------------------------------
# lattice construction / single step
# ---------------------------------------------------------------------------


def _initial_state(spec: CaseSpec) -> torch.Tensor:
    ny, nx = spec.ny, spec.nx
    dev = torch.device("cpu")
    dt = spec.dtype_
    rho = torch.ones((ny, nx), dtype=dt, device=dev)
    if spec.kind == "channel_poiseuille":
        ux = torch.zeros((ny, nx), dtype=dt, device=dev)
        uy = torch.zeros((ny, nx), dtype=dt, device=dev)
    else:
        xs = torch.arange(nx, dtype=dt, device=dev).view(1, nx)
        ys = torch.arange(ny, dtype=dt, device=dev).view(ny, 1)
        kx = spec.k * xs
        ky = spec.k * ys
        ux = spec.u0 * torch.cos(kx) * torch.sin(ky)
        uy = -spec.u0 * torch.sin(kx) * torch.cos(ky)
    return equilibrium(rho, ux, uy)


def _wall_mask(spec: CaseSpec, dtype: torch.dtype) -> torch.Tensor:
    m = torch.zeros((spec.ny, spec.nx), dtype=torch.bool)
    m[0, :] = True
    m[-1, :] = True
    return m


def _inward_masks(spec: CaseSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """(mask_open, mask_closed) picking the directions to reflect.

    A fluid node reflects direction ``i`` (i.e. takes its post-collision
    population from ``OPPOSITE[i]``) when the node shifted by ``-c_i`` is a
    wall.  Only the two wall rows exist, so only the rows next to them can be
    closed; this keeps the reflection explicit instead of hiding it inside a
    masked scatter.
    """
    ny, nx = spec.ny, spec.nx
    open_ = torch.zeros((9, ny, nx), dtype=torch.bool)
    closed = torch.zeros((9, ny, nx), dtype=torch.bool)
    for i in range(9):
        cy = int(C[i, 1])
        if cy == 0:
            open_[i, :, :] = True
            continue
        ys = torch.arange(ny).view(ny, 1).expand(ny, nx) - cy
        valid = (ys >= 0) & (ys <= ny - 1)
        wall = valid & ((ys == 0) | (ys == ny - 1))
        closed[i] = wall
        open_[i] = ~wall
    return open_, closed


def _guo_forcing(
    ux: torch.Tensor, uy: torch.Tensor, spec: CaseSpec, fx: float, fy: float
) -> torch.Tensor:
    """Full Guo (2002) source term ``S_i`` for a constant body force.

    ``S_i = (1 - 1/(2 tau)) w_i [ (c_i - u)/cs^2 + (c_i.u) c_i / cs^4 ] . F``
    with ``cs^2 = 1/3``.  Both halves of the bracket are required: the
    ``(c_i.u)c_i`` part is what makes ``sum_i S_i = 0`` (no spurious mass
    source) while ``sum_i c_i S_i = (1 - 1/(2 tau)) F`` supplies the momentum.
    Dropping it leaves a mass sink of order ``g u / nu`` -- measured as a
    uniform 2.5e-3 density deficit after 12k steps in the channel case.
    """
    dtype, device = ux.dtype, ux.device
    c = C.to(device).to(dtype)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    w = torch.tensor(W_EXACT, dtype=dtype, device=device).view(9, 1, 1)
    ux_ = ux.unsqueeze(0)
    uy_ = uy.unsqueeze(0)
    pre = (1.0 - 0.5 / spec.tau) * w
    cu = cx * ux_ + cy * uy_
    return pre * (
        3.0 * ((cx - ux_) * fx + (cy - uy_) * fy)
        + 9.0 * cu * (cx * fx + cy * fy)
    )


def _collide(f: torch.Tensor, spec: CaseSpec) -> torch.Tensor:
    """BGK collision (+ Guo source when a body force is present)."""
    rho, ux, uy = macroscopic(f)
    if spec.body_force:
        rho_safe = torch.clamp(rho, min=1e-12)
        ux = ux + 0.5 * spec.body_force / rho_safe
        feq = equilibrium(rho, ux, uy)
        return (
            f - (f - feq) / spec.tau
            + _guo_forcing(ux, uy, spec, spec.body_force, 0.0)
        )
    return collide_bgk(f, spec.tau)


def _step(
    f: torch.Tensor,
    spec: CaseSpec,
    open_: torch.Tensor,
    closed: torch.Tensor,
) -> torch.Tensor:
    """One LBM time step: collide+force -> stream -> half-way bounce-back.

    The collision is plain BGK on ``f^eq(rho, u + F/(2 rho))`` -- identical to
    :func:`tensorlbm.solver.collide_bgk` when ``body_force == 0`` (asserted in
    ``tests/test_example_harness.py``), so the runtime cannot drift from the
    library kernel.
    """
    f_post = _collide(f, spec)
    out = stream(f_post)
    if spec.kind == "channel_poiseuille":
        # half-way bounce-back: reflected population is the post-collision
        # opposite one, placed back on the fluid node.
        for i in range(1, 9):
            if bool(closed[i].any()):
                reflect = f_post[OPPOSITE[i]][closed[i]]
                out[i][closed[i]] = reflect
    return out


def _velocity(f: torch.Tensor, spec: CaseSpec) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rho, ux, uy = macroscopic(f)
    if spec.body_force:
        # second-order (Guo) velocity: half of the force is already in flight
        ux = ux + 0.5 * spec.body_force / torch.clamp(rho, min=1e-12)
    return rho, ux, uy


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _provenance(spec: CaseSpec) -> dict[str, Any]:
    import tensorlbm

    return {
        "spec_digest": spec.digest(),
        "tensorlbm_version": getattr(tensorlbm, "__version__", "unknown"),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(),
    }


def _field_digest(u: torch.Tensor) -> str:
    return hashlib.sha256(u.detach().contiguous().numpy().tobytes()).hexdigest()


def run_case(spec: CaseSpec, keep_history: bool = True) -> dict[str, Any]:
    """Run ``spec`` and return a JSON-serialisable run record.

    The record is the artifact of the example: it carries the spec, its
    digest, the analytic reference, the measured KPIs, a field digest for
    bit-level reproducibility and (optionally) a coarse time history.
    """
    f = _initial_state(spec)
    open_, closed = _inward_masks(spec)
    ny, nx = spec.ny, spec.nx

    ref = analytic_reference(spec)
    history: list[dict[str, float]] = []
    converged_at: int | None = None
    prev_snapshot: torch.Tensor | None = None
    t0 = __import__("time").time()

    rho, ux, uy = _velocity(f, spec)
    u_prev = ux.clone()
    for step in range(1, spec.steps + 1):
        f = _step(f, spec, open_, closed)
        if keep_history and spec.sample_every and step % spec.sample_every == 0:
            r, u, v = _velocity(f, spec)
            sample = {"step": float(step)}
            if spec.kind == "taylor_green_2d":
                sample["kinetic_energy"] = float(
                    (0.5 * (u * u + v * v)).mean().item()
                )
            else:
                sample["u_max"] = float(u.abs().max().item())
                sample["mass_drift"] = float((r.mean() - 1.0).item())
            history.append(sample)
        if spec.steady_window and step % spec.steady_window == 0:
            r, u, v = _velocity(f, spec)
            if spec.kind == "channel_poiseuille":
                denom = max(float(u.abs().max().item()), 1e-30)
                delta = float((u - u_prev).abs().max().item()) / denom
                if delta < spec.steady_tol:
                    converged_at = step
                    u_prev = u.clone()
                    break
            u_prev = u.clone()
    elapsed = __import__("time").time() - t0

    rho, ux, uy = _velocity(f, spec)
    kpis: dict[str, Any] = {
        "steps_run": step,
        "elapsed_s": elapsed,
        "mlups": (ny * nx * step) / max(elapsed, 1e-12) / 1e6,
        "rho_min": float(rho.min().item()),
        "rho_max": float(rho.max().item()),
        "finite": bool(torch.isfinite(f).all().item()),
        "field_digest": _field_digest(ux),
    }
    if spec.kind == "channel_poiseuille":
        fluid = slice(1, ny - 1)
        u_center = ux[ny // 2, 0]
        kpis["u_max"] = float(u_center.abs().item())
        kpis["u_max_global"] = float(ux[fluid].abs().max().item())
        # analytic profile on the measured rows (half-way wall at y=1/2)
        # node j sits at y = j; the no-slip planes are at y = 1/2 and
        # y = ny - 3/2, so the parabolic reference is
        # u(y) = g/(2 nu) (y - 1/2) (ny - 3/2 - y)
        y = torch.arange(ny, dtype=spec.dtype_)[fluid]
        prof_ref = spec.body_force / (2.0 * spec.nu) * (y - 0.5) * (ny - 1.5 - y)
        prof_got = ux[fluid, 0]
        kpis["profile_rel_l2"] = float(
            (torch.linalg.norm(prof_got - prof_ref) / torch.linalg.norm(prof_ref)).item()
        )
        kpis["profile_maxabs_rel"] = float(
            ((prof_got - prof_ref).abs().max() / prof_ref.abs().max()).item()
        )
        # interior rows only: the two rows next to the walls carry the
        # first-order half-way bounce-back error and must not be allowed to
        # hide (or to define) the bulk accuracy.
        inner = slice(2, ny - 2)
        kpis["profile_interior_rel_l2"] = float(
            (
                torch.linalg.norm(ux[inner, 0] - prof_ref[1:-1])
                / torch.linalg.norm(prof_ref[1:-1])
            ).item()
        )
        kpis["mass_drift"] = float((rho[fluid].mean() - 1.0).item())
        kpis["converged_at"] = converged_at
        kpis["relative_error_vs_analytic"] = abs(
            kpis["u_max"] - ref["u_max"]
        ) / ref["u_max"]
    else:
        energy = 0.5 * (ux * ux + uy * uy)
        # fit ln E over the sampled history: least squares on (step, ln E)
        if history:
            ts = [h["step"] for h in history]
            ls = [math.log(max(h["kinetic_energy"], 1e-300)) for h in history]
            n = len(ts)
            mt = sum(ts) / n
            ml = sum(ls) / n
            num = sum((t - mt) * (l - ml) for t, l in zip(ts, ls))
            den = sum((t - mt) ** 2 for t in ts)
            rate = -num / den if den else float("nan")
            kpis["energy_density"] = float(energy.mean().item())
            kpis["decay_rate_measured"] = rate
            kpis["relative_error_vs_analytic"] = abs(rate - ref["decay_rate"]) / ref[
                "decay_rate"
            ]

    profile_out: list[float] | None = None
    if spec.kind == "channel_poiseuille":
        profile_out = [float(v) for v in ux[1:-1, 0]]

    return {
        "case": spec.name or spec.kind,
        "profile": profile_out,
        "spec": spec.canonical(),
        "analytic": ref,
        "kpis": kpis,
        "provenance": _provenance(spec),
        "history": history,
    }


def record_digest(record: Mapping[str, Any]) -> str:
    """Stable digest of a run record's numeric payload (excludes wall clock)."""
    payload = {
        "spec": record["spec"],
        "analytic": record["analytic"],
        "kpis": {k: v for k, v in record["kpis"].items() if k not in ("elapsed_s", "mlups", "steps_run")},
        "history": record.get("history", []),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(blob.encode()).hexdigest()

# ---------------------------------------------------------------------------
# bridge: declarative instruction-DSL programs (XVAL-LBM-INSTRUCTION-DSL-TESTS-SH)
# ---------------------------------------------------------------------------
def run_instruction_dsl_program(program, *, out=None, sample_every=0):
    """Execute a declarative instruction-DSL program and return its run record.

    ``program`` is either instruction text (contains a newline or starts with
    ``stencil``) or a path to a ``*.lbm`` file.  The heavy lifting stays in
    :mod:`tensorlbm.instruction_dsl`; this wrapper exists so the example
    runtime (and therefore every tool built on it) can drive a DSL program
    without importing the internals.
    """
    import json as _json
    from pathlib import Path as _Path

    from .instruction_dsl import parse, run

    is_path = ("\n" not in str(program)
               and not str(program).lstrip().startswith("stencil"))
    text = _Path(program).read_text(encoding="utf-8") if is_path else str(program)
    prog = parse(text)
    res = run(prog, sample_every=sample_every)
    record = dict(res.summary())
    record["program_digest"] = prog.digest
    record["program_path"] = str(program) if is_path else None
    record["domain"] = {"nx": prog.nx, "ny": prog.ny}
    if out:
        _Path(out).parent.mkdir(parents=True, exist_ok=True)
        _Path(out).write_text(_json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record
