"""Solver-in-the-loop closure calibration on the differentiable bounded path.

This module productises the B3 step of the roadmap: *data -> closure ->
solver*.  The A6 differentiable reference path (:mod:`tensorlbm.autograd_path`)
gives gradients through the solver; A6+ added velocity-inlet / zero-gradient-
outlet boundaries so a rollout represents a bounded campaign point; what was
still manual (``examples/solver_in_the_loop.py``) is the calibration loop
itself.  Here a *closure* — the Smagorinsky constant, constant or
Re-dependent — is identified from drag observations at several Reynolds
numbers by gradient descent through bounded rollouts, and evaluated at
held-out Re.

The observable is the windowed momentum-exchange drag coefficient

.. math:: C_D(Re) = \\frac{\\langle F_x \\rangle_{\\text{window}}}
    {\\tfrac{1}{2}\\,\\rho_0\\,u_{in}^2\\,A}

with :math:`A = \\pi r^2` the sphere's projected area, evaluated on the
post-stream / pre-bounce-back probe (the production sampling phase).  Every
part of the forward map — inlet condition, LES collision, streaming, drag
sampling — carries autograd gradients, so the closure parameters are trained
by plain backprop.

Typical use — recover a power-law closure from synthetic campaign data::

    box = BoxCase(nz=16, ny=16, nx=40, steps=300)
    truth = cs_power(0.12, -0.25, re_ref=16.0)
    targets = synthetic_targets(box, re_values=(8, 16, 32), closure=truth)
    result = calibrate(targets, box, kind="power")
    result.eval  # per-Re predicted vs target C_D, including held-out Re

The synthetic-observation route (ground truth produced by the same solver)
is the *verification* mode: the recoverable optimum is known exactly.  Real
campaign observations enter the same way as :class:`DragTarget` rows:

* :class:`HullCase` swaps the calibration sphere for the *production* SUBOFF
  voxel mask (:func:`tensorlbm.suboff_cad.build_suboff_mask`, same
  placement/length as the ``suboff_n128`` case), keeps the production tau
  relation ``tau = 0.5 + 3 u_in L / Re`` with ``L = 0.6 nx`` and closes the
  lateral planes with free-stream faces like the campaign case, so the
  calibration rollout differs from the measured campaign point only in the
  collision model (Smagorinsky-BGK here vs cumulant in production);
* :func:`drag_targets_from_sidecars` turns ``drag_history.json`` sidecars
  (``tensorlbm.drag-history/v1``, written by ``tensorlbm.scan_drag``) into
  ``DragTarget`` rows with the same ``2F/(rho u^2 S_proj)`` normalisation
  the campaign report uses.

The honest caveat for real data: model error (Smagorinsky-BGK absorbing
what production does with a cumulant collision) folds into the identified
constant — the closure reproduces the measured drag, not necessarily the
true sub-grid physics.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tensorlbm.autograd_path import InletSpec, OutletSpec, WallSpec, obstacle_force, rollout
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.scan_drag import DRAG_HISTORY_SCHEMA
from tensorlbm.solver3d import collide_mrt3d
from tensorlbm.turbulence import (
    collide_smagorinsky_bgk3d,
    collide_smagorinsky_mrt3d,
    collide_wale_bgk3d,
    collide_wale_mrt3d,
)

__all__ = [
    "BoxCase",
    "CUMULANT27_RATES",
    "COLLISION27_FAMILIES",
    "CalibResult",
    "Collision27CalibResult",
    "DragHistory",
    "DragTarget",
    "HullCase",
    "MRT27_RATES",
    "SGS_MODELS",
    "CAMPAIGN_SEMANTICS",
    "CampaignCollision27CalibResult",
    "CampaignObservables",
    "CampaignRateCalibResult",
    "CampaignSemantics",
    "LEGACY27_SEMANTICS",
    "LEGACY_SEMANTICS",
    "bounded_drag",
    "calibrate_collision27_campaign",
    "calibrate_mrt_rates_campaign",
    "campaign_chain19",
    "campaign_chain27",
    "campaign_rollout19",
    "campaign_rollout27",
    "press_profile27_campaign",
    "press_profile_campaign",
    "calibrate",
    "calibrate_collision27",
    "cd_from_force",
    "collide_cumulant27_diffable",
    "cs_power",
    "drag_targets_from_sidecars",
    "evaluate",
    "load_drag_history",
    "obstacle_force27",
    "press_profile",
    "press_profile27",
    "rate_fd_response",
    "rate_fd_response27",
    "rollout27",
    "step27",
    "synthetic_targets",
    "windowed_cd",
]

#: Callable mapping a Reynolds number (float or 0-dim tensor) to a Smagorinsky
#: constant (float or 0-dim tensor) — the closure being calibrated.
ClosureFn = Callable[[Any], Any]


def cs_power(c0: float, b: float, re_ref: float = 1.0) -> ClosureFn:
    """Power-law closure ``C_s(Re) = c0 · (Re/re_ref)^b``."""

    def closure(re: Any) -> Any:
        return c0 * (re / re_ref) ** b

    return closure


@dataclass(frozen=True)
class BoxCase:
    """The bounded calibration domain: a sphere in a slip-free campaign box.

    The sphere sits on the x axis at ``cx = 0.3·nx`` (SUBOFF convention),
    centred laterally.  ``tau`` follows the house Reynolds relation
    ``tau = 0.5 + 3·u_in·D/Re`` with ``D = 2r``; keep ``tau`` comfortably
    above 0.5 across the swept Re range (BGK stability).
    """

    nz: int
    ny: int
    nx: int
    radius: int = 4
    u_in: float = 0.15
    steps: int = 300
    window_start: int = 200
    inlet_method: str = "equilibrium"
    wall_method: str = "periodic"
    checkpoint: bool = False
    checkpoint_block: int = 1
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.window_start >= self.steps:
            raise ValueError("window_start must be < steps")
        if self.inlet_method not in ("equilibrium", "zouhe"):
            raise ValueError("inlet_method must be 'equilibrium' or 'zouhe'")
        _check_wall_method(self.wall_method)
        if self.checkpoint_block < 1:
            raise ValueError("checkpoint_block must be >= 1")
        if 2 * self.radius >= min(self.nz, self.ny):
            raise ValueError("sphere must fit laterally: radius < min(nz, ny)/2")

    def make_mask(self) -> torch.Tensor:
        """Boolean solid mask of shape ``(nz, ny, nx)``."""
        device = torch.device(self.device)
        zs, ys, xs = (
            torch.arange(n, device=device, dtype=torch.float64) for n in (self.nz, self.ny, self.nx)
        )
        Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")
        cx, cy, cz = 0.3 * self.nx, self.ny / 2, self.nz / 2
        return (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= self.radius**2

    def tau_of_re(self, re: float) -> torch.Tensor:
        """Relaxation time for the house relation (returns a graph-leaf)."""
        return torch.tensor(
            0.5 + 3.0 * self.u_in * (2.0 * self.radius) / re,
            dtype=self.dtype,
            device=torch.device(self.device),
        )

    def ref_area(self, mask: torch.Tensor | None = None) -> float:
        """C_D normalisation area: the sphere's projected area ``pi r^2``."""
        return math.pi * self.radius**2


def _check_wall_method(method: str) -> None:
    if method not in ("periodic", "free-slip", "freestream"):
        raise ValueError("wall_method must be 'periodic', 'free-slip' or 'freestream'")


#: Available collision families.  ``"bgk"`` is the historical calibration
#: path; ``"mrt"`` (:func:`tensorlbm.solver3d.collide_mrt3d` rates) brackets
#: production drag that sits below the BGK zero-SGS floor.
COLLISION_FAMILIES = ("bgk", "mrt")

#: Available sub-grid models.  ``"smagorinsky"`` reads the closure constant
#: as ``C_s``; ``"wale"`` reads it as ``C_w`` (near-wall vanishing eddy
#: viscosity, so a different C_D response — and identifiability — at the
#: same tau).
SGS_MODELS = ("smagorinsky", "wale")


def _check_families(collision: str, sgs: str) -> None:
    if collision not in COLLISION_FAMILIES:
        raise ValueError(f"collision must be one of {COLLISION_FAMILIES}, got {collision!r}")
    if sgs not in SGS_MODELS:
        raise ValueError(f"sgs must be one of {SGS_MODELS}, got {sgs!r}")


def _make_collide(
    cs: float | torch.Tensor | None,
    collision: str,
    sgs: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Callable[..., torch.Tensor] | None:
    """Collision operator for the (collision family, SGS model) axes.

    ``cs=None`` means *no* sub-grid model: plain ``collide_bgk3d`` (the
    historical default, returned as ``None`` so the rollout's fast path is
    bit-for-bit unchanged) or ``collide_mrt3d`` under ``collision="mrt"``.
    A non-None *cs* selects the SGS kernel — the constant lands in ``C_s``
    for Smagorinsky and ``C_w`` for WALE — as a 0-dim tensor of *dtype* on
    *device* so it stays in the autograd graph.
    """
    _check_families(collision, sgs)
    if cs is None:
        if collision == "bgk":
            return None
        return lambda f, t: collide_mrt3d(f, t)
    const = torch.as_tensor(cs, dtype=dtype, device=device)
    if collision == "bgk":
        if sgs == "smagorinsky":
            return lambda f, t, _c=const: collide_smagorinsky_bgk3d(f, t, C_s=_c)
        return lambda f, t, _c=const: collide_wale_bgk3d(f, t, C_w=_c)
    if sgs == "smagorinsky":
        return lambda f, t, _c=const: collide_smagorinsky_mrt3d(f, t, C_s=_c)
    return lambda f, t, _c=const: collide_wale_mrt3d(f, t, C_w=_c)


@dataclass(frozen=True)
class HullCase:
    """Bounded calibration domain: the *production* SUBOFF hull voxel mask.

    The real-geometry counterpart of :class:`BoxCase` for calibrating
    against campaign drag data.  The mask comes from the production voxel
    builder (:func:`tensorlbm.suboff_cad.build_suboff_mask`) with the exact
    ``suboff_n128`` placement — hull centred at ``cx = 0.35·nx``, length
    ``0.6·nx``, laterally centred, ``hull_type``/``sail_scale``/``fin_scale``
    the campaign geometry axis (bare hull, 1.0/1.0 in the 2026-08-21 drag
    dataset) — and ``tau`` follows the *production* relation
    ``tau = 0.5 + 3·u_in·L/Re`` with ``L = 0.6·nx`` (at ``u_in = 0.1``,
    ``n128``: ``tau = 0.5 + 23.04/Re``), so a rollout matches the campaign
    point's viscosity exactly.  The lateral planes default to free-stream
    faces, the production far-field condition.

    The C_D normalisation is the *projected* area of the actual mask — the
    ``(y, z)`` columns containing at least one solid cell, the same
    convention as ``tensorlbm.drag_survey.projected_area`` used on the data
    side (69 cells² for the n128 bare hull; note ``pi·r_max²`` would give
    63.1, a 9% bookkeeping bias).  Using the projection of the very mask
    that is simulated makes the two sides agree by construction, whatever
    the geometry scales.

    Defaults follow the measured convergence of the campaign drag histories
    (``docs/closure_calibration.md``, "Real observations"): windowed C_D at
    ``[1000, 1200)`` sits within 0.3–0.5% of the 4000-step tail mean across
    the whole Re 50–800 sweep, so ``steps=1200`` / ``window_start=1000``
    buys a ~3x shorter rollout for a sub-percent systematic offset.
    """

    nz: int
    ny: int
    nx: int
    u_in: float = 0.1
    hull_length: float | None = None
    steps: int = 1200
    window_start: int = 1000
    inlet_method: str = "equilibrium"
    wall_method: str = "freestream"
    hull_type: str = "bare_hull"
    sail_scale: float = 1.0
    fin_scale: float = 1.0
    checkpoint: bool = True
    checkpoint_block: int = 25
    dtype: torch.dtype = torch.float32
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.window_start >= self.steps:
            raise ValueError("window_start must be < steps")
        if self.inlet_method not in ("equilibrium", "zouhe"):
            raise ValueError("inlet_method must be 'equilibrium' or 'zouhe'")
        _check_wall_method(self.wall_method)
        if self.checkpoint_block < 1:
            raise ValueError("checkpoint_block must be >= 1")
        if self.u_in <= 0.0:
            raise ValueError("u_in must be positive")
        if self.hull_length is None:
            object.__setattr__(self, "hull_length", 0.6 * self.nx)
        elif self.hull_length <= 0.0:
            raise ValueError("hull_length must be positive")

    def make_mask(self) -> torch.Tensor:
        """Production SUBOFF voxel mask of shape ``(nz, ny, nx)``."""
        from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask

        nx, ny, nz = self.nx, self.ny, self.nz
        mask, _stats = build_suboff_mask(
            hull_type=self.hull_type,
            nx=nx,
            ny=ny,
            nz=nz,
            cx=nx * 0.35,
            cy=ny / 2.0,
            cz=nz / 2.0,
            length=self.hull_length,
            config=SuboffConfig(sail_scale=self.sail_scale, fin_scale=self.fin_scale),
            device=self.device,
        )
        return mask

    def tau_of_re(self, re: float) -> torch.Tensor:
        """Production relation ``tau = 0.5 + 3·u_in·L/Re`` (graph leaf)."""
        return torch.tensor(
            0.5 + 3.0 * self.u_in * self.hull_length / re,
            dtype=self.dtype,
            device=torch.device(self.device),
        )

    def ref_area(self, mask: torch.Tensor | None = None) -> float:
        """Projected area ``(y, z)`` columns with solid — the data-side convention."""
        if mask is None:
            mask = self.make_mask()
        return float(mask.any(dim=2).sum())


@dataclass(frozen=True)
class DragTarget:
    """One drag observation: a Reynolds number and its measured C_D."""

    re: float
    cd: float
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Real observations: drag_history sidecars -> targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DragHistory:
    """A ``drag_history.json`` sample sequence (``tensorlbm.drag-history/v1``).

    ``steps`` are the solver steps at which the campaign observer sampled;
    ``force`` holds the ``(force_x, force_y, force_z)`` triple per sample in
    lattice units (momentum per step; see :mod:`tensorlbm.scan_drag`).
    """

    steps: np.ndarray
    force: np.ndarray

    def __post_init__(self) -> None:
        if self.steps.ndim != 1 or self.force.shape != (len(self.steps), 3):
            raise ValueError(
                f"DragHistory needs steps (n,) and force (n, 3), got "
                f"{self.steps.shape} / {self.force.shape}"
            )


def load_drag_history(path: str | Path) -> DragHistory:
    """Read one ``drag_history.json`` sidecar (schema-checked).

    Raises ``ValueError`` on any other schema tag, so a future sidecar
    format cannot silently feed a mis-reduced drag into a calibration.
    """
    data = json.loads(Path(path).read_text())
    schema = data.get("schema")
    if schema != DRAG_HISTORY_SCHEMA:
        raise ValueError(f"{path}: unsupported drag-history schema {schema!r}")
    samples = data.get("samples")
    if not samples:
        raise ValueError(f"{path}: drag history has no samples")
    steps = np.asarray([s["step"] for s in samples], dtype=np.int64)
    force = np.asarray([[s["force_x"], s["force_y"], s["force_z"]] for s in samples], dtype=float)
    return DragHistory(steps=steps, force=force)


def cd_from_force(force_x: float, u_in: float, ref_area: float, rho0: float = 1.0) -> float:
    """Campaign C_D normalisation ``2F / (rho0 u_in^2 A)`` (lattice units).

    The exact form used on the data side (``docs/benchmarks/
    suboff_cd_re_20260821.md``); :func:`bounded_drag` divides by
    ``0.5 rho u^2 A`` — the same number.
    """
    return 2.0 * force_x / (rho0 * u_in**2 * ref_area)


def windowed_cd(
    history: DragHistory,
    start_step: int,
    end_step: int,
    u_in: float,
    ref_area: float,
    rho0: float = 1.0,
) -> float:
    """C_D of the samples in ``start_step <= step < end_step`` (half-open).

    The convergence-probe reducer: windowed means of the same history at
    increasing offsets answer "how long a rollout does drag need" directly
    from campaign data (see "Real observations" in ``docs/
    closure_calibration.md``).
    """
    sel = (history.steps >= start_step) & (history.steps < end_step)
    if not sel.any():
        raise ValueError(f"no drag samples in [{start_step}, {end_step})")
    return cd_from_force(float(history.force[sel, 0].mean()), u_in, ref_area, rho0)


def drag_targets_from_sidecars(
    sidecars: Iterable[str | Path],
    *,
    u_in: float,
    ref_area: float,
    rho0: float = 1.0,
    tail_frac: float = 0.25,
    re_values: Sequence[float] | None = None,
) -> list[DragTarget]:
    """Turn campaign ``drag_history.json`` sidecars into calibration targets.

    Each sidecar is reduced exactly like the campaign report: tail mean of
    ``force_x`` over the last *tail_frac* of its samples (40 of 160 samples
    = steps 3025–4000 at the survey interval 25), then
    ``C_D = 2F/(rho0 u_in^2 ref_area)``.  ``ref_area`` must be the projected
    area of the campaign mask (69 cells² for ``suboff_n128`` bare hull;
    :meth:`HullCase.ref_area` computes it from the same mask).

    The sidecar itself carries no Reynolds number; *re* is taken either
    from *re_values* (one per sidecar) or, by default, from the ``params.re``
    field of the sibling ``status.json`` written by ``scan_runner`` next to
    every ``drag_history.json``.
    """
    paths = [Path(p) for p in sidecars]
    if re_values is None:
        res = []
        for p in paths:
            status = p.parent / "status.json"
            if not status.is_file():
                raise ValueError(f"{p}: no re_values given and no sibling {status} to read Re from")
            res.append(float(json.loads(status.read_text())["params"]["re"]))
    else:
        res = [float(re) for re in re_values]
        if len(res) != len(paths):
            raise ValueError(f"re_values has {len(res)} entries for {len(paths)} sidecars")
    if not 0.0 < tail_frac <= 1.0:
        raise ValueError(f"tail_frac must be in (0, 1], got {tail_frac}")

    targets = []
    for p, re in zip(paths, res):
        history = load_drag_history(p)
        n_tail = max(1, int(len(history.steps) * tail_frac))
        tail = float(history.force[-n_tail:, 0].mean())
        targets.append(DragTarget(re=re, cd=cd_from_force(tail, u_in, ref_area, rho0)))
    return targets


def _case_mask(box: BoxCase | HullCase) -> torch.Tensor:
    return box.make_mask()


def bounded_drag(
    box: BoxCase | HullCase,
    re: float,
    cs: float | torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    *,
    collision: str = "bgk",
    sgs: str = "smagorinsky",
) -> torch.Tensor:
    """One differentiable bounded rollout; returns the scalar windowed C_D.

    The full map — inlet Dirichlet value, (optional) SGS collision with
    constant *cs*, streaming, momentum-exchange drag — is autograd-
    connected, so gradients flow to ``cs`` (tensor) and, when the inlet
    velocity is a tensor, through the boundary condition as well.

    Works for any case object providing the interface both case classes
    share: ``make_mask``/``tau_of_re``/``ref_area(mask)`` plus the rollout
    fields (``u_in``, ``steps``, ``window_start``, ``inlet_method``,
    ``wall_method``, ``checkpoint``, ``dtype``, ``device``) — i.e.
    :class:`BoxCase` (sphere, periodic sides) and :class:`HullCase`
    (production SUBOFF mask, free-stream sides).

    Args:
        box: Domain and rollout configuration.
        re: Reynolds number (sets ``tau`` via the case's relation).
        cs: Sub-grid constant (Smagorinsky ``C_s`` or WALE ``C_w``
            depending on *sgs*); a 0-dim tensor stays in the graph.
            ``None`` runs the collision family without any sub-grid model
            — the zero-SGS "floor".
        mask: Pre-built solid mask (defaults to ``box.make_mask()``).
        collision: Collision family, ``"bgk"`` (historical default) or
            ``"mrt"`` (:func:`tensorlbm.solver3d.collide_mrt3d` rates).
        sgs: Sub-grid model the constant *cs* feeds, ``"smagorinsky"``
            (historical default) or ``"wale"``.

    Returns:
        0-dim tensor: mean x-force over the window ``[window_start, steps)``
        normalised by ``0.5·rho0·u_in²·A`` with ``A = box.ref_area(mask)``.
    """
    device = torch.device(box.device)
    mask = _case_mask(box) if mask is None else mask
    tau = box.tau_of_re(re)
    inlet = InletSpec(
        ux=torch.as_tensor(box.u_in, dtype=box.dtype, device=device),
        method=box.inlet_method,
    )
    walls = None if box.wall_method == "periodic" else WallSpec(method=box.wall_method, ux=box.u_in)
    collide = _make_collide(cs, collision, sgs, box.dtype, device)

    rho0 = torch.ones((box.nz, box.ny, box.nx), dtype=box.dtype, device=device)
    u = torch.zeros((3, box.nz, box.ny, box.nx), dtype=box.dtype, device=device)
    u[0] = box.u_in
    f0 = equilibrium3d(rho0, u[0], u[1], u[2])

    # probes only from window_start: the transient probes are never
    # materialised (memory proportional to the window, not the rollout)
    if box.checkpoint and box.checkpoint_block > 1:
        force = _blocked_window_force(
            f0, box, tau, mask, collide=collide, inlet=inlet, outlet=OutletSpec(), walls=walls
        )
    else:
        _f, probes = rollout(
            f0,
            box.steps,
            tau,
            mask,
            collide=collide,
            checkpoint=box.checkpoint,
            inlet=inlet,
            outlet=OutletSpec(),
            walls=walls,
            return_probes=True,
            probe_start=box.window_start,
        )
        force = sum(obstacle_force(p, mask)[0] for p in probes) / len(probes)
    return force / (0.5 * box.u_in**2 * box.ref_area(mask))


def _blocked_window_force(
    f0: torch.Tensor,
    box: BoxCase | HullCase,
    tau: torch.Tensor,
    mask: torch.Tensor,
    *,
    collide: Callable[..., torch.Tensor] | None,
    inlet: InletSpec,
    outlet: OutletSpec,
    walls: WallSpec | None,
) -> torch.Tensor:
    """Windowed x-force with *block-level* gradient checkpointing.

    Per-step checkpointing still retains ~3 ``(19, nz, ny, nx)`` tensors per
    step (measured 37 MiB/step at ``suboff_n128`` scale, i.e. ~47 GiB for a
    1200-step rollout).  Checkpointing blocks of ``box.checkpoint_block``
    steps instead keeps one input per block alive; the probes are summed to
    a scalar *inside* each block, so they exist only during the forward and
    the backward recompute — peak memory stops growing with the rollout
    length (measured ~7 GiB for the production-scale case, any length).
    Gradients are identical to the plain rollout (checkpointing recomputes
    the exact same deterministic ops).
    """
    from torch.utils.checkpoint import checkpoint as _checkpoint

    def run_block(f: torch.Tensor, n_steps: int, probe_from: int):
        # probe_from < 0: no probes needed (pure transient block)
        if probe_from < 0:
            return rollout(
                f, n_steps, tau, mask, collide=collide, inlet=inlet, outlet=outlet, walls=walls
            ), None
        _f, probes = rollout(
            f,
            n_steps,
            tau,
            mask,
            collide=collide,
            inlet=inlet,
            outlet=outlet,
            walls=walls,
            return_probes=True,
            probe_start=probe_from,
        )
        # graph scalar: sum of per-probe forces (count kept outside)
        return _f, sum(obstacle_force(p, mask)[0] for p in probes)

    block = box.checkpoint_block
    f = f0
    force_sum = torch.zeros((), dtype=box.dtype, device=f0.device)
    n_probes = 0
    for start in range(0, box.steps, block):
        n = min(block, box.steps - start)
        if start + n <= box.window_start:  # fully inside the transient
            f = _checkpoint(run_block, f, n, -1, use_reentrant=False)[0]
        else:
            f, block_sum = _checkpoint(
                run_block, f, n, max(0, box.window_start - start), use_reentrant=False
            )
            force_sum = force_sum + block_sum
            n_probes += n - max(0, box.window_start - start)
    return force_sum / n_probes


# ---------------------------------------------------------------------------
# Field observables: what does the closure actually move? (B3-next)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClosureObservables:
    """Window-mean *field* observables of one bounded rollout (B3-next).

    Produced by :func:`bounded_observables`.  Every entry is the mean over
    the case window ``[window_start, steps)`` of quantities read on the
    post-stream / pre-bounce-back probe state — the same sampling phase as
    the drag observable — so entries are directly comparable across
    closures.  Motivation: windowed C_D is closure-blind at production
    resolution (``|d ln C_D / d ln C_s| ~ 0.004-0.03``, and the
    collision-family axis moves it only ~0.2-0.8%), while the surface
    pressure profile reads the collision axis at 5-8% with a ~33x
    signal-to-drift ratio (``docs/closure_calibration.md``, "Observable
    swap").
    """

    cd: float
    #: ``(press_bins,)`` mean ``rho - 1`` on solid-adjacent wet cells, binned axially
    press_profile: torch.Tensor
    #: ``(nx,)`` mean ``ux`` on the centreline ``(z=nz/2, y=ny/2)``
    centerline_ux: torch.Tensor
    #: per wake plane, the mean deficit ``u_in - ux`` on its wet disk cells
    wake_deficit: list[torch.Tensor]
    #: per wake plane, the mean cross-flow ``uy`` on the same cells
    wake_cross: list[torch.Tensor]
    #: mean over the window of the per-step minimum ``ux`` in the wake slab
    wake_min_ux: float
    #: the x indices of the wake planes (rows of the two wake matrices)
    planes: tuple[int, ...]
    #: cells per pressure bin (diagnostics; bins past the body are empty)
    press_counts: torch.Tensor


def _default_wake_planes(mask: torch.Tensor) -> tuple[int, ...]:
    """Geometric default probe planes: 20/45/70 % into the wake.

    The body tail is read from *mask* itself (last x with any solid cell),
    so the rule works for any case geometry.
    """
    nx = mask.shape[2]
    tail = int(mask.any(dim=(0, 1)).nonzero(as_tuple=True)[0][-1]) + 1
    span = nx - 1 - tail
    return tuple(min(nx - 2, tail + max(1, round(f * span))) for f in (0.2, 0.45, 0.7))


def _solid_adjacent_wet(mask: torch.Tensor) -> torch.Tensor:
    """Wet cells with at least one of their 6 face neighbours solid."""
    wet = ~mask
    adj = torch.zeros_like(wet)
    adj[1:] |= mask[:-1]
    adj[:-1] |= mask[1:]
    adj[:, 1:] |= mask[:, :-1]
    adj[:, :-1] |= mask[:, 1:]
    adj[:, :, 1:] |= mask[:, :, :-1]
    adj[:, :, :-1] |= mask[:, :, 1:]
    return wet & adj


@torch.no_grad()
def bounded_observables(
    box: BoxCase | HullCase,
    re: float,
    cs: float | torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    *,
    collision: str = "bgk",
    sgs: str = "smagorinsky",
    plane_xs: Sequence[int] | None = None,
    wake_radius: float = 20.0,
    press_bins: int = 32,
    chunk: int = 40,
) -> ClosureObservables:
    """One no-grad bounded rollout returning window-mean field observables.

    The same rollout configuration as :func:`bounded_drag` (identical
    inlet/outlet/walls/collision wiring; ``cs=None`` is the zero-SGS
    floor), but instead of reducing the probe states to the windowed drag
    scalar it accumulates the *fields* a closure could be identified from:

    - ``cd`` — the drag observable itself (control; known closure-blind);
    - ``press_profile`` — axial surface-pressure proxy ``rho - 1`` on
      solid-adjacent wet cells.  In lattice units ``p = c_s^2 rho``, so
      this is the gauge pressure profile ``C_p`` up to a constant.  The
      collision axis moves it 5-8% on the production hull (vs 0.2-0.8% for
      C_D) with the response steady across the window halves (drift
      ~0.002 absolute vs ~0.05 response);
    - ``centerline_ux`` — mean streamwise velocity on the centreline;
    - ``wake_deficit`` / ``wake_cross`` — ``u_in - ux`` and ``uy`` on the
      (z, y) planes at *plane_xs*, restricted to wet cells within
      *wake_radius* of the body axis.  The deficit is the smooth part of
      the wake; the cross-flow field is phase-sensitive (shedding) and
      needs long windows to be usable;
    - ``wake_min_ux`` — per-step minimum ``ux`` over the wet wake slab
      ``[tail, nx-1)``, window-averaged.  Zero means no recirculation.

    Args:
        box: Domain and rollout configuration (see :func:`bounded_drag`).
        re: Reynolds number (sets ``tau`` via the case's relation).
        cs: Sub-grid constant (or ``None`` for the zero-SGS floor).
        mask: Pre-built solid mask (defaults to ``box.make_mask()``).
        collision: Collision family, ``"bgk"`` or ``"mrt"``.
        sgs: Sub-grid model the constant *cs* feeds.
        plane_xs: Wake-plane x indices.  ``None`` picks three planes at
            20/45/70 % into the wake of the *actual* mask.
        wake_radius: Lateral radius (cells) of the wake sampling disk.
        press_bins: Number of axial bins for the pressure profile.
        chunk: Probe-processing chunk (memory: ~``chunk`` probe states).

    Returns:
        :class:`ClosureObservables` with window-mean values.
    """
    device = torch.device(box.device)
    mask = _case_mask(box) if mask is None else mask
    nz, ny, nx = box.nz, box.ny, box.nx
    cy, cz = ny // 2, nz // 2
    tau = box.tau_of_re(re)
    inlet = InletSpec(
        ux=torch.as_tensor(box.u_in, dtype=box.dtype, device=device),
        method=box.inlet_method,
    )
    walls = None if box.wall_method == "periodic" else WallSpec(method=box.wall_method, ux=box.u_in)
    collide = _make_collide(cs, collision, sgs, box.dtype, device)

    rho0 = torch.ones((nz, ny, nx), dtype=box.dtype, device=device)
    u0 = torch.zeros((3, nz, ny, nx), dtype=box.dtype, device=device)
    u0[0] = box.u_in
    f = equilibrium3d(rho0, u0[0], u0[1], u0[2])

    planes = tuple(int(x) for x in plane_xs) if plane_xs is not None else _default_wake_planes(mask)
    yy, zz = torch.meshgrid(
        torch.arange(ny, device=device), torch.arange(nz, device=device), indexing="ij"
    )
    disk = ((yy - cy) ** 2 + (zz - cz) ** 2) <= wake_radius**2
    adj = _solid_adjacent_wet(mask)
    adj_idx = adj.nonzero(as_tuple=False)
    n_bins = min(press_bins, nx)
    bin_w = max(1, nx // n_bins)
    tail = int(mask.any(dim=(0, 1)).nonzero(as_tuple=True)[0][-1]) + 1

    def process(p: torch.Tensor, acc: dict) -> None:
        rho, ux, uy, _uz = macroscopic3d(p)
        acc["fx"] += float(obstacle_force(p, mask)[0])
        prof = torch.zeros(n_bins, dtype=torch.float64, device=device)
        xs = (adj_idx[:, 2] // bin_w).long()
        vals = (rho[adj_idx[:, 0], adj_idx[:, 1], adj_idx[:, 2]] - 1.0).double()
        prof.scatter_add_(0, xs, vals)
        acc["press"] = acc["press"] + prof
        acc["cl"] = acc["cl"] + ux[cz, cy, :].double()
        for i, xp in enumerate(planes):
            wet_disk = disk & ~mask[:, :, xp].T  # both (ny, nz)
            d = (box.u_in - ux[:, :, xp]).T[wet_disk].double()
            c = uy[:, :, xp].T[wet_disk].double()
            acc["wdef"][i] = acc["wdef"][i] + d
            acc["wcross"][i] = acc["wcross"][i] + c
        wk, wm = ux[:, :, tail : nx - 1], ~mask[:, :, tail : nx - 1]
        acc["minux"] += float(wk[wm].min()) if wm.any() else 0.0

    acc: dict = {
        "fx": 0.0,
        "minux": 0.0,
        "press": torch.zeros(n_bins, dtype=torch.float64, device=device),
        "cl": torch.zeros(nx, dtype=torch.float64, device=device),
        "wdef": [0 for _ in planes],
        "wcross": [0 for _ in planes],
    }
    f = rollout(
        f,
        box.window_start,
        tau,
        mask,
        collide=collide,
        inlet=inlet,
        outlet=OutletSpec(),
        walls=walls,
    )
    n_probe = 0
    while n_probe < box.steps - box.window_start:
        n = min(chunk, box.steps - box.window_start - n_probe)
        _f, probes = rollout(
            f,
            n,
            tau,
            mask,
            collide=collide,
            inlet=inlet,
            outlet=OutletSpec(),
            walls=walls,
            return_probes=True,
            probe_start=0,
        )
        f = _f
        for p in probes:
            process(p, acc)
        n_probe += n
        del probes

    n = float(n_probe)
    press_counts = torch.zeros(n_bins, dtype=torch.float64, device=device)
    press_counts.scatter_add_(
        0,
        (adj_idx[:, 2] // bin_w).long(),
        torch.ones(len(adj_idx), dtype=torch.float64, device=device),
    )
    return ClosureObservables(
        cd=acc["fx"] / n / (0.5 * box.u_in**2 * box.ref_area(mask)),
        press_profile=acc["press"] / n,
        centerline_ux=acc["cl"] / n,
        wake_deficit=[v / n for v in acc["wdef"]],
        wake_cross=[v / n for v in acc["wcross"]],
        wake_min_ux=acc["minux"] / n,
        planes=planes,
        press_counts=press_counts,
    )


def observable_response(
    a: ClosureObservables, b: ClosureObservables, ref: ClosureObservables
) -> dict[str, float]:
    """Relative response ``||a_k - b_k|| / ||ref_k||`` per observable.

    The identifiability metric of the observable swap: for a vector
    observable the relative L2 distance between the two runs, normalised
    by the reference run's norm; for the scalars the relative absolute
    difference.  Divide by ``ln(cs_hi/cs_lo)`` to express per e-fold of a
    parameter sweep (the experiment tables in ``docs/closure_calibration.md``
    report exactly that).
    """
    out: dict[str, float] = {}

    def vec(key: str, ka: torch.Tensor, kb: torch.Tensor, kr: torch.Tensor) -> None:
        nr = float(torch.linalg.norm(kr.double()))
        out[key] = float(torch.linalg.norm(ka.double() - kb.double())) / nr if nr else 0.0

    out["cd"] = abs(a.cd - b.cd) / abs(ref.cd) if ref.cd else 0.0
    out["wake_min_ux"] = (
        abs(a.wake_min_ux - b.wake_min_ux) / abs(ref.wake_min_ux) if ref.wake_min_ux else 0.0
    )
    vec("press_profile", a.press_profile, b.press_profile, ref.press_profile)
    vec("centerline_ux", a.centerline_ux, b.centerline_ux, ref.centerline_ux)
    for i, xp in enumerate(ref.planes):
        vec(f"wake_deficit@{xp}", a.wake_deficit[i], b.wake_deficit[i], ref.wake_deficit[i])
        vec(f"wake_cross@{xp}", a.wake_cross[i], b.wake_cross[i], ref.wake_cross[i])
    return out


@torch.no_grad()
def synthetic_targets(
    box: BoxCase | HullCase,
    re_values: Sequence[float],
    closure: ClosureFn,
    *,
    collision: str = "bgk",
    sgs: str = "smagorinsky",
) -> list[DragTarget]:
    """Produce verification-mode observations with a known ground truth."""
    return [
        DragTarget(
            re=re,
            cd=float(bounded_drag(box, re, cs=closure(re), collision=collision, sgs=sgs)),
        )
        for re in re_values
    ]


def _closure_parameters(
    kind: str, cs0: float, re_ref: float, device: torch.device, dtype: torch.dtype
) -> tuple[list[torch.Tensor], ClosureFn, tuple[str, ...], tuple[torch.Tensor, ...]]:
    """Trainable parameters, the closure they encode, names, and log-space ones.

    Returns ``(params, closure, names, log_params)``: *params* are the Adam
    leaves (constants trained in log space for positivity), *closure* maps
    ``Re -> C_s`` reading them live, *names* label the human-readable values
    (already exp-ed where trained in log space), and *log_params* is the
    subset to bound during optimisation.

    The power law is centred at *re_ref* (the geometric mean of the training
    Re, chosen by :func:`calibrate`): with ``re_ref = 1`` the intercept
    ``c0 = C_s(1)`` sits orders of magnitude away from the physical
    ``C_s ~ 0.1`` range — and outside the positivity clamp — so the family
    could not represent steeply falling closures at all.
    """
    if kind == "scalar":
        log_cs = torch.tensor(math.log(cs0), dtype=dtype, device=device, requires_grad=True)

        def closure(re: Any, _l=log_cs) -> Any:
            return torch.exp(_l)

        return [log_cs], closure, ("cs",), (log_cs,)

    if kind == "power":
        log_c0 = torch.tensor(math.log(cs0), dtype=dtype, device=device, requires_grad=True)
        b = torch.tensor(-0.1, dtype=dtype, device=device, requires_grad=True)

        def closure(re: Any, _c0=log_c0, _b=b, _ref=re_ref) -> Any:
            return torch.exp(_c0) * (float(re) / _ref) ** _b

        return [log_c0, b], closure, ("c0", "b"), (log_c0,)

    raise ValueError(f"kind must be 'scalar' or 'power', got {kind!r}")


@dataclass
class CalibResult:
    """Outcome of :func:`calibrate`.

    For ``kind="power"``, ``params["c0"]`` is the closure value at
    ``re_ref`` (the geometric mean of the training Re): ``c0`` alone is
    meaningless without it.
    """

    kind: str
    params: dict[str, float]
    loss_history: list[float] = field(default_factory=list)
    eval: dict[str, dict[str, float]] = field(default_factory=dict)
    closure: ClosureFn | None = None
    re_ref: float = 1.0
    collision: str = "bgk"
    sgs: str = "smagorinsky"

    def cs(self, re: float) -> float:
        """Evaluate the identified closure at ``re``."""
        assert self.closure is not None
        return float(self.closure(re))


def calibrate(
    targets: Sequence[DragTarget],
    box: BoxCase | HullCase,
    kind: str = "power",
    cs0: float = 0.05,
    iters: int = 150,
    lr: float = 0.03,
    log_every: int = 0,
    *,
    collision: str = "bgk",
    sgs: str = "smagorinsky",
) -> CalibResult:
    """Identify a closure from drag observations by backprop through rollouts.

    Args:
        targets: Observations ``(Re, C_D)``; weights scale the squared
            residuals.
        box: Domain configuration shared by all rollouts.
        kind: ``"scalar"`` — one constant ``C_s``; ``"power"`` —
            ``C_s(Re) = c0·Re^b`` (two parameters, trained in log space).
        cs0: Initial guess for the (geometric-mean) constant.
        iters: Adam iterations (cosine-decayed learning rate).
        lr: Peak learning rate.
        log_every: Print progress every n iterations (0: silent).
        collision: Collision family of the rollouts (see
            :func:`bounded_drag`).
        sgs: Sub-grid model the closure constant feeds (see
            :func:`bounded_drag`).

    Returns:
        :class:`CalibResult` with identified parameters, the loss history and
        per-target predictions.  The calibration Re grid is *not* baked into
        the closure: evaluate :meth:`CalibResult.cs` at any Re.
    """
    if not targets:
        raise ValueError("targets must be non-empty")
    _check_families(collision, sgs)
    device = torch.device(box.device)
    mask = box.make_mask()
    # centre the power law at the geometric mean of the training Re so the
    # intercept c0 stays in the physical C_s range (see _closure_parameters)
    re_ref = math.exp(sum(math.log(t.re) for t in targets) / len(targets))

    params, closure, names, log_params = _closure_parameters(kind, cs0, re_ref, device, box.dtype)
    optim = torch.optim.Adam(params, lr=lr)
    history: list[float] = []
    for it in range(iters):
        for group in optim.param_groups:
            group["lr"] = lr * 0.5 * (1.0 + math.cos(math.pi * it / iters))
        residuals = []
        for tgt in targets:
            cd_pred = bounded_drag(
                box, tgt.re, cs=closure(tgt.re), mask=mask, collision=collision, sgs=sgs
            )
            residuals.append(tgt.weight * (cd_pred - tgt.cd) ** 2)
        loss = torch.stack(residuals).sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            for p in log_params:  # keep the (log-space) constants positive-side
                p.clamp_(math.log(1e-4), math.log(1.0))
        history.append(float(loss.detach()))
        if log_every and (it == 0 or (it + 1) % log_every == 0):
            current = ", ".join(
                f"{n}={float(p.detach().exp()) if any(p is q for q in log_params) else float(p.detach()):.5f}"
                for n, p in zip(names, params)
            )
            print(f"  iter {it + 1:4d}  loss={float(loss.detach()):.6e}  {current}")

    human: dict[str, float] = {}
    for name, p in zip(names, params):
        human[name] = (
            float(p.detach().exp()) if any(p is q for q in log_params) else float(p.detach())
        )
    return CalibResult(
        kind=kind,
        params=human,
        loss_history=history,
        closure=closure,
        re_ref=re_ref,
        collision=collision,
        sgs=sgs,
    )


def evaluate(
    result: CalibResult,
    targets: Sequence[DragTarget],
    box: BoxCase | HullCase,
    *,
    collision: str | None = None,
    sgs: str | None = None,
) -> dict[str, dict[str, float]]:
    """Predict C_D at each target Re and compare against the observation.

    Fills ``result.eval`` and returns it: ``{re: {target, pred, abs_err,
    rel_err_pct}}``.  Use with held-out Re to measure closure extrapolation.
    By default the (collision, sgs) axes recorded in *result* — i.e. the
    ones :func:`calibrate` identified the closure under — are reused, so
    an evaluation cannot silently switch families.
    """
    assert result.closure is not None
    _collision = result.collision if collision is None else collision
    _sgs = result.sgs if sgs is None else sgs
    mask = box.make_mask()
    out: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for tgt in targets:
            cd_pred = float(
                bounded_drag(
                    box,
                    tgt.re,
                    cs=result.closure(tgt.re),
                    mask=mask,
                    collision=_collision,
                    sgs=_sgs,
                )
            )
            out[f"{tgt.re:g}"] = {
                "target": tgt.cd,
                "pred": cd_pred,
                "abs_err": abs(cd_pred - tgt.cd),
                "rel_err_pct": 100.0 * abs(cd_pred - tgt.cd) / abs(tgt.cd),
            }
    result.eval = out
    return out


# ---------------------------------------------------------------------------
# B3 stage 4: MRT moment-rate calibration against shell-pressure profiles
# ---------------------------------------------------------------------------

DEFAULT_MRT_RATES: dict[str, float] = {"s_e": 1.19, "s_eps": 1.4, "s_q": 1.2}
"""Geier-style default moment rates used by :func:`press_profile` etc."""


def _rate_collide(
    tau: float | torch.Tensor, rates: Mapping[str, float | torch.Tensor]
) -> Callable[..., torch.Tensor]:
    def collide(f: torch.Tensor, _tau: float | torch.Tensor) -> torch.Tensor:
        return collide_mrt3d(f, tau, s_e=rates["s_e"], s_eps=rates["s_eps"], s_q=rates["s_q"])

    return collide


def _plane_shell(
    mask: torch.Tensor, bins: int
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Centre-plane shell indexing for the axial pressure profile.

    Returns ``(cz, ays, axs, bin_idx)`` for the cells of the z=cz plane
    that are wet and 4-neighbour-adjacent to solid, binned along x.
    """
    cz = mask.shape[0] // 2
    smask = mask[cz]
    adj = torch.zeros_like(smask)
    adj[:, 1:] |= smask[:, :-1]
    adj[:, :-1] |= smask[:, 1:]
    adj[1:, :] |= smask[:-1, :]
    adj[:-1, :] |= smask[1:, :]
    adj &= ~smask
    ays, axs = adj.nonzero(as_tuple=True)
    nx = mask.shape[2]
    bin_idx = torch.minimum(axs // (nx // bins), torch.as_tensor(bins - 1, device=mask.device))
    return cz, ays, axs, bin_idx.long()


def press_profile(
    box: BoxCase | HullCase,
    re: float,
    rates: Mapping[str, float] | None = None,
    *,
    mask: torch.Tensor | None = None,
    bins: int = 32,
    chunk: int = 20,
) -> tuple[torch.Tensor, float]:
    """Window-averaged shell-pressure axial profile and windowed C_D.

    Runs the MRT family with moment *rates* (defaults = Geier values) over
    the case's window ``[window_start, steps)`` and reduces the centre
    plane (``z = nz//2``, the campaign snapshot convention) to *bins*
    axial means of ``rho - 1`` over shell cells — the cells adjacent
    (4-neighbourhood, in-plane) to solid.  ``rho`` is phase-invariant
    under collision and bounce-back, so any sub-step sampling agrees;
    measured equal to the probe-phase profile within 1e-4 relL2.

    Returns ``(profile, cd)``: fp64 tensor of shape ``(bins,)`` (empty
    bins are exactly zero) and the windowed drag coefficient.
    """
    device = torch.device(box.device)
    mask = _case_mask(box) if mask is None else mask
    rates = dict(DEFAULT_MRT_RATES) if rates is None else dict(rates)
    tau = box.tau_of_re(re)
    inlet = InletSpec(
        ux=torch.as_tensor(box.u_in, dtype=box.dtype, device=device),
        method=box.inlet_method,
    )
    walls = None if box.wall_method == "periodic" else WallSpec(method=box.wall_method, ux=box.u_in)
    collide = _rate_collide(tau, rates)
    cz, ays, axs, bin_idx = _plane_shell(mask, bins)
    counts = torch.bincount(bin_idx, minlength=bins).clamp(min=1).double()

    rho0 = torch.ones((box.nz, box.ny, box.nx), dtype=box.dtype, device=device)
    u = torch.zeros((3, box.nz, box.ny, box.nx), dtype=box.dtype, device=device)
    u[0] = box.u_in
    with torch.no_grad():
        f = equilibrium3d(rho0, u[0], u[1], u[2])
        f = rollout(
            f,
            box.window_start,
            tau,
            mask,
            collide=collide,
            inlet=inlet,
            outlet=OutletSpec(),
            walls=walls,
        )
        n_p = 0
        fx = 0.0
        prof = torch.zeros(bins, dtype=torch.float64, device=device)
        while n_p < box.steps - box.window_start:
            n = min(chunk, box.steps - box.window_start - n_p)
            f, probes = rollout(
                f,
                n,
                tau,
                mask,
                collide=collide,
                inlet=inlet,
                outlet=OutletSpec(),
                walls=walls,
                return_probes=True,
                probe_start=0,
            )
            for p in probes:
                fx += float(obstacle_force(p, mask)[0])
                rho = macroscopic3d(p)[0]
                prof = prof.index_add(0, bin_idx, (rho[cz][ays, axs] - 1.0).double())
            n_p += n
            del probes
    n_avg = float(n_p)
    return prof / n_avg / counts, fx / n_avg / (0.5 * box.u_in**2 * box.ref_area(mask))


def rate_fd_response(
    box: BoxCase | HullCase,
    re: float,
    rates: Mapping[str, float] | None = None,
    *,
    frac: float = 0.2,
    mask: torch.Tensor | None = None,
    bins: int = 32,
) -> dict[str, dict[str, float]]:
    """Finite-difference identifiability probe of the MRT moment rates.

    Varies each rate by +-``frac`` in ratio around its value and reports
    the profile's and C_D's response *per e-fold* of rate change:
    ``relL2(p_hi, p_lo)/ln(hi/lo)``.  A rate whose profile response is at
    the measurement noise floor cannot be identified from pressure data.
    """
    base = dict(DEFAULT_MRT_RATES) if rates is None else dict(rates)
    p_ref, cd_ref = press_profile(box, re, base, mask=mask, bins=bins)
    out: dict[str, dict[str, float]] = {}
    for name, val in base.items():
        lo, hi = dict(base), dict(base)
        lo[name], hi[name] = val * (1.0 - frac), val * (1.0 + frac)
        p_lo, cd_lo = press_profile(box, re, lo, mask=mask, bins=bins)
        p_hi, cd_hi = press_profile(box, re, hi, mask=mask, bins=bins)
        dl = abs(math.log((1.0 + frac) / (1.0 - frac)))
        out[name] = {
            "press_per_efold": float(
                torch.linalg.vector_norm(p_hi - p_lo) / torch.linalg.vector_norm(p_ref)
            )
            / dl,
            "cd_per_efold": abs(math.log(cd_hi / cd_lo)) / dl,
        }
    return out


@dataclass(frozen=True)
class RateCalibResult:
    """Outcome of :func:`calibrate_mrt_rates`."""

    rates: dict[str, float]
    loss_history: list[float] = field(default_factory=list)
    eval: dict[str, dict[str, float]] = field(default_factory=dict)
    rate0: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MRT_RATES))

    def collide(self, re: float, box: BoxCase | HullCase) -> Callable[..., torch.Tensor]:
        """Collision operator with the identified rates at *re*."""
        return _rate_collide(box.tau_of_re(re), self.rates)


def calibrate_mrt_rates(
    targets: Mapping[float, torch.Tensor],
    box: BoxCase | HullCase,
    *,
    mask: torch.Tensor | None = None,
    bins: int = 32,
    iters: int = 15,
    lr: float = 0.03,
    bounds: tuple[float, float] = (0.5, 2.0),
    block: int = 25,
    log_every: int = 0,
) -> RateCalibResult:
    """Identify MRT moment rates from shell-pressure profiles by backprop.

    The transient up to ``window_start`` runs no-grad; the window itself is
    differentiated through a block-checkpointed rollout (in-block profile
    reduction — memory holds only the block-boundary states, not the
    window).  Loss = mean over targets of the squared relative L2 between
    simulated and target profiles; Adam on the raw rates, clipped to
    *bounds* after every step (rates stay physical).

    Args:
        targets: ``{re: profile}`` shell-pressure targets, e.g. from
            campaign centre-plane snapshots binned the same way
            (:func:`press_profile` defines the binning).
        box: Domain and rollout configuration (MRT family, no SGS).
        mask: Pre-built solid mask (defaults to ``box.make_mask()``).
        bins: Axial bin count (must match the targets' binning).
        iters: Adam iterations.
        lr: Adam learning rate.
        bounds: ``(lo, hi)`` clamp for every rate after each step.
        block: Checkpoint block size in window steps; the window length
            ``steps - window_start`` must be divisible by it.
        log_every: Print loss every n iterations (0: silent).

    Returns:
        :class:`RateCalibResult`; ``eval`` holds, per target Re, the
        profile gap and C_D before (default rates) and after calibration.
    """
    from torch.utils.checkpoint import checkpoint

    device = torch.device(box.device)
    mask = _case_mask(box) if mask is None else mask
    window = box.steps - box.window_start
    if window % block:
        raise ValueError(f"window {window} not divisible by block {block}")
    cz, ays, axs, bin_idx = _plane_shell(mask, bins)
    counts = torch.bincount(bin_idx, minlength=bins).clamp(min=1).double()
    inlet = InletSpec(
        ux=torch.as_tensor(box.u_in, dtype=box.dtype, device=device),
        method=box.inlet_method,
    )
    walls = None if box.wall_method == "periodic" else WallSpec(method=box.wall_method, ux=box.u_in)
    tgt = {re: t.to(device=device, dtype=torch.float64) for re, t in targets.items()}
    tnorm = {re: float(torch.linalg.vector_norm(t) ** 2) for re, t in tgt.items()}

    rates = {
        k: torch.tensor(v, dtype=box.dtype, device=device, requires_grad=True)
        for k, v in DEFAULT_MRT_RATES.items()
    }

    def diff_profile(re: float) -> torch.Tensor:
        tau = box.tau_of_re(re)
        collide = _rate_collide(tau, rates)
        rho0 = torch.ones((box.nz, box.ny, box.nx), dtype=box.dtype, device=device)
        u = torch.zeros((3, box.nz, box.ny, box.nx), dtype=box.dtype, device=device)
        u[0] = box.u_in
        with torch.no_grad():
            f = equilibrium3d(rho0, u[0], u[1], u[2])
            f = rollout(
                f,
                box.window_start,
                tau,
                mask,
                collide=collide,
                inlet=inlet,
                outlet=OutletSpec(),
                walls=walls,
            ).detach()

        def run_block(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            acc = torch.zeros(bins, dtype=box.dtype, device=device)
            for _ in range(block):
                f = rollout(
                    f,
                    1,
                    tau,
                    mask,
                    collide=collide,
                    inlet=inlet,
                    outlet=OutletSpec(),
                    walls=walls,
                )
                rho = macroscopic3d(f)[0]
                acc = acc.index_add(0, bin_idx, rho[cz][ays, axs] - 1.0)
            return f, acc

        total = torch.zeros(bins, dtype=box.dtype, device=device)
        for _ in range(window // block):
            f, acc = checkpoint(run_block, f, use_reentrant=False)
            total = total + acc
        return (total.double() / window) / counts

    opt = torch.optim.Adam(rates.values(), lr=lr)
    hist: list[float] = []
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        losses = [((diff_profile(re) - t) ** 2).sum() / tnorm[re] for re, t in tgt.items()]
        total = torch.stack(losses).mean()
        torch.autograd.backward(total)
        opt.step()
        with torch.no_grad():
            for r in rates.values():
                r.clamp_(*bounds)
        hist.append(float(total.detach()))
        if log_every and it % log_every == 0:
            cur = {k: round(float(v.detach()), 4) for k, v in rates.items()}
            print(f"it{it:02d} loss={hist[-1]:.6f} {cur}")

    final = {k: float(v.detach()) for k, v in rates.items()}
    evaluation: dict[str, dict[str, float]] = {}
    for re, t in tgt.items():
        p0, cd0 = press_profile(box, re, DEFAULT_MRT_RATES, mask=mask, bins=bins)
        p1, cd1 = press_profile(box, re, final, mask=mask, bins=bins)
        n = torch.linalg.vector_norm(t)
        evaluation[f"{re:g}"] = {
            "press_before": float(torch.linalg.vector_norm(p0 - t) / n),
            "press_after": float(torch.linalg.vector_norm(p1 - t) / n),
            "cd_before": cd0,
            "cd_after": cd1,
        }
    return RateCalibResult(rates=final, loss_history=hist, eval=evaluation)


# ---------------------------------------------------------------------------
# B3 stage 5: differentiable D3Q27 collision families (cumulant / MRT)
# ---------------------------------------------------------------------------
#
# Stage 4 (``calibrate_mrt_rates``) left the held-out Re = 148 shell-pressure
# residual essentially untouched (0.0709 -> 0.0690) and concluded the residual
# is *structural*: the production campaign (``scan_suboff_re_drag_20260821``)
# runs a D3Q19 cumulant collision, and the three-rate D3Q19 MRT family cannot
# represent its low-Re behaviour.  Stage 5 swaps the collision family instead
# of fitting more rates: the two D3Q27 operators of the repo —
#
# * ``cumulant`` — :func:`tensorlbm.cumulant.collide_cumulant_d3q27`, the
#   central-moment/cumulant operator with bulk (``omega_b``), 3rd-order
#   (``omega_odd``) and >=4th-order (``omega_even``) ghost-mode rates.  This
#   is the family the stage-4 conclusions named ("chasing it requires a
#   differentiable cumulant target"): same operator class as production, one
#   isotropy order up.
# * ``mrt`` — :func:`tensorlbm.d3q27.collide_mrt27` with rates
#   ``s_e / s_eps / s_q`` (``s_pi`` follows ``s_e``, the historical default),
#   the D3Q27 analogue of the stage-4 family.
#
# The stage-4 API above is untouched; everything here is new surface.

#: Production-default D3Q27 cumulant rates (``collide_cumulant_d3q27`` defaults).
CUMULANT27_RATES: dict[str, float] = {"omega_b": 1.0, "omega_odd": 1.0, "omega_even": 1.0}

#: D3Q27 MRT rates at the historical defaults of :func:`tensorlbm.d3q27.collide_mrt27`.
MRT27_RATES: dict[str, float] = {"s_e": 1.19, "s_eps": 1.4, "s_q": 1.2}

#: Supported D3Q27 collision families for the stage-5 calibration path.
COLLISION27_FAMILIES: tuple[str, ...] = ("cumulant", "mrt")


def _default27_rates(family: str) -> dict[str, float]:
    if family == "cumulant":
        return dict(CUMULANT27_RATES)
    if family == "mrt":
        return dict(MRT27_RATES)
    raise ValueError(f"family must be one of {COLLISION27_FAMILIES}, got {family!r}")


def _central_to_cumulant_diffable(k: torch.Tensor) -> torch.Tensor:
    """Autograd-clean rewrite of :func:`tensorlbm.cumulant._central_to_cumulant`.

    The original builds its output with ``clone`` + per-index assignment;
    the saved-for-backward slice views of that clone are then modified by
    later assignments (6th order reads the 4th-order results), which aborts
    backward with an in-place-modified error.  This version writes the same
    arithmetic in static-index form (read-only component access + one
    ``torch.cat``), so gradients flow; for float and tensor inputs the result
    is bitwise identical to the original (guarded by test).
    """
    k200, k020, k002 = k[4], k[5], k[6]
    k110, k101, k011 = k[7], k[8], k[9]
    c17 = k[17] - k200 * k020 - 2.0 * k110 * k110
    c18 = k[18] - k200 * k002 - 2.0 * k101 * k101
    c19 = k[19] - k020 * k002 - 2.0 * k011 * k011
    c20 = k[20] - k200 * k011 - 2.0 * k101 * k110
    c21 = k[21] - k020 * k101 - 2.0 * k110 * k011
    c22 = k[22] - k002 * k110 - 2.0 * k101 * k011
    c23 = k[23] - k200 * k[13] - 2.0 * k110 * k[16]
    c24 = k[24] - k200 * k[15] - 2.0 * k101 * k[16]
    c25 = k[25] - k020 * k[14] - 2.0 * k011 * k[16]
    c26 = (
        k[26]
        - k200 * c19
        - k020 * c18
        - k002 * c17
        - 2.0 * (k110 * c22 + k101 * c21 + k011 * c20)
        + 2.0 * k200 * k020 * k002
        + 4.0 * k110 * k101 * k011
        + 2.0 * k110 * k110 * k002
        + 2.0 * k101 * k101 * k020
        + 2.0 * k011 * k011 * k200
    )
    return torch.cat(
        [
            k[0:17],
            c17[None],
            c18[None],
            c19[None],
            c20[None],
            c21[None],
            c22[None],
            c23[None],
            c24[None],
            c25[None],
            c26[None],
        ],
        dim=0,
    )


def _cumulant_to_central_diffable(C: torch.Tensor) -> torch.Tensor:
    """Autograd-clean inverse of :func:`_central_to_cumulant_diffable`.

    Same arithmetic as :func:`tensorlbm.cumulant._cumulant_to_central` in
    static-index form (bitwise identical, guarded by test).
    """
    C200, C020, C002 = C[4], C[5], C[6]
    C110, C101, C011 = C[7], C[8], C[9]
    k17 = C[17] + C200 * C020 + 2.0 * C110 * C110
    k18 = C[18] + C200 * C002 + 2.0 * C101 * C101
    k19 = C[19] + C020 * C002 + 2.0 * C011 * C011
    k20 = C[20] + C200 * C011 + 2.0 * C101 * C110
    k21 = C[21] + C020 * C101 + 2.0 * C110 * C011
    k22 = C[22] + C002 * C110 + 2.0 * C101 * C011
    k23 = C[23] + C200 * C[13] + 2.0 * C110 * C[16]
    k24 = C[24] + C200 * C[15] + 2.0 * C101 * C[16]
    k25 = C[25] + C020 * C[14] + 2.0 * C011 * C[16]
    k26 = (
        C[26]
        + C200 * C[19]
        + C020 * C[18]
        + C002 * C[17]
        + 2.0 * (C110 * C[22] + C101 * C[21] + C011 * C[20])
        - 2.0 * C200 * C020 * C002
        - 4.0 * C110 * C101 * C011
        - 2.0 * C110 * C110 * C002
        - 2.0 * C101 * C101 * C020
        - 2.0 * C011 * C011 * C200
    )
    return torch.cat(
        [
            C[0:17],
            k17[None],
            k18[None],
            k19[None],
            k20[None],
            k21[None],
            k22[None],
            k23[None],
            k24[None],
            k25[None],
            k26[None],
        ],
        dim=0,
    )


def _relax_cumulants_diffable(
    C: torch.Tensor,
    omega_shear: float | torch.Tensor,
    omega_bulk: float | torch.Tensor,
    omega_3: float | torch.Tensor,
    omega_46: float | torch.Tensor,
) -> torch.Tensor:
    """Autograd-clean rewrite of :func:`tensorlbm.cumulant._relax_cumulants_d3q27`.

    Same trace/deviatoric split and per-order rates, written without
    index assignment so tensor rates keep their gradients.  ``omega_4/5/6``
    share one rate here (the production default ties them to ``omega_even``).
    """
    Cxx, Cyy, Czz = C[4], C[5], C[6]
    Cxy, Cxz, Cyz = C[7], C[8], C[9]
    trace = Cxx + Cyy + Czz
    trace_s = (1.0 - omega_bulk) * trace
    n4 = (1.0 - omega_shear) * (Cxx - trace / 3.0) + trace_s / 3.0
    n5 = (1.0 - omega_shear) * (Cyy - trace / 3.0) + trace_s / 3.0
    n6 = (1.0 - omega_shear) * (Czz - trace / 3.0) + trace_s / 3.0
    n7 = (1.0 - omega_shear) * Cxy
    n8 = (1.0 - omega_shear) * Cxz
    n9 = (1.0 - omega_shear) * Cyz
    n3 = (1.0 - omega_3) * C[10:17]
    n46 = (1.0 - omega_46) * C[17:26]
    n26 = (1.0 - omega_46) * C[26]
    return torch.cat(
        [C[0:4], n4[None], n5[None], n6[None], n7[None], n8[None], n9[None], n3, n46, n26[None]],
        dim=0,
    )


def collide_cumulant27_diffable(
    f: torch.Tensor,
    tau: float | torch.Tensor,
    omega_b: float | torch.Tensor = 1.0,
    omega_odd: float | torch.Tensor = 1.0,
    omega_even: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Differentiable D3Q27 cumulant collision (stage-5 calibration family).

    Mirrors :func:`tensorlbm.cumulant.collide_cumulant_d3q27` (``C_s = 0``)
    with the nonlinear transforms rewritten in autograd-safe form, so the
    ghost-mode rates — and everything else in the chain — carry gradients.
    Bitwise identical to the original for float rates (guarded by test);
    unlike it, backward through the operator works.

    Args:
        f: Distribution tensor, shape ``(27, nz, ny, nx)``.
        tau: Shear relaxation time (sets ``omega_shear = 1/tau``).
        omega_b: Bulk (trace) relaxation rate.
        omega_odd: 3rd-order ghost-mode rate.
        omega_even: 4th/5th/6th-order ghost-mode rate.

    Returns:
        Post-collision distribution, shape ``(27, nz, ny, nx)``.
    """
    from tensorlbm.cumulant import _get_matrices, _to_central, _to_raw
    from tensorlbm.d3q27 import equilibrium27, macroscopic27

    omega = 1.0 / tau
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq
    M, M_inv = _get_matrices(f.device, f.dtype)
    nz, ny, nx = f_neq.shape[1], f_neq.shape[2], f_neq.shape[3]
    m_neq = (M @ f_neq.reshape(27, -1)).reshape(27, nz, ny, nx)
    k_neq = _to_central(m_neq, ux, uy, uz)
    C_neq = _central_to_cumulant_diffable(k_neq)
    C_star = _relax_cumulants_diffable(
        C_neq,
        omega_shear=omega,
        omega_bulk=omega_b,
        omega_3=omega_odd,
        omega_46=omega_even,
    )
    k_star = _cumulant_to_central_diffable(C_star)
    m_star = _to_raw(k_star, ux, uy, uz)
    f_neq_star = (M_inv @ m_star.reshape(27, -1)).reshape(27, nz, ny, nx)
    return feq + f_neq_star


def obstacle_force27(f_probe: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Momentum-exchange force on the bounce-back obstacle, D3Q27 (Ladd).

    Same convention as :func:`tensorlbm.autograd_path.obstacle_force`:
    ``F_alpha = 2 sum_{x in solid} sum_q c_{q,alpha} f[q, x]`` on the
    post-stream / pre-bounce-back probe state.
    """
    from tensorlbm.d3q27 import C as C27

    c = C27.to(device=f_probe.device, dtype=f_probe.dtype)
    momentum = (f_probe * mask.to(dtype=f_probe.dtype)).sum(dim=(1, 2, 3))
    return 2.0 * torch.matmul(momentum, c)


def _far_field_faces27(f: torch.Tensor, u_in: float) -> torch.Tensor:
    """Free-stream faces without bounce-back: the probe keeps its phase.

    Applies exactly the six-plane closure of
    :func:`tensorlbm.boundaries_d3q27.far_field_bc_27` — equilibrium inlet
    (x = 0), zero-gradient outlet (x = nx-1), equilibrium lateral faces —
    but stops before the obstacle bounce-back so the returned state is the
    post-stream / post-BC / pre-bounce-back probe (the production sampling
    phase).
    """
    from tensorlbm.d3q27 import equilibrium27

    rho1 = torch.ones((f.shape[1], f.shape[2], f.shape[3]), dtype=f.dtype, device=f.device)
    feq = equilibrium27(
        rho1, torch.full_like(rho1, u_in), torch.zeros_like(rho1), torch.zeros_like(rho1)
    )
    out = f.clone()
    out[:, :, :, 0] = feq[:, :, :, 0]  # inlet (free stream)
    out[:, :, :, -1] = out[:, :, :, -2]  # outlet (zero gradient)
    out[:, 0, :, :] = feq[:, 0, :, :]  # y- lateral
    out[:, -1, :, :] = feq[:, -1, :, :]  # y+ lateral
    out[:, :, 0, :] = feq[:, :, 0, :]  # z- lateral
    out[:, :, -1, :] = feq[:, :, -1, :]  # z+ lateral
    return out


def step27(
    f: torch.Tensor,
    tau: float | torch.Tensor,
    mask: torch.Tensor,
    collide: Callable[..., torch.Tensor],
    u_in: float,
    *,
    return_probe: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """One autograd-clean D3Q27 bounded step.

    The D3Q27 counterpart of
    :func:`tensorlbm.autograd_path.differentiable_step` with the HullCase
    boundary set: collide (NoDynamics inside the solid) -> stream
    (:func:`tensorlbm.d3q27.stream27_roll`) -> free-stream faces ->
    full-way bounce-back, with the probe returned between the faces and the
    bounce-back (the production sampling phase).
    """
    from tensorlbm.boundaries_d3q27 import bounce_back_cells_27
    from tensorlbm.d3q27 import stream27_roll

    f_col = torch.where(mask.unsqueeze(0), f, collide(f, tau))
    fs = stream27_roll(f_col)
    fs = _far_field_faces27(fs, u_in)
    if return_probe:
        return bounce_back_cells_27(fs, mask), fs
    return bounce_back_cells_27(fs, mask)


def rollout27(
    f: torch.Tensor,
    n_steps: int,
    tau: float | torch.Tensor,
    mask: torch.Tensor,
    collide: Callable[..., torch.Tensor],
    u_in: float,
    *,
    return_probes: bool = False,
    probe_start: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Roll out :func:`step27` for *n_steps*, keeping the autograd graph.

    With ``return_probes=True`` collects the per-step pre-bounce-back probe
    states from step ``probe_start`` on (windowed observables; see
    :func:`tensorlbm.autograd_path.rollout` for the D3Q19 semantics).
    """
    if return_probes and not 0 <= probe_start < max(n_steps, 1):
        raise ValueError(f"probe_start must be in [0, {n_steps}) when return_probes=True")
    probes: list[torch.Tensor] = []
    for step in range(n_steps):
        if return_probes:
            f, probe = step27(f, tau, mask, collide, u_in, return_probe=True)
            if step >= probe_start:
                probes.append(probe)
        else:
            f = step27(f, tau, mask, collide, u_in)
    if return_probes:
        return f, probes
    return f


def _rate27_collide(
    tau: float | torch.Tensor, rates: Mapping[str, Any], family: str
) -> Callable[..., torch.Tensor]:
    if family == "cumulant":
        return lambda f, _t: collide_cumulant27_diffable(
            f,
            tau,
            omega_b=rates["omega_b"],
            omega_odd=rates["omega_odd"],
            omega_even=rates["omega_even"],
        )
    from tensorlbm.d3q27 import collide_mrt27

    return lambda f, _t: collide_mrt27(
        f, tau, s_e=rates["s_e"], s_eps=rates["s_eps"], s_q=rates["s_q"], s_pi=rates["s_e"]
    )


@torch.no_grad()
def press_profile27(
    box: BoxCase | HullCase,
    re: float,
    rates: Mapping[str, float] | None = None,
    *,
    family: str = "cumulant",
    mask: torch.Tensor | None = None,
    bins: int = 32,
    chunk: int = 20,
) -> tuple[torch.Tensor, float]:
    """D3Q27 shell-pressure profile and windowed C_D (stage-5 observable).

    The exact D3Q27 counterpart of :func:`press_profile`: same case window,
    same centre-plane shell binning (``_plane_shell``), same fp64 profile
    reduction on the post-stream / pre-bounce-back probe — only the
    collision family and the 27-population rollout change.

    Returns:
        ``(profile, cd)``: fp64 tensor ``(bins,)`` of window-mean ``rho - 1``
        over shell cells, and the windowed drag coefficient
        (``obstacle_force27`` normalisation).
    """
    from tensorlbm.d3q27 import equilibrium27, macroscopic27

    rates = _default27_rates(family) if rates is None else dict(rates)
    device = torch.device(box.device)
    mask = _case_mask(box) if mask is None else mask
    tau = box.tau_of_re(re)
    collide = _rate27_collide(tau, rates, family)
    cz, ays, axs, bin_idx = _plane_shell(mask, bins)
    counts = torch.bincount(bin_idx, minlength=bins).clamp(min=1).double()

    rho0 = torch.ones((box.nz, box.ny, box.nx), dtype=box.dtype, device=device)
    u = torch.zeros((3, box.nz, box.ny, box.nx), dtype=box.dtype, device=device)
    u[0] = box.u_in
    f = equilibrium27(rho0, u[0], u[1], u[2])
    f = rollout27(f, box.window_start, tau, mask, collide, box.u_in)
    n_p = 0
    fx = 0.0
    prof = torch.zeros(bins, dtype=torch.float64, device=device)
    while n_p < box.steps - box.window_start:
        n = min(chunk, box.steps - box.window_start - n_p)
        f, probes = rollout27(f, n, tau, mask, collide, box.u_in, return_probes=True, probe_start=0)
        for p in probes:
            fx += float(obstacle_force27(p, mask)[0])
            rho = macroscopic27(p)[0]
            prof = prof.index_add(0, bin_idx, (rho[cz][ays, axs] - 1.0).double())
        n_p += n
        del probes
    n_avg = float(n_p)
    return prof / n_avg / counts, fx / n_avg / (0.5 * box.u_in**2 * box.ref_area(mask))


def rate_fd_response27(
    box: BoxCase | HullCase,
    re: float,
    rates: Mapping[str, float] | None = None,
    *,
    family: str = "cumulant",
    frac: float = 0.2,
    mask: torch.Tensor | None = None,
    bins: int = 32,
) -> dict[str, dict[str, float]]:
    """Finite-difference identifiability probe of the D3Q27 rates.

    Same protocol as :func:`rate_fd_response`: each rate varied by
    +-``frac`` in ratio, profile and C_D response per e-fold of rate change.
    """
    base = _default27_rates(family) if rates is None else dict(rates)
    p_ref, cd_ref = press_profile27(box, re, base, family=family, mask=mask, bins=bins)
    out: dict[str, dict[str, float]] = {}
    for name, val in base.items():
        lo, hi = dict(base), dict(base)
        lo[name], hi[name] = val * (1.0 - frac), val * (1.0 + frac)
        p_lo, cd_lo = press_profile27(box, re, lo, family=family, mask=mask, bins=bins)
        p_hi, cd_hi = press_profile27(box, re, hi, family=family, mask=mask, bins=bins)
        dl = abs(math.log((1.0 + frac) / (1.0 - frac)))
        out[name] = {
            "press_per_efold": float(
                torch.linalg.vector_norm(p_hi - p_lo) / torch.linalg.vector_norm(p_ref)
            )
            / dl,
            "cd_per_efold": abs(math.log(cd_hi / cd_lo)) / dl,
        }
    return out


@dataclass(frozen=True)
class Collision27CalibResult:
    """Outcome of :func:`calibrate_collision27`."""

    family: str
    rates: dict[str, float]
    loss_history: list[float] = field(default_factory=list)
    eval: dict[str, dict[str, float]] = field(default_factory=dict)

    def collide(self, re: float, box: BoxCase | HullCase) -> Callable[..., torch.Tensor]:
        """Collision operator with the identified rates at *re*."""
        return _rate27_collide(box.tau_of_re(re), self.rates, self.family)


def calibrate_collision27(
    targets: Mapping[float, torch.Tensor],
    box: BoxCase | HullCase,
    *,
    family: str = "cumulant",
    mask: torch.Tensor | None = None,
    bins: int = 32,
    iters: int = 15,
    lr: float = 0.03,
    bounds: tuple[float, float] = (0.5, 2.0),
    block: int = 25,
    log_every: int = 0,
) -> Collision27CalibResult:
    """Identify D3Q27 collision rates from shell-pressure profiles by backprop.

    The stage-5 counterpart of :func:`calibrate_mrt_rates` with the D3Q27
    ``family`` (``"cumulant"`` or ``"mrt"``) in place of the D3Q19 MRT:
    no-grad transient, block-checkpointed differentiable window, in-block
    profile reduction, Adam on the raw rates clamped to *bounds* after every
    step.  Loss = mean over targets of the squared relative L2 between
    simulated and target profiles (non-conserved observable; see the
    probe-degeneracy note in ``docs/closure_calibration.md``).

    Returns:
        :class:`Collision27CalibResult`; ``eval`` holds, per target Re, the
        profile gap and C_D before (family-default rates) and after
        calibration.
    """
    from torch.utils.checkpoint import checkpoint

    from tensorlbm.d3q27 import equilibrium27, macroscopic27

    if family not in COLLISION27_FAMILIES:
        raise ValueError(f"family must be one of {COLLISION27_FAMILIES}, got {family!r}")
    device = torch.device(box.device)
    mask = _case_mask(box) if mask is None else mask
    window = box.steps - box.window_start
    if window % block:
        raise ValueError(f"window {window} not divisible by block {block}")
    cz, ays, axs, bin_idx = _plane_shell(mask, bins)
    counts = torch.bincount(bin_idx, minlength=bins).clamp(min=1).double()
    defaults = _default27_rates(family)
    tgt = {re: t.to(device=device, dtype=torch.float64) for re, t in targets.items()}
    tnorm = {re: float(torch.linalg.vector_norm(t) ** 2) for re, t in tgt.items()}

    rates = {
        k: torch.tensor(v, dtype=box.dtype, device=device, requires_grad=True)
        for k, v in defaults.items()
    }

    def diff_profile(re: float) -> torch.Tensor:
        tau = box.tau_of_re(re)
        collide = _rate27_collide(tau, rates, family)
        rho0 = torch.ones((box.nz, box.ny, box.nx), dtype=box.dtype, device=device)
        u = torch.zeros((3, box.nz, box.ny, box.nx), dtype=box.dtype, device=device)
        u[0] = box.u_in
        with torch.no_grad():
            f = equilibrium27(rho0, u[0], u[1], u[2])
            f = rollout27(f, box.window_start, tau, mask, collide, box.u_in).detach()

        def run_block(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            acc = torch.zeros(bins, dtype=box.dtype, device=device)
            for _ in range(block):
                f, probe = step27(f, tau, mask, collide, box.u_in, return_probe=True)
                rho = macroscopic27(probe)[0]
                acc = acc.index_add(0, bin_idx, rho[cz][ays, axs] - 1.0)
            return f, acc

        total = torch.zeros(bins, dtype=box.dtype, device=device)
        for _ in range(window // block):
            f, acc = checkpoint(run_block, f, use_reentrant=False)
            total = total + acc
        return (total.double() / window) / counts

    opt = torch.optim.Adam(rates.values(), lr=lr)
    hist: list[float] = []
    # D3Q27 rates beyond ~1.6-1.9 over-relax the ghost modes and the bounded
    # rollout diverges (NaN loss, measured on the HullCase n128 campaign
    # points); the guard reverts to the last finite rates and stops instead
    # of poisoning the result.  ``loss_history`` records only finite losses.
    last_good = dict(defaults)
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        losses = [((diff_profile(re) - t) ** 2).sum() / tnorm[re] for re, t in tgt.items()]
        total = torch.stack(losses).mean()
        if not math.isfinite(float(total.detach())):
            with torch.no_grad():
                for k, r in rates.items():
                    r.fill_(last_good[k])
            print(
                f"{family}: rollout diverged at iter {it}; "
                f"reverted to last finite rates {last_good}"
            )
            break
        torch.autograd.backward(total)
        opt.step()
        with torch.no_grad():
            for r in rates.values():
                r.clamp_(*bounds)
            last_good = {k: float(v.detach()) for k, v in rates.items()}
        hist.append(float(total.detach()))
        if log_every and it % log_every == 0:
            cur = {k: round(float(v.detach()), 4) for k, v in rates.items()}
            print(f"{family} it{it:02d} loss={hist[-1]:.6f} {cur}")

    final = {k: float(v.detach()) for k, v in rates.items()}
    evaluation: dict[str, dict[str, float]] = {}
    for re, t in tgt.items():
        p0, cd0 = press_profile27(box, re, defaults, family=family, mask=mask, bins=bins)
        p1, cd1 = press_profile27(box, re, final, family=family, mask=mask, bins=bins)
        n = torch.linalg.vector_norm(t)
        evaluation[f"{re:g}"] = {
            "press_before": float(torch.linalg.vector_norm(p0 - t) / n),
            "press_after": float(torch.linalg.vector_norm(p1 - t) / n),
            "cd_before": cd0,
            "cd_after": cd1,
        }
    return Collision27CalibResult(family=family, rates=final, loss_history=hist, eval=evaluation)


# ---------------------------------------------------------------------------
# B3 stage 6: campaign-semantics alignment knobs (pipeline-definition residual)
# ---------------------------------------------------------------------------
#
# Stage 5 (``runs/b3_stage5_20260824/diag_samefamily.json``) rolled the
# campaign's *own* D3Q19 cumulant operator through the calibration pipeline
# and showed that half the low-Re residual is the periodic mass correction
# (Re 148: 0.068 -> 0.040) while the rest persists across Re and across
# collision families: it is *pipeline definition*, not physics.  Stage 6
# enumerates the definitions that differ between the campaign chain
# (``scan_runner.run_scan_point`` on the ``suboff_n128`` case) and the
# differentiable calibration chain (``press_profile`` / ``press_profile27``),
# pins each one behind a switch (:class:`CampaignSemantics`) and re-runs the
# diagnosis and the calibration under the fully aligned semantics.
#
# The definitional differences (campaign side, file:line):
#
# 1. initialisation inside the solid — campaign ``initial_pu`` sets
#    ``ux = 0`` inside the hull (``cases/suboff.py:124``); the calibration
#    chain initialises free-stream equilibrium everywhere;
# 2. collision inside the solid — the campaign collides *every* cell
#    (``cases/base.py:245`` -> ``collide_advanced_3d`` has no solid skip);
#    the calibration chain is NoDynamics in the solid
#    (``autograd_path._collide_skip_solid``; ``step27`` same);
# 3. outlet closure — the campaign's ``far_field_bc_3d`` copies the whole
#    outlet plane (all populations, ``boundaries3d.py:546``); the
#    calibration outlet replaces only the unknown outgoing directions
#    (``autograd_path._apply_outlet``).  The D3Q27 chain already uses the
#    full-plane copy (``_far_field_faces27``);
# 4. mass correction — the campaign rescales every 10th step at absolute
#    step numbers, post-step (``scan_runner.py:874``: ``step % mass_every
#    == 0`` on the 1-indexed loop; interval from ``cases/suboff.py:44``),
#    from step 10 (transient included); the stage-5 diagnosis corrected
#    every 20 steps, window only;
# 5. observation readout — the campaign observable is a *snapshot export*
#    (``FieldSampleReporter`` dispatched post-correction,
#    ``scan_runner.py:892``; the final step is always exported); the
#    calibration observable is the window mean over per-step probes.
#
# Everything below is additive: the stage-4/5 surface is untouched and the
# historical behaviour is exactly ``CampaignSemantics()`` with all defaults
# (verified bitwise by ``tests/test_campaign_semantics.py``).


@dataclass(frozen=True)
class CampaignSemantics:
    """Pipeline-definition switches aligning a rollout with the campaign chain.

    Every field names one definitional axis; the default of each field is the
    *historical calibration-pipeline* behaviour, the campaign value is
    :data:`CAMPAIGN_SEMANTICS`.  The knobs are independent so the residual
    can be decomposed by switching them on one at a time (or all at once).

    Attributes:
        init_solid: ``"freestream"`` (legacy — equilibrium at ``u_in``
            everywhere, solid included) or ``"rest"`` (campaign — ``ux = 0``
            inside the solid, ``rho = 1`` everywhere).
        collide_solid: ``False`` (legacy — NoDynamics inside the solid) or
            ``True`` (campaign — collide every cell).
        outlet: ``"unknown"`` (legacy — only the outgoing directions of the
            outlet plane are copied) or ``"full"`` (campaign — the whole
            outlet plane is copied from the interior neighbour).
        mass_every: mass-correction interval in solver steps (0 = never);
            corrections land at solver steps ``s`` with
            ``s % mass_every == mass_phase`` and ``s >= mass_first``
            (campaign: 10/0/10 — every 10th step from step 10, post-step,
            ``scan_runner.py:874``).
        mass_phase: phase of the correction grid (see ``mass_every``).
        mass_first: first correctable step (see ``mass_every``).
        observe_frames: campaign snapshot readout.  Empty (legacy) reads the
            window mean over per-step probes; otherwise the profile is the
            mean of the post-correction states at exactly these solver steps
            — the ``FieldSampleReporter`` export definition.
    """

    init_solid: str = "freestream"
    collide_solid: bool = False
    outlet: str = "unknown"
    mass_every: int = 0
    mass_phase: int = 0
    mass_first: int = 1
    observe_frames: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.init_solid not in ("freestream", "rest"):
            raise ValueError(f"init_solid must be 'freestream' or 'rest', got {self.init_solid!r}")
        if not isinstance(self.collide_solid, bool):
            raise ValueError("collide_solid must be a bool")
        if self.outlet not in ("unknown", "full"):
            raise ValueError(f"outlet must be 'unknown' or 'full', got {self.outlet!r}")
        for name, value in (("mass_every", self.mass_every), ("mass_first", self.mass_first)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"CampaignSemantics.{name} must be a non-negative int, got {value!r}"
                )
        if not 0 <= self.mass_phase < max(self.mass_every, 1):
            raise ValueError(
                f"mass_phase must be in [0, {max(self.mass_every, 1)}), got {self.mass_phase}"
            )
        if self.mass_every and self.mass_first < 1:
            raise ValueError("mass_first must be >= 1 (solver steps are 1-indexed)")
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in self.observe_frames):
            raise ValueError(f"observe_frames must be positive ints, got {self.observe_frames!r}")

    def correct_at(self, step: int) -> bool:
        """Whether the mass correction lands after solver step *step*."""
        return (
            self.mass_every > 0
            and step >= self.mass_first
            and step % self.mass_every == self.mass_phase
        )

    def with_frames(self, *steps: int) -> CampaignSemantics:
        """Copy with the campaign snapshot readout at *steps* (no args: window)."""
        from dataclasses import replace

        return replace(self, observe_frames=tuple(steps))


#: Fully campaign-aligned semantics (``scan_runner`` + ``suboff_n128``).
CAMPAIGN_SEMANTICS = CampaignSemantics(
    init_solid="rest",
    collide_solid=True,
    outlet="full",
    mass_every=10,
    mass_phase=0,
    mass_first=10,
)

#: The historical D3Q19 calibration-pipeline semantics (all defaults).
LEGACY_SEMANTICS = CampaignSemantics()

#: The historical D3Q27 bounded-chain semantics (its outlet is the production
#: full-plane copy already, so only that value is legal on the 27 chain).
LEGACY27_SEMANTICS = CampaignSemantics(outlet="full")


def _initial_f19(
    box: BoxCase | HullCase, mask: torch.Tensor, sem: CampaignSemantics
) -> torch.Tensor:
    """Initial D3Q19 populations under *sem* (campaign or legacy initialisation)."""
    device = torch.device(box.device)
    rho0 = torch.ones((box.nz, box.ny, box.nx), dtype=box.dtype, device=device)
    u = torch.zeros((3, box.nz, box.ny, box.nx), dtype=box.dtype, device=device)
    u[0] = box.u_in
    if sem.init_solid == "rest":
        # campaign cases/suboff.py:124 — ux = 0 inside the hull, rho = 1 everywhere
        u[0] = torch.where(mask, torch.zeros_like(u[0]), u[0])
    return equilibrium3d(rho0, u[0], u[1], u[2])


def _initial_f27(
    box: BoxCase | HullCase, mask: torch.Tensor, sem: CampaignSemantics
) -> torch.Tensor:
    """Initial D3Q27 populations under *sem*."""
    from tensorlbm.d3q27 import equilibrium27

    device = torch.device(box.device)
    rho0 = torch.ones((box.nz, box.ny, box.nx), dtype=box.dtype, device=device)
    u = torch.zeros((3, box.nz, box.ny, box.nx), dtype=box.dtype, device=device)
    u[0] = box.u_in
    if sem.init_solid == "rest":
        u[0] = torch.where(mask, torch.zeros_like(u[0]), u[0])
    return equilibrium27(rho0, u[0], u[1], u[2])


def campaign_chain19(
    box: BoxCase | HullCase,
    collide: Callable[..., torch.Tensor],
    mask: torch.Tensor,
    sem: CampaignSemantics,
    tau: float | torch.Tensor = 0.9,
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Build the one-step D3Q19 closure ``(f) -> (f_new, probe)`` under *sem*.

    With all-legacy knobs the closure is bit-for-bit
    :func:`tensorlbm.autograd_path.differentiable_step` at the same boundary
    specs (guarded by test); with ``sem.outlet == "full"`` the boundary step
    is the production ``far_field_bc_3d`` face closure without the obstacle
    bounce-back (so the probe keeps the production sampling phase) and the
    bounce-back is the production :func:`bounce_back_cells_3d`.
    """
    from tensorlbm.autograd_path import _apply_inlet, _apply_outlet, _apply_walls
    from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
    from tensorlbm.d3q19 import OPPOSITE
    from tensorlbm.solver3d import stream3d

    device = torch.device(box.device)
    inlet = InletSpec(
        ux=torch.as_tensor(box.u_in, dtype=box.dtype, device=device),
        method=box.inlet_method,
    )
    walls = None if box.wall_method == "periodic" else WallSpec(method=box.wall_method, ux=box.u_in)
    outlet = OutletSpec()
    opp = OPPOSITE.to(device=device)

    def collided(f: torch.Tensor) -> torch.Tensor:
        f_col = collide(f, tau)
        if sem.collide_solid or mask is None:
            return f_col
        return torch.where(mask.unsqueeze(0), f, f_col)

    if sem.outlet == "full":
        bc_config = {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []}

        def step(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            probe = far_field_bc_3d(
                stream3d(collided(f)), box.u_in, obstacle_mask=None, bc_config=bc_config
            )
            return bounce_back_cells_3d(probe, mask), probe

    else:

        def step(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            f_str = stream3d(collided(f))
            if walls is not None:
                f_str = _apply_walls(f_str, walls)
            f_str = _apply_inlet(f_str, inlet)
            f_str = _apply_outlet(f_str, outlet, inlet, f[..., -1:])
            return torch.where(mask.unsqueeze(0), f_str[opp], f_str), f_str

    return step


def campaign_chain27(
    box: BoxCase | HullCase,
    collide: Callable[..., torch.Tensor],
    mask: torch.Tensor,
    sem: CampaignSemantics,
    tau: float | torch.Tensor = 0.9,
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """D3Q27 counterpart of :func:`campaign_chain19`.

    The D3Q27 bounded chain (:func:`step27`) already uses the production
    full-plane outlet copy (``_far_field_faces27``), so only the
    collide-in-solid knob applies on top of it; ``sem.outlet`` must be
    ``"full"`` (the unknown-directions outlet has no counterpart on the
    27-stencil chain here).
    """
    from tensorlbm.boundaries_d3q27 import bounce_back_cells_27
    from tensorlbm.d3q27 import stream27_roll

    if sem.outlet != "full":
        raise ValueError(
            "the D3Q27 bounded chain only implements the production full-copy outlet; "
            f"got sem.outlet={sem.outlet!r}"
        )

    def step(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f_col = collide(f, tau)
        if not sem.collide_solid:
            f_col = torch.where(mask.unsqueeze(0), f, f_col)
        probe = _far_field_faces27(stream27_roll(f_col), box.u_in)
        return bounce_back_cells_27(probe, mask), probe

    return step


def _shell_acc(
    rho: torch.Tensor,
    cz: int,
    ays: torch.Tensor,
    axs: torch.Tensor,
    bin_idx: torch.Tensor,
    bins: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Binned centre-plane shell profile of ``rho - 1`` for one state."""
    out = torch.zeros(bins, dtype=dtype, device=device)
    return out.index_add(0, bin_idx, rho[cz][ays, axs] - 1.0)


@dataclass(frozen=True)
class CampaignObservables:
    """Readouts of one :func:`campaign_rollout19` / :func:`campaign_rollout27` run.

    ``window_profile`` / ``window_cd`` are the historical observable
    (per-step probe mean over ``(window_start, steps]``, Ladd momentum
    exchange on the pre-bounce probe); ``frames`` holds the campaign
    snapshot profiles (post-step, post-mass-correction states — the
    ``FieldSampleReporter`` export phase).
    """

    window_profile: torch.Tensor
    window_cd: float
    frames: dict[int, torch.Tensor]
    m0: float
    n_window: int


def _run_campaign(
    box: BoxCase | HullCase,
    collide: Callable[..., torch.Tensor],
    mask: torch.Tensor,
    sem: CampaignSemantics,
    chain: Callable[..., Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]],
    initial_f: Callable[[BoxCase | HullCase, torch.Tensor, CampaignSemantics], torch.Tensor],
    macroscopic: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
    force: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    bins: int = 32,
    chunk: int = 20,
) -> CampaignObservables:
    """Shared driver: roll the *sem* chain over ``[1, box.steps]`` and read it out."""
    from tensorlbm.solver3d import correct_mass3d

    device = torch.device(box.device)
    cz, ays, axs, bin_idx = _plane_shell(mask, bins)
    counts = torch.bincount(bin_idx, minlength=bins).clamp(min=1).double()
    frames = sorted(sem.observe_frames)
    for s in frames:
        if s > box.steps:
            raise ValueError(f"observe frame {s} is beyond box.steps={box.steps}")
    step = chain(box, collide, mask, sem)
    with torch.no_grad():
        f = initial_f(box, mask, sem)
        m0 = float(f.sum())  # campaign initial mass (scan_runner.py:802)
        prof = torch.zeros(bins, dtype=torch.float64, device=device)
        frame_profs = {s: torch.zeros(bins, dtype=torch.float64, device=device) for s in frames}
        fx = 0.0
        n_w = 0
        s = 0
        while s < box.steps:
            n = min(chunk, box.steps - s)
            for _ in range(n):
                f, probe = step(f)
                s += 1
                if s > box.window_start:
                    fx += float(force(probe, mask)[0])
                    rho = macroscopic(probe)[0]
                    prof = prof.index_add(0, bin_idx, (rho[cz][ays, axs] - 1.0).double())
                    n_w += 1
                if sem.correct_at(s):
                    f = correct_mass3d(f, m0)
                if s in frame_profs:
                    # campaign export phase: post-step, post-correction, post-bounce
                    rho = macroscopic(f)[0]
                    frame_profs[s] = frame_profs[s].index_add(
                        0, bin_idx, (rho[cz][ays, axs] - 1.0).double()
                    )
    window_profile = prof / n_w / counts
    cd = fx / n_w / (0.5 * box.u_in**2 * box.ref_area(mask))
    return CampaignObservables(
        window_profile=window_profile,
        window_cd=cd,
        frames={s: p / counts for s, p in frame_profs.items()},
        m0=m0,
        n_window=n_w,
    )


def campaign_rollout19(
    box: BoxCase | HullCase,
    collide: Callable[..., torch.Tensor],
    sem: CampaignSemantics = LEGACY_SEMANTICS,
    *,
    mask: torch.Tensor | None = None,
    bins: int = 32,
    chunk: int = 20,
) -> CampaignObservables:
    """Run the D3Q19 *sem* chain and collect window + snapshot observables."""
    mask = _case_mask(box) if mask is None else mask
    return _run_campaign(
        box,
        collide,
        mask,
        sem,
        campaign_chain19,
        _initial_f19,
        macroscopic3d,
        obstacle_force,
        bins=bins,
        chunk=chunk,
    )


def campaign_rollout27(
    box: BoxCase | HullCase,
    collide: Callable[..., torch.Tensor],
    sem: CampaignSemantics = LEGACY27_SEMANTICS,
    *,
    mask: torch.Tensor | None = None,
    bins: int = 32,
    chunk: int = 20,
) -> CampaignObservables:
    """Run the D3Q27 *sem* chain and collect window + snapshot observables."""
    from tensorlbm.d3q27 import macroscopic27

    mask = _case_mask(box) if mask is None else mask
    return _run_campaign(
        box,
        collide,
        mask,
        sem,
        campaign_chain27,
        _initial_f27,
        macroscopic27,
        obstacle_force27,
        bins=bins,
        chunk=chunk,
    )


def _sem_profile(out: CampaignObservables, sem: CampaignSemantics) -> torch.Tensor:
    """The *sem*-selected profile of a finished run (window mean or frames)."""
    if sem.observe_frames:
        return torch.stack([out.frames[s] for s in sem.observe_frames]).mean(dim=0)
    return out.window_profile


def press_profile_campaign(
    box: BoxCase | HullCase,
    re: float,
    rates: Mapping[str, float] | None = None,
    sem: CampaignSemantics = CAMPAIGN_SEMANTICS,
    *,
    mask: torch.Tensor | None = None,
    bins: int = 32,
    chunk: int = 20,
) -> tuple[torch.Tensor, float]:
    """D3Q19-MRT shell-pressure profile and windowed C_D under *sem*.

    The stage-6 counterpart of :func:`press_profile`: same MRT family and
    rates, same shell binning, but the chain definitions (initialisation,
    collide-in-solid, outlet, mass correction, snapshot readout) follow
    *sem* — :data:`CAMPAIGN_SEMANTICS` reproduces the production campaign
    pipeline definition.  Returns the *sem*-selected profile (window mean,
    or the mean of the campaign snapshot frames) and the windowed Ladd C_D
    (kept in the historical convention so C_D-neutrality comparisons stay
    like-for-like).
    """
    rates = dict(DEFAULT_MRT_RATES) if rates is None else dict(rates)
    collide = _rate_collide(box.tau_of_re(re), rates)
    out = campaign_rollout19(box, collide, sem, mask=mask, bins=bins, chunk=chunk)
    return _sem_profile(out, sem), out.window_cd


def press_profile27_campaign(
    box: BoxCase | HullCase,
    re: float,
    rates: Mapping[str, float] | None = None,
    sem: CampaignSemantics = CAMPAIGN_SEMANTICS,
    *,
    family: str = "cumulant",
    mask: torch.Tensor | None = None,
    bins: int = 32,
    chunk: int = 20,
) -> tuple[torch.Tensor, float]:
    """D3Q27 counterpart of :func:`press_profile_campaign`."""
    rates = _default27_rates(family) if rates is None else dict(rates)
    collide = _rate27_collide(box.tau_of_re(re), rates, family)
    out = campaign_rollout27(box, collide, sem, mask=mask, bins=bins, chunk=chunk)
    return _sem_profile(out, sem), out.window_cd


def _diff_campaign_profile(
    box: BoxCase | HullCase,
    re: float,
    rates: dict[str, torch.Tensor],
    sem: CampaignSemantics,
    mask: torch.Tensor,
    bins: int,
    block: int,
    chain: Callable[..., Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]],
    collide_of: Callable[[float, dict[str, torch.Tensor]], Callable[..., torch.Tensor]],
    initial_f: Callable[[BoxCase | HullCase, torch.Tensor, CampaignSemantics], torch.Tensor],
    macroscopic: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
) -> torch.Tensor:
    """Differentiable *sem* profile at *re* (block-checkpointed window).

    Mirrors the ``diff_profile`` closure of :func:`calibrate_mrt_rates` /
    :func:`calibrate_collision27`: no-grad transient (with the *sem* mass
    corrections), then the window ``(window_start, steps]`` differentiated
    through ``block``-step checkpoints.  The readout follows ``sem``: the
    per-step probe mean (legacy) or the campaign snapshot frames listed in
    ``sem.observe_frames`` (each must fall inside the window; the final-step
    frame is the campaign export definition).
    """
    from functools import partial

    from torch.utils.checkpoint import checkpoint

    from tensorlbm.solver3d import correct_mass3d

    device = torch.device(box.device)
    cz, ays, axs, bin_idx = _plane_shell(mask, bins)
    counts = torch.bincount(bin_idx, minlength=bins).clamp(min=1).double()
    window = box.steps - box.window_start
    frames = sorted(sem.observe_frames)
    for s in frames:
        if not box.window_start < s <= box.steps:
            raise ValueError(
                f"observe frame {s} must lie inside the differentiable window "
                f"({box.window_start}, {box.steps}]"
            )
    step = chain(box, collide_of(re, rates), mask, sem)

    with torch.no_grad():
        f = initial_f(box, mask, sem)
        m0 = float(f.sum())
        for s in range(1, box.window_start + 1):
            f, _probe = step(f)
            if sem.correct_at(s):
                f = correct_mass3d(f, m0)
        f = f.detach()

    def run_block(f: torch.Tensor, s0: int) -> tuple[torch.Tensor, torch.Tensor]:
        acc = torch.zeros(bins, dtype=box.dtype, device=device)
        for k in range(block):
            f, probe = step(f)
            s = s0 + k + 1
            if sem.correct_at(s):
                f = correct_mass3d(f, m0)
            if frames:
                if s in frames:
                    rho = macroscopic(f)[0]  # post-correction export state
                    acc = acc + _shell_acc(rho, cz, ays, axs, bin_idx, bins, box.dtype, device)
            else:
                rho = macroscopic(probe)[0]
                acc = acc + _shell_acc(rho, cz, ays, axs, bin_idx, bins, box.dtype, device)
        return f, acc

    total = torch.zeros(bins, dtype=box.dtype, device=device)
    for b in range(window // block):
        f, acc = checkpoint(
            partial(run_block, s0=box.window_start + b * block), f, use_reentrant=False
        )
        total = total + acc
    n_avg = float(len(frames)) if frames else float(window)
    return (total.double() / n_avg) / counts


@dataclass(frozen=True)
class CampaignRateCalibResult:
    """Outcome of :func:`calibrate_mrt_rates_campaign` (stage 6)."""

    sem: CampaignSemantics
    rates: dict[str, float]
    loss_history: list[float] = field(default_factory=list)
    eval: dict[str, dict[str, float]] = field(default_factory=dict)


def _evaluate_campaign(
    box: BoxCase | HullCase,
    sem: CampaignSemantics,
    mask: torch.Tensor,
    bins: int,
    targets: Mapping[float, torch.Tensor],
    final_rates: Mapping[str, float],
    press: Callable[..., tuple[torch.Tensor, float]],
) -> dict[str, dict[str, float]]:
    """Before/after evaluation: frame + window profile gaps and C_D, per Re."""
    sem_win = sem.with_frames()
    evaluation: dict[str, dict[str, float]] = {}
    for re, t in targets.items():
        n = torch.linalg.vector_norm(t)
        p0, cd0 = press(box, re, None, sem, mask=mask, bins=bins)
        p1, cd1 = press(box, re, dict(final_rates), sem, mask=mask, bins=bins)
        w0, _ = press(box, re, None, sem_win, mask=mask, bins=bins)
        w1, _ = press(box, re, dict(final_rates), sem_win, mask=mask, bins=bins)
        evaluation[f"{re:g}"] = {
            "press_before": float(torch.linalg.vector_norm(p0 - t) / n),
            "press_after": float(torch.linalg.vector_norm(p1 - t) / n),
            "window_before": float(torch.linalg.vector_norm(w0 - t) / n),
            "window_after": float(torch.linalg.vector_norm(w1 - t) / n),
            "cd_before": cd0,
            "cd_after": cd1,
        }
    return evaluation


def calibrate_mrt_rates_campaign(
    targets: Mapping[float, torch.Tensor],
    box: BoxCase | HullCase,
    sem: CampaignSemantics = CAMPAIGN_SEMANTICS,
    *,
    mask: torch.Tensor | None = None,
    bins: int = 32,
    iters: int = 15,
    lr: float = 0.03,
    bounds: tuple[float, float] = (0.5, 2.0),
    block: int = 25,
    log_every: int = 0,
) -> CampaignRateCalibResult:
    """Stage-6 :func:`calibrate_mrt_rates` under campaign semantics *sem*.

    Same protocol (Adam on the raw rates, clamped to *bounds*, loss = mean
    squared relative L2 against the targets, block-checkpointed
    differentiable window) on the *sem* chain; ``sem.observe_frames``
    selects the export definition the loss reads — pass the campaign
    snapshot step(s) to fit the production observable definition.
    """
    device = torch.device(box.device)
    mask = _case_mask(box) if mask is None else mask
    window = box.steps - box.window_start
    if window % block:
        raise ValueError(f"window {window} not divisible by block {block}")
    tgt = {re: t.to(device=device, dtype=torch.float64) for re, t in targets.items()}
    tnorm = {re: float(torch.linalg.vector_norm(t) ** 2) for re, t in tgt.items()}

    rates = {
        k: torch.tensor(v, dtype=box.dtype, device=device, requires_grad=True)
        for k, v in DEFAULT_MRT_RATES.items()
    }

    def _collide_of(re: float, r: dict[str, torch.Tensor]) -> Callable[..., torch.Tensor]:
        return _rate_collide(box.tau_of_re(re), r)

    def diff_profile(re: float) -> torch.Tensor:
        return _diff_campaign_profile(
            box,
            re,
            rates,
            sem,
            mask,
            bins,
            block,
            campaign_chain19,
            _collide_of,
            _initial_f19,
            macroscopic3d,
        )

    opt = torch.optim.Adam(rates.values(), lr=lr)
    hist: list[float] = []
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        losses = [((diff_profile(re) - t) ** 2).sum() / tnorm[re] for re, t in tgt.items()]
        total = torch.stack(losses).mean()
        if not math.isfinite(float(total.detach())):
            print(f"mrt-campaign: rollout diverged at iter {it}; stopping", flush=True)
            break
        torch.autograd.backward(total)
        opt.step()
        with torch.no_grad():
            for r in rates.values():
                r.clamp_(*bounds)
        hist.append(float(total.detach()))
        if log_every and it % log_every == 0:
            cur = {k: round(float(v.detach()), 4) for k, v in rates.items()}
            print(f"mrt-camp it{it:02d} loss={hist[-1]:.6f} {cur}", flush=True)

    final = {k: float(v.detach()) for k, v in rates.items()}

    def _press(
        b: BoxCase | HullCase,
        re: float,
        r: Mapping[str, float] | None,
        s: CampaignSemantics,
        *,
        mask: torch.Tensor | None = None,
        bins: int = 32,
    ) -> tuple[torch.Tensor, float]:
        return press_profile_campaign(
            b, re, None if r is None else dict(r), s, mask=mask, bins=bins
        )

    evaluation = _evaluate_campaign(box, sem, mask, bins, tgt, final, _press)
    return CampaignRateCalibResult(sem=sem, rates=final, loss_history=hist, eval=evaluation)


@dataclass(frozen=True)
class CampaignCollision27CalibResult:
    """Outcome of :func:`calibrate_collision27_campaign` (stage-6 control)."""

    family: str
    sem: CampaignSemantics
    rates: dict[str, float]
    loss_history: list[float] = field(default_factory=list)
    eval: dict[str, dict[str, float]] = field(default_factory=dict)


def calibrate_collision27_campaign(
    targets: Mapping[float, torch.Tensor],
    box: BoxCase | HullCase,
    sem: CampaignSemantics = CAMPAIGN_SEMANTICS,
    *,
    family: str = "cumulant",
    mask: torch.Tensor | None = None,
    bins: int = 32,
    iters: int = 15,
    lr: float = 0.03,
    bounds: tuple[float, float] = (0.6, 1.6),
    block: int = 25,
    log_every: int = 0,
) -> CampaignCollision27CalibResult:
    """Stage-6 :func:`calibrate_collision27` under campaign semantics *sem*.

    The D3Q27 control: same guarded protocol (NaN revert-and-stop, tighter
    default bounds — 27-stencil ghost-rate calibration is not C_D-neutral,
    stage 5) on the *sem* chain.  Only the knobs expressible on the 27 chain
    apply (initialisation, collide-in-solid, mass correction, snapshot
    readout; the outlet is the production full copy either way).
    """
    from tensorlbm.d3q27 import macroscopic27

    if family not in COLLISION27_FAMILIES:
        raise ValueError(f"family must be one of {COLLISION27_FAMILIES}, got {family!r}")
    device = torch.device(box.device)
    mask = _case_mask(box) if mask is None else mask
    window = box.steps - box.window_start
    if window % block:
        raise ValueError(f"window {window} not divisible by block {block}")
    tgt = {re: t.to(device=device, dtype=torch.float64) for re, t in targets.items()}
    tnorm = {re: float(torch.linalg.vector_norm(t) ** 2) for re, t in tgt.items()}
    defaults = _default27_rates(family)

    rates = {
        k: torch.tensor(v, dtype=box.dtype, device=device, requires_grad=True)
        for k, v in defaults.items()
    }

    def _collide_of(re: float, r: dict[str, torch.Tensor]) -> Callable[..., torch.Tensor]:
        return _rate27_collide(box.tau_of_re(re), r, family)

    def diff_profile(re: float) -> torch.Tensor:
        return _diff_campaign_profile(
            box,
            re,
            rates,
            sem,
            mask,
            bins,
            block,
            campaign_chain27,
            _collide_of,
            _initial_f27,
            macroscopic27,
        )

    opt = torch.optim.Adam(rates.values(), lr=lr)
    hist: list[float] = []
    last_good = dict(defaults)
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        losses = [((diff_profile(re) - t) ** 2).sum() / tnorm[re] for re, t in tgt.items()]
        total = torch.stack(losses).mean()
        if not math.isfinite(float(total.detach())):
            with torch.no_grad():
                for k, r in rates.items():
                    r.fill_(last_good[k])
            print(
                f"{family}-campaign: rollout diverged at iter {it}; "
                f"reverted to last finite rates {last_good}",
                flush=True,
            )
            break
        torch.autograd.backward(total)
        opt.step()
        with torch.no_grad():
            for r in rates.values():
                r.clamp_(*bounds)
            last_good = {k: float(v.detach()) for k, v in rates.items()}
        hist.append(float(total.detach()))
        if log_every and it % log_every == 0:
            cur = {k: round(float(v.detach()), 4) for k, v in rates.items()}
            print(f"{family}-camp it{it:02d} loss={hist[-1]:.6f} {cur}", flush=True)

    final = {k: float(v.detach()) for k, v in rates.items()}

    def _press(
        b: BoxCase | HullCase,
        re: float,
        r: Mapping[str, float] | None,
        s: CampaignSemantics,
        *,
        mask: torch.Tensor | None = None,
        bins: int = 32,
    ) -> tuple[torch.Tensor, float]:
        rr = None if r is None else dict(r)
        return press_profile27_campaign(b, re, rr, s, family=family, mask=mask, bins=bins)

    evaluation = _evaluate_campaign(box, sem, mask, bins, tgt, final, _press)
    return CampaignCollision27CalibResult(
        family=family, sem=sem, rates=final, loss_history=hist, eval=evaluation
    )
