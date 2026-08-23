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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tensorlbm.autograd_path import InletSpec, OutletSpec, WallSpec, obstacle_force, rollout
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.scan_drag import DRAG_HISTORY_SCHEMA
from tensorlbm.turbulence import collide_smagorinsky_bgk3d

__all__ = [
    "BoxCase",
    "CalibResult",
    "DragHistory",
    "DragTarget",
    "HullCase",
    "bounded_drag",
    "calibrate",
    "cd_from_force",
    "cs_power",
    "drag_targets_from_sidecars",
    "evaluate",
    "load_drag_history",
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
) -> torch.Tensor:
    """One differentiable bounded rollout; returns the scalar windowed C_D.

    The full map — inlet Dirichlet value, (optional) Smagorinsky collision
    with constant *cs*, streaming, momentum-exchange drag — is autograd-
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
        cs: Smagorinsky constant; a 0-dim tensor stays in the graph.
            ``None`` runs plain BGK.
        mask: Pre-built solid mask (defaults to ``box.make_mask()``).

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
    collide = None
    if cs is not None:
        collide_cs = torch.as_tensor(cs, dtype=box.dtype, device=device)
        collide = lambda f, t, _cs=collide_cs: collide_smagorinsky_bgk3d(f, t, C_s=_cs)  # noqa: E731

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


@torch.no_grad()
def synthetic_targets(
    box: BoxCase | HullCase,
    re_values: Sequence[float],
    closure: ClosureFn,
) -> list[DragTarget]:
    """Produce verification-mode observations with a known ground truth."""
    return [DragTarget(re=re, cd=float(bounded_drag(box, re, cs=closure(re)))) for re in re_values]


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

    Returns:
        :class:`CalibResult` with identified parameters, the loss history and
        per-target predictions.  The calibration Re grid is *not* baked into
        the closure: evaluate :meth:`CalibResult.cs` at any Re.
    """
    if not targets:
        raise ValueError("targets must be non-empty")
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
            cd_pred = bounded_drag(box, tgt.re, cs=closure(tgt.re), mask=mask)
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
    )


def evaluate(
    result: CalibResult,
    targets: Sequence[DragTarget],
    box: BoxCase | HullCase,
) -> dict[str, dict[str, float]]:
    """Predict C_D at each target Re and compare against the observation.

    Fills ``result.eval`` and returns it: ``{re: {target, pred, abs_err,
    rel_err_pct}}``.  Use with held-out Re to measure closure extrapolation.
    """
    assert result.closure is not None
    mask = box.make_mask()
    out: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for tgt in targets:
            cd_pred = float(bounded_drag(box, tgt.re, cs=result.closure(tgt.re), mask=mask))
            out[f"{tgt.re:g}"] = {
                "target": tgt.cd,
                "pred": cd_pred,
                "abs_err": abs(cd_pred - tgt.cd),
                "rel_err_pct": 100.0 * abs(cd_pred - tgt.cd) / abs(tgt.cd),
            }
    result.eval = out
    return out
