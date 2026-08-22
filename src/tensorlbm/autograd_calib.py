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
campaign observations (e.g. a ``scan_runner`` SUBOFF dataset) enter the same
way as :class:`DragTarget` rows — the calibration then finds the closure
that reproduces the measured drag, with the honest caveat that model error
(Smagorinsky itself) is absorbed into the identified constant.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from tensorlbm.autograd_path import InletSpec, OutletSpec, obstacle_force, rollout
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.turbulence import collide_smagorinsky_bgk3d

__all__ = [
    "BoxCase",
    "CalibResult",
    "DragTarget",
    "bounded_drag",
    "calibrate",
    "cs_power",
    "synthetic_targets",
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
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.window_start >= self.steps:
            raise ValueError("window_start must be < steps")
        if self.inlet_method not in ("equilibrium", "zouhe"):
            raise ValueError("inlet_method must be 'equilibrium' or 'zouhe'")
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


@dataclass(frozen=True)
class DragTarget:
    """One drag observation: a Reynolds number and its measured C_D."""

    re: float
    cd: float
    weight: float = 1.0


def _sphere_mask(box: BoxCase) -> torch.Tensor:
    return box.make_mask()


def bounded_drag(
    box: BoxCase,
    re: float,
    cs: float | torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One differentiable bounded rollout; returns the scalar windowed C_D.

    The full map — inlet Dirichlet value, (optional) Smagorinsky collision
    with constant *cs*, streaming, momentum-exchange drag — is autograd-
    connected, so gradients flow to ``cs`` (tensor) and, when the inlet
    velocity is a tensor, through the boundary condition as well.

    Args:
        box: Domain and rollout configuration.
        re: Reynolds number (sets ``tau`` via the house relation).
        cs: Smagorinsky constant; a 0-dim tensor stays in the graph.
            ``None`` runs plain BGK.
        mask: Pre-built solid mask (defaults to ``box.make_mask()``).

    Returns:
        0-dim tensor: mean x-force over the window ``[window_start, steps)``
        normalised by ``0.5·rho0·u_in²·π r²``.
    """
    device = torch.device(box.device)
    mask = _sphere_mask(box) if mask is None else mask
    tau = box.tau_of_re(re)
    inlet = InletSpec(
        ux=torch.as_tensor(box.u_in, dtype=box.dtype, device=device),
        method=box.inlet_method,
    )
    collide = None
    if cs is not None:
        collide_cs = torch.as_tensor(cs, dtype=box.dtype, device=device)
        collide = lambda f, t, _cs=collide_cs: collide_smagorinsky_bgk3d(f, t, C_s=_cs)  # noqa: E731

    rho0 = torch.ones((box.nz, box.ny, box.nx), dtype=box.dtype, device=device)
    u = torch.zeros((3, box.nz, box.ny, box.nx), dtype=box.dtype, device=device)
    u[0] = box.u_in
    f0 = equilibrium3d(rho0, u[0], u[1], u[2])

    _f, probes = rollout(
        f0,
        box.steps,
        tau,
        mask,
        collide=collide,
        inlet=inlet,
        outlet=OutletSpec(),
        return_probes=True,
    )
    window = probes[box.window_start :]
    force = sum(obstacle_force(p, mask)[0] for p in window) / len(window)
    area = math.pi * box.radius**2
    return force / (0.5 * box.u_in**2 * area)


@torch.no_grad()
def synthetic_targets(
    box: BoxCase,
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
    box: BoxCase,
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
    box: BoxCase,
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
