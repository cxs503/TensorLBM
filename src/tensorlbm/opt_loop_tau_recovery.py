"""End-to-end ``tau`` / ``Re`` recovery on the Taylor-Green decay (optimisation loop).

Motivation (paper牵引, JOUR-DIFFLBM section 4 / inverse experiment): the
differentiable step chain (:mod:`tensorlbm.autograd_path`) puts the discrete
time-stepping operator itself in the backward graph, so a *physical parameter
of the solver* - here the BGK relaxation time :math:`\\tau`, hence the
kinematic viscosity :math:`\\nu = (\\tau - 1/2)/3` and the Reynolds number -
can be identified from observed macroscopic fields by plain gradient descent.
What was missing was an **end-to-end, machine-checkable demo**: a decay case
whose answer is known independently, a documented method selection, a machine
readable convergence record, and a reproducibility protocol.

Physics setup
-------------
Taylor-Green vortex in a periodic box, two velocity components in the plane::

    u_x(x, y, t) =  A u_0 sin(kx) cos(ky) exp(-2 \\nu k^2 t)
    u_y(x, y, t) = -A u_0 cos(kx) sin(ky) exp(-2 \\nu k^2 t)

with :math:`k = 2\\pi/L` and :math:`L = n` lattice spacings (one wavelength per
box).  This is an **exact** solution of the incompressible Navier-Stokes
equations and, at the same time, the LBM decay mode under BGK: with the
equilibrium initial condition the simulated field decays as
:math:`\\exp(-2\\nu k^2 t)` with :math:`\\nu = (\\tau-1/2)/3` in lattice units.
The D3Q19 lattice in a single-``z`` plane reproduces the two-dimensional decay
bit-for-bit in this configuration (checked: ``nz = 1, 2, 4`` give identical
``max|u_x|``), so ``nz = 1`` is used (`test_opt_loop_tau_recovery.py`).

Method selection (what was tried, what is used, why)
---------------------------------------------------
The observable is a *sequence* of macroscopic velocity fields sampled at a few
times after the acoustic startup transient; the unknown is :math:`\\tau` (plus,
optionally, the initial amplitude ``A`` as a nuisance parameter).  Loss::

    L(tau, A) = (1/N_t) \\sum_t ||u_sim(t; tau, A) - u_obs(t)||^2 / ||u_obs(t)||^2

Two selections matter and both are recorded as tracks of the demo:

1. **Amplitude nuisance is required.**  The initial condition is built from the
   lattice-sampled analytic field; the discrete maximum of
   ``sin(kx)cos(ky)`` on the cell centres is :math:`0.962` of the continuous
   one at :math:`n=16`, and the equilibrium start adds a small compressibility
   offset.  The simulated field is then the analytic one times a nearly
   *constant* factor (:math:`\\approx 0.9755`), which a fit with a pinned
   amplitude can only absorb by *slowing the decay*, i.e. by biasing
   :math:`\\nu` upward.  Measured bias at :math:`n=16`: ``+3.9%`` (fails the
   3% gate); with the amplitude profiled out: ``-0.10%``.  Track
   ``twin16_fixed`` keeps the biased selection on purpose, as the control arm.
2. **Optimise** :math:`\\log \\nu` (not :math:`\\tau`) so the positivity /
   stability bound :math:`\\tau > 1/2` is a box constraint after the step
   instead of a barrier in the loss; Adam with cosine-decayed learning rate,
   the gradient coming from back-propagation through every LBM step.

Observations are injectable: ``analytic`` fields (physics anchor: exact NS
solution, so the residual error measures the *whole* pipeline against an
external truth) or fields produced by the same solver at a known
``tau_true`` (twin experiment: verifies the optimiser itself, the recoverable
optimum is exact).  Gaussian noise can be added for the robustness track.

Reproducibility: every run is seeded (initial guess jitter **and** observation
noise), the full per-iteration history is written to a machine-readable JSON
(:data:`SCHEMA`), and the same seed reproduces the same numbers.

Usage::

    python -m tensorlbm.opt_loop_tau_recovery --out artifacts/opt_loop_tau_recovery.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from tensorlbm.autograd_path import rollout
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d

__all__ = [
    "SCHEMA",
    "Observations",
    "RecoverResult",
    "TaylorGreenCase",
    "analytic_observations",
    "gradient_check",
    "recover",
    "run_demo",
    "twin_observations",
]

SCHEMA = "tensorlbm/opt-loop-tau-recovery/v1"

#: BGK stability window used for the box constraint on ``tau``.
TAU_MIN = 0.5005
TAU_MAX = 1.9


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaylorGreenCase:
    """Periodic Taylor-Green box, one wavelength across ``n`` lattice spacings.

    Attributes
    ----------
    n:
        Number of nodes per side (``k = 2 pi / n``); ``n`` is also the box
        length used in :math:`Re = u_0 L / \\nu`.
    u0:
        Peak initial velocity in lattice units (``0.05`` keeps
        :math:`Ma \\approx 0.087`, i.e. compressibility error of order
        :math:`Ma^2 \\approx 0.8\\%`).
    nz:
        Lattice layers in ``z``; one layer is enough for this 2-D decay (see
        module docstring) and is the default.
    dtype:
        Torch dtype; ``float64`` is the default because the demonstration is
        about derivative/optimiser accuracy, not throughput.
    """

    n: int = 16
    u0: float = 0.05
    nz: int = 1
    dtype: torch.dtype = torch.float64

    @property
    def k(self) -> float:
        return 2.0 * math.pi / self.n

    def tau_from_nu(self, nu: float) -> float:
        return 0.5 + 3.0 * nu

    def nu_from_tau(self, tau: float) -> float:
        return (tau - 0.5) / 3.0

    def reynolds(self, nu: float) -> float:
        """``Re = u0 * L / nu`` with ``L = n`` lattice units (box length)."""
        return self.u0 * self.n / nu

    # -- fields -----------------------------------------------------------

    def analytic(self, t: float, nu: float, amplitude: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact Taylor-Green velocity field sampled on the cell centres."""
        k = self.k
        x = (torch.arange(self.n, dtype=self.dtype) + 0.5) * k
        y, xx = torch.meshgrid(x, x, indexing="ij")
        decay = math.exp(-2.0 * nu * k * k * t)
        scale = amplitude * self.u0 * decay
        ux = scale * torch.sin(xx) * torch.cos(y)
        uy = -scale * torch.cos(xx) * torch.sin(y)
        shape = (self.nz, self.n, self.n)
        return ux.expand(shape).contiguous(), uy.expand(shape).contiguous()

    def initial_field(self, amplitude: float | torch.Tensor = 1.0):
        """Equilibrium populations carrying the analytic field at ``t = 0``.

        ``amplitude`` may be a graph-connected scalar (the nuisance parameter
        of the inversion), in which case its gradient reaches the loss.
        """
        ux, uy = self.analytic(0.0, 0.0, amplitude=1.0)
        zeros = torch.zeros_like(ux)
        rho = torch.ones((self.nz, self.n, self.n), dtype=self.dtype)
        return equilibrium3d(rho, ux * amplitude, uy * amplitude, zeros)

    def times(self, nu: float, *, decay: float = 1.0, skip: float = 0.1, n_obs: int = 4) -> tuple[int, ...]:
        """Observation steps spanning ``decay`` e-folds of amplitude.

        The first ``skip`` fraction of the window is dropped so that the
        acoustic startup transient (the equilibrium initial condition does not
        carry the :math:`O(Ma^2)` pressure field) is not part of the data.
        """
        rate = 2.0 * nu * self.k * self.k
        t_end = decay / rate
        step_end = int(round(t_end))
        start = int(round(skip * step_end))
        span = step_end - start
        return tuple(start + max(1, round(span * i / (n_obs - 1))) for i in range(n_obs))


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@dataclass
class Observations:
    """Velocity fields at the observation steps, plus provenance."""

    times: tuple[int, ...]
    fields: list[tuple[torch.Tensor, torch.Tensor]]
    source: str = "analytic"
    noise: float = 0.0
    seed: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "times": list(self.times),
            "source": self.source,
            "noise": self.noise,
            "seed": self.seed,
            "n_fields": len(self.fields),
            "norms": [float(sum((c * c).sum() for c in pair)) for pair in self.fields],
        }


def _add_noise(
    fields: Iterable[tuple[torch.Tensor, torch.Tensor]],
    case: TaylorGreenCase,
    noise: float,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if noise <= 0.0:
        return [tuple(pair) for pair in fields]  # type: ignore[misc]
    gen = torch.Generator().manual_seed(int(seed) + 7919)
    sigma = noise * case.u0
    out = []
    for ux, uy in fields:
        out.append(
            (
                ux + sigma * torch.randn(ux.shape, generator=gen, dtype=case.dtype),
                uy + sigma * torch.randn(uy.shape, generator=gen, dtype=case.dtype),
            )
        )
    return out


def analytic_observations(
    case: TaylorGreenCase,
    nu: float,
    times: Sequence[int],
    *,
    noise: float = 0.0,
    seed: int = 0,
) -> Observations:
    """Observations from the exact Navier-Stokes solution (physics anchor)."""
    raw = [case.analytic(float(t), nu) for t in times]
    return Observations(
        times=tuple(int(t) for t in times),
        fields=_add_noise(raw, case, noise, seed),
        source="analytic",
        noise=noise,
        seed=seed,
    )


def twin_observations(
    case: TaylorGreenCase,
    tau_true: float,
    times: Sequence[int],
    *,
    noise: float = 0.0,
    seed: int = 0,
) -> Observations:
    """Observations from the same differentiable solver at ``tau_true``.

    This is the *verification* mode: the recoverable optimum is exactly known
    (no model discrepancy), so the residual error measures the optimiser and
    the gradient, not the physics.
    """
    tau = torch.tensor(float(tau_true), dtype=case.dtype)
    f = case.initial_field()
    raw: list[tuple[torch.Tensor, torch.Tensor]] = []
    t_prev = 0
    with torch.no_grad():
        for t in times:
            f = rollout(f, int(t) - t_prev, tau)
            t_prev = int(t)
            _, ux, uy, _ = macroscopic3d(f)
            raw.append((ux.clone(), uy.clone()))
    return Observations(
        times=tuple(int(t) for t in times),
        fields=_add_noise(raw, case, noise, seed),
        source="twin",
        noise=noise,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Forward observable and loss
# ---------------------------------------------------------------------------


def simulate(
    case: TaylorGreenCase,
    tau: torch.Tensor | float,
    times: Sequence[int],
    amplitude: torch.Tensor | float = 1.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Differentiable velocity fields at ``times``; gradients reach ``tau``/``amplitude``."""
    f = case.initial_field(amplitude)
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    t_prev = 0
    for t in times:
        n_steps = int(t) - t_prev
        if n_steps > 0:
            f = rollout(f, n_steps, tau)
            t_prev = int(t)
        _, ux, uy, _ = macroscopic3d(f)
        out.append((ux, uy))
    return out


def loss_and_norms(
    sim: Sequence[tuple[torch.Tensor, torch.Tensor]],
    obs: Observations,
) -> tuple[torch.Tensor, list[float]]:
    """Relative-field loss: each observation time contributes one unit of weight."""
    total = None
    per_time: list[float] = []
    for (sx, sy), (ox, oy) in zip(sim, obs.fields):
        num = ((sx - ox) ** 2).mean() + ((sy - oy) ** 2).mean()
        den = (ox**2).mean() + (oy**2).mean()
        term = num / den
        total = term if total is None else total + term
        per_time.append(float(term.detach()))
    assert total is not None
    return total / len(sim), per_time


# ---------------------------------------------------------------------------
# Inversion
# ---------------------------------------------------------------------------


@dataclass
class RecoverResult:
    """Outcome of one seeded inversion, with the full convergence history."""

    track: str
    seed: int
    n: int
    times: tuple[int, ...]
    fit_amplitude: bool
    tau_true: float
    nu_true: float
    re_true: float
    tau_init: float
    iters: int
    lr: float
    tau_hat: float
    nu_hat: float
    re_hat: float
    amplitude_hat: float
    rel_err_pct: float
    loss_initial: float
    loss_final: float
    converged: bool
    wall_s: float
    history: list[dict[str, float]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return abs(self.rel_err_pct) < 3.0

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed_under_3pct"] = self.passed
        return payload


def recover(
    case: TaylorGreenCase,
    obs: Observations,
    *,
    tau_true: float,
    track: str = "demo",
    tau_init: float | None = None,
    iters: int = 20,
    lr: float = 0.05,
    fit_amplitude: bool = True,
    seed: int = 0,
    tol: float = 1e-10,
    polish: int = 0,
) -> RecoverResult:
    """Recover ``tau`` (hence ``nu`` and ``Re``) from ``obs`` by gradient descent.

    The unknown is ``log nu`` so ``tau = 1/2 + 3 nu`` stays physical; Adam with
    a cosine-decayed learning rate; per-iteration ``tau``, ``nu``, ``Re``,
    relative error and loss are recorded for the convergence JSON.
    """
    t_start = time.time()
    nu_true = case.nu_from_tau(tau_true)
    if tau_init is None:
        gen = torch.Generator().manual_seed(int(seed) + 104729)
        jitter = 1.0 + 0.25 * float(2.0 * torch.rand(1, generator=gen, dtype=case.dtype) - 1.0)
        tau_init = case.tau_from_nu(nu_true * jitter)
    nu_init = case.nu_from_tau(tau_init)

    log_nu = torch.tensor(math.log(nu_init), dtype=case.dtype, requires_grad=True)
    amplitude = torch.tensor(1.0, dtype=case.dtype, requires_grad=bool(fit_amplitude))
    params = [log_nu] + ([amplitude] if fit_amplitude else [])
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, iters))

    history: list[dict[str, float]] = []
    loss_value = float("inf")
    loss_prev = float("inf")
    converged = False

    for it in range(iters):
        opt.zero_grad()
        nu = torch.exp(log_nu)
        tau = 0.5 + 3.0 * nu
        sim = simulate(case, tau, obs.times, amplitude=amplitude if fit_amplitude else 1.0)
        loss, per_time = loss_and_norms(sim, obs)
        loss.backward()
        grad_log_nu = float(log_nu.grad) if log_nu.grad is not None else float("nan")
        grad_amp = float(amplitude.grad) if (fit_amplitude and amplitude.grad is not None) else float("nan")
        opt.step()
        sched.step()
        with torch.no_grad():
            lo = math.log((TAU_MIN - 0.5) / 3.0)
            hi = math.log((TAU_MAX - 0.5) / 3.0)
            log_nu.clamp_(min=lo, max=hi)

        nu_v = float(torch.exp(log_nu).detach())
        tau_v = 0.5 + 3.0 * nu_v
        re_v = case.reynolds(nu_v)
        loss_value = float(loss.detach())
        history.append(
            {
                "iter": float(it),
                "phase": "adam",
                "loss": loss_value,
                "per_time": [float(v) for v in per_time],
                "log_nu": float(log_nu.detach()),
                "nu": nu_v,
                "tau": tau_v,
                "reynolds": re_v,
                "rel_err_pct": 100.0 * (nu_v - nu_true) / nu_true,
                "amplitude": float(amplitude.detach()) if fit_amplitude else 1.0,
                "grad_log_nu": grad_log_nu,
                "grad_amplitude": grad_amp,
                "lr": float(sched.get_last_lr()[0]),
            }
        )
        if it > 2 and abs(loss_prev - loss_value) < tol * max(1.0, abs(loss_prev)):
            converged = True
            break
        loss_prev = loss_value

    if polish > 0 and loss_value == loss_value:  # L-BFGS polish on the same objective
        opt_l = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=int(polish),
            history_size=10,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-14,
            tolerance_change=1e-16,
        )

        def closure() -> torch.Tensor:
            opt_l.zero_grad()
            nu_t = torch.exp(log_nu)
            sim_t = simulate(case, 0.5 + 3.0 * nu_t, obs.times, amplitude=amplitude if fit_amplitude else 1.0)
            value, _ = loss_and_norms(sim_t, obs)
            value.backward()
            # every line-search evaluation is part of the convergence record
            nu_e = float(nu_t.detach())
            history.append(
                {
                    "iter": float(len(history)),
                    "phase": "lbfgs_eval",
                    "loss": float(value.detach()),
                    "per_time": [],
                    "log_nu": float(log_nu.detach()),
                    "nu": nu_e,
                    "tau": 0.5 + 3.0 * nu_e,
                    "reynolds": case.reynolds(nu_e),
                    "rel_err_pct": 100.0 * (nu_e - nu_true) / nu_true,
                    "amplitude": float(amplitude.detach()) if fit_amplitude else 1.0,
                    "grad_log_nu": float(log_nu.grad) if log_nu.grad is not None else float("nan"),
                    "grad_amplitude": None,
                    "lr": 1.0,
                }
            )
            return value

        before = loss_value
        opt_l.step(closure)
        with torch.no_grad():
            log_nu.clamp_(min=math.log((TAU_MIN - 0.5) / 3.0), max=math.log((TAU_MAX - 0.5) / 3.0))
        nu_v = float(torch.exp(log_nu.detach()))
        tau_v = 0.5 + 3.0 * nu_v
        with torch.no_grad():
            sim_eval = simulate(case, tau_v, obs.times, amplitude=float(amplitude.detach()) if fit_amplitude else 1.0)
            loss_value = float(loss_and_norms(sim_eval, obs)[0])
        history.append(
            {
                "iter": float(len(history)),
                "phase": "lbfgs",
                "loss": loss_value,
                "per_time": [],
                "log_nu": float(log_nu.detach()),
                "nu": nu_v,
                "tau": tau_v,
                "reynolds": case.reynolds(nu_v),
                "rel_err_pct": 100.0 * (nu_v - nu_true) / nu_true,
                "amplitude": float(amplitude.detach()) if fit_amplitude else 1.0,
                "grad_log_nu": float(log_nu.grad) if log_nu.grad is not None else float("nan"),
                "grad_amplitude": float(amplitude.grad) if (fit_amplitude and amplitude.grad is not None) else float("nan"),
                "lr": 1.0,
                "polish_gain": before - loss_value,
            }
        )
        converged = True

    nu_hat = float(torch.exp(log_nu.detach()))
    tau_hat = 0.5 + 3.0 * nu_hat
    return RecoverResult(
        track=track,
        seed=int(seed),
        n=case.n,
        times=tuple(obs.times),
        fit_amplitude=bool(fit_amplitude),
        tau_true=float(tau_true),
        nu_true=float(nu_true),
        re_true=float(case.reynolds(nu_true)),
        tau_init=float(tau_init),
        iters=len(history),
        lr=float(lr),
        tau_hat=float(tau_hat),
        nu_hat=float(nu_hat),
        re_hat=float(case.reynolds(nu_hat)),
        amplitude_hat=float(amplitude.detach()) if fit_amplitude else 1.0,
        rel_err_pct=100.0 * (nu_hat - nu_true) / nu_true,
        loss_initial=history[0]["loss"],
        loss_final=history[-1]["loss"],
        converged=bool(converged),
        wall_s=time.time() - t_start,
        history=history,
    )


def gradient_check(
    case: TaylorGreenCase,
    obs: Observations,
    *,
    tau: float = 0.55,
    eps: float = 1e-3,
    amplitude: float = 1.0,
) -> dict[str, float]:
    """Compare ``dL/d log nu`` with a central finite difference of the same objective.

    This is the honest-gradient check the inverse experiment stands on: the
    autograd derivative through the whole LBM rollout must agree with the
    finite difference of the *identical* discrete objective.
    """
    log_nu = torch.tensor(math.log(case.nu_from_tau(tau)), dtype=case.dtype, requires_grad=True)

    def objective(log_nu_value: float | torch.Tensor) -> torch.Tensor:
        nu = torch.exp(log_nu_value) if torch.is_tensor(log_nu_value) else math.exp(log_nu_value)
        amp = torch.tensor(amplitude, dtype=case.dtype)
        sim = simulate(case, 0.5 + 3.0 * nu, obs.times, amplitude=amp)
        loss, _ = loss_and_norms(sim, obs)
        return loss

    loss = objective(log_nu)
    loss.backward()
    analytic = float(log_nu.grad)
    hi = objective(math.log(case.nu_from_tau(tau)) + eps)
    lo = objective(math.log(case.nu_from_tau(tau)) - eps)
    fd = float((hi - lo) / (2.0 * eps))
    denom = max(abs(fd), 1e-30)
    return {
        "autograd": analytic,
        "finite_difference": fd,
        "rel_diff": abs(analytic - fd) / denom,
        "eps": eps,
    }


# ---------------------------------------------------------------------------
# Demo driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackSpec:
    """One arm of the demonstration: grid, data source, selection under test."""

    name: str
    n: int
    source: str  # "twin" | "analytic"
    tau_true: float
    fit_amplitude: bool
    noise: float = 0.0
    decay: float = 1.0
    control_arm: bool = False
    iters: int = 2
    lr: float = 0.3
    polish: int = 10


DEFAULT_TRACKS: tuple[TrackSpec, ...] = (
    # physics anchor: exact NS solution, amplitude nuisance profiled out
    TrackSpec("analytic16", 16, "analytic", 0.55, True),
    TrackSpec("analytic32", 32, "analytic", 0.55, True),
    # verification arm: same solver, the recoverable optimum is exactly known
    TrackSpec("twin16", 16, "twin", 0.55, True),
    # control arm: amplitude pinned -> the documented bias (method selection evidence);
    # it only bites on the analytic observations (the twin data carry exactly A = 1)
    TrackSpec("analytic16_fixed", 16, "analytic", 0.55, False, control_arm=True),
    # robustness: observation noise
    TrackSpec("twin16_noise0.2", 16, "twin", 0.55, True, noise=0.002),
    TrackSpec("twin16_noise1.0", 16, "twin", 0.55, True, noise=0.01),
)

TRACK_ORDER = {spec.name: i for i, spec in enumerate(DEFAULT_TRACKS)}


def _track_spec(spec: TrackSpec) -> dict[str, Any]:
    return {
        "source": spec.source,
        "fit_amplitude": spec.fit_amplitude,
        "noise_rel": spec.noise,
        "decay_e_folds": spec.decay,
        "iters": spec.iters,
        "lr": spec.lr,
        "lbfgs_polish": spec.polish,
        "control_arm": spec.control_arm,
    }


def run_job(spec: TrackSpec, seed: int, *, with_gradient_check: bool = False) -> dict[str, Any]:
    """Run a single (track, seed) inversion and return a self-describing record."""
    case = TaylorGreenCase(n=spec.n)
    nu_true = case.nu_from_tau(spec.tau_true)
    times = case.times(nu_true, decay=spec.decay)
    if spec.source == "twin":
        obs = twin_observations(case, spec.tau_true, times, noise=spec.noise, seed=int(seed))
    else:
        obs = analytic_observations(case, nu_true, times, noise=spec.noise, seed=int(seed))
    grad = None
    if with_gradient_check:
        grad = gradient_check(case, obs, tau=spec.tau_true, eps=1e-3)
    result = recover(
        case,
        obs,
        tau_true=spec.tau_true,
        track=spec.name,
        iters=spec.iters,
        lr=spec.lr,
        fit_amplitude=spec.fit_amplitude,
        seed=int(seed),
        polish=spec.polish,
    )
    return {
        "track": spec.name,
        "seed": int(seed),
        "selection": _track_spec(spec),
        "case": {
            "n": case.n,
            "nz": case.nz,
            "u0": case.u0,
            "k": case.k,
            "box_length_lattice_units": case.n,
            "dtype": str(case.dtype),
            "tau_true": spec.tau_true,
            "nu_true": nu_true,
            "re_true": case.reynolds(nu_true),
            "observation_times": list(times),
        },
        "observation": obs.to_payload(),
        "gradient_check": grad,
        "run": result.to_payload(),
    }


def _track_entry(track: str, jobs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    jobs = sorted(jobs, key=lambda j: j["seed"])
    first = jobs[0]
    errs = [j["run"]["rel_err_pct"] for j in jobs]
    grad = next((j["gradient_check"] for j in jobs if j.get("gradient_check")), None)
    return {
        "track": track,
        "selection": first["selection"],
        "case": first["case"],
        "observation": first["observation"],
        "gradient_check": grad,
        "runs": [j["run"] for j in jobs],
        "summary": {
            "seeds": [j["seed"] for j in jobs],
            "n_runs": len(jobs),
            "rel_err_pct": errs,
            "max_abs_rel_err_pct": max(abs(e) for e in errs),
            "mean_abs_rel_err_pct": sum(abs(e) for e in errs) / len(errs),
            "spread_pct": max(errs) - min(errs),
            "all_under_3pct": all(abs(e) < 3.0 for e in errs),
            "tau_hat": [j["run"]["tau_hat"] for j in jobs],
            "re_hat": [j["run"]["re_hat"] for j in jobs],
            "amplitude_hat": [j["run"]["amplitude_hat"] for j in jobs],
            "wall_s": [j["run"]["wall_s"] for j in jobs],
        },
    }


def assemble(jobs: Sequence[dict[str, Any]], seeds: Sequence[int]) -> dict[str, Any]:
    """Group single-run records into the canonical machine-readable payload."""
    by_track: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        by_track.setdefault(job["track"], []).append(job)
    order = sorted(by_track, key=lambda t: TRACK_ORDER.get(t, 10**6))
    tracks = [_track_entry(t, by_track[t]) for t in order]
    return {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "torch_version": torch.__version__,
        "seeds": list(int(s) for s in seeds),
        "tracks": tracks,
        "summary": {
            "tracks": [t["track"] for t in tracks],
            "by_track": {
                t["track"]: {
                    "max_abs_rel_err_pct": t["summary"]["max_abs_rel_err_pct"],
                    "all_under_3pct": t["summary"]["all_under_3pct"],
                    "tau_hat": t["summary"]["tau_hat"],
                    "re_hat": t["summary"]["re_hat"],
                }
                for t in tracks
            },
            "all_tracks_pass": all(
                t["summary"]["all_under_3pct"] for t in tracks if not t["selection"].get("control_arm")
            ),
            "control_arms": [t["track"] for t in tracks if t["selection"].get("control_arm")],
            "criterion": "TG decay tau/Re recovery < 3%; 3 seeds reproducible; machine-readable convergence JSON",
        },
    }


def run_demo(
    *,
    seeds: Sequence[int] = (0, 1, 2),
    tracks: Sequence[TrackSpec] = DEFAULT_TRACKS,
    progress: bool = True,
) -> dict[str, Any]:
    """Run every track over every seed (single process; the CLI shards for HPC)."""
    jobs = []
    for spec in tracks:
        for k, seed in enumerate(seeds):
            job = run_job(spec, int(seed), with_gradient_check=(k == 0))
            jobs.append(job)
            if progress:
                print(
                    f"[{spec.name} seed={seed}] tau_hat={job['run']['tau_hat']:.6f} "
                    f"Re_hat={job['run']['re_hat']:.4f} err={job['run']['rel_err_pct']:+.4f}%",
                    flush=True,
                )
    return assemble(jobs, seeds)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/opt_loop_tau_recovery.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--tracks", nargs="+", default=None, help="subset of track names")
    parser.add_argument("--iters", type=int, default=None, help="override the track iteration count")
    parser.add_argument("--polish", type=int, default=None, help="override L-BFGS polish iterations")
    parser.add_argument("--threads", type=int, default=0, help="torch intra-op threads (0 = leave default)")
    parser.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"), default=None,
                        help="run only jobs i::N of the (track x seed) product; writes per-job records")
    parser.add_argument("--merge", nargs="+", type=Path, default=None,
                        help="merge shard files into the canonical payload")
    args = parser.parse_args(argv)

    if args.threads:
        torch.set_num_threads(int(args.threads))

    if args.merge:
        seeds = tuple(int(s) for s in args.seeds)
        merged = []
        for path in args.merge:
            data = json.loads(path.read_text())
            merged.extend(data["jobs"] if isinstance(data, dict) and "jobs" in data else data)
        payload = assemble(merged, seeds)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"merged {len(merged)} runs -> {args.out}")
        return 0 if payload["summary"]["all_tracks_pass"] else 1

    tracks = DEFAULT_TRACKS
    if args.tracks:
        wanted = set(args.tracks)
        tracks = tuple(t for t in DEFAULT_TRACKS if t.name in wanted)
        missing = wanted - {t.name for t in tracks}
        if missing:
            raise SystemExit(f"unknown track(s): {sorted(missing)}")
    if args.iters is not None or args.polish is not None:
        tracks = tuple(
            TrackSpec(
                **{
                    **asdict(t),
                    "iters": int(args.iters) if args.iters is not None else t.iters,
                    "polish": int(args.polish) if args.polish is not None else t.polish,
                }
            )
            for t in tracks
        )

    seeds = tuple(int(s) for s in args.seeds)

    if args.shard:
        i, n = args.shard
        if not (0 <= i < n):
            raise SystemExit("shard index out of range")
        jobs_all = [(spec, seed) for spec in tracks for seed in seeds]
        picked = jobs_all[i::n]
        jobs = []
        for k, (spec, seed) in enumerate(picked):
            job = run_job(spec, seed, with_gradient_check=(seed == seeds[0]))
            jobs.append(job)
            print(
                f"[shard {i}/{n}] {spec.name} seed={seed} tau_hat={job['run']['tau_hat']:.6f} "
                f"err={job['run']['rel_err_pct']:+.4f}%",
                flush=True,
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"schema": SCHEMA, "shard": [i, n], "jobs": jobs}, indent=2) + "\n")
        print(f"wrote {args.out}")
        return 0

    payload = run_demo(seeds=seeds, tracks=tracks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0 if payload["summary"]["all_tracks_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
