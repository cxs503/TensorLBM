"""Graph-unroll BPTT calibration of an LES parameter, plus independent controls.

The calibrated quantity is the sub-grid model constant ``Cs`` (the Smagorinsky
constant in ``tau_from_cs``), trained by backpropagation through time across an
unrolled LBM trajectory against a sequence of reference velocity fields:

.. math:: J(\\mathrm{Cs}) = \\frac{1}{K}\\sum_{k=1}^{K}
    \\frac{\\lVert \\bar u_{\\rm LES}(t_k) - \\bar u_{\\rm ref}(t_k)\\rVert^2}
         {\\lVert \\bar u_{\\rm ref}(t_k)\\rVert^2}

Four independent routes are provided so the BPTT result can be cross-validated
rather than taken on trust:

* :func:`bptt_calibrate` — autograd through the unrolled graph (the OSS
  ``graph_unroll`` idea).
* :func:`fd_gradient` — central finite difference of the *same* loss (validates
  the gradient, not just the optimum).
* :func:`sweep_loss` / :func:`golden_minimize` — derivative-free evaluation of
  the same loss (validates the optimum, independent of any autograd).
* :func:`measure_eddy_viscosity` — closes the loop on the closure itself: the
  realised eddy viscosity is measured from the linear decay of a single shear
  mode and compared with the analytic prediction :math:`\\nu_t=C_s^2|S|`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .solver2d import collide_stream, equilibrium, macroscopic, unroll

__all__ = [
    "CalibProblem",
    "field_loss",
    "snapshot_errors",
    "bptt_calibrate",
    "fd_gradient",
    "sweep_loss",
    "golden_minimize",
    "measure_eddy_viscosity",
]


@dataclass(frozen=True)
class CalibProblem:
    """Unrolled-trajectory field-matching problem.

    Attributes:
        f0: coarse-grid initial distribution, shape ``(9, nx, ny)``.
        ref_snapshots: ``{step: (ux_ref, uy_ref)}`` filtered reference
            velocities at the *same* grid as ``f0``; every field is matched in
            the relative L2 norm.
        tau0: lattice BGK relaxation time of the coarse grid (same physical
            viscosity as the reference, so ``tau0`` is close to 0.5).
        n_steps: unroll length; must cover ``max(ref_snapshots)``.
        closure: ``"smagorinsky"`` (default) or ``"oss_proxy"``.
    """

    f0: torch.Tensor
    ref_snapshots: dict = field(default_factory=dict)
    tau0: float = 0.5156
    n_steps: int = 300
    closure: str = "smagorinsky"

    @property
    def snapshot_steps(self) -> tuple[int, ...]:
        return tuple(sorted(int(k) for k in self.ref_snapshots))


def _velocities(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, ux, uy = macroscopic(f)
    return ux, uy


def snapshot_errors(problem: CalibProblem, f0: torch.Tensor | None = None,
                    cs=0.0) -> dict[int, float]:
    """Per-snapshot relative L2 velocity error ``||u-u_ref||/||u_ref||``."""
    f_start = problem.f0 if f0 is None else f0
    _, snaps = unroll(
        f_start,
        problem.n_steps,
        problem.tau0,
        cs,
        problem.closure,
        snapshots=problem.snapshot_steps,
    )
    out: dict[int, float] = {}
    for k in problem.snapshot_steps:
        ux, uy = _velocities(snaps[k])
        rx, ry = problem.ref_snapshots[k]
        num = float(((ux - rx) ** 2 + (uy - ry) ** 2).sum())
        den = float((rx * rx + ry * ry).sum())
        out[k] = math.sqrt(num / den)
    return out


def field_loss(problem: CalibProblem, cs=0.0, f0: torch.Tensor | None = None):
    """Relative-L2 snapshot loss (mean over snapshots), differentiable in ``cs``."""
    f_start = problem.f0 if f0 is None else f0
    _, snaps = unroll(
        f_start,
        problem.n_steps,
        problem.tau0,
        cs,
        problem.closure,
        snapshots=problem.snapshot_steps,
    )
    total = None
    for k in problem.snapshot_steps:
        ux, uy = _velocities(snaps[k])
        rx, ry = problem.ref_snapshots[k]
        den = (rx * rx + ry * ry).sum()
        term = ((ux - rx) ** 2 + (uy - ry) ** 2).sum() / den
        total = term if total is None else total + term
    return total / len(problem.snapshot_steps)


def bptt_calibrate(
    problem: CalibProblem,
    cs_init: float = 0.17,
    *,
    iters: int = 40,
    lr: float = 0.05,
    log_param: bool = True,
    lower: float = 1e-3,
    upper: float = 1.0,
):
    """Minimise :func:`field_loss` by BPTT (Adam on ``log10 Cs``).

    Returns a dict with the trajectory of ``(cs, loss)`` pairs, the final
    constant, the final loss and the loss at ``cs_init``.
    """
    raw = torch.tensor(
        math.log10(cs_init) if log_param else float(cs_init), dtype=torch.float64,
        requires_grad=True,
    )
    opt = torch.optim.Adam([raw], lr=lr)

    def to_cs(v):
        return torch.clamp(10.0 ** v if log_param else v, lower, upper)

    history: list[tuple[float, float]] = []
    loss0 = float(field_loss(problem, to_cs(raw.detach())).detach())
    best = (loss0, float(to_cs(raw.detach())))
    for _ in range(int(iters)):
        opt.zero_grad()
        cs = to_cs(raw)
        loss = field_loss(problem, cs)
        loss.backward()
        opt.step()
        with torch.no_grad():
            cur = float(to_cs(raw))
            hist_loss = float(loss)
        history.append((cur, hist_loss))
        if hist_loss < best[0]:
            best = (hist_loss, cur)
    return {
        "cs_init": float(cs_init),
        "cs": best[1],
        "loss": best[0],
        "loss_init": loss0,
        "history": history,
        "iters": int(iters),
    }


def fd_gradient(problem: CalibProblem, cs: float, h: float = 1e-3) -> float:
    """Central finite-difference ``dJ/dCs`` (no autograd involved)."""
    with torch.no_grad():
        jp = float(field_loss(problem, float(cs) + h))
        jm = float(field_loss(problem, float(cs) - h))
    return (jp - jm) / (2.0 * h)


def autograd_gradient(problem: CalibProblem, cs: float) -> float:
    """Autograd ``dJ/dCs`` through the unrolled graph (the BPTT gradient)."""
    v = torch.tensor(float(cs), dtype=torch.float64, requires_grad=True)
    loss = field_loss(problem, v)
    loss.backward()
    return float(v.grad)


def sweep_loss(problem: CalibProblem, grid) -> list[tuple[float, float]]:
    """Loss on an explicit ``Cs`` grid (derivative-free evaluation)."""
    out = []
    with torch.no_grad():
        for cs in grid:
            out.append((float(cs), float(field_loss(problem, float(cs)))))
    return out


def golden_minimize(problem: CalibProblem, lo: float, hi: float, iters: int = 40) -> dict:
    """Golden-section minimisation of :func:`field_loss` on ``[lo, hi]``."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = float(field_loss(problem, c)), float(field_loss(problem, d))
    for _ in range(int(iters)):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = float(field_loss(problem, c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = float(field_loss(problem, d))
    cs = 0.5 * (a + b)
    return {"cs": cs, "loss": float(field_loss(problem, cs)), "bracket": (a, b)}


def measure_eddy_viscosity(
    cs: float,
    *,
    tau0: float = 0.6,
    amp: float = 0.05,
    nx: int = 8,
    ny: int = 32,
    ky: int = 1,
    steps: int = 6,
    closure: str = "smagorinsky",
) -> dict:
    """Realised viscosity of the closure, measured from a decaying shear mode.

    A small-amplitude single-mode shear ``u_x = A sin(k y)`` is the cleanest
    probe of the closure prefactor: ``nu_t = Cs^2 |S|`` with
    ``|S| = A k`` analytically, while the mode itself decays as
    ``ln a(t) = ln a0 - nu k^2 t``.  Running the same probe at ``cs=0`` and at
    ``cs`` isolates ``nu_t`` from lattice discretisation error.

    Returns ``{"nu_bgk", "nu_les", "nu_t_measured", "nu_t_predicted"}`` in
    lattice units.
    """
    k = 2.0 * math.pi * float(ky) / float(ny)  # lattice wavenumber

    def decay(c):
        yy = torch.arange(ny, dtype=torch.float64).view(1, ny) * (2 * math.pi / ny)
        ux = amp * torch.sin(ky * yy) * torch.ones((nx, 1), dtype=torch.float64)
        rho = torch.ones_like(ux)
        f = equilibrium(rho, ux, torch.zeros_like(ux))
        amps = []
        for _ in range(int(steps)):
            f = collide_stream(f, tau0, c, closure)
            _, uxa, _ = macroscopic(f)
            amps.append(float((uxa * torch.sin(ky * yy)).mean()) * 2.0)
        lt = [float(t) for t in range(1, int(steps) + 1)]
        la = [math.log(max(a, 1e-300)) for a in amps]
        n = len(lt)
        mt, ma = sum(lt) / n, sum(la) / n
        slope = sum((t - mt) * (a - ma) for t, a in zip(lt, la)) / sum((t - mt) ** 2 for t in lt)
        return -slope / (k * k)

    nu0 = decay(0.0)
    nu_cs = decay(cs)
    return {
        "cs": float(cs),
        "nu_bgk": nu0,
        "nu_les": nu_cs,
        "nu_t_measured": nu_cs - nu0,
        "nu_t_predicted": (cs ** 2) * amp * k,
        "closure": closure,
    }
