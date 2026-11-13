"""Objective *selection* layer for the graph-unroll BPTT pipeline (TGV anchor).

Round 1 of this XVAL family (``docs/XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER.md``)
showed that the OSS ``graph_unroll``/BPTT *mechanism* is faithfully reproduced
but that a pointwise field-matching objective selects a sub-grid model that is
physically wrong.  This module turns that observation into a **selection
problem**: a finite, pre-declared candidate set

    candidates = {closure} × {objective}

is trained by BPTT (and, as an autograd-free control, by a sweep), the winner
is chosen by the *training* loss only, and the quality of that choice is then
judged on a **held-out** snapshot that no objective ever saw.

Everything here is deterministic: no random seeds, fixed initial guesses, fixed
iteration counts.  All quantities are differentiable tensors until they are
converted to ``float`` for reporting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .solver2d import macroscopic, unroll

__all__ = [
    "OBJECTIVES",
    "SelectionCase",
    "snapshot_fields",
    "kinetic_energy",
    "enstrophy",
    "objective_loss",
    "bptt_select",
    "fd_gradient",
    "sweep_select",
    "golden_select",
    "holdout_metrics",
    "spectrum_np",
    "select_and_judge",
]

#: Objectives supported by :func:`objective_loss`.
#:
#: ``field``       relative-L2 velocity-field mismatch (round-1 / OSS-style)
#: ``energy``      relative error of the kinetic-energy decay curve (integral)
#: ``enstrophy``   relative error of the enstrophy decay curve (integral)
OBJECTIVES = ("field", "energy", "enstrophy")


@dataclass(frozen=True)
class SelectionCase:
    """One BPTT trajectory plus the reference it is compared with.

    Attributes:
        f0: coarse-grid initial distribution ``(9, nx, ny)``.
        ref: ``{step: (ux, uy)}`` reference velocities (numpy, same grid).
        tau0: coarse-grid BGK relaxation time.
        n_steps: unroll length; must be ``>= max(ref)``.
        closure: ``"smagorinsky"`` or ``"oss_proxy"``.
        train_steps: snapshots the objectives are evaluated on.
        holdout_step: snapshot that no objective may see (judged only).
    """

    f0: torch.Tensor
    ref: dict = field(default_factory=dict)
    tau0: float = 0.515563
    n_steps: int = 200
    closure: str = "smagorinsky"
    train_steps: tuple = (50, 100, 150)
    holdout_step: int = 200

    @property
    def all_steps(self) -> tuple:
        return tuple(sorted(set(int(s) for s in self.train_steps) | {int(self.holdout_step)}))


def _ref_tensors(case: SelectionCase, steps) -> dict:
    out = {}
    for k in steps:
        ux, uy = case.ref[int(k)]
        out[int(k)] = (
            torch.as_tensor(np.asarray(ux), dtype=torch.float64),
            torch.as_tensor(np.asarray(uy), dtype=torch.float64),
        )
    return out


def snapshot_fields(case: SelectionCase, theta, *, steps=None) -> dict:
    """Unroll once and return ``{step: (ux, uy)}`` (autograd-connected in θ)."""
    want = case.all_steps if steps is None else tuple(int(s) for s in steps)
    _, snaps = unroll(
        case.f0, case.n_steps, case.tau0, theta, case.closure, snapshots=tuple(want)
    )
    out = {}
    for k in want:
        _, ux, uy = macroscopic(snaps[k])
        out[k] = (ux, uy)
    return out


def kinetic_energy(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Domain-averaged kinetic energy ``0.5 <u²>`` (differentiable)."""
    return 0.5 * (ux * ux + uy * uy).mean()


def _vorticity(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Central-difference vorticity on the periodic torus (differentiable)."""
    duy_dx = 0.5 * (torch.roll(uy, -1, dims=0) - torch.roll(uy, 1, dims=0))
    dux_dy = 0.5 * (torch.roll(ux, -1, dims=1) - torch.roll(ux, 1, dims=1))
    return duy_dx - dux_dy


def enstrophy(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Domain-averaged enstrophy ``0.5 <ω²>`` (differentiable)."""
    w = _vorticity(ux, uy)
    return 0.5 * (w * w).mean()


def objective_loss(case: SelectionCase, objective: str, theta, *, steps=None):
    """Differentiable objective on the *training* snapshots.

    Args:
        case: selection case.
        objective: one of :data:`OBJECTIVES`.
        theta: sub-grid constant (Python float or graph-connected tensor).
        steps: override the training snapshots (secondary window study).
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    tn = tuple(int(s) for s in (case.train_steps if steps is None else steps))
    fields = snapshot_fields(case, theta, steps=tn)
    ref = _ref_tensors(case, tn)

    if objective == "field":
        total = None
        for k in tn:
            ux, uy = fields[k]
            rx, ry = ref[k]
            term = ((ux - rx) ** 2 + (uy - ry) ** 2).sum() / (rx * rx + ry * ry).sum()
            total = term if total is None else total + term
        return total / len(tn)

    fn = kinetic_energy if objective == "energy" else enstrophy
    num = None
    den = None
    for k in tn:
        ux, uy = fields[k]
        rx, ry = ref[k]
        q = fn(ux, uy)
        qr = fn(rx, ry)
        d = (q - qr) ** 2
        num = d if num is None else num + d
        den = qr * qr if den is None else den + qr * qr
    return num / den


def bptt_select(
    case: SelectionCase,
    objective: str,
    theta_init: float = 0.02,
    *,
    iters: int = 20,
    lr: float = 0.1,
    lower: float = 1e-3,
    upper: float = 1.0,
    steps=None,
    keep_best: bool = True,
) -> dict:
    """Select θ by BPTT (Adam on ``log10 θ``) against ``objective``.

    ``keep_best=True`` reports the best training loss seen on the trajectory
    (round-1 convention, comparable across starts); ``False`` reports the last
    iterate.  Both are returned.
    """
    raw = torch.tensor(math.log10(float(theta_init)), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)

    def to_theta(v):
        return torch.clamp(10.0 ** v, lower, upper)

    hist: list[tuple[float, float]] = []
    j0 = float(objective_loss(case, objective, to_theta(raw.detach()), steps=steps).detach())
    best = (j0, float(to_theta(raw.detach())))
    for _ in range(int(iters)):
        opt.zero_grad()
        th = to_theta(raw)
        loss = objective_loss(case, objective, th, steps=steps)
        loss.backward()
        opt.step()
        with torch.no_grad():
            cur = float(to_theta(raw))
            cur_loss = float(loss)
        hist.append((cur, cur_loss))
        if cur_loss < best[0]:
            best = (cur_loss, cur)
    last = hist[-1] if hist else (float(theta_init), j0)
    chosen = best if keep_best else last
    return {
        "objective": objective,
        "closure": case.closure,
        "theta_init": float(theta_init),
        "theta": float(chosen[1]),
        "loss": float(chosen[0]),
        "loss_init": j0,
        "theta_last": float(last[0]),
        "loss_last": float(last[1]),
        "history": hist,
        "iters": int(iters),
        "lr": float(lr),
    }


def fd_gradient(case: SelectionCase, objective: str, theta: float, h: float = 1e-3,
                *, steps=None) -> float:
    """Central finite difference of the same loss (no autograd)."""
    with torch.no_grad():
        jp = float(objective_loss(case, objective, float(theta) + h, steps=steps))
        jm = float(objective_loss(case, objective, float(theta) - h, steps=steps))
    return (jp - jm) / (2.0 * h)


def autograd_gradient(case: SelectionCase, objective: str, theta: float, *,
                      steps=None) -> float:
    """Autograd (BPTT) derivative of the loss w.r.t. θ."""
    v = torch.tensor(float(theta), dtype=torch.float64, requires_grad=True)
    objective_loss(case, objective, v, steps=steps).backward()
    return float(v.grad)


def sweep_select(case: SelectionCase, objective: str, grid) -> dict:
    """Derivative-free loss on an explicit θ grid; returns grid + argmin."""
    pts = []
    with torch.no_grad():
        for th in grid:
            pts.append((float(th), float(objective_loss(case, objective, float(th)))))
    best_th, best_j = min(pts, key=lambda p: p[1])
    return {"objective": objective, "closure": case.closure,
            "grid": pts, "theta": best_th, "loss": best_j}


def golden_select(case: SelectionCase, objective: str, lo: float, hi: float,
                  iters: int = 40) -> dict:
    """Golden-section minimisation of the same loss (autograd-free control)."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c, d = b - phi * (b - a), a + phi * (b - a)
    with torch.no_grad():
        fc = float(objective_loss(case, objective, c))
        fd = float(objective_loss(case, objective, d))
        for _ in range(int(iters)):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = float(objective_loss(case, objective, c))
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = float(objective_loss(case, objective, d))
    th = 0.5 * (a + b)
    return {"objective": objective, "closure": case.closure, "theta": th,
            "loss": float(objective_loss(case, objective, th)), "bracket": (a, b)}


def spectrum_np(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Normalised radially averaged 2-D energy spectrum (numpy, no autograd)."""
    n = ux.shape[0]
    uhx = np.fft.rfft2(ux)
    uhy = np.fft.rfft2(uy)
    p = 0.5 * (np.abs(uhx) ** 2 + np.abs(uhy) ** 2)
    kx = np.fft.fftfreq(n, d=1.0 / n)
    ky = np.fft.rfftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    kr = np.sqrt(KX * KX + KY * KY).astype(int)
    nb = kr.max() + 1
    e = np.bincount(kr.ravel(), weights=p.ravel(), minlength=nb)
    e = e / max(e.sum(), 1e-300)
    return e


def holdout_metrics(case: SelectionCase, theta: float) -> dict:
    """Metrics at the held-out snapshot (no objective above ever sees it)."""
    k = int(case.holdout_step)
    with torch.no_grad():
        fields = snapshot_fields(case, float(theta), steps=(k,))
    ux, uy = fields[k]
    rx, ry = _ref_tensors(case, (k,))[k]
    num = float(((ux - rx) ** 2 + (uy - ry) ** 2).sum())
    den = float((rx * rx + ry * ry).sum())
    ke = float(kinetic_energy(ux, uy))
    ke_r = float(kinetic_energy(rx, ry))
    z = float(enstrophy(ux, uy))
    z_r = float(enstrophy(rx, ry))
    s = spectrum_np(ux.numpy(), uy.numpy())
    s_r = spectrum_np(rx.numpy(), ry.numpy())
    return {
        "step": k,
        "theta": float(theta),
        "closure": case.closure,
        "field_err": math.sqrt(num / den),
        "ke": ke,
        "ke_ref": ke_r,
        "ke_err": abs(ke - ke_r) / ke_r,
        "z": z,
        "z_ref": z_r,
        "z_err": abs(z - z_r) / z_r,
        "spec_err": float(np.linalg.norm(s - s_r) / np.linalg.norm(s_r)),
    }


def select_and_judge(
    case: SelectionCase,
    objective: str,
    *,
    theta_init: float = 0.02,
    iters: int = 20,
    lr: float = 0.1,
    grid=None,
    steps=None,
) -> dict:
    """Full BPTT → control → hold-out chain for one (closure, objective) pair."""
    bptt = bptt_select(case, objective, theta_init, iters=iters, lr=lr, steps=steps)
    ref = sweep_select(case, objective, grid) if grid is not None else None
    grad_bptt = autograd_gradient(case, objective, bptt["theta"], steps=steps)
    grad_fd = fd_gradient(case, objective, bptt["theta"], steps=steps)
    hold = holdout_metrics(case, bptt["theta"])
    return {
        "closure": case.closure,
        "objective": objective,
        "bptt": bptt,
        "sweep": ref,
        "grad": {"autograd": grad_bptt, "fd": grad_fd},
        "holdout": hold,
        "train_steps": list(case.train_steps if steps is None else steps),
    }