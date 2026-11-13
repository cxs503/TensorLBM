"""Differentiable D2Q9 BGK lattice-Boltzmann kernel with an in-graph LES closure.

This is the *graph-unroll* building block: every operator below takes and
returns plain tensors, nothing is written back into an in-place buffer, so a
Python ``for`` loop over time steps stays inside the autograd graph and one
``loss.backward()`` yields the derivative of a terminal (or multi-snapshot)
field mismatch with respect to the sub-grid model constant.

Cross-validation target (OSS, zero transcription — see
``artifacts/xval_graph_unroll_bptt_les/oss_source_notes.json``): the
TensorFlow-1.x implementation ``loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow``
provides a ``graph_unroll`` branch in which ``collide``/``stream`` return
in-graph tensors instead of assigning to a ``Variable``, an ``Unroll()``
helper chains ``N`` steps on the Python side and returns the state chain, and
``tau`` is recomputed *inside* the graph from the local field and the scalar
``Sc`` so gradients reach the model constant from the field ``N`` steps
later.

Two closures are available:

``"smagorinsky"``
    The literature-consistent closure used for the calibration under test.
    With :math:`\\Pi^{neq}_{ab}=\\sum_i (f_i-f_i^{eq})c_{ia}c_{ib}`,
    :math:`Q=\\sqrt{\\Pi^{neq}_{ab}\\Pi^{neq}_{ab}}` and
    :math:`|S|=\\sqrt{2S_{ab}S_{ab}}` one gets
    :math:`|S| = 3\\sqrt{2}\\,Q/(2\\rho\\tau)` from
    :math:`\\Pi^{neq}_{ab} = -(2/3)\\tau\\rho S_{ab}`, so that
    :math:`\\nu_t=C_s^2|S|` and :math:`\\tau=\\tfrac12+3(\\nu_0+\\nu_t)`
    (lattice units, :math:`\\Delta=1`) give the closed form

    .. math:: \\tau = \\tfrac12\\left(\\tau_0 +
        \\sqrt{\\tau_0^2 + 18\\sqrt2\\,C_s^2 Q/\\rho}\\right)

    The prefactor is not taken on faith: :func:`closure_eddy_viscosity` is
    checked against the analytic viscous decay of a single shear mode in
    ``tests/test_les_graph_unroll_bptt.py`` (<15% on the decay exponent).

``"oss_proxy"``
    The upstream scalar proxy, reproduced for the difference table only:
    :math:`Q=\\sqrt{2\\sum_i (f_i-f_i^{eq})^2 \\mathrm{EEk}_i}` with
    :math:`\\mathrm{EEk}_i=\\sum_{n,m}|c_{in}c_{im}|`, then
    :math:`\\tau=\\tfrac12(\\tau_0+\\sqrt{\\tau_0^2+6 Q S_c/\\rho})`.
    As the upstream inventory notes, this is *not* equivalent to the
    literature form (:math:`18\\sqrt2 C_s^2 Q/\\rho`): the effective constant
    is rescaled, which is why the two closures cannot share one ``Sc`` value.

Conventions: ``f`` has shape ``(9, nx, ny)`` with ``nx`` the first spatial
axis; lattice spacing and time step are 1; the domain is periodic (streaming
is a set of ``torch.roll`` calls, differentiable by construction).
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "E",
    "W",
    "OPPOSITE",
    "CLOSURES",
    "equilibrium",
    "macroscopic",
    "neq_stress",
    "tau_from_cs",
    "collide_stream",
    "unroll",
    "shear_wave_ic",
]

_DTYPE = torch.float64

#: D2Q9 lattice velocities, ordering ``(0, E, N, W, S, NE, NW, SW, SE)``.
E = torch.tensor(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=_DTYPE,
)
W = torch.tensor(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=_DTYPE
)
OPPOSITE = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.int64)

CLOSURES = ("smagorinsky", "oss_proxy")

# |c_n c_m| summed over the two lattice components for *each* direction:
# used only by the ``oss_proxy`` closure (upstream EEk table).
_EEK = torch.tensor(
    [sum(abs(int(a) * int(b)) for a in (cx, cy) for b in (cx, cy)) for cx, cy in E.tolist()],
    dtype=_DTYPE,
)


def _ex() -> torch.Tensor:
    return E[:, 0].view(9, 1, 1)


def _ey() -> torch.Tensor:
    return E[:, 1].view(9, 1, 1)


def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Second-order D2Q9 equilibrium ``feq`` with shape ``(9, nx, ny)``."""
    ex, ey = _ex(), _ey()
    w = W.view(9, 1, 1)
    eu = ex * ux + ey * uy
    usq = ux * ux + uy * uy
    return w * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq)


def macroscopic(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Density and velocity ``(rho, ux, uy)`` from ``f``."""
    rho = f.sum(dim=0)
    ux = (f * _ex()).sum(dim=0) / rho
    uy = (f * _ey()).sum(dim=0) / rho
    return rho, ux, uy


def neq_stress(f: torch.Tensor, feq: torch.Tensor) -> torch.Tensor:
    """Non-equilibrium momentum-flux tensor ``Pi_ab`` -> ``(3, nx, ny)`` (xx, xy, yy)."""
    fneq = f - feq
    ex, ey = _ex(), _ey()
    pxx = (fneq * ex * ex).sum(dim=0)
    pxy = (fneq * ex * ey).sum(dim=0)
    pyy = (fneq * ey * ey).sum(dim=0)
    return torch.stack([pxx, pxy, pyy])


def _q_literature(pi: torch.Tensor) -> torch.Tensor:
    """``Q = sqrt(Pi_ab Pi_ab)``."""
    return torch.sqrt(pi[0] * pi[0] + 2.0 * pi[1] * pi[1] + pi[2] * pi[2] + 1e-300)


def _q_oss(f: torch.Tensor, feq: torch.Tensor) -> torch.Tensor:
    """Upstream scalar proxy ``sqrt(2 sum_i (f_i-f_i^eq)^2 EEk_i)``."""
    fneq = f - feq
    return torch.sqrt(2.0 * (fneq * fneq * _EEK.view(9, 1, 1)).sum(dim=0) + 1e-300)


def tau_from_cs(
    f: torch.Tensor,
    feq: torch.Tensor,
    rho: torch.Tensor,
    tau0,
    cs,
    closure: str = "smagorinsky",
):
    """Total relaxation time from the local field and the model constant.

    ``cs`` and ``tau0`` may be Python floats or graph-connected 0-dim/1-dim
    tensors broadcastable to the field shape; the returned ``tau`` is a field
    and is differentiable w.r.t. both.
    """
    if closure not in CLOSURES:
        raise ValueError(f"closure must be one of {CLOSURES}, got {closure!r}")
    if closure == "smagorinsky":
        q = _q_literature(neq_stress(f, feq))
        c = 18.0 * math.sqrt(2.0)
    else:
        q = _q_oss(f, feq)
        c = 6.0
    root = torch.sqrt(tau0 * tau0 + c * cs * cs * q / rho)
    return 0.5 * (tau0 + root)


def collide_stream(
    f: torch.Tensor,
    tau0=0.6,
    cs=0.0,
    closure: str = "smagorinsky",
    *,
    return_tau: bool = False,
):
    """One differentiable collide -> stream step on the periodic torus.

    ``cs=0`` (for either closure) reduces to plain BGK with ``tau0``.
    """
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    tau = tau_from_cs(f, feq, rho, tau0, cs, closure)
    f_post = f - (f - feq) / tau
    out = torch.empty_like(f_post)
    for i in range(9):
        out[i] = torch.roll(f_post[i], shifts=(int(E[i, 0]), int(E[i, 1])), dims=(0, 1))
    if return_tau:
        return out, tau
    return out


def unroll(
    f0: torch.Tensor,
    n_steps: int,
    tau0=0.6,
    cs=0.0,
    closure: str = "smagorinsky",
    *,
    snapshots: tuple[int, ...] = (),
):
    """Chain ``n_steps`` graph steps; return ``(f_final, {step: f})``.

    This is the Python-side ``Unroll()`` loop of the upstream implementation:
    the state chain stays in the autograd graph, so a loss on ``f_final`` (or
    on any snapshot) differentiates back to ``cs``.
    """
    wanted = set(int(s) for s in snapshots)
    snaps: dict[int, torch.Tensor] = {}
    f = f0
    for k in range(1, int(n_steps) + 1):
        f = collide_stream(f, tau0, cs, closure)
        if k in wanted:
            snaps[k] = f
    return f, snaps


def shear_wave_ic(nx: int, ny: int, *, amp: float = 0.01, kx: int = 1, ky: int = 1):
    """Single-mode initial condition ``u = amp (sin(k.x), -sin(k.y))``."""
    x = torch.arange(nx, dtype=_DTYPE).view(nx, 1) * (2.0 * math.pi / nx)
    y = torch.arange(ny, dtype=_DTYPE).view(1, ny) * (2.0 * math.pi / ny)
    ux = amp * torch.sin(kx * x) * torch.cos(ky * y)
    uy = -amp * torch.sin(ky * y) * torch.cos(kx * x)
    rho = torch.ones((nx, ny), dtype=_DTYPE)
    return equilibrium(rho, ux, uy), ux, uy