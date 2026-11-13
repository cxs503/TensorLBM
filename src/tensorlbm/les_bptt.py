"""Differentiable D2Q9 lattice Boltzmann with graph unrolling and BPTT
calibration of the LES (Smagorinsky-type) closure parameter.

Why this module exists
----------------------
The platform's D2Q9 kernels (:mod:`tensorlbm.solver`) are written for
*forward* simulation: collision/streaming overwrite fields in place, so no
gradient can flow from a late-time field back to a model constant.  A growing
line of work (data-driven LES closures) needs the opposite: keep the whole
time-integration inside an autograd graph and back-propagate through it
("back-propagation through time", BPTT) to identify a sub-grid model constant
from reference data.

This module provides that missing capability in a dependency-light way
(PyTorch only, float32/float64, CPU):

* :func:`collide_les` - BGK collision with the LES relaxation time
  ``tau = 1/2 (tau0 + sqrt(tau0^2 + 6 Q Sc / rho))``, ``Q = sqrt(2 sum_i
  (f_i - f_i^eq)^2 E_i)``, ``E_i = (|c_ix| + |c_iy|)^2`` - i.e. the same
  non-equilibrium stress proxy used by the upstream reference implementation
  (see ``OSS_DIFFERENCES`` in the module docs and
  ``xval/XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER/``).
* :func:`stream` - streaming as a single fixed 3x3 convolution with periodic
  (torus) padding, the "conv-as-streaming" formulation; mathematically
  identical to :func:`tensorlbm.solver.stream` but fully differentiable and
  vectorised over channels.
* :func:`unroll` - chain ``n`` collide+stream steps inside the graph and
  return the state history (the ``Unroll``/``CollideSC(graph_unroll=True)``
  pattern of the reference implementation).
* :func:`calibrate_les_parameter` - BPTT identification of ``Sc`` from a
  reference velocity history (adjoint-style objective, Adam in log-space).

The operator is *exactly* the D2Q9 BGK operator with a Smagorinsky-type
effective relaxation time, so the calibrated ``Sc`` is transferable to the
forward kernels: :func:`tensorlbm.solver.collide_bgk_les` delegates here.

Deviations from the reference implementation (documented, not silent)
--------------------------------------------------------------------
``OSS_DIFFERENCES`` (also written out in the xval report):

1. TensorFlow graph-mode ``tf.Variable`` + ``assign`` is replaced by plain
   tensors carried through Python; state chaining is by object identity, which
   is the same graph-unroll contract (no in-place writes on graph tensors).
2. ``Sc`` is a ``torch`` scalar with autograd instead of a TF variable fed to
   an optimizer; the gradient is identical (chain rule through ``sqrt``).
3. Column-major/row-major bookkeeping of the stream table is replaced by an
   explicit velocity-indexed kernel, verified against the reference table
   (``tests/test_les_bptt.py``) and against a roll-based gather.
4. float64 is the default here (float32 selectable) because parameter
   identification is noise-limited; the reference uses float32 throughout.
5. No ``Zou-He`` in/outflow callback is ported: this module is periodic only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "D2Q9_C",
    "D2Q9_W",
    "D2Q9_OPP",
    "D2Q9_EEK",
    "density_velocity",
    "equilibrium",
    "les_relaxation_time",
    "collide_les",
    "stream",
    "unroll",
    "shear_wave_state",
    "analytic_shear_decay",
    "calibrate_les_parameter",
    "velocity_mse_loss",
]

# --------------------------------------------------------------------------
# D2Q9 tables.  Ordering is the one shared with the rest of TensorLBM
# (turbulence_closure.lattice.D2Q9) and with the upstream reference tables:
#   0:(0,0) 1:(1,0) 2:(0,1) 3:(-1,0) 4:(0,-1) 5:(1,1) 6:(-1,1) 7:(-1,-1) 8:(1,-1)
# --------------------------------------------------------------------------
_C_LIST = (
    (0, 0),
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
    (1, -1),
)
_W_LIST = (
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
)

D2Q9_C = torch.tensor(_C_LIST, dtype=torch.float64)
D2Q9_W = torch.tensor(_W_LIST, dtype=torch.float64)
D2Q9_OPP = torch.tensor(
    [
        next(i for i, c in enumerate(_C_LIST) if c == (-cx, -cy))
        for cx, cy in _C_LIST
    ],
    dtype=torch.long,
)
# E_i = sum_{n,m in {x,y,z}} |c_n c_m| evaluated per direction, exactly as the
# reference implementation accumulates it (the z component is zero in 2-D, so
# E_i = (|c_ix| + |c_iy|)^2 for the D2Q9 set: 0,1,1,1,1,4,4,4,4).
D2Q9_EEK = torch.tensor(
    [
        sum(
            abs(c[n]) * abs(c[m])
            for n in range(3)
            for m in range(3)
            for c in ((cx, cy, 0),)
        )
        for cx, cy in _C_LIST
    ],
    dtype=torch.float64,
)

_TABLE_CACHE: dict[tuple[str, str], tuple[torch.Tensor, ...]] = {}


def _tables(dtype: torch.dtype, device: torch.device):
    """Lattice tables cast once per (dtype, device) pair."""
    key = (str(dtype), str(device))
    cached = _TABLE_CACHE.get(key)
    if cached is None:
        c = D2Q9_C.to(dtype=dtype, device=device)
        w = D2Q9_W.to(dtype=dtype, device=device)
        opp = D2Q9_OPP.to(device=device)
        eek = D2Q9_EEK.to(dtype=dtype, device=device)
        # Streaming kernel: out_q(x) = f_q(x - c_q).  With a 1-cell circular
        # pad and VALID cross-correlation, kernel[q, q, 1 - c_y, 1 - c_x] = 1.
        kern = torch.zeros(9, 9, 3, 3, dtype=dtype, device=device)
        for q, (cx, cy) in enumerate(_C_LIST):
            kern[q, q, 1 - cy, 1 - cx] = 1.0
        _TABLE_CACHE[key] = (c, w, opp, eek, kern)
    return _TABLE_CACHE[key]


# --------------------------------------------------------------------------
# Primitive operators
# --------------------------------------------------------------------------


def density_velocity(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Moments of a ``[B, 9, H, W]`` distribution (``Cs = 1``)."""
    c, _, _, _, _ = _tables(f.dtype, f.device)
    rho = f.sum(dim=1, keepdim=True)
    vel = (f.unsqueeze(2) * c.view(1, 9, 2, 1, 1)).sum(dim=1)
    return rho, vel / rho


def equilibrium(rho: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
    """Second-order Maxwell-Boltzmann equilibrium, ``[B, 9, H, W]``."""
    c, w, _, _, _ = _tables(rho.dtype, rho.device)
    cu = (vel.unsqueeze(1) * c.view(1, 9, 2, 1, 1)).sum(dim=2)  # [B,9,H,W]
    usq = (vel * vel).sum(dim=1, keepdim=True)
    return w.view(1, 9, 1, 1) * rho * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq
    )


def les_relaxation_time(
    tau0: float | torch.Tensor,
    sc: torch.Tensor | float,
    rho: torch.Tensor,
    f: torch.Tensor,
    feq: torch.Tensor,
    les: bool = True,
) -> torch.Tensor | float:
    """Smagorinsky-type effective relaxation time.

    ``tau = 1/2 (tau0 + sqrt(tau0^2 + 6 Q Sc / rho))`` with
    ``Q = sqrt(2 sum_i (f_i - f_i^eq)^2 E_i)``.  This is the closure used by
    the upstream reference implementation; ``Sc`` is the lumped parameter to be
    identified (``Sc -> 0`` recovers plain BGK).
    """
    if not les:
        return tau0
    _, _, _, eek, _ = _tables(f.dtype, f.device)
    noneq = f - feq
    q = (noneq * noneq * eek.view(1, 9, 1, 1)).sum(dim=1, keepdim=True)
    q = torch.sqrt(2.0 * q.clamp_min(0.0))
    tau0_t = torch.as_tensor(tau0, dtype=f.dtype, device=f.device)
    return 0.5 * (tau0_t + torch.sqrt(tau0_t * tau0_t + 6.0 * q * sc / rho))


def collide_les(
    f: torch.Tensor,
    tau0: float = 0.56,
    sc: torch.Tensor | float = 0.17,
    boundary: torch.Tensor | None = None,
    les: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BGK collision + half-way bounce-back + LES closure, one time step.

    Parameters
    ----------
    f : ``[B, 9, H, W]`` distribution (valid ``f.sum`` gives ``rho``).
    tau0 : bare (molecular) relaxation time, ``tau0 = 3 nu0 + 1/2``.
    sc : LES parameter (scalar tensor if gradients are wanted).
    boundary : ``[B, 1, H, W]`` mask, ``1`` = solid (bounce-back cell).

    Returns ``(f_post_collision, rho, vel)``.
    """
    c, _, opp, _, _ = _tables(f.dtype, f.device)
    del c
    rho = f.sum(dim=1, keepdim=True)
    vel = (f.unsqueeze(2) * _tables(f.dtype, f.device)[0].view(1, 9, 2, 1, 1)).sum(
        dim=1
    )
    vel = vel / rho
    feq = equilibrium(rho, vel)
    tau = les_relaxation_time(tau0, sc, rho, f, feq, les=les)
    out = f - (f - feq) / tau
    if boundary is not None:
        # f_boundary = conv(F_old * boundary, BOUNCE)  ->  out_q = f_old_{opp(q)}
        f_b = (f * boundary)[:, opp]
        out = out * (1.0 - boundary) + f_b
    return out, rho, vel


def stream(f: torch.Tensor) -> torch.Tensor:
    """Periodic streaming as one 3x3 depth-wise-ish convolution.

    Equivalent to ``torch.roll(f, shifts=(c_y, c_x))`` per direction; the
    equivalence is asserted in ``tests/test_les_bptt.py``.
    """
    _, _, _, _, kern = _tables(f.dtype, f.device)
    fp = torch.cat((f[..., -1:, :], f, f[..., :1, :]), dim=2)
    fp = torch.cat((fp[..., -1:], fp, fp[..., :1]), dim=3)
    return F.conv2d(fp, kern)


# --------------------------------------------------------------------------
# Graph unrolling / BPTT
# --------------------------------------------------------------------------


@dataclass
class UnrollState:
    """One recorded step of an unrolled rollout."""

    step: int
    f: torch.Tensor
    rho: torch.Tensor
    vel: torch.Tensor


def unroll(
    f0: torch.Tensor,
    n_steps: int,
    *,
    tau0: float = 0.56,
    sc: torch.Tensor | float = 0.17,
    boundary: torch.Tensor | None = None,
    les: bool = True,
) -> list[UnrollState]:
    """Run ``n_steps`` of ``collide -> stream`` keeping the autograd graph.

    Returns the state *after* each stream step (the reference implementation's
    ``Unroll`` returns the same chain).  ``sc`` may be a tensor with
    ``requires_grad=True``; gradients then flow back through ``n_steps``.
    """
    states: list[UnrollState] = []
    f = f0
    rho = vel = None
    for step in range(n_steps):
        f, rho, vel = collide_les(f, tau0=tau0, sc=sc, boundary=boundary, les=les)
        f = stream(f)
        rho = f.sum(dim=1, keepdim=True)
        vel = (f.unsqueeze(2) * _tables(f.dtype, f.device)[0].view(1, 9, 2, 1, 1)).sum(
            dim=1
        ) / rho
        states.append(UnrollState(step=step, f=f, rho=rho, vel=vel))
    return states


def velocity_mse_loss(
    states: Sequence[UnrollState],
    target_vel: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Weighted mean-square velocity mismatch over an unrolled window."""
    if len(states) != len(target_vel):
        raise ValueError("states and target_vel must have the same length")
    if weights is None:
        weights = [1.0] * len(states)
    total = None
    for st, tgt, w in zip(states, target_vel, weights):
        term = w * ((st.vel - tgt) ** 2).mean()
        total = term if total is None else total + term
    assert total is not None
    return total / float(sum(weights))


def calibrate_les_parameter(
    target_vel: Sequence[torch.Tensor],
    f0: torch.Tensor,
    *,
    tau0: float = 0.56,
    sc_init: float = 0.05,
    boundary: torch.Tensor | None = None,
    les: bool = True,
    iters: int = 120,
    lr: float = 0.05,
    threshold: float = 1e-14,
    method: str = "adam",
) -> dict:
    """Identify ``Sc`` by BPTT from a reference velocity history.

    The unrolled window is ``len(target_vel)`` steps long; the optimiser works
    in ``log Sc`` so that positivity is structural.  Returns a history dict
    (``sc_history``, ``loss_history``, ``grad_history``, ``sc``, ``loss``).
    """
    n_steps = len(target_vel)
    if method == "gradient_root":
        return _calibrate_by_gradient_root(
            target_vel, f0, tau0=tau0, sc_init=sc_init, boundary=boundary,
            les=les, maxiter=iters,
        )
    if method != "adam":
        raise ValueError(f"unknown method {method!r}")
    log_sc = torch.tensor(math.log(sc_init), dtype=f0.dtype, requires_grad=True)
    opt = torch.optim.Adam([log_sc], lr=lr)
    hist = {"sc_history": [], "loss_history": [], "grad_history": []}
    # cosine decay of the step size for a stable terminal contraction
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1))
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        sc = torch.exp(log_sc)
        states = unroll(f0, n_steps, tau0=tau0, sc=sc, boundary=boundary, les=les)
        loss = velocity_mse_loss(states, target_vel)
        loss.backward()
        hist["sc_history"].append(float(torch.exp(log_sc).detach()))
        hist["loss_history"].append(float(loss.detach()))
        hist["grad_history"].append(float(log_sc.grad.detach()) if log_sc.grad is not None else float("nan"))
        opt.step()
        sched.step()
        if float(loss.detach()) < threshold:
            break
    sc = float(torch.exp(log_sc).detach())
    hist["sc"] = sc
    hist["loss"] = hist["loss_history"][-1]
    hist["iterations"] = len(hist["loss_history"])
    return hist


# --------------------------------------------------------------------------
# Analytic shear-wave anchor (validation of the forward operator itself)
# --------------------------------------------------------------------------


def shear_wave_state(
    n: int = 48,
    u0: float = 0.05,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local-equilibrium initial condition with ``u_x = u0 sin(2 pi y / n)``.

    Returns ``(f0, boundary)`` with a *fluid* mask (no solids), i.e. a
    periodic shear layer: the standard linear-response probe for an effective
    viscosity, and the cheapest flow in which a sub-grid model is identifiable
    (the LES term raises the decay rate of the fundamental mode).
    """
    j = torch.arange(n, dtype=dtype, device=device)
    ux = u0 * torch.sin(2.0 * math.pi * j / n).view(1, 1, n, 1).expand(1, 1, n, n)
    uy = torch.zeros_like(ux)
    vel = torch.cat((ux, uy), dim=1)
    rho = torch.ones(1, 1, n, n, dtype=dtype, device=device)
    f0 = equilibrium(rho, vel)
    boundary = torch.zeros(1, 1, n, n, dtype=dtype, device=device)
    return f0, boundary


def analytic_shear_decay(u0: float, nu: float, k: float, t: float) -> float:
    """Linear-regime decay of a shear wave: ``u0 exp(-nu k^2 t)``."""
    return u0 * math.exp(-nu * k * k * t)


def fundamental_amplitude(f: torch.Tensor) -> torch.Tensor:
    """Amplitude of the ``sin(2 pi y / n)`` mode of ``u_x`` (per batch)."""
    _, vel = density_velocity(f)
    n = f.shape[2]
    j = torch.arange(n, dtype=f.dtype, device=f.device)
    basis = torch.sin(2.0 * math.pi * j / n)
    prof = vel[:, 0].mean(dim=2)  # [B, n] (uniform in x)
    return 2.0 * (prof * basis.view(1, n)).mean(dim=1)


def profile_amplitude(f: torch.Tensor) -> float:
    """Scalar convenience wrapper (batch of one)."""
    return float(fundamental_amplitude(f)[0])

# --------------------------------------------------------------------------
# Taylor-Green vortex anchor (analytic linear-regime decay)
# --------------------------------------------------------------------------


def taylor_green_state(
    n: int = 48,
    u0: float = 0.1,
    k: int = 2,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local-equilibrium 2-D Taylor-Green vortex.

    ``u_x = u0 sin(kx) cos(ky)``, ``u_y = -u0 cos(kx) sin(ky)`` with
    ``k = 2 pi k_mode / n``.  In the linear (small ``u0``) regime the velocity
    amplitude decays as ``exp(-2 nu k^2 t)``, which gives an analytic reference
    for the *forward* operator and a flow with non-zero strain everywhere -
    the minimal setting in which a sub-grid parameter is identifiable.
    """
    x = torch.arange(n, dtype=dtype, device=device)
    xx = x.view(1, 1, n, 1).expand(1, 1, n, n)
    yy = x.view(1, 1, 1, n).expand(1, 1, n, n)
    kk = 2.0 * math.pi * k / n
    ux = u0 * torch.sin(kk * xx) * torch.cos(kk * yy)
    uy = -u0 * torch.cos(kk * xx) * torch.sin(kk * yy)
    vel = torch.cat((ux, uy), dim=1)
    rho = torch.ones(1, 1, n, n, dtype=dtype, device=device)
    return equilibrium(rho, vel), torch.zeros(1, 1, n, n, dtype=dtype, device=device)


def analytic_taylor_green_decay(u0: float, nu: float, k_phys: float, t: float) -> float:
    """``u0 exp(-2 nu k^2 t)`` (linear regime, 2-D Taylor-Green)."""
    return u0 * math.exp(-2.0 * nu * k_phys * k_phys * t)


def tg_amplitude(f: torch.Tensor, k: int = 2) -> torch.Tensor:
    """Amplitude of the ``k``-th Taylor-Green mode of ``u_x``.

    ``u_x = A sin(kx) cos(ky)`` projects as ``4 <u_x sin(kx) cos(ky)> = A``
    (the mean of ``sin^2 cos^2`` over one period is 1/4), so this returns the
    amplitude in velocity units and can be compared directly with
    :func:`analytic_taylor_green_decay`.
    """
    _, vel = density_velocity(f)
    n = f.shape[2]
    m = 2.0 * math.pi * k / n
    x = torch.arange(n, dtype=f.dtype, device=f.device)
    sx = torch.sin(m * x).view(1, 1, n, 1)
    cy = torch.cos(m * x).view(1, 1, 1, n)
    return 4.0 * (vel[:, 0:1] * sx * cy).mean(dim=(1, 2, 3))


def _calibrate_by_gradient_root(
    target_vel: Sequence[torch.Tensor],
    f0: torch.Tensor,
    *,
    tau0: float,
    sc_init: float,
    boundary: torch.Tensor | None,
    les: bool,
    maxiter: int = 20,
    lo: float = 1e-3,
    hi: float = 5.0,
) -> dict:
    """Identify ``Sc`` by a bracketed root solve on the BPTT gradient.

    The parameter is one-dimensional, so the stationary point of the unrolled
    loss is found with Brent's method applied to ``dL/dSc`` obtained from the
    retained graph.  Superlinear (each evaluation costs one forward + one
    backward sweep), and it terminates on an explicit stationarity residual
    instead of a step-count budget.  Falls back to a bisection refinement if the
    bracket does not change sign.
    """
    from scipy.optimize import brentq

    hist = {"sc_history": [], "loss_history": [], "grad_history": []}

    def dldsc(val: float) -> float:
        sc = torch.tensor(val, dtype=f0.dtype, requires_grad=True)
        states = unroll(
            f0, len(target_vel), tau0=tau0, sc=sc, boundary=boundary, les=les
        )
        loss = velocity_mse_loss(states, target_vel)
        loss.backward()
        grad = float(sc.grad)
        hist["sc_history"].append(float(val))
        hist["loss_history"].append(float(loss.detach()))
        hist["grad_history"].append(grad)
        return grad

    try:
        sc_root = float(brentq(dldsc, lo, hi, xtol=1e-6, rtol=1e-8,
                               maxiter=maxiter))
    except ValueError:  # no sign change: keep the best sampled point
        grid = [lo, sc_init, 0.05, 0.1, 0.2, 0.35, 0.6, hi]
        sc_root = min(grid, key=lambda v: abs(dldsc(v)))
    hist["sc"] = sc_root
    hist["loss"] = float(
        velocity_mse_loss(
            unroll(f0, len(target_vel), tau0=tau0, sc=sc_root,
                   boundary=boundary, les=les),
            target_vel,
        )
    )
    hist["iterations"] = len(hist["loss_history"])
    hist["method"] = "gradient_root"
    return hist
