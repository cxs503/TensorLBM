"""Fused-kernel-type selector for the D2Q9 BGK update (XVAL vs lbmpy).

The mature open-source implementation ``lssfau/lbmpy`` exposes a single knob,
``kernel_type``, in :func:`lbmpy.creationfunctions.create_lb_config`
(``src/lbmpy/creationfunctions.py``) that selects *the granularity at which the
streaming and the collision are fused into one kernel*:

``'default_stream_collide'``
    Pull-streaming and collision are emitted as one kernel: the pdfs are pulled
    from the neighbours, the macroscopic moments are built from the pulled
    values and the collision is applied to them in the same pass (see
    ``src/lbmpy/updatekernels.py::create_lbm_kernel``).
``'collide_only'``
    Only the collision is emitted; streaming is the caller's business.
``'stream_pull_only'``
    Only the pull streaming is emitted, no collision.

``lbmpy`` validates the value and raises ``ValueError`` for anything else.
Because the three granularities only regroup operations that are algebraically
identical (the BGK collision operator is local and the pull streaming is a
per-direction translation, so the two commute), the *numerical result* of the
three variants must agree to round-off -- the choice is a performance knob, not
a physics knob.  That equivalence is what the cross-validation in
``xval/fused_kernel_type_selector/`` measures against the OSS kernel.

This module ports the selector to :mod:`tensorlbm`, which previously only had
*fused collision* kernels (:func:`tensorlbm.solver.collide_bgk_fused`) and a
separate :func:`tensorlbm.solver.stream`; a single-pass
stream+collide kernel was missing.

Scope / limitations (mirrored in the XVAL report):

* The pull streaming here is **periodic** (``torch.roll`` on both axes), exactly
  like :func:`tensorlbm.solver.stream`.  Walls/boundaries are applied by the
  caller *after* the fused step, as in ``tensorlbm.sediment.channel``.
* Only the BGK (single-relaxation-time) collision is fused.  MRT/TRT/cumulant
  remain available through their own entry points in
  :mod:`tensorlbm.solver`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

__all__ = [
    "KERNEL_TYPES",
    "collide_stream_bgk",
    "resolve_kernel_type",
    "step",
]

#: Kernel granularities accepted by :func:`step`, mirroring lbmpy's
#: ``LBMConfig.kernel_type`` (``creationfunctions.py``).
KERNEL_TYPES: tuple[str, ...] = (
    "default_stream_collide",
    "collide_only",
    "stream_pull_only",
)


def resolve_kernel_type(kernel_type: str) -> str:
    """Validate ``kernel_type`` and return it (lbmpy-compatible rejection).

    ``lbmpy`` raises ``ValueError("Invalid value of parameter 'kernel_type'")``
    for an unknown granularity; keeping the same behaviour here means a
    configuration written for ``lbmpy`` fails loudly instead of silently
    falling back to a default kernel.
    """
    if kernel_type not in KERNEL_TYPES:
        raise ValueError(
            f"Invalid value of parameter 'kernel_type': {kernel_type!r}; "
            f"expected one of {KERNEL_TYPES}"
        )
    return kernel_type


def collide_stream_bgk(f: "torch.Tensor", tau: float,
                       moments: str = "pre_stream") -> "torch.Tensor":
    """Single-pass periodic pull-streaming **fused with** BGK collision.

    Equivalent to ``stream(collide_bgk(f, tau))`` but written as one pass over
    the nine directions: each direction is pulled from its upstream neighbour,
    and the three moments are accumulated from the pulled values in the same
    loop, so no full ``(9, ny, nx)`` intermediate streaming field is needed.

    Args:
        f: distribution of shape ``(9, ny, nx)``.
        tau: BGK relaxation time, must be finite and ``> 0.5``.
        moments: where the macroscopic moments fed to the equilibrium are
            evaluated.  ``"pre_stream"`` (default) is the textbook update
            ``stream(collide_bgk(f, tau))`` -- collision at the local cell,
            then pull.  ``"post_stream"`` builds the moments from the *pulled*
            population, i.e. ``collide_bgk(stream(f), tau)``; that is the
            convention lbmpy's ``default_stream_collide`` kernel implements
            (measured to round-off in ``xval/fused_kernel_type_selector``), so
            it is the branch to use when cross-checking against lbmpy.  The two
            orders are algebraically different (the moments of a pull-shifted
            population are not the moments of any single cell): on a random
            non-equilibrium field they differ by ~14 %, which is why the order
            has to be stated rather than assumed.

    Returns:
        ``f_new`` with ``f_new_i(x) = g_i(x) - (g_i(x) - feq_i(x)) / tau`` where
        ``g_i(x) = f_i(x - c_i)`` is the pull-streamed field and ``feq`` is the
        equilibrium of the moments of ``g``.
    """
    import torch

    from .d2q9 import _c_on, _w_for
    from .solver import _relaxation_time

    f = torch.as_tensor(f)
    if f.ndim != 3 or f.shape[0] != 9:
        raise ValueError(f"f must have shape (9, ny, nx); got {tuple(f.shape)}")
    tau = _relaxation_time(tau, where="fused_kernel_selector.collide_stream_bgk")

    device = f.device
    dtype = f.dtype
    c = _c_on(device)
    w = _w_for(dtype, device)
    C_x = c[:, 0].to(dtype)
    C_y = c[:, 1].to(dtype)

    # Moments of the *local* (pre-collision) population: the collision is a
    # pointwise operator, so it is applied at the source cell and the resulting
    # ``post`` pdfs are then pulled from the upstream neighbour -- exactly the
    # semantics of lbmpy's stream-pull accessor reading the post-collision
    # field (``updatekernels.create_lbm_kernel``).
    if moments not in ("pre_stream", "post_stream"):
        raise ValueError(f"moments must be 'pre_stream' or 'post_stream'; got {moments!r}")
    pulled = tuple(
        torch.roll(f[i], shifts=(int(C_y[i]), int(C_x[i])), dims=(0, 1)) for i in range(9)
    )
    momentum_source = f if moments == "pre_stream" else pulled

    rho = torch.zeros_like(f[0])
    jx = torch.zeros_like(f[0])
    jy = torch.zeros_like(f[0])
    for i in range(9):
        rho = rho + momentum_source[i]
        jx = jx + C_x[i] * momentum_source[i]
        jy = jy + C_y[i] * momentum_source[i]

    rho_safe = torch.clamp(rho, min=1e-12)
    ux = jx / rho_safe
    uy = jy / rho_safe
    u_sq = ux * ux + uy * uy
    inv_tau = 1.0 / tau
    coeff_pre = 1.0 - inv_tau

    source = momentum_source
    out = []
    for i in range(9):
        cu = C_x[i] * ux + C_y[i] * uy
        feq_i = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq)
        post_i = coeff_pre * source[i] + inv_tau * feq_i
        if moments == "pre_stream":
            # pull the post-collision pdfs: out_i(x) = post_i(x - c_i), the same
            # convention as tensorlbm.solver.stream (periodic wrap).
            out.append(torch.roll(post_i, shifts=(int(C_y[i]), int(C_x[i])), dims=(0, 1)))
        else:
            # lbmpy order: the pulled population is already in place, so the
            # kernel output is just the collision applied to it.
            out.append(post_i)
    return torch.stack(out, dim=0)


def step(f: "torch.Tensor", tau: float, kernel_type: str = "default_stream_collide",
         moments: str = "pre_stream") -> "torch.Tensor":
    """Advance ``f`` by one lattice step using the requested kernel granularity.

    Args:
        f: distribution of shape ``(9, ny, nx)``.
        tau: BGK relaxation time (``> 0.5``).
        kernel_type: one of :data:`KERNEL_TYPES`; validated like ``lbmpy``.

    Returns:
        The updated distribution.  ``default_stream_collide`` and
        ``collide_only`` + :func:`tensorlbm.solver.stream` are numerically
        identical; ``stream_pull_only`` returns the streamed (un-collided)
        field.
    """
    from .solver import collide_bgk, stream

    kernel_type = resolve_kernel_type(kernel_type)
    if kernel_type == "default_stream_collide":
        return collide_stream_bgk(f, tau, moments=moments)
    if kernel_type == "collide_only":
        return collide_bgk(f, tau)
    return stream(f)