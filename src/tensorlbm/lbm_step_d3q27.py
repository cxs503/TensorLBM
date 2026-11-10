"""Verified D3Q27 main loop (half-way bounce-back + far-field BC).

This closes gaps **G7/G8** of ``benchmarks/gaps.md``: before this module the
:class:`tensorlbm.general_sim.GeneralSimEngine` enumerated ``LatticeModel.D3Q27``
but every stage of the pipeline (equilibrium init, streaming, bounce-back,
far-field BC, mass correction) hard-coded the D3Q19 stencil.

The step below is the exact D3Q27 counterpart of
:func:`tensorlbm.lbm_step_correct.lbm_step_correct` (BB mode), reusing the
library's D3Q27 primitives:

  1. Collision (``collide_bgk27`` / ``collide_mrt27``)
  2. NoDynamics (solid cells keep the pre-collision population) — OpenLB
     convention, required for half-way bounce-back
  3. Half-way bounce-back (swap to the opposite direction, BEFORE streaming)
  4. Streaming (``d3q27.stream27_roll`` — periodic wrap on all axes)
  5. Far-field BC (free-stream equilibrium on inlet + selected lateral faces,
     zero-gradient outlet); faces not listed are left to the periodic wrap
  6. Optional global mass correction (``d3q27.correct_mass27``)

Verified properties (see ``tests/test_general_sim_d3q27_activation.py``):
uniform-equilibrium fixed point, mass conservation, degenerate agreement with
the D3Q19 route on a D3Q19-supported population, and the D3Q27 isotropy of the
equilibrium (3rd-moment residual < 1e-14).
"""

from __future__ import annotations

import torch

from .d3q27 import OPPOSITE as OPPOSITE27
from .d3q27 import equilibrium27, stream27_roll

__all__ = ["far_field_bc_27_faces", "lbm_step_correct_27", "check_d3q27_state"]

Q27 = 27


def check_d3q27_state(f: torch.Tensor, solid: torch.Tensor | None = None) -> None:
    """Validate a D3Q27 population tensor (negative-control guard)."""
    if f.dim() != 4 or f.shape[0] != Q27:
        raise ValueError(
            f"D3Q27 step expects f with shape (27, nz, ny, nx); got {tuple(f.shape)}"
        )
    if solid is not None:
        if solid.shape != f.shape[1:]:
            raise ValueError(
                f"solid mask shape {tuple(solid.shape)} != f spatial shape {tuple(f.shape[1:])}"
            )
        if solid.dtype != torch.bool:
            raise ValueError(f"solid mask must be bool, got {solid.dtype}")


def far_field_bc_27_faces(
    f: torch.Tensor,
    u_in: float,
    bc_config: dict | None = None,
    uy: float = 0.0,
    uz: float = 0.0,
    obstacle_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Face-selective far-field BC for D3Q27.

    Exact D3Q27 counterpart of
    :func:`tensorlbm.boundary.boundaries3d.far_field_bc_3d`: same face labels,
    same inlet/outlet semantics, same ``bc_config`` contract —

    * inlet ``x-``  → free-stream equilibrium (unless periodic),
    * outlet ``x+`` → zero gradient (copy the neighbour plane, unless periodic),
    * lateral ``y±``/``z±`` → free-stream equilibrium iff listed in
      ``bc_config['far_field_faces']`` and not periodic; otherwise the periodic
      wrap performed by ``stream27_roll`` is preserved.

    ``bc_config=None`` reproduces the legacy behaviour: all four lateral faces
    receive the free-stream equilibrium.
    """
    check_d3q27_state(f)
    ones = torch.ones((1, 1, 1), dtype=f.dtype, device=f.device)
    feq = equilibrium27(
        ones,
        torch.full_like(ones, u_in),
        torch.full_like(ones, uy),
        torch.full_like(ones, uz),
        device=f.device,
    )  # (27, 1, 1, 1)

    if bc_config is not None:
        ff = set(bc_config.get("far_field_faces", []))
        periodic = set(bc_config.get("periodic_faces", []))
    else:
        ff = {"y-", "y+", "z-", "z+"}
        periodic = set()

    f = f.clone()
    if "x-" not in periodic:
        f[:, :, :, 0] = feq[:, :, :, 0]
    if "x+" not in periodic:
        f[:, :, :, -1] = f[:, :, :, -2]

    _, nz, ny, nx = f.shape
    if "y-" in ff and "y-" not in periodic:
        f[:, :, 0, :] = feq[:, :, 0, :].expand(Q27, nz, nx).contiguous()
    if "y+" in ff and "y+" not in periodic:
        f[:, :, -1, :] = feq[:, :, -1, :].expand(Q27, nz, nx).contiguous()
    if "z-" in ff and "z-" not in periodic:
        f[:, 0, :, :] = feq[:, 0, :, :]
    if "z+" in ff and "z+" not in periodic:
        f[:, -1, :, :] = feq[:, -1, :, :]
    return f


def lbm_step_correct_27(
    f: torch.Tensor,
    collide_fn,
    tau: float,
    solid: torch.Tensor,
    u_in: float,
    far_field_bc_fn=None,
    correct_mass_fn=None,
    target_mass: float | None = None,
    step: int = 0,
    mass_interval: int = 200,
    wall_treatment: str = "bb",
    **collide_kwargs,
) -> torch.Tensor:
    """One D3Q27 LBM step: collide → NoDynamics → BB → stream → BC → mass.

    Parameters mirror :func:`tensorlbm.lbm_step_correct.lbm_step_correct`
    (BB mode only).  ``wall_treatment`` other than ``"bb"`` raises
    ``NotImplementedError`` — wall functions and BFL interpolation are not yet
    available on D3Q27 in this engine, and silently falling back to D3Q19
    kernels would be a correctness bug.
    """
    if wall_treatment != "bb":
        raise NotImplementedError(
            f"lbm_step_correct_27 supports wall_treatment='bb' only; got {wall_treatment!r}"
        )
    check_d3q27_state(f, solid)

    # 1+2+3: fused NoDynamics + half-way bounce-back (bit-exact form of the
    # two-op sequence: solid cells end up holding f_pre[OPPOSITE[q]]).
    f_pre_opp = f[OPPOSITE27.to(f.device)]
    f = collide_fn(f, tau=tau, **collide_kwargs)
    f = torch.where(solid.unsqueeze(0), f_pre_opp, f)

    # 4: streaming (periodic wrap on every axis)
    f = stream27_roll(f)

    # 5: far-field / periodic faces
    if far_field_bc_fn is not None:
        f = far_field_bc_fn(f, u_in)

    # 6: global mass correction
    if correct_mass_fn is not None and target_mass is not None:
        if step % mass_interval == 0:
            f = correct_mass_fn(f, target_mass)
    return f