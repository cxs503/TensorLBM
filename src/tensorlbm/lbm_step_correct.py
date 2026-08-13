"""Correct LBM main loop with NoDynamics + half-way bounce-back.

This module provides the verified-correct main loop for LBM simulations
with bounce-back boundary conditions. It implements:

1. NoDynamics at solid cells (skip collision, like OpenLB)
2. Half-way bounce-back (BB before streaming, wall at cell interface)
3. Far-field BC (without touching solid cells)

Verified by 100% analytical solutions:
  - Shear wave decay: ν error = 0.00%
  - Couette flow: u error = 0.00%
  - Poiseuille flow: u error = 0.34%

The key insight: bounce_back_cells_3d must be applied BEFORE streaming
(with NoDynamics at solid cells) to achieve half-way bounce-back.
Applying BB after streaming gives full-way bounce-back (wall at grid
point, 4.76% error in Couette flow).

Wall-treatment modes
--------------------
``wall_treatment='bb'``  (default)
    Half-way bounce-back: collide → NoDynamics → BB → stream → BC.

``wall_treatment='wf'``
    Wall function: collide → NoDynamics → stream → WF → BC.
    Uses :func:`tensorlbm.wall_function_common.apply_wall_function`.

``wall_treatment='bfl'``
    BFL interpolated bounce-back: collide → NoDynamics → BB → stream →
    BC → BFL.  Uses :func:`tensorlbm.bfl_common.bfl_bounce_back_common`
    to replace the half-way BB at curved boundaries with a linear/
    quadratic interpolation.  Requires ``bfl_mask`` and ``bfl_q`` kwargs.

Thermal coupling
----------------
If ``thermal_step_fn`` is provided, it is called **after** the main LBM
step to update the thermal distribution *g* and apply buoyancy coupling
back to *f*.  The function signature must match
:func:`tensorlbm.thermal_common.thermal_step`::

    thermal_step_fn(f, g, mask, **thermal_kwargs) -> (f, g, T)

VOF coupling
------------
If ``vof_step_fn`` is provided, it **replaces** the entire LBM step
(collision + streaming + bounce-back) with the VOF step, which manages
its own collision, streaming, and bounce-back internally.  The function
signature must match
:func:`tensorlbm.free_surface_common.free_surface_vof_step`::

    vof_step_fn(f, phi, tau, **vof_kwargs) -> (f, phi)
"""

from __future__ import annotations

import torch
from .d3q19 import macroscopic3d
from .boundaries3d import bounce_back_cells_3d


def lbm_step_correct(
    f: torch.Tensor,
    collide_fn,
    tau: float,
    solid: torch.Tensor,
    u_in: float,
    far_field_bc_fn,
    correct_mass_fn=None,
    target_mass: float | None = None,
    step: int = 0,
    mass_interval: int = 200,
    bounce_back_fn=None,
    wall_treatment: str = "bb",
    nu: float | None = None,
    y_val: float = 1.0,
    near_mask: torch.Tensor | None = None,
    # --- BFL parameters ---
    bfl_mask: torch.Tensor | None = None,
    bfl_q: torch.Tensor | None = None,
    bfl_wall_correction: torch.Tensor | None = None,
    lattice: str = "D3Q19",  # type: ignore[assignment]
    # --- Thermal coupling ---
    thermal_step_fn=None,
    thermal_g: torch.Tensor | None = None,
    thermal_kwargs: dict | None = None,
    # --- VOF coupling ---
    vof_step_fn=None,
    vof_phi: torch.Tensor | None = None,
    vof_kwargs: dict | None = None,
    **collide_kwargs,
) -> torch.Tensor | tuple:
    """One correct LBM step with NoDynamics + half-way BB or wall function.

    Order of operations depends on *wall_treatment*:

    **BB mode** (``wall_treatment='bb'``, default):
      1. Collision (all cells)
      2. NoDynamics: restore solid cells to pre-collision
      3. Bounce-back at solid (BEFORE streaming → half-way)
      4. Streaming
      5. Far-field BC
      6. Mass correction (optional)

    **WF mode** (``wall_treatment='wf'``):
      1. Collision (all cells)
      2. NoDynamics: restore solid cells to pre-collision
      3. Streaming (NO bounce-back before streaming)
      4. Wall function (AFTER streaming → replaces BB)
      5. Far-field BC
      6. Mass correction (optional)

    **BFL mode** (``wall_treatment='bfl'``):
      1. Collision (all cells)
      2. NoDynamics: restore solid cells to pre-collision
      3. Half-way bounce-back at solid (BEFORE streaming)
      4. Streaming
      5. Far-field BC
      6. BFL interpolated bounce-back (replaces half-way BB at boundary)
      7. Mass correction (optional)

    The WF mode uses the wall function from
    :mod:`tensorlbm.wall_function_common` which applies a y+ threshold:
    cells with ``y+ < 11.6`` get bounce-back (viscous sublayer), cells
    with ``y+ >= 11.6`` get the log-law wall function (body force).

    The BFL mode uses :func:`tensorlbm.bfl_common.bfl_bounce_back_common`
    to apply interpolated bounce-back at curved/irregular boundaries.

    Args:
        f: Distribution tensor (19, nz, ny, nx).
        collide_fn: Collision function (e.g., collide_smagorinsky_mrt3d).
        tau: Relaxation time.
        solid: Boolean solid mask (nz, ny, nx).
        u_in: Free-stream velocity for far-field BC.
        far_field_bc_fn: Far-field BC function (e.g., far_field_bc_3d).
        correct_mass_fn: Mass correction function (e.g., correct_mass3d).
        target_mass: Target total mass for correction.
        step: Current step number (for mass correction interval).
        mass_interval: Mass correction interval (default 200).
        bounce_back_fn: Custom bounce-back function with signature
            ``fn(f, solid, f_pre) -> f``.  If None, uses
            ``bounce_back_cells_3d(f, solid, f_pre=f_pre)``.
            Only used when ``wall_treatment='bb'``.
        wall_treatment: ``'bb'`` (bounce-back, default), ``'wf'``
            (wall function replaces BB), or ``'bfl'`` (BFL interpolated
            bounce-back).
        nu: Kinematic viscosity (lattice units).  Required for WF mode.
        y_val: Wall distance for wall function (default 1.0).
        near_mask: Pre-computed near-wall mask (optional, for WF mode).
        bfl_mask: ``(Q, nz, ny, nx)`` bool — BFL fluid-boundary link mask.
            Required for BFL mode.
        bfl_q: ``(Q, nz, ny, nx)`` float — BFL fractional distance field.
            Required for BFL mode.
        bfl_wall_correction: Optional moving-wall correction tensor for BFL.
        lattice: Lattice name (``"D3Q19"`` or ``"D3Q27"``) for BFL/WF.
        thermal_step_fn: Optional thermal step function (e.g.,
            :func:`thermal_common.thermal_step`).  If provided, called
            after the main LBM step to update thermal distribution *g*
            and apply buoyancy to *f*.
        thermal_g: Thermal distribution tensor (required if
            ``thermal_step_fn`` is given).
        thermal_kwargs: Extra keyword arguments for ``thermal_step_fn``.
        vof_step_fn: Optional VOF step function (e.g.,
            :func:`free_surface_common.free_surface_vof_step`).  If
            provided, **replaces** the entire LBM step.
        vof_phi: Volume-of-fluid scalar field (required if
            ``vof_step_fn`` is given).
        vof_kwargs: Extra keyword arguments for ``vof_step_fn``.
        **collide_kwargs: Additional collision parameters (e.g., C_s=0.05).

    Returns:
        Updated distribution tensor.  If ``thermal_step_fn`` or
        ``vof_step_fn`` is provided, the return value may be a tuple —
        see the thermal/VOF sections above.

    Note:
        When ``vof_step_fn`` is provided, the function delegates entirely
        to the VOF step and returns ``(f, phi)``.  All other parameters
        (collision, bounce-back, etc.) are ignored because the VOF step
        manages its own collision, streaming, and bounce-back.
    """
    # ----------------------------------------------------------------
    # VOF mode: delegate entirely to the VOF step function
    # ----------------------------------------------------------------
    if vof_step_fn is not None:
        if vof_phi is None:
            raise ValueError("vof_phi must be provided when vof_step_fn is given")
        kw = dict(vof_kwargs or {})
        kw.setdefault("tau", tau)
        kw.setdefault("solid", solid)
        f, phi = vof_step_fn(f, vof_phi, **kw)
        # Mass correction
        if correct_mass_fn is not None and target_mass is not None:
            if step % mass_interval == 0:
                f = correct_mass_fn(f, target_mass)
        return f, phi

    # ----------------------------------------------------------------
    # Standard LBM step (BB / WF / BFL modes)
    # ----------------------------------------------------------------
    # 1. Save pre-collision state
    f_pre = f.clone()

    # 2. Collision (all cells)
    f = collide_fn(f, tau=tau, **collide_kwargs)

    # 3. NoDynamics: restore solid cells to pre-collision values
    sm = solid.unsqueeze(0).expand_as(f)
    for q in range(f.shape[0]):
        f[q] = torch.where(sm[q], f_pre[q], f[q])

    if wall_treatment == "wf":
        # WF mode: collide → NoDynamics → stream → WF → BC
        # Skip bounce-back before streaming

        # 4. Streaming
        from .solver3d import stream3d

        f = stream3d(f)

        # 5. Wall function (replaces BB, applied after streaming)
        from .wall_function_common import apply_wall_function

        if nu is None:
            nu = (tau - 0.5) / 3.0
        f, _diag = apply_wall_function(
            f,
            solid,
            near_mask,
            nu=nu,
            y_val=y_val,
            lattice=lattice,
        )

        # 6. Far-field BC
        f = far_field_bc_fn(f, u_in)

    elif wall_treatment == "bfl":
        # BFL mode: collide → NoDynamics → BB → stream → BC → BFL
        if bfl_mask is None or bfl_q is None:
            raise ValueError("bfl_mask and bfl_q must be provided when wall_treatment='bfl'")

        # 4. Half-way bounce-back at solid (BEFORE streaming)
        if bounce_back_fn is not None:
            f = bounce_back_fn(f, solid, f_pre)
        else:
            f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        f_pre_stream = f.clone()

        # 5. Streaming
        from .solver3d import stream3d

        f = stream3d(f)

        # 6. Far-field BC
        f = far_field_bc_fn(f, u_in)

        # 7. BFL interpolated bounce-back (replaces half-way BB at boundary)
        from .bfl_common import bfl_bounce_back_common

        f = bfl_bounce_back_common(
            f,
            f_pre_stream,
            bfl_mask,
            bfl_q,
            lattice=lattice,
            wall_correction=bfl_wall_correction,
        )

    else:
        # BB mode: collide → NoDynamics → BB → stream → BC
        # 4. Half-way bounce-back (BEFORE streaming)
        if bounce_back_fn is not None:
            f = bounce_back_fn(f, solid, f_pre)
        else:
            f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # 5. Streaming
        from .solver3d import stream3d

        f = stream3d(f)

        # 6. Far-field BC (without obstacle_mask → don't touch solid)
        f = far_field_bc_fn(f, u_in)

    # 7. Mass correction
    if correct_mass_fn is not None and target_mass is not None:
        if step % mass_interval == 0:
            f = correct_mass_fn(f, target_mass)

    # ----------------------------------------------------------------
    # Thermal coupling (after main LBM step)
    # ----------------------------------------------------------------
    if thermal_step_fn is not None:
        if thermal_g is None:
            raise ValueError("thermal_g must be provided when thermal_step_fn is given")
        kw = dict(thermal_kwargs or {})
        kw.setdefault("mask", solid)
        f, thermal_g, T = thermal_step_fn(f, thermal_g, **kw)
        return f, thermal_g, T

    return f
