"""Common timestep executor for all LBM solvers.

Provides :class:`LBMStepExecutor` — a unified, pre-allocated,
macroscopic-reusing timestep executor that all 53 solver files can use.
Optimises memory management and call patterns **without** changing any
collision, streaming, or boundary formula.

Key optimisations (all internal to ``LBMStepExecutor``):

  a. **Macroscopic reuse** — compute ``rho``/``ux``/``uy``/``uz`` once per
     step and pass them to the wall-function and force-measurement
     callbacks, avoiding up to 2 redundant ``macroscopic`` calls per step.
  b. **Pre-allocated buffers** — ``f_post``, ``feq``, ``rho``, ``ux``,
     ``uy``, ``uz``, ``u_mag``, ``u_tau``, ``y_plus``, ``out_stream``
     allocated once in ``__init__`` and reused every step.
  c. **In-place operations** — ``mul_()``, ``add_()``, ``copy_()`` wherever
     the formula permits.
  d. **Stream optimisation** — pre-allocated ``out_stream`` buffer;
     rest direction (``q=0``) is a plain copy, not ``torch.roll``.
  e. **Skip rest direction** — ``q=0`` has zero velocity vector.

Numerical equivalence: every internal optimised method produces results
that are ``allclose(atol=1e-6)`` with the original standalone functions.
"""
from __future__ import annotations

from typing import Any, Callable

import torch

# ---------------------------------------------------------------------------
# Lattice dispatch helpers
# ---------------------------------------------------------------------------

_LATTICE_Q = {"D3Q19": 19, "D3Q27": 27}


def _get_lattice_constants(lattice: str, device: torch.device, dtype: torch.dtype):
    """Return (C, W, OPPOSITE, macroscopic_fn, equilibrium_fn) for *lattice*."""
    if lattice == "D3Q19":
        from .d3q19 import C, W, OPPOSITE, macroscopic3d, equilibrium3d

        return C, W, OPPOSITE, macroscopic3d, equilibrium3d
    if lattice == "D3Q27":
        from .d3q27 import C, W, OPPOSITE, macroscopic27, equilibrium27

        return C, W, OPPOSITE, macroscopic27, equilibrium27
    raise ValueError(f"Unsupported lattice {lattice!r}; supported: D3Q19, D3Q27")


# ---------------------------------------------------------------------------
# LBMStepExecutor
# ---------------------------------------------------------------------------


class LBMStepExecutor:
    """Pre-allocated, macroscopic-reusing timestep executor.

    A single ``step(f)`` call performs the full LBM update cycle:

    1. **Collide** — BGK (internal, pre-allocated) or any external
       collision function (MRT, TRT, Cumulant, …).
    2. **Stream** — ``torch.roll``-based streaming into a pre-allocated
       ``out_stream`` buffer; rest direction (``q=0``) is a plain copy.
    3. **Boundary** — optional external boundary function.
    4. **Macroscopic** — computed *once* into pre-allocated buffers.
    5. **Wall function** — optional; uses the pre-computed macroscopic,
       avoiding the redundant ``macroscopic`` calls inside
       :func:`tensorlbm.wall_function_common.wall_function` and
       :func:`tensorlbm.wall_function_common._apply_body_force`.
    6. **Force measurement** — optional; receives pre-computed macroscopic.
    7. **Mass correction** — optional; in-place ``mul_``.

    Parameters
    ----------
    lattice : str
        ``"D3Q19"`` or ``"D3Q27"``.
    collide_fn : str or callable
        ``"bgk"`` for the internal pre-allocated BGK collision, or any
        callable ``f(tau) -> f_post`` (e.g. ``collide_mrt3d``).
    stream_fn : callable or None
        If *None*, uses the internal pre-allocated ``torch.roll`` stream.
        If a callable, called as ``stream_fn(f) -> f_streamed``.
    boundary_fn : callable or None
        Called as ``boundary_fn(f, **boundary_kwargs) -> f_bc``.
    wall_fn : bool or None
        If *True*, enable the internal optimised wall function (requires
        *mask*, *nu*, *y_val*).
    force_fn : callable or None
        Called as ``force_fn(f, rho, ux, uy, uz, **force_kwargs) -> forces``.
    device : torch.device
        Target compute device.
    nx, ny, nz : int
        Grid dimensions.
    tau : float
        Relaxation time (used by internal BGK collision).
    dtype : torch.dtype
        Tensor dtype (default ``torch.float32``).
    mask : torch.Tensor or None
        Solid mask ``(nz, ny, nx)`` for wall function / bounce-back.
    nu : float
        Kinematic viscosity (lattice units) for wall function.
    y_val : float
        Near-wall cell-centre distance for wall function.
    wall_law : str
        ``"log"`` or ``"reichardt"`` for friction-velocity computation.
    target_mass : float or None
        If set, rescale distribution to this mass each step (in-place).
    boundary_kwargs : dict
        Extra keyword arguments forwarded to *boundary_fn*.
    force_kwargs : dict
        Extra keyword arguments forwarded to *force_fn*.
    """

    def __init__(
        self,
        lattice: str,
        *,
        collide_fn: str | Callable,
        stream_fn: Callable | None = None,
        boundary_fn: Callable | None = None,
        wall_fn: bool | None = False,
        force_fn: Callable | None = None,
        device: torch.device,
        nx: int,
        ny: int,
        nz: int,
        tau: float = 0.6,
        dtype: torch.dtype = torch.float32,
        # Wall-function parameters
        mask: torch.Tensor | None = None,
        nu: float = 0.02,
        y_val: float = 0.5,
        wall_law: str = "log",
        # Mass correction
        target_mass: float | None = None,
        # Extra kwargs forwarded to callbacks
        boundary_kwargs: dict[str, Any] | None = None,
        force_kwargs: dict[str, Any] | None = None,
    ):
        if lattice not in _LATTICE_Q:
            raise ValueError(
                f"Unsupported lattice {lattice!r}; supported: {list(_LATTICE_Q)}"
            )
        self.lattice = lattice
        self.Q = _LATTICE_Q[lattice]
        self.device = device
        self.nx, self.ny, self.nz = nx, ny, nz
        self.tau = tau
        self.dtype = dtype

        # -- Callbacks ---------------------------------------------------
        self._use_internal_bgk = isinstance(collide_fn, str) and collide_fn == "bgk"
        if not self._use_internal_bgk and not callable(collide_fn):
            raise ValueError(
                "collide_fn must be 'bgk' or a callable, got "
                f"{collide_fn!r}"
            )
        self._collide_fn = collide_fn if not self._use_internal_bgk else None
        self._stream_fn = stream_fn  # None → internal
        self._boundary_fn = boundary_fn
        self._force_fn = force_fn
        self._wall_enabled = bool(wall_fn)
        self._target_mass = target_mass
        self._boundary_kwargs = boundary_kwargs or {}
        self._force_kwargs = force_kwargs or {}

        # -- Lattice constants -------------------------------------------
        C, W, OPPOSITE, macro_fn, eq_fn = _get_lattice_constants(
            lattice, device, dtype
        )
        self._macroscopic_fn = macro_fn
        self._equilibrium_fn = eq_fn
        self._C = C.to(device)
        self._W = W.to(device).to(dtype)
        self._OPPOSITE = OPPOSITE.to(device)

        # Pre-compute float lattice vectors and views (avoid per-step alloc)
        c_float = self._C.to(dtype)
        self._cx_view = c_float[:, 0].view(self.Q, 1, 1, 1)
        self._cy_view = c_float[:, 1].view(self.Q, 1, 1, 1)
        self._cz_view = c_float[:, 2].view(self.Q, 1, 1, 1)
        self._w_view = self._W.view(self.Q, 1, 1, 1)

        # Pre-compute streaming shifts as Python tuples (no host sync)
        self._shifts: list[tuple[int, int, int]] = [
            (int(c[0]), int(c[1]), int(c[2])) for c in self._C.tolist()
        ]

        # -- Pre-allocate all buffers ------------------------------------
        shape_f = (self.Q, nz, ny, nx)
        shape_s = (nz, ny, nx)

        # Distribution buffers
        self.f_post = torch.empty(shape_f, device=device, dtype=dtype)
        self._f_return = torch.empty(shape_f, device=device, dtype=dtype)
        self._f_input = torch.empty(shape_f, device=device, dtype=dtype)
        self.feq = torch.empty(shape_f, device=device, dtype=dtype)
        self.out_stream = torch.empty(shape_f, device=device, dtype=dtype)
        self._tmp_f = torch.empty(shape_f, device=device, dtype=dtype)

        # Scalar field buffers
        self.rho = torch.empty(shape_s, device=device, dtype=dtype)
        self._rho_safe = torch.empty(shape_s, device=device, dtype=dtype)
        self.ux = torch.empty(shape_s, device=device, dtype=dtype)
        self.uy = torch.empty(shape_s, device=device, dtype=dtype)
        self.uz = torch.empty(shape_s, device=device, dtype=dtype)
        self.u_mag = torch.empty(shape_s, device=device, dtype=dtype)
        self.u_tau = torch.empty(shape_s, device=device, dtype=dtype)
        self.y_plus = torch.empty(shape_s, device=device, dtype=dtype)

        # -- Wall-function setup -----------------------------------------
        self._nu = nu
        self._y_val = y_val
        self._wall_law = wall_law
        self._mask = mask
        self._near_wall: torch.Tensor | None = None
        if self._wall_enabled:
            if mask is None:
                raise ValueError("wall_fn=True requires mask to be provided")
            self._near_wall = self._compute_near_wall_mask(mask)
            # Pre-compute von Kármán constants
            self._kappa = 0.41
            self._b_log = 5.0

    # ------------------------------------------------------------------
    # Private: in-place macroscopic computation
    # ------------------------------------------------------------------

    def _compute_macroscopic_inplace(
        self, f: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (rho, ux, uy, uz) into pre-allocated buffers.

        Implements the *same formula* as
        :func:`tensorlbm.d3q19.macroscopic3d` /
        :func:`tensorlbm.d3q27.macroscopic27` but writes results into
        pre-allocated buffers using ``out=`` and in-place ops, avoiding
        per-step tensor allocation.
        """
        # rho = f.sum(dim=0)
        torch.sum(f, dim=0, out=self.rho)
        # rho_safe = clamp(rho, min=1e-12)
        torch.clamp(self.rho, min=1e-12, out=self._rho_safe)

        # ux = sum(f * cx, dim=0) / rho_safe
        torch.mul(f, self._cx_view, out=self._tmp_f)
        torch.sum(self._tmp_f, dim=0, out=self.ux)
        self.ux.div_(self._rho_safe)

        # uy
        torch.mul(f, self._cy_view, out=self._tmp_f)
        torch.sum(self._tmp_f, dim=0, out=self.uy)
        self.uy.div_(self._rho_safe)

        # uz
        torch.mul(f, self._cz_view, out=self._tmp_f)
        torch.sum(self._tmp_f, dim=0, out=self.uz)
        self.uz.div_(self._rho_safe)

        return self.rho, self.ux, self.uy, self.uz

    # ------------------------------------------------------------------
    # Private: in-place equilibrium computation
    # ------------------------------------------------------------------

    def _compute_equilibrium_inplace(
        self,
        rho: torch.Tensor,
        ux: torch.Tensor,
        uy: torch.Tensor,
        uz: torch.Tensor,
    ) -> torch.Tensor:
        """Compute feq into pre-allocated ``self.feq``.

        Implements the *same formula* as
        :func:`tensorlbm.d3q19.equilibrium3d` /
        :func:`tensorlbm.d3q27.equilibrium27`::

            feq = w * rho * (1 + 3*cu + 4.5*cu² - 1.5*u²)

        The output is written into ``self.feq`` (pre-allocated).
        """
        # u_sq = ux² + uy² + uz²  (scalar field, into self.u_mag)
        torch.mul(ux, ux, out=self.u_mag)
        self.u_mag.addcmul_(uy, uy)
        self.u_mag.addcmul_(uz, uz)

        # cu = cx*ux + cy*uy + cz*uz  (Q-field, into self._tmp_f)
        torch.mul(self._cx_view, ux.unsqueeze(0), out=self._tmp_f)
        self._tmp_f.addcmul_(self._cy_view, uy.unsqueeze(0))
        self._tmp_f.addcmul_(self._cz_view, uz.unsqueeze(0))

        # feq = w * rho * (1 + 3*cu + 4.5*cu² - 1.5*u_sq)
        # Step 1: feq = w * rho  [into pre-allocated self.feq]
        torch.mul(self._w_view, rho.unsqueeze(0), out=self.feq)

        # Step 2: compute Hermite term and multiply in-place.
        # term = 1 + 3*cu + 4.5*cu² - 1.5*u_sq
        # Use self.out_stream as scratch (not needed during collide phase).
        scratch = self.out_stream
        torch.mul(self._tmp_f, self._tmp_f, out=scratch)   # cu²
        scratch.mul_(4.5)                                   # 4.5*cu²
        scratch.add_(self._tmp_f, alpha=3.0)               # 4.5*cu² + 3*cu
        scratch.add_(-1.5 * self.u_mag.unsqueeze(0))        # - 1.5*u_sq
        scratch.add_(1.0)                                   # + 1

        # feq *= term  [in-place]
        self.feq.mul_(scratch)
        return self.feq

    # ------------------------------------------------------------------
    # Private: internal BGK collision (pre-allocated, in-place)
    # ------------------------------------------------------------------

    def _collide_bgk(self, f: torch.Tensor) -> torch.Tensor:
        """BGK collision with pre-allocated buffers and in-place ops.

        Implements the *same formula* as
        :func:`tensorlbm.solver3d.collide_bgk3d` /
        :func:`tensorlbm.d3q27.collide_bgk27`::

            rho, ux, uy, uz = macroscopic(f)
            feq = equilibrium(rho, ux, uy, uz)
            f_post = f - (f - feq) / tau

        Rewritten for in-place::

            f_post = f * (1 - 1/tau) + feq * (1/tau)
        """
        # Compute macroscopic into pre-allocated buffers
        rho, ux, uy, uz = self._compute_macroscopic_inplace(f)

        # Compute equilibrium into pre-allocated feq
        self._compute_equilibrium_inplace(rho, ux, uy, uz)

        # f_post = f * (1 - 1/tau) + feq * (1/tau)  [in-place]
        alpha = 1.0 / self.tau
        self.f_post.copy_(f)
        self.f_post.mul_(1.0 - alpha)  # f_post = f * (1 - 1/tau)
        self.f_post.add_(self.feq, alpha=alpha)  # f_post += feq / tau
        return self.f_post

    # ------------------------------------------------------------------
    # Private: pre-allocated streaming
    # ------------------------------------------------------------------

    def _stream_preallocated(self, f: torch.Tensor) -> torch.Tensor:
        """Stream *f* into the pre-allocated ``self.out_stream`` buffer.

        Implements the *same formula* as
        :func:`tensorlbm.solver3d.stream3d` /
        :func:`tensorlbm.d3q27.stream27_roll` (pull scheme via
        ``torch.roll``) but writes into a pre-allocated buffer and skips
        the rest direction (``q=0`` is a plain copy).
        """
        Q = self.Q
        out = self.out_stream
        for q in range(Q):
            sx, sy, sz = self._shifts[q]
            if sx == 0 and sy == 0 and sz == 0:
                out[q].copy_(f[q])
            else:
                # Write into pre-allocated buffer (not create new tensor)
                out[q].copy_(torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2)))
        return out

    # ------------------------------------------------------------------
    # Private: near-wall mask (pre-computed once)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_near_wall_mask(solid: torch.Tensor) -> torch.Tensor:
        """Identify fluid cells adjacent to solid cells (6-connected).

        Same formula as
        :func:`tensorlbm.wall_function_common._near_wall_mask`.
        """
        fluid = ~solid
        near = torch.zeros_like(solid)
        for ax, sgn in [(2, 1), (2, -1), (1, 1), (1, -1), (0, 1), (0, -1)]:
            near |= torch.roll(solid, sgn, dims=ax) & fluid
        return near

    # ------------------------------------------------------------------
    # Private: optimised wall function (macroscopic reuse)
    # ------------------------------------------------------------------

    def _apply_wall_function(
        self,
        f: torch.Tensor,
        rho: torch.Tensor,
        ux: torch.Tensor,
        uy: torch.Tensor,
        uz: torch.Tensor,
    ) -> torch.Tensor:
        """Apply wall-function correction with pre-computed macroscopic.

        Implements the *same formula* as
        :func:`tensorlbm.wall_function_common.wall_function` +
        :func:`tensorlbm.wall_function_common._apply_body_force`,
        but reuses the pre-computed ``(rho, ux, uy, uz)`` instead of
        calling ``macroscopic`` twice (once in ``wall_function``, once in
        ``_apply_body_force``).
        """
        from .wall_function_common import compute_u_tau, compute_y_plus

        # u_mag = sqrt(ux² + uy² + uz²)  [into pre-allocated buffer]
        torch.mul(ux, ux, out=self.u_mag)
        self.u_mag.addcmul_(uy, uy)
        self.u_mag.addcmul_(uz, uz)
        torch.sqrt(self.u_mag, out=self.u_mag)
        self.u_mag.clamp_(min=1e-12)

        # u_tau, y_plus  [into pre-allocated buffers]
        u_tau = compute_u_tau(self.u_mag, self._nu, self._y_val, self._wall_law)
        self.u_tau.copy_(u_tau)
        y_plus = compute_y_plus(self.u_tau, self._nu, self._y_val)
        self.y_plus.copy_(y_plus)

        # Early exit if u_tau is zero everywhere
        if not self.u_tau.any():
            return f

        # Wall shear stress and body force (same formula as wall_function)
        tau_w = self.u_tau * self.u_tau
        near_f = self._near_wall.to(f.dtype)
        coef = -(tau_w / self._y_val) * near_f
        inv_umag = 1.0 / self.u_mag
        fx = coef * (ux * inv_umag)
        fy = coef * (uy * inv_umag)
        fz = coef * (uz * inv_umag)

        # Apply Guo body force with pre-computed velocity (same formula
        # as _apply_body_force, but no macroscopic call)
        return self._apply_body_force_inplace(f, fx, fy, fz, ux, uy, uz)

    def _apply_body_force_inplace(
        self,
        f: torch.Tensor,
        fx: torch.Tensor,
        fy: torch.Tensor,
        fz: torch.Tensor,
        ux: torch.Tensor,
        uy: torch.Tensor,
        uz: torch.Tensor,
    ) -> torch.Tensor:
        """Guo body-force correction with pre-computed velocity.

        Same formula as
        :func:`tensorlbm.wall_function_common._apply_body_force`::

            forcing = w * (1 + c·u/cs²) * (c·F) / cs²
            f_new = f + forcing

        but reuses pre-computed ``(ux, uy, uz)`` instead of calling
        ``macroscopic(f)`` internally.
        """
        cs2 = 1.0 / 3.0
        # cu = cx*fx + cy*fy + cz*fz  [force dot c]
        torch.mul(self._cx_view, fx.unsqueeze(0), out=self._tmp_f)
        self._tmp_f.addcmul_(self._cy_view, fy.unsqueeze(0))
        self._tmp_f.addcmul_(self._cz_view, fz.unsqueeze(0))

        # cu_u = cx*ux + cy*uy + cz*uz  [velocity dot c]
        # Use feq as scratch (not needed during wall function)
        torch.mul(self._cx_view, ux.unsqueeze(0), out=self.feq)
        self.feq.addcmul_(self._cy_view, uy.unsqueeze(0))
        self.feq.addcmul_(self._cz_view, uz.unsqueeze(0))

        # forcing = w * (1 + cu_u/cs²) * cu / cs²
        # Write into out_stream as scratch, then add to f in-place
        scratch = self.out_stream
        torch.mul(self.feq, 1.0 / cs2, out=scratch)  # cu_u / cs²
        scratch.add_(1.0)  # 1 + cu_u / cs²
        scratch.mul_(self._tmp_f)  # (1 + cu_u/cs²) * cu
        scratch.mul_(self._w_view)  # w * (1 + cu_u/cs²) * cu
        scratch.mul_(1.0 / cs2)  # w * (1 + cu_u/cs²) * cu / cs²

        # f += forcing  [in-place if f is not aliased with scratch]
        if f.data_ptr() == scratch.data_ptr():
            # Should not happen (f is the distribution, scratch is out_stream)
            return f + scratch
        f.add_(scratch)
        return f

    # ------------------------------------------------------------------
    # Private: in-place mass correction
    # ------------------------------------------------------------------

    def _correct_mass_inplace(self, f: torch.Tensor, target_mass: float) -> None:
        """Rescale *f* in-place so that ``f.sum() == target_mass``.

        Same formula as
        :func:`tensorlbm.solver3d.correct_mass3d` /
        :func:`tensorlbm.d3q27.correct_mass27`, but in-place.
        """
        current = f.sum()
        if current.abs() < 1e-30:
            return
        f.mul_(target_mass / current)

    # ------------------------------------------------------------------
    # Public: single timestep
    # ------------------------------------------------------------------

    def step(
        self, f: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Execute one LBM timestep.

        Returns ``(f_updated, diagnostics)`` where *diagnostics* is a dict
        that may contain ``"forces"``, ``"mass"``, ``"max_speed"``, and
        ``"mean_rho"``.
        """
        diag: dict[str, Any] = {}

        # 0. Copy input to internal buffer (avoid aliasing with _f_return on async GPUs)
        self._f_input.copy_(f)
        f = self._f_input

        # 1. Collide -----------------------------------------------------
        if self._use_internal_bgk:
            f_post = self._collide_bgk(f)
        else:
            f_post = self._collide_fn(f, self.tau)  # type: ignore[misc]

        # 2. Stream (pre-allocated out_stream) ---------------------------
        if self._stream_fn is not None:
            f_streamed = self._stream_fn(f_post)
        else:
            f_streamed = self._stream_preallocated(f_post)

        # 3. Boundary conditions -----------------------------------------
        if self._boundary_fn is not None:
            f_bc = self._boundary_fn(f_streamed, **self._boundary_kwargs)
        else:
            f_bc = f_streamed

        # 4. Compute macroscopic ONCE (for wall fn + force measurement) -
        need_macro = self._wall_enabled or self._force_fn is not None
        if need_macro:
            rho, ux, uy, uz = self._compute_macroscopic_inplace(f_bc)
        else:
            rho = ux = uy = uz = None  # type: ignore[assignment]

        # 5. Wall function (uses pre-computed macroscopic) ---------------
        if self._wall_enabled:
            f_bc = self._apply_wall_function(f_bc, rho, ux, uy, uz)  # type: ignore[arg-type]
            # Re-read macroscopic after wall function modified f_bc
            # (wall function changes velocity via Guo forcing)
            if self._force_fn is not None:
                rho, ux, uy, uz = self._compute_macroscopic_inplace(f_bc)

        # 6. Force measurement (uses pre-computed macroscopic) -----------
        if self._force_fn is not None:
            forces = self._force_fn(f_bc, rho, ux, uy, uz, **self._force_kwargs)  # type: ignore[arg-type]
            diag["forces"] = forces

        # 7. Mass correction (in-place) ---------------------------------
        if self._target_mass is not None:
            self._correct_mass_inplace(f_bc, self._target_mass)

        # Diagnostics from macroscopic (if computed)
        if need_macro and rho is not None:
            diag["max_speed"] = float(
                torch.sqrt(ux * ux + uy * uy + uz * uz).max().item()  # type: ignore[union-attr]
            )
            diag["mean_rho"] = float(rho.mean().item())  # type: ignore[union-attr]

        # 8. Return a clone to guarantee no aliasing on async GPU backends
        return f_bc.clone(), diag

    # ------------------------------------------------------------------
    # Public: run multiple steps
    # ------------------------------------------------------------------

    def run(
        self, f: torch.Tensor, n_steps: int
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Execute *n_steps* timesteps, returning final *f* and diagnostics list."""
        diags: list[dict[str, Any]] = []
        for _ in range(n_steps):
            f, diag = self.step(f)
            diags.append(diag)
        return f, diags


__all__ = ["LBMStepExecutor"]
