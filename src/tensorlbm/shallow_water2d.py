"""2-D shallow-water dam-break solver + Ritter (1892) self-similar anchor.

Why this module exists
----------------------
:mod:`tensorlbm.shallow_water` integrates the depth-averaged SWE in **1-D**
(``h(x, t)``).  The dam-break verification face needs a genuinely two-dimensional
solver (a 2-D free-surface field that a mesh-refinement study can be run on)
whose analytic anchor ``h(x, t)`` is the same Ritter solution.  This module is
that solver: a Cartesian finite-volume SWE integrator in flux form.

Equations (flat bed ``z_b = 0``, no friction)
--------------------------------------------

.. math::

    \partial_t h + \partial_x (hu) + \partial_y (hv) = 0

    \partial_t (hu) + \partial_x\!\left(hu^2 + \tfrac12 g h^2\right)
        + \partial_y (huv) = 0

    \partial_t (hv) + \partial_x (huv)
        + \partial_y\!\left(hv^2 + \tfrac12 g h^2\right) = 0

Numerics
--------
* **Flux form** on a uniform Cartesian grid -> discrete mass conservation to
  round-off (the mass gate of the verification face is a *structural* property,
  not a tuned one).
* **HLL** approximate Riemann solver with Davis/Einfeldt wave-speed estimates,
  positivity preserving for the wet/dry problem.
* **Hydrostatic reconstruction** (Audusse et al. 2004) with a vanishing dry-bed
  depth ``h_dry``: exactly preserves lake-at-rest and keeps the wetting front
  free of negative depths.
* **MUSCL-Hancock** piecewise-linear predictor-corrector (van Leer slopes with a
  ``minmod`` limiter) as an opt-in second-order mode.  ``order=1`` reproduces
  the classical first-order Godunov scheme.
* Reflective (free-slip) or transmissive (zero-gradient) boundaries per side.

The 1-D Ritter solution is used as the analytic anchor:

.. math::

    h(x,t) = \frac{h_0}{9}\left(2 - \frac{x - x_{\rm dam}}{c_0 t}\right)^2,
    \qquad c_0 = \sqrt{g h_0},
    \qquad 0 \le x - x_{\rm dam} \le 2 c_0 t,

``h = h_0`` behind the rarefaction head ``x_dam - c_0 t`` and ``h = 0`` beyond
the front ``x_dam + 2 c_0 t``.  For a *finite* column of width ``a`` against a
wall, the wall state stays ``h_0`` while ``c_0 t <= a``, so the anchor applies
unchanged in that window.

References
----------
Ritter (1892); Harten, Lax & van Leer (1983); Audusse et al. (2004);
Toro, *Shock-Capturing Methods for Free-Surface Flows* (2001).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

H_DRY = 1.0e-6
"""Depth below which a cell is treated as dry (momentum is set to zero)."""


@dataclass
class SWEState:
    """Conservative state of the 2-D shallow-water solver."""

    h: np.ndarray
    hu: np.ndarray
    hv: np.ndarray
    dx: float = 1.0
    dy: float = 1.0
    g: float = 1.0
    t: float = 0.0
    history: dict[str, list[float]] = field(default_factory=dict)

    def copy(self) -> "SWEState":
        return SWEState(
            h=self.h.copy(),
            hu=self.hu.copy(),
            hv=self.hv.copy(),
            dx=self.dx,
            dy=self.dy,
            g=self.g,
            t=self.t,
            history={key: list(val) for key, val in self.history.items()},
        )

    def mass(self) -> float:
        """Total water volume ``sum(h) dx dy`` (the conserved quantity)."""
        return float(self.h.sum() * self.dx * self.dy)


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """minmod limiter: zero at local extrema, upwind-biased otherwise."""
    same_sign = a * b > 0.0
    return np.where(same_sign, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _slopes_x(q: np.ndarray) -> np.ndarray:
    """Limited x-slopes on cell centres (zero-gradient at the domain ends)."""
    dq = np.zeros_like(q)
    fwd = q[..., 2:] - q[..., 1:-1]
    bwd = q[..., 1:-1] - q[..., :-2]
    dq[..., 1:-1] = _minmod(fwd, bwd)
    return dq


def _slopes_y(q: np.ndarray) -> np.ndarray:
    """Limited y-slopes on cell centres (zero-gradient at the domain ends)."""
    dq = np.zeros_like(q)
    fwd = q[2:, :] - q[1:-1, :]
    bwd = q[1:-1, :] - q[:-2, :]
    dq[1:-1, :] = _minmod(fwd, bwd)
    return dq


def _velocity(state: SWEState) -> tuple[np.ndarray, np.ndarray]:
    """Desingularised velocities ``u = hu / max(h, h_dry)``."""
    h_safe = np.maximum(state.h, H_DRY)
    return state.hu / h_safe, state.hv / h_safe


def _hll_flux_x(
    h_l: np.ndarray,
    hu_l: np.ndarray,
    hv_l: np.ndarray,
    h_r: np.ndarray,
    hu_r: np.ndarray,
    hv_r: np.ndarray,
    g: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HLL flux in the x-direction for the 2-D SWE."""
    u_l = hu_l / np.maximum(h_l, H_DRY)
    v_l = hv_l / np.maximum(h_l, H_DRY)
    u_r = hu_r / np.maximum(h_r, H_DRY)
    v_r = hv_r / np.maximum(h_r, H_DRY)
    c_l = np.sqrt(g * np.maximum(h_l, 0.0))
    c_r = np.sqrt(g * np.maximum(h_r, 0.0))
    s_l = np.minimum(u_l - c_l, u_r - c_r)
    s_r = np.maximum(u_l + c_l, u_r + c_r)

    f_h_l, f_hu_l, f_hv_l = hu_l, hu_l * u_l + 0.5 * g * h_l**2, hu_l * v_l
    f_h_r, f_hu_r, f_hv_r = hu_r, hu_r * u_r + 0.5 * g * h_r**2, hu_r * v_r

    denom = s_r - s_l
    denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)
    inv = 1.0 / denom
    mid_h = (s_r * hu_l - s_l * hu_r + s_l * s_r * (h_r - h_l)) * inv
    mid_hu = (s_r * f_hu_l - s_l * f_hu_r + s_l * s_r * (hu_r - hu_l)) * inv
    mid_hv = (s_r * f_hv_l - s_l * f_hv_r + s_l * s_r * (hv_r - hv_l)) * inv

    left = s_l >= 0.0
    right = s_r <= 0.0
    f_h = np.where(left, f_h_l, np.where(right, f_h_r, mid_h))
    f_hu = np.where(left, f_hu_l, np.where(right, f_hu_r, mid_hu))
    f_hv = np.where(left, f_hv_l, np.where(right, f_hv_r, mid_hv))
    return f_h, f_hu, f_hv


def _hydrostatic_reconstruct(
    h_centre: np.ndarray,
    hu_centre: np.ndarray,
    hv_centre: np.ndarray,
    h_face: np.ndarray,
    hu_face: np.ndarray,
    hv_face: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positivity fix-up of a reconstructed face state (flat bed).

    On a flat bed the Audusse reconstruction reduces to ``h_face* = h_face``;
    what remains is the *dry* branch: a face state that is dry must carry no
    momentum (otherwise the Riemann solver sees a spurious jet), and a
    reconstructed depth that undershoots zero is clipped.
    """
    depth = np.maximum(h_face, 0.0)
    wet = depth > H_DRY
    hu_out = np.where(wet, hu_face, 0.0)
    hv_out = np.where(wet, hv_face, 0.0)
    return depth, hu_out, hv_out


def _face_states_x(state: SWEState, slopes: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """MUSCL face states in x: (h_l, hu_l, hv_l, h_r, hu_r, hv_r) on x-faces."""
    h = state.h
    hu = state.hu
    hv = state.hv
    dh = slopes["hx"]
    dhu = slopes["hux"]
    dhv = slopes["hvx"]
    # Left/right states for faces between cells (j, i) and (j, i+1).
    h_l = h[:, :-1] + 0.5 * dh[:, :-1]
    hu_l = hu[:, :-1] + 0.5 * dhu[:, :-1]
    hv_l = hv[:, :-1] + 0.5 * dhv[:, :-1]
    h_r = h[:, 1:] - 0.5 * dh[:, 1:]
    hu_r = hu[:, 1:] - 0.5 * dhu[:, 1:]
    hv_r = hv[:, 1:] - 0.5 * dhv[:, 1:]
    h_l, hu_l, hv_l = _hydrostatic_reconstruct(h[:, :-1], hu[:, :-1], hv[:, :-1], h_l, hu_l, hv_l)
    h_r, hu_r, hv_r = _hydrostatic_reconstruct(h[:, 1:], hu[:, 1:], hv[:, 1:], h_r, hu_r, hv_r)
    return h_l, hu_l, hv_l, h_r, hu_r, hv_r


def _face_states_y(state: SWEState, slopes: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """MUSCL face states in y: (h_l, hu_l, hv_l, h_r, hu_r, hv_r) on y-faces."""
    h = state.h
    hu = state.hu
    hv = state.hv
    dh = slopes["hy"]
    dhu = slopes["huy"]
    dhv = slopes["hvy"]
    h_l = h[:-1, :] + 0.5 * dh[:-1, :]
    hu_l = hu[:-1, :] + 0.5 * dhu[:-1, :]
    hv_l = hv[:-1, :] + 0.5 * dhv[:-1, :]
    h_r = h[1:, :] - 0.5 * dh[1:, :]
    hu_r = hu[1:, :] - 0.5 * dhu[1:, :]
    hv_r = hv[1:, :] - 0.5 * dhv[1:, :]
    h_l, hu_l, hv_l = _hydrostatic_reconstruct(h[:-1, :], hu[:-1, :], hv[:-1, :], h_l, hu_l, hv_l)
    h_r, hu_r, hv_r = _hydrostatic_reconstruct(h[1:, :], hu[1:, :], hv[1:, :], h_r, hu_r, hv_r)
    return h_l, hu_l, hv_l, h_r, hu_r, hv_r


def _fluxes_x(state: SWEState, slopes: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """HLL fluxes on all interior x-faces (shape ``(ny, nx - 1)``)."""
    h_l, hu_l, hv_l, h_r, hu_r, hv_r = _face_states_x(state, slopes)
    return _hll_flux_x(h_l, hu_l, hv_l, h_r, hu_r, hv_r, state.g)


def _fluxes_y(state: SWEState, slopes: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """HLL fluxes on all interior y-faces (shape ``(ny - 1, nx)``)."""
    h_l, hu_l, hv_l, h_r, hu_r, hv_r = _face_states_y(state, slopes)
    # Reuse the x-flux kernel by swapping the role of the normal velocity.
    f_h, f_hv, f_hu = _hll_flux_x(h_l, hv_l, hu_l, h_r, hv_r, hu_r, state.g)
    return f_h, f_hu, f_hv


def _zero_slopes(state: SWEState) -> dict[str, np.ndarray]:
    z = np.zeros_like(state.h)
    return {key: z.copy() for key in ("hx", "hux", "hvx", "hy", "huy", "hvy")}


def _compute_slopes(state: SWEState) -> dict[str, np.ndarray]:
    return {
        "hx": _slopes_x(state.h),
        "hux": _slopes_x(state.hu),
        "hvx": _slopes_x(state.hv),
        "hy": _slopes_y(state.h),
        "huy": _slopes_y(state.hu),
        "hvy": _slopes_y(state.hv),
    }


def max_wave_speed(state: SWEState) -> float:
    """Largest ``|u| + sqrt(g h)`` in the domain (for the CFL time step)."""
    u, v = _velocity(state)
    c = np.sqrt(state.g * np.maximum(state.h, 0.0))
    return float(np.max(np.maximum(np.abs(u) + c, np.abs(v) + c)))


def time_step(state: SWEState, cfl: float = 0.4) -> float:
    """CFL-limited stable time step."""
    speed = max_wave_speed(state)
    if speed <= 0.0:
        return float("inf")
    return cfl * min(state.dx, state.dy) / speed


def step(
    state: SWEState,
    dt: float,
    *,
    order: int = 2,
    boundaries: str = "reflective",
) -> SWEState:
    """Advance ``state`` by ``dt`` with one finite-volume update.

    ``order=1`` is the first-order Godunov scheme (piecewise-constant states),
    ``order=2`` activates the MUSCL-Hancock predictor-corrector.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    slopes = _compute_slopes(state) if order == 2 else _zero_slopes(state)
    h, hu, hv = state.h.copy(), state.hu.copy(), state.hv.copy()

    if order == 2:
        # MUSCL-Hancock half-step predictor (unsplit, both directions).
        fx = _fluxes_x(state, slopes)
        fy = _fluxes_y(state, slopes)
        dh = np.zeros_like(h)
        dhu = np.zeros_like(h)
        dhv = np.zeros_like(h)
        # dh/dt = -(F_R - F_L)/dx; the half step is applied by the 0.5*dt below.
        dh[:, 1:-1] += -(fx[0][:, 1:] - fx[0][:, :-1]) / state.dx
        dh[1:-1, :] += -(fy[0][1:, :] - fy[0][:-1, :]) / state.dy
        dhu[:, 1:-1] += -(fx[1][:, 1:] - fx[1][:, :-1]) / state.dx
        dhu[1:-1, :] += -(fy[1][1:, :] - fy[1][:-1, :]) / state.dy
        dhv[:, 1:-1] += -(fx[2][:, 1:] - fx[2][:, :-1]) / state.dx
        dhv[1:-1, :] += -(fy[2][1:, :] - fy[2][:-1, :]) / state.dy
        h_pred = h + 0.5 * dt * dh
        hu_pred = hu + 0.5 * dt * dhu
        hv_pred = hv + 0.5 * dt * dhv
        # Dry cells carry no momentum (predictor positivity guard).
        dry = h_pred <= H_DRY
        h_pred = np.where(dry, np.maximum(h_pred, 0.0), h_pred)
        hu_pred = np.where(dry, 0.0, hu_pred)
        hv_pred = np.where(dry, 0.0, hv_pred)
        pred = SWEState(
            h=h_pred,
            hu=hu_pred,
            hv=hv_pred,
            dx=state.dx,
            dy=state.dy,
            g=state.g,
            t=state.t + 0.5 * dt,
        )
        pred_slopes = _compute_slopes(pred)
        fx = _fluxes_x(pred, pred_slopes)
        fy = _fluxes_y(pred, pred_slopes)
    else:
        fx = _fluxes_x(state, slopes)
        fy = _fluxes_y(state, slopes)

    # Boundary fluxes (reflective -> zero mass flux, mirrored momentum flux is
    # handled by the zero-gradient state; transmissive -> first interior face).
    if boundaries == "reflective":
        # The wall face depth is linearly extrapolated from the adjacent cell so
        # the pressure flux at the wall is second-order accurate; a
        # cell-centred value would inject an O(dx) error that propagates into
        # the interior at the wave speed (the first-order floor of the scheme).
        h_edge = _wall_depth_extrapolated(state.h, order)
        fx_left = _wall_flux_x(h_edge[0], state.hu[:, 0], state.hv[:, 0], state.g)
        fx_right = _wall_flux_x(h_edge[1], state.hu[:, -1], state.hv[:, -1], state.g)
        fy_bottom = _wall_flux_y(h_edge[2], state.hu[0, :], state.hv[0, :], state.g)
        fy_top = _wall_flux_y(h_edge[3], state.hu[-1, :], state.hv[-1, :], state.g)
    elif boundaries == "transmissive":
        fx_left = (fx[0][:, 0], fx[1][:, 0], fx[2][:, 0])
        fx_right = (fx[0][:, -1], fx[1][:, -1], fx[2][:, -1])
        fy_bottom = (fy[0][0, :], fy[1][0, :], fy[2][0, :])
        fy_top = (fy[0][-1, :], fy[1][-1, :], fy[2][-1, :])
    else:
        raise ValueError(f"boundaries must be 'reflective' or 'transmissive', got {boundaries!r}")

    f_h = np.zeros((h.shape[0], h.shape[1] + 1))
    f_hu = np.zeros_like(f_h)
    f_hv = np.zeros_like(f_h)
    f_h[:, 0], f_hu[:, 0], f_hv[:, 0] = fx_left
    f_h[:, 1:-1] = fx[0]
    f_hu[:, 1:-1] = fx[1]
    f_hv[:, 1:-1] = fx[2]
    f_h[:, -1], f_hu[:, -1], f_hv[:, -1] = fx_right

    g_h = np.zeros((h.shape[0] + 1, h.shape[1]))
    g_hu = np.zeros_like(g_h)
    g_hv = np.zeros_like(g_h)
    g_h[0], g_hu[0], g_hv[0] = fy_bottom
    g_h[1:-1] = fy[0]
    g_hu[1:-1] = fy[1]
    g_hv[1:-1] = fy[2]
    g_h[-1], g_hu[-1], g_hv[-1] = fy_top

    h_new = (
        h - dt / state.dx * (f_h[:, 1:] - f_h[:, :-1]) - dt / state.dy * (g_h[1:, :] - g_h[:-1, :])
    )
    hu_new = (
        hu
        - dt / state.dx * (f_hu[:, 1:] - f_hu[:, :-1])
        - dt / state.dy * (g_hu[1:, :] - g_hu[:-1, :])
    )
    hv_new = (
        hv
        - dt / state.dx * (f_hv[:, 1:] - f_hv[:, :-1])
        - dt / state.dy * (g_hv[1:, :] - g_hv[:-1, :])
    )

    # Wet/dry bookkeeping: drained cells become exactly dry and momentum-free.
    dry = h_new <= H_DRY
    h_new = np.where(dry, 0.0, h_new)
    hu_new = np.where(dry, 0.0, hu_new)
    hv_new = np.where(dry, 0.0, hv_new)

    out = SWEState(
        h=h_new,
        hu=hu_new,
        hv=hv_new,
        dx=state.dx,
        dy=state.dy,
        g=state.g,
        t=state.t + dt,
        history=state.history,
    )
    return out


def _wall_depth_extrapolated(
    h: np.ndarray, order: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Depth at the four reflective walls, extrapolated to the wall face.

    ``order=1`` uses the adjacent cell-centred depth, ``order=2`` the linear
    extrapolation to the face (``1.5 h_0 - 0.5 h_1``).
    """
    if order == 2:
        left = h[:, 0] - 0.5 * (h[:, 1] - h[:, 0])
        right = h[:, -1] + 0.5 * (h[:, -1] - h[:, -2])
        bottom = h[0, :] - 0.5 * (h[1, :] - h[0, :])
        top = h[-1, :] + 0.5 * (h[-1, :] - h[-2, :])
    else:
        left, right = h[:, 0], h[:, -1]
        bottom, top = h[0, :], h[-1, :]
    return (
        np.maximum(left, 0.0),
        np.maximum(right, 0.0),
        np.maximum(bottom, 0.0),
        np.maximum(top, 0.0),
    )


def _wall_flux_x(
    h: np.ndarray, hu: np.ndarray, hv: np.ndarray, g: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reflective-wall x-flux: zero mass/momentum flux across the wall face."""
    return np.zeros_like(h), 0.5 * g * h**2, np.zeros_like(h)


def _wall_flux_y(
    h: np.ndarray, hu: np.ndarray, hv: np.ndarray, g: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reflective-wall y-flux: zero mass/momentum flux across the wall face."""
    return np.zeros_like(h), np.zeros_like(h), 0.5 * g * h**2


def dam_break_initial_state(
    ny: int,
    nx: int,
    *,
    depth: float,
    column_width: float,
    span_extent: float | None = None,
    dx: float = 1.0,
    dy: float = 1.0,
    g: float = 1.0,
) -> SWEState:
    """Still water of ``depth`` released onto a dry bed.

    ``span_extent=None`` gives the **planar** dam break used for the 1-D Ritter
    anchor: the depth field is uniform in ``y`` and equals ``depth`` for
    ``x < column_width``.  A finite ``span_extent`` instead wets only
    ``0 <= y < span_extent``, i.e. a rectangular patch that also spreads in
    ``y`` (the 2-D-in-plan case; it has no 1-D analytic counterpart).
    """
    h = np.zeros((ny, nx))
    i_end = int(round(column_width / dx))
    h[:, :i_end] = depth
    if span_extent is not None:
        j_end = int(round(span_extent / dy))
        h[j_end:, :] = 0.0
    zeros = np.zeros_like(h)
    return SWEState(h=h, hu=zeros.copy(), hv=zeros.copy(), dx=dx, dy=dy, g=g)


def simulate(
    state: SWEState,
    *,
    t_end: float,
    cfl: float = 0.4,
    order: int = 2,
    boundaries: str = "reflective",
    max_steps: int = 2_000_000,
) -> SWEState:
    """Integrate ``state`` to ``t_end``; records ``t`` and ``mass`` in ``history``."""
    state = state.copy()
    state.history = {"t": [state.t], "mass": [state.mass()]}
    steps = 0
    while state.t < t_end - 1e-12:
        dt = min(time_step(state, cfl), t_end - state.t)
        state = step(state, dt, order=order, boundaries=boundaries)
        state.history["t"].append(state.t)
        state.history["mass"].append(state.mass())
        steps += 1
        if steps > max_steps:
            raise RuntimeError("simulate exceeded max_steps; check the CFL condition")
        if not np.all(np.isfinite(state.h)):
            raise RuntimeError(f"non-finite depth at t={state.t:.6g} (step {steps})")
    return state


def depth_profile_x(state: SWEState, iters: int = 4) -> np.ndarray:
    """Spanwise-averaged depth ``h(x)`` (mean over y of the interior rows)."""
    rows = state.h
    if rows.shape[0] > 2 * iters:
        rows = rows[iters:-iters, :]
    return np.asarray(rows.mean(axis=0))


def mass_drift(state: SWEState) -> float:
    """Relative mass drift ``|M(t) - M(0)| / M(0)`` from the recorded history."""
    mass = np.asarray(state.history["mass"], dtype=float)
    if mass.size == 0 or mass[0] == 0.0:
        return 0.0
    return float(np.max(np.abs(mass - mass[0])) / mass[0])
