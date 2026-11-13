"""XVAL: vivsim ``jax-pure-function-aa-pattern-static-slice`` cross-validation.

Anchor: XVAL-LBM-JAX-PURE-FUNCTION-AA-PATTERN-STATIC-SLIC
OSS reference: haimingz/vivsim  (inventoried, slug NEW:LBM-GPU-JAX-FUSION)

The reference pattern (read from vivsim source, zero hand-copy of numbers):
  * AA layout: distribution ``f`` shaped ``(q, NX, NY)`` so the spatial dims are
    contiguous and XLA can coalesce memory access.
  * Static-slice streaming: the 9 D2Q9 unit shifts are spelled out with
    ``concatenate`` of static slices instead of a generic ``roll``; the work is
    visible to the compiler as simple concatenates.
  * Pure functions: every operator takes ``f`` and returns a new ``f``.
  * Time-loop fusion: ``jax.jit(donate_argnums=0)`` + ``jax.lax.scan``.
  * Benchmark method: warmup + repeat + ``block_until_ready``, per-operator.

This module ports the *streaming pattern* to TensorLBM's PyTorch backend and
its axis convention ``f: (9, ny, nx)`` (axis1 = y, axis2 = x; note vivsim uses
``(9, NX, NY)`` i.e. axis1 = x, axis2 = y).  Three streaming implementations
are provided so the pattern can be isolated:

  * :func:`stream_static_slice`  -- port of vivsim ``lbm.streaming`` (standard
    pull: ``f_out[i] = f[i] shifted by -c_i``).  THE reference pattern.
  * :func:`stream_roll_std`     -- standard pull via ``torch.roll`` (identical
    math to static-slice; the performance baseline for the pattern).
  * :func:`stream_roll`         -- TensorLBM's existing ``d2q9.stream``
    (``shifts=(-cy,-cx)``), kept verbatim to report how the current in-repo
    streaming behaves against the analytical solution.

Physics (2D D2Q9 force-driven Poiseuille channel, matching vivsim's
``examples/2d/poiseuille_channel.py`` parameter block):
  * BGK collision + EDM (Exact Difference Method) forcing.
  * Force-corrected NEBB (Zou/He) at top/bottom walls.
  * Full-distribution convention (TensorLBM): ``rho = sum(f)``.  vivsim uses a
    zero-centred convention ``f_i = f_i^full - w_i`` so ``rho = 1 + sum(f)``;
    that convention difference is documented, not silently mixed in here.
"""

from __future__ import annotations

import torch

from .d2q9 import C, W, CS2

# D2Q9 velocity set (cx, cy) -- identical to tensorlbm.d2q9.C
#   0:(0,0) 1:(1,0) 2:(0,1) 3:(-1,0) 4:(0,-1) 5:(1,1) 6:(-1,1) 7:(-1,-1) 8:(1,-1)
_CX = C[:, 0]
_CY = C[:, 1]


# --------------------------------------------------------------------------- #
# Streaming implementations
# --------------------------------------------------------------------------- #
def _shift_x_pos(f: torch.Tensor) -> torch.Tensor:
    """Pull from x-1: result[..., x] = f[..., x-1]  (last dim = x)."""
    return torch.cat((f[..., -1:], f[..., :-1]), dim=-1)


def _shift_x_neg(f: torch.Tensor) -> torch.Tensor:
    """Pull from x+1: result[..., x] = f[..., x+1]  (last dim = x)."""
    return torch.cat((f[..., 1:], f[..., :1]), dim=-1)


def _shift_y_pos(f: torch.Tensor) -> torch.Tensor:
    """Pull from y-1: result[y, ...] = f[y-1, ...]  (dim -2 = y)."""
    return torch.cat((f[:, -1:, :], f[:, :-1, :]), dim=1)


def _shift_y_neg(f: torch.Tensor) -> torch.Tensor:
    """Pull from y+1: result[y, ...] = f[y+1, ...]  (dim -2 = y)."""
    return torch.cat((f[:, 1:, :], f[:, :1, :]), dim=1)


def stream_static_slice(f: torch.Tensor) -> torch.Tensor:
    """Static-slice (concatenate) streaming -- port of vivsim ``lbm.streaming``.

    Standard pull convention: ``f_out[i][y, x] = f[i][y - cy_i, x - cx_i]``.
    Periodic wrap-around in both directions.  This is the reference pattern.
    """
    return torch.cat(
        (
            f[0:1],
            _shift_x_pos(f[1:2]),
            _shift_y_pos(f[2:3]),
            _shift_x_neg(f[3:4]),
            _shift_y_neg(f[4:5]),
            _shift_x_pos(_shift_y_pos(f[5:6])),
            _shift_x_neg(_shift_y_pos(f[6:7])),
            _shift_x_neg(_shift_y_neg(f[7:8])),
            _shift_x_pos(_shift_y_neg(f[8:9])),
        ),
        dim=0,
    )


def stream_roll_std(f: torch.Tensor) -> torch.Tensor:
    """Standard pull streaming via ``torch.roll`` (same physics as static-slice).

    ``f_out[i][y, x] = f[i][y - cy_i, x - cx_i]``  ->  ``roll(shifts=(cy, cx))``.
    Used as the performance baseline for the pattern (identical math to
    :func:`stream_static_slice`).
    """
    f_out = torch.empty_like(f)
    for i in range(f.shape[0]):
        cx, cy = int(_CX[i]), int(_CY[i])
        if cx == 0 and cy == 0:
            f_out[i] = f[i]
        else:
            f_out[i] = torch.roll(f[i], shifts=(cy, cx), dims=(0, 1))
    return f_out


def stream_roll(f: torch.Tensor) -> torch.Tensor:
    """TensorLBM's existing ``d2q9.stream`` (verbatim logic).

    ``torch.roll(f[i], shifts=(-cy, -cx), dims=(0, 1))``.  Kept verbatim so the
    XVAL can report how the *current* in-repo streaming behaves against the
    analytical solution.
    """
    f_out = torch.empty_like(f)
    for i in range(f.shape[0]):
        cx, cy = int(_CX[i]), int(_CY[i])
        if cx == 0 and cy == 0:
            f_out[i] = f[i]
        else:
            f_out[i] = torch.roll(f[i], shifts=(-cy, -cx), dims=(0, 1))
    return f_out


# --------------------------------------------------------------------------- #
# Collision + forcing + boundary (ported from vivsim, full-distribution form)
# --------------------------------------------------------------------------- #
def macroscopic(f: torch.Tensor):
    """rho, ux, uy from full distributions.  rho = sum(f)."""
    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    cx = _CX.to(dtype=f.dtype).view(9, 1, 1)
    cy = _CY.to(dtype=f.dtype).view(9, 1, 1)
    ux = (f * cx).sum(dim=0) / rho_safe
    uy = (f * cy).sum(dim=0) / rho_safe
    return rho, ux, uy


def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q9 equilibrium (full distribution)."""
    c = C.to(dtype=rho.dtype)
    w = W.to(dtype=rho.dtype)
    cu = c[:, 0].view(9, 1, 1) * ux.unsqueeze(0) + c[:, 1].view(9, 1, 1) * uy.unsqueeze(0)
    u2 = ux * ux + uy * uy
    return w.view(9, 1, 1) * rho.unsqueeze(0) * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2.unsqueeze(0)
    )


def collide_bgk(f: torch.Tensor, feq: torch.Tensor, omega: float) -> torch.Tensor:
    """BGK (SRT) collision: f - omega (f - feq)."""
    return f - omega * (f - feq)


def get_guo_forcing_term(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Guo forcing term -- verbatim port of vivsim ``forcing/guo.py``.

    ``term_i = w_i [ g_x (3(c_ix - u_x) + 9 uc c_ix) + g_y (3(c_iy - u_y) + 9 uc c_iy) ]``
    with ``uc = u_x c_ix + u_y c_iy``.  ``g`` and ``u`` are ``(2, ny, nx)``.
    """
    c = C.to(dtype=g.dtype)
    w = W.to(dtype=g.dtype)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    gx, gy = g[0], g[1]
    ux, uy = u[0], u[1]
    uc = ux.unsqueeze(0) * cx + uy.unsqueeze(0) * cy
    return w.view(9, 1, 1) * (
        gx.unsqueeze(0) * (3.0 * (cx - ux.unsqueeze(0)) + 9.0 * uc * cx)
        + gy.unsqueeze(0) * (3.0 * (cy - uy.unsqueeze(0)) + 9.0 * uc * cy)
    )


def forcing_edm(f: torch.Tensor, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """EDM forcing -- vivsim ``forcing/edm.py``: ``f + get_guo_forcing_term(g, u)``.

    Algebraically ``feq(rho, u + g/2rho) - feq(rho, u - g/2rho)``; equivalent to
    Guo's scheme.  Uses the *uncorrected* velocity ``u``.
    """
    return f + get_guo_forcing_term(g, u)


def forcing_guo_bgk(f: torch.Tensor, g: torch.Tensor, u: torch.Tensor, omega: float) -> torch.Tensor:
    """Guo forcing for BGK -- vivsim ``forcing_guo_bgk``: scale by ``(1 - 0.5 omega)``."""
    return f + get_guo_forcing_term(g, u) * (1.0 - 0.5 * omega)


def get_omega(nu: float) -> float:
    """Lattice relaxation rate omega = 1 / (3 nu + 0.5)."""
    return 1.0 / (3.0 * nu + 0.5)


def _nebb_wall(f: torch.Tensor, loc: str, rho_wall: float, ux_wall: float, uy_wall: float) -> torch.Tensor:
    """Non-equilibrium bounce-back (Zou/He) on a horizontal wall row.

    ``loc`` in {"top", "bottom"}.  Ported from vivsim ``boundary_nebb`` with the
    axis convention of this module (axis1 = y).  Only the three unknown incoming
    distributions are rebuilt; outgoing ones are left untouched.
    """
    if loc == "top":
        row = -1
        in_dirs = [4, 7, 8]     # cy < 0 (point into domain from top)
        out_dirs = [2, 5, 6]    # cy > 0
        tan = [1, 3]           # x directions
        normal_sign = -1.0
        un = -uy_wall          # outward normal at top is +y
        ut = ux_wall
    else:  # bottom
        row = 0
        in_dirs = [2, 5, 6]    # cy > 0
        out_dirs = [4, 7, 8]
        tan = [1, 3]
        normal_sign = 1.0
        un = uy_wall
        ut = ux_wall

    wall = f[:, row, :]  # (9, nx)
    shear = 0.5 * (wall[tan[0]] - wall[tan[1]]) * normal_sign
    normal_term = (1.0 / 6.0) * un * rho_wall
    tangential_term = 0.5 * ut * rho_wall
    new_vals = torch.stack(
        [
            wall[out_dirs[0]] + (2.0 / 3.0) * un * rho_wall,
            wall[out_dirs[1]] - shear + normal_term + tangential_term,
            wall[out_dirs[2]] + shear + normal_term - tangential_term,
        ]
    )
    new_wall = wall.clone()
    for k, d in enumerate(in_dirs):
        new_wall[d] = new_vals[k]
    f = f.clone()
    f[:, row, :] = new_wall
    return f


def boundary_force_corrected_nebb(
    f: torch.Tensor, loc: str, gx_wall: float, rho_wall: float = 1.0
) -> torch.Tensor:
    """Force-corrected NEBB: wall velocity corrected by ``-g/(2 rho)`` first.

    vivsim convention: ``u_wall^lattice = u_wall - g/(2 rho)`` with ``rho_wall=1``.
    The wall is at rest (ux_wall = uy_wall = 0) so the corrected tangential
    velocity is ``-gx_wall / (2 rho_wall)``.
    """
    ux_c = 0.0 - gx_wall / (2.0 * rho_wall)
    uy_c = 0.0
    return _nebb_wall(f, loc, rho_wall, ux_c, uy_c)


def make_poiseuille_step(
    stream_fn,
    omega: float,
    gx: float,
    ny: int,
    nx: int,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
):
    """Build a pure ``step(f) -> f`` for the Poiseuille channel.

    Order (matches vivsim BGK+EDM recipe): macroscopic -> equilibrium ->
    BGK collide -> EDM force -> stream -> force-corrected NEBB (top, bottom).
    """
    g = torch.zeros(2, ny, nx, dtype=dtype, device=device)
    g[0] = gx

    def step(f: torch.Tensor) -> torch.Tensor:
        rho, ux, uy = macroscopic(f)
        u = torch.stack([ux, uy])
        feq = equilibrium(rho, ux, uy)
        f = collide_bgk(f, feq, omega)
        f = forcing_edm(f, g, u)
        f = stream_fn(f)
        f = boundary_force_corrected_nebb(f, "top", gx_wall=gx)
        f = boundary_force_corrected_nebb(f, "bottom", gx_wall=gx)
        return f

    return step


def init_equilibrium(ny: int, nx: int, gx: float, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Initial distribution at rest with the force-corrected velocity ``-g/2``."""
    rho = torch.ones(ny, nx, dtype=dtype, device=device)
    ux = torch.full((ny, nx), -gx / 2.0, dtype=dtype, device=device)
    uy = torch.zeros(ny, nx, dtype=dtype, device=device)
    return equilibrium(rho, ux, uy)


def run_poiseuille(
    stream_fn,
    ny: int,
    nx: int,
    nu: float,
    gx: float,
    n_steps: int,
    dtype=torch.float64,
    device: str = "cpu",
    profile_every: int = 0,
):
    """Run the Poiseuille channel; return (f, rho, ux, uy, history).

    ``history`` collects (step, ux_profile) when ``profile_every > 0``.
    """
    omega = get_omega(nu)
    step = make_poiseuille_step(stream_fn, omega, gx, ny, nx, dtype, device)
    f = init_equilibrium(ny, nx, gx, dtype, device)
    history = []
    for t in range(n_steps):
        f = step(f)
        if profile_every and (t + 1) % profile_every == 0:
            _, ux, _ = macroscopic(f)
            history.append((t + 1, ux[:, nx // 2].clone()))
    rho, ux, uy = macroscopic(f)
    return f, rho, ux, uy, history


def analytical_ux(ny: int, gx: float, nu: float) -> torch.Tensor:
    """Analytical steady Poiseuille profile ``u = gx/(2 nu) y (H - y)``, H = ny-1."""
    H = ny - 1
    y = torch.arange(ny, dtype=torch.float64)
    return (gx / (2.0 * nu)) * y * (H - y)
