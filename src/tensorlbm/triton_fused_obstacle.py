"""Triton fused D3Q19 LBM step with wall mask, LES Smagorinsky, and BC helpers.

This module extends :mod:`tensorlbm.triton_fused` with the three things
the periodic-only kernel cannot do, in a single Triton launch:

  1. **Wall bounce-back** at obstacle cells.  The kernel reads an
     ``obstacle: int8[NZ,NY,NX]`` mask.  When a fluid cell's pull-stream
     source cell is an obstacle, the population at that direction is
     replaced by the population going the *opposite* direction at the
     fluid cell itself (``f[18-q]``).  This is the standard
     "wet-node" bounce-back used by most CFD-grade LBM codes.

  2. **LES Smagorinsky** eddy-viscosity.  After the BGK equilibrium
     is computed, the kernel derives the strain-rate tensor ``S_ij``
     from the non-equilibrium populations (``f - f_eq``) using the
     standard ``S_ij = -1/(2τ_mol) * Σ_q fneq_q c_qi c_qj / ρ``,
     forms ``|S| = sqrt(2 S_ij S_ij)``, and uses ``ν_t = (Cs·Δ)²·|S|``
     to replace the constant τ with a per-cell effective τ.  The
     molecular τ stays fixed; only the additional eddy viscosity
     varies per cell.

  3. **Inflow / outflow BC helpers** (in PyTorch, on host).  These
     apply the Zou-He velocity inlet on the leftmost z-plane and the
     zero-gradient outflow on the rightmost z-plane.  They touch one
     or two cells per step and are cheap.

Measured on RTX 5090 at n=256 with the SUBOFF obstacle (≈ 8% of cells
are walls) the fused obstacle+LES kernel hits 6.1 GLUPS, ≈ 71% of
the periodic-kernel ceiling — i.e. the wall overhead is small because
the obstacle mask is mostly 0 and the bounce-back path adds only a
single ``tl.where`` per population read.

Limitations
-----------
D3Q19, fp32 storage only.  Wall bounce-back is first-order; for
quantitative drag prediction on SUBOFF-class flows the user should
verify against a finer grid or a multi-relaxation-time (MRT) variant.
Smagorinsky Cs is a constant (no dynamic procedure).
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

try:
    # Local / pre-deployment path: files live at the repo root with
    # underscore-separated names.
    from tensorlbm_triton_fused import (
        DEFAULT_BLOCK_X,
        DEFAULT_BLOCK_Y,
        DEFAULT_NUM_STAGES,
        DEFAULT_NUM_WARPS,
        _CX,
        _CY,
        _CZ,
        _Q,
        _Q_PAD,
        _W,
        make_lattice_tensors,
    )
except ImportError:
    # Deployed path: modules live inside the ``tensorlbm`` package.
    from tensorlbm.triton_fused import (
        DEFAULT_BLOCK_X,
        DEFAULT_BLOCK_Y,
        DEFAULT_NUM_STAGES,
        DEFAULT_NUM_WARPS,
        _CX,
        _CY,
        _CZ,
        _Q,
        _Q_PAD,
        _W,
        make_lattice_tensors,
    )


__all__ = [
    "triton_fused_obstacle_les",
    "triton_fused_obstacle_xfar_les",
    "triton_obstacle_force_reduction",
    "apply_far_field_bc_6face",
    "apply_mass_correction",
    "apply_inflow_zou_he",
    "apply_outflow_zero_gradient",
    "create_suboff_obstacle_torch",
]


# D3Q19 opposite-direction index table.  Import from the canonical
# ``tensorlbm.d3q19.OPPOSITE`` to avoid drift.  OPPOSITE[q] gives the
# index whose lattice vector is -c_q; the naive 18-q formula is WRONG
# for this lattice ordering (verified by direct comparison).
try:
    from tensorlbm.d3q19 import OPPOSITE as _D3Q19_OPPOSITE  # type: ignore
except ImportError:
    # Fallback for pre-deployment where d3q19 isn't importable.
    _D3Q19_OPPOSITE = torch.tensor(
        [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17],
        dtype=torch.int64,
    )
_OPPOSITE = _D3Q19_OPPOSITE.to(torch.int32)
OPPOSITE_PY: list[int] = list(_OPPOSITE.tolist())
_CX_T = torch.tensor(_CX, dtype=torch.float32)
_CY_T = torch.tensor(_CY, dtype=torch.float32)
_CZ_T = torch.tensor(_CZ, dtype=torch.float32)
_W_T = torch.tensor(_W, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Phase 2 — collision-family dispatch (BGK / CM / CUMULANT)
# ---------------------------------------------------------------------------
# The fused kernel can now apply any of three collision families inside the
# stream+BB+BC+force fusion.  KBC stays on PyTorch (entropic bisection loop
# doesn't map to a Triton kernel).  All three families share the same stream,
# bounce-back, macroscopic, and far-field BC scaffolding; only the collide
# stage differs and is selected by the ``COLLISION: tl.constexpr`` tag.

# Hermite projection factor: 1 / (2 * cs^4) with cs^2 = 1/3 → 9/2.
_HERMITE_FACTOR: float = 9.0 / 2.0

# D3Q19 moment transform matrices and the Hermite polynomial table for
# CUMULANT reconstruction.  Imported lazily so that the kernel module
# remains importable on hosts without the tensorlbm package (the pre-deploy
# path is the production one — these tables only resolve when the kernel is
# invoked from inside the runner).
_CM_TABLES: dict[str, np.ndarray] | None = None
_CUMULANT_HERMITE: np.ndarray | None = None
_CM_SHIFT_X: tuple[tuple[int, int, int], ...] | None = None
_CM_SHIFT_Y: tuple[tuple[int, int, int], ...] | None = None
_CM_SHIFT_Z: tuple[tuple[int, int, int], ...] | None = None
_CM_ORDER_BOUNDS: dict[str, tuple[int, int]] | None = None


def _ensure_collision_tables() -> None:
    """Resolve the CM/CUMULANT constexpr tables on first use.

    Reads from ``tensorlbm.cascaded_collision`` and ``tensorlbm.cumulant``.
    Module-level cache so the import + numpy conversion happens once.
    """
    global _CM_TABLES, _CUMULANT_HERMITE
    global _CM_SHIFT_X, _CM_SHIFT_Y, _CM_SHIFT_Z, _CM_ORDER_BOUNDS
    if _CM_TABLES is not None:
        return

    from tensorlbm.cascaded_collision import (  # type: ignore
        _M19_DATA,
        _M19_INV_DATA,
        _D3Q19_SHIFT_GROUPS,
        _D3Q19_ORDER_BOUNDS,
    )

    M = np.asarray(_M19_DATA, dtype=np.float32)
    M_inv = np.asarray(_M19_INV_DATA, dtype=np.float32)
    assert M.shape == (19, 19) and M_inv.shape == (19, 19)
    _CM_TABLES = {"M": M, "M_inv": M_inv}

    # Shift groups are returned as a 3-tuple (x, y, z); each entry is itself
    # a tuple of (i0, i1, i2) index triplets.
    sx, sy, sz = _D3Q19_SHIFT_GROUPS
    _CM_SHIFT_X = tuple(tuple(int(c) for c in trip) for trip in sx)
    _CM_SHIFT_Y = tuple(tuple(int(c) for c in trip) for trip in sy)
    _CM_SHIFT_Z = tuple(tuple(int(c) for c in trip) for trip in sz)
    _CM_ORDER_BOUNDS = dict(_D3Q19_ORDER_BOUNDS)

    # Hermite polynomial table for CUMULANT: row = population index,
    # cols = (h_xx, h_yy, h_zz, h_xy, h_xz, h_yz) where
    #   h_xx = cx^2 - cs^2, h_yy = cy^2 - cs^2, h_zz = cz^2 - cs^2,
    #   h_xy = cx*cy,        h_xz = cx*cz,        h_yz = cy*cz,
    # with cs^2 = 1/3.
    cx = np.asarray(_CX, dtype=np.float32)
    cy = np.asarray(_CY, dtype=np.float32)
    cz = np.asarray(_CZ, dtype=np.float32)
    cs2 = np.float32(1.0 / 3.0)
    hermite = np.stack(
        [
            cx * cx - cs2,
            cy * cy - cs2,
            cz * cz - cs2,
            cx * cy,
            cx * cz,
            cy * cz,
        ],
        axis=1,
    )  # (19, 6)
    _CUMULANT_HERMITE = hermite.astype(np.float32)


# Collision-family name → Triton constexpr tag.
_COLLISION_BGK = 0
_COLLISION_CM = 1
_COLLISION_CUMULANT = 2


def _dispatch_collision(collision: str) -> int:
    """Map a collision family name to its Triton ``COLLISION`` tag.

    Returns one of ``{_COLLISION_BGK, _COLLISION_CM, _COLLISION_CUMULANT}``.
    Raises ``ValueError`` for unsupported names (KBC and unknown).
    """
    c = collision.upper()
    if c == "BGK":
        return _COLLISION_BGK
    if c == "CM":
        return _COLLISION_CM
    if c == "CUMULANT":
        return _COLLISION_CUMULANT
    raise ValueError(
        f"unsupported collision {collision!r} for Triton kernel; "
        f"valid: BGK, CM, CUMULANT (KBC stays on PyTorch)"
    )


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@triton.jit
def _fused_collide_stream_obstacle_les_kernel(
    f_ptr, fnew_ptr,
    obstacle_ptr,
    opp_ptr,
    cxi_ptr, cyi_ptr, czi_ptr,
    cxf_ptr, cyf_ptr, czf_ptr, w_ptr,
    nu_lb,
    Cs_delta_sq,
    nz, ny, nx,
    stride_q, stride_z, stride_y, stride_x,
    Q_PAD: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """Pull-stream + BGK collide + wet-node bounce-back + LES Smagorinsky.

    Wall handling: at a fluid cell, when the source cell of population
    ``q`` is an obstacle, replace the loaded value with the population
    going the *opposite* direction at the same fluid cell. The opposite
    index is given by the precomputed ``OPPOSITE[q]`` table (matching
    ``tensorlbm.d3q19.OPPOSITE``); the naive ``18 - q`` formula does
    NOT produce opposites for this lattice ordering.

    LES: replaces the constant BGK rate ``tau_inv`` with a per-cell
    effective rate that includes the Smagorinsky eddy viscosity.
    """
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)
    pid_z = tl.program_id(2)

    offs_y = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    offs_x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    offs_q = tl.arange(0, Q_PAD)

    mask_q = offs_q < 19
    spatial_mask = (offs_y < ny)[:, None] & (offs_x < nx)[None, :]
    rw_mask = mask_q[:, None, None] & spatial_mask[None, :, :]

    # Compute source coords with periodic wrap.
    cx_i = tl.load(cxi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]
    cy_i = tl.load(cyi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]
    cz_i = tl.load(czi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]

    src_z = pid_z - cz_i
    src_z = tl.where(src_z < 0, src_z + nz, tl.where(src_z >= nz, src_z - nz, src_z))
    src_y = offs_y[None, :, None] - cy_i
    src_y = tl.where(src_y < 0, src_y + ny, tl.where(src_y >= ny, src_y - ny, src_y))
    src_x = offs_x[None, None, :] - cx_i
    src_x = tl.where(src_x < 0, src_x + nx, tl.where(src_x >= nx, src_x - nx, src_x))

    src_offs = (offs_q.to(tl.int64)[:, None, None] * stride_q
                + src_z.to(tl.int64) * stride_z
                + src_y.to(tl.int64) * stride_y
                + src_x.to(tl.int64) * stride_x)
    f_in = tl.load(f_ptr + src_offs, mask=rw_mask, other=0.0)

    # Load obstacle at source cell (does src belong to the wall?).
    # src_obst_offs is (Q_PAD, BY, BX) — one source cell per (q, y, x).
    # Mask must match: replicate spatial mask along the q-axis.
    src_obst_offs = (src_z.to(tl.int64) * stride_z
                     + src_y.to(tl.int64) * stride_y
                     + src_x.to(tl.int64) * stride_x)
    src_obst_mask = mask_q[:, None, None] & spatial_mask[None, :, :]
    src_obst = tl.load(obstacle_ptr + src_obst_offs,
                       mask=src_obst_mask, other=0)
    src_is_wall = src_obst > 0  # (Q_PAD, BY, BX)

    # Load f at OWN cell at OPPOSITE direction for bounce-back.
    # f_own_opp[q] = f_pre[OPPOSITE[q], own_cell]
    opp_q = tl.load(opp_ptr + offs_q, mask=mask_q, other=0)
    rev_offs = (opp_q.to(tl.int64)[:, None, None] * stride_q
                + pid_z.to(tl.int64) * stride_z
                + offs_y.to(tl.int64)[None, :, None] * stride_y
                + offs_x.to(tl.int64)[None, None, :] * stride_x)
    f_own_opp = tl.load(f_ptr + rev_offs, mask=rw_mask, other=0.0)

    # Wet-node bounce-back: if src is wall, use f_own_opp (the population
    # going the opposite direction at the fluid cell).  Otherwise normal.
    # src_is_wall is (Q, BY, BX); f_own_opp and f_in are (Q, BY, BX).
    f_eff = tl.where(src_is_wall, f_own_opp, f_in)

    # Macroscopic variables and equilibrium.
    cx_b = tl.load(cxf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    cy_b = tl.load(cyf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    cz_b = tl.load(czf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    w_b = tl.load(w_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]

    rho = tl.sum(f_eff, axis=0)
    rho_safe = tl.where(rho > 1e-12, rho, 1e-12)
    ux = tl.sum(cx_b * f_eff, axis=0) / rho_safe
    uy = tl.sum(cy_b * f_eff, axis=0) / rho_safe
    uz = tl.sum(cz_b * f_eff, axis=0) / rho_safe
    usq = ux * ux + uy * uy + uz * uz

    cu = cx_b * ux[None, :, :] + cy_b * uy[None, :, :] + cz_b * uz[None, :, :]
    feq = (rho_safe[None, :, :] * w_b
           * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None, :, :]))

    # ----- LES Smagorinsky -----
    # fneq = f_eff - feq
    fneq = f_eff - feq
    # grad_u_ij = -1/(2 τ_mol) * Σ_q fneq_q * c_qi * c_qj / ρ
    tau_mol = 3.0 * nu_lb + 0.5
    prefactor = -0.5 / tau_mol / rho_safe  # (BLOCK_Y, BLOCK_X)

    g00 = prefactor * tl.sum(fneq * (cx_b * cx_b), axis=0)
    g11 = prefactor * tl.sum(fneq * (cy_b * cy_b), axis=0)
    g22 = prefactor * tl.sum(fneq * (cz_b * cz_b), axis=0)
    g01 = prefactor * tl.sum(fneq * (cx_b * cy_b), axis=0)
    g02 = prefactor * tl.sum(fneq * (cx_b * cz_b), axis=0)
    g12 = prefactor * tl.sum(fneq * (cy_b * cz_b), axis=0)

    # |S| = sqrt(2 * Σ_ij S_ij²); S_ij is symmetric so S_ij = grad_u_ij here.
    S_sq = (2.0 * (g00 * g00 + g11 * g11 + g22 * g22)
            + 4.0 * (g01 * g01 + g02 * g02 + g12 * g12))
    S_mag = tl.sqrt(S_sq + 1e-20)

    nu_t = Cs_delta_sq * S_mag
    tau_eff = 3.0 * (nu_lb + nu_t) + 0.5
    omega_eff = 1.0 / tau_eff

    # BGK with effective omega.
    f_post = f_eff - omega_eff[None, :, :] * (f_eff - feq)

    # Store.  We write to all cells including wall cells (no
    # wall-skip), matching the production periodic kernel pattern.
    # Wall cells' f_post is never read by neighbours — the bounce-back
    # at fluid neighbours already populated what those directions
    # should expose — so the wasted writes are harmless.
    # Single-expression offset pattern (matches prod triton_fused.py).
    dst_offs = (offs_q.to(tl.int64)[:, None, None] * stride_q
                + pid_z.to(tl.int64) * stride_z
                + offs_y.to(tl.int64)[None, :, None] * stride_y
                + offs_x.to(tl.int64)[None, None, :] * stride_x)
    tl.store(fnew_ptr + dst_offs, f_post, mask=rw_mask)


# ---------------------------------------------------------------------------
# X-streamwise fused kernel (matches production SUBOFF runner's convention)
# ---------------------------------------------------------------------------
#
# Differs from ``_fused_collide_stream_obstacle_les_kernel`` only in the
# axis convention: ``pid_x`` iterates over single x-planes (the streamwise
# axis in production), with y and z tiled.  Periodic wrap is preserved on
# all three axes; the production runner overwrites the 6 boundary faces
# in a separate ``apply_far_field_bc_6face`` post-op, exactly matching
# ``boundaries3d.far_field_bc_3d`` semantics.
#
# Lattice vector signs match ``d3q19.C`` (verified by sphere drag unit test).
# ---------------------------------------------------------------------------

@triton.jit
def _fused_collide_stream_obstacle_xfar_les_kernel(
    f_ptr, fnew_ptr,
    obstacle_ptr,
    opp_ptr,
    cxi_ptr, cyi_ptr, czi_ptr,
    cxf_ptr, cyf_ptr, czf_ptr, w_ptr,
    nu_lb,
    Cs_delta_sq,
    nz, ny, nx,
    stride_q, stride_z, stride_y, stride_x,
    tau_eff_ptr,
    tau_eff_stride_z, tau_eff_stride_y, tau_eff_stride_x,
    Q_PAD: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    BLOCK_Z: tl.constexpr,
    COLLISION: tl.constexpr,
    M_CM: tl.constexpr,
    M_INV_CM: tl.constexpr,
    HERMITE_CUM: tl.constexpr,
    SHIFT_X: tl.constexpr,
    SHIFT_Y: tl.constexpr,
    SHIFT_Z: tl.constexpr,
    USE_EXTERNAL_TAU: tl.constexpr,
):
    """Pull-stream + collide (BGK/CM/CUMULANT) + wet-node bounce-back.

    x-streamwise: ``pid_x = tl.program_id(2)`` indexes single x-planes.
    y and z are tiled with ``BLOCK_Y x BLOCK_Z``.  Periodic wrap on all
    three axes; BC writes happen in a post-op (``apply_far_field_bc_6face``).

    ``COLLISION`` selects the collide sub-branch:
    * 0 = BGK — single-relaxation-time with internal Smagorinsky LES tau-eff
    * 1 = CM — cascaded central moments (D3Q19 moment matrix M/M_inv +
      1-D binomial shifts + per-mode relaxation)
    * 2 = CUMULANT — non-equilibrium stress + Hermite polynomial
      reconstruction
    """
    pid_z = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(2)

    offs_z = pid_z * BLOCK_Z + tl.arange(0, BLOCK_Z)
    offs_y = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    offs_q = tl.arange(0, Q_PAD)

    mask_q = offs_q < 19
    spatial_mask = (offs_z < nz)[:, None] & (offs_y < ny)[None, :]
    rw_mask = mask_q[:, None, None] & spatial_mask[None, :, :]

    # Pull-stream source coords with periodic wrap.
    cx_i = tl.load(cxi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]
    cy_i = tl.load(cyi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]
    cz_i = tl.load(czi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]

    src_x = pid_x - cx_i
    src_x = tl.where(src_x < 0, src_x + nx, tl.where(src_x >= nx, src_x - nx, src_x))
    src_y = offs_y[None, :, None] - cy_i
    src_y = tl.where(src_y < 0, src_y + ny, tl.where(src_y >= ny, src_y - ny, src_y))
    src_z = offs_z[None, None, :] - cz_i  # NB: offs_z varies along axis 0 of (Z,Y) tile
    src_z = tl.where(src_z < 0, src_z + nz, tl.where(src_z >= nz, src_z - nz, src_z))

    # Layout: f is (Q, nz, ny, nx) — axis 1 = nz, axis 2 = ny, axis 3 = nx.
    # Tile is (BLOCK_Z, BLOCK_Y) over (nz, ny); src_x is per-pid (scalar).
    src_offs = (offs_q.to(tl.int64)[:, None, None] * stride_q
                + src_z.to(tl.int64) * stride_z
                + src_y.to(tl.int64) * stride_y
                + src_x.to(tl.int64) * stride_x)
    f_in = tl.load(f_ptr + src_offs, mask=rw_mask, other=0.0)

    # Wall mask at source cell.
    src_obst_offs = (src_z.to(tl.int64) * stride_z
                     + src_y.to(tl.int64) * stride_y
                     + src_x.to(tl.int64) * stride_x)
    src_obst_mask = mask_q[:, None, None] & spatial_mask[None, :, :]
    src_obst = tl.load(obstacle_ptr + src_obst_offs,
                       mask=src_obst_mask, other=0)
    src_is_wall = src_obst > 0

    # Wet-node bounce-back: own cell at OPPOSITE direction.
    opp_q = tl.load(opp_ptr + offs_q, mask=mask_q, other=0)
    rev_offs = (opp_q.to(tl.int64)[:, None, None] * stride_q
                + offs_z.to(tl.int64)[None, None, :] * stride_z
                + offs_y.to(tl.int64)[None, :, None] * stride_y
                + pid_x.to(tl.int64) * stride_x)
    f_own_opp = tl.load(f_ptr + rev_offs, mask=rw_mask, other=0.0)

    f_eff = tl.where(src_is_wall, f_own_opp, f_in)

    # Macroscopic + equilibrium (POST-stream moments → valid stream-then-collide).
    cx_b = tl.load(cxf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    cy_b = tl.load(cyf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    cz_b = tl.load(czf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    w_b = tl.load(w_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]

    rho = tl.sum(f_eff, axis=0)
    rho_safe = tl.where(rho > 1e-12, rho, 1e-12)
    ux = tl.sum(cx_b * f_eff, axis=0) / rho_safe
    uy = tl.sum(cy_b * f_eff, axis=0) / rho_safe
    uz = tl.sum(cz_b * f_eff, axis=0) / rho_safe
    usq = ux * ux + uy * uy + uz * uz
    cu = cx_b * ux[None, :, :] + cy_b * uy[None, :, :] + cz_b * uz[None, :, :]
    feq = (rho_safe[None, :, :] * w_b
           * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None, :, :]))

    # === Per-cell effective tau (Phase 3 — external SGS coupling) ===
    # When USE_EXTERNAL_TAU is True, the kernel loads omega_eff from the
    # ``tau_eff_ptr`` tensor (shape [nz, ny, nx]) and skips the internal
    # Smagorinsky |S| computation.  Default fallback for masked cells is
    # the molecular ``tau_mol = 3*nu_lb + 0.5`` so wet-node BCs stay sane.
    tau_mol = 3.0 * nu_lb + 0.5
    if USE_EXTERNAL_TAU:
        # tau_eff_ptr is unused when USE_EXTERNAL_TAU is False — pass a
        # placeholder pointer in Python (e.g., obstacle_ptr) to satisfy
        # Triton's signature check.
        tau_offs = (offs_z[None, None, :] * tau_eff_stride_z
                    + offs_y[None, :, None] * tau_eff_stride_y
                    + pid_x * tau_eff_stride_x)
        tau_eff = tl.load(
            tau_eff_ptr + tau_offs,
            mask=spatial_mask[None, :, :],
            other=tau_mol,
        )
        omega_eff = 1.0 / tau_eff
    else:
        omega_eff = None  # sentinel; BGK computes its own below

    # === Collision-family dispatch (Phase 2) ===
    # All three families operate on the streamed, bounce-back'd
    # distribution ``f_eff``.  The macroscopic moments above and the
    # equilibrium ``feq`` are reused for BGK; CM and CUMULANT compute
    # their own collide-specific intermediates.  Stream + BB scaffolding
    # above is family-agnostic.
    if COLLISION == 0:
        # ---- BGK + Smagorinsky LES (legacy single-relaxation-time) ----
        if USE_EXTERNAL_TAU:
            # External tau_eff tensor supplied by the runner (WALE / Vreman /
            # externally-supplied Smagorinsky).  Skip the internal |S|
            # computation entirely; omega_eff was loaded above with shape
            # ``(1, BZ, BY)`` so it broadcasts cleanly against
            # ``f_eff - feq`` of shape ``(Q_PAD, BZ, BY)`` — no extra
            # leading dim needed (unlike ``omega_eff_local`` which is
            # 2-D and requires ``[None, :, :]`` to broadcast).
            f_post = f_eff - omega_eff * (f_eff - feq)
        else:
            # Internal Smagorinsky LES (legacy default).
            fneq = f_eff - feq
            prefactor = -0.5 / tau_mol / rho_safe
            g00 = prefactor * tl.sum(fneq * (cx_b * cx_b), axis=0)
            g11 = prefactor * tl.sum(fneq * (cy_b * cy_b), axis=0)
            g22 = prefactor * tl.sum(fneq * (cz_b * cz_b), axis=0)
            g01 = prefactor * tl.sum(fneq * (cx_b * cy_b), axis=0)
            g02 = prefactor * tl.sum(fneq * (cx_b * cz_b), axis=0)
            g12 = prefactor * tl.sum(fneq * (cy_b * cz_b), axis=0)
            S_sq = (2.0 * (g00 * g00 + g11 * g11 + g22 * g22)
                    + 4.0 * (g01 * g01 + g02 * g02 + g12 * g12))
            S_mag = tl.sqrt(S_sq + 1e-20)
            nu_t = Cs_delta_sq * S_mag
            tau_eff = 3.0 * (nu_lb + nu_t) + 0.5
            omega_eff_local = 1.0 / tau_eff
            f_post = f_eff - omega_eff_local[None, :, :] * (f_eff - feq)
    elif COLLISION == 1:
        # ---- CM (cascaded central moments, Premnath-Banerjee 2009) ----
        # Pipeline: populations → raw moments → central moments →
        # relaxation (trace/deviatoric split at 2nd order, uniform
        # rates at 3rd/4th) → raw moments → populations.
        #
        # Default relaxation rates: s_bulk=s_3=s_4=1 (only the shear
        # rate ω = 1/τ at 2nd order is the active knob).  This matches
        # ``collide_cascaded_d3q19`` with s_bulk=s_3=s_4=1.
        #
        # Uses M_CM (19×19 moment matrix) and M_INV_CM (inverse) as
        # ``tl.constexpr`` flat tuples; SHIFT_X/Y/Z are the 1-D
        # binomial-shift triplet tables (5 triplets per axis for D3Q19).
        fneq = f_eff - feq

        # === Build M, M_inv as (19, 19) compile-time constant tensors ===
        # Triton has no direct "constant tensor from constexpr tuple"
        # idiom, so we assemble via nested ``tl.static_range`` with
        # ``tl.where``.  The compiler folds these into a single
        # constant.  M[i, j] = M_CM[i*19 + j] (flat layout).
        #
        # ``tl.arange`` requires a power-of-2 range; use 32 (Q_PAD)
        # and mask to 19 with the constexpr table itself being 0
        # outside j < 19.
        idx_mq = tl.arange(0, 32)
        idx_mm = tl.arange(0, 32)
        M_2d = tl.zeros((32, 32), dtype=tl.float32)
        M_inv_2d = tl.zeros((32, 32), dtype=tl.float32)
        for i in tl.static_range(0, 19):
            for j in tl.static_range(0, 19):
                mask_ij = (idx_mq[:, None] == i) & (idx_mm[None, :] == j)
                M_2d = tl.where(mask_ij, M_CM[i * 19 + j], M_2d)
                M_inv_2d = tl.where(mask_ij, M_INV_CM[i * 19 + j], M_inv_2d)

        # === Raw moments of f_neq ===
        # m[i, z, y] = Σ_j M[i, j] * fneq[j, z, y].  fneq is zero for
        # j ≥ 19 (masked at load), so we don't need an explicit mask.
        m_neq = tl.sum(M_2d[:, :, None, None] * fneq[None, :, :, :], axis=1)

        # === 1-D forward shifts: x, then y, then z ===
        # For each triplet (i0, i1, i2):
        #   m'[i0] = m[i0]
        #   m'[i1] = m[i1] − u·m[i0]
        #   m'[i2] = m[i2] − 2u·m[i1] + u²·m[i0]
        # ``idx_row`` is 32-wide (power of 2 required by ``tl.arange``);
        # m_neq rows 19..31 are zero so masking is implicit.
        idx_row = tl.arange(0, 32)

        for g in tl.static_range(0, 5):
            i0 = SHIFT_X[g * 3 + 0]
            i1 = SHIFT_X[g * 3 + 1]
            i2 = SHIFT_X[g * 3 + 2]
            m_i0 = tl.sum(m_neq * (idx_row == i0).to(tl.float32)[:, None, None], axis=0)
            m_i1 = tl.sum(m_neq * (idx_row == i1).to(tl.float32)[:, None, None], axis=0)
            m_i2 = tl.sum(m_neq * (idx_row == i2).to(tl.float32)[:, None, None], axis=0)
            m_i1_new = m_i1 - ux * m_i0
            m_i2_new = m_i2 - 2.0 * ux * m_i1 + ux * ux * m_i0
            m_neq = tl.where((idx_row == i1)[:, None, None], m_i1_new[None, :, :], m_neq)
            m_neq = tl.where((idx_row == i2)[:, None, None], m_i2_new[None, :, :], m_neq)
        for g in tl.static_range(0, 5):
            i0 = SHIFT_Y[g * 3 + 0]
            i1 = SHIFT_Y[g * 3 + 1]
            i2 = SHIFT_Y[g * 3 + 2]
            m_i0 = tl.sum(m_neq * (idx_row == i0).to(tl.float32)[:, None, None], axis=0)
            m_i1 = tl.sum(m_neq * (idx_row == i1).to(tl.float32)[:, None, None], axis=0)
            m_i2 = tl.sum(m_neq * (idx_row == i2).to(tl.float32)[:, None, None], axis=0)
            m_i1_new = m_i1 - uy * m_i0
            m_i2_new = m_i2 - 2.0 * uy * m_i1 + uy * uy * m_i0
            m_neq = tl.where((idx_row == i1)[:, None, None], m_i1_new[None, :, :], m_neq)
            m_neq = tl.where((idx_row == i2)[:, None, None], m_i2_new[None, :, :], m_neq)
        for g in tl.static_range(0, 5):
            i0 = SHIFT_Z[g * 3 + 0]
            i1 = SHIFT_Z[g * 3 + 1]
            i2 = SHIFT_Z[g * 3 + 2]
            m_i0 = tl.sum(m_neq * (idx_row == i0).to(tl.float32)[:, None, None], axis=0)
            m_i1 = tl.sum(m_neq * (idx_row == i1).to(tl.float32)[:, None, None], axis=0)
            m_i2 = tl.sum(m_neq * (idx_row == i2).to(tl.float32)[:, None, None], axis=0)
            m_i1_new = m_i1 - uz * m_i0
            m_i2_new = m_i2 - 2.0 * uz * m_i1 + uz * uz * m_i0
            m_neq = tl.where((idx_row == i1)[:, None, None], m_i1_new[None, :, :], m_neq)
            m_neq = tl.where((idx_row == i2)[:, None, None], m_i2_new[None, :, :], m_neq)

        # === Relaxation (defaults: s_bulk=ω, s_3=1, s_4=1, only ω active) ===
        if USE_EXTERNAL_TAU:
            omega = omega_eff  # per-cell external SGS coupling
        else:
            omega = 1.0 / (3.0 * nu_lb + 0.5)

        # 2nd-order trace/deviatoric split (indices 4-6)
        m4 = tl.sum(m_neq * (idx_row == 4).to(tl.float32)[:, None, None], axis=0)
        m5 = tl.sum(m_neq * (idx_row == 5).to(tl.float32)[:, None, None], axis=0)
        m6 = tl.sum(m_neq * (idx_row == 6).to(tl.float32)[:, None, None], axis=0)
        trace = m4 + m5 + m6
        m4_new = (1.0 - omega) * (m4 - trace / 3.0)
        m5_new = (1.0 - omega) * (m5 - trace / 3.0)
        m6_new = (1.0 - omega) * (m6 - trace / 3.0)
        # NB: ``m4_new`` already broadcasts against ``(32, 1, 1)`` from
        # ``(idx_row == 4)[:, None, None]`` — adding ``[None, :, :]`` would
        # introduce a spurious 4th dim that breaks ``tl.store`` downstream.
        m_neq = tl.where((idx_row == 4)[:, None, None], m4_new, m_neq)
        m_neq = tl.where((idx_row == 5)[:, None, None], m5_new, m_neq)
        m_neq = tl.where((idx_row == 6)[:, None, None], m6_new, m_neq)

        # 2nd-order shear (indices 7-9): (1 − ω) · m_i
        for i in tl.static_range(7, 10):
            m_i = tl.sum(m_neq * (idx_row == i).to(tl.float32)[:, None, None], axis=0)
            m_i_new = (1.0 - omega) * m_i
            m_neq = tl.where((idx_row == i)[:, None, None], m_i_new, m_neq)

        # 3rd order (indices 10-15): zero (s_3 = 1)
        for i in tl.static_range(10, 16):
            m_neq = tl.where((idx_row == i)[:, None, None], 0.0, m_neq)

        # 4th order (indices 16-18): zero (s_4 = 1)
        for i in tl.static_range(16, 19):
            m_neq = tl.where((idx_row == i)[:, None, None], 0.0, m_neq)

        # === 1-D inverse shifts: z, then y, then x ===
        # Same triplets, but with +u (opposite sign of forward shift).
        for g in tl.static_range(0, 5):
            i0 = SHIFT_Z[g * 3 + 0]
            i1 = SHIFT_Z[g * 3 + 1]
            i2 = SHIFT_Z[g * 3 + 2]
            m_i0 = tl.sum(m_neq * (idx_row == i0).to(tl.float32)[:, None, None], axis=0)
            m_i1 = tl.sum(m_neq * (idx_row == i1).to(tl.float32)[:, None, None], axis=0)
            m_i2 = tl.sum(m_neq * (idx_row == i2).to(tl.float32)[:, None, None], axis=0)
            m_i1_new = m_i1 + uz * m_i0
            m_i2_new = m_i2 + 2.0 * uz * m_i1 + uz * uz * m_i0
            m_neq = tl.where((idx_row == i1)[:, None, None], m_i1_new[None, :, :], m_neq)
            m_neq = tl.where((idx_row == i2)[:, None, None], m_i2_new[None, :, :], m_neq)
        for g in tl.static_range(0, 5):
            i0 = SHIFT_Y[g * 3 + 0]
            i1 = SHIFT_Y[g * 3 + 1]
            i2 = SHIFT_Y[g * 3 + 2]
            m_i0 = tl.sum(m_neq * (idx_row == i0).to(tl.float32)[:, None, None], axis=0)
            m_i1 = tl.sum(m_neq * (idx_row == i1).to(tl.float32)[:, None, None], axis=0)
            m_i2 = tl.sum(m_neq * (idx_row == i2).to(tl.float32)[:, None, None], axis=0)
            m_i1_new = m_i1 + uy * m_i0
            m_i2_new = m_i2 + 2.0 * uy * m_i1 + uy * uy * m_i0
            m_neq = tl.where((idx_row == i1)[:, None, None], m_i1_new[None, :, :], m_neq)
            m_neq = tl.where((idx_row == i2)[:, None, None], m_i2_new[None, :, :], m_neq)
        for g in tl.static_range(0, 5):
            i0 = SHIFT_X[g * 3 + 0]
            i1 = SHIFT_X[g * 3 + 1]
            i2 = SHIFT_X[g * 3 + 2]
            m_i0 = tl.sum(m_neq * (idx_row == i0).to(tl.float32)[:, None, None], axis=0)
            m_i1 = tl.sum(m_neq * (idx_row == i1).to(tl.float32)[:, None, None], axis=0)
            m_i2 = tl.sum(m_neq * (idx_row == i2).to(tl.float32)[:, None, None], axis=0)
            m_i1_new = m_i1 + ux * m_i0
            m_i2_new = m_i2 + 2.0 * ux * m_i1 + ux * ux * m_i0
            m_neq = tl.where((idx_row == i1)[:, None, None], m_i1_new[None, :, :], m_neq)
            m_neq = tl.where((idx_row == i2)[:, None, None], m_i2_new[None, :, :], m_neq)

        # === Reconstruct populations ===
        # fneq_star[i, z, y] = Σ_j M_inv[i, j] · m_post[j, z, y]
        fneq_star = tl.sum(M_inv_2d[:, :, None, None] * m_neq[None, :, :, :], axis=1)

        f_post = feq + fneq_star
    elif COLLISION == 2:
        # ---- CUMULANT (D3Q19, Geier et al. 2015) ----
        # 1. fneq = f_eff - feq
        # 2. Strain-rate stress tensor: pi_αβ = Σ_q c_{q,α} c_{q,β} fneq_q
        #    (masked to active Q=19 directions so padded slots don't leak).
        # 3. Relax: pi_αβ_s = pi_αβ - omega·pi_αβ - (omega_b - omega)·trace/3·δαβ
        # 4. Hermite reconstruction of regularised non-equilibrium.
        # 5. Higher-order residual relaxed at omega_even (=1 → zero).
        fneq = f_eff - feq
        pi_xx = tl.sum(cx_b * cx_b * fneq * mask_q[:, None, None].to(tl.float32), axis=0)
        pi_yy = tl.sum(cy_b * cy_b * fneq * mask_q[:, None, None].to(tl.float32), axis=0)
        pi_zz = tl.sum(cz_b * cz_b * fneq * mask_q[:, None, None].to(tl.float32), axis=0)
        pi_xy = tl.sum(cx_b * cy_b * fneq * mask_q[:, None, None].to(tl.float32), axis=0)
        pi_xz = tl.sum(cx_b * cz_b * fneq * mask_q[:, None, None].to(tl.float32), axis=0)
        pi_yz = tl.sum(cy_b * cz_b * fneq * mask_q[:, None, None].to(tl.float32), axis=0)
        # Scalar relaxation rate from molecular nu_lb.  Phase 3 replaces this
        # with an external per-cell tau_eff tensor for WALE / Vreman coupling.
        if USE_EXTERNAL_TAU:
            omega = omega_eff  # per-cell external SGS coupling
        else:
            omega = 1.0 / (3.0 * nu_lb + 0.5)
        omega_b = 1.0
        omega_even = 1.0
        trace = pi_xx + pi_yy + pi_zz
        delta_b = (omega_b - omega) * trace / 3.0
        pi_xx_s = pi_xx - omega * pi_xx - delta_b
        pi_yy_s = pi_yy - omega * pi_yy - delta_b
        pi_zz_s = pi_zz - omega * pi_zz - delta_b
        pi_xy_s = pi_xy - omega * pi_xy
        pi_xz_s = pi_xz - omega * pi_xz
        pi_yz_s = pi_yz - omega * pi_yz
        # Hermite polynomials H_αβ = cx_α·cx_β - cs²·δ_αβ
        cs2 = 1.0 / 3.0
        h_xx = cx_b * cx_b - cs2
        h_yy = cy_b * cy_b - cs2
        h_zz = cz_b * cz_b - cs2
        h_xy = cx_b * cy_b
        h_xz = cx_b * cz_b
        h_yz = cy_b * cz_b
        # Regularised non-equilibrium from relaxed Π
        fneq_reg = 4.5 * w_b * (
            h_xx * pi_xx_s + h_yy * pi_yy_s + h_zz * pi_zz_s
            + 2.0 * h_xy * pi_xy_s + 2.0 * h_xz * pi_xz_s + 2.0 * h_yz * pi_yz_s
        )
        # Higher-order residual: fneq minus the 2nd-order-Hermite projection
        # of the UNRELAXED stress (so the residual carries the >2nd-order
        # modes).
        fneq_ho_unrel = 4.5 * w_b * (
            h_xx * pi_xx + h_yy * pi_yy + h_zz * pi_zz
            + 2.0 * h_xy * pi_xy + 2.0 * h_xz * pi_xz + 2.0 * h_yz * pi_yz
        )
        fneq_ho = fneq - fneq_ho_unrel
        fneq_ho_s = (1.0 - omega_even) * fneq_ho
        f_post = feq + fneq_reg + fneq_ho_s
    else:
        # Unreachable: validated by ``_dispatch_collision``.
        f_post = f_eff

    dst_offs = (offs_q.to(tl.int64)[:, None, None] * stride_q
                + offs_z.to(tl.int64)[None, None, :] * stride_z
                + offs_y.to(tl.int64)[None, :, None] * stride_y
                + pid_x.to(tl.int64) * stride_x)
    tl.store(fnew_ptr + dst_offs, f_post, mask=rw_mask)


# ---------------------------------------------------------------------------
# V2 — Q=19 buffer + X/Y/Z tiling (BLOCK_X innermost for coalesced loads)
# ---------------------------------------------------------------------------
# Replaces the x-streamwise kernel above.  The Q=32 padded iteration
# wasted 41% of compute lanes (and provided no speedup since the kernel
# was bandwidth-pattern bound, not launch-overhead bound).  Tile shape is
# now ``(Q_PAD=32 internal arange, BLOCK_Z, BLOCK_Y, BLOCK_X)`` with
# BLOCK_X innermost: stride_x=1 in memory, so each warp's 32 threads
# issue a single coalesced 128-byte transaction per q-lane.
#
# Buffer shape is ``(19, nz, ny, nx)`` matching production Q=19 — the 32
# is only the *arange size* for Triton's power-of-2 constraint; the
# actual loads/stores are masked to lanes 0..18.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_v2_kernel_xfar_les(
    f_ptr, fnew_ptr,
    obstacle_ptr,
    opp_ptr,
    cxi_ptr, cyi_ptr, czi_ptr,
    cxf_ptr, cyf_ptr, czf_ptr, w_ptr,
    nu_lb,
    Cs_delta_sq,
    nz, ny, nx,
    stride_q, stride_z, stride_y, stride_x,
    tau_eff_ptr,
    tau_eff_stride_z, tau_eff_stride_y, tau_eff_stride_x,
    fx_buf_ptr, fy_buf_ptr, fz_buf_ptr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    BLOCK_Z: tl.constexpr,
    COLLISION: tl.constexpr,
    M_CM: tl.constexpr,
    M_INV_CM: tl.constexpr,
    HERMITE_CUM: tl.constexpr,
    SHIFT_X: tl.constexpr,
    SHIFT_Y: tl.constexpr,
    SHIFT_Z: tl.constexpr,
    USE_EXTERNAL_TAU: tl.constexpr,
    COMPUTE_FORCE: tl.constexpr,
):
    """Pull-stream + collide (BGK/CM/CUMULANT) + wet-node bounce-back.

    Tile: ``(Q=32 arange, BLOCK_Z, BLOCK_Y, BLOCK_X)`` with BLOCK_X
    innermost (stride_x=1 → coalesced).  Grid: ``(cdiv(nx, BLOCK_X),
    cdiv(ny, BLOCK_Y), cdiv(nz, BLOCK_Z))``.  Public buffer is Q=19;
    lanes 19..31 are masked off via ``mask_q = offs_q < 19``.

    ``COLLISION`` selects the collide sub-branch:
    * 0 = BGK — single-relaxation-time with internal Smagorinsky LES
    * 1 = CM — cascaded central moments (D3Q19 moment matrix + 1-D
      binomial shifts + per-mode relaxation)
    * 2 = CUMULANT — non-equilibrium stress + Hermite reconstruction
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)

    offs_x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    offs_y = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    offs_z = pid_z * BLOCK_Z + tl.arange(0, BLOCK_Z)
    offs_q = tl.arange(0, 32)

    mask_q = offs_q < 19
    spatial_mask = ((offs_z < nz)[:, None, None]
                    & (offs_y < ny)[None, :, None]
                    & (offs_x < nx)[None, None, :])
    rw_mask = mask_q[:, None, None, None] & spatial_mask[None, :, :, :]

    # Lattice constants: (32,) shape with 19 active lanes.
    cx_i = tl.load(cxi_ptr + offs_q, mask=mask_q, other=0)
    cy_i = tl.load(cyi_ptr + offs_q, mask=mask_q, other=0)
    cz_i = tl.load(czi_ptr + offs_q, mask=mask_q, other=0)

    # Source coords per-q per-axis: src_x:(32, BX), src_y:(32, BY), src_z:(32, BZ).
    src_x = offs_x[None, :] - cx_i[:, None]
    src_x = tl.where(src_x < 0, src_x + nx,
                     tl.where(src_x >= nx, src_x - nx, src_x))
    src_y = offs_y[None, :] - cy_i[:, None]
    src_y = tl.where(src_y < 0, src_y + ny,
                     tl.where(src_y >= ny, src_y - ny, src_y))
    src_z = offs_z[None, :] - cz_i[:, None]
    src_z = tl.where(src_z < 0, src_z + nz,
                     tl.where(src_z >= nz, src_z - nz, src_z))

    # 4-D source offsets: (32, BZ, BY, BX).  Inner dim BX has stride_x=1 → coalesced.
    src_offs = (offs_q[:, None, None, None].to(tl.int64) * stride_q
                + src_z[:, :, None, None].to(tl.int64) * stride_z
                + src_y[:, None, :, None].to(tl.int64) * stride_y
                + src_x[:, None, None, :].to(tl.int64) * stride_x)
    f_in = tl.load(f_ptr + src_offs, mask=rw_mask, other=0.0)

    # Wall mask at source cell.  Same addresses as f_in (no Q-multiplier).
    src_obst_offs = (src_z[:, :, None, None].to(tl.int64) * stride_z
                     + src_y[:, None, :, None].to(tl.int64) * stride_y
                     + src_x[:, None, None, :].to(tl.int64) * stride_x)
    src_obst = tl.load(obstacle_ptr + src_obst_offs,
                       mask=spatial_mask[None, :, :, :], other=0)
    src_is_wall = src_obst > 0

    # Own-cell wall mask: BB must only fire when OWN is fluid.  Without
    # this guard, the BB swap happens at interior solid cells too,
    # causing a period-2 oscillation of f at solid cells and a
    # corresponding sign-flip in the Ladd force.
    own_obst_offs = (offs_z[:, None, None].to(tl.int64) * stride_z
                     + offs_y[None, :, None].to(tl.int64) * stride_y
                     + offs_x[None, None, :].to(tl.int64) * stride_x)
    own_obst = tl.load(obstacle_ptr + own_obst_offs,
                       mask=spatial_mask, other=0)
    own_is_wall = (own_obst > 0)[None, :, :, :]
    bb_fires = src_is_wall & (~own_is_wall)

    # Wet-node bounce-back: own cell at OPPOSITE direction.
    opp_q = tl.load(opp_ptr + offs_q, mask=mask_q, other=0)
    rev_offs = (opp_q[:, None, None, None].to(tl.int64) * stride_q
                + offs_z[None, :, None, None].to(tl.int64) * stride_z
                + offs_y[None, None, :, None].to(tl.int64) * stride_y
                + offs_x[None, None, None, :].to(tl.int64) * stride_x)
    f_own_opp = tl.load(f_ptr + rev_offs, mask=rw_mask, other=0.0)

    f_eff = tl.where(bb_fires, f_own_opp, f_in)

    # === Swap-at-solid (matches PyTorch ``bounce_back_cells_3d``) ===
    # PyTorch's BB applies ``f[q, x_solid] = f[opp_q, x_solid]`` to
    # EVERY solid cell after streaming — interior solid cells included.
    # This zeros u at solid cells in the next collide, enforcing
    # no-slip at the fluid-solid interface.  We add it as a SECOND
    # pass over ``f_eff`` AFTER wet-node BB so the order is:
    #   fluid cell, src=solid: wet-node BB fires → use f_own_opp
    #   solid cell (any): swap-at-solid fires → use f_own_opp
    #   fluid cell, src=fluid: untouched (f_in)
    # Critically this is computed BEFORE the Ladd force block, so the
    # force samples the post-stream, pre-BB state via ``f_in`` (the
    # bb swap writes to ``f_eff``, not ``f_in``).
    f_eff = tl.where(own_is_wall, f_own_opp, f_eff)

    # === Macroscopic + equilibrium (post-stream moments). ===
    cx_b = tl.load(cxf_ptr + offs_q, mask=mask_q, other=0.0)
    cy_b = tl.load(cyf_ptr + offs_q, mask=mask_q, other=0.0)
    cz_b = tl.load(czf_ptr + offs_q, mask=mask_q, other=0.0)
    w_b = tl.load(w_ptr + offs_q, mask=mask_q, other=0.0)
    cx_b = cx_b[:, None, None, None]
    cy_b = cy_b[:, None, None, None]
    cz_b = cz_b[:, None, None, None]
    w_b = w_b[:, None, None, None]

    rho = tl.sum(f_eff, axis=0)  # (BZ, BY, BX)
    rho_safe = tl.where(rho > 1e-12, rho, 1e-12)
    ux = tl.sum(cx_b * f_eff, axis=0) / rho_safe
    uy = tl.sum(cy_b * f_eff, axis=0) / rho_safe
    uz = tl.sum(cz_b * f_eff, axis=0) / rho_safe
    usq = ux * ux + uy * uy + uz * uz
    cu = (cx_b * ux[None, :, :, :]
          + cy_b * uy[None, :, :, :]
          + cz_b * uz[None, :, :, :])
    feq = (rho_safe[None, :, :, :] * w_b
           * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None, :, :, :]))

    tau_mol = 3.0 * nu_lb + 0.5

    if USE_EXTERNAL_TAU:
        tau_offs = (offs_z[None, :, None, None] * tau_eff_stride_z
                    + offs_y[None, None, :, None] * tau_eff_stride_y
                    + offs_x[None, None, None, :] * tau_eff_stride_x)
        tau_eff_loaded = tl.load(
            tau_eff_ptr + tau_offs,
            mask=spatial_mask[None, :, :, :],
            other=tau_mol,
        )
        omega_eff = 1.0 / tau_eff_loaded
    else:
        omega_eff = None  # sentinel

    # === Collision-family dispatch ===
    if COLLISION == 0:
        if USE_EXTERNAL_TAU:
            f_post = f_eff - omega_eff * (f_eff - feq)
        else:
            fneq = f_eff - feq
            prefactor = -0.5 / tau_mol / rho_safe
            g00 = prefactor * tl.sum(fneq * (cx_b * cx_b), axis=0)
            g11 = prefactor * tl.sum(fneq * (cy_b * cy_b), axis=0)
            g22 = prefactor * tl.sum(fneq * (cz_b * cz_b), axis=0)
            g01 = prefactor * tl.sum(fneq * (cx_b * cy_b), axis=0)
            g02 = prefactor * tl.sum(fneq * (cx_b * cz_b), axis=0)
            g12 = prefactor * tl.sum(fneq * (cy_b * cz_b), axis=0)
            S_sq = (2.0 * (g00 * g00 + g11 * g11 + g22 * g22)
                    + 4.0 * (g01 * g01 + g02 * g02 + g12 * g12))
            S_mag = tl.sqrt(S_sq + 1e-20)
            nu_t = Cs_delta_sq * S_mag
            tau_eff_l = 3.0 * (nu_lb + nu_t) + 0.5
            omega_eff_local = 1.0 / tau_eff_l
            f_post = f_eff - omega_eff_local[None, :, :, :] * (f_eff - feq)
    elif COLLISION == 1:
        # ---- CM (cascaded central moments) ----
        fneq = f_eff - feq

        idx_mq = tl.arange(0, 32)
        idx_mm = tl.arange(0, 32)
        M_2d = tl.zeros((32, 32), dtype=tl.float32)
        M_inv_2d = tl.zeros((32, 32), dtype=tl.float32)
        for i in tl.static_range(0, 19):
            for j in tl.static_range(0, 19):
                mask_ij = (idx_mq[:, None] == i) & (idx_mm[None, :] == j)
                M_2d = tl.where(mask_ij, M_CM[i * 19 + j], M_2d)
                M_inv_2d = tl.where(mask_ij, M_INV_CM[i * 19 + j], M_inv_2d)

        # m_neq shape: (32, BZ, BY, BX)
        m_neq = tl.sum(
            M_2d[:, :, None, None, None] * fneq[None, :, :, :, :],
            axis=1,
        )

        idx_row = tl.arange(0, 32)

        # Forward shifts: x, y, z.
        for g in tl.static_range(0, 5):
            i0 = SHIFT_X[g * 3 + 0]
            i1 = SHIFT_X[g * 3 + 1]
            i2 = SHIFT_X[g * 3 + 2]
            m_i0 = tl.sum(m_neq * (idx_row == i0).to(tl.float32)[:, None, None, None], axis=0)
            m_i1 = tl.sum(m_neq * (idx_row == i1).to(tl.float32)[:, None, None, None], axis=0)
            m_i2 = tl.sum(m_neq * (idx_row == i2).to(tl.float32)[:, None, None, None], axis=0)
            m_i1_new = m_i1 - ux * m_i0
            m_i2_new = m_i2 - 2.0 * ux * m_i1 + ux * ux * m_i0
            m_neq = tl.where((idx_row == i1)[:, None, None, None], m_i1_new[None, :, :, :], m_neq)
            m_neq = tl.where((idx_row == i2)[:, None, None, None], m_i2_new[None, :, :, :], m_neq)
        for g in tl.static_range(0, 5):
            i0 = SHIFT_Y[g * 3 + 0]
            i1 = SHIFT_Y[g * 3 + 1]
            i2 = SHIFT_Y[g * 3 + 2]
            m_i0 = tl.sum(m_neq * (idx_row == i0).to(tl.float32)[:, None, None, None], axis=0)
            m_i1 = tl.sum(m_neq * (idx_row == i1).to(tl.float32)[:, None, None, None], axis=0)
            m_i2 = tl.sum(m_neq * (idx_row == i2).to(tl.float32)[:, None, None, None], axis=0)
            m_i1_new = m_i1 - uy * m_i0
            m_i2_new = m_i2 - 2.0 * uy * m_i1 + uy * uy * m_i0
            m_neq = tl.where((idx_row == i1)[:, None, None, None], m_i1_new[None, :, :, :], m_neq)
            m_neq = tl.where((idx_row == i2)[:, None, None, None], m_i2_new[None, :, :, :], m_neq)
        for g in tl.static_range(0, 5):
            i0 = SHIFT_Z[g * 3 + 0]
            i1 = SHIFT_Z[g * 3 + 1]
            i2 = SHIFT_Z[g * 3 + 2]
            m_i0 = tl.sum(m_neq * (idx_row == i0).to(tl.float32)[:, None, None, None], axis=0)
            m_i1 = tl.sum(m_neq * (idx_row == i1).to(tl.float32)[:, None, None, None], axis=0)
            m_i2 = tl.sum(m_neq * (idx_row == i2).to(tl.float32)[:, None, None, None], axis=0)
            m_i1_new = m_i1 - uz * m_i0
            m_i2_new = m_i2 - 2.0 * uz * m_i1 + uz * uz * m_i0
            m_neq = tl.where((idx_row == i1)[:, None, None, None], m_i1_new[None, :, :, :], m_neq)
            m_neq = tl.where((idx_row == i2)[:, None, None, None], m_i2_new[None, :, :, :], m_neq)

        if USE_EXTERNAL_TAU:
            omega = omega_eff
        else:
            omega = 1.0 / (3.0 * nu_lb + 0.5)

        # Trace/deviatoric split at indices 4..6.
        m4 = tl.sum(m_neq * (idx_row == 4).to(tl.float32)[:, None, None, None], axis=0)
        m5 = tl.sum(m_neq * (idx_row == 5).to(tl.float32)[:, None, None, None], axis=0)
        m6 = tl.sum(m_neq * (idx_row == 6).to(tl.float32)[:, None, None, None], axis=0)
        trace = m4 + m5 + m6
        m4_new = (1.0 - omega) * (m4 - trace / 3.0)
        m5_new = (1.0 - omega) * (m5 - trace / 3.0)
        m6_new = (1.0 - omega) * (m6 - trace / 3.0)
        m_neq = tl.where((idx_row == 4)[:, None, None, None], m4_new, m_neq)
        m_neq = tl.where((idx_row == 5)[:, None, None, None], m5_new, m_neq)
        m_neq = tl.where((idx_row == 6)[:, None, None, None], m6_new, m_neq)

        # Shear at 7..9.
        for i in tl.static_range(7, 10):
            m_i = tl.sum(m_neq * (idx_row == i).to(tl.float32)[:, None, None, None], axis=0)
            m_i_new = (1.0 - omega) * m_i
            m_neq = tl.where((idx_row == i)[:, None, None, None], m_i_new, m_neq)

        # 3rd order zero.
        for i in tl.static_range(10, 16):
            m_neq = tl.where((idx_row == i)[:, None, None, None], 0.0, m_neq)

        # 4th order zero.
        for i in tl.static_range(16, 19):
            m_neq = tl.where((idx_row == i)[:, None, None, None], 0.0, m_neq)

        # Inverse shifts: z, y, x.
        for g in tl.static_range(0, 5):
            i0 = SHIFT_Z[g * 3 + 0]
            i1 = SHIFT_Z[g * 3 + 1]
            i2 = SHIFT_Z[g * 3 + 2]
            m_i0 = tl.sum(m_neq * (idx_row == i0).to(tl.float32)[:, None, None, None], axis=0)
            m_i1 = tl.sum(m_neq * (idx_row == i1).to(tl.float32)[:, None, None, None], axis=0)
            m_i2 = tl.sum(m_neq * (idx_row == i2).to(tl.float32)[:, None, None, None], axis=0)
            m_i1_new = m_i1 + uz * m_i0
            m_i2_new = m_i2 + 2.0 * uz * m_i1 + uz * uz * m_i0
            m_neq = tl.where((idx_row == i1)[:, None, None, None], m_i1_new[None, :, :, :], m_neq)
            m_neq = tl.where((idx_row == i2)[:, None, None, None], m_i2_new[None, :, :, :], m_neq)
        for g in tl.static_range(0, 5):
            i0 = SHIFT_Y[g * 3 + 0]
            i1 = SHIFT_Y[g * 3 + 1]
            i2 = SHIFT_Y[g * 3 + 2]
            m_i0 = tl.sum(m_neq * (idx_row == i0).to(tl.float32)[:, None, None, None], axis=0)
            m_i1 = tl.sum(m_neq * (idx_row == i1).to(tl.float32)[:, None, None, None], axis=0)
            m_i2 = tl.sum(m_neq * (idx_row == i2).to(tl.float32)[:, None, None, None], axis=0)
            m_i1_new = m_i1 + uy * m_i0
            m_i2_new = m_i2 + 2.0 * uy * m_i1 + uy * uy * m_i0
            m_neq = tl.where((idx_row == i1)[:, None, None, None], m_i1_new[None, :, :, :], m_neq)
            m_neq = tl.where((idx_row == i2)[:, None, None, None], m_i2_new[None, :, :, :], m_neq)
        for g in tl.static_range(0, 5):
            i0 = SHIFT_X[g * 3 + 0]
            i1 = SHIFT_X[g * 3 + 1]
            i2 = SHIFT_X[g * 3 + 2]
            m_i0 = tl.sum(m_neq * (idx_row == i0).to(tl.float32)[:, None, None, None], axis=0)
            m_i1 = tl.sum(m_neq * (idx_row == i1).to(tl.float32)[:, None, None, None], axis=0)
            m_i2 = tl.sum(m_neq * (idx_row == i2).to(tl.float32)[:, None, None, None], axis=0)
            m_i1_new = m_i1 + ux * m_i0
            m_i2_new = m_i2 + 2.0 * ux * m_i1 + ux * ux * m_i0
            m_neq = tl.where((idx_row == i1)[:, None, None, None], m_i1_new[None, :, :, :], m_neq)
            m_neq = tl.where((idx_row == i2)[:, None, None, None], m_i2_new[None, :, :, :], m_neq)

        fneq_star = tl.sum(
            M_inv_2d[:, :, None, None, None] * m_neq[None, :, :, :, :],
            axis=1,
        )
        f_post = feq + fneq_star
    elif COLLISION == 2:
        # ---- CUMULANT ----
        fneq = f_eff - feq
        mq = mask_q[:, None, None, None].to(tl.float32)
        pi_xx = tl.sum(cx_b * cx_b * fneq * mq, axis=0)
        pi_yy = tl.sum(cy_b * cy_b * fneq * mq, axis=0)
        pi_zz = tl.sum(cz_b * cz_b * fneq * mq, axis=0)
        pi_xy = tl.sum(cx_b * cy_b * fneq * mq, axis=0)
        pi_xz = tl.sum(cx_b * cz_b * fneq * mq, axis=0)
        pi_yz = tl.sum(cy_b * cz_b * fneq * mq, axis=0)
        if USE_EXTERNAL_TAU:
            omega = omega_eff
        else:
            omega = 1.0 / (3.0 * nu_lb + 0.5)
        omega_b = 1.0
        omega_even = 1.0
        trace = pi_xx + pi_yy + pi_zz
        delta_b = (omega_b - omega) * trace / 3.0
        pi_xx_s = pi_xx - omega * pi_xx - delta_b
        pi_yy_s = pi_yy - omega * pi_yy - delta_b
        pi_zz_s = pi_zz - omega * pi_zz - delta_b
        pi_xy_s = pi_xy - omega * pi_xy
        pi_xz_s = pi_xz - omega * pi_xz
        pi_yz_s = pi_yz - omega * pi_yz
        cs2 = 1.0 / 3.0
        h_xx = cx_b * cx_b - cs2
        h_yy = cy_b * cy_b - cs2
        h_zz = cz_b * cz_b - cs2
        h_xy = cx_b * cy_b
        h_xz = cx_b * cz_b
        h_yz = cy_b * cz_b
        fneq_reg = 4.5 * w_b * (
            h_xx * pi_xx_s + h_yy * pi_yy_s + h_zz * pi_zz_s
            + 2.0 * h_xy * pi_xy_s + 2.0 * h_xz * pi_xz_s + 2.0 * h_yz * pi_yz_s
        )
        fneq_ho_unrel = 4.5 * w_b * (
            h_xx * pi_xx + h_yy * pi_yy + h_zz * pi_zz
            + 2.0 * h_xy * pi_xy + 2.0 * h_xz * pi_xz + 2.0 * h_yz * pi_yz
        )
        fneq_ho = fneq - fneq_ho_unrel
        fneq_ho_s = (1.0 - omega_even) * fneq_ho
        f_post = feq + fneq_reg + fneq_ho_s
    else:
        f_post = f_eff

    # Float mask (BZ, BY, BX) — 1.0 at solid cells, 0.0 at fluid cells.
    # Reuses the own-cell wall mask already loaded for the BB guard.
    own_is_wall_f = own_is_wall.to(tl.float32)

    # === Ladd (1994) momentum-exchange force reduction ===
    # Production PyTorch order is: collide → stream → **sample force
    # pre-bounce-back** → bounce-back → BC.  Inside the fused kernel
    # collide+stream+BB are fused in a single launch, but ``f_in`` is
    # the post-stream value at the own cell (``f_in[q, x] =
    # f_pre[q, x - c_q]``), and ``where(bb_fires, f_own_opp, f_in)``
    # writes to a NEW register ``f_eff`` — ``f_in`` itself is
    # preserved.  Sampling ``f_in`` masked by ``own_is_wall_f`` gives
    # exactly the populations PyTorch's ``compute_obstacle_forces_3d``
    # would see at this phase, with zero extra reads.
    if COMPUTE_FORCE:
        fx_cell = tl.sum(cx_i[:, None, None, None].to(tl.float32) * f_in,
                         axis=0)
        fy_cell = tl.sum(cy_i[:, None, None, None].to(tl.float32) * f_in,
                         axis=0)
        fz_cell = tl.sum(cz_i[:, None, None, None].to(tl.float32) * f_in,
                         axis=0)

        # Mask fluid cells (multiply by 0) and apply Ladd ×2 — matches
        # ``compute_obstacle_forces_3d`` exactly.
        fx_cell = 2.0 * fx_cell * own_is_wall_f
        fy_cell = 2.0 * fy_cell * own_is_wall_f
        fz_cell = 2.0 * fz_cell * own_is_wall_f

        # Reduce over tile → per-program scalar, then atomic_add.
        tl.atomic_add(fx_buf_ptr, tl.sum(fx_cell))
        tl.atomic_add(fy_buf_ptr, tl.sum(fy_cell))
        tl.atomic_add(fz_buf_ptr, tl.sum(fz_cell))

    # === Write output (32, BZ, BY, BX) — same shape as f_eff ===
    dst_offs = (offs_q[:, None, None, None].to(tl.int64) * stride_q
                + offs_z[None, :, None, None].to(tl.int64) * stride_z
                + offs_y[None, None, :, None].to(tl.int64) * stride_y
                + offs_x[None, None, None, :].to(tl.int64) * stride_x)
    tl.store(fnew_ptr + dst_offs, f_post, mask=rw_mask)


# ---------------------------------------------------------------------------
# Force reduction kernel (parallel sum across all cells in (nz, ny, nx))
# ---------------------------------------------------------------------------
#
# Reads the post-collide distribution ``f`` and the obstacle mask, computes
# F_x = 2 * Σ_q Σ_x_solid c_{q,x} f[q, x_solid] (and similarly for y, z).
# Returns three scalar tensors on device.
#
# Sampling phase is *post-collide+stream+BB*: the fused obstacle kernel
# writes a distribution that already includes wet-node bounce-back at
# fluid cells adjacent to walls.  Production's
# ``obstacles.compute_obstacle_forces_3d`` samples pre-bounce-back.
# See the public wrapper's docstring for the precision implication.
# ---------------------------------------------------------------------------

@triton.jit
def _obstacle_force_reduction_kernel(
    f_ptr,
    obstacle_ptr,
    cxi_ptr, cyi_ptr, czi_ptr,
    fx_buf_ptr, fy_buf_ptr, fz_buf_ptr,
    nz, ny, nx,
    stride_q, stride_z, stride_y, stride_x,
    BLOCK: tl.constexpr,
    Q_PAD: tl.constexpr,
):
    """Reduce force over a slab of cells.

    Each program processes ``BLOCK`` contiguous flat-indices ``i`` into
    the spatial tensor (shape (nz, ny, nx)).  Splits work over the
    (Q, nz*ny*nx) volume.

    Output: atomic add into ``fx_buf_ptr``, ``fy_buf_ptr``, ``fz_buf_ptr``.
    """
    pid = tl.program_id(0)
    base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total = nz * ny * nx
    mask = offs < total

    # Decompose flat index into (z, y, x).
    z = offs // (ny * nx)
    rem = offs - z * (ny * nx)
    y = rem // nx
    x = rem - y * nx
    z = tl.where(mask, z, 0)
    y = tl.where(mask, y, 0)
    x = tl.where(mask, x, 0)

    # Obstacle mask at this cell.
    obst_offs = z * stride_z + y * stride_y + x * stride_x
    obst = tl.load(obstacle_ptr + obst_offs, mask=mask, other=0)
    is_wall = obst > 0
    coeff = tl.where(is_wall, 1.0, 0.0).to(tl.float32)

    # Sum q over c_{q,α} * f[q, z, y, x] for α ∈ {x, y, z}.
    fx_acc = tl.zeros((), dtype=tl.float32)
    fy_acc = tl.zeros((), dtype=tl.float32)
    fz_acc = tl.zeros((), dtype=tl.float32)
    offs_q = tl.arange(0, Q_PAD)
    mask_q = offs_q < 19
    cx_q = tl.load(cxi_ptr + offs_q, mask=mask_q, other=0)[:, None]
    cy_q = tl.load(cyi_ptr + offs_q, mask=mask_q, other=0)[:, None]
    cz_q = tl.load(czi_ptr + offs_q, mask=mask_q, other=0)[:, None]
    f_offs = (offs_q.to(tl.int64)[:, None] * stride_q
              + z.to(tl.int64)[None, :] * stride_z
              + y.to(tl.int64)[None, :] * stride_y
              + x.to(tl.int64)[None, :] * stride_x)
    cell_mask = mask_q[:, None] & mask[None, :]
    f_vals = tl.load(f_ptr + f_offs, mask=cell_mask, other=0.0)
    fx_local = tl.sum(cx_q * f_vals, axis=0)
    fy_local = tl.sum(cy_q * f_vals, axis=0)
    fz_local = tl.sum(cz_q * f_vals, axis=0)
    fx_acc += tl.sum(fx_local * coeff)
    fy_acc += tl.sum(fy_local * coeff)
    fz_acc += tl.sum(fz_local * coeff)

    tl.atomic_add(fx_buf_ptr, 2.0 * fx_acc)
    tl.atomic_add(fy_buf_ptr, 2.0 * fy_acc)
    tl.atomic_add(fz_buf_ptr, 2.0 * fz_acc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def triton_fused_obstacle_les(
    f: torch.Tensor,
    nu_lb: float,
    obstacle: torch.Tensor,
    Cs: float,
    delta: float,
    *,
    out: torch.Tensor | None = None,
    block_x: int = DEFAULT_BLOCK_X,
    block_y: int = DEFAULT_BLOCK_Y,
    num_warps: int = DEFAULT_NUM_WARPS,
    num_stages: int = DEFAULT_NUM_STAGES,
) -> torch.Tensor:
    """One LBM step with bounce-back walls + LES Smagorinsky + BGK collide.

    Args:
        f: Distribution tensor of shape ``(Q, nz, ny, nx)``, fp32.
        nu_lb: Molecular kinematic viscosity in lattice units.
        obstacle: ``int8[NZ, NY, NX]`` mask (1 = wall, 0 = fluid).
        Cs: Smagorinsky constant (typically 0.10-0.18).
        delta: LES filter width in lattice units (typically 1.0).
        out: Optional output tensor (must match shape/dtype of ``f``).

    Returns:
        Tensor holding the post-step distribution.  Wall cells are
        skipped on the write side; their values reflect the previous
        step's state and are not used downstream.
    """
    if obstacle.shape != f.shape[1:]:
        raise ValueError(
            f"obstacle shape {tuple(obstacle.shape)} does not match "
            f"f's spatial shape {tuple(f.shape[1:])}"
        )
    Q, nz, ny, nx = f.shape
    if Q != _Q_PAD:
        raise ValueError(
            f"Expected Q={_Q_PAD} (padded D3Q19), got Q={Q}"
        )
    if out is None:
        out = torch.empty_like(f)

    lat = make_lattice_tensors(str(f.device))
    grid = (triton.cdiv(ny, block_y), triton.cdiv(nx, block_x), nz)
    opp = _OPPOSITE.to(f.device)
    _fused_collide_stream_obstacle_les_kernel[grid](
        f, out,
        obstacle, opp,
        lat["cxi"], lat["cyi"], lat["czi"],
        lat["cxf"], lat["cyf"], lat["czf"], lat["w"],
        nu_lb, Cs * Cs * delta * delta,
        nz, ny, nx,
        f.stride(0), f.stride(1), f.stride(2), f.stride(3),
        Q_PAD=_Q_PAD, BLOCK_X=block_x, BLOCK_Y=block_y,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def triton_fused_obstacle_xfar_les(
    f: torch.Tensor,
    nu_lb: float,
    obstacle: torch.Tensor,
    Cs: float,
    delta: float,
    *,
    collision: str = "BGK",
    out: torch.Tensor | None = None,
    block_x: int = 32,
    block_y: int = 1,
    block_z: int | None = None,
    num_warps: int = 1,
    num_stages: int = DEFAULT_NUM_STAGES,
    tau_eff: torch.Tensor | None = None,
    fx_buf: torch.Tensor | None = None,
    fy_buf: torch.Tensor | None = None,
    fz_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """X-streamwise fused collide + stream + BB + LES Smagorinsky (V2).

    Matches the production SUBOFF runner's convention: x is the
    streamwise axis (axis 2 of ``(Q, nz, ny, nx)``).  The kernel is
    periodic on all three axes; the caller must overwrite the 6 boundary
    faces with :func:`apply_far_field_bc_6face` (matching
    ``boundaries3d.far_field_bc_3d``).

    V2 rewrites the tile to ``(Q=32 arange, BZ, BY, BX)`` with BX
    innermost (stride_x=1 → coalesced loads), and accepts ``Q=19``
    buffers (matching production).  Tile spans all three spatial axes.

    Args:
        f: Distribution tensor of shape ``(Q=19, nz, ny, nx)``, fp32.
        obstacle: ``int8[NZ, NY, NX]`` mask (1 = wall, 0 = fluid).
        Cs: Smagorinsky constant.
        delta: LES filter width (lattice units).
        collision: One of ``"BGK"``, ``"CM"``, ``"CUMULANT"``.  Selects
            the collide sub-branch in the fused kernel via the
            ``COLLISION: tl.constexpr`` tag.  KBC is not supported
            (use the PyTorch 5-op chain).
        out: Optional output tensor (must match shape/dtype of ``f``).
        block_x: X-axis tile size (innermost; must be power-of-2 ≥ 16
            for coalesced loads).  Default 32.
        block_y, block_z: Y/Z axis tile sizes.  ``block_z`` defaults to
            ``block_y`` if None.  Default 4.
        tau_eff: Optional ``[NZ, NY, NX]`` per-cell effective relaxation
            time (lattice units).  When supplied, BGK skips its internal
            |S|-based Smagorinsky LES and CM/CUMULANT use
            ``omega_eff = 1/tau_eff`` element-wise.  Useful for WALE and
            Vreman SGS coupling — pass the output of
            :func:`tensorlbm.suboff_cmk_kbc_runner._compute_sgs_tau_eff`.
        fx_buf, fy_buf, fz_buf: Optional scalar ``torch.float32``
            tensors.  When all three are supplied, the kernel computes
            the Ladd (1994) wet-node momentum-exchange force in the same
            launch as the collide+stream+BB step (each program
            accumulates 2 · Σ_q c_q · f_in[q] over its OWN wall cells
            then ``tl.atomic_add`` into these buffers).  ``f_in`` is the
            post-stream, **pre-bounce-back** population — the exact
            sampling phase of production
            :func:`tensorlbm.obstacles.compute_obstacle_forces_3d`
            (called between ``stream3d`` and ``far_field_bc_3d``); the
            bounce-back swap writes to ``f_eff`` and never disturbs
            ``f_in``.  The caller is
            responsible for **zeroing them before each call** — the
            kernel uses ``atomic_add``, not assignment.  Pass ``None``
            to disable the fused force computation (the standalone
            :func:`triton_obstacle_force_reduction` is still available
            for that case, but it samples post-bounce-back and therefore
            deviates from the production phase).

    Returns:
        Tensor holding the post-step distribution.
    """
    if obstacle.shape != f.shape[1:]:
        raise ValueError(
            f"obstacle shape {tuple(obstacle.shape)} does not match "
            f"f's spatial shape {tuple(f.shape[1:])}"
        )
    Q, nz, ny, nx = f.shape
    if Q != 19:
        raise ValueError(
            f"Expected Q=19 (production D3Q19), got Q={Q}"
        )
    if out is None:
        out = torch.empty_like(f)
    if block_z is None:
        block_z = block_y

    use_external_tau = tau_eff is not None
    if use_external_tau:
        if tau_eff.shape != (nz, ny, nx):
            raise ValueError(
                f"tau_eff shape {tuple(tau_eff.shape)} does not match "
                f"f's spatial shape {(nz, ny, nx)}"
            )
        if tau_eff.dtype != torch.float32:
            raise ValueError(
                f"tau_eff must be float32, got {tau_eff.dtype}"
            )
        if tau_eff.device != f.device:
            raise ValueError(
                f"tau_eff device {tau_eff.device} does not match "
                f"f device {f.device}"
            )

    # Force-fusion dispatch: when all three scalar buffers are supplied
    # the kernel computes the Ladd force via ``tl.atomic_add``; otherwise
    # the fused-force block is constexpr-eliminated.  When the buffers
    # are not supplied, pass ``fx_buf`` etc. through as a placeholder
    # tensor — Triton requires non-null pointers even when unused.
    compute_force = (
        fx_buf is not None and fy_buf is not None and fz_buf is not None
    )
    if not compute_force:
        # ``tl.atomic_add`` requires fp32 pointers.  The output buffer
        # ``out`` is always fp32 (Q, nz, ny, nx) and is unused after this
        # kernel returns, so it's a safe placeholder when the force
        # buffers are not supplied.  The kernel sees ``COMPUTE_FORCE=False``
        # and prunes the entire force block.
        fx_buf = out
        fy_buf = out
        fz_buf = out

    collision_tag = _dispatch_collision(collision)

    # Resolve and bake the CM/CUMULANT constexpr tables on first call.
    # Triton needs these as tl.constexpr at compile time; we pass them
    # as 2-D numpy arrays of float32.  Empty arrays are passed for the
    # BGK branch — the kernel's constexpr dead-code elimination ignores
    # them when ``COLLISION == 0``.
    _ensure_collision_tables()
    assert _CM_TABLES is not None
    assert _CUMULANT_HERMITE is not None
    assert _CM_SHIFT_X is not None
    assert _CM_SHIFT_Y is not None
    assert _CM_SHIFT_Z is not None

    # Flatten the shift triplets into a single 1-D tuple of 15 ints.
    # Triton constexpr must be hashable; numpy arrays are not.  We convert
    # all constexpr tables to nested tuples here.
    def _flat(triplets: tuple[tuple[int, int, int], ...]) -> tuple[int, ...]:
        out: list[int] = []
        for trip in triplets:
            out.extend(int(c) for c in trip)
        return tuple(out)

    def _to_f32_tuple(arr: np.ndarray) -> tuple:
        # Convert 2-D float32 array to nested tuple of Python floats.
        return tuple(float(x) for row in arr for x in row)

    shift_x_flat = _flat(_CM_SHIFT_X)
    shift_y_flat = _flat(_CM_SHIFT_Y)
    shift_z_flat = _flat(_CM_SHIFT_Z)
    M_cm_tuple = _to_f32_tuple(_CM_TABLES["M"])
    M_inv_cm_tuple = _to_f32_tuple(_CM_TABLES["M_inv"])
    Hermite_cum_tuple = _to_f32_tuple(_CUMULANT_HERMITE)

    lat = make_lattice_tensors(str(f.device))
    grid = (triton.cdiv(nx, block_x), triton.cdiv(ny, block_y),
            triton.cdiv(nz, block_z))
    opp = _OPPOSITE.to(f.device)

    # When USE_EXTERNAL_TAU is False, ``tau_eff_ptr`` is unused — pass a
    # placeholder (obstacle_ptr) so the kernel signature is satisfied.
    tau_eff_ptr_arg = tau_eff if use_external_tau else obstacle
    if use_external_tau:
        tau_eff_sz, tau_eff_sy, tau_eff_sx = (
            tau_eff.stride(0), tau_eff.stride(1), tau_eff.stride(2),
        )
    else:
        tau_eff_sz, tau_eff_sy, tau_eff_sx = 1, 1, 1  # ignored

    _fused_v2_kernel_xfar_les[grid](
        f, out,
        obstacle, opp,
        lat["cxi"], lat["cyi"], lat["czi"],
        lat["cxf"], lat["cyf"], lat["czf"], lat["w"],
        nu_lb, Cs * Cs * delta * delta,
        nz, ny, nx,
        f.stride(0), f.stride(1), f.stride(2), f.stride(3),
        tau_eff_ptr_arg,
        tau_eff_sz, tau_eff_sy, tau_eff_sx,
        fx_buf, fy_buf, fz_buf,
        BLOCK_X=block_x, BLOCK_Y=block_y, BLOCK_Z=block_z,
        COLLISION=collision_tag,
        M_CM=M_cm_tuple,
        M_INV_CM=M_inv_cm_tuple,
        HERMITE_CUM=Hermite_cum_tuple,
        SHIFT_X=shift_x_flat,
        SHIFT_Y=shift_y_flat,
        SHIFT_Z=shift_z_flat,
        USE_EXTERNAL_TAU=use_external_tau,
        COMPUTE_FORCE=compute_force,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def triton_obstacle_force_reduction(
    f: torch.Tensor,
    obstacle: torch.Tensor,
    *,
    fx_buf: torch.Tensor | None = None,
    fy_buf: torch.Tensor | None = None,
    fz_buf: torch.Tensor | None = None,
    block: int = 4096,
    num_warps: int = 4,
    num_stages: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ladd (1994) wet-node momentum-exchange force reduction.

    Computes F_α = 2 · Σ_q c_{q,α} f[q, x_solid] over the obstacle
    cells.  Returns three scalar tensors on device (one per axis).

    Sampling phase: this reads ``f`` *after* the fused collide+stream
    + bounce-back kernel has written its post-collide distribution
    into ``f``.  This means the values at fluid cells whose pull-stream
    source was a wall have already been replaced with the opposite-
    direction population at the same fluid cell (wet-node bounce-back
    substitution).  The Ladd formula is applied to those substituted
    populations, *not* to the pre-substitution streamed values.

    Production's :func:`tensorlbm.obstacles.compute_obstacle_forces_3d`
    is called **pre-bounce-back** (after ``stream3d``, before
    ``far_field_bc_3d``).  At solid cells the fused kernel stores the
    swap-at-solid + collided distribution, whose momentum moment
    ``Σ_q c_q f_q`` flips sign each step and decays toward zero — so
    this post-bounce-back reduction collapses (measured ≈ 0 vs the
    production-phase value on an n=64 sphere, 30 steps).  For
    quantitative forces use the **fused** force buffers of
    :func:`triton_fused_obstacle_xfar_les` (``fx_buf``/``fy_buf``/
    ``fz_buf``), which sample ``f_in`` pre-bounce-back and match
    production to fp32-atomic round-off (~1e-5 relative).

    The buffers ``fx_buf``, ``fy_buf``, ``fz_buf`` are reused across
    calls — they are zeroed at the start of this function so callers
    can pass persistent buffers.

    Args:
        f: Distribution tensor ``(Q, nz, ny, nx)``, fp32.
        obstacle: ``int8[NZ, NY, NX]`` mask.
        fx_buf, fy_buf, fz_buf: Optional persistent scalar buffers
            (fp32).  Allocated if None.
        block: Cells per program (default 4096).
        num_warps: Triton warp count.
        num_stages: Triton pipeline stages.

    Returns:
        ``(fx, fy, fz)`` scalar tensors on the same device as ``f``.
    """
    if obstacle.shape != f.shape[1:]:
        raise ValueError(
            f"obstacle shape {tuple(obstacle.shape)} does not match "
            f"f's spatial shape {tuple(f.shape[1:])}"
        )
    if f.shape[0] != 19:
        raise ValueError(
            f"Expected Q=19 (production D3Q19), got Q={f.shape[0]}"
        )
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    total = nz * ny * nx

    if fx_buf is None:
        fx_buf = torch.zeros((), dtype=torch.float32, device=f.device)
    if fy_buf is None:
        fy_buf = torch.zeros((), dtype=torch.float32, device=f.device)
    if fz_buf is None:
        fz_buf = torch.zeros((), dtype=torch.float32, device=f.device)
    fx_buf.zero_()
    fy_buf.zero_()
    fz_buf.zero_()

    lat = make_lattice_tensors(str(f.device))
    grid = (triton.cdiv(total, block),)
    _obstacle_force_reduction_kernel[grid](
        f, obstacle,
        lat["cxi"], lat["cyi"], lat["czi"],
        fx_buf, fy_buf, fz_buf,
        nz, ny, nx,
        f.stride(0), f.stride(1), f.stride(2), f.stride(3),
        BLOCK=block, Q_PAD=_Q_PAD,
        num_warps=num_warps, num_stages=num_stages,
    )
    return fx_buf, fy_buf, fz_buf


# ---------------------------------------------------------------------------
# Wetted-area drag normalization (production Ct convention)
# ---------------------------------------------------------------------------

def _voxel_wetted_area(mask: torch.Tensor, dx: float) -> float:
    """Wetted surface area of a voxelized obstacle, in lattice units.

    Counts every solid face exposed to a non-solid neighbour (or to the
    domain exterior) and multiplies by ``dx²``.  This is the production
    SUBOFF drag normalization: ``suboff_cmk_kbc_runner`` computes
    ``dynamic_pressure = 0.5 · ρ · u_in² · _voxel_wetted_area(solid, dx)``
    and reports ``Ct = F_x / dynamic_pressure`` — the reference area is
    the *wetted* area, not the frontal area.

    Matches :func:`tensorlbm.suboff_resistance._voxel_wetted_area`
    semantics exactly (host-side PyTorch ops; never enters the Triton
    kernel).

    Args:
        mask: ``(nz, ny, nx)`` solid mask (bool or integer).
        dx: Cell size in lattice units; area = face_count · dx².

    Returns:
        Wetted area as a Python float.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.ndim != 3:
        raise ValueError("mask must be a 3D tensor")
    m = mask
    area_faces = torch.tensor(0, dtype=torch.int64, device=m.device)
    # Faces on the domain boundary (solid cell touching the exterior).
    area_faces += m[:, :, 0].sum()
    area_faces += m[:, :, -1].sum()
    area_faces += m[:, 0, :].sum()
    area_faces += m[:, -1, :].sum()
    area_faces += m[0, :, :].sum()
    area_faces += m[-1, :, :].sum()
    # Interior faces: solid on one side, fluid on the other.
    area_faces += (m[:, :, 1:] != m[:, :, :-1]).sum()
    area_faces += (m[:, 1:, :] != m[:, :-1, :]).sum()
    area_faces += (m[1:, :, :] != m[:-1, :, :]).sum()
    return float(area_faces.item()) * dx * dx


def obstacle_drag_coefficient(
    fx: float,
    obstacle: torch.Tensor,
    u_in: float,
    *,
    dx: float = 1.0,
    rho_lu: float = 1.0,
    s_ref: str = "wetted",
    streamwise_axis: int = 2,
    s_ref_value: float | None = None,
) -> dict[str, float]:
    """Normalize a Ladd momentum-exchange force into a drag coefficient.

    Production convention (``suboff_cmk_kbc_runner``): ``C_D = F_x /
    (0.5 · ρ · u_in² · S_ref)`` with ``S_ref`` the **wetted** voxel
    area.  The frontal (projected) area is offered only for
    side-by-side reporting — SUBOFF resistance coefficients (Ct) are
    defined on the wetted area.

    Args:
        fx: Streamwise force from the Ladd reduction (lattice units).
        obstacle: ``(nz, ny, nx)`` solid mask.
        u_in: Free-stream velocity (lattice units).
        dx: Cell size (lattice units).
        rho_lu: Density (lattice units, default 1.0).
        s_ref: ``"wetted"`` (production default) or ``"frontal"``.
        streamwise_axis: Spatial axis of the free stream — 2 for x
            (the ``*_xfar_*`` kernels / production layout
            ``(Q, nz, ny, nx)``), 0 for z.  Only used when
            ``s_ref="frontal"``.
        s_ref_value: Explicit reference area override (lattice²).  When
            given it takes precedence over ``s_ref``.

    Returns:
        Dict with keys ``C_D``, ``S_ref``, ``s_ref_kind``,
        ``dynamic_pressure``, ``fx``.
    """
    if s_ref_value is not None:
        s_ref_kind = "explicit"
        area = float(s_ref_value)
    elif s_ref == "wetted":
        s_ref_kind = "wetted"
        area = _voxel_wetted_area(obstacle, dx)
    elif s_ref == "frontal":
        s_ref_kind = "frontal"
        area = float(obstacle.bool().any(dim=streamwise_axis).sum().item()) * dx * dx
    else:
        raise ValueError(f"s_ref must be 'wetted' or 'frontal', got {s_ref!r}")

    dynamic_pressure = 0.5 * rho_lu * u_in * u_in * area
    c_d = float(fx) / dynamic_pressure if dynamic_pressure > 0.0 else 0.0
    return {
        "C_D": c_d,
        "S_ref": area,
        "s_ref_kind": s_ref_kind,
        "dynamic_pressure": dynamic_pressure,
        "fx": float(fx),
    }


# ---------------------------------------------------------------------------
# Boundary-condition helpers (PyTorch, host-side)
# ---------------------------------------------------------------------------

def apply_inflow_zou_he(
    f: torch.Tensor,
    vel: torch.Tensor,
    *,
    plane: int = 0,
    direction: int = 1,
) -> None:
    """Zou-He velocity inlet on the given z-plane.

    Sets the populations pointing *into* the domain (towards +z if
    direction=1, towards -z if direction=-1) to match a prescribed
    velocity ``vel[3, ny, nx]``.  Modifies ``f`` in place.

    Reference: Zou & He (1997), \"On pressure and velocity boundary
    conditions for the lattice Boltzmann BGK model\".

    Args:
        f: ``(Q, nz, ny, nx)`` distribution.  ``plane`` is the global
            z index at which the inlet is applied.
        vel: ``(3, ny, nx)`` velocity field at the inlet plane.
        direction: +1 for inlet at z=0 with flow into +z; -1 for the
            opposite face.
    """
    Q, nz, ny, nx = f.shape
    if vel.shape != (3, ny, nx):
        raise ValueError(
            f"vel shape {tuple(vel.shape)} does not match (3, {ny}, {nx})"
        )
    if not (0 <= plane < nz):
        raise ValueError(f"plane={plane} out of range [0, {nz})")

    # D3Q19 direction sets per axis (matches tensorlbm.boundaries3d._D3Q19_*).
    # The naive 18-q formula does NOT produce opposites for this lattice;
    # use the OPPOSITE table.
    #
    # For z=0 inlet with +z flow (cz>0): unknown populations are 5,11,14,15,18.
    # For z=nz-1 inlet with -z flow (cz<0): unknown populations are 6,12,13,16,17.
    if direction == 1 and plane == 0:
        in_q = torch.tensor([5, 11, 14, 15, 18], device=f.device)
        out_q = torch.tensor([6, 12, 13, 16, 17], device=f.device)
    elif direction == -1 and plane == nz - 1:
        in_q = torch.tensor([6, 12, 13, 16, 17], device=f.device)
        out_q = torch.tensor([5, 11, 14, 15, 18], device=f.device)
    else:
        raise NotImplementedError(
            f"Inflow at plane={plane} direction={direction} not yet "
            "supported; only z=0/+z and z=nz-1/-z inlet faces."
        )

    # Sum of outgoing populations at this plane.
    sum_out = f[out_q, plane].sum(0)
    # Density from Zou-He (with safety against divide-by-zero).
    denom = 1.0 - vel[2] + 1e-10
    rho = (1.0 + sum_out) / denom  # (ny, nx)
    usq = vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2  # (ny, nx)
    # Vectorised non-equilibrium bounce-back (no Python loop).
    # in_q and out_q are tensors on f.device.  OPPOSITE is also on device.
    opp_t = _OPPOSITE.to(f.device)
    in_opp = opp_t[in_q]                                    # (n_in,)
    # f[:, plane] for the incoming and opp directions:
    f_in_pl = f[in_q, plane]                                # (n_in, ny, nx)
    f_opp_pl = f[in_opp, plane]                             # (n_in, ny, nx)
    # Per-direction cx,cy,cz and weights as device tensors indexed by in_q.
    cxq = _CX_T.to(f.device)[in_q]
    cyq = _CY_T.to(f.device)[in_q]
    czq = _CZ_T.to(f.device)[in_q]
    wq = _W_T.to(f.device)[in_q]
    cxop = _CX_T.to(f.device)[in_opp]
    cyop = _CY_T.to(f.device)[in_opp]
    czop = _CZ_T.to(f.device)[in_opp]
    wop = _W_T.to(f.device)[in_opp]
    # cu = 3 * c . u, with u = (vel[0], vel[1], vel[2])  (ny, nx each).
    cu_q = 3.0 * (cxq[:, None, None] * vel[0][None]
                  + cyq[:, None, None] * vel[1][None]
                  + czq[:, None, None] * vel[2][None])      # (n_in, ny, nx)
    cu_op = 3.0 * (cxop[:, None, None] * vel[0][None]
                   + cyop[:, None, None] * vel[1][None]
                   + czop[:, None, None] * vel[2][None])
    feq_q = (wq[:, None, None] * rho[None]
             * (1.0 + cu_q + 0.5 * cu_q * cu_q - 1.5 * usq[None]))
    feq_op = (wop[:, None, None] * rho[None]
              * (1.0 + cu_op + 0.5 * cu_op * cu_op - 1.5 * usq[None]))
    f[in_q, plane] = feq_q + f_opp_pl - feq_op


def apply_outflow_zero_gradient(
    f: torch.Tensor,
    *,
    plane: int | None = None,
    direction: int = -1,
) -> None:
    """Zero-gradient (Neumann) outflow on the given z-plane.

    Copies populations from the adjacent plane: ``f[:, plane, :, :] =
    f[:, plane + direction, :, :]``.  This is the simplest and most
    common outflow BC; it lets the wake exit the domain with minimal
    reflection.  Modifies ``f`` in place.

    Args:
        f: ``(Q, nz, ny, nx)`` distribution.
        plane: z index for the outlet (default: ``nz - 1`` for +z outflow).
        direction: +1 for outlet at z=0 with flow exiting in -z direction;
            -1 for outlet at z=nz-1 with flow exiting in +z direction.
    """
    nz = f.shape[1]
    if plane is None:
        plane = nz - 1 if direction == -1 else 0
    if not (0 <= plane < nz):
        raise ValueError(f"plane={plane} out of range [0, {nz})")
    src_plane = plane + direction
    if not (0 <= src_plane < nz):
        # Already at the boundary; nothing to copy from.
        return
    f[:, plane, :, :] = f[:, src_plane, :, :]


# ---------------------------------------------------------------------------
# Production-style BC writes (6-face far-field) and mass correction
# ---------------------------------------------------------------------------

def apply_far_field_bc_6face(
    f: torch.Tensor,
    u_in: float,
    *,
    uy: float = 0.0,
    uz: float = 0.0,
    feq_pad_buf: torch.Tensor | None = None,
    feq_vec: torch.Tensor | None = None,
) -> None:
    """Far-field Dirichlet BC on all 6 domain faces (production convention).

    Matches :func:`tensorlbm.boundaries3d.far_field_bc_3d` semantics for a
    3D domain with x as the streamwise axis:

    * ``f[:, :, :, 0]`` (x=0 inlet) → free-stream equilibrium with
      ``u = (u_in, uy, uz)``, ``rho = 1``.
    * ``f[:, :, :, -1]`` (x=nx-1 outlet) → zero-gradient copy from
      ``x = nx-2``.
    * ``f[:, :, 0, :]`` (y=0) and ``f[:, :, -1, :]`` (y=ny-1) → free-stream
      equilibrium.
    * ``f[:, 0, :, :]`` (z=0) and ``f[:, -1, :, :]`` (z=nz-1) → free-stream
      equilibrium.

    Accepts ``f`` padded to ``_Q_PAD`` (32 for D3Q19); pads the
    equilibrium to match before the boundary writes.  Slots beyond
    ``Q=19`` remain zero (kernel masks them off).

    Modifies ``f`` in place.  These writes are O(Q·area) — far cheaper
    than the volume-bound fused collide+stream step, so they remain
    in PyTorch rather than being fused into the Triton kernel.

    Args:
        feq_vec: Optional pre-computed equilibrium vector ``(Q,)`` for
            ``u = (u_in, uy, uz)``, ``rho = 1``.  When provided, skips
            the full-grid equilibrium compute (~200 MB of memory
            traffic at n=128, ~3 ms) and uses this constant vector
            for all 5 boundary planes via broadcasting.  Pass the
            pre-allocated ``TritonStepState.feq_vec_buf`` after
            filling it once (typically via
            :func:`_compute_uniform_equilibrium_vec`).
    """
    from tensorlbm.d3q19 import equilibrium3d  # local import to avoid cycle

    Q, nz, ny, nx = f.shape

    if feq_vec is None:
        # Slow path: compute equilibrium on a (1,1,1) grid (microseconds).
        rho1 = torch.ones((1, 1, 1), dtype=f.dtype, device=f.device)
        feq = equilibrium3d(
            rho1,
            torch.full_like(rho1, u_in),
            torch.full_like(rho1, uy),
            torch.full_like(rho1, uz),
            device=f.device,
        )
        feq_vec = feq[:, 0, 0, 0].contiguous()  # shape (Q,)
    elif feq_vec.dim() != 1:
        raise ValueError(
            f"feq_vec must be 1-D shape (Q,), got shape "
            f"{tuple(feq_vec.shape)}"
        )

    # Pad feq_vec from Q_phys → Q if f's buffer is _Q_PAD wide.
    # Use the caller's pre-allocated scratch (1-D, shape (Q,)) when
    # provided to avoid per-step allocation.
    if feq_vec.shape[0] < Q:
        if feq_pad_buf is None or feq_pad_buf.shape != (Q,) or \
                feq_pad_buf.dtype != f.dtype or \
                feq_pad_buf.device != f.device:
            feq_pad_buf = torch.zeros(
                (Q,), dtype=f.dtype, device=f.device,
            )
        else:
            feq_pad_buf.zero_()
        feq_pad_buf[:feq_vec.shape[0]] = feq_vec
        feq_vec = feq_pad_buf
    elif feq_vec.shape[0] > Q:
        feq_vec = feq_vec[:Q]

    # Inlet (x=0): shape (Q, 1, 1) broadcasts to (Q, nz, ny).
    f[:, :, :, 0] = feq_vec[:, None, None]
    # Outlet (x=nx-1): zero-gradient copy from x=nx-2.
    if nx >= 2:
        f[:, :, :, -1] = f[:, :, :, -2]
    # Lateral y± Dirichlet: shape (Q, 1, 1) broadcasts to (Q, nz, nx).
    f[:, :, 0, :] = feq_vec[:, None, None]
    if ny >= 2:
        f[:, :, -1, :] = feq_vec[:, None, None]
    # Lateral z± Dirichlet.  In 2D-extruded mode (nz <= 4) the z-axis
    # is left as the kernel's periodic wrap; in full 3D both z faces
    # are overwritten.  Guards use the same ``nz > 4`` threshold as
    # production's ``boundaries3d.far_field_bc_3d``.
    if nz > 4:
        f[:, 0, :, :] = feq_vec[:, None, None]
        if nz >= 2:
            f[:, -1, :, :] = feq_vec[:, None, None]


def _compute_uniform_equilibrium_vec(
    u_in: float,
    uy: float,
    uz: float,
    Q: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute equilibrium distribution ``f_eq`` for uniform ``u=(u_in, uy, uz)``,
    ``rho=1`` as a single ``(Q,)`` vector.

    Used to pre-fill :attr:`TritonStepState.feq_vec_buf` so the BC
    function can avoid the per-step full-grid equilibrium computation.
    """
    from tensorlbm.d3q19 import equilibrium3d
    rho1 = torch.ones((1, 1, 1), dtype=dtype, device=device)
    feq = equilibrium3d(
        rho1,
        torch.full_like(rho1, u_in),
        torch.full_like(rho1, uy),
        torch.full_like(rho1, uz),
        device=device,
    )
    return feq[:, 0, 0, 0].contiguous()


def apply_mass_correction(
    f: torch.Tensor,
    target_mass: float,
) -> torch.Tensor:
    """Rescale ``f`` so ``f.sum() == target_mass``.

    Mirrors :func:`tensorlbm.solver3d.correct_mass3d` exactly.  Returns
    ``f`` itself (modified in place; same identity for call-site
    compatibility).
    """
    current = float(f.sum().item())
    if abs(current) < 1e-30:
        return f
    f.mul_(target_mass / current)
    return f


# ---------------------------------------------------------------------------
# SUBOFF geometry (Darpa Suboff, hull + sail + stern appendages)
# ---------------------------------------------------------------------------

def create_suboff_obstacle_torch(
    nx: int, ny: int, nz: int,
    *,
    device: str | torch.device = "cuda:0",
    dx: float = 1.0,
    scale: float = 1.0,
    L_factor: float = 1.0,
    with_sail: bool = True,
    with_stern: bool = True,
) -> torch.Tensor:
    """Build the SUBOFF obstacle mask ``int8[nz, ny, nx]``.

    Uses the DARPA SUBOFF cross-section polynomials (in lattice units,
    scaled by ``dx``).  ``L_factor > 1`` stretches the geometry to
    fit a longer domain.

    The hull is centred at ``(nx//2, ny//2, nz//2)``.

    The output axis order matches the LBM distribution tensor
    ``f[Q, nz, ny, nx]``: streamwise = z (axis 0), span = y (axis 1),
    lateral = x (axis 2).  This is what
    :func:`triton_fused_obstacle_les` expects.
    """
    dev = torch.device(device)
    # Build the geometry directly on a (nz, ny, nx) meshgrid so that
    # SUBOFF x (the long axis, 14.29 ft) maps to LBM axis 0 (streamwise,
    # since flow is in +z).  Previous ordering used arange(nx) for SUBOFF x
    # and permute(2,1,0), which placed the long axis in LBM x (lateral) —
    # a sideways hull, not the slender along-stream hull the SUBOFF benchmark
    # expects.
    g = torch.arange(nx, dtype=torch.float32, device=dev)
    X, Y, Z = torch.meshgrid(torch.arange(nz, dtype=torch.float32, device=dev),
                             torch.arange(ny, dtype=torch.float32, device=dev),
                             g,
                             indexing="ij")
    cx, cy, cz = nz // 2, ny // 2, nx // 2
    x_local = (X - cx) * dx
    y_local = (Y - cy) * dx
    z_local = (Z - cz) * dx

    # Convert to feet (1 ft = 0.3048 m).  SUBOFF polynomials are
    # parameterised in feet from the nose.
    ft_per_lx = 1.0 / 0.3048
    x_ft = x_local * ft_per_lx

    R = torch.zeros_like(x_ft)

    # Nose: 0 <= x_ft <= 3.333333 (parabolic).
    m1 = (x_ft >= 0) & (x_ft <= 3.333333)
    tmp = 0.3 * x_ft - 1.0
    a1 = (1.126395101 * x_ft * tmp ** 4
          + 0.442874707 * x_ft ** 2 * tmp ** 3
          + 1.0 - tmp ** 4 * (1.2 * x_ft + 1.0))
    R1 = 0.8333333 * torch.sqrt(torch.clamp(a1, min=0))
    R = torch.where(m1, R1, R)

    # Cylinder: 3.333333 < x_ft <= 10.645833.
    m2 = (x_ft > 3.333333) & (x_ft <= 10.645833)
    R = torch.where(m2, torch.full_like(R, 0.254), R)

    # Tail taper: 10.645833 < x_ft <= 13.979167 (5th-order polynomial).
    m3 = (x_ft > 10.645833) & (x_ft <= 13.979167)
    r1 = 0.1175
    k0, k1 = 10.0, 44.6244
    ksi = (13.979167 - x_ft) / 3.333333
    a3 = (r1 * r1 + r1 * k0 * ksi ** 2
          + (20 - 20 * r1 * r1 - 4 * r1 * k0 - k1 / 3) * ksi ** 3
          + (-45 + 45 * r1 * r1 + 6 * r1 * k0 + k1) * ksi ** 4
          + (36 - 36 * r1 * r1 - 4 * r1 * k0 - k1) * ksi ** 5
          + (-10 + 10 * r1 * r1 + r1 * k0 + k1 / 3) * ksi ** 6)
    R3 = 0.8333333 * torch.sqrt(torch.clamp(a3, min=0))
    R = torch.where(m3, R3, R)

    # End cap: 13.979167 < x_ft <= 14.291667 (elliptical).
    m4 = (x_ft > 13.979167) & (x_ft <= 14.291667)
    a4 = 1.0 - (3.2 * x_ft - 44.733333) ** 2
    R4 = 0.8333333 * torch.sqrt(torch.clamp(a4, min=0)) * 0.1175
    R = torch.where(m4, R4, R)

    # Convert R back to lattice units.
    R_lx = R / ft_per_lx / dx
    # NOTE: y_local, z_local are in METRES.  Compare against R (also in metres),
    # not R_lx (lattice units).  Comparing m^2 to lx^2 inflated the cross-section
    # radius by 1/dx ≈ 23.5x, turning the slender SUBOFF hull into a solid
    # prism filling the entire y-z plane.  See BUG_REPORT_SUBOFF_unit_mismatch.md.
    hull = ((y_local ** 2 + z_local ** 2) < R ** 2).to(torch.int8)

    obstacle = hull

    # Sail (fairwater): rectangular box near x_ft ≈ 10.
    if with_sail:
        sail_x_lo, sail_x_hi = 9.5, 11.0
        sail_y_max = 0.18  # ft
        sail_z_max = 0.32
        m_sail = ((x_ft >= sail_x_lo) & (x_ft <= sail_x_hi)
                  & (y_local.abs() <= (sail_y_max / ft_per_lx))
                  & (z_local <= sail_z_max / ft_per_lx))
        obstacle = torch.clamp(obstacle + m_sail.to(torch.int8), 0, 1)

    # Stern appendages: simple flat plates.
    if with_stern:
        stern_x_lo, stern_x_hi = 12.0, 13.5
        stern_thickness = 0.04 / ft_per_lx
        m_stern = ((x_ft >= stern_x_lo) & (x_ft <= stern_x_hi)
                   & ((y_local.abs() < stern_thickness)
                      | (z_local.abs() < stern_thickness)))
        obstacle = torch.clamp(obstacle + m_stern.to(torch.int8), 0, 1)

    # The meshgrid now produces shape (nz, ny, nx) directly because the
    # SUBOFF x arange is over nz.  No permute needed; the layout already
    # matches the kernel's expected (nz, ny, nx) format.
    # Streamwise = z (axis 0), span = y (axis 1), lateral = x (axis 2).
    return obstacle.contiguous()