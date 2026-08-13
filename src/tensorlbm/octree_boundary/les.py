"""LES sub-grid model on octree-shell leaves via neighbor-table gathers.

The WALE/Smagorinsky operators in ``tensorlbm.turbulence`` are written for
regular grid tensors (torch.roll neighbours).  Octree-shell leaves are a
sparse SoA (``f_leaf (Q, n_leaf)``) whose spatial neighbours are only known
through ``neighbor_table`` (Q, n_leaf) with sentinels
``SHELL_OUTSIDE / SOLID / DOMAIN_OUT / FANOUT``.

Here we compute the velocity-gradient tensor with neighbour *gathers*
(``f[..., neighbor_table[d, :]]``), so the sub-grid eddy viscosity is
physically meaningful on the shell.

Gradients are one-sided when the opposite neighbour is missing (sentinel),
which is the correct fallback for a shell whose inner neighbour is SOLID
(wall) and outer neighbour is SHELL_OUTSIDE (ghost-filled).
"""
from __future__ import annotations

import functools

import torch

from tensorlbm.d3q27 import macroscopic27
from tensorlbm.d3q19 import macroscopic3d

# sentinels (must match geometry.py)
SHELL_OUTSIDE = -1
SOLID = -2
DOMAIN_OUT = -3
FANOUT = -4

# D3Q19 and D3Q27 share the first six axis directions.  The old octree LES
# code accidentally used direction indices (2, 1, 0) as if they were spatial
# axes; index 0 is the rest population, so the z-gradient was identically
# zero.  Keep the mapping explicit and use both sides whenever available.
_AXIS_DIRS = ((1, 2), (3, 4), (5, 6))  # x, y, z: (+, -)


@functools.lru_cache(maxsize=16)
def _leaf_mrt_matrices(
    Q: int, device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache native sparse-LES MRT matrices in the population dtype."""
    if Q == 27:
        from tensorlbm.turbulence import _get_d3q27_mrt_matrices
        M, M_inv = _get_d3q27_mrt_matrices(device)
    elif Q == 19:
        from tensorlbm.solver3d import _get_d3q19_mrt_matrices
        M, M_inv = _get_d3q19_mrt_matrices(device)
    else:
        raise NotImplementedError(f"leaf LES supports D3Q19/D3Q27, got Q={Q}")
    return M.to(dtype=dtype), M_inv.to(dtype=dtype)


def _leaf_macros(f: torch.Tensor, Q: int):
    """(rho, ux, uy, uz) per leaf from (Q, n_leaf) populations."""
    if Q == 27:
        rho, ux, uy, uz = macroscopic27(f.view(Q, 1, 1, -1))
    else:
        rho, ux, uy, uz = macroscopic3d(f.view(Q, 1, 1, -1))
    return (rho.reshape(-1), ux.reshape(-1), uy.reshape(-1), uz.reshape(-1))


def _gather_velocity(
    f: torch.Tensor, d: int, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
    neighbor_table: torch.Tensor, n_leaf: int,
    *,
    neighbor_velocity: torch.Tensor | None = None,
    neighbor_distance: torch.Tensor | None = None,
):
    """Velocity of the neighbour in direction d.

    ``neighbor_velocity`` is supplied by the sharded octree stepper when a
    neighbour lives on another device (or is a coarse/fine fan-out).  The
    legacy path gathers from the local leaf table.  ``neighbor_distance`` is
    a positive distance for valid neighbours and zero for sentinels.
    """
    if neighbor_velocity is not None:
        if neighbor_velocity.ndim != 3 or neighbor_velocity.shape[0] != 3:
            raise ValueError(
                "neighbor_velocity must have shape (3,Q,n_leaf)",
            )
        gx = neighbor_velocity[0, d]
        gy = neighbor_velocity[1, d]
        gz = neighbor_velocity[2, d]
        if neighbor_distance is None:
            valid = torch.isfinite(gx) & torch.isfinite(gy) & torch.isfinite(gz)
        else:
            valid = neighbor_distance[d] > 0.0
        return gx, gy, gz, valid
    nb = neighbor_table[d]
    valid = nb >= 0
    idx = nb.clamp(min=0)
    # zeros for invalid entries (masked later)
    gx = torch.where(valid, ux[idx], torch.zeros_like(ux))
    gy = torch.where(valid, uy[idx], torch.zeros_like(uy))
    gz = torch.where(valid, uz[idx], torch.zeros_like(uz))
    return gx, gy, gz, valid

def _gradient(f, d: int, u, neighbor_table, n_leaf, dx):
    """One-sided central difference ∂u/∂x_d via neighbour gather.

    ``dx`` may be a scalar or a per-leaf tensor ``(n_leaf,)``; cross-level
    neighbours use the *actual* centre distance along the axis (leaf_center
    based) so mixed-level shells (d_max=2) get physically correct gradients.
    """
    dp, dm = _AXIS_DIRS[d]
    gp, _, _, vp = _gather_velocity(
        f, dp, u, torch.zeros_like(u), torch.zeros_like(u), neighbor_table, n_leaf,
    )
    gm, _, _, vm = _gather_velocity(
        f, dm, u, torch.zeros_like(u), torch.zeros_like(u), neighbor_table, n_leaf,
    )
    dx_t = torch.as_tensor(dx, dtype=u.dtype, device=u.device)
    central = (gp - gm) / (2.0 * dx_t)
    one_sided_p = (gp - u) / dx_t
    one_sided_m = (u - gm) / dx_t
    return torch.where(vp & vm, central, torch.where(vp, one_sided_p,
                       torch.where(vm, one_sided_m, torch.zeros_like(u))))


def _gradient_centers(
    f, d: int, u, neighbor_table, leaf_center, n_leaf,
    *, neighbor_velocity: torch.Tensor | None = None,
    neighbor_distance: torch.Tensor | None = None,
):
    """Gradient using actual neighbour centre distances along spatial axis.

    ``d`` is a spatial axis (0=x, 1=y, 2=z), not a lattice direction index.
    Both the positive and negative lattice directions are used when present;
    a one-sided estimate is retained at a wall or shell interface.
    """
    dp, dm = _AXIS_DIRS[d]
    gp, _, _, vp = _gather_velocity(
        f, dp, u, torch.zeros_like(u), torch.zeros_like(u), neighbor_table, n_leaf,
        neighbor_velocity=neighbor_velocity,
        neighbor_distance=neighbor_distance,
    )
    gm, _, _, vm = _gather_velocity(
        f, dm, u, torch.zeros_like(u), torch.zeros_like(u), neighbor_table, n_leaf,
        neighbor_velocity=neighbor_velocity,
        neighbor_distance=neighbor_distance,
    )
    if neighbor_distance is not None:
        dp_t = neighbor_distance[dp].to(dtype=u.dtype, device=u.device)
        dm_t = neighbor_distance[dm].to(dtype=u.dtype, device=u.device)
    else:
        def _distance(direction: int, valid: torch.Tensor) -> torch.Tensor:
            nb = neighbor_table[direction]
            idx = nb.clamp(min=0)
            delta = leaf_center[:, :] - leaf_center[idx, :]
            axis_vec = torch.argmax(delta.abs(), dim=1)
            row = torch.arange(leaf_center.shape[0], device=leaf_center.device)
            return torch.where(
                valid,
                delta[row, axis_vec].abs().clamp_min(1e-12),
                torch.ones_like(u),
            )
        dp_t = _distance(dp, vp)
        dm_t = _distance(dm, vm)
    central = (gp - gm) / (dp_t + dm_t).clamp_min(1e-12)
    one_sided_p = (gp - u) / dp_t.clamp_min(1e-12)
    one_sided_m = (u - gm) / dm_t.clamp_min(1e-12)
    return torch.where(vp & vm, central, torch.where(vp, one_sided_p,
                       torch.where(vm, one_sided_m, torch.zeros_like(u))))


def _dx_per_leaf(leaf_level: torch.Tensor, base_dx: float = 0.5) -> torch.Tensor:
    """Per-leaf spacing in L0 coarse units: dx = 2^(-level).

    Leaf volume is 2^(-3*level) of the L0 coarse cell (geometry.py:298), so
    the leaf spacing is dx = 2^(-level): level 1 -> 0.5, level 2 -> 0.25
    (i.e. L0 dx=1 refined by 2^level per axis).  ``base_dx`` is ignored;
    kept for call-site compatibility.
    """
    return 2.0 ** (-leaf_level.to(torch.float64))


def leaf_wale_nu_t(
    f: torch.Tensor,
    neighbor_table: torch.Tensor,
    C_w: float,
    dx: float | torch.Tensor = 0.5,
    leaf_level: torch.Tensor | None = None,
    leaf_center: torch.Tensor | None = None,
    neighbor_velocity: torch.Tensor | None = None,
    neighbor_distance: torch.Tensor | None = None,
) -> torch.Tensor:
    """WALE eddy viscosity on octree leaves (neighbour-gather gradients).

    Args:
        f: (Q, n_leaf) populations.
        neighbor_table: (Q, n_leaf) int64 with sentinels.
        C_w: WALE constant.
        dx: leaf spacing; a scalar (legacy) or a per-leaf tensor
            ``(n_leaf,)``.  When ``leaf_level`` is given it overrides
            ``dx`` via ``dx = 2^(-leaf_level)`` (L0 units) — the physically
            correct per-leaf spacing for mixed-level shells (d_max=2).
        leaf_level: (n_leaf,) int64 leaf refinement level (1 or 2).
        leaf_center: (n_leaf, 3) world coordinates; when given, gradients
            use actual neighbour centre distances (correct across levels).
        neighbor_velocity: optional ``(3,Q,n_leaf)`` neighbour velocity field
            assembled by the sharded stepper, including REMOTE/FANOUT links.
        neighbor_distance: optional ``(Q,n_leaf)`` positive centre distances
            matching ``neighbor_velocity``.

    Returns:
        nu_t (n_leaf,) >= 0.
    """
    Q = f.shape[0]
    rho, ux, uy, uz = _leaf_macros(f, Q)
    n_leaf = f.shape[1]
    nbt = neighbor_table
    import os

    if os.environ.get("OCTREE_DEBUG_NAN"):
        bad_f = ~torch.isfinite(f)
        bad_m = ~(torch.isfinite(rho) & torch.isfinite(ux)
                  & torch.isfinite(uy) & torch.isfinite(uz))
        if bool(bad_f.any()):
            print(
                f"[dbg] leaf_wale_nu_t: f INPUT non-finite {int(bad_f.sum().item())} "
                f"leaves={int(bad_f.any(dim=0).sum().item())} "
                f"fmin={float(f.min().item()):.4e} fmax={float(f.max().item()):.4e}",
                flush=True,
            )
        elif bool(bad_m.any()):
            print(
                f"[dbg] leaf_wale_nu_t: f finite but macros non-finite "
                f"{int(bad_m.sum().item())} leaves "
                f"levels={leaf_level[bad_m][:5].tolist() if leaf_level is not None else '?'} "
                f"centers={leaf_center[bad_m][:3].tolist() if leaf_center is not None else '?'} "
                f"rho_bad={rho[bad_m][:3].tolist()} "
                f"|f|max_bad={f[:, bad_m].abs().max(dim=0).values[:3].tolist()}",
                flush=True,
            )
        else:
            bad = ~(torch.isfinite(rho) & (rho > 0.0))
            if bool(bad.any()):
                print(
                    f"[dbg] leaf_wale_nu_t: rho<=0 {int(bad.sum().item())} leaves "
                    f"rho_min={float(rho.min().item()):.4e}",
                    flush=True,
                )
    if leaf_level is not None:
        dx = _dx_per_leaf(leaf_level)
    grad = (
        (lambda f, d, u, nbt, n_leaf: _gradient_centers(
            f, d, u, nbt, leaf_center, n_leaf,
            neighbor_velocity=neighbor_velocity,
            neighbor_distance=neighbor_distance))
        if leaf_center is not None else
        (lambda f, d, u, nbt, n_leaf, dx=dx: _gradient(f, d, u, nbt, n_leaf, dx))
    )

    # velocity gradient tensor g_ij = ∂u_i / ∂x_j (axis 0=x, 1=y, 2=z)
    g11 = grad(f, 0, ux, nbt, n_leaf)   # ∂ux/∂x
    g12 = grad(f, 1, ux, nbt, n_leaf)   # ∂ux/∂y
    g13 = grad(f, 2, ux, nbt, n_leaf)   # ∂ux/∂z
    g21 = grad(f, 0, uy, nbt, n_leaf)
    g22 = grad(f, 1, uy, nbt, n_leaf)
    g23 = grad(f, 2, uy, nbt, n_leaf)
    g31 = grad(f, 0, uz, nbt, n_leaf)
    g32 = grad(f, 1, uz, nbt, n_leaf)
    g33 = grad(f, 2, uz, nbt, n_leaf)

    # S_ij = (g_ij + g_ji)/2
    S11 = 0.5 * (g11 + g11); S12 = 0.5 * (g12 + g21); S13 = 0.5 * (g13 + g31)
    S21 = S12;              S22 = 0.5 * (g22 + g22); S23 = 0.5 * (g23 + g32)
    S31 = S13;              S32 = S23;              S33 = 0.5 * (g33 + g33)

    S2 = (S11**2 + S22**2 + S33**2
          + 2 * (S12**2 + S13**2 + S23**2))          # |S|²
    S3 = (S11 * (S22 * S33 - S23 * S32)
          - S12 * (S21 * S33 - S23 * S31)
          + S13 * (S21 * S32 - S22 * S31))           # det(S)

    # WALE: nu_t = (C_w dx)² * S3^(2/3) / (S2^(5/2) + S3^(5/3))
    # Use sign-preserving real power for possibly-negative S3, and guard the
    # 0/0 case (no shear -> nu_t must be exactly 0, not NaN).
    S2_safe = S2.clamp_min(1e-30)
    S3_abs = S3.abs().clamp_min(1e-30)
    num = S3_abs ** (2.0 / 3.0)
    den = S2_safe ** (2.5) + S3_abs ** (5.0 / 3.0) + 1e-12
    nu_t = (C_w * dx) ** 2 * num / den
    nu_t = torch.where(S2 > 1e-24, nu_t, torch.zeros_like(nu_t))
    return torch.clamp(nu_t, min=0.0)


def leaf_smagorinsky_nu_t(
    f: torch.Tensor,
    neighbor_table: torch.Tensor,
    C_s: float,
    dx: float | torch.Tensor = 0.5,
    leaf_level: torch.Tensor | None = None,
    leaf_center: torch.Tensor | None = None,
    neighbor_velocity: torch.Tensor | None = None,
    neighbor_distance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Smagorinsky eddy viscosity on octree leaves (neighbour-gather gradients)."""
    Q = f.shape[0]
    rho, ux, uy, uz = _leaf_macros(f, Q)
    n_leaf = f.shape[1]
    nbt = neighbor_table
    if leaf_level is not None:
        dx = _dx_per_leaf(leaf_level)
    grad = (
        (lambda f, d, u, nbt, n_leaf: _gradient_centers(
            f, d, u, nbt, leaf_center, n_leaf,
            neighbor_velocity=neighbor_velocity,
            neighbor_distance=neighbor_distance))
        if leaf_center is not None else
        (lambda f, d, u, nbt, n_leaf, dx=dx: _gradient(f, d, u, nbt, n_leaf, dx))
    )

    g11 = grad(f, 0, ux, nbt, n_leaf)
    g12 = grad(f, 1, ux, nbt, n_leaf)
    g13 = grad(f, 2, ux, nbt, n_leaf)
    g21 = grad(f, 0, uy, nbt, n_leaf)
    g22 = grad(f, 1, uy, nbt, n_leaf)
    g23 = grad(f, 2, uy, nbt, n_leaf)
    g31 = grad(f, 0, uz, nbt, n_leaf)
    g32 = grad(f, 1, uz, nbt, n_leaf)
    g33 = grad(f, 2, uz, nbt, n_leaf)

    S2 = (g11**2 + g22**2 + g33**2
          + 0.5 * ((g12 + g21)**2 + (g13 + g31)**2 + (g23 + g32)**2))
    nu_t = (C_s * dx) ** 2 * torch.sqrt(S2.clamp(min=0.0))
    return torch.clamp(nu_t, min=0.0)


def leaf_les_collide(
    f: torch.Tensor,
    tau: float,
    neighbor_table: torch.Tensor,
    *,
    model: str = "wale",
    C_w: float = 0.5,
    C_s: float = 0.05,
    dx: float = 0.5,
    leaf_level: torch.Tensor | None = None,
    leaf_center: torch.Tensor | None = None,
    neighbor_velocity: torch.Tensor | None = None,
    neighbor_distance: torch.Tensor | None = None,
) -> torch.Tensor:
    """MRT collision with LES on octree leaves (stable at tau->0.5).

    The sub-grid eddy viscosity comes from neighbour-table gathers (spatially
    correct on the SoA shell); D3Q19 and D3Q27 both use native MRT matrices,
    so a sparse Morton leaf set is never passed through a regular-grid roll.
    """
    Q = f.shape[0]
    if model == "wale":
        nu_t = leaf_wale_nu_t(f, neighbor_table, C_w, leaf_level=leaf_level,
                              leaf_center=leaf_center,
                              neighbor_velocity=neighbor_velocity,
                              neighbor_distance=neighbor_distance)
    else:
        nu_t = leaf_smagorinsky_nu_t(f, neighbor_table, C_s, leaf_level=leaf_level,
                                    leaf_center=leaf_center,
                                    neighbor_velocity=neighbor_velocity,
                                    neighbor_distance=neighbor_distance)

    # Cap the eddy viscosity so tau_eff stays healthy (nu_t >> nu_molecular
    # would over-relax the MRT shear rows and destabilise the shell).
    # Physical LES requires nu_t ~ O(nu) to O(10*nu); clamp to 100*nu.
    nu_mol = (tau - 0.5) / 3.0
    nu_t = torch.clamp(nu_t, max=100.0 * max(nu_mol, 1e-12))
    if bool(torch.isnan(nu_t).any()) or bool(torch.isinf(nu_t).any()):
        bad = torch.isnan(nu_t) | torch.isinf(nu_t)
        raise FloatingPointError(
            f"leaf LES nu_t non-finite: {int(bad.sum().item())}/{nu_t.numel()} "
            f"levels={leaf_level[bad][:5].tolist() if leaf_level is not None else '?'} "
            f"centers={leaf_center[bad][:3].tolist() if leaf_center is not None else '?'}",
        )

    device = f.device
    from tensorlbm.turbulence import _nu_t_to_tau_eff
    if Q == 27:
        from tensorlbm.d3q27 import macroscopic27, equilibrium27
        rho, ux, uy, uz = macroscopic27(f.view(Q, 1, 1, -1))
        feq = equilibrium27(rho, ux, uy, uz)
    elif Q == 19:
        from tensorlbm.d3q19 import macroscopic3d, equilibrium3d
        rho, ux, uy, uz = macroscopic3d(f.view(Q, 1, 1, -1))
        feq = equilibrium3d(rho, ux, uy, uz)
    else:
        raise NotImplementedError(f"leaf LES supports D3Q19/D3Q27, got Q={Q}")
    M, M_inv = _leaf_mrt_matrices(Q, device, f.dtype)
    tau_eff = _nu_t_to_tau_eff(tau, nu_t)
    s_nu_flat = (1.0 / tau_eff).reshape(-1)

    # moment space: m = M f ; feq_m = M feq ; relax non-conserved rows
    f4 = f.view(Q, 1, 1, -1)
    n = f4.shape[-1]
    m = M @ f4.view(Q, -1)
    meq = M @ feq.view(Q, -1)
    s = torch.ones(Q, n, device=device, dtype=f.dtype)
    if Q == 27:
        # row 0..3 conserved (rho, jx, jy, jz): s=0 keeps them unchanged
        s[0:4] = 0.0
        s[4] = 1.19
        s[10:19] = 1.2
        s[19] = 1.4
        s[20:27] = 1.19
        # D3Q27 shear rows 5..9
        s[5:10] = s_nu_flat.view(1, -1).expand(5, n)
    else:
        # Match collide_wale_mrt3d: D3Q19 conserved rows are 0,3,5,7;
        # shear rows are 9..13 and the remaining kinetic rows retain the
        # standard MRT rates.
        s[0] = 0.0
        s[3] = 0.0
        s[5] = 0.0
        s[7] = 0.0
        s[1] = 1.19
        s[2] = 1.4
        s[4] = 1.2
        s[6] = 1.2
        s[8] = 1.2
        s[14:16] = 1.19
        s[16:19] = 1.0
        s[9:14] = s_nu_flat.view(1, -1).expand(5, n)
    m_post = m + s * (meq - m)
    f_out = (M_inv @ m_post).view(Q, 1, 1, n)
    return f_out
