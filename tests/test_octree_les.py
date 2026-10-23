"""Regression tests for sparse-leaf LES direction and shard context."""

from __future__ import annotations

import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.octree_boundary.geometry import SOLID
from tensorlbm.octree_boundary.les import _gradient, leaf_les_collide


def test_sparse_gradient_uses_spatial_axis_not_rest_direction() -> None:
    # Three leaves along z.  Direction 5 is +z and 6 is -z; direction 0 is
    # rest and must never be used as the z derivative stencil.
    n = 3
    nt = torch.full((19, n), SOLID, dtype=torch.int64)
    nt[5, 0], nt[5, 1] = 1, 2
    nt[6, 1], nt[6, 2] = 0, 1
    u = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    f = torch.zeros((19, n), dtype=torch.float64)
    grad_z = _gradient(f, 2, u, nt, n, 1.0)
    torch.testing.assert_close(grad_z, torch.ones_like(u), atol=0.0, rtol=0.0)


def test_leaf_les_supports_d3q19() -> None:
    n = 8
    f = equilibrium3d(
        torch.ones((1, 1, n), dtype=torch.float64),
        torch.zeros((1, 1, n), dtype=torch.float64),
        torch.zeros((1, 1, n), dtype=torch.float64),
        torch.zeros((1, 1, n), dtype=torch.float64),
    )
    nt = torch.full((19, n), SOLID, dtype=torch.int64)
    centers = torch.zeros((n, 3), dtype=torch.float32)
    out = leaf_les_collide(
        f,
        0.51,
        nt,
        leaf_level=torch.ones(n, dtype=torch.int64),
        leaf_center=centers,
    )
    assert out.shape == f.shape
    assert bool(torch.isfinite(out).all())


# ---------------------------------------------------------------------------
# WALE operator variant (wall-chain decomposition fix, 2026-10-12)
# ---------------------------------------------------------------------------

from tensorlbm.octree_boundary.les import leaf_wale_nu_t  # noqa: E402
from tensorlbm.turbulence import _wale_nu_t_3d  # noqa: E402


def _strain_chain_leaves(eps: float, n: int = 5, dtype=torch.float64):
    """Leaves on a 1-D x-chain carrying a uniform strain du_x/dx = eps.

    All six axis neighbours of the interior leaves are valid, so the
    gather gradients of the interior leaves are centred differences of the
    exact linear field.
    """
    nt = torch.full((19, n), SOLID, dtype=torch.int64)
    nt[1, : n - 1] = torch.arange(1, n)  # +x neighbour
    nt[2, 1:] = torch.arange(0, n - 1)  # -x neighbour
    x = torch.arange(n, dtype=dtype)
    ux = eps * x
    f = equilibrium3d(
        torch.ones((1, 1, n), dtype=dtype),
        ux.view(1, 1, n),
        torch.zeros((1, 1, n), dtype=dtype),
        torch.zeros((1, 1, n), dtype=dtype),
    ).view(19, n)
    return f, nt


def _strain_grid_leaves(eps: float, shape=(5, 5, 5), dtype=torch.float64):
    """3-D grid of leaves carrying a uniform full-rank strain of size eps.

    ``u = eps * (x + y, x - y, z)`` gives ``g = eps * [[1,1,0],[1,-1,0],
    [0,0,1]]``: ``det(S) = -2 eps^3`` (non-degenerate, the legacy
    operator's numerator), ``S^d = eps^2 * diag(1/3, 1/3, -2/3)`` (the
    Nicoud-Ducros numerator), so under ``eps -> eps/10`` the new operator
    scales down 10x and the legacy one up 1000x.
    """
    f, nt, _ux, _uy, _uz, interior = _regular_grid_leaves(shape=shape, dtype=dtype)
    nz, ny, nx = shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=dtype),
        torch.arange(ny, dtype=dtype),
        torch.arange(nx, dtype=dtype),
        indexing="ij",
    )
    ux = eps * (xx + yy)
    uy = eps * (xx - yy)
    uz = eps * zz
    f_strain = equilibrium3d(torch.ones((nz, ny, nx), dtype=dtype), ux, uy, uz).view(
        19, nz * ny * nx
    )
    return f_strain, nt, interior


def test_wale_nicoud_ducros_vanishes_as_strain_goes_to_zero() -> None:
    """Regression (wall-chain decomposition 2026-10-11, factor F3).

    The legacy determinant operator scales as eps^-3 and DIVERGES as the
    shear -> 0 (measured on the Re=100 sphere control: median raw
    nu_t/nu_mol = 9.2e5, clamps saturated on ~100% of the band).  The
    Nicoud-Ducros operator is linear in the strain and must vanish.
    """
    for eps in (1.0e-2, 1.0e-3, 1.0e-4):
        f, nt, interior = _strain_grid_leaves(eps)
        nu_new = leaf_wale_nu_t(f, nt, 0.5, dx=1.0)[interior]
        nu_old = leaf_wale_nu_t(f, nt, 0.5, dx=1.0, wale_variant="legacy_det")[interior]
        new_i = float(nu_new.mean())
        old_i = float(nu_old.mean())
        # new operator: nu_t -> 0 (linear in eps, bounded by C_w^2 * eps)
        assert 0.0 <= new_i <= 0.5 * 0.5 * eps * 2.0
        # old operator at the same state grows ~ eps^-3 (the divergence
        # being fixed); at control-like eps it is >= 1e6x the new operator
        assert old_i / max(new_i, 1.0e-300) > 1.0e5
    # scaling: new operator linear (eps/10 -> nu_t/10) ...
    f1, nt1, m1 = _strain_grid_leaves(1.0e-3)
    f2, nt2, m2 = _strain_grid_leaves(1.0e-4)
    n1 = float(leaf_wale_nu_t(f1, nt1, 0.5, dx=1.0)[m1].mean())
    n2 = float(leaf_wale_nu_t(f2, nt2, 0.5, dx=1.0)[m2].mean())
    assert abs(n2 / n1 - 0.1) < 0.02
    # ... legacy operator eps^-3 (eps/10 -> nu_t x 1000).  NB: the pure
    # eps^-3 regime needs the physical denominator to dominate the
    # formula's own +1e-12 guard, i.e. eps >= ~2e-3 — at control-like
    # eps ~ 1e-3 the guard caps the growth (the measured control state sat
    # in exactly that saturated regime).
    f3, nt3, m3 = _strain_grid_leaves(1.0e-1)
    f4, nt4, m4 = _strain_grid_leaves(1.0e-2)
    o3 = float(leaf_wale_nu_t(f3, nt3, 0.5, dx=1.0, wale_variant="legacy_det")[m3].mean())
    o4 = float(leaf_wale_nu_t(f4, nt4, 0.5, dx=1.0, wale_variant="legacy_det")[m4].mean())
    assert abs(o4 / o3 - 1.0e3) < 0.02e3


def _regular_grid_leaves(shape=(6, 6, 6), dtype=torch.float64):
    """Flattened regular grid with axis-direction neighbour linkage.

    ``neighbor_table[d, i]`` carries the flattened index of leaf ``i``'s
    neighbour along lattice direction ``d`` (1/2 = +/-x, 3/4 = +/-y,
    5/6 = +/-z) or ``SOLID`` at the grid edge, so interior leaves exercise
    exactly the centred stencil of the regular-grid operator.
    """
    nz, ny, nx = shape
    n = nz * ny * nx
    idx = torch.arange(n).view(nz, ny, nx)
    nt = torch.full((19, n), SOLID, dtype=torch.int64)

    def link(direction: int, axis: int, delta: int) -> None:
        neighbour = torch.full_like(idx, -1)
        valid = torch.zeros_like(idx, dtype=torch.bool)
        dst = [slice(None)] * 3
        src = [slice(None)] * 3
        val = [slice(None)] * 3
        size = idx.shape[axis]
        if delta > 0:
            dst[axis], src[axis], val[axis] = (
                slice(0, size - delta),
                slice(delta, size),
                slice(0, size - delta),
            )
        else:
            dst[axis], src[axis], val[axis] = (
                slice(-delta, size),
                slice(0, size + delta),
                slice(-delta, size),
            )
        neighbour[tuple(dst)] = idx[tuple(src)]
        valid[tuple(val)] = True
        flat = neighbour.reshape(-1)
        v = valid.reshape(-1)
        cur = nt[direction].clone()
        cur[v] = flat[v]
        nt[direction] = cur

    link(1, 2, 1)  # +x
    link(2, 2, -1)  # -x
    link(3, 1, 1)  # +y
    link(4, 1, -1)  # -y
    link(5, 0, 1)  # +z
    link(6, 0, -1)  # -z

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=dtype),
        torch.arange(ny, dtype=dtype),
        torch.arange(nx, dtype=dtype),
        indexing="ij",
    )
    ux = 0.02 + 0.003 * xx + 0.001 * yy * zz
    uy = 0.01 - 0.002 * yy + 0.0005 * xx
    uz = 0.004 + 0.0015 * zz - 0.0008 * xx * yy
    f = equilibrium3d(torch.ones((nz, ny, nx), dtype=dtype), ux, uy, uz).view(19, n)
    interior_mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
    interior_mask[1:-1, 1:-1, 1:-1] = True
    return f, nt, ux, uy, uz, interior_mask.reshape(-1)


def test_wale_leaf_operator_matches_regular_grid_nicoud_ducros() -> None:
    """Parity: leaf gather operator == regular-grid _wale_nu_t_3d."""
    f, nt, ux, uy, uz, interior = _regular_grid_leaves()
    C_w = 0.5
    leaf_nu = leaf_wale_nu_t(f, nt, C_w, dx=1.0)  # default = nicoud_ducros
    reg_nu = _wale_nu_t_3d(ux, uy, uz, C_w).reshape(-1)
    sel = interior
    torch.testing.assert_close(leaf_nu[sel], reg_nu[sel], rtol=1.0e-10, atol=1.0e-14)


def test_wale_healthy_magnitude_at_control_like_shear() -> None:
    """At the control's near-wall shear the new operator must be inert.

    Control scale: leaf tau 0.5864 (nu_mol = 0.0288 leaf-lu), D = 48,
    u = 0.06 -> characteristic strain ~ u/D ~ 1.25e-3.  The new operator
    gives nu_t/nu_mol << 1 there (an LES model on a laminar field must be
    inert); the legacy operator reads ~9e5 (the measured control median
    was 9.22e5 — this synthetic reproduces that magnitude).
    """
    tau_leaf = 0.5864
    nu_mol = (tau_leaf - 0.5) / 3.0
    eps = 0.06 / 48.0
    f, nt, interior = _strain_grid_leaves(eps)
    nu_new = leaf_wale_nu_t(f, nt, 0.5, dx=0.5)[interior]
    nu_old = leaf_wale_nu_t(f, nt, 0.5, dx=0.5, wale_variant="legacy_det")[interior]
    new_ratio = float(nu_new.mean()) / nu_mol
    old_ratio = float(nu_old.mean()) / nu_mol
    assert new_ratio < 1.0
    assert old_ratio > 1.0e4


def test_leaf_wale_rejects_unknown_variant() -> None:
    f, nt = _strain_chain_leaves(1.0e-3)
    try:
        leaf_wale_nu_t(f, nt, 0.5, dx=1.0, wale_variant="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown wale_variant must raise ValueError")


# ---------------------------------------------------------------------------
# fp32 overflow fallback (highre/v3.2 library robustness fix, 2026-10-12)
# ---------------------------------------------------------------------------

from tensorlbm.octree_boundary import les as _les_module  # noqa: E402


def _old_path_nicoud_ducros(
    f: torch.Tensor,
    nt: torch.Tensor,
    C_w: float,
    dx: float | torch.Tensor = 0.5,
    leaf_level: torch.Tensor | None = None,
    leaf_center: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verbatim re-implementation of the pre-v3.2 (bc4479c) nicoud_ducros
    pipeline — the A/B oracle.

    Uses the module's own gradient helpers, so the only difference from the
    patched ``leaf_wale_nu_t`` is the absence of the float64 fallback: the
    oracle returns whatever the raw fp32/fp64 arithmetic produces, NaN
    included.
    """
    from tensorlbm.octree_boundary.les import (
        _dx_per_leaf,
        _gradient,
        _gradient_centers,
        _leaf_macros,
    )

    Q = f.shape[0]
    _rho, ux, uy, uz = _leaf_macros(f, Q)
    n_leaf = f.shape[1]
    if leaf_level is not None:
        dx = _dx_per_leaf(leaf_level)
    grad = (
        (lambda fd, d, u, nbt, n: _gradient_centers(fd, d, u, nbt, leaf_center, n))
        if leaf_center is not None
        else (lambda fd, d, u, nbt, n, dx=dx: _gradient(fd, d, u, nbt, n, dx))
    )
    g11 = grad(f, 0, ux, nt, n_leaf)
    g12 = grad(f, 1, ux, nt, n_leaf)
    g13 = grad(f, 2, ux, nt, n_leaf)
    g21 = grad(f, 0, uy, nt, n_leaf)
    g22 = grad(f, 1, uy, nt, n_leaf)
    g23 = grad(f, 2, uy, nt, n_leaf)
    g31 = grad(f, 0, uz, nt, n_leaf)
    g32 = grad(f, 1, uz, nt, n_leaf)
    g33 = grad(f, 2, uz, nt, n_leaf)
    S11 = 0.5 * (g11 + g11)
    S12 = 0.5 * (g12 + g21)
    S13 = 0.5 * (g13 + g31)
    S22 = 0.5 * (g22 + g22)
    S23 = 0.5 * (g23 + g32)
    S33 = 0.5 * (g33 + g33)
    S2 = S11**2 + S22**2 + S33**2 + 2 * (S12**2 + S13**2 + S23**2)
    g2_11 = g11 * g11 + g12 * g21 + g13 * g31
    g2_12 = g11 * g12 + g12 * g22 + g13 * g32
    g2_13 = g11 * g13 + g12 * g23 + g13 * g33
    g2_21 = g21 * g11 + g22 * g21 + g23 * g31
    g2_22 = g21 * g12 + g22 * g22 + g23 * g32
    g2_23 = g21 * g13 + g22 * g23 + g23 * g33
    g2_31 = g31 * g11 + g32 * g21 + g33 * g31
    g2_32 = g31 * g12 + g32 * g22 + g33 * g32
    g2_33 = g31 * g13 + g32 * g23 + g33 * g33
    tr_g2 = g2_11 + g2_22 + g2_33
    inv3 = 1.0 / 3.0
    Sd_11 = g2_11 - tr_g2 * inv3
    Sd_22 = g2_22 - tr_g2 * inv3
    Sd_33 = g2_33 - tr_g2 * inv3
    Sd_12 = 0.5 * (g2_12 + g2_21)
    Sd_13 = 0.5 * (g2_13 + g2_31)
    Sd_23 = 0.5 * (g2_23 + g2_32)
    Sd_norm2 = Sd_11**2 + Sd_22**2 + Sd_33**2 + 2.0 * (Sd_12**2 + Sd_13**2 + Sd_23**2)
    eps = 1e-30
    num = torch.clamp(Sd_norm2, min=0.0) ** 1.5
    den = torch.clamp(S2, min=0.0) ** 2.5 + torch.clamp(Sd_norm2, min=0.0) ** 1.25 + eps
    nu_t = (C_w * dx) ** 2 * num / den
    nu_t = torch.where(S2 > 1e-24, nu_t, torch.zeros_like(nu_t))
    return torch.clamp(nu_t, min=0.0)


def _extreme_gradient_leaves(
    eps: float = 0.1, spacing: float = 1.0e-9, n: int = 5, dtype=torch.float32
):
    """fp32 leaf chain reproducing the a1 step-150 overflow state.

    Leaf centres sit ``spacing`` apart while the velocity advances ``eps``
    per leaf, so the gather gradients are ``|g| = eps/spacing = 1e8`` —
    past the fp32 overflow band of the Nicoud-Ducros ratio (num ~ g^6 hits
    +Inf at |g| >~ 2.6e6; den ~ g^5 stays finite to |g| >~ 3.6e7; beyond
    that both are Inf and Inf/Inf = NaN, which the 100*nu_mol clamp cannot
    rescue).  Velocities stay O(0.1), so the macroscopic recovery is
    well-conditioned — only the ratio overflows, exactly like the formal
    run (leaf centres are fp32 in the octree grid; geometry.py:710).
    """
    f, nt = _strain_chain_leaves(eps, n=n, dtype=dtype)
    centers = torch.zeros((n, 3), dtype=dtype)
    centers[:, 0] = torch.arange(n, dtype=dtype) * spacing
    return f, nt, centers


def test_extreme_gradients_old_path_nan_new_path_finite() -> None:
    """(a) The raw pre-fix formula NaNs on the extreme state; the patched
    operator returns finite nu_t and the fail-closed collide guard no
    longer fires (the 2026-10-12 a1 step-150 failure mode)."""
    f, nt, centers = _extreme_gradient_leaves()
    C_w, dx = 0.5, 0.5

    old = _old_path_nicoud_ducros(f, nt, C_w, dx=dx, leaf_center=centers)
    assert bool(torch.isnan(old).all()), "old path must reproduce Inf/Inf NaN"

    nu = leaf_wale_nu_t(f, nt, C_w, dx=dx, leaf_center=centers)
    assert bool(torch.isfinite(nu).all())
    assert bool((nu > 0.0).all())

    # closed form for a pure du_x/dx gradient (g = diag(a, 0, 0)):
    #   Sd_norm2 = (2/3) a^4, S2 = a^2
    #   nu_t = (C_w dx)^2 (2/3)^1.5 a / (1 + (2/3)^1.25)
    from tensorlbm.octree_boundary.les import _gradient_centers, _leaf_macros

    _rho, ux, _uy, _uz = _leaf_macros(f, f.shape[0])
    a = _gradient_centers(f, 0, ux, nt, centers, f.shape[1])
    expect = (C_w * dx) ** 2 * (2.0 / 3.0) ** 1.5 * a.double() / (1.0 + (2.0 / 3.0) ** 1.25)
    torch.testing.assert_close(nu.double(), expect, rtol=1.0e-5, atol=1.0e-30)

    # the value entering the collision is the 100*nu_mol clamp of a finite
    # nu_t -> finite and cap-respecting; leaf_les_collide completes.
    tau = 0.55
    cap = 100.0 * max((tau - 0.5) / 3.0, 1e-12)
    assert bool(torch.isfinite(nu.clamp(max=cap)).all())
    out = leaf_les_collide(f, tau, nt, model="wale", C_w=C_w, dx=dx, leaf_center=centers)
    assert out.reshape(f.shape).shape == f.shape
    assert bool(torch.isfinite(out).all())

    # per-leaf tensor dx path (leaf_level) must be rescued too
    nu_lvl = leaf_wale_nu_t(
        f,
        nt,
        C_w,
        leaf_level=torch.ones(f.shape[1], dtype=torch.int64),
        leaf_center=centers,
    )
    assert bool(torch.isfinite(nu_lvl).all())


def test_wale_bit_identity_on_healthy_states() -> None:
    """(b) Healthy states: patched output bitwise-equal to the pre-fix
    formula (the fallback must not perturb finite arithmetic)."""
    cases = []
    f64, nt64, ux64, uy64, uz64, interior64 = _regular_grid_leaves(dtype=torch.float64)
    cases.append(("fp64 grid, scalar dx", f64, nt64, dict(dx=1.0)))
    f32, nt32, _ux, _uy, _uz, _interior = _regular_grid_leaves(dtype=torch.float32)
    cases.append(("fp32 grid, scalar dx", f32, nt32, dict(dx=1.0)))
    fch, ntch = _strain_chain_leaves(1.0e-3, dtype=torch.float32)
    centers_ch = torch.zeros((fch.shape[1], 3), dtype=torch.float32)
    centers_ch[:, 0] = torch.arange(fch.shape[1], dtype=torch.float32)
    cases.append(("fp32 chain, leaf_center", fch, ntch, dict(dx=0.5, leaf_center=centers_ch)))
    cases.append(
        (
            "fp32 chain, leaf_level (tensor dx)",
            fch,
            ntch,
            dict(
                leaf_level=torch.ones(fch.shape[1], dtype=torch.int64),
                leaf_center=centers_ch,
            ),
        )
    )
    for name, f, nt, kwargs in cases:
        old = _old_path_nicoud_ducros(f, nt, 0.5, **kwargs)
        new = leaf_wale_nu_t(f, nt, 0.5, **kwargs)
        assert torch.equal(new, old), f"bit-identity violated: {name}"
        assert bool(torch.isfinite(new).all())


def test_mixed_state_only_bad_leaves_touched() -> None:
    """(c) A mixed healthy+extreme state: the counter increments by exactly
    the oracle-NaN leaf count, and the healthy leaves keep their pre-fix
    bits."""
    f_h, nt_h, _u_h, _v_h, _w_h, _m = _regular_grid_leaves(dtype=torch.float32)
    n_h = f_h.shape[1]
    f_x, nt_x, centers_x = _extreme_gradient_leaves()
    n_x = f_x.shape[1]
    nt_x = torch.where(nt_x >= 0, nt_x + n_h, nt_x)  # shift into the tail
    f = torch.cat([f_h, f_x], dim=1)
    nt = torch.cat([nt_h, nt_x], dim=1)
    n = f.shape[1]
    centers = torch.zeros((n, 3), dtype=torch.float32)
    centers[:n_h, 0] = torch.remainder(
        torch.arange(n_h, dtype=torch.float32), 6.0
    )  # x-coordinate of the 6x6x6 grid (flattened (z, y, x))
    centers[:n_h, 1] = torch.remainder(
        torch.div(torch.arange(n_h, dtype=torch.float32), 6.0, rounding_mode="floor"),
        6.0,
    )
    centers[:n_h, 2] = torch.div(
        torch.arange(n_h, dtype=torch.float32), 36.0, rounding_mode="floor"
    )
    centers[n_h:, :] = centers_x

    old = _old_path_nicoud_ducros(f, nt, 0.5, dx=0.5, leaf_center=centers)
    n_bad = int(torch.isnan(old).sum())
    assert n_bad == n_x, "exactly the extreme tail must overflow"

    _les_module.WALE_FP64_FALLBACK_LEAVES = 0
    healthy_before = _les_module.WALE_FP64_FALLBACK_LEAVES
    leaf_wale_nu_t(f, nt, 0.5, dx=0.5)  # healthy-only call: no centers -> small grads
    healthy_delta = _les_module.WALE_FP64_FALLBACK_LEAVES - healthy_before
    assert healthy_delta == 0, "counter must not move on healthy states"

    new = leaf_wale_nu_t(f, nt, 0.5, dx=0.5, leaf_center=centers)
    delta = _les_module.WALE_FP64_FALLBACK_LEAVES - healthy_before
    assert delta == n_bad, "counter must count exactly the rescued leaves"
    assert bool(torch.isfinite(new).all())
    assert torch.equal(new[:n_h], old[:n_h]), "healthy leaves must be bit-identical"
    _les_module.WALE_FP64_FALLBACK_LEAVES = 0
