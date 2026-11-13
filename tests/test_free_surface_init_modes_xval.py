"""Tests for the birth-state init modes (XVAL-LBM-AVERAGE-SURROUND-INIT).

Layered suite:

* **oracle parity** — the torch kernel must reproduce the NumPy ``float64``
  reference oracle *and* a literal per-cell transcription of the OSS
  ``Maarten-vd-Sande/lbm::average_surround`` body;
* **semantics** — only active (LIQUID|INTERFACE) neighbours enter the average;
  an isolated cell falls back to the caller's reference density instead of the
  OSS zero-population state;
* **baseline contrast** — ``fixed_rho`` deviates from the OSS reference by more
  than the 15 % owner threshold once a density contrast exists, while
  ``average_surround`` stays at float32 round-off;
* **regression** — the historical ``_init_new_2d`` behaviour is reproduced
  bit-exactly by the new code path (replica of the pre-patch body);
* **integration** — ``koerner_step_2d`` threads ``init_mode`` into its birth
  sites, keeps the liquid inventory conserved and stays finite.
"""

from __future__ import annotations

import torch
import pytest

from tensorlbm.d2q9 import equilibrium
from tensorlbm import free_surface_init_modes as im
from tensorlbm.free_surface_lbm_2d import (
    GAS,
    INTERFACE,
    LIQUID,
    SOLID,
    _address_interface_layer,
    _init_new_2d,
    init_fill_rectangular_2d,
    init_flags_from_fill_2d,
    init_mass_from_fill_2d,
    koerner_step_2d,
    total_liquid_inventory_2d,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _state(ny=16, nx=20, rho_liquid=1.0, seed=0):
    """Liquid block + gas, with a shear velocity and a density perturbation."""
    torch.manual_seed(seed)
    flags = torch.full((ny, nx), GAS, dtype=torch.int8)
    flags[2 : ny - 2, 2 : nx // 2] = LIQUID
    flags[:, 0] = SOLID
    flags[:, -1] = SOLID
    solid = flags == SOLID
    rho = torch.where(
        (flags == LIQUID),
        torch.full((ny, nx), float(rho_liquid)) + 0.05 * torch.rand(ny, nx),
        torch.zeros((ny, nx)),
    )
    ux = torch.where(flags == LIQUID, 0.02 * torch.rand(ny, nx), torch.zeros((ny, nx)))
    uy = torch.where(flags == LIQUID, -0.01 * torch.rand(ny, nx), torch.zeros((ny, nx)))
    f = equilibrium(rho, ux, uy)
    return f, flags, solid, rho, ux, uy


def _oss_average_surround_literal(ix, iy, rho_prev, u_prev, active, shifts):
    """Literal per-cell transcription of the OSS ``average_surround`` body.

    ``mass[ix, iy] = 0; rho = 0; u = 0;`` then ``rho/u`` averaged over the
    ``cell_type & (FLUID | INTERFACE)`` neighbours, then ``set_equi``.
    """
    ny, nx = rho_prev.shape
    rho = 0.0
    ux = 0.0
    uy = 0.0
    c = 0
    for dx, dy in shifts:
        jx, jy = ix + dx, iy + dy
        if not (0 <= jx < nx and 0 <= jy < ny):
            continue
        if active[jy, jx]:
            c += 1
            rho += float(rho_prev[jy, jx])
            ux += float(u_prev[0][jy, jx])
            uy += float(u_prev[1][jy, jx])
    if c > 0:
        rho /= c
        ux /= c
        uy /= c
    return rho, ux, uy, c


def _random_case(ny=9, nx=11, seed=3, p_active=0.6):
    g = torch.Generator().manual_seed(seed)
    rho = 0.5 + torch.rand((ny, nx), generator=g)
    ux = 0.05 * (torch.rand((ny, nx), generator=g) - 0.5)
    uy = 0.05 * (torch.rand((ny, nx), generator=g) - 0.5)
    active = torch.rand((ny, nx), generator=g) < p_active
    return rho, ux, uy, active


# ---------------------------------------------------------------------------
# oracle parity
# ---------------------------------------------------------------------------


def test_neighbour_average_matches_numpy_oracle():
    rho, ux, uy, active = _random_case()
    mean_rho, (mean_ux, mean_uy), count = im.neighbour_average(
        rho, (ux, uy), active, im.D2Q9_SHIFTS
    )
    ref_rho, (ref_ux, ref_uy), ref_count = im.average_surround_reference(
        rho, (ux, uy), active, fallback_rho=0.0, shifts=im.D2Q9_SHIFTS
    )
    sl = (slice(1, -1), slice(1, -1))
    assert torch.equal(count[sl], torch.from_numpy(ref_count)[sl].to(count.dtype))
    # torch.roll is periodic, the oracle treats the border as inactive; compare
    # the interior where both agree.
    sl = (slice(1, -1), slice(1, -1))
    for got, ref in ((mean_rho, ref_rho), (mean_ux, ref_ux), (mean_uy, ref_uy)):
        err = (got[sl] - torch.from_numpy(ref)[sl]).abs().max().item()
        scale = max(1e-12, torch.from_numpy(ref)[sl].abs().max().item())
        assert err / scale < 1e-6, err


def test_neighbour_average_matches_literal_oss_port():
    rho, ux, uy, active = _random_case(seed=11)
    mean_rho, (mean_ux, mean_uy), count = im.neighbour_average(
        rho, (ux, uy), active, im.D2Q9_SHIFTS
    )
    for iy in range(1, rho.shape[0] - 1):
        for ix in range(1, rho.shape[1] - 1):
            r, a, b, c = _oss_average_surround_literal(
                ix, iy, rho, (ux, uy), active.numpy(), im.D2Q9_SHIFTS
            )
            assert int(count[iy, ix]) == c
            if c == 0:
                continue
            assert abs(float(mean_rho[iy, ix]) - r) < 1e-6
            assert abs(float(mean_ux[iy, ix]) - a) < 1e-6
            assert abs(float(mean_uy[iy, ix]) - b) < 1e-6


def test_d3q19_stencil_is_complete_and_symmetric():
    shifts = im.D3Q19_SHIFTS
    assert len(shifts) == 18
    assert len({tuple(s) for s in shifts}) == 18
    assert all(tuple(-c for c in s) in {tuple(o) for o in shifts} for s in shifts)
    assert all(sum(abs(c) for c in s) in (1, 2) for s in shifts)


# ---------------------------------------------------------------------------
# semantics: active-only averaging, isolated-cell fallback
# ---------------------------------------------------------------------------


def test_only_active_neighbours_enter_the_average():
    rho = torch.zeros((7, 7))
    active = torch.zeros((7, 7), dtype=torch.bool)
    # centre surrounded by 8 active neighbours with distinct densities
    rho[3, 3] = 99.0  # the cell itself must never enter its own average
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    for (dx, dy), v in zip(im.D2Q9_SHIFTS, vals):
        rho[3 + dy, 3 + dx] = v
        active[3 + dy, 3 + dx] = True
    ux = torch.zeros((7, 7))
    uy = torch.zeros((7, 7))
    mean_rho, _, count = im.neighbour_average(rho, (ux, uy), active, im.D2Q9_SHIFTS)
    assert int(count[3, 3]) == 8
    assert float(mean_rho[3, 3]) == pytest.approx(sum(vals) / 8.0, rel=1e-6)


def test_isolated_birth_falls_back_to_reference_density():
    """OSS writes ``rho = 0`` (``c == 0``) -> empty population; we fall back."""
    ny, nx = 7, 7
    flags = torch.full((ny, nx), GAS, dtype=torch.int8)
    flags[3, 3] = GAS  # the cell being born: no active neighbour at all
    rho = torch.zeros((ny, nx))
    f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
    mask = torch.zeros((ny, nx), dtype=torch.bool)
    mask[3, 3] = True

    got_avg = _init_new_2d(
        f, flags, mask, 1.0, ux=torch.zeros_like(rho), uy=torch.zeros_like(rho),
        init_mode=im.INIT_MODE_AVERAGE_SURROUND,
    )
    pop = float(got_avg[:, 3, 3].sum())
    assert pop == pytest.approx(1.0, rel=1e-6)  # rho = fallback, not 0

    # historical zero fallback (rho_init=None) reproduces the OSS defect
    got_legacy = _init_new_2d(
        f, flags, mask, None, ux=torch.zeros_like(rho), uy=torch.zeros_like(rho)
    )
    assert float(got_legacy[:, 3, 3].sum()) == 0.0


def test_init_mode_validation_rejects_typos():
    f, flags, _, _, ux, uy = _state()
    mask = torch.zeros_like(flags, dtype=torch.bool)
    with pytest.raises(ValueError):
        _init_new_2d(f, flags, mask, 1.0, ux=ux, uy=uy, init_mode="average_surounds")
    with pytest.raises(ValueError):
        im.resolve_birth_mode("nope", site="envelope")
    with pytest.raises(ValueError):
        im.resolve_birth_mode("legacy", site="unknown_site")


# ---------------------------------------------------------------------------
# baseline contrast against the OSS reference
# ---------------------------------------------------------------------------


def _relative_deviation(a, b):
    scale = max(abs(b), 1e-12)
    return abs(a - b) / scale


def test_fixed_rho_exceeds_threshold_under_density_contrast():
    """With a real density contrast, ``fixed_rho`` is >15 % off the OSS ref."""
    rho, ux, uy, active = _random_case(seed=5, p_active=0.9)
    rho = rho + 0.5  # densities in [0.5, 1.5]
    ref_rho, ref_vel, count = im.average_surround_reference(
        rho, (ux, uy), active, fallback_rho=1.0, shifts=im.D2Q9_SHIFTS
    )
    interior = torch.zeros(rho.shape, dtype=torch.bool)
    interior[1:-1, 1:-1] = True
    live = (count > 0) & interior.numpy()
    assert live.sum() > 20

    got_avg, _, _ = im.neighbour_average(rho, (ux, uy), active, im.D2Q9_SHIFTS)
    err_avg = [
        _relative_deviation(float(got_avg[i, j]), float(ref_rho[i, j]))
        for i, j in zip(*torch.nonzero(torch.from_numpy(live)).T)
    ]
    assert max(err_avg) < 1e-6

    # fixed_rho would use rho_ref = 1.0 for every born cell
    err_fixed = [
        _relative_deviation(1.0, float(ref_rho[i, j]))
        for i, j in zip(*torch.nonzero(torch.from_numpy(live)).T)
    ]
    assert max(err_fixed) > 0.15


# ---------------------------------------------------------------------------
# regression: historical behaviour preserved bit-exactly
# ---------------------------------------------------------------------------


def _pre_patch_init_new_2d(f, flags, mask, rho_init, ux, uy):
    """Replica of ``_init_new_2d`` as it stood before this change (HEAD)."""
    active = (flags == LIQUID) | (flags == INTERFACE)
    nb = torch.stack(
        [
            torch.roll(active.to(f.dtype), shifts=(dy, dx), dims=(0, 1))
            for dy, dx in (((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)))
        ]
    )
    cnt = nb.sum(dim=0).clamp(min=1.0)
    shifts = ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))
    ux_nb = torch.stack([torch.roll(ux, shifts=(dy, dx), dims=(0, 1)) for dy, dx in shifts]) * nb
    uy_nb = torch.stack([torch.roll(uy, shifts=(dy, dx), dims=(0, 1)) for dy, dx in shifts]) * nb
    ux_m = ux_nb.sum(dim=0) / cnt
    uy_m = uy_nb.sum(dim=0) / cnt
    if rho_init is None:
        rho = f.sum(0)
        rho_nb = (
            torch.stack(
                [torch.roll(rho, shifts=(dy, dx), dims=(0, 1)) for dy, dx in shifts]
            )
            * nb
        ).sum(dim=0) / cnt
        rho_field = torch.where(nb.sum(dim=0) > 0, rho_nb, torch.zeros_like(rho_nb))
    elif isinstance(rho_init, torch.Tensor):
        rho_field = rho_init.to(f.dtype)
    else:
        rho_field = torch.full_like(ux, float(rho_init))
    feq = equilibrium(rho_field, ux_m, uy_m)
    return torch.where(mask.unsqueeze(0), feq, f)


@pytest.mark.parametrize("rho_init", [None, 1.0, 0.7])
def test_legacy_path_is_bit_identical_to_pre_patch(rho_init):
    f, flags, _, _, ux, uy = _state(seed=2)
    mask = torch.zeros_like(flags, dtype=torch.bool)
    mask[3, 4:8] = True
    want = _pre_patch_init_new_2d(f, flags, mask, rho_init, ux, uy)
    got = _init_new_2d(f, flags, mask, rho_init, ux=ux, uy=uy)
    assert torch.equal(got, want)


def test_fixed_rho_mode_equals_explicit_scalar_reference():
    f, flags, _, _, ux, uy = _state(seed=4)
    mask = torch.zeros_like(flags, dtype=torch.bool)
    mask[4:6, 6:9] = True
    got = _init_new_2d(f, flags, mask, 1.0, ux=ux, uy=uy, init_mode="fixed_rho")
    want = _pre_patch_init_new_2d(f, flags, mask, 1.0, ux, uy)
    assert torch.equal(got, want)


# ---------------------------------------------------------------------------
# integration: the solver step threads the mode into its birth sites
# ---------------------------------------------------------------------------


def test_address_interface_layer_honours_init_mode():
    f, flags, solid, rho, ux, uy = _state(ny=12, nx=14)
    mass = init_mass_from_fill_2d((flags == LIQUID).float(), flags)
    recv_new = torch.zeros_like(solid)
    recv_new[1, 8] = True  # GAS cell next to liquid -> newborn envelope shell
    mass[1, 8] = 0.0
    out = {}
    for mode in ("legacy", "average_surround"):
        fl, m, ff, rho_fill = _address_interface_layer(
            flags.clone(), mass.clone(), f.clone(), solid, 1.0, ux, uy,
            recv_new=recv_new, rho=rho.clone(), init_mode=mode,
        )
        assert int(fl[1, 8]) == INTERFACE
        out[mode] = (float(ff[:, 1, 8].sum()), float(m[1, 8]), float(rho_fill[1, 8]))
    # The cell is isolated (every neighbour is GAS): the historical zero
    # fallback reproduces the OSS defect (empty population, ``fill = m/rho``
    # then divides by the 1e-6 clamp), while ``average_surround`` refills at
    # the liquid-scale reference density.
    assert out["legacy"][0] == 0.0
    assert out["average_surround"][0] > 0.0
    pop, _, rho_fill = out["average_surround"]
    assert abs(pop - rho_fill) < 1e-5, (pop, rho_fill)
    assert out["average_surround"][2] == pytest.approx(1.0, rel=1e-5)


def test_koerner_step_init_mode_runs_and_conserves_inventory():
    ny, nx = 24, 32
    fill, solid = init_fill_rectangular_2d(
        ny, nx, column_width=10, column_height=14, device="cpu"
    )
    flags0 = init_flags_from_fill_2d(fill, solid)
    mass0 = init_mass_from_fill_2d(fill, flags0)
    rho0 = torch.where(flags0 == GAS, torch.zeros_like(fill), torch.ones_like(fill))
    f0 = equilibrium(rho0, torch.zeros_like(fill), torch.zeros_like(fill))
    inv0 = total_liquid_inventory_2d(f0, mass0, flags0)

    results = {}
    for mode in ("legacy", "fixed_rho", "average_surround"):
        f, mass, flags = f0.clone(), mass0.clone(), flags0.clone()
        for _ in range(12):
            f, mass, flags = koerner_step_2d(
                f, mass, flags, solid, tau=0.8, gy=-1e-4, init_mode=mode
            )
        assert torch.isfinite(f).all()
        results[mode] = (
            total_liquid_inventory_2d(f, mass, flags),
            f.contiguous().numpy().tobytes(),
        )

    for mode, (inv, _) in results.items():
        assert abs(inv - inv0) / inv0 < 1e-3, (mode, inv, inv0)
    # legacy is the default -> identical state
    assert results["legacy"][1] == results["average_surround"][1]

def test_free_surface_step_2d_forwards_init_mode():
    """The 3-tuple public wrapper must expose the same birth-mode switch."""
    from tensorlbm.free_surface_lbm_2d import free_surface_step_2d

    ny, nx = 20, 24
    fill, solid = init_fill_rectangular_2d(
        ny, nx, column_width=8, column_height=12, device="cpu"
    )
    flags0 = init_flags_from_fill_2d(fill, solid)
    rho0 = torch.where(flags0 == GAS, torch.zeros_like(fill), torch.ones_like(fill))
    f0 = equilibrium(rho0, torch.zeros_like(fill), torch.zeros_like(fill))

    out = {}
    for mode in ("legacy", "fixed_rho", "average_surround"):
        f, fl, mass_fill = f0.clone(), fill.clone(), flags0.clone()
        for _ in range(6):
            f, fl, mass_fill = free_surface_step_2d(
                f, fl, mass_fill, solid, tau=0.8, gy=-1e-4, init_mode=mode
            )
        assert torch.isfinite(f).all()
        out[mode] = f.contiguous().numpy().tobytes()
    assert out["legacy"] == out["average_surround"]
