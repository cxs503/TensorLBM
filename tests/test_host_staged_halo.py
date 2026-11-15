"""Tests for the host-staged halo transport (XVAL m18373).

The reference is ProjectPhysX/FluidX3D; the facts asserted here are the ones
quoted with ``file:line`` in ``src/tensorlbm/host_staged_halo.py``:

* ``kernel.cpp:904-931`` — the D3Q19 transfer table has exactly 5 crossing
  distributions per face and side, and those indices are the ``c_axis = ±1``
  set of the reference's *own* velocity set (``kernel.cpp:871-877``).
* ``lbm.cpp:721-733`` — trimming to ``N/D`` multiples, ``N/D + 2*(D>1)`` local
  extents, ghost layers at ``0`` and ``N-1``, extract at ``1`` and ``N-2``.
* ``lbm.cpp:1305-1316`` — two host staging buffers of
  ``Amax x max(5*sizeof(fpxx), 17)`` bytes per device.
* ``lbm.cpp:1355-1382`` — per-direction extract/insert with a barrier between,
  host pointer swap instead of a host memcpy.

Plus the behavioural requirements of the port: decomposition invariance, zero
host copy volume in the swap arm, and mass conservation.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.host_staged_halo import (
    D3Q19_C,
    FLUIDX3D_D3Q19_INDEX_TRANSFER,
    FLUIDX3D_D3Q19_VELOCITY_SET,
    REF_TRANSFERS_PER_FACE,
    HostStagedD3Q19,
    HostStagedHalo,
    HostStagedLayout,
    crossing_indices,
)


def abc_velocity(n: int, u0: float = 0.05) -> torch.Tensor:
    k = 2 * math.pi / n
    z, y, x = torch.meshgrid(torch.arange(n) * k, torch.arange(n) * k, torch.arange(n) * k, indexing="ij")
    return torch.stack(
        [u0 * torch.sin(y) * torch.cos(z), u0 * torch.sin(z) * torch.cos(x), u0 * torch.sin(x) * torch.cos(y)]
    )


# ---------------------------------------------------------------------------
# Reference table fidelity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("side", ["p", "m"])
def test_reference_transfer_table_is_the_crossing_set(axis: int, side: str) -> None:
    """FluidX3D's own indices must be exactly the ``c_axis = ±1`` populations."""
    sign = +1 if side == "p" else -1
    ref_indices = FLUIDX3D_D3Q19_INDEX_TRANSFER[(axis, side)]
    assert len(ref_indices) == REF_TRANSFERS_PER_FACE == 5
    geometric = {
        q for q, c in enumerate(FLUIDX3D_D3Q19_VELOCITY_SET) if c[axis] == sign
    }
    assert set(ref_indices) == geometric
    # and the same geometric set in the TensorLBM ordering used by the port
    assert set(crossing_indices(axis, sign)) == {q for q, c in enumerate(D3Q19_C) if c[axis] == sign}


def test_crossing_sets_partition_the_velocity_set() -> None:
    for axis in (0, 1, 2):
        plus = set(crossing_indices(axis, +1))
        minus = set(crossing_indices(axis, -1))
        assert plus.isdisjoint(minus)
        assert len(plus) == len(minus) == 5


# ---------------------------------------------------------------------------
# Layout / ghost geometry
# ---------------------------------------------------------------------------


def test_layout_matches_reference_trimming_and_extents() -> None:
    lay = HostStagedLayout(30, 30, 30, 4, 1, 1)
    # lbm.cpp:722-724 — trim to a multiple of the device count
    assert lay.trimmed and (lay.gnx, lay.gny, lay.gnz) == (28, 30, 30)
    # lbm.cpp:733 — N/D + 2*(D>1)
    assert lay.owned_xyz == (7, 30, 30)
    assert lay.local_xyz == (9, 30, 30)
    assert lay.halo == (1, 0, 0)


def test_layout_amax_is_largest_decomposed_face() -> None:
    lay = HostStagedLayout(16, 16, 16, 2, 1, 2)
    # x face: (ly, lx)?  face perpendicular to x has area ly*lz; local extents
    assert lay.face_area(0) == lay.local_xyz[1] * lay.local_xyz[2]
    assert lay.face_area(2) == lay.local_xyz[0] * lay.local_xyz[1]
    assert lay.amax == max(lay.face_area(0), lay.face_area(2))


def test_reference_buffer_allocation() -> None:
    lay = HostStagedLayout(16, 16, 16, 2, 1, 1)
    halo = HostStagedHalo(lay, dtype=torch.float32)
    per_buffer = lay.amax * max(REF_TRANSFERS_PER_FACE * 4, 17)
    assert halo.buffer_bytes() == per_buffer
    assert halo.staging_allocation_bytes() == 2 * lay.n_devices * per_buffer


def test_ghost_layers_receive_neighbour_interior_layers() -> None:
    """extract(1, N-2) -> insert(N-1, 0) with the crossing-population payload."""
    n = 8
    lay = HostStagedLayout(n, n, n, 1, 1, 2)
    g = torch.randn(lay.slab_shape(), dtype=torch.float64)
    slabs = [torch.zeros(lay.slab_shape(), dtype=torch.float64) for _ in range(lay.n_devices)]
    for d, slab in enumerate(slabs):
        x0, y0, z0 = lay.global_offset(d)
        ox, oy, oz = lay.owned_xyz
        # place a distinguishable payload per device: device index in every slot
        slab[lay.owned_slice()] = (d + 1) * 10.0
    halo = HostStagedHalo(lay, dtype=torch.float64)
    halo.exchange(slabs)
    n_local = lay.local_extent(2)  # z is the decomposed axis
    for d, slab in enumerate(slabs):
        plus = lay.neighbour(d, 2, +1)
        minus = lay.neighbour(d, 2, -1)
        upper = slab[halo._ghost_slice(2, n_local - 1)]
        lower = slab[halo._ghost_slice(2, 0)]
        for q in crossing_indices(2, -1):
            assert upper[q].min() == pytest.approx((plus + 1) * 10.0)
        for q in crossing_indices(2, +1):
            assert lower[q].min() == pytest.approx((minus + 1) * 10.0)
    del g


# ---------------------------------------------------------------------------
# Transport accounting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("devs", [(1, 1, 2), (2, 1, 1), (2, 2, 2)])
def test_staging_volume_equals_reference_formula(devs: tuple[int, int, int]) -> None:
    lay = HostStagedLayout(16, 16, 16, *devs)
    halo = HostStagedHalo(lay, dtype=torch.float32)
    slabs = [torch.zeros(lay.slab_shape(), dtype=torch.float32) for _ in range(lay.n_devices)]
    halo.exchange(slabs)
    expected = sum(
        lay.n_devices * 2 * 2 * lay.face_area(axis) * REF_TRANSFERS_PER_FACE * 4
        for axis in lay.decomposed_axes
    )
    stats = halo.stats()
    assert stats["pcie_bytes"] == expected
    assert stats["barriers"] == len(lay.decomposed_axes)


def test_pointer_swap_removes_host_copy_volume() -> None:
    lay = HostStagedLayout(16, 16, 16, 2, 2, 2)
    slabs = [torch.zeros(lay.slab_shape(), dtype=torch.float32) for _ in range(lay.n_devices)]
    swap = HostStagedHalo(lay, dtype=torch.float32, staging="swap")
    swap.exchange(slabs)
    assert swap.stats()["host_copy_bytes"] == 0
    copy = HostStagedHalo(lay, dtype=torch.float32, staging="copy")
    copy.exchange(slabs)
    assert copy.stats()["host_copy_bytes"] > 0
    # both arms move the same device<->host payload
    assert copy.stats()["pcie_bytes"] == swap.stats()["pcie_bytes"]


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("devs", [(1, 1, 2), (2, 1, 1), (2, 2, 2)])
@pytest.mark.parametrize("staging", ["swap", "copy"])
def test_decomposition_invariance(devs: tuple[int, int, int], staging: str) -> None:
    n, steps = 12, 25
    u0 = abc_velocity(n)
    ref = HostStagedD3Q19(HostStagedLayout(n, n, n), nu=0.02, dtype=torch.float64)
    ref.set_velocity(u0)
    ref.step(steps)
    lay = HostStagedLayout(n, n, n, *devs)
    solver = HostStagedD3Q19(lay, nu=0.02, dtype=torch.float64, staging=staging)
    solver.set_velocity(u0)
    solver.step(steps)
    u_ref, u = ref.project_velocity(), solver.project_velocity()
    rel = float((u - u_ref).norm() / u_ref.norm())
    assert rel < 1e-10, rel
    assert abs(solver.kinetic_energy() - ref.kinetic_energy()) / ref.kinetic_energy() < 1e-10


def test_mass_conservation() -> None:
    n, steps = 12, 40
    lay = HostStagedLayout(n, n, n, 1, 2, 2)
    solver = HostStagedD3Q19(lay, nu=0.02, dtype=torch.float64)
    solver.set_velocity(abc_velocity(n))
    rho0 = solver.project_density().sum()
    solver.step(steps)
    rho1 = solver.project_density().sum()
    assert abs(float(rho1 - rho0)) / float(rho0) < 1e-12


def _tg_velocity(n: int, u0: float = 0.05) -> torch.Tensor:
    """Periodic Taylor-Green field on an ``n^3`` torus (wavenumber applied once)."""
    k = 2 * math.pi / n
    z, y, x = torch.meshgrid(
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        indexing="ij",
    )
    return torch.stack(
        [u0 * torch.sin(k * x) * torch.cos(k * y), -u0 * torch.cos(k * x) * torch.sin(k * y), torch.zeros_like(x)]
    )


def _tg_decay(n: int, nu: float, devs: tuple[int, int, int], t1: int = 20, t2: int = 100) -> float:
    solver = HostStagedD3Q19(HostStagedLayout(n, n, n, *devs), nu=nu, dtype=torch.float64)
    solver.set_velocity(_tg_velocity(n))
    solver.step(t1)
    k1 = solver.kinetic_energy()
    solver.step(t2 - t1)
    return math.log(k1 / solver.kinetic_energy()) / (t2 - t1)


def test_taylor_green_initial_condition_is_a_periodic_single_mode() -> None:
    """The physics arm is only meaningful if its IC *is* the analytic field.

    Regression for the m18373 defect: ``sin(k * i * k)`` (wavenumber applied
    twice) is not periodic on the torus and silently turned the decay arm into a
    measurement of a discontinuity.
    """
    n, u0 = 16, 0.05
    u = _tg_velocity(n, u0)
    # a mode that is not an integer number of periods on the torus is
    # under-sampled: its discrete peak is strictly smaller than u0
    assert abs(float(u.abs().max()) - u0) < 1e-14
    dudx = (torch.roll(u[0], -1, 2) - torch.roll(u[0], 1, 2)) / 2
    dudy = (torch.roll(u[1], -1, 1) - torch.roll(u[1], 1, 1)) / 2
    assert float((dudx + dudy).abs().max()) < 1e-14
    amp = torch.fft.fft2(u[0][0].to(torch.complex128)).abs()  # (z, y, x) -> z = 0 plane
    total = float(amp.sum())
    mode = float(amp[(-1, 1)] + amp[(-1, -1)] + amp[(1, 1)] + amp[(1, -1)])
    assert (total - mode) / total < 1e-14
    assert abs(0.5 * float((u * u).sum(dim=0).mean()) / (u0**2 / 4.0) - 1.0) < 1e-14


def test_taylor_green_decay_matches_analytic_and_is_decomposition_invariant() -> None:
    """TG kinetic energy must decay at ``4*nu*k^2`` on every device grid."""
    n, nu = 16, 0.02
    gamma_theory = 4.0 * nu * (2 * math.pi / n) ** 2
    gammas = {}
    for devs in ((1, 1, 1), (1, 1, 2), (2, 2, 2)):
        gammas[devs] = _tg_decay(n, nu, devs)
    for devs, g in gammas.items():
        assert abs(g - gamma_theory) / gamma_theory < 0.15, (devs, g, gamma_theory)
    ref = gammas[(1, 1, 1)]
    for devs, g in gammas.items():
        assert abs(g - ref) / ref < 1e-10, (devs, g, ref)


def test_dense_reference_step_is_reproduced_bitwise() -> None:
    """The slab/collide/stream port must equal an independent dense LBM step.

    The reference below is written from the definitions (pull streaming with
    explicit ``torch.roll`` on ``(z, y, x)`` axes, BGK equilibrium), so any
    index/axis/velocity-set slip in the port shows up as a non-zero diff.
    """
    from tensorlbm.host_staged_halo import D3Q19_C, D3Q19_W

    n, nu = 12, 0.02
    c = torch.tensor(D3Q19_C, dtype=torch.float64)
    w = torch.tensor(D3Q19_W, dtype=torch.float64).view(19, 1, 1, 1)
    shifts = [(int(v[2]), int(v[1]), int(v[0])) for v in D3Q19_C]  # (z, y, x) order

    def feq(rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        cu = torch.einsum("qa,azyx->qzyx", c, u)
        u2 = (u * u).sum(dim=0)
        return w * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * u2)

    f = feq(torch.ones(1, n, n, n, dtype=torch.float64), _tg_velocity(n))
    solver = HostStagedD3Q19(HostStagedLayout(n, n, n), nu=nu, dtype=torch.float64)
    solver.set_velocity(_tg_velocity(n))
    omega = 1.0 / (3 * nu + 0.5)
    for _ in range(3):
        rho = f.sum(dim=0)
        u = torch.einsum("qa,qzyx->azyx", c, f) / rho
        f = f - omega * (f - feq(rho, u))
        f = torch.stack([torch.roll(f[q], shifts[q], dims=(0, 1, 2)) for q in range(19)])
        solver.step(1)
    assert float((solver.slabs[0] - f).abs().max()) == 0.0
