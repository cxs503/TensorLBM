"""TDD tests for the device-agnostic multi-card domain-decomposition module.

Validates that ``DomainDecomposition``, ``halo_exchange_3d``,
``auto_decompose`` and the new ``MultiDeviceSolver3D`` common class work
with arbitrary device strings (cuda / sdaa / cpu) and that the per-card
collision / streaming / boundary kernels run unmodified on each device.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.multi_gpu import (
    DomainDecomposition,
    MultiDeviceSolver3D,
    auto_decompose,
    halo_exchange_3d,
)
from tensorlbm.solver3d import collide_bgk3d, stream3d


# --------------------------------------------------------------------------- #
# Device-availability helpers
# --------------------------------------------------------------------------- #

def _sdaa_available() -> bool:
    return hasattr(torch, "sdaa") and torch.sdaa.is_available()


def _sdaa_count() -> int:
    return torch.sdaa.device_count() if _sdaa_available() else 0


skip_no_sdaa = pytest.mark.skipif(
    not _sdaa_available(), reason="no SDAA backend available"
)
skip_few_sdaa = pytest.mark.skipif(
    _sdaa_count() < 2, reason="fewer than 2 SDAA devices"
)
skip_few_sdaa4 = pytest.mark.skipif(
    _sdaa_count() < 4, reason="fewer than 4 SDAA devices"
)


# D3Q19 velocity set (for reference pull-stream in tests)
D3Q19_C = (
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
    (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
    (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
)


def _pull_stream(f: torch.Tensor) -> torch.Tensor:
    """Periodic pull stream expressed as per-population tensor rolls."""
    out = torch.empty_like(f)
    for q, (cx, cy, cz) in enumerate(D3Q19_C):
        out[q] = torch.roll(f[q], shifts=(cz, cy, cx), dims=(0, 1, 2))
    return out


def _equilibrium_global(nz: int, ny: int, nx: int, u: float, device: str, dtype: torch.dtype):
    """Uniform-equilibrium global distribution on *device*."""
    rho = torch.ones((nz, ny, nx), dtype=dtype, device=device)
    ux = torch.full((nz, ny, nx), u, dtype=dtype, device=device)
    uy = torch.zeros((nz, ny, nx), dtype=dtype, device=device)
    uz = torch.zeros((nz, ny, nx), dtype=dtype, device=device)
    return equilibrium3d(rho, ux, uy, uz)


# --------------------------------------------------------------------------- #
# 1. DomainDecomposition — device-agnostic
# --------------------------------------------------------------------------- #

class TestDomainDecompositionDeviceAgnostic:
    """DomainDecomposition must accept arbitrary device strings."""

    def test_constructor_accepts_sdaa_strings(self):
        dd = DomainDecomposition(devices=["sdaa:0", "sdaa:1"], nx_global=8)
        assert dd.n_devices == 2
        assert dd.devices == ["sdaa:0", "sdaa:1"]
        assert dd.slabs == [(0, 4), (4, 8)]

    def test_constructor_accepts_mixed_strings(self):
        dd = DomainDecomposition(devices=["cpu", "sdaa:0"], nx_global=6)
        assert dd.n_devices == 2
        assert dd.slabs == [(0, 3), (3, 6)]

    def test_from_devices_cuda_default(self):
        dd = DomainDecomposition.from_devices([0, 1], nx_global=8)
        assert dd.devices == ["cuda:0", "cuda:1"]

    def test_from_devices_sdaa(self):
        dd = DomainDecomposition.from_devices(
            [0, 1, 2], nx_global=9, device_type="sdaa",
        )
        assert dd.devices == ["sdaa:0", "sdaa:1", "sdaa:2"]
        assert dd.slabs == [(0, 3), (3, 6), (6, 9)]

    def test_from_devices_cpu(self):
        dd = DomainDecomposition.from_devices(
            [0, 1], nx_global=8, device_type="cpu",
        )
        assert dd.devices == ["cpu", "cpu"]

    def test_balanced_slabs_with_remainder(self):
        dd = DomainDecomposition(devices=["cpu"] * 3, nx_global=10)
        widths = [e - s for s, e in dd.slabs]
        assert widths == [4, 3, 3]  # remainder distributed to first slabs


# --------------------------------------------------------------------------- #
# 2. auto_decompose — SDAA detection
# --------------------------------------------------------------------------- #

class TestAutoDecomposeSdaa:
    """auto_decompose must detect SDAA when CUDA is unavailable."""

    def test_explicit_sdaa_type(self):
        f = torch.zeros(19, 2, 2, 8)
        dd = auto_decompose(f, device_type="sdaa", n_devices=2)
        assert dd.devices == ["sdaa:0", "sdaa:1"]
        assert dd.nx_global == 8

    def test_explicit_cpu_type(self):
        f = torch.zeros(19, 2, 2, 6)
        dd = auto_decompose(f, device_type="cpu", n_devices=0)
        assert dd.devices == ["cpu"]

    @skip_no_sdaa
    def test_auto_detect_sdaa_when_no_cuda(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA is available — cannot test SDAA auto-detect")
        f = torch.zeros(19, 2, 2, 8)
        dd = auto_decompose(f)
        assert all(d.startswith("sdaa:") for d in dd.devices)
        assert dd.n_devices == _sdaa_count()

    def test_n_devices_override(self):
        f = torch.zeros(19, 2, 2, 8)
        dd = auto_decompose(f, device_type="sdaa", n_devices=4)
        assert len(dd.devices) == 4
        assert dd.devices == [f"sdaa:{i}" for i in range(4)]


# --------------------------------------------------------------------------- #
# 3. halo_exchange_3d — cross-SDAA communication
# --------------------------------------------------------------------------- #

@skip_few_sdaa
class TestHaloExchange3DCrossSdaa:
    """halo_exchange_3d must copy ghost planes between SDAA devices."""

    def test_two_sdaa_slabs_exchange_ghosts(self):
        ov = 1
        nz, ny, nx_local = 3, 4, 5
        # Slab 0 on sdaa:0, slab 1 on sdaa:1
        s0 = torch.randn(19, nz, ny, nx_local + 2 * ov, device="sdaa:0")
        s1 = torch.randn(19, nz, ny, nx_local + 2 * ov, device="sdaa:1")
        dd = DomainDecomposition(
            devices=["sdaa:0", "sdaa:1"], nx_global=2 * nx_local, overlap=ov,
        )
        slabs = [s0, s1]
        halo_exchange_3d(slabs, dd)

        # Right ghost of s0 must equal interior-left of s1
        expected_right_ghost = s1[:, :, :, ov:2 * ov].to("sdaa:0")
        assert torch.equal(slabs[0][:, :, :, -ov:], expected_right_ghost)

        # Left ghost of s1 must equal interior-right of s0
        expected_left_ghost = s0[:, :, :, -2 * ov:-ov].to("sdaa:1")
        assert torch.equal(slabs[1][:, :, :, :ov], expected_left_ghost)

    def test_three_sdaa_slabs_periodic_ring(self):
        ov = 1
        nz, ny, nx_local = 2, 3, 4
        devices = ["sdaa:0", "sdaa:1", "sdaa:2"]
        slabs = [
            torch.randn(19, nz, ny, nx_local + 2 * ov, device=d)
            for d in devices
        ]
        dd = DomainDecomposition(
            devices=devices, nx_global=3 * nx_local, overlap=ov,
        )
        # snapshot interior-right and interior-left before exchange
        right_interiors = [s[:, :, :, -2 * ov:-ov].clone() for s in slabs]
        left_interiors = [s[:, :, :, ov:2 * ov].clone() for s in slabs]

        halo_exchange_3d(slabs, dd)

        n = len(slabs)
        for i in range(n):
            left_src = right_interiors[(i - 1) % n].to(slabs[i].device)
            right_src = left_interiors[(i + 1) % n].to(slabs[i].device)
            assert torch.equal(slabs[i][:, :, :, :ov], left_src)
            assert torch.equal(slabs[i][:, :, :, -ov:], right_src)


# --------------------------------------------------------------------------- #
# 4. MultiDeviceSolver3D — contracts
# --------------------------------------------------------------------------- #

class TestMultiDeviceSolver3DContracts:
    """The common multi-device solver class must accept functions + device list."""

    def test_constructor_accepts_device_list_and_functions(self):
        f = _equilibrium_global(3, 4, 8, 0.0, "cpu", torch.float64)
        solver = MultiDeviceSolver3D(
            f_global=f,
            devices=["cpu", "cpu"],
            collide_fn=lambda f: f,
            stream_fn=_pull_stream,
        )
        assert solver.n_devices == 2
        assert solver.decomp.nx_global == 8

    def test_step_returns_none_without_force_fn(self):
        f = _equilibrium_global(3, 4, 8, 0.0, "cpu", torch.float64)
        solver = MultiDeviceSolver3D(
            f_global=f,
            devices=["cpu", "cpu"],
            collide_fn=lambda f: f,
            stream_fn=_pull_stream,
        )
        result = solver.step()
        assert result is None

    def test_step_returns_aggregated_force_with_force_fn(self):
        f = _equilibrium_global(3, 4, 8, 0.0, "cpu", torch.float64)
        # force_fn returns total mass of the slab (sum of all populations)
        def force_fn(slab):
            return slab.sum().reshape(1)
        solver = MultiDeviceSolver3D(
            f_global=f,
            devices=["cpu", "cpu"],
            collide_fn=lambda f: f,
            stream_fn=_pull_stream,
            force_fn=force_fn,
        )
        result = solver.step()
        assert result is not None
        # Total mass should be preserved (equilibrium at rest, periodic stream)
        expected = f.sum()
        assert torch.allclose(result.cpu(), expected.cpu().reshape(1), rtol=1e-10)

    def test_gather_reconstructs_global_shape(self):
        f = _equilibrium_global(3, 4, 8, 0.0, "cpu", torch.float64)
        solver = MultiDeviceSolver3D(
            f_global=f,
            devices=["cpu", "cpu"],
            collide_fn=lambda f: f,
            stream_fn=_pull_stream,
        )
        g = solver.gather()
        assert g.shape == f.shape
        assert g.dtype == f.dtype

    def test_gather_macroscopic_returns_global_fields(self):
        f = _equilibrium_global(3, 4, 8, 0.1, "cpu", torch.float64)
        solver = MultiDeviceSolver3D(
            f_global=f,
            devices=["cpu", "cpu"],
            collide_fn=lambda f: f,
            stream_fn=_pull_stream,
        )
        rho, ux, uy, uz = solver.gather_macroscopic()
        assert rho.shape == (3, 4, 8)
        # Weights W are stored as float32, so expect ~1e-8 rounding.
        assert torch.allclose(rho, torch.ones_like(rho), atol=1e-7)
        assert torch.allclose(ux, torch.full_like(ux, 0.1), atol=1e-7)


# --------------------------------------------------------------------------- #
# 5. MultiDeviceSolver3D — equivalence with monolithic solver
# --------------------------------------------------------------------------- #

class TestMultiDeviceSolver3DEquivalence:
    """Multi-card periodic stream must match monolithic periodic stream."""

    @pytest.mark.parametrize("n_devices", [2, 3, 4])
    def test_multi_card_matches_monolithic_periodic_stream(self, n_devices):
        torch.manual_seed(20260716)
        nz, ny, nx = 3, 4, 12
        initial = torch.randn(19, nz, ny, nx, dtype=torch.float64)

        # Monolithic reference (periodic)
        expected = initial.clone()
        for _ in range(3):
            expected = _pull_stream(expected)

        # Multi-card
        solver = MultiDeviceSolver3D(
            f_global=initial,
            devices=["cpu"] * n_devices,
            collide_fn=lambda f: f,
            stream_fn=_pull_stream,
        )
        for _ in range(3):
            solver.step()
        actual = solver.gather()

        mismatch = (actual != expected).sum(dim=(1, 2, 3))
        assert torch.equal(
            mismatch, torch.zeros(19, dtype=mismatch.dtype),
        ), f"directions with mismatches: {mismatch.tolist()}"

    def test_multi_card_with_bgk_collision_matches_monolithic(self):
        """BGK collision + periodic stream must match across 4 cards."""
        torch.manual_seed(20260716)
        nz, ny, nx = 3, 4, 12
        rho0 = torch.ones((nz, ny, nx), dtype=torch.float64)
        ux0 = torch.full((nz, ny, nx), 0.05, dtype=torch.float64)
        uy0 = torch.zeros((nz, ny, nx), dtype=torch.float64)
        uz0 = torch.zeros((nz, ny, nx), dtype=torch.float64)
        initial = equilibrium3d(rho0, ux0, uy0, uz0)
        # Add perturbation
        initial = initial + 0.01 * torch.randn_like(initial)

        tau = 0.8

        # Monolithic reference
        expected = initial.clone()
        for _ in range(2):
            expected = collide_bgk3d(expected, tau)
            expected = _pull_stream(expected)

        # Multi-card (4 CPU "cards")
        solver = MultiDeviceSolver3D(
            f_global=initial,
            devices=["cpu"] * 4,
            collide_fn=lambda f: collide_bgk3d(f, tau),
            stream_fn=_pull_stream,
        )
        for _ in range(2):
            solver.step()
        actual = solver.gather()

        assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_force_all_reduce_matches_global(self):
        """Sum of per-card forces must equal the global force."""
        torch.manual_seed(20260716)
        nz, ny, nx = 3, 4, 12
        initial = torch.randn(19, nz, ny, nx, dtype=torch.float64)

        # Global force: total x-momentum
        c = torch.tensor(D3Q19_C, dtype=torch.float64)
        cx = c[:, 0].view(19, 1, 1, 1)
        global_force = (initial * cx).sum().reshape(1)

        def force_fn(slab):
            return (slab * cx.to(slab.device)).sum().reshape(1)

        solver = MultiDeviceSolver3D(
            f_global=initial,
            devices=["cpu"] * 4,
            collide_fn=lambda f: f,
            stream_fn=_pull_stream,
            force_fn=force_fn,
        )
        result = solver.step()
        assert torch.allclose(result.cpu(), global_force.cpu().reshape(1), rtol=1e-10)


# --------------------------------------------------------------------------- #
# 6. SDAA cross-device integration
# --------------------------------------------------------------------------- #

@skip_few_sdaa
class TestMultiDeviceSolver3DSdaaIntegration:
    """MultiDeviceSolver3D must run on real SDAA hardware."""

    def test_two_sdaa_cards_periodic_stream_equivalence(self):
        torch.manual_seed(20260716)
        nz, ny, nx = 3, 4, 12
        initial = torch.randn(19, nz, ny, nx, dtype=torch.float32)

        # Monolithic reference on CPU
        expected = initial.clone()
        for _ in range(2):
            expected = _pull_stream(expected)

        solver = MultiDeviceSolver3D(
            f_global=initial,
            devices=["sdaa:0", "sdaa:1"],
            collide_fn=lambda f: f,
            stream_fn=_pull_stream,
        )
        for _ in range(2):
            solver.step()
        actual = solver.gather()

        assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_four_sdaa_cards_bgk_collision_equivalence(self):
        torch.manual_seed(20260716)
        nz, ny, nx = 4, 5, 16
        rho0 = torch.ones((nz, ny, nx), dtype=torch.float32)
        ux0 = torch.full((nz, ny, nx), 0.05, dtype=torch.float32)
        initial = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))
        initial = initial + 0.01 * torch.randn_like(initial)
        tau = 0.8

        # Monolithic reference on CPU
        expected = initial.clone()
        for _ in range(2):
            expected = collide_bgk3d(expected, tau)
            expected = _pull_stream(expected)

        solver = MultiDeviceSolver3D(
            f_global=initial,
            devices=[f"sdaa:{i}" for i in range(4)],
            collide_fn=lambda f: collide_bgk3d(f, tau),
            stream_fn=_pull_stream,
        )
        for _ in range(2):
            solver.step()
        actual = solver.gather()

        assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_sdaa_force_all_reduce(self):
        """Force all-reduce across SDAA cards matches global."""
        torch.manual_seed(20260716)
        nz, ny, nx = 3, 4, 16
        initial = torch.randn(19, nz, ny, nx, dtype=torch.float32)
        c = torch.tensor(D3Q19_C, dtype=torch.float32)
        cx = c[:, 0].view(19, 1, 1, 1)
        global_force = (initial * cx).sum().reshape(1)

        def force_fn(slab):
            return (slab * cx.to(slab.device)).sum().reshape(1)

        solver = MultiDeviceSolver3D(
            f_global=initial,
            devices=[f"sdaa:{i}" for i in range(4)],
            collide_fn=lambda f: f,
            stream_fn=_pull_stream,
            force_fn=force_fn,
        )
        result = solver.step()
        assert torch.allclose(result.cpu(), global_force.cpu().reshape(1), rtol=1e-4, atol=1e-5)

    def test_sdaa_gather_macroscopic(self):
        nz, ny, nx = 3, 4, 16
        rho0 = torch.ones((nz, ny, nx), dtype=torch.float32)
        ux0 = torch.full((nz, ny, nx), 0.1, dtype=torch.float32)
        initial = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))

        solver = MultiDeviceSolver3D(
            f_global=initial,
            devices=[f"sdaa:{i}" for i in range(4)],
            collide_fn=lambda f: f,
            stream_fn=_pull_stream,
        )
        rho, ux, uy, uz = solver.gather_macroscopic()
        assert rho.shape == (nz, ny, nx)
        assert torch.allclose(rho, torch.ones_like(rho), rtol=1e-4, atol=1e-5)
        assert torch.allclose(ux, torch.full_like(ux, 0.1), rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------- #
# 7. SUBOFF-scale validation (480×240×240 on 4 or 8 cards)
# --------------------------------------------------------------------------- #

@skip_few_sdaa4
class TestSuboffScaleMultiCardValidation:
    """Validate domain decomposition on a SUBOFF-scale 480×240×240 grid.

    Uses 4 SDAA cards (120 cells/card).  Verifies that the multi-card
    periodic stream matches the monolithic reference and that force
    all-reduce produces the correct global value.
    """

    @pytest.mark.slow
    def test_480x240x240_4card_periodic_stream_equivalence(self):
        nz, ny, nx = 240, 240, 480
        n_cards = 4
        dtype = torch.float32

        # Uniform equilibrium at rest (no perturbation → exact conservation)
        rho0 = torch.ones((nz, ny, nx), dtype=dtype)
        ux0 = torch.full((nz, ny, nx), 0.0, dtype=dtype)
        initial = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))

        # Monolithic reference: 1 step of periodic stream on CPU
        expected = _pull_stream(initial)

        solver = MultiDeviceSolver3D(
            f_global=initial,
            devices=[f"sdaa:{i}" for i in range(n_cards)],
            collide_fn=lambda f: f,
            stream_fn=_pull_stream,
        )
        solver.step()
        actual = solver.gather()

        assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.slow
    def test_480x240x240_4card_force_all_reduce(self):
        nz, ny, nx = 240, 240, 480
        n_cards = 4
        dtype = torch.float32

        rho0 = torch.ones((nz, ny, nx), dtype=dtype)
        ux0 = torch.full((nz, ny, nx), 0.05, dtype=dtype)
        initial = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))

        c = torch.tensor(D3Q19_C, dtype=dtype)
        cx = c[:, 0].view(19, 1, 1, 1)
        global_force = (initial * cx).sum().reshape(1)

        def force_fn(slab):
            return (slab * cx.to(slab.device)).sum().reshape(1)

        solver = MultiDeviceSolver3D(
            f_global=initial,
            devices=[f"sdaa:{i}" for i in range(n_cards)],
            collide_fn=lambda f: f,
            stream_fn=_pull_stream,
            force_fn=force_fn,
        )
        result = solver.step()
        assert torch.allclose(result.cpu(), global_force.cpu().reshape(1), rtol=1e-3, atol=1e-4)

    @pytest.mark.slow
    def test_480x240x240_8card_bgk_equivalence(self):
        nz, ny, nx = 240, 240, 480
        n_cards = 8
        dtype = torch.float32

        if _sdaa_count() < 8:
            pytest.skip("fewer than 8 SDAA devices")

        rho0 = torch.ones((nz, ny, nx), dtype=dtype)
        ux0 = torch.full((nz, ny, nx), 0.05, dtype=dtype)
        initial = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))

        tau = 0.8
        # Monolithic reference: 1 step collide + stream on CPU
        expected = collide_bgk3d(initial, tau)
        expected = _pull_stream(expected)

        solver = MultiDeviceSolver3D(
            f_global=initial,
            devices=[f"sdaa:{i}" for i in range(n_cards)],
            collide_fn=lambda f: collide_bgk3d(f, tau),
            stream_fn=_pull_stream,
        )
        solver.step()
        actual = solver.gather()

        assert torch.allclose(actual, expected, rtol=1e-3, atol=1e-4)
