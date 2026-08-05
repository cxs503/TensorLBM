"""Regression equivalence verification: wall function + multi-GPU domain decomposition.

This test suite performs three categories of verification:

1. **Bug identification** — detects known bugs in the original implementation
   (wall_model.py, wall_shear.py, roughness.py, multi_gpu.py) to establish
   whether the original code is "带病上岗" (running with known defects).

2. **Equivalence verification** — verifies that:
   - wall_function_common.wall_function ≡ wall_model.wall_function_3d (D3Q19)
   - CPU single-card ≡ CPU multi-card domain decomposition
   - halo_exchange_3d with .contiguous() ≡ without .contiguous() on CPU
   - MultiGPUSolver3D ≡ MultiDeviceSolver3D

3. **Combination tests** — verifies that wall_function + collision + multi_gpu
   compose correctly and produce results equivalent to the single-card reference.

TDD: tests are written to fail first (red), then pass after verification (green).
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C, W
from tensorlbm.solver3d import stream3d, collide_bgk3d
from tensorlbm.multi_gpu import (
    DomainDecomposition,
    MultiGPUSolver3D,
    MultiDeviceSolver3D,
    halo_exchange_3d,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_domain():
    """A small 3-D D3Q19 domain with a bottom-wall solid mask."""
    nz, ny, nx = 4, 8, 16
    f = equilibrium3d(
        torch.ones(nz, ny, nx),
        torch.full((nz, ny, nx), 0.1),
        torch.zeros(nz, ny, nx),
        torch.zeros(nz, ny, nx),
    )
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool)
    solid[0, :, :] = True  # bottom wall
    return f, solid, nz, ny, nx


@pytest.fixture
def domain_params():
    """Standard simulation parameters."""
    return {"nu": 0.02, "tau": 0.8, "y_val": 0.5}


# ===========================================================================
# 1. BUG IDENTIFICATION
# ===========================================================================

class TestBugIdentification:
    """Identify known bugs in the original implementation."""

    def test_wss_from_fneq_3d_import_error_equilibrium_vs_equilibrium3d(self):
        """BUG: wall_shear.py imports 'equilibrium' from d3q19, but the
        function is named 'equilibrium3d'. This causes an ImportError when
        wss_from_fneq_3d is called, making the entire 3-D WSS function
        non-functional (带病上岗).
        """
        from tensorlbm.wall_shear import wss_from_fneq_3d

        nz, ny, nx = 4, 4, 4
        f = equilibrium3d(
            torch.ones(nz, ny, nx),
            torch.full((nz, ny, nx), 0.1),
            torch.zeros(nz, ny, nx),
            torch.zeros(nz, ny, nx),
        )
        rho, ux, uy, uz = macroscopic3d(f)
        mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
        mask[0, :, :] = True

        with pytest.raises(ImportError, match="cannot import name 'equilibrium'"):
            wss_from_fneq_3d(f, rho, ux, uy, uz, tau=0.8, mask=mask)

    def test_wss_from_fneq_3d_W_to_device_discarded(self):
        """CODE SMELL: wall_shear.py line 192 discards the result of
        _W3D.to(device). The weights tensor is never actually moved to the
        device. This is dead code that does nothing useful.
        """
        import inspect
        from tensorlbm import wall_shear

        source = inspect.getsource(wall_shear.wss_from_fneq_3d)
        # The line "_W3D.to(device)" appears but its result is not assigned
        assert "_W3D.to(device)" in source
        # It should be "_W3D = _W3D.to(device)" or removed entirely
        assert "_W3D = _W3D.to(device)" not in source

    def test_wall_model_y_val_inconsistency(self):
        """INCONSISTENCY: compute_wall_slip_velocity hardcodes y_val=1.5,
        while wall_function_3d defaults to y_val=0.5. These are different
        functions, but the inconsistency is a known design wart.
        """
        import inspect
        from tensorlbm.wall_model import compute_wall_slip_velocity, wall_function_3d

        slip_src = inspect.getsource(compute_wall_slip_velocity)
        wf_src = inspect.getsource(wall_function_3d)

        assert "y_val = 1.5" in slip_src  # hardcoded in slip velocity
        assert "y_val: float = 0.5" in wf_src  # default in wall_function_3d

    def test_wall_function_3d_works_despite_warts(self, small_domain, domain_params):
        """Despite the y_val inconsistency, wall_function_3d itself is
        functional and produces a modified distribution."""
        from tensorlbm.wall_model import wall_function_3d

        f, solid, nz, ny, nx = small_domain
        nu = domain_params["nu"]
        y_val = domain_params["y_val"]

        f_out, drag_fric, drag_pres = wall_function_3d(
            f, solid, nu, y_val=y_val, wall_law="log"
        )

        assert f_out.shape == f.shape
        assert not torch.equal(f, f_out), "wall_function_3d should modify f"
        assert drag_fric > 0, "friction drag should be positive with a wall"
        assert torch.isfinite(f_out).all(), "output should be finite"

    def test_halo_exchange_3d_contiguous_fix_present(self):
        """The .contiguous() fix is present in the current code."""
        import inspect
        from tensorlbm.multi_gpu import halo_exchange_3d

        source = inspect.getsource(halo_exchange_3d)
        assert ".contiguous()" in source, (
            "halo_exchange_3d should contain .contiguous() calls for "
            "non-contiguous slice safety"
        )


# ===========================================================================
# 2. EQUIVALENCE VERIFICATION
# ===========================================================================

class TestWallFunctionEquivalence:
    """Verify wall_function_common ≡ wall_model.wall_function_3d (D3Q19)."""

    def test_wall_function_common_equals_wall_model_3d(
        self, small_domain, domain_params
    ):
        """wall_function_common.wall_function with pre-computed u_tau/y_plus
        must produce identical results to wall_model.wall_function_3d."""
        from tensorlbm.wall_model import wall_function_3d
        from tensorlbm.wall_function_common import (
            wall_function,
            compute_u_tau,
            compute_y_plus,
            _near_wall_mask,
        )

        f, solid, nz, ny, nx = small_domain
        nu = domain_params["nu"]
        y_val = domain_params["y_val"]

        # Original path
        f_orig, _, _ = wall_function_3d(f, solid, nu, y_val=y_val, wall_law="log")

        # Common module path
        rho, ux, uy, uz = macroscopic3d(f)
        u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
        near = _near_wall_mask(solid)
        u_tau = compute_u_tau(u_mag, nu, y_val=y_val, wall_law="log")
        y_plus = compute_y_plus(u_tau, nu, y_val=y_val)
        # Zero out non-near-wall cells (matching wall_function_3d behaviour)
        u_tau_near = u_tau * near.to(u_tau.dtype)
        y_plus_near = y_plus * near.to(y_plus.dtype)
        f_common = wall_function(
            f, solid, u_tau_near, y_plus_near,
            lattice="D3Q19", nu=nu, y_val=y_val,
        )

        assert torch.equal(f_orig, f_common), (
            "wall_function_common.wall_function must be identical to "
            "wall_model.wall_function_3d"
        )

    def test_wall_function_common_reichardt_equals_wall_model_reichardt(
        self, small_domain, domain_params
    ):
        """Same equivalence for the Reichardt wall law."""
        from tensorlbm.wall_model import wall_function_3d
        from tensorlbm.wall_function_common import (
            wall_function,
            compute_u_tau,
            compute_y_plus,
            _near_wall_mask,
        )

        f, solid, nz, ny, nx = small_domain
        nu = domain_params["nu"]
        y_val = domain_params["y_val"]

        f_orig, _, _ = wall_function_3d(f, solid, nu, y_val=y_val, wall_law="reichardt")

        rho, ux, uy, uz = macroscopic3d(f)
        u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
        near = _near_wall_mask(solid)
        u_tau = compute_u_tau(u_mag, nu, y_val=y_val, wall_law="reichardt")
        y_plus = compute_y_plus(u_tau, nu, y_val=y_val)
        u_tau_near = u_tau * near.to(u_tau.dtype)
        y_plus_near = y_plus * near.to(y_plus.dtype)
        f_common = wall_function(
            f, solid, u_tau_near, y_plus_near,
            lattice="D3Q19", nu=nu, y_val=y_val,
        )

        assert torch.equal(f_orig, f_common)


class TestMultiGPUEquivalence:
    """Verify CPU single-card ≡ CPU multi-card domain decomposition."""

    def test_single_vs_multi_2_cards(self, small_domain, domain_params):
        """1 CPU card vs 2 CPU cards: collide+stream must be identical."""
        f, solid, nz, ny, nx = small_domain
        tau = domain_params["tau"]

        # Single-card reference
        f_ref = collide_bgk3d(f, tau)
        f_ref = stream3d(f_ref)

        # Multi-card (2 CPU)
        dd = DomainDecomposition.from_devices([0, 1], nx_global=nx, device_type="cpu")
        solver = MultiGPUSolver3D(f, dd)
        solver.step(lambda f: collide_bgk3d(f, tau), stream3d)
        f_multi = solver.gather()

        assert torch.equal(f_ref, f_multi)

    def test_single_vs_multi_4_cards(self, small_domain, domain_params):
        """1 CPU card vs 4 CPU cards: must be identical."""
        f, solid, nz, ny, nx = small_domain
        tau = domain_params["tau"]

        f_ref = collide_bgk3d(f, tau)
        f_ref = stream3d(f_ref)

        dd = DomainDecomposition.from_devices(
            [0, 1, 2, 3], nx_global=nx, device_type="cpu"
        )
        solver = MultiGPUSolver3D(f, dd)
        solver.step(lambda f: collide_bgk3d(f, tau), stream3d)
        f_multi = solver.gather()

        assert torch.equal(f_ref, f_multi)

    def test_multi_step_single_vs_multi(self, small_domain, domain_params):
        """Multi-step (5 steps) single-card vs multi-card: must be identical."""
        f, solid, nz, ny, nx = small_domain
        tau = domain_params["tau"]
        n_steps = 5

        f_ref = f.clone()
        for _ in range(n_steps):
            f_ref = collide_bgk3d(f_ref, tau)
            f_ref = stream3d(f_ref)

        dd = DomainDecomposition.from_devices([0, 1], nx_global=nx, device_type="cpu")
        solver = MultiGPUSolver3D(f, dd)
        for _ in range(n_steps):
            solver.step(lambda f: collide_bgk3d(f, tau), stream3d)
        f_multi = solver.gather()

        assert torch.equal(f_ref, f_multi)

    def test_multi_gpu_solver_vs_multi_device_solver(self, small_domain, domain_params):
        """MultiGPUSolver3D ≡ MultiDeviceSolver3D on CPU."""
        f, solid, nz, ny, nx = small_domain
        tau = domain_params["tau"]

        dd = DomainDecomposition.from_devices([0, 1], nx_global=nx, device_type="cpu")
        solver_gpu = MultiGPUSolver3D(f, dd)
        solver_gpu.step(lambda f: collide_bgk3d(f, tau), stream3d)
        f_gpu = solver_gpu.gather()

        solver_dev = MultiDeviceSolver3D(
            f, ["cpu", "cpu"],
            collide_fn=lambda f: collide_bgk3d(f, tau),
            stream_fn=stream3d,
        )
        solver_dev.step()
        f_dev = solver_dev.gather()

        assert torch.equal(f_gpu, f_dev)


class TestHaloExchangeContiguousEquivalence:
    """Verify halo_exchange_3d with .contiguous() ≡ without on CPU."""

    @staticmethod
    def _halo_exchange_3d_no_contig(slabs, decomp):
        """halo_exchange_3d without .contiguous() — the pre-fix version."""
        ov = decomp.overlap
        n_slabs = len(slabs)
        for i, slab in enumerate(slabs):
            left = slabs[(i - 1) % n_slabs]
            right = slabs[(i + 1) % n_slabs]
            left_ghost = slab[:, :, :, :ov]
            right_ghost = slab[:, :, :, -ov:]
            left_ghost.copy_(left[:, :, :, -2 * ov:-ov].to(left_ghost.device))
            right_ghost.copy_(right[:, :, :, ov:2 * ov].to(right_ghost.device))
        return slabs

    def test_contiguous_vs_no_contiguous_cpu(self):
        """On CPU, .contiguous() must not change results."""
        nz, ny, nx_local = 2, 4, 6
        ov = 1
        n_slabs = 3
        slabs_a = [torch.randn(19, nz, ny, nx_local + 2 * ov) for _ in range(n_slabs)]
        slabs_b = [s.clone() for s in slabs_a]

        dd = DomainDecomposition(
            devices=["cpu"] * n_slabs, nx_global=nx_local * n_slabs, overlap=ov
        )

        halo_exchange_3d(slabs_a, dd)
        self._halo_exchange_3d_no_contig(slabs_b, dd)

        for i in range(n_slabs):
            assert torch.equal(slabs_a[i], slabs_b[i]), (
                f"Slab {i} differs between contiguous and non-contiguous versions"
            )

    def test_contiguous_vs_no_contiguous_with_streaming(self, small_domain, domain_params):
        """Full step: contiguous vs non-contiguous halo exchange + streaming."""
        f, solid, nz, ny, nx = small_domain
        tau = domain_params["tau"]

        # With .contiguous() (current code)
        dd = DomainDecomposition.from_devices([0, 1], nx_global=nx, device_type="cpu")
        solver_a = MultiGPUSolver3D(f, dd)
        for i, slab in enumerate(solver_a.slabs):
            solver_a.slabs[i] = collide_bgk3d(slab, tau)
        halo_exchange_3d(solver_a.slabs, solver_a.decomp)
        for i, slab in enumerate(solver_a.slabs):
            solver_a.slabs[i] = stream3d(slab)
        f_a = solver_a.gather()

        # Without .contiguous()
        solver_b = MultiGPUSolver3D(f, dd)
        for i, slab in enumerate(solver_b.slabs):
            solver_b.slabs[i] = collide_bgk3d(slab, tau)
        self._halo_exchange_3d_no_contig(solver_b.slabs, solver_b.decomp)
        for i, slab in enumerate(solver_b.slabs):
            solver_b.slabs[i] = stream3d(slab)
        f_b = solver_b.gather()

        assert torch.equal(f_a, f_b)


# ===========================================================================
# 3. COMBINATION TESTS
# ===========================================================================

class TestCombination:
    """Verify wall_function + collision + multi_gpu compose correctly."""

    def test_wall_function_plus_collision_plus_multi_gpu(
        self, small_domain, domain_params
    ):
        """wall_function + collision + multi_gpu: multi-card result must
        equal single-card reference."""
        from tensorlbm.wall_model import wall_function_3d

        f, solid, nz, ny, nx = small_domain
        nu = domain_params["nu"]
        tau = domain_params["tau"]
        y_val = domain_params["y_val"]
        ov = 1

        # Single-card reference: collide -> wall_function -> stream
        f_ref = collide_bgk3d(f, tau)
        f_ref, _, _ = wall_function_3d(f_ref, solid, nu, y_val=y_val, wall_law="log")
        f_ref = stream3d(f_ref)

        # Multi-card: decompose, per-slab collide+wall+halo+stream
        dd = DomainDecomposition.from_devices([0, 1], nx_global=nx, device_type="cpu")
        solver = MultiGPUSolver3D(f, dd)

        # Decompose solid mask to match slabs (with ghost layers via modulo)
        solid_slabs = []
        for dev, (x0, x1) in zip(dd.devices, dd.slabs):
            x_indices = torch.arange(x0 - ov, x1 + ov) % nx
            solid_slab = solid[:, :, x_indices].to(dev).contiguous()
            solid_slabs.append(solid_slab)

        # Per-slab: collide -> wall_function
        for i, slab in enumerate(solver.slabs):
            solver.slabs[i] = collide_bgk3d(slab, tau)
            solver.slabs[i], _, _ = wall_function_3d(
                solver.slabs[i], solid_slabs[i], nu, y_val=y_val, wall_law="log"
            )
        # Halo exchange
        halo_exchange_3d(solver.slabs, solver.decomp)
        # Stream
        for i, slab in enumerate(solver.slabs):
            solver.slabs[i] = stream3d(slab)
        f_multi = solver.gather()

        assert torch.equal(f_ref, f_multi)

    def test_wall_function_common_plus_collision_plus_multi_gpu(
        self, small_domain, domain_params
    ):
        """Same combination but using wall_function_common instead of
        wall_model.wall_function_3d."""
        from tensorlbm.wall_function_common import (
            wall_function,
            compute_u_tau,
            compute_y_plus,
            _near_wall_mask,
        )

        f, solid, nz, ny, nx = small_domain
        nu = domain_params["nu"]
        tau = domain_params["tau"]
        y_val = domain_params["y_val"]
        ov = 1

        def apply_wall_function_common(f_slab, solid_slab):
            """Apply wall_function_common to a slab."""
            rho, ux, uy, uz = macroscopic3d(f_slab)
            u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
            near = _near_wall_mask(solid_slab)
            u_tau = compute_u_tau(u_mag, nu, y_val=y_val, wall_law="log")
            y_plus = compute_y_plus(u_tau, nu, y_val=y_val)
            u_tau_near = u_tau * near.to(u_tau.dtype)
            y_plus_near = y_plus * near.to(y_plus.dtype)
            return wall_function(
                f_slab, solid_slab, u_tau_near, y_plus_near,
                lattice="D3Q19", nu=nu, y_val=y_val,
            )

        # Single-card reference
        f_ref = collide_bgk3d(f, tau)
        f_ref = apply_wall_function_common(f_ref, solid)
        f_ref = stream3d(f_ref)

        # Multi-card
        dd = DomainDecomposition.from_devices([0, 1], nx_global=nx, device_type="cpu")
        solver = MultiGPUSolver3D(f, dd)

        solid_slabs = []
        for dev, (x0, x1) in zip(dd.devices, dd.slabs):
            x_indices = torch.arange(x0 - ov, x1 + ov) % nx
            solid_slab = solid[:, :, x_indices].to(dev).contiguous()
            solid_slabs.append(solid_slab)

        for i, slab in enumerate(solver.slabs):
            solver.slabs[i] = collide_bgk3d(slab, tau)
            solver.slabs[i] = apply_wall_function_common(solver.slabs[i], solid_slabs[i])
        halo_exchange_3d(solver.slabs, solver.decomp)
        for i, slab in enumerate(solver.slabs):
            solver.slabs[i] = stream3d(slab)
        f_multi = solver.gather()

        assert torch.equal(f_ref, f_multi)

    def test_multi_gpu_with_boundary_fn(self, small_domain, domain_params):
        """Multi-GPU solver with boundary_fn (simple Zou-He-like BC)."""
        f, solid, nz, ny, nx = small_domain
        tau = domain_params["tau"]

        def simple_boundary(f_slab):
            """Simple boundary: clamp density at x=0 to equilibrium."""
            # No-op boundary for equivalence test (periodic domain)
            return f_slab

        # Single-card
        f_ref = collide_bgk3d(f, tau)
        f_ref = stream3d(f_ref)
        f_ref = simple_boundary(f_ref)

        # Multi-card with boundary_fn
        dd = DomainDecomposition.from_devices([0, 1], nx_global=nx, device_type="cpu")
        solver = MultiGPUSolver3D(f, dd)
        solver.step(
            lambda f: collide_bgk3d(f, tau),
            stream3d,
            boundary_fn=simple_boundary,
        )
        f_multi = solver.gather()

        assert torch.equal(f_ref, f_multi)

    def test_multi_step_multi_gpu_with_wall_function(
        self, small_domain, domain_params
    ):
        """Multi-step: collide -> wall_function -> stream, repeated 3 steps."""
        from tensorlbm.wall_model import wall_function_3d

        f, solid, nz, ny, nx = small_domain
        nu = domain_params["nu"]
        tau = domain_params["tau"]
        y_val = domain_params["y_val"]
        ov = 1
        n_steps = 3

        # Single-card reference
        f_ref = f.clone()
        for _ in range(n_steps):
            f_ref = collide_bgk3d(f_ref, tau)
            f_ref, _, _ = wall_function_3d(f_ref, solid, nu, y_val=y_val, wall_law="log")
            f_ref = stream3d(f_ref)

        # Multi-card
        dd = DomainDecomposition.from_devices([0, 1], nx_global=nx, device_type="cpu")
        solver = MultiGPUSolver3D(f, dd)

        solid_slabs = []
        for dev, (x0, x1) in zip(dd.devices, dd.slabs):
            x_indices = torch.arange(x0 - ov, x1 + ov) % nx
            solid_slab = solid[:, :, x_indices].to(dev).contiguous()
            solid_slabs.append(solid_slab)

        for _ in range(n_steps):
            for i, slab in enumerate(solver.slabs):
                solver.slabs[i] = collide_bgk3d(slab, tau)
                solver.slabs[i], _, _ = wall_function_3d(
                    solver.slabs[i], solid_slabs[i], nu, y_val=y_val, wall_law="log"
                )
            halo_exchange_3d(solver.slabs, solver.decomp)
            for i, slab in enumerate(solver.slabs):
                solver.slabs[i] = stream3d(slab)
        f_multi = solver.gather()

        assert torch.equal(f_ref, f_multi)


# ===========================================================================
# 4. ROUGHNESS MODULE SANITY
# ===========================================================================

class TestRoughnessSanity:
    """Sanity checks for the roughness module (not equivalence, but
    confirms the module is functional and not 带病上岗)."""

    def test_roughness_b_correction_smooth_regime(self):
        """ks+ < 2.25 → no correction (ΔB = 0)."""
        from tensorlbm.roughness import roughness_b_correction

        ks_plus = torch.tensor([0.5, 1.0, 2.0])
        db = roughness_b_correction(ks_plus)
        assert torch.all(db == 0.0), "Smooth regime should have zero correction"

    def test_roughness_b_correction_fully_rough_regime(self):
        """ks+ > 90 → Colebrook correction (ΔB > 0)."""
        from tensorlbm.roughness import roughness_b_correction

        ks_plus = torch.tensor([100.0, 200.0])
        db = roughness_b_correction(ks_plus)
        assert torch.all(db > 0), "Fully rough regime should have positive correction"

    def test_roughness_b_correction_transitional_regime(self):
        """2.25 ≤ ks+ ≤ 90 → blended correction (0 < ΔB < full_rough)."""
        from tensorlbm.roughness import roughness_b_correction

        ks_plus = torch.tensor([10.0, 50.0])
        db = roughness_b_correction(ks_plus)
        assert torch.all(db > 0), "Transitional should have positive correction"
        assert torch.all(db < 100), "Transitional correction should be bounded"

    def test_compute_rough_wall_slip_velocity_functional(self, small_domain, domain_params):
        """compute_rough_wall_slip_velocity should run without error and
        produce finite output."""
        from tensorlbm.roughness import compute_rough_wall_slip_velocity

        f, solid, nz, ny, nx = small_domain
        nu = domain_params["nu"]
        rho, ux, uy, uz = macroscopic3d(f)

        ux_s, uy_s, uz_s = compute_rough_wall_slip_velocity(
            ux, uy, uz, solid, nu, ks=1.0
        )
        assert ux_s.shape == ux.shape
        assert torch.isfinite(ux_s).all()
        assert torch.isfinite(uy_s).all()
        assert torch.isfinite(uz_s).all()


# ===========================================================================
# 5. WALL SHEAR MODULE BUG DETAIL
# ===========================================================================

class TestWallShearBugs:
    """Detailed bug verification for wall_shear.py."""

    def test_wss_from_fneq_2d_functional(self):
        """wss_from_fneq_2d should work (d2q9 has 'equilibrium')."""
        from tensorlbm.d2q9 import equilibrium, macroscopic
        from tensorlbm.wall_shear import wss_from_fneq_2d

        ny, nx = 8, 8
        rho = torch.ones(ny, nx)
        ux = torch.full((ny, nx), 0.1)
        uy = torch.zeros(ny, nx)
        f = equilibrium(rho, ux, uy)
        mask = torch.zeros(ny, nx, dtype=torch.bool)
        mask[0, :] = True

        wss = wss_from_fneq_2d(f, rho, ux, uy, tau=0.8, mask=mask)
        assert wss.shape == (ny, nx)
        assert torch.isfinite(wss).all()

    def test_wss_from_velocity_2d_functional(self):
        """wss_from_velocity_2d (FD method) should work."""
        from tensorlbm.wall_shear import wss_from_velocity_2d

        ny, nx = 8, 8
        ux = torch.full((ny, nx), 0.1)
        uy = torch.zeros(ny, nx)
        mask = torch.zeros(ny, nx, dtype=torch.bool)
        mask[0, :] = True

        wss = wss_from_velocity_2d(ux, uy, mask, nu=0.02)
        assert wss.shape == (ny, nx)
        assert torch.isfinite(wss).all()

    def test_wss_from_fneq_3d_broken_import(self):
        """CONFIRMED BUG: wss_from_fneq_3d is non-functional due to
        importing 'equilibrium' instead of 'equilibrium3d' from d3q19."""
        from tensorlbm.wall_shear import wss_from_fneq_3d

        nz, ny, nx = 4, 4, 4
        f = equilibrium3d(
            torch.ones(nz, ny, nx),
            torch.full((nz, ny, nx), 0.1),
            torch.zeros(nz, ny, nx),
            torch.zeros(nz, ny, nx),
        )
        rho, ux, uy, uz = macroscopic3d(f)
        mask = torch.zeros(nz, ny, nx, dtype=torch.bool)

        with pytest.raises(ImportError):
            wss_from_fneq_3d(f, rho, ux, uy, uz, tau=0.8, mask=mask)
