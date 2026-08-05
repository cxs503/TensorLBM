"""TDD tests for LBMStepExecutor — numerical equivalence + performance.

Verifies that LBMStepExecutor produces results that are allclose(atol=1e-6)
with the original manual-loop implementations, across:

  * D3Q19 and D3Q27 lattices
  * BGK, MRT, and Cumulant collision operators
  * With and without wall function
  * With far-field and bounce-back boundaries
  * With and without mass correction

Also includes a performance comparison (MLUPS) between the executor and
the manual loop.
"""
from __future__ import annotations

import time

import pytest
import torch

from tensorlbm.d3q19 import (
    C as C19,
    W as W19,
    equilibrium3d,
    macroscopic3d,
)
from tensorlbm.d3q27 import (
    C as C27,
    W as W27,
    collide_bgk27,
    collide_mrt27,
    equilibrium27,
    macroscopic27,
    stream27_roll,
)
from tensorlbm.solver3d import (
    collide_bgk3d,
    collide_mrt3d,
    stream3d,
    correct_mass3d,
)
from tensorlbm.cumulant import collide_cumulant_d3q19, collide_cumulant_d3q27
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.boundaries_d3q27 import far_field_bc_27, bounce_back_cells_27
from tensorlbm.wall_function_common import (
    compute_u_tau,
    compute_y_plus,
    wall_function,
)
from tensorlbm.lbm_step import LBMStepExecutor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SDAA = "sdaa:0"
DEVICE = torch.device(SDAA if torch.sdaa.is_available() else "cpu")
DTYPE = torch.float32
ATOL = 1e-6

NZ, NY, NX = 16, 16, 16


def _make_f(Q: int, seed: int = 42) -> torch.Tensor:
    """Create a valid (positive, rho≈1) distribution for testing."""
    torch.manual_seed(seed)
    # Start from equilibrium at rho=1, u=0.05
    nz, ny, nx = NZ, NY, NX
    rho0 = torch.ones(nz, ny, nx, device=DEVICE, dtype=DTYPE)
    ux0 = torch.full_like(rho0, 0.05)
    uy0 = torch.full_like(rho0, 0.02)
    uz0 = torch.full_like(rho0, 0.01)
    if Q == 19:
        f = equilibrium3d(rho0, ux0, uy0, uz0, device=DEVICE)
    else:
        f = equilibrium27(rho0, ux0, uy0, uz0, device=DEVICE)
    # Add small perturbation
    f = f + 0.001 * torch.randn_like(f)
    return f


def _make_mask() -> torch.Tensor:
    """Create a simple obstacle mask (small sphere in the centre)."""
    nz, ny, nx = NZ, NY, NX
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=DEVICE),
        torch.arange(ny, device=DEVICE),
        torch.arange(nx, device=DEVICE),
        indexing="ij",
    )
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    r = min(nz, ny, nx) // 6
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < r * r
    return mask


# ---------------------------------------------------------------------------
# 1. Macroscopic equivalence
# ---------------------------------------------------------------------------

class TestMacroscopicEquivalence:
    """_compute_macroscopic_inplace must match macroscopic3d/macroscopic27."""

    @pytest.mark.parametrize("lattice,Q", [("D3Q19", 19), ("D3Q27", 27)])
    def test_macroscopic_allclose(self, lattice, Q):
        f = _make_f(Q)
        ex = LBMStepExecutor(
            lattice, collide_fn="bgk", device=DEVICE,
            nx=NX, ny=NY, nz=NZ, tau=0.6,
        )
        rho_e, ux_e, uy_e, uz_e = ex._compute_macroscopic_inplace(f)

        if lattice == "D3Q19":
            rho_r, ux_r, uy_r, uz_r = macroscopic3d(f)
        else:
            rho_r, ux_r, uy_r, uz_r = macroscopic27(f)

        assert torch.allclose(rho_e, rho_r, atol=ATOL), f"rho mismatch ({lattice})"
        assert torch.allclose(ux_e, ux_r, atol=ATOL), f"ux mismatch ({lattice})"
        assert torch.allclose(uy_e, uy_r, atol=ATOL), f"uy mismatch ({lattice})"
        assert torch.allclose(uz_e, uz_r, atol=ATOL), f"uz mismatch ({lattice})"


# ---------------------------------------------------------------------------
# 2. Equilibrium equivalence
# ---------------------------------------------------------------------------

class TestEquilibriumEquivalence:
    """_compute_equilibrium_inplace must match equilibrium3d/equilibrium27."""

    @pytest.mark.parametrize("lattice,Q", [("D3Q19", 19), ("D3Q27", 27)])
    def test_equilibrium_allclose(self, lattice, Q):
        f = _make_f(Q)
        ex = LBMStepExecutor(
            lattice, collide_fn="bgk", device=DEVICE,
            nx=NX, ny=NY, nz=NZ, tau=0.6,
        )
        rho, ux, uy, uz = ex._compute_macroscopic_inplace(f)
        feq_e = ex._compute_equilibrium_inplace(rho, ux, uy, uz)

        if lattice == "D3Q19":
            feq_r = equilibrium3d(rho, ux, uy, uz, device=DEVICE)
        else:
            feq_r = equilibrium27(rho, ux, uy, uz, device=DEVICE)

        assert torch.allclose(feq_e, feq_r, atol=ATOL), f"feq mismatch ({lattice})"


# ---------------------------------------------------------------------------
# 3. BGK collision equivalence
# ---------------------------------------------------------------------------

class TestCollideBGKEquivalence:
    """_collide_bgk must match collide_bgk3d/collide_bgk27."""

    @pytest.mark.parametrize("lattice,Q", [("D3Q19", 19), ("D3Q27", 27)])
    @pytest.mark.parametrize("tau", [0.6, 0.8, 1.0])
    def test_collide_bgk_allclose(self, lattice, Q, tau):
        f = _make_f(Q)
        ex = LBMStepExecutor(
            lattice, collide_fn="bgk", device=DEVICE,
            nx=NX, ny=NY, nz=NZ, tau=tau,
        )
        f_post_e = ex._collide_bgk(f)

        if lattice == "D3Q19":
            f_post_r = collide_bgk3d(f, tau)
        else:
            f_post_r = collide_bgk27(f, tau)

        assert torch.allclose(f_post_e, f_post_r, atol=ATOL), (
            f"BGK collide mismatch ({lattice}, tau={tau})"
        )


# ---------------------------------------------------------------------------
# 4. Streaming equivalence
# ---------------------------------------------------------------------------

class TestStreamEquivalence:
    """_stream_preallocated must match stream3d/stream27_roll."""

    @pytest.mark.parametrize("lattice,Q", [("D3Q19", 19), ("D3Q27", 27)])
    def test_stream_allclose(self, lattice, Q):
        f = _make_f(Q)
        ex = LBMStepExecutor(
            lattice, collide_fn="bgk", device=DEVICE,
            nx=NX, ny=NY, nz=NZ, tau=0.6,
        )
        f_streamed_e = ex._stream_preallocated(f)
        # Clone because _stream_preallocated writes into a reusable buffer
        f_streamed_e = f_streamed_e.clone()

        if lattice == "D3Q19":
            f_streamed_r = stream3d(f)
        else:
            f_streamed_r = stream27_roll(f)

        assert torch.allclose(f_streamed_e, f_streamed_r, atol=ATOL), (
            f"stream mismatch ({lattice})"
        )


# ---------------------------------------------------------------------------
# 5. Wall function equivalence
# ---------------------------------------------------------------------------

class TestWallFunctionEquivalence:
    """_apply_wall_function must match the manual wall_function pipeline."""

    @pytest.mark.parametrize("lattice,Q", [("D3Q19", 19), ("D3Q27", 27)])
    def test_wall_function_allclose(self, lattice, Q):
        f = _make_f(Q)
        mask = _make_mask()
        nu = 0.02
        y_val = 0.5

        ex = LBMStepExecutor(
            lattice, collide_fn="bgk", device=DEVICE,
            nx=NX, ny=NY, nz=NZ, tau=0.6,
            wall_fn=True, mask=mask, nu=nu, y_val=y_val,
        )
        # Compute macroscopic once
        rho, ux, uy, uz = ex._compute_macroscopic_inplace(f)
        # Apply executor's wall function
        f_e = ex._apply_wall_function(f.clone(), rho, ux, uy, uz)
        f_e = f_e.clone()

        # Manual pipeline: compute u_tau, y_plus, then wall_function
        u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
        u_tau = compute_u_tau(u_mag, nu, y_val, "log")
        y_plus = compute_y_plus(u_tau, nu, y_val)
        f_r = wall_function(
            f.clone(), mask, u_tau, y_plus,
            lattice=lattice, nu=nu, y_val=y_val,
        )

        assert torch.allclose(f_e, f_r, atol=ATOL), (
            f"wall_function mismatch ({lattice})"
        )


# ---------------------------------------------------------------------------
# 6. Full-step equivalence: BGK + stream + boundary
# ---------------------------------------------------------------------------

class TestFullStepEquivalence:
    """Full step() must match the manual collide→stream→boundary loop."""

    @pytest.mark.parametrize("lattice,Q", [("D3Q19", 19), ("D3Q27", 27)])
    def test_full_step_bgk_farfield(self, lattice, Q):
        """BGK + stream + far_field boundary."""
        f0 = _make_f(Q)
        tau = 0.6
        u_in = 0.05
        mask = _make_mask()

        # --- Manual loop (reference) ---
        f_ref = f0.clone()
        if lattice == "D3Q19":
            f_ref = collide_bgk3d(f_ref, tau)
            f_ref = stream3d(f_ref)
            f_ref = far_field_bc_3d(f_ref, u_in=u_in, obstacle_mask=mask)
        else:
            f_ref = collide_bgk27(f_ref, tau)
            f_ref = stream27_roll(f_ref)
            f_ref = far_field_bc_27(f_ref, u_in=u_in, obstacle_mask=mask)

        # --- Executor ---
        if lattice == "D3Q19":
            bfn = far_field_bc_3d
        else:
            bfn = far_field_bc_27
        ex = LBMStepExecutor(
            lattice, collide_fn="bgk", device=DEVICE,
            nx=NX, ny=NY, nz=NZ, tau=tau,
            boundary_fn=bfn,
            boundary_kwargs={"u_in": u_in, "obstacle_mask": mask},
        )
        f_ex, _ = ex.step(f0.clone())

        assert torch.allclose(f_ex, f_ref, atol=ATOL), (
            f"full step (farfield) mismatch ({lattice})"
        )

    @pytest.mark.parametrize("lattice,Q", [("D3Q19", 19), ("D3Q27", 27)])
    def test_full_step_bgk_bounceback(self, lattice, Q):
        """BGK + stream + bounce_back boundary."""
        f0 = _make_f(Q)
        tau = 0.6
        mask = _make_mask()

        # --- Manual loop (reference) ---
        f_ref = f0.clone()
        if lattice == "D3Q19":
            f_ref = collide_bgk3d(f_ref, tau)
            f_ref = stream3d(f_ref)
            f_ref = bounce_back_cells_3d(f_ref, mask)
        else:
            f_ref = collide_bgk27(f_ref, tau)
            f_ref = stream27_roll(f_ref)
            f_ref = bounce_back_cells_27(f_ref, mask)

        # --- Executor ---
        if lattice == "D3Q19":
            bfn = bounce_back_cells_3d
        else:
            bfn = bounce_back_cells_27
        ex = LBMStepExecutor(
            lattice, collide_fn="bgk", device=DEVICE,
            nx=NX, ny=NY, nz=NZ, tau=tau,
            boundary_fn=bfn,
            boundary_kwargs={"mask": mask},
        )
        f_ex, _ = ex.step(f0.clone())

        assert torch.allclose(f_ex, f_ref, atol=ATOL), (
            f"full step (bounceback) mismatch ({lattice})"
        )

    @pytest.mark.parametrize("lattice,Q", [("D3Q19", 19), ("D3Q27", 27)])
    def test_full_step_bgk_wall_function(self, lattice, Q):
        """BGK + stream + bounce_back + wall function."""
        f0 = _make_f(Q)
        tau = 0.6
        mask = _make_mask()
        nu = 0.02
        y_val = 0.5

        # --- Manual loop (reference) ---
        f_ref = f0.clone()
        if lattice == "D3Q19":
            f_ref = collide_bgk3d(f_ref, tau)
            f_ref = stream3d(f_ref)
            f_ref = bounce_back_cells_3d(f_ref, mask)
            # Wall function pipeline
            rho, ux, uy, uz = macroscopic3d(f_ref)
            u_mag = torch.sqrt(ux*ux + uy*uy + uz*uz).clamp(min=1e-12)
            u_tau = compute_u_tau(u_mag, nu, y_val, "log")
            y_plus = compute_y_plus(u_tau, nu, y_val)
            f_ref = wall_function(f_ref, mask, u_tau, y_plus,
                                   lattice=lattice, nu=nu, y_val=y_val)
        else:
            f_ref = collide_bgk27(f_ref, tau)
            f_ref = stream27_roll(f_ref)
            f_ref = bounce_back_cells_27(f_ref, mask)
            rho, ux, uy, uz = macroscopic27(f_ref)
            u_mag = torch.sqrt(ux*ux + uy*uy + uz*uz).clamp(min=1e-12)
            u_tau = compute_u_tau(u_mag, nu, y_val, "log")
            y_plus = compute_y_plus(u_tau, nu, y_val)
            f_ref = wall_function(f_ref, mask, u_tau, y_plus,
                                   lattice=lattice, nu=nu, y_val=y_val)

        # --- Executor ---
        if lattice == "D3Q19":
            bfn = bounce_back_cells_3d
        else:
            bfn = bounce_back_cells_27
        ex = LBMStepExecutor(
            lattice, collide_fn="bgk", device=DEVICE,
            nx=NX, ny=NY, nz=NZ, tau=tau,
            boundary_fn=bfn,
            boundary_kwargs={"mask": mask},
            wall_fn=True, mask=mask, nu=nu, y_val=y_val,
        )
        f_ex, _ = ex.step(f0.clone())

        assert torch.allclose(f_ex, f_ref, atol=ATOL), (
            f"full step (wall fn) mismatch ({lattice})"
        )

    @pytest.mark.parametrize("lattice,Q", [("D3Q19", 19), ("D3Q27", 27)])
    def test_full_step_mass_correction(self, lattice, Q):
        """BGK + stream + mass correction."""
        f0 = _make_f(Q)
        tau = 0.6
        target_mass = float(f0.sum().item())

        # --- Manual loop (reference) ---
        f_ref = f0.clone()
        if lattice == "D3Q19":
            f_ref = collide_bgk3d(f_ref, tau)
            f_ref = stream3d(f_ref)
            f_ref = correct_mass3d(f_ref, target_mass)
        else:
            from tensorlbm.d3q27 import correct_mass27
            f_ref = collide_bgk27(f_ref, tau)
            f_ref = stream27_roll(f_ref)
            f_ref = correct_mass27(f_ref, target_mass)

        # --- Executor ---
        ex = LBMStepExecutor(
            lattice, collide_fn="bgk", device=DEVICE,
            nx=NX, ny=NY, nz=NZ, tau=tau,
            target_mass=target_mass,
        )
        f_ex, _ = ex.step(f0.clone())

        assert torch.allclose(f_ex, f_ref, atol=ATOL), (
            f"full step (mass correction) mismatch ({lattice})"
        )


# ---------------------------------------------------------------------------
# 7. External collision operator equivalence (MRT, Cumulant)
# ---------------------------------------------------------------------------

class TestExternalCollisionEquivalence:
    """Executor with external collide_fn must match manual loop."""

    @pytest.mark.parametrize("lattice,Q,collide_ref", [
        ("D3Q19", 19, collide_mrt3d),
        ("D3Q27", 27, collide_mrt27),
    ])
    def test_mrt_full_step(self, lattice, Q, collide_ref):
        f0 = _make_f(Q)
        tau = 0.6
        mask = _make_mask()

        # Manual
        f_ref = f0.clone()
        f_ref = collide_ref(f_ref, tau)
        if lattice == "D3Q19":
            f_ref = stream3d(f_ref)
            f_ref = bounce_back_cells_3d(f_ref, mask)
        else:
            f_ref = stream27_roll(f_ref)
            f_ref = bounce_back_cells_27(f_ref, mask)

        # Executor
        if lattice == "D3Q19":
            bfn = bounce_back_cells_3d
        else:
            bfn = bounce_back_cells_27
        ex = LBMStepExecutor(
            lattice, collide_fn=collide_ref, device=DEVICE,
            nx=NX, ny=NY, nz=NZ, tau=tau,
            boundary_fn=bfn,
            boundary_kwargs={"mask": mask},
        )
        f_ex, _ = ex.step(f0.clone())

        assert torch.allclose(f_ex, f_ref, atol=ATOL), (
            f"MRT full step mismatch ({lattice})"
        )

    @pytest.mark.parametrize("lattice,Q,collide_ref", [
        ("D3Q19", 19, collide_cumulant_d3q19),
        ("D3Q27", 27, collide_cumulant_d3q27),
    ])
    def test_cumulant_full_step(self, lattice, Q, collide_ref):
        f0 = _make_f(Q)
        tau = 0.6
        mask = _make_mask()

        # Manual
        f_ref = f0.clone()
        f_ref = collide_ref(f_ref, tau)
        if lattice == "D3Q19":
            f_ref = stream3d(f_ref)
            f_ref = bounce_back_cells_3d(f_ref, mask)
        else:
            f_ref = stream27_roll(f_ref)
            f_ref = bounce_back_cells_27(f_ref, mask)

        # Executor
        if lattice == "D3Q19":
            bfn = bounce_back_cells_3d
        else:
            bfn = bounce_back_cells_27
        ex = LBMStepExecutor(
            lattice, collide_fn=collide_ref, device=DEVICE,
            nx=NX, ny=NY, nz=NZ, tau=tau,
            boundary_fn=bfn,
            boundary_kwargs={"mask": mask},
        )
        f_ex, _ = ex.step(f0.clone())

        assert torch.allclose(f_ex, f_ref, atol=ATOL), (
            f"Cumulant full step mismatch ({lattice})"
        )


# ---------------------------------------------------------------------------
# 8. Multi-step equivalence
# ---------------------------------------------------------------------------

class TestMultiStepEquivalence:
    """Multiple steps must remain allclose (no error accumulation)."""

    @pytest.mark.parametrize("lattice,Q", [("D3Q19", 19), ("D3Q27", 27)])
    def test_10_steps_allclose(self, lattice, Q):
        f0 = _make_f(Q)
        tau = 0.6
        mask = _make_mask()
        n_steps = 10

        # Manual
        f_ref = f0.clone()
        for _ in range(n_steps):
            if lattice == "D3Q19":
                f_ref = collide_bgk3d(f_ref, tau)
                f_ref = stream3d(f_ref)
                f_ref = bounce_back_cells_3d(f_ref, mask)
            else:
                f_ref = collide_bgk27(f_ref, tau)
                f_ref = stream27_roll(f_ref)
                f_ref = bounce_back_cells_27(f_ref, mask)

        # Executor
        if lattice == "D3Q19":
            bfn = bounce_back_cells_3d
        else:
            bfn = bounce_back_cells_27
        ex = LBMStepExecutor(
            lattice, collide_fn="bgk", device=DEVICE,
            nx=NX, ny=NY, nz=NZ, tau=tau,
            boundary_fn=bfn,
            boundary_kwargs={"mask": mask},
        )
        f_ex, _ = ex.run(f0.clone(), n_steps)

        assert torch.allclose(f_ex, f_ref, atol=ATOL), (
            f"multi-step mismatch ({lattice}, {n_steps} steps)"
        )


# ---------------------------------------------------------------------------
# 9. Performance comparison
# ---------------------------------------------------------------------------

class TestPerformance:
    """Compare MLUPS: executor vs manual loop."""

    @pytest.mark.parametrize("lattice,Q", [("D3Q19", 19), ("D3Q27", 27)])
    def test_mlups_comparison(self, lattice, Q):
        nz, ny, nx = 64, 64, 64
        n_steps = 50
        tau = 0.6
        mask = torch.zeros(nz, ny, nx, dtype=torch.bool, device=DEVICE)

        # Initialise with equilibrium
        rho0 = torch.ones(nz, ny, nx, device=DEVICE, dtype=DTYPE)
        ux0 = torch.full_like(rho0, 0.05)
        uy0 = torch.full_like(rho0, 0.02)
        uz0 = torch.full_like(rho0, 0.01)
        if Q == 19:
            f_init = equilibrium3d(rho0, ux0, uy0, uz0, device=DEVICE)
            bfn = bounce_back_cells_3d
            collide_ref = collide_bgk3d
            stream_ref = stream3d
        else:
            f_init = equilibrium27(rho0, ux0, uy0, uz0, device=DEVICE)
            bfn = bounce_back_cells_27
            collide_ref = collide_bgk27
            stream_ref = stream27_roll

        n_cells = nz * ny * nx

        # --- Manual loop ---
        f_man = f_init.clone()
        # Warmup
        for _ in range(3):
            f_man = collide_ref(f_man, tau)
            f_man = stream_ref(f_man)
            f_man = bfn(f_man, mask)
        if DEVICE.type == "sdaa":
            torch.sdaa.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            f_man = collide_ref(f_man, tau)
            f_man = stream_ref(f_man)
            f_man = bfn(f_man, mask)
        if DEVICE.type == "sdaa":
            torch.sdaa.synchronize()
        t_man = time.perf_counter() - t0
        mlups_man = n_cells * n_steps / t_man / 1e6

        # --- Executor ---
        ex = LBMStepExecutor(
            lattice, collide_fn="bgk", device=DEVICE,
            nx=nx, ny=ny, nz=nz, tau=tau,
            boundary_fn=bfn,
            boundary_kwargs={"mask": mask},
        )
        f_ex = f_init.clone()
        # Warmup
        for _ in range(3):
            f_ex, _ = ex.step(f_ex)
        if DEVICE.type == "sdaa":
            torch.sdaa.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            f_ex, _ = ex.step(f_ex)
        if DEVICE.type == "sdaa":
            torch.sdaa.synchronize()
        t_ex = time.perf_counter() - t0
        mlups_ex = n_cells * n_steps / t_ex / 1e6

        # Verify numerical equivalence after n_steps
        assert torch.allclose(f_ex, f_man, atol=ATOL), (
            f"performance test numerical mismatch ({lattice})"
        )

        print(f"\n[{lattice}] Manual: {mlups_man:.1f} MLUPS, "
              f"Executor: {mlups_ex:.1f} MLUPS, "
              f"Speedup: {mlups_ex/mlups_man:.2f}x")

    @pytest.mark.parametrize("lattice,Q", [("D3Q19", 19), ("D3Q27", 27)])
    def test_mlups_wall_function(self, lattice, Q):
        """Wall-function step: executor saves 2 macroscopic calls per step."""
        nz, ny, nx = 32, 32, 32
        n_steps = 50
        tau = 0.6
        nu = 0.02
        y_val = 0.5

        # Create obstacle mask
        zz, yy, xx = torch.meshgrid(
            torch.arange(nz, device=DEVICE),
            torch.arange(ny, device=DEVICE),
            torch.arange(nx, device=DEVICE),
            indexing="ij",
        )
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        r = min(nz, ny, nx) // 6
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < r * r

        # Initialise
        rho0 = torch.ones(nz, ny, nx, device=DEVICE, dtype=DTYPE)
        ux0 = torch.full_like(rho0, 0.05)
        uy0 = torch.full_like(rho0, 0.02)
        uz0 = torch.full_like(rho0, 0.01)
        if Q == 19:
            f_init = equilibrium3d(rho0, ux0, uy0, uz0, device=DEVICE)
            bfn = bounce_back_cells_3d
            collide_ref = collide_bgk3d
            stream_ref = stream3d
            macro_ref = macroscopic3d
        else:
            f_init = equilibrium27(rho0, ux0, uy0, uz0, device=DEVICE)
            bfn = bounce_back_cells_27
            collide_ref = collide_bgk27
            stream_ref = stream27_roll
            macro_ref = macroscopic27

        n_cells = nz * ny * nx

        # --- Manual loop (with wall function) ---
        f_man = f_init.clone()
        # Warmup
        for _ in range(3):
            f_man = collide_ref(f_man, tau)
            f_man = stream_ref(f_man)
            f_man = bfn(f_man, mask)
            rho, ux, uy, uz = macro_ref(f_man)
            u_mag = torch.sqrt(ux*ux + uy*uy + uz*uz).clamp(min=1e-12)
            u_tau = compute_u_tau(u_mag, nu, y_val, "log")
            y_plus = compute_y_plus(u_tau, nu, y_val)
            f_man = wall_function(f_man, mask, u_tau, y_plus,
                                   lattice=lattice, nu=nu, y_val=y_val)
        if DEVICE.type == "sdaa":
            torch.sdaa.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            f_man = collide_ref(f_man, tau)
            f_man = stream_ref(f_man)
            f_man = bfn(f_man, mask)
            rho, ux, uy, uz = macro_ref(f_man)
            u_mag = torch.sqrt(ux*ux + uy*uy + uz*uz).clamp(min=1e-12)
            u_tau = compute_u_tau(u_mag, nu, y_val, "log")
            y_plus = compute_y_plus(u_tau, nu, y_val)
            f_man = wall_function(f_man, mask, u_tau, y_plus,
                                   lattice=lattice, nu=nu, y_val=y_val)
        if DEVICE.type == "sdaa":
            torch.sdaa.synchronize()
        t_man = time.perf_counter() - t0
        mlups_man = n_cells * n_steps / t_man / 1e6

        # --- Executor (with wall function, macroscopic reuse) ---
        ex = LBMStepExecutor(
            lattice, collide_fn="bgk", device=DEVICE,
            nx=nx, ny=ny, nz=nz, tau=tau,
            boundary_fn=bfn,
            boundary_kwargs={"mask": mask},
            wall_fn=True, mask=mask, nu=nu, y_val=y_val,
        )
        f_ex = f_init.clone()
        # Warmup
        for _ in range(3):
            f_ex, _ = ex.step(f_ex)
        if DEVICE.type == "sdaa":
            torch.sdaa.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            f_ex, _ = ex.step(f_ex)
        if DEVICE.type == "sdaa":
            torch.sdaa.synchronize()
        t_ex = time.perf_counter() - t0
        mlups_ex = n_cells * n_steps / t_ex / 1e6

        # Verify numerical equivalence
        assert torch.allclose(f_ex, f_man, atol=ATOL), (
            f"wall-fn performance test numerical mismatch ({lattice})"
        )

        print(f"\n[{lattice} wall-fn] Manual: {mlups_man:.1f} MLUPS, "
              f"Executor: {mlups_ex:.1f} MLUPS, "
              f"Speedup: {mlups_ex/mlups_man:.2f}x")
