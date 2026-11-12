"""Tests for the T2 unified GPU core path (``tensorlbm.gpu_core_path``).

These pin the properties the module claims: the solver is constructible and
runs (the m15657 defect was a config attribute missing at construction), the
kernels reproduce an independent BGK+Guo reference, mass is conserved without
forcing, both analytic anchors (Poiseuille, Taylor–Green) are met, and the
backend dispatch refuses to pretend when CUDA/triton is absent.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from tensorlbm.d3q19 import C, OPPOSITE, W
from tensorlbm.gpu_core_path import (
    DistributedGPUCorePathSolver,
    GPUCorePathConfig,
    GPUCorePathSolver,
    apply_channel_walls,
    channel_analytic_velocity,
    collide_guo_into,
    get_best_available_backend,
    run_channel_case,
    run_taylor_green_case,
    stream_into,
    taylor_green_analytic_energy,
)


def test_config_derived_physics_is_self_consistent():
    cfg = GPUCorePathConfig(ny=18, tau=0.8, u_max_target=0.05)
    assert cfg.height == 16.0
    assert cfg.nu == pytest.approx(0.1)
    assert cfg.body_force == pytest.approx(8 * cfg.nu * cfg.u_max_target / cfg.height**2)
    # u_max = u_tau Re_tau / 2 (the analytic signature of a body-force channel)
    assert cfg.u_tau * cfg.re_tau / 2 == pytest.approx(cfg.u_max_target, rel=1e-12)


def test_config_rejects_bad_input():
    for bad in (dict(tau=0.5), dict(ny=4), dict(init="nonsense"), dict(steps=0)):
        with pytest.raises(ValueError):
            GPUCorePathConfig(**bad).validate()


def test_solver_constructs_and_steps():
    """Regression pin for the m15657 defect (AttributeError at construction)."""
    cfg = GPUCorePathConfig(ny=18, tau=0.8, steps=5, backend="eager")
    solver = GPUCorePathSolver(cfg)
    assert solver.backend == "eager"
    solver.step(5)
    assert solver.steps == 5
    assert torch.isfinite(solver._fa).all()
    # the two state buffers are reused (ping-pong), not re-allocated
    ptrs = {solver._fa.data_ptr(), solver._fb.data_ptr()}
    solver.step(3)
    assert {solver._fa.data_ptr(), solver._fb.data_ptr()} == ptrs


def test_collision_matches_independent_reference():
    torch.manual_seed(0)
    f = torch.rand(19, 2, 6, 4, dtype=torch.float64) * 0.1 + 0.02
    f = f / f.sum(dim=0, keepdim=True)  # rho = 1
    tau = 0.8
    acc = 3.0e-4
    out = torch.empty_like(f)
    collide_guo_into(f, tau, out, C.to(torch.float64), W.to(torch.float64), body_force=acc)

    # independent reference: equilibrium at u* then BGK then Guo force term
    rho = f.sum(0)
    u = torch.stack([(f * C[:, i].view(-1, 1, 1, 1)).sum(0) for i in range(3)]) / rho
    fx = acc * rho
    us = u.clone()
    us[0] += 0.5 * fx / rho
    cu = sum(C[:, i].view(-1, 1, 1, 1) * us[i] for i in range(3))
    u2 = (us**2).sum(0)
    feq = W.view(-1, 1, 1, 1) * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
    ref = f - (f - feq) / tau
    cpu = sum(C[:, i].view(-1, 1, 1, 1) * u[i] for i in range(3))
    ref = ref + W.view(-1, 1, 1, 1) * (1 - 1 / (2 * tau)) * (
        3 * (C[:, 0].view(-1, 1, 1, 1) - u[0]) * fx + 9 * cpu * C[:, 0].view(-1, 1, 1, 1) * fx
    )
    assert torch.allclose(out, ref, atol=1e-10, rtol=1e-10)


def test_streaming_moves_population_along_its_velocity():
    f = torch.zeros(19, 2, 4, 4, dtype=torch.float64)
    f[1, 0, 0, 0] = 1.0  # +x direction
    out = torch.empty_like(f)
    stream_into(f, out)
    assert out[1, 0, 0, 1] == 1.0  # moved one cell in +x
    f2 = torch.zeros_like(f)
    f2[3, 0, 0, 0] = 1.0  # +y
    stream_into(f2, out)
    assert out[3, 0, 1, 0] == 1.0


def test_half_way_bounce_back_reflects_at_wall_rows():
    f = torch.zeros(19, 1, 6, 3, dtype=torch.float64)
    fpost = torch.zeros_like(f)
    fpost[OPPOSITE[3], :, 1, :] = 0.25  # opposite of +y at the first fluid row
    apply_channel_walls(f, fpost)
    assert f[3, :, 1, :].unique().tolist() == [0.25]
    fpost2 = torch.zeros_like(f)
    fpost2[OPPOSITE[4], :, -2, :] = 0.5  # opposite of -y at the last fluid row
    apply_channel_walls(f, fpost2)
    assert f[4, :, -2, :].unique().tolist() == [0.5]


def test_mass_conserved_without_body_force():
    cfg = GPUCorePathConfig(nx=16, ny=16, nz=4, tau=0.8, init="taylor_green",
                            tg_u0=0.01, perturbation_amplitude=1e-2, steps=40,
                            backend="eager")
    solver = GPUCorePathSolver(cfg)
    m0 = solver.mass()
    solver.step(40)
    # Periodic streaming + a mass-conserving collision: the only drift left is
    # round-off accumulated over the steps, far below any physical scale.
    drift = abs(solver.mass() - m0)
    assert drift < 1e-10, drift


def test_taylor_green_decay_matches_exact_solution():
    cfg = GPUCorePathConfig(nx=32, ny=32, nz=4, tau=0.6, init="taylor_green",
                            tg_u0=0.02, steps=120, backend="eager")
    row = run_taylor_green_case(cfg, samples=3)
    assert row["ke0_rel_err"] < 1e-3
    assert row["max_rel_err"] < 0.03, row["samples"]


def test_channel_matches_analytic_parabola():
    cfg = GPUCorePathConfig(ny=18, tau=0.8, u_max_target=0.05, steps=3000,
                            backend="eager", label="pytest_channel")
    row = run_channel_case(cfg, check_every=500, steady_tol=2e-5)
    assert row["u_max_rel_err"] < 0.03, row
    assert row["profile_rel_l2"] < 0.03, row
    assert row["mass_drift"] < 1e-4, row


def test_precision_tier_is_honoured():
    cfg = GPUCorePathConfig(ny=18, tau=0.8, steps=20, precision="FP32FP16",
                            backend="eager")
    solver = GPUCorePathSolver(cfg)
    assert solver.compute_dtype is torch.float32
    assert solver.store_dtype is torch.float16
    assert solver.f.dtype is torch.float16
    solver.step(5)
    assert torch.isfinite(solver._fa).all()


def test_forced_triton_backend_raises_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA host: triton path is available")
    with pytest.raises(RuntimeError):
        get_best_available_backend("triton")


def test_distributed_solver_requires_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA host")
    with pytest.raises(RuntimeError):
        DistributedGPUCorePathSolver(GPUCorePathConfig(), rank=0, world_size=1)


@pytest.mark.skipif(
    os.environ.get("TL_TEST_COMPILE", "0") != "1",
    reason="torch.compile test is slow; enable with TL_TEST_COMPILE=1",
)
def test_compiled_backend_matches_eager():
    base = dict(ny=18, tau=0.8, steps=20, init="parabolic", precision="FP64FP64")
    a = GPUCorePathSolver(GPUCorePathConfig(backend="eager", **base))
    b = GPUCorePathSolver(GPUCorePathConfig(backend="torch_compile", **base))
    a.step(20)
    b.step(20)
    assert torch.allclose(a._fa, b._fa, atol=1e-12, rtol=0)
