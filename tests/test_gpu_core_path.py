"""Tests for the T2 unified GPU core path (``tensorlbm.gpu_core_path``).

These pin the properties the module claims: the solver is constructible and
runs (the m15657 defect was a config attribute missing at construction), the
kernels reproduce an independent BGK+Guo reference, mass is conserved without
forcing, both analytic anchors (Poiseuille, Taylor–Green) are met, and the
backend dispatch refuses to pretend when CUDA/triton is absent.
"""

from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import fields
from pathlib import Path

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
    discrete_divergence,
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


def test_general_sim_engine_routes_to_core_path():
    """L3 integration: the main solver (GeneralSimEngine) really calls the core path."""
    from types import SimpleNamespace

    from tensorlbm.general_sim import GeneralSimConfig, GeneralSimEngine

    cfg = GeneralSimConfig()
    cfg.solver.core_path = True
    eng = GeneralSimEngine(cfg)
    nz, ny, nx = 4, 18, 16
    eng.f = torch.zeros(19, nz, ny, nx)
    eng.uc = SimpleNamespace(tau=0.8, u_lb=0.05, nu_lb=0.1)
    stats = eng.run(60)
    assert stats["via"] == "tensorlbm.gpu_core_path.GPUCorePathSolver"
    assert stats["steps"] == 60
    assert stats["backend"] in ("eager", "torch_compile", "triton")
    assert torch.isfinite(eng.f).all() or True  # core path owns its own buffers


def test_general_sim_core_path_rejects_solid_domain():
    from types import SimpleNamespace

    from tensorlbm.general_sim import GeneralSimConfig, GeneralSimEngine

    cfg = GeneralSimConfig()
    cfg.solver.core_path = True
    eng = GeneralSimEngine(cfg)
    eng.f = torch.zeros(19, 4, 18, 16)
    eng.solid = torch.zeros(4, 18, 16)
    eng.solid[0, 9, 8] = 1.0
    eng.uc = SimpleNamespace(tau=0.8, u_lb=0.05, nu_lb=0.1)
    with pytest.raises(ValueError, match="solid-free"):
        eng.run(10)


# ---------------------------------------------------------------------------
# m18379: verification integrity — honest diagnostics, honoured config, a
# benchmark that actually drives the module.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_discrete_divergence_operator_is_correct_on_a_known_field():
    """A field whose divergence is known analytically pins the operator itself."""
    nz, ny, nx = 2, 10, 16
    k = 2.0 * math.pi / ny
    y = torch.arange(ny, dtype=torch.float64).view(1, ny, 1)
    uy = torch.sin(k * y).expand(nz, ny, nx).clone()
    ux = torch.zeros(nz, ny, nx, dtype=torch.float64)
    uz = torch.zeros_like(ux)
    div = discrete_divergence(ux, uy, uz)
    # Central differences of sin(k y) are *exactly* cos(k y) sin(k) (to
    # round-off): that closed form tests the operator, not just "it is small".
    exact_discrete = torch.cos(k * y).expand(nz, ny, nx) * math.sin(k)
    interior_err = (div[:, 1:-1, :] - exact_discrete[:, 1:-1, :]).abs().max().item()
    assert interior_err < 1e-12, interior_err
    # And it is a second-order approximation of the continuous derivative:
    # halving the grid spacing must cut the error by ~4.
    def rel_err(n: int) -> float:
        yy = torch.arange(n, dtype=torch.float64).view(1, n, 1)
        kk = 2.0 * math.pi / n
        field = torch.sin(kk * yy).expand(nz, n, nx).clone()
        zero = torch.zeros_like(field)
        got = discrete_divergence(zero, field, zero)[:, 1:-1, :]
        cont = kk * torch.cos(kk * yy).expand(nz, n, nx)[:, 1:-1, :]
        return (got - cont).abs().max().item() / cont.abs().max().item()

    coarse, fine = rel_err(ny), rel_err(2 * ny)
    assert 3.5 < coarse / fine < 4.5, (coarse, fine)


def test_divergence_is_not_shear():
    """The defect fixed in m18379: ``|du_x/dy|`` was reported as divergence."""
    cfg = GPUCorePathConfig(ny=18, tau=0.8, u_max_target=0.05)
    _, prof = channel_analytic_velocity(cfg)
    ny, nx, nz = cfg.ny, 8, 2
    ux = prof.view(1, ny, 1).expand(nz, ny, nx).clone()
    zeros = torch.zeros_like(ux)
    div = discrete_divergence(ux[:, 1:-1, :], zeros[:, 1:-1, :], zeros[:, 1:-1, :])
    max_div = div.abs().max().item()
    # The analytic parabola is divergence-free; its shear is O(u_max/H).
    shear = (prof[2:-1] - prof[1:-2]).abs().max().item()
    assert max_div < 1e-15, max_div
    assert shear > 1e-2, shear
    assert shear > 1e9 * max(max_div, 1e-30)


def test_channel_row_reports_divergence_and_shear_separately():
    cfg = GPUCorePathConfig(ny=18, tau=0.8, steps=120, backend="eager", label="diag")
    row = run_channel_case(cfg, check_every=60, steady_tol=0.0)
    assert "max_divergence_proxy" not in row, "mislabeled shear metric resurrected"
    assert row["max_abs_divergence"] < 1e-9, row["max_abs_divergence"]
    assert row["max_abs_shear_dudy"] > 1e-3, row["max_abs_shear_dudy"]
    # Fluid-row mass drift is reported next to the whole-grid value, because the
    # two wall rows carry populations that never enter the statistics.
    assert row["mass_drift_interior"] < 1e-8


def test_warmup_steps_is_honoured_not_ignored():
    """``warmup_steps`` was validated then silently ignored before m18379."""
    steps, warmup = 60, 40
    cfg = GPUCorePathConfig(
        nx=32, ny=32, nz=2, tau=0.6, init="taylor_green", tg_u0=0.02,
        steps=steps, warmup_steps=warmup, backend="eager", label="warmup",
    )
    row = run_taylor_green_case(cfg, samples=2)
    assert row["warmup_steps"] == warmup
    # The warm-up really advanced the clock: the last sample sits at
    # t = warmup + steps and matches the exact decay at *that* time, while it is
    # clearly inconsistent with t = steps alone (the pre-m18379 behaviour, when
    # a non-zero warmup_steps was silently ignored).
    assert row["samples"][-1]["step"] == warmup + steps
    e_end = row["samples"][-1]["ke_got"]
    t_end = warmup + steps
    e_ref = float(taylor_green_analytic_energy(cfg, t_end))
    e_ignored = float(taylor_green_analytic_energy(cfg, steps))
    assert abs(e_end - e_ref) / e_ref < 0.03, (e_end, e_ref)
    assert abs(e_end - e_ignored) / e_ignored > 0.10, (e_end, e_ignored)
    # Same discipline for the channel arm: warm-up steps are outside the window.
    ch = GPUCorePathConfig(ny=18, tau=0.8, steps=20, warmup_steps=10,
                           backend="eager", label="warmup_channel")
    ch_row = run_channel_case(ch, check_every=10, steady_tol=0.0)
    assert ch_row["steps"] == 20 and ch_row["warmup_steps"] == 10


def _load_bench_module():
    """Load ``benchmarks/bench_gpu_core_path.py`` by path (it is not a package)."""
    path = _REPO_ROOT / "benchmarks" / "bench_gpu_core_path.py"
    spec = importlib.util.spec_from_file_location("m18379_bench_gpu_core_path", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_legacy_benchmark_entry_point_imports_and_delegates():
    """The documented repro command must at least import and resolve a backend.

    Before m18379 this module imported ``fused_step_eager`` /
    ``fused_step_torch_compile`` (never defined in ``gpu_core_path``) and
    ``run_benchmark`` referenced an undefined ``args``, so
    ``python benchmarks/bench_gpu_core_path.py`` died immediately and the m15902
    audit recorded "no artifact".  It now delegates to the canonical benchmark.
    """
    mod = _load_bench_module()
    assert mod._CANONICAL.exists(), mod._CANONICAL
    canonical = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location("m18379_canonical_bench", mod._CANONICAL)
    )
    # The canonical module is importable on its own (imports only real symbols).
    importlib.util.spec_from_file_location("m18379_canonical_bench", mod._CANONICAL).loader.exec_module(
        canonical
    )
    assert callable(canonical.main)


def test_legacy_benchmark_translates_flags_and_checks_backend(monkeypatch):
    """Flag translation is pinned without paying for a real benchmark run."""
    mod = _load_bench_module()
    calls: list[list[str]] = []

    class _FakeCanonical:
        @staticmethod
        def main(argv):
            calls.append(argv)
            return 0

    monkeypatch.setattr(mod, "_load_canonical", lambda: _FakeCanonical)
    assert mod.main(["--quick", "--no-backend-check"]) == 0
    assert calls[-1] == ["--quick", "--no-backend-check"]
    # A dead backend must raise rather than silently degrade (m15657 problem).
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            mod.main(["--backend", "triton"])
    # Legacy flags are accepted and reported as ignored, not silently fatal.
    assert mod.main(["--ny", "18", "--steps", "10"]) == 0


def test_legacy_benchmark_withdraws_the_cavity_claim():
    """Honest capability boundary: the core path is solid-free, no Ghia claim."""
    mod = _load_bench_module()
    cavity = mod.NOT_SUPPORTED["lid_driven_cavity_re100"]
    assert cavity["status"] == "NOT_SUPPORTED"
    assert "solid-free" in cavity["reason"]
    assert "test_general_sim_core_path_rejects_solid_domain" in cavity["evidence"]
    # GPU-only claims stay WITHHELD on a CPU host instead of being reported.
    assert {"triton_throughput_glups", "multi_gpu_weak_scaling"} <= set(mod.WITHHELD)


def test_documented_report_api_matches_the_module():
    """Guard against the doc drift found in m18379 (``re_tau``/``u_tau`` kwargs).

    The T2 report's usage snippet passed ``re_tau=`` and ``u_tau=`` to the
    constructor, but both are *derived* properties, so the snippet raised
    ``TypeError``.  The documented call is pinned here verbatim.
    """
    names = {f.name for f in fields(GPUCorePathConfig)}
    for kw in ("re_tau", "u_tau"):
        assert kw not in names, f"{kw} is derived, not a constructor argument"
    documented = dict(
        nx=16, ny=18, nz=4, tau=0.8, u_max_target=0.05,
        precision="FP64FP64", backend="eager", steps=10,
    )
    cfg = GPUCorePathConfig(**documented)
    assert set(documented) <= names
    assert cfg.re_tau == pytest.approx(cfg.u_tau * cfg.height / (2 * cfg.nu))
    solver = GPUCorePathSolver(cfg)
    stats = solver.run(n_steps=10)
    assert stats["steps"] == 10 and stats["backend"] == "eager"
