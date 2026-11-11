"""Regression tests for TensorLBM's core input contract (audit D5/D6/D7/D9 follow-up).

``docs/AUDIT_RUNTEST_6_TENSORLBM.md`` (2026-11-11) recorded that the same illegal
input was rejected by some entry points and silently accepted by others:

* D7 — ``collide_bgk`` accepted ``tau<=0.5`` while ``collide_advanced_3d`` raised;
* D6 — ``zou_he_outlet_pressure(rho_out=0)`` returned an all-NaN field silently;
* D5 — ``GeneralSimConfig`` accepted ``viscosity=0`` / ``resolution=0`` / ``max_steps<0``;
* D9 — ``poiseuille_2d.run_case(H=0)`` died with a bare ``ZeroDivisionError``.

Each test below asserts the *contract* (a ``ValueError`` with a message naming the
offending parameter), not merely that "something was raised", and each valid-input
test asserts a physical invariant (mass conservation / finite field) so the guard
cannot be satisfied by blanket rejection.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.input_contract import (
    check_positive_int,
    check_positive_scalar,
    check_relaxation_time,
)
from tensorlbm.solver import collide_bgk, collide_bgk_fused

TAU_BAD = [0.5, 0.4, 0.0, -0.7, float("nan"), float("inf")]


def _bulk(n: int = 8, rho: float = 1.0):
    from tensorlbm.d2q9 import equilibrium

    rr = torch.full((n, n), rho)
    z = torch.zeros((n, n))
    return equilibrium(rr, z, z)


class TestRelaxationTimeGuard:
    @pytest.mark.parametrize("tau", TAU_BAD)
    def test_collide_bgk_rejects_nonpositive_viscosity(self, tau):
        f = _bulk()
        with pytest.raises(ValueError, match="tau"):
            collide_bgk(f, tau)

    @pytest.mark.parametrize("tau", TAU_BAD)
    def test_collide_bgk_fused_rejects_nonpositive_viscosity(self, tau):
        f = _bulk()
        with pytest.raises(ValueError, match="tau"):
            collide_bgk_fused(f, tau)

    def test_guard_matches_sibling_module_behaviour(self):
        """The contract must not depend on which entry point the caller picked."""
        from tensorlbm.advanced_collision_contract import collide_advanced_3d

        f3 = torch.full((19, 4, 4, 4), 1.0 / 19.0)
        with pytest.raises(ValueError, match="tau"):
            collide_advanced_3d("D3Q19", "BGK", f3, tau=0.5)
        with pytest.raises(ValueError, match="tau"):
            check_relaxation_time(0.5, where="test")

    def test_valid_tau_still_works_and_conserves_mass(self):
        f = _bulk()
        # give the field a shear so the collision is not a no-op
        f[1] *= 1.05
        f[3] *= 0.95
        out = collide_bgk(f.clone(), 0.6)
        assert torch.isfinite(out).all()
        assert out.sum().item() == pytest.approx(f.sum().item(), rel=1e-6)
        # tau=0.5+1e-9 is legal but nearly inviscid: finite, mass-conserving
        out2 = collide_bgk(f.clone(), 0.5 + 1e-9)
        assert torch.isfinite(out2).all()
        assert out2.sum().item() == pytest.approx(f.sum().item(), rel=1e-6)

    def test_check_relaxation_time_passes_tensors_through(self):
        """Tensor tau (per-cell, e.g. Smagorinsky) is validated by its own path."""
        t = torch.full((2, 2), 0.6)
        assert check_relaxation_time(t) is t


class TestPressureOutletGuard:
    @pytest.mark.parametrize("rho_out", [0.0, -1.0, float("nan"), float("inf")])
    def test_d2q9_outlet_rejects_invalid_density(self, rho_out):
        from tensorlbm.boundary.boundaries import zou_he_outlet_pressure

        f = torch.full((9, 6, 6), 1.0 / 9.0)
        with pytest.raises(ValueError, match="rho_out"):
            zou_he_outlet_pressure(f, rho_out=rho_out)

    @pytest.mark.parametrize("rho_out", [0.0, -1.0])
    def test_3d_outlets_reject_invalid_density(self, rho_out):
        from tensorlbm.boundary.boundaries3d import (
            zou_he_outlet_pressure_3d,
            zou_he_outlet_pressure_z,
        )
        from tensorlbm.boundary.boundaries_d3q27 import zou_he_outlet_pressure_27

        with pytest.raises(ValueError, match="rho_out"):
            zou_he_outlet_pressure_3d(torch.full((19, 4, 4, 4), 1.0 / 19.0), rho_out=rho_out)
        with pytest.raises(ValueError, match="rho_out"):
            zou_he_outlet_pressure_z(torch.full((19, 4, 4, 4), 1.0 / 19.0), rho_out=rho_out)
        with pytest.raises(ValueError, match="rho_out"):
            zou_he_outlet_pressure_27(torch.full((27, 4, 4, 4), 1.0 / 27.0), rho_out=rho_out)

    def test_valid_outlet_stays_finite(self):
        from tensorlbm.boundary.boundaries import zou_he_outlet_pressure

        f = torch.full((9, 6, 6), 1.0 / 9.0)
        out = zou_he_outlet_pressure(f, rho_out=1.0)
        assert torch.isfinite(out).all()


class TestGeneralSimConfigGuard:
    def test_rejects_degenerate_viscosity(self):
        from tensorlbm.general_sim import GeneralSimConfig, PhysicsConfig

        with pytest.raises(ValueError, match="viscosity"):
            GeneralSimConfig(physics=PhysicsConfig(viscosity=0.0)).validate()
        with pytest.raises(ValueError, match="viscosity"):
            GeneralSimConfig(physics=PhysicsConfig(viscosity=-1e-6)).validate()
        with pytest.raises(ValueError, match="viscosity"):
            GeneralSimConfig(physics=PhysicsConfig(viscosity=float("nan"))).validate()

    def test_rejects_degenerate_resolution_and_steps(self):
        from tensorlbm.general_sim import GeneralSimConfig, SolverConfig

        with pytest.raises(ValueError, match="resolution"):
            GeneralSimConfig(solver=SolverConfig(resolution=0)).validate()
        with pytest.raises(ValueError, match="resolution"):
            GeneralSimConfig(solver=SolverConfig(resolution=3)).validate()
        with pytest.raises(ValueError, match="max_steps"):
            GeneralSimConfig(solver=SolverConfig(max_steps=-5)).validate()

    def test_engine_enforces_config_on_construction(self):
        from tensorlbm.general_sim import GeneralSimConfig, GeneralSimEngine, PhysicsConfig

        with pytest.raises(ValueError, match="viscosity"):
            GeneralSimEngine(GeneralSimConfig(physics=PhysicsConfig(viscosity=0.0)))

    def test_default_config_valid_and_returns_self(self):
        from tensorlbm.general_sim import GeneralSimConfig

        cfg = GeneralSimConfig()
        assert cfg.validate() is cfg
        assert math.isfinite(cfg.reynolds_number) and cfg.reynolds_number > 0


class TestBenchmarkCliGuards:
    def test_poiseuille_2d_rejects_zero_height(self):
        import importlib.util
        from pathlib import Path

        run = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "poiseuille_2d" / "run.py"
        if not run.exists():  # pragma: no cover - packaging corner
            pytest.skip("poiseuille_2d benchmark not present")
        spec = importlib.util.spec_from_file_location("p2d_run", run)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with pytest.raises(ValueError, match="H"):
            mod.run_case(0, tau=0.8, u_max_target=0.04, collision="bgk",
                         min_steps=2, max_steps=4, out_path="/tmp/_guard_p2d.json")


class TestGuardHelpers:
    def test_positive_scalar(self):
        assert check_positive_scalar(1.0, "x") == 1.0
        assert check_positive_scalar(0.0, "x", allow_zero=True) == 0.0
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="x"):
                check_positive_scalar(bad, "x")

    def test_positive_int(self):
        assert check_positive_int(8, "n") == 8
        assert check_positive_int(0, "n", min_value=0) == 0
        for bad in (-1, 2.5, "8", True):
            with pytest.raises(ValueError, match="n"):
                check_positive_int(bad, "n")
        with pytest.raises(ValueError, match="n"):
            check_positive_int(3, "n", min_value=4)