"""Tests for ``tensorlbm.compile_utils`` — the shared torch.compile layer.

Two layers:

1.  Mode validation + wrapper semantics (CPU, fast):
    ``None`` passthrough, the two proven modes, cudagraph-class /
    unknown modes rejected with a reason, warn-only ``.item()`` sniffing.
2.  Generality: the wrapper is exercised on a **non-SUBOFF pure-tensor
    whole-step example** — a D2Q9 periodic BGK shear-wave decay built
    from the package's own ``solver.collide_bgk``/``solver.stream`` —
    checking eager vs compiled agreement (allclose rtol=1e-4, isfinite)
    and recording (not hard-asserting) the ms/step speedup.

The SUBOFF runners' use of the same shared whitelist is regression-tested
through ``SuboffCmkKbcConfig`` validation (same ValueError behaviour as
before the deduplication into ``tensorlbm.compile_utils``).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import pytest
import torch

from tensorlbm.compile_utils import (
    ALLOWED_COMPILE_MODES,
    compile_step,
    validate_compile_mode,
)

_CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _CUDA, reason="no CUDA device")


# ---------------------------------------------------------------------------
# 1. Mode validation
# ---------------------------------------------------------------------------


class TestValidateCompileMode:
    @pytest.mark.parametrize("mode", [None, "default", "max-autotune-no-cudagraphs"])
    def test_allowed_modes_pass(self, mode: str | None) -> None:
        assert mode in ALLOWED_COMPILE_MODES
        validate_compile_mode(mode)  # must not raise

    @pytest.mark.parametrize(
        "mode",
        [
            "reduce-overhead",  # cudagraphs
            "max-autotune",  # cudagraphs
            "cudagraphs",  # literal
            "turbo",  # unknown
            "",  # junk
            123,  # not a string
        ],
    )
    def test_disallowed_modes_raise(self, mode: Any) -> None:
        with pytest.raises(ValueError, match="compile_mode"):
            validate_compile_mode(mode)

    @pytest.mark.parametrize(
        "mode", ["reduce-overhead", "max-autotune", "cudagraphs"]
    )
    def test_cudagraph_modes_carry_reason(self, mode: str) -> None:
        """Cudagraph-class rejections must state the structural reason."""
        with pytest.raises(ValueError, match="cudagraph replay overwrites"):
            validate_compile_mode(mode)


# ---------------------------------------------------------------------------
# 2. compile_step wrapper semantics (CPU)
# ---------------------------------------------------------------------------


def _double_plus_one(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 + 1.0


def _sync_step(f: torch.Tensor) -> torch.Tensor:
    """Whole-step stand-in whose own source calls .item()."""
    scale = float(f.sum().item())  # host sync — the thing we sniff for
    return f * (1.0 / max(scale, 1.0))


class TestCompileStepWrapper:
    def test_none_returns_same_object(self) -> None:
        fn = _double_plus_one
        assert compile_step(fn, None) is fn

    def test_default_mode_wraps_and_matches_eager(self) -> None:
        compiled = compile_step(_double_plus_one, "default")
        assert compiled is not _double_plus_one
        x = torch.linspace(-1.0, 1.0, 64)
        torch.testing.assert_close(compiled(x), _double_plus_one(x))

    def test_item_sync_source_warns_but_wraps(self) -> None:
        with pytest.warns(UserWarning, match=r"\.item\(\)"):
            compiled = compile_step(_sync_step, "default")
        assert compiled is not _sync_step  # warn-only: still wrapped

    def test_unknown_mode_raises_before_wrapping(self) -> None:
        with pytest.raises(ValueError, match="compile_mode"):
            compile_step(_double_plus_one, "reduce-overhead")

    def test_warmup_hint_is_informational(self) -> None:
        compiled = compile_step(
            _double_plus_one, "default", warmup_hint="one guarded graph"
        )
        assert compiled is not _double_plus_one


# ---------------------------------------------------------------------------
# 3. Generality: non-SUBOFF pure-tensor whole step (D2Q9 periodic BGK)
# ---------------------------------------------------------------------------


def _make_bgk2d_step(tau: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Whole-step D2Q9 periodic BGK evolution from the package solvers.

    Non-SUBOFF, obstacle-free, BC-free: collide (BGK relaxation towards
    equilibrium) + stream (periodic pull).  Pure tensor ops end to end —
    exactly the shape of step that ``compile_step`` targets.
    """
    from tensorlbm.solver import collide_bgk, stream

    def step(f: torch.Tensor) -> torch.Tensor:
        return stream(collide_bgk(f, tau))

    return step


def _shear_wave_init(ny: int, nx: int, device: torch.device) -> torch.Tensor:
    """Decaying shear wave ux = u0 * sin(2*pi*y/ny): a standard LBM unit test."""
    from tensorlbm.d2q9 import equilibrium

    y = torch.arange(ny, device=device, dtype=torch.float32)
    u0 = 0.02
    ux = u0 * torch.sin(2.0 * math.pi * y / ny)
    ones = torch.ones(ny, nx, device=device)
    ux = ux.unsqueeze(1) * ones
    uy = torch.zeros_like(ux)
    rho = torch.ones_like(ux)
    return equilibrium(rho, ux, uy)


def _run_steps(
    step: Callable[[torch.Tensor], torch.Tensor],
    f0: torch.Tensor,
    n: int,
) -> torch.Tensor:
    f = f0.clone()
    for _ in range(n):
        f = step(f)
    return f


def _ms_per_step(
    step: Callable[[torch.Tensor], torch.Tensor],
    f0: torch.Tensor,
    n_warmup: int,
    n_timed: int,
) -> float:
    """CUDA-event ms/step (median-free single window, events on both ends)."""
    f = f0.clone()
    for _ in range(n_warmup):
        f = step(f)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_timed):
        f = step(f)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_timed


class TestPeriodicBgk2DCompiled:
    """Eager vs compiled on a pure-tensor D2Q9 periodic BGK shear wave."""

    NY, NX = 256, 256
    N_STEPS = 20
    TAU = 0.6  # nu = 1/30, comfortably stable at u0 = 0.02

    @pytest.fixture()
    def f0(self) -> torch.Tensor:
        device = torch.device("cuda")
        return _shear_wave_init(self.NY, self.NX, device)

    @requires_cuda
    def test_compiled_matches_eager(self, f0: torch.Tensor) -> None:
        eager = _make_bgk2d_step(self.TAU)
        f_eager = _run_steps(eager, f0, self.N_STEPS)
        f_compiled = _run_steps(compile_step(eager, "default"), f0, self.N_STEPS)

        assert torch.isfinite(f_eager).all().item()
        assert torch.isfinite(f_compiled).all().item()
        assert torch.allclose(f_eager, f_compiled, rtol=1e-4, atol=1e-7), (
            "compiled whole-step drifted from eager beyond rtol=1e-4"
        )
        # A little physics: the shear wave must decay, not blow up.
        from tensorlbm.d2q9 import macroscopic

        _, ux0, _ = macroscopic(f0)
        _, ux_final, _ = macroscopic(f_eager)
        assert ux_final.abs().max() < ux0.abs().max()

    @requires_cuda
    def test_compiled_ms_per_step_recorded(self, f0: torch.Tensor) -> None:
        """Run the compiled path once and record ms/step (no hard speedup assert)."""
        eager = _make_bgk2d_step(self.TAU)
        compiled = compile_step(eager, "default")

        ms_eager = _ms_per_step(eager, f0, n_warmup=10, n_timed=50)
        ms_compiled = _ms_per_step(compiled, f0, n_warmup=10, n_timed=50)

        assert ms_compiled > 0.0 and math.isfinite(ms_compiled)
        speedup = ms_eager / ms_compiled
        # Record only: compile on this launch-bound 2D chain is expected
        # to be >=1x, but the number is machine-dependent, so no assert.
        print(
            f"\nD2Q9 periodic BGK {self.NY}x{self.NX} fp32: "
            f"eager {ms_eager:.3f} ms/step, compiled(default) "
            f"{ms_compiled:.3f} ms/step, speedup {speedup:.2f}x"
        )


# ---------------------------------------------------------------------------
# 4. SUBOFF runners route through the shared whitelist (regression)
# ---------------------------------------------------------------------------


class TestRunnerDedupRegression:
    """Same ValueError surface as before the dedup into compile_utils."""

    def test_config_rejects_cudagraph_mode(self) -> None:
        from tensorlbm.suboff_cmk_kbc_runner import SuboffCmkKbcConfig

        with pytest.raises(ValueError, match="compile_mode"):
            SuboffCmkKbcConfig(compile_mode="reduce-overhead")

    def test_config_rejects_unknown_mode(self) -> None:
        from tensorlbm.suboff_cmk_kbc_runner import SuboffCmkKbcConfig

        with pytest.raises(ValueError, match="compile_mode"):
            SuboffCmkKbcConfig(compile_mode="turbo")

    def test_config_accepts_proven_modes(self) -> None:
        from tensorlbm.suboff_cmk_kbc_runner import SuboffCmkKbcConfig

        for mode in ("default", "max-autotune-no-cudagraphs"):
            cfg = SuboffCmkKbcConfig(compile_mode=mode)
            assert cfg.compile_mode == mode
        assert SuboffCmkKbcConfig().compile_mode is None  # default unchanged

    def test_config_rejects_kbc_with_compile(self) -> None:
        from tensorlbm.suboff_cmk_kbc_runner import SuboffCmkKbcConfig

        with pytest.raises(ValueError, match="KBC"):
            SuboffCmkKbcConfig(collision="KBC", compile_mode="default")
