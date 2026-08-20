"""Tests for ``benchmarks/compile_route`` — the benchmark-side adapter over
``tensorlbm.compile_utils``.

Layers:

1.  Mode normalisation + eager passthrough identity (CPU, fast): the CLI
    spelling ``"eager"`` maps to the canonical ``None``; cudagraph-class and
    unknown modes raise the *shared* ``validate_compile_mode`` ValueError;
    ``mode=None`` returns the very same function object (byte-identical
    eager path).
2.  Argparse plumbing: ``add_compile_mode_arg``/``compile_mode_from_args``
    round-trip the three CLI spellings.
3.  Numerics: a small D2Q9 whole-step chain routed through
    ``route_step(mode="default")`` agrees with the eager routing on the
    same device (allclose) — run on CPU by default (small grid), which also
    proves the wrapper is device-agnostic like ``compile_utils`` itself.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_compile_route():
    """Import benchmarks/compile_route.py (not a package member of tests)."""
    spec = importlib.util.spec_from_file_location(
        "benchmarks_compile_route", _REPO_ROOT / "benchmarks" / "compile_route.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("benchmarks_compile_route", mod)
    spec.loader.exec_module(mod)
    return mod


cr = _load_compile_route()


# ---------------------------------------------------------------------------
# 1. Mode normalisation
# ---------------------------------------------------------------------------


class TestNormalizeCompileMode:
    def test_eager_spellings_map_to_none(self) -> None:
        assert cr.normalize_compile_mode("eager") is None
        assert cr.normalize_compile_mode("EAGER") is None
        assert cr.normalize_compile_mode("") is None
        assert cr.normalize_compile_mode(None) is None

    def test_proven_modes_pass_through(self) -> None:
        assert cr.normalize_compile_mode("default") == "default"
        assert (
            cr.normalize_compile_mode("max-autotune-no-cudagraphs") == "max-autotune-no-cudagraphs"
        )

    @pytest.mark.parametrize(
        "mode", ["reduce-overhead", "max-autotune", "cudagraphs", "turbo", 123]
    )
    def test_bad_modes_raise_shared_valueerror(self, mode) -> None:
        with pytest.raises(ValueError, match="compile_mode"):
            cr.normalize_compile_mode(mode)

    def test_cudagraph_reason_forwarded(self) -> None:
        with pytest.raises(ValueError, match="cudagraph"):
            cr.normalize_compile_mode("reduce-overhead")


class TestRouteStep:
    def test_eager_passthrough_is_identity(self) -> None:
        def step(f):
            return f + 1.0

        assert cr.route_step(step, None, name="t", quiet=True) is step
        assert cr.route_step(step, "eager", name="t", quiet=True) is step

    def test_default_returns_compiled_wrapper(self) -> None:
        def step(f):
            return f + 1.0

        wrapped = cr.route_step(step, "default", name="t", quiet=True)
        assert wrapped is not step
        assert callable(wrapped)


# ---------------------------------------------------------------------------
# 2. Argparse plumbing
# ---------------------------------------------------------------------------


class TestArgparsePlumbing:
    def test_roundtrip_default_is_compiled(self) -> None:
        ap = argparse.ArgumentParser()
        cr.add_compile_mode_arg(ap)
        args = ap.parse_args([])
        assert args.compile_mode == "default"  # new benchmark standard
        assert cr.compile_mode_from_args(args) == "default"

    def test_roundtrip_eager_and_autotune(self) -> None:
        ap = argparse.ArgumentParser()
        cr.add_compile_mode_arg(ap)
        args = ap.parse_args(["--compile-mode", "eager"])
        assert cr.compile_mode_from_args(args) is None
        args = ap.parse_args(["--compile-mode", "max-autotune-no-cudagraphs"])
        assert cr.compile_mode_from_args(args) == "max-autotune-no-cudagraphs"

    def test_bad_cli_spelling_rejected_by_argparse(self) -> None:
        ap = argparse.ArgumentParser()
        cr.add_compile_mode_arg(ap)
        with pytest.raises(SystemExit):
            ap.parse_args(["--compile-mode", "reduce-overhead"])


# ---------------------------------------------------------------------------
# 3. Numerics: routed whole-step vs eager on the same device
# ---------------------------------------------------------------------------


def _make_step(device: torch.device):
    """Small D2Q9 whole-step chain from the package's own primitives."""
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.solver import collide_bgk, stream

    ny = nx = 16
    tau = 0.8
    y = torch.arange(ny, device=device, dtype=torch.float32)
    x = torch.arange(nx, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    f0 = equilibrium(
        torch.ones((ny, nx), device=device),
        0.05 * torch.sin(2 * torch.pi * yy / ny),
        torch.zeros((ny, nx), device=device),
    )

    def _step(f):
        return collide_bgk(stream(f), tau)

    return f0, _step


class TestRoutedStepNumerics:
    def test_default_mode_matches_eager_cpu(self) -> None:
        device = torch.device("cpu")
        f_eager, step = _make_step(device)
        f_comp, _ = _make_step(device)

        eager_fn = cr.route_step(step, None, name="num-test", quiet=True)
        comp_fn = cr.route_step(step, "default", name="num-test", quiet=True)
        for _ in range(10):
            f_eager = eager_fn(f_eager)
            f_comp = comp_fn(f_comp)
        torch.testing.assert_close(f_eager, f_comp, rtol=1e-4, atol=1e-6)
        assert torch.isfinite(f_comp).all()
