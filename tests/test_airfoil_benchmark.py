"""Tests for airfoil benchmark module."""
from __future__ import annotations

import torch
from tensorlbm.airfoil_benchmark import (
    AirfoilConfig,
    naca4_surface,
    build_airfoil_mask,
    run_airfoil_benchmark,
    reference_cl_cd,
)


def test_naca4_surface_symmetric() -> None:
    xc = torch.linspace(0.0, 1.0, 50)
    y_upper, y_lower = naca4_surface(xc, m=0.0, p=0.0, t=0.12)
    assert torch.allclose(y_upper, -y_lower)
    assert y_upper.max() > 0.04  # ~6% half-thickness
    assert y_upper.argmax() > 10  # max thickness near 30% chord


def test_build_airfoil_mask() -> None:
    mask = build_airfoil_mask(nx=100, ny=40, chord=30, alpha_deg=4.0)
    assert mask.shape == (40, 100)
    solid = mask.sum().item()
    assert solid > 0
    assert solid < 2000  # ~30*4 cells


def test_reference_cl_cd() -> None:
    ref0 = reference_cl_cd(0.0, 1000.0)
    assert abs(ref0["cl"]) < 0.01
    assert ref0["cd"] > 0.0

    ref4 = reference_cl_cd(4.0, 1000.0)
    assert ref4["cl"] > 0.1  # positive lift
    assert ref4["cd"] > 0.05


def test_airfoil_benchmark_runs() -> None:
    """Minimal smoke test."""
    cfg = AirfoilConfig(
        chord=30, alpha_deg=2.0, re=100,
        nx=100, ny=40, n_steps=200, warmup_steps=50,
        device="cpu",
    )
    result = run_airfoil_benchmark(cfg)
    assert "cl_sim" in result
    assert "cd_sim" in result
    assert abs(result["cl_sim"]) < 10.0  # sanity
    assert result["cd_sim"] > 0.0


__all__ = [
    "test_naca4_surface_symmetric",
    "test_build_airfoil_mask",
    "test_reference_cl_cd",
    "test_airfoil_benchmark_runs",
]
