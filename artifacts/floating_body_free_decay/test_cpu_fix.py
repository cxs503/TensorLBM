#!/usr/bin/env python3
"""Test: compute_retardation_function .cpu() fix.

Verifies that compute_retardation_function works correctly when radiation
tensors are on CPU (the default path exercised by the benchmark), and that
the .cpu() guard is present in the source.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import torch

from tensorlbm.rigid_body_6dof import (
    RadiationData,
    compute_retardation_function,
)


def test_retardation_function_cpu():
    """compute_retardation_function returns correct shape and values on CPU."""
    n_freq = 16
    omega = torch.linspace(0.02, 5.0, n_freq)
    added = torch.zeros(n_freq, 6, 6)
    damping = torch.zeros(n_freq, 6, 6)
    # Constant damping B(ω)=10 on DOF 2 (heave)
    damping[:, 2, 2] = 10.0
    radiation = RadiationData(omega, added, damping, added[0].clone())

    t = torch.linspace(0.0, 10.0, 50)
    K = compute_retardation_function(radiation, t, n_omega_integration=256)

    assert K.shape == (50, 6, 6), f"Expected (50,6,6), got {K.shape}"
    assert torch.all(torch.isfinite(K)), "K contains non-finite values"
    # K(0) should be positive for positive damping (integral of B cos(0) = integral of B)
    assert K[0, 2, 2].item() > 0, f"K(0,2,2) should be > 0, got {K[0,2,2].item()}"
    # Off-diagonal should be zero
    assert K[0, 0, 1].item() == 0.0, "Off-diagonal should be zero"
    print(f"  K.shape={K.shape}, K(0,2,2)={K[0,2,2].item():.4f} ✓")


def test_retardation_function_source_has_cpu_guard():
    """Source code must contain .cpu().numpy() pattern."""
    src = Path("src/tensorlbm/rigid_body_6dof.py").read_text()
    # Check that .cpu().numpy() is used (not bare .numpy())
    assert ".cpu().numpy()" in src, "Source missing .cpu().numpy() guard"
    # Ensure no bare omega.numpy() or B[...].numpy() without .cpu()
    # The specific line should have .cpu() before .numpy()
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if ".numpy()" in line and "omega" in line.lower():
            assert ".cpu()" in line, f"Line {i+1}: bare .numpy() without .cpu(): {line.strip()}"
    print("  Source contains .cpu().numpy() guard ✓")


def test_retardation_fine_grid_interpolation():
    """When n_omega_integration > N_freq, interpolation path is exercised (the fixed code path)."""
    n_freq = 8  # small, so n_omega_integration=256 > 8 triggers interpolation
    omega = torch.linspace(0.0, 3.0, n_freq)
    added = torch.zeros(n_freq, 6, 6)
    damping = torch.zeros(n_freq, 6, 6)
    damping[:, 3, 3] = 50.0  # roll DOF
    radiation = RadiationData(omega, added, damping, added[0].clone())

    t = torch.tensor([0.0, 0.5, 1.0, 2.0])
    K = compute_retardation_function(radiation, t, n_omega_integration=256)

    assert K.shape == (4, 6, 6)
    assert torch.all(torch.isfinite(K))
    # K should be symmetric in the sense that K(t) for constant B is even-ish
    # K(0) > K(2) for decaying kernel
    assert K[0, 3, 3].item() >= K[3, 3, 3].item(), "K should decay"
    print(f"  Interpolation path: K(0,3,3)={K[0,3,3].item():.4f} >= K(2,3,3)={K[3,3,3].item():.4f} ✓")


if __name__ == "__main__":
    print("Running retardation function tests...")
    test_retardation_function_cpu()
    test_retardation_function_source_has_cpu_guard()
    test_retardation_fine_grid_interpolation()
    print("\nAll tests PASSED ✓")
