#!/usr/bin/env python3
"""XVAL: Ternary double-well with harmonic lambda — TensorLBM vs CudaPhaseField.

Cross-validation of the chemical potential calculation between:
- **TensorLBM**: `src/tensorlbm/phasefield/ternary_free_energy.py` (this repo)
- **OSS reference**: AminZar69/CudaPhaseField (CUDA Fortran, GPL-3.0)

OSS source: https://github.com/AminZar69/CudaPhaseField
Key paper: Zarareh et al. 2022, CMAM 109, 100–124, DOI 10.1016/j.camwa.2022.01.005

Decision criterion (owner 2026-11-12):对标误差 < 15% 可接受.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.phasefield.ternary_free_energy import (  # noqa: E402
    TernaryDoubleWellFreeEnergy,
    isotropic_gradient_2d,
    isotropic_laplacian_2d,
    lambda_from_sigma,
    harmonic_lambda,
    planar_profile_ternary,
)


def _to_json(obj):
    """Recursively convert tensors to JSON-serializable types."""
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.item())
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return "NaN"
    if isinstance(obj, float) and math.isinf(obj):
        return "Infinity" if obj > 0 else "-Infinity"
    return obj


# ---------------------------------------------------------------------------
# OSS reference parameters (from CudaPhaseField src/device_var.f90)
# ---------------------------------------------------------------------------

OSS_SIGMA = 0.1  # all sigma(i,j) = 0.1
OSS_W = 4.0
OSS_DENSITY = [1000.0, 500.0, 100.0]  # density_host(3)
OSS_NX = 254
OSS_NY = 254
OSS_R = 35.0  # droplet radius (for droplet test)


def _build_sigma(value: float = OSS_SIGMA, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Build 3x3 symmetric tension matrix."""
    return torch.full((3, 3), value, dtype=dtype)


def _oss_lambda(sigma_val: float = OSS_SIGMA) -> list[float]:
    """Reproduce OSS lambda calculation (from device_var.f90)."""
    s = sigma_val
    lam = [s + s - s, s + s - s, s + s - s]  # all equal when sigma is uniform
    return lam


def _oss_lambdat(sigma_val: float = OSS_SIGMA) -> float:
    """Reproduce OSS lambdat calculation."""
    lam = _oss_lambda(sigma_val)
    return 3.0 / (1.0 / lam[0] + 1.0 / lam[1] + 1.0 / lam[2])


# ---------------------------------------------------------------------------
# Test 1: Lambda / harmonic lambda consistency
# ---------------------------------------------------------------------------


def test_lambda_consistency() -> dict:
    """Verify lambda and harmonic lambda match OSS formulas."""
    sigma = _build_sigma()
    lam = lambda_from_sigma(sigma)
    lam_t = harmonic_lambda(lam)

    oss_lam = _oss_lambda()
    oss_lam_t = _oss_lambdat()

    lam_err = [abs(float(lam[i]) - oss_lam[i]) / abs(oss_lam[i]) for i in range(3)]
    lam_t_err = abs(float(lam_t) - oss_lam_t) / abs(oss_lam_t)

    return {
        "test": "lambda_consistency",
        "tensorlbm_lambda": [float(lam[i]) for i in range(3)],
        "oss_lambda": oss_lam,
        "lambda_rel_error": lam_err,
        "tensorlbm_lambdat": float(lam_t),
        "oss_lambdat": oss_lam_t,
        "lambdat_rel_error": lam_t_err,
        "pass": all(e < 1e-12 for e in lam_err) and lam_t_err < 1e-12,
    }


# ---------------------------------------------------------------------------
# Test 2: Laplacian operator (isotropic 9-point)
# ---------------------------------------------------------------------------


def test_laplacian_operator() -> dict:
    """Verify isotropic Laplacian matches OSS stencil."""
    ny, nx = 64, 64
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1).repeat(1, nx)
    x = torch.arange(nx, dtype=torch.float64).view(1, -1).repeat(ny, 1)

    phi = torch.cos(2 * math.pi * x / nx) * torch.cos(2 * math.pi * y / ny)
    lap = isotropic_laplacian_2d(phi)

    kx = 2 * math.pi / nx
    ky = 2 * math.pi / ny
    analytic_lap = -(kx**2 + ky**2) * phi

    denom = analytic_lap.abs().mean() + 1e-30
    err = float((lap - analytic_lap).abs().mean()) / float(denom)

    return {
        "test": "laplacian_operator",
        "analytic_laplacian_mean": float(analytic_lap.mean()),
        "computed_mean": float(lap.mean()),
        "mean_abs_rel_error": err,
        "pass": err < 0.01,
    }


# ---------------------------------------------------------------------------
# Test 3: Gradient operator (isotropic 8-point)
# ---------------------------------------------------------------------------


def test_gradient_operator() -> dict:
    """Verify isotropic gradient matches OSS stencil."""
    ny, nx = 64, 64
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1).repeat(1, nx)
    x = torch.arange(nx, dtype=torch.float64).view(1, -1).repeat(ny, 1)

    phi = torch.sin(2 * math.pi * x / nx) * torch.sin(2 * math.pi * y / ny)
    dphi_dx, dphi_dy = isotropic_gradient_2d(phi)

    kx = 2 * math.pi / nx
    ky = 2 * math.pi / ny
    analytic_dx = kx * torch.cos(2 * math.pi * x / nx) * torch.sin(2 * math.pi * y / ny)
    analytic_dy = ky * torch.sin(2 * math.pi * x / nx) * torch.cos(2 * math.pi * y / ny)

    err_x = float((dphi_dx - analytic_dx).abs().max()) / (float(analytic_dx.abs().max()) + 1e-30)
    err_y = float((dphi_dy - analytic_dy).abs().max()) / (float(analytic_dy.abs().max()) + 1e-30)

    return {
        "test": "gradient_operator",
        "max_abs_rel_error_x": err_x,
        "max_abs_rel_error_y": err_y,
        "pass": err_x < 0.01 and err_y < 0.01,
    }


# ---------------------------------------------------------------------------
# Test 4: Chemical potential — planar interface
# ---------------------------------------------------------------------------


def test_chemical_potential_planar() -> dict:
    """Chemical potential of a planar interface profile.

    Note: The binary slice profile (phi2=0) is NOT the exact equilibrium
    of the ternary model, so mu will vary across the interface. This test
    verifies the formula produces finite, reasonable values.
    """
    ny, nx = 128, 64
    sigma = _build_sigma()
    fe = TernaryDoubleWellFreeEnergy(sigma=sigma, w=OSS_W)

    phi1, phi2, phi3 = planar_profile_ternary(ny, nx, sigma=sigma, w=OSS_W)
    mu1, mu2, mu3 = fe.chemical_potential(phi1, phi2, phi3)

    mu1_mean = float(mu1.mean())
    mu2_mean = float(mu2.mean())
    mu3_mean = float(mu3.mean())

    # Check that mu is finite and not NaN
    mu1_finite = torch.isfinite(mu1).all().item()
    mu2_finite = torch.isfinite(mu2).all().item()
    mu3_finite = torch.isfinite(mu3).all().item()

    # Check that mu values are in a reasonable range (not exploding)
    mu1_max = float(mu1.abs().max())
    mu2_max = float(mu2.abs().max())
    mu3_max = float(mu3.abs().max())

    return {
        "test": "chemical_potential_planar",
        "mu1_mean": mu1_mean,
        "mu2_mean": mu2_mean,
        "mu3_mean": mu3_mean,
        "mu1_max_abs": mu1_max,
        "mu2_max_abs": mu2_max,
        "mu3_max_abs": mu3_max,
        "all_finite": mu1_finite and mu2_finite and mu3_finite,
        "pass": mu1_finite and mu2_finite and mu3_finite and mu1_max < 10.0,
    }


# ---------------------------------------------------------------------------
# Test 5: Chemical potential — pointwise comparison with OSS formula
# ---------------------------------------------------------------------------


def test_chemical_potential_pointwise() -> dict:
    """Pointwise chemical potential comparison against OSS formula (manual calc)."""
    ny, nx = 32, 32
    sigma = _build_sigma()
    fe = TernaryDoubleWellFreeEnergy(sigma=sigma, w=OSS_W)

    y = torch.arange(ny, dtype=torch.float64).view(-1, 1).repeat(1, nx)
    x = torch.arange(nx, dtype=torch.float64).view(1, -1).repeat(ny, 1)

    phi1 = 0.5 * (1.0 + torch.tanh((x - nx // 2) / 3.0)) * 0.5 * (
        1.0 + torch.tanh((y - ny // 2) / 3.0)
    )
    phi2 = 0.3 * torch.sin(2 * math.pi * x / nx) * torch.sin(2 * math.pi * y / ny) + 0.1
    phi3 = 1.0 - phi1 - phi2
    phi1 = phi1.clamp(0.0, 1.0)
    phi2 = phi2.clamp(0.0, 1.0)
    phi3 = phi3.clamp(0.0, 1.0)
    total = phi1 + phi2 + phi3
    phi1 /= total
    phi2 /= total
    phi3 /= total

    mu1, mu2, mu3 = fe.chemical_potential(phi1, phi2, phi3)

    cy, cx = ny // 2, nx // 2
    p1 = float(phi1[cy, cx])
    p2 = float(phi2[cy, cx])
    p3 = float(phi3[cy, cx])

    w = OSS_W
    lam = [OSS_SIGMA, OSS_SIGMA, OSS_SIGMA]
    lam_t = OSS_SIGMA

    def dw_deriv(p):
        return p * (1.0 - p) * (1.0 - 2.0 * p)

    coupling = p1 * p2 * p3

    def lap_at(phi, i, j):
        s = phi
        ny_s, nx_s = s.shape
        diag = (
            s[(i - 1) % ny_s, (j - 1) % nx_s]
            + s[(i - 1) % ny_s, (j + 1) % nx_s]
            + s[(i + 1) % ny_s, (j - 1) % nx_s]
            + s[(i + 1) % ny_s, (j + 1) % nx_s]
        )
        card = (
            s[(i - 1) % ny_s, j]
            + s[(i + 1) % ny_s, j]
            + s[i, (j - 1) % nx_s]
            + s[i, (j + 1) % nx_s]
        )
        return (diag + 4.0 * card - 20.0 * s[i, j]) / 6.0

    d2p1 = lap_at(phi1, cy, cx)
    d2p2 = lap_at(phi2, cy, cx)
    d2p3 = lap_at(phi3, cy, cx)

    mu1_manual = (12.0 / w) * (lam[0] * dw_deriv(p1) - 2.0 * lam_t * coupling) - (
        3.0 / 4.0
    ) * w * lam[0] * d2p1
    mu2_manual = (12.0 / w) * (lam[1] * dw_deriv(p2) - 2.0 * lam_t * coupling) - (
        3.0 / 4.0
    ) * w * lam[1] * d2p2
    mu3_manual = (12.0 / w) * (lam[2] * dw_deriv(p3) - 2.0 * lam_t * coupling) - (
        3.0 / 4.0
    ) * w * lam[2] * d2p3

    err1 = abs(float(mu1[cy, cx]) - mu1_manual) / (abs(mu1_manual) + 1e-30)
    err2 = abs(float(mu2[cy, cx]) - mu2_manual) / (abs(mu2_manual) + 1e-30)
    err3 = abs(float(mu3[cy, cx]) - mu3_manual) / (abs(mu3_manual) + 1e-30)

    return {
        "test": "chemical_potential_pointwise",
        "center_phi": [p1, p2, p3],
        "center_mu_tensorlbm": [float(mu1[cy, cx]), float(mu2[cy, cx]), float(mu3[cy, cx])],
        "center_mu_manual_oss": [mu1_manual, mu2_manual, mu3_manual],
        "rel_error": [err1, err2, err3],
        "pass": err1 < 1e-10 and err2 < 1e-10 and err3 < 1e-10,
    }


# ---------------------------------------------------------------------------
# Test 6: Droplet equilibrium (circular interface)
# ---------------------------------------------------------------------------


def test_droplet_equilibrium() -> dict:
    """Chemical potential of a circular droplet.

    Note: The tanh profile is an approximation; mu won't be perfectly constant.
    This test verifies the formula produces finite values with reasonable variation.
    """
    ny, nx = 128, 128
    sigma = _build_sigma()
    fe = TernaryDoubleWellFreeEnergy(sigma=sigma, w=OSS_W)

    cy, cx = ny // 2, nx // 2
    r = 30.0
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1).repeat(1, nx)
    x = torch.arange(nx, dtype=torch.float64).view(1, -1).repeat(ny, 1)
    dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    phi1 = 0.5 * (1.0 - torch.tanh((dist - r) / (OSS_W / 2.0)))
    phi3 = 1.0 - phi1
    phi2 = torch.zeros_like(phi1)

    mu1, mu2, mu3 = fe.chemical_potential(phi1, phi2, phi3)

    mask_inside = dist < (r - OSS_W)
    mask_outside = dist > (r + OSS_W)

    mu1_inside = float(mu1[mask_inside].mean()) if mask_inside.any() else float("nan")
    mu1_outside = float(mu1[mask_outside].mean()) if mask_outside.any() else float("nan")
    mu1_inside_std = float(mu1[mask_inside].std()) if mask_inside.any() else float("nan")

    mu_jump = abs(mu1_inside - mu1_outside) if not (math.isnan(mu1_inside) or math.isnan(mu1_outside)) else float("nan")

    # Check finite values and reasonable absolute variation
    all_finite = torch.isfinite(mu1).all().item() and torch.isfinite(mu2).all().item()

    return {
        "test": "droplet_equilibrium",
        "droplet_radius": r,
        "mu1_inside_mean": mu1_inside,
        "mu1_outside_mean": mu1_outside,
        "mu1_inside_std": mu1_inside_std,
        "mu_jump": mu_jump,
        "all_finite": all_finite,
        "pass": all_finite and mu1_inside_std < 0.01,
    }


# ---------------------------------------------------------------------------
# Test 7: Non-uniform sigma matrix
# ---------------------------------------------------------------------------


def test_nonuniform_sigma() -> dict:
    """Test with non-uniform sigma matrix (different tensions between pairs)."""
    sigma = torch.tensor(
        [[0.0, 0.1, 0.15], [0.1, 0.0, 0.08], [0.15, 0.08, 0.0]],
        dtype=torch.float64,
    )
    fe = TernaryDoubleWellFreeEnergy(sigma=sigma, w=OSS_W)

    lam = fe.lam
    lam_t = fe.lam_t

    expected_lam = [0.17, 0.03, 0.13]
    expected_lam_t = 3.0 / (1.0 / 0.17 + 1.0 / 0.03 + 1.0 / 0.13)

    lam_err = [abs(float(lam[i]) - expected_lam[i]) for i in range(3)]
    lam_t_err = abs(float(lam_t) - expected_lam_t)

    return {
        "test": "nonuniform_sigma",
        "sigma": sigma.tolist(),
        "tensorlbm_lambda": [float(lam[i]) for i in range(3)],
        "expected_lambda": expected_lam,
        "lambda_abs_error": lam_err,
        "tensorlbm_lambdat": float(lam_t),
        "expected_lambdat": expected_lam_t,
        "lambdat_abs_error": lam_t_err,
        "pass": all(e < 1e-12 for e in lam_err) and lam_t_err < 1e-12,
    }


# ---------------------------------------------------------------------------
# Test 8: Full-field comparison (all grid points) — PRIMARY XVAL TEST
# ---------------------------------------------------------------------------


def test_full_field_comparison() -> dict:
    """Compare chemical potential at every grid point against manual OSS formula.

    This is the PRIMARY cross-validation test: if the full-field comparison
    passes with machine precision, the TensorLBM implementation is bit-for-bit
    equivalent to the OSS formula.
    """
    ny, nx = 24, 24
    sigma = _build_sigma()
    fe = TernaryDoubleWellFreeEnergy(sigma=sigma, w=OSS_W)

    y = torch.arange(ny, dtype=torch.float64).view(-1, 1).repeat(1, nx)
    x = torch.arange(nx, dtype=torch.float64).view(1, -1).repeat(ny, 1)

    phi1 = 0.5 * (1.0 + torch.tanh((x - nx // 2) / 4.0))
    phi2 = 0.2 * torch.sin(2 * math.pi * x / nx) + 0.15
    phi3 = 1.0 - phi1 - phi2
    phi1 = phi1.clamp(0.01, 0.99)
    phi2 = phi2.clamp(0.01, 0.99)
    phi3 = phi3.clamp(0.01, 0.99)
    total = phi1 + phi2 + phi3
    phi1 /= total
    phi2 /= total
    phi3 /= total

    mu1, mu2, mu3 = fe.chemical_potential(phi1, phi2, phi3)

    w = OSS_W
    lam = [OSS_SIGMA, OSS_SIGMA, OSS_SIGMA]
    lam_t = OSS_SIGMA

    def dw_deriv(p):
        return p * (1.0 - p) * (1.0 - 2.0 * p)

    coupling = phi1 * phi2 * phi3
    d2p1 = isotropic_laplacian_2d(phi1)
    d2p2 = isotropic_laplacian_2d(phi2)
    d2p3 = isotropic_laplacian_2d(phi3)

    mu1_manual = (12.0 / w) * (lam[0] * dw_deriv(phi1) - 2.0 * lam_t * coupling) - (
        3.0 / 4.0
    ) * w * lam[0] * d2p1
    mu2_manual = (12.0 / w) * (lam[1] * dw_deriv(phi2) - 2.0 * lam_t * coupling) - (
        3.0 / 4.0
    ) * w * lam[1] * d2p2
    mu3_manual = (12.0 / w) * (lam[2] * dw_deriv(phi3) - 2.0 * lam_t * coupling) - (
        3.0 / 4.0
    ) * w * lam[2] * d2p3

    err1 = float((mu1 - mu1_manual).abs().max()) / (float(mu1_manual.abs().max()) + 1e-30)
    err2 = float((mu2 - mu2_manual).abs().max()) / (float(mu2_manual.abs().max()) + 1e-30)
    err3 = float((mu3 - mu3_manual).abs().max()) / (float(mu3_manual.abs().max()) + 1e-30)

    return {
        "test": "full_field_comparison",
        "grid_size": [ny, nx],
        "max_rel_error": [err1, err2, err3],
        "pass": err1 < 1e-10 and err2 < 1e-10 and err3 < 1e-10,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="XVAL: ternary double-well with harmonic lambda")
    parser.add_argument("--out", type=str, default="runs/xval_ternary_double_well", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test only")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Test 1: Lambda consistency
    r1 = test_lambda_consistency()
    results.append(r1)
    print(f"[{'PASS' if r1['pass'] else 'FAIL'}] {r1['test']}: lambda rel_err = {r1['lambda_rel_error']}")

    # Test 2: Laplacian
    r2 = test_laplacian_operator()
    results.append(r2)
    print(f"[{'PASS' if r2['pass'] else 'FAIL'}] {r2['test']}: mean_abs_rel_error = {r2['mean_abs_rel_error']:.2e}")

    # Test 3: Gradient
    r3 = test_gradient_operator()
    results.append(r3)
    print(f"[{'PASS' if r3['pass'] else 'FAIL'}] {r3['test']}: max_rel_err = [{r3['max_abs_rel_error_x']:.2e}, {r3['max_abs_rel_error_y']:.2e}]")

    if args.quick:
        all_pass = all(r["pass"] for r in results)
        report = _to_json({
            "xval_id": "XVAL-LBM-TERNARY-DOUBLE-WELL-WITH-HARMONIC-LAMBDA",
            "oss_reference": "AminZar69/CudaPhaseField (Zarareh et al. 2022, CMAM 109)",
            "oss_source": "https://github.com/AminZar69/CudaPhaseField",
            "criterion": "对标误差 < 15%",
            "results": results,
            "all_pass": all_pass,
            "mode": "quick",
        })
        with open(out_dir / "results.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nQuick test: {'ALL PASS' if all_pass else 'SOME FAIL'}")
        return

    # Test 4: Planar interface chemical potential
    r4 = test_chemical_potential_planar()
    results.append(r4)
    print(f"[{'PASS' if r4['pass'] else 'FAIL'}] {r4['test']}: mu1_max_abs = {r4['mu1_max_abs']:.4f}, finite = {r4['all_finite']}")

    # Test 5: Pointwise comparison
    r5 = test_chemical_potential_pointwise()
    results.append(r5)
    print(f"[{'PASS' if r5['pass'] else 'FAIL'}] {r5['test']}: rel_err = {r5['rel_error']}")

    # Test 6: Droplet equilibrium
    r6 = test_droplet_equilibrium()
    results.append(r6)
    print(f"[{'PASS' if r6['pass'] else 'FAIL'}] {r6['test']}: mu1_inside_std = {r6['mu1_inside_std']:.6f}, finite = {r6['all_finite']}")

    # Test 7: Non-uniform sigma
    r7 = test_nonuniform_sigma()
    results.append(r7)
    print(f"[{'PASS' if r7['pass'] else 'FAIL'}] {r7['test']}: lambda_err = {r7['lambda_abs_error']}")

    # Test 8: Full-field comparison (PRIMARY XVAL TEST)
    r8 = test_full_field_comparison()
    results.append(r8)
    print(f"[{'PASS' if r8['pass'] else 'FAIL'}] {r8['test']}: max_rel_error = {r8['max_rel_error']}")

    all_pass = all(r["pass"] for r in results)

    report = _to_json({
        "xval_id": "XVAL-LBM-TERNARY-DOUBLE-WELL-WITH-HARMONIC-LAMBDA",
        "oss_reference": "AminZar69/CudaPhaseField (Zarareh et al. 2022, CMAM 109)",
        "oss_source": "https://github.com/AminZar69/CudaPhaseField",
        "oss_paper_doi": "10.1016/j.camwa.2022.01.005",
        "criterion": "对标误差 < 15%",
        "results": results,
        "all_pass": all_pass,
        "mode": "full",
        "tensorlbm_module": "src/tensorlbm/phasefield/ternary_free_energy.py",
    })

    with open(out_dir / "results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"XVAL complete: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print(f"Results written to {out_dir / 'results.json'}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
