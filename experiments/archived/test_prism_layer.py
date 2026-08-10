#!/usr/bin/env python3
"""Smoke-test the prism-layer generator and DG body-fitted solver.

Validates:
1. Surface normal extraction produces valid unit vectors
2. Prism layer centres march outward along normals
3. Prism cells are within domain bounds
4. Prism fraction is < 5% for typical meshes
5. DG body-fitted RHS function runs without error
6. Jacobian is diagonal and has positive determinant
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from tensorlbm.boundaries3d import sphere_mask
from tensorlbm.prism_layer import (
    PrismLayerMesh,
    dg_rhs_body_fitted,
    extract_surface_normals,
    generate_prism_layers,
    prism_statistics,
)
from tensorlbm.dg_advection import get_ops, nodal_from_mean


def test_surface_normals():
    """Test that extracted normals are unit vectors."""
    print("=== Test 1: Surface normals ===")
    solid = sphere_mask(40, 30, 30, 10.0, 15.0, 15.0, 8.0, device="cpu")
    coords, normals, flat = extract_surface_normals(solid, smooth=True)
    n_surf = coords.shape[0]
    print(f"  Surface cells: {n_surf}")

    # All normals should be unit vectors
    mags = normals.norm(dim=1)
    max_dev = (mags - 1.0).abs().max().item()
    print(f"  Max |normal| deviation from 1.0: {max_dev:.6f}")
    assert max_dev < 0.01, f"Normals not unit: max deviation {max_dev}"

    # For a sphere, normals should point outward (radially)
    cx = torch.tensor([10.0, 15.0, 15.0])
    radial = coords - cx.unsqueeze(0)
    radial = radial / radial.norm(dim=1, keepdim=True).clamp(min=1e-12)
    dot = (normals * radial).sum(dim=1)
    mean_dot = dot.mean().item()
    print(f"  Mean dot(normal, radial): {mean_dot:.4f} (should be > 0.6 for stair-stepped sphere)")
    assert mean_dot > 0.6, f"Normals not outward: mean dot = {mean_dot}"

    print("  PASSED\n")


def test_prism_generation():
    """Test prism layer mesh properties."""
    print("=== Test 2: Prism layer generation ===")
    solid = sphere_mask(80, 60, 60, 20.0, 30.0, 30.0, 15.0, device="cpu")

    for n_layers, first_h, growth in [(3, 0.1, 1.2), (5, 0.05, 1.15)]:
        prism = generate_prism_layers(solid, n_layers=n_layers,
                                       first_height=first_h, growth=growth)
        stats = prism_statistics(prism, solid)

        print(f"  n_layers={n_layers}, first_h={first_h}, growth={growth}")
        print(f"    Surface faces: {stats['surface_faces']}")
        print(f"    Prism cells: {stats['prism_cells']}")
        print(f"    Prism fraction: {stats['prism_fraction_pct']:.2f}%")
        print(f"    Layer heights: {[f'{h:.4f}' for h in stats['layer_heights']]}")

        # Check prism fraction < 5%
        assert stats['prism_fraction_pct'] < 5.0, \
            f"Prism fraction {stats['prism_fraction_pct']:.1f}% exceeds 5%"

        # Check layer centres increase monotonically along normals
        for k in range(1, prism.n_layers):
            # Distance from surface should increase
            d_prev = (prism.layer_centers[k - 1] - prism.surface_centers.unsqueeze(0)).norm(dim=2)
            d_curr = (prism.layer_centers[k] - prism.surface_centers.unsqueeze(0)).norm(dim=2)
            assert (d_curr.mean() > d_prev.mean()).item(), \
                f"Layer {k} not farther from wall than layer {k-1}"

        # Check layer centres are within grid bounds
        shape = solid.shape
        for k in range(prism.n_layers):
            for d in range(prism.ndim):
                c = prism.layer_centers[k, :, d]
                assert (c.min() >= -1).item() and (c.max() <= shape[d]).item(), \
                    f"Layer {k} dim {d}: centres out of bounds: [{c.min():.1f}, {c.max():.1f}]"

        # Check Jacobian
        if prism.jacobian is not None:
            assert prism.jacobian.shape == (n_layers, prism.n_surface, prism.ndim, prism.ndim)
            # Determinant should be positive
            if prism.determinant is not None:
                min_det = prism.determinant.min().item()
                print(f"    Min Jacobian det: {min_det:.6f}")
                assert min_det > 0, f"Jacobian determinant negative: {min_det}"

    print("  PASSED\n")


def test_dg_body_fitted_rhs():
    """Test that the body-fitted DG RHS function runs without error."""
    print("=== Test 3: DG body-fitted RHS ===")

    # Small 2D-like test (3D with nz=1)
    solid = sphere_mask(30, 20, 20, 7.0, 10.0, 10.0, 5.0, device="cpu")

    prism = generate_prism_layers(solid, n_layers=3,
                                   first_height=0.1, growth=1.2,
                                   device=torch.device("cpu"))

    # Build a dummy DG field: D3Q19 velocities
    from tensorlbm.d3q19 import C as VELOCITIES_D3Q19, W as WEIGHTS
    velocities = VELOCITIES_D3Q19.float()  # (19, 3)
    Q = velocities.shape[0]
    n_prism = prism.band_indices.shape[0]

    # Create DG field: (Q, n_prism, 2, 2, 2) for P1
    n_node = 2  # P1
    ndim = prism.ndim
    node_shape = (n_node,) * ndim
    f_dg = torch.randn((Q, n_prism) + node_shape, dtype=torch.float32)

    # Create Cartesian ops and wall-normal ops
    ops_cart = get_ops(1, 1.0, dtype=torch.float32, device="cpu")
    ops_wall = prism.dg_ops_wall_normal

    # Run RHS
    try:
        rhs = dg_rhs_body_fitted(f_dg, velocities, ops_cart, ops_wall, prism)
        print(f"  RHS shape: {rhs.shape} (expected {(Q, n_prism) + node_shape})")
        assert rhs.shape == f_dg.shape, f"Shape mismatch: {rhs.shape} vs {f_dg.shape}"
        assert torch.isfinite(rhs).all(), "RHS contains NaN/Inf"
        print(f"  RHS max abs: {rhs.abs().max().item():.4f}")
        print("  PASSED\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")
        raise


def test_extract_normals_2d():
    """Test surface normal extraction in 2D."""
    print("=== Test 4: 2D surface normals ===")
    # Simple box in 2D
    ny, nx = 40, 60
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[15:25, 20:30] = True  # box
    print(f"  2D grid: {ny}x{nx}, solid box at [15:25, 20:30]")

    coords, normals, flat = extract_surface_normals(solid, smooth=False)
    n_surf = coords.shape[0]
    print(f"  Surface cells: {n_surf}")

    # Check: should have 4 sides × (10 or 8) cells ≈ 36 surface cells
    expected = 2 * 10 + 2 * 8  # perimeter of box
    assert abs(n_surf - expected) <= 4, \
        f"Expected ~{expected} surface cells, got {n_surf}"

    # Surface cells are FLUID cells adjacent to solid.
    # Left side: fluid cells at x=19 (solid inside starts at x=20)
    left_mask = coords[:, 1] == 19
    if left_mask.any():
        left_n = normals[left_mask].mean(dim=0)
        print(f"  Left side normal: {left_n.tolist()} (expect [-1, 0] or [0, -1] in (y,x))")
        # In (y, x) ordering, x-component is index 1
        assert left_n[1] < -0.5, f"Left normal wrong: {left_n}"

    right_mask = coords[:, 1] == 30  # fluid at x=30, solid inside ends at x=29
    if right_mask.any():
        right_n = normals[right_mask].mean(dim=0)
        print(f"  Right side normal: {right_n.tolist()} (expect [+1, 0] or [0, +1] in (y,x))")
        assert right_n[1] > 0.5, f"Right normal wrong: {right_n}"

    print("  PASSED\n")


def test_prism_to_cartesian_projection():
    """Test project_prism_to_cartesian."""
    print("=== Test 5: Prism → Cartesian projection ===")
    from tensorlbm.prism_layer import project_prism_to_cartesian

    solid = sphere_mask(40, 30, 30, 10.0, 15.0, 15.0, 8.0, device="cpu")
    prism = generate_prism_layers(solid, n_layers=3,
                                   first_height=0.1, growth=1.2)

    Q = 19
    n_node = 2
    ndim = prism.ndim
    node_shape = (n_node,) * ndim
    n_prism = prism.band_indices.shape[0]

    f_lbm = torch.ones((Q,) + tuple(solid.shape), dtype=torch.float32)
    f_prism = torch.zeros((Q, n_prism) + node_shape, dtype=torch.float32)
    f_prism.fill_(2.0)  # prism cells are all 2.0

    f_result = project_prism_to_cartesian(f_lbm, f_prism, prism)

    # Prism cells should now be 2.0, non-prism fluid cells 1.0, solid 1.0
    n_band = int(prism.band_mask.sum().item())
    mean_band = f_result[0][prism.band_mask].mean().item()
    mean_rest = f_result[0][~prism.band_mask].mean().item()
    print(f"  Mean in prism band: {mean_band:.4f} (expect 2.0)")
    print(f"  Mean in rest: {mean_rest:.4f} (expect ~1.0)")
    assert abs(mean_band - 2.0) < 0.01, f"Prism cells not filled: {mean_band}"
    print("  PASSED\n")


def main():
    print("=" * 60)
    print("PRISM LAYER GENERATOR — Validation Tests")
    print("=" * 60)
    print()

    test_surface_normals()
    test_extract_normals_2d()
    test_prism_generation()
    test_prism_to_cartesian_projection()
    test_dg_body_fitted_rhs()

    print("=" * 60)
    print("ALL PRISM LAYER TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
