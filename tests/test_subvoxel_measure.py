"""P248-N02: sub-voxel surface measure contracts.

The kernel (``plane_cube_measure``) is checked against an *independent*
reference: the plane/cube cut volume is computed by Gauss-Legendre
quadrature of ``clip(u(x, y) + 1/2, 0, 1)`` with the quadrature segments
split at every kink, which is exact to machine precision for a piecewise
polynomial integrand.  The cut area is then obtained as ``dV/ds``.  Nothing
in the reference shares code with the polygon-clipping kernel.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.subvoxel_measure import (
    band_measure_nodal,
    mask_to_levelset,
    plane_box_measure,
    plane_cube_measure,
    staircase_surface_area,
    subvoxel_measure,
)

GL_X, GL_W = np.polynomial.legendre.leggauss(24)


def _segments(breaks):
    b = np.unique(np.clip(np.sort(np.asarray(breaks, float)), -0.5, 0.5))
    b = np.concatenate([[-0.5], b[(b > -0.5) & (b < 0.5)], [0.5]])
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def volume_reference(normal, offset):
    """Volume of ``{n.x <= offset}`` in the unit cube, machine precision."""
    n = np.asarray(normal, float)
    k = int(np.argmax(np.abs(n)))
    if n[k] < 0.0:
        return 1.0 - volume_reference(-n, -offset)
    idx = [i for i in range(3) if i != k]
    nz = n[k]
    a = offset / nz + 0.5
    b1, b2 = -n[idx[0]] / nz, -n[idx[1]] / nz
    total = 0.0
    ab = abs(b2)
    xbreaks = [-0.5, 0.5]
    if abs(b1) > 1e-14:
        for wv in (-ab / 2.0, ab / 2.0, 1.0 - ab / 2.0, 1.0 + ab / 2.0):
            xbreaks.append((wv - a) / b1)
    for lo, hi in _segments(xbreaks):
        for x, wx in zip(0.5 * (hi - lo) * GL_X + 0.5 * (hi + lo), GL_W):
            w = a + b1 * x
            if abs(b2) < 1e-15:
                inner = float(np.clip(w, 0.0, 1.0))
            else:
                inner = 0.0
                for ylo, yhi in _segments([(-w) / b2, (1.0 - w) / b2]):
                    for y, wy in zip(0.5 * (yhi - ylo) * GL_X + 0.5 * (yhi + ylo), GL_W):
                        inner += 0.5 * (yhi - ylo) * wy * float(np.clip(w + b2 * y, 0.0, 1.0))
            total += 0.5 * (hi - lo) * wx * inner
    return total


def area_reference(normal, offset, step=1e-5):
    return (volume_reference(normal, offset + step)
            - volume_reference(normal, offset - step)) / (2.0 * step)


def _plane_cases():
    rng = np.random.default_rng(7)
    cases = []
    for _ in range(12):
        v = rng.normal(size=3)
        cases.append((v / np.linalg.norm(v), float(rng.uniform(-0.55, 0.55))))
    # axis aligned, oblique and edge-on planes, kept off the level-set
    # breakpoints where a finite-difference reference is meaningless
    for v, s in [((0, 0, 1.0), 0.31), ((1, 0, 0), -0.22), ((0, 1, 0), 0.11),
                 ((1, 1, 1), 0.05), ((0, 1, 1), 0.24), ((2, 1, 0.5), -0.13),
                 ((0.1, 0.2, 0.97), 0.31), ((1, 1, 0), 0.2)]:
        w = np.array(v, float)
        cases.append((w / np.linalg.norm(w), float(s)))
    return cases


@pytest.mark.unit
def test_plane_cube_volume_matches_independent_quadrature():
    cases = _plane_cases()
    normals = np.array([c[0] for c in cases])
    offsets = np.array([c[1] for c in cases])
    fraction, _ = plane_cube_measure(normals, offsets)
    for i, (n, s) in enumerate(cases):
        assert fraction[i] == pytest.approx(volume_reference(n, s), abs=2e-9)


@pytest.mark.unit
def test_plane_cube_cut_area_matches_dv_ds():
    for n, s in _plane_cases():
        _, area = plane_cube_measure(np.array([n]), np.array([s]))
        assert area[0] == pytest.approx(area_reference(n, s), rel=1e-6, abs=1e-8)


@pytest.mark.unit
def test_plane_cube_degenerate_orientations_are_exact():
    # face-on (area = 1) and body-diagonal hexagon (area = 3*sqrt(3)/4)
    frac, area = plane_cube_measure(np.array([[0.0, 0.0, 1.0]]), np.array([0.25]))
    assert frac[0] == pytest.approx(0.75)
    assert area[0] == pytest.approx(1.0)
    diag = np.array([[1.0, 1.0, 1.0]]) / np.sqrt(3.0)
    frac, area = plane_cube_measure(diag, np.array([0.0]))
    assert frac[0] == pytest.approx(0.5)
    assert area[0] == pytest.approx(3.0 * np.sqrt(3.0) / 4.0, rel=1e-12)


@pytest.mark.unit
def test_plane_box_measure_axis_aligned_cut():
    # a plane at 0.3 h above the cell centre of an anisotropic box
    normal = np.array([[0.0, 0.0, 1.0]])
    frac, area, inside = plane_box_measure(normal, np.array([0.3]), np.array([2.0, 1.0, 4.0]))
    assert inside[0]
    assert frac[0] == pytest.approx(0.575)        # (h_z/2 + 0.3) / h_z
    assert area[0] == pytest.approx(2.0 * 1.0)    # full face


@pytest.mark.unit
def test_anisotropic_box_area_scaling():
    # a vertical plane in a box h_z = 5 h_xy must cut h_xy * h_z, independent of h_z
    normal = np.array([[1.0, 0.0, 0.0]])
    for hz in (1.0, 2.0, 5.0):
        _, area, inside = plane_box_measure(normal, np.array([0.0]),
                                            np.array([1.0, 1.0, hz]))
        assert inside[0]
        assert area[0] == pytest.approx(hz)


def _sphere_field(radius, spacing=1.0, centre=(0.31, 0.07, 0.19), pad=0.2):
    lo = -radius - pad * radius
    hi = radius + pad * radius
    xs = np.arange(int((hi - lo) / spacing) + 1) * spacing + lo + centre[0] * spacing
    ys = np.arange(int((hi - lo) / spacing) + 1) * spacing + lo + centre[1] * spacing
    zs = np.arange(int((hi - lo) / spacing) + 1) * spacing + lo + centre[2] * spacing
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    return np.sqrt(X**2 + Y**2 + Z**2) - radius


@pytest.mark.unit
def test_nodal_measure_is_second_order_on_a_sphere():
    errs = []
    hs = []
    volume_errors = []
    for radius in (8.0, 16.0, 32.0):
        phi = _sphere_field(radius)
        area, volume, n_cut, _ = band_measure_nodal(phi, 1.0)
        reference = 4.0 * np.pi * radius**2
        errs.append(abs(area - reference) / reference)
        hs.append(1.0 / radius)
        assert n_cut > 0
        vol_err = abs(volume - 4.0 / 3.0 * np.pi * radius**3) / (4.0 / 3.0 * np.pi * radius**3)
        assert vol_err < 1e-1
        volume_errors.append(vol_err)
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert slope >= 1.8, f"observed order {slope:.2f} < 1.8"
    assert errs[-1] <= 5e-3
    vol_slope = np.polyfit(np.log(hs), np.log(volume_errors), 1)[0]
    assert vol_slope >= 1.8, f"volume order {vol_slope:.2f} < 1.8"
    assert volume_errors[-1] < 8e-3


@pytest.mark.unit
def test_nodal_measure_is_invariant_to_z_anisotropy_for_a_ruled_wall():
    # a vertical wall is exactly representable, so h_z must not matter
    areas = []
    for hz in (1.0, 2.0, 4.0):
        xs = np.arange(41) - 20.0 + 0.13
        zs = np.arange(int(40.0 / hz) + 1) * hz - 20.0
        Z, Y, X = np.meshgrid(zs, xs, xs, indexing="ij")
        phi = np.sqrt(X**2 + Y**2) - 12.0
        area, *_ = band_measure_nodal(phi, (1.0, 1.0, hz))
        reference = 2.0 * np.pi * 12.0 * 40.0
        areas.append(abs(area - reference) / reference)
    assert max(areas) < 1e-2
    assert max(areas) - min(areas) < 1e-6


@pytest.mark.unit
def test_gradient_band_measure_matches_nodal_on_a_distance_function():
    radius = 16.0
    phi = _sphere_field(radius)
    reference = 4.0 * np.pi * radius**2
    area_grad = subvoxel_measure(phi, 1.0).area  # finite-difference gradient path
    area_nodal = band_measure_nodal(phi, 1.0)[0]
    assert abs(area_grad - reference) / reference < 5e-3
    assert abs(area_nodal - reference) / reference < 5e-3
    assert abs(area_grad - area_nodal) / reference < 5e-3


@pytest.mark.unit
def test_subvoxel_measure_beats_the_staircase_negative_control():
    radius = 12.0
    phi = _sphere_field(radius)
    mask = phi < 0.0
    reference = 4.0 * np.pi * radius**2
    result = subvoxel_measure(phi, 1.0)
    stair = staircase_surface_area(mask, 1.0)
    assert abs(result.area - reference) / reference < 5e-3
    assert abs(stair - reference) / reference > 0.4          # legacy fails badly
    assert result.volume == pytest.approx(4.0 / 3.0 * np.pi * radius**3, rel=3e-2)


@pytest.mark.unit
def test_measure_is_deterministic():
    phi = _sphere_field(10.0)
    first = subvoxel_measure(phi, 1.0)
    second = subvoxel_measure(phi.copy(), 1.0)
    assert first.area == second.area
    assert first.volume == second.volume
    assert first.n_cut_cells == second.n_cut_cells


@pytest.mark.unit
def test_mask_route_recovers_sub_voxel_area_from_occupancy_only():
    # the mask route has thrown the sub-voxel information away, so it can only
    # be first-order accurate; it must still beat the face count substantially
    radius = 16.0
    phi = _sphere_field(radius)
    mask = phi < 0.0
    reference = 4.0 * np.pi * radius**2
    level = mask_to_levelset(mask, 1.0)
    result = subvoxel_measure(level, 1.0)
    stair = staircase_surface_area(mask, 1.0)
    mask_err = abs(result.area - reference) / reference
    stair_err = abs(stair - reference) / reference
    assert mask_err < stair_err / 2.0
    assert mask_err < 0.05


@pytest.mark.unit
def test_empty_and_full_domains_are_handled():
    phi = np.ones((4, 4, 4))
    area, volume, n_cut, n_solid = band_measure_nodal(phi, 1.0)
    assert area == 0.0 and volume == 0.0 and n_cut == 0 and n_solid == 0
    phi = -np.ones((4, 4, 4))
    area, volume, n_cut, n_solid = band_measure_nodal(phi, 1.0)
    assert area == 0.0 and n_cut == 0
    assert volume == pytest.approx(3.0**3)
    assert n_solid == 27