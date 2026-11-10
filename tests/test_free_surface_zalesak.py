"""Tests for the Zalesak slotted-cylinder rotation benchmark (P189-N06).

The tests cover, in increasing cost:

1. geometry (analytic vs sub-cell-integrated volume, centroid, mirror symmetry),
2. the analytic velocity field (discrete divergence, lattice-speed calibration),
3. the transport operator's structural invariants (exact mass conservation,
   boundedness ``0 <= phi <= 1``, monotonicity, uniform-advection exactness),
4. the interface metrics themselves (thickness of a known profile, overlap,
   Hausdorff distance, convergence order, GCI),
5. a short full-revolution smoke test in 2-D and a 3-D ``z``-invariance check.

Heavy certification runs live in
``benchmarks/zalesak_rotation/run_zalesak_campaign.py``; these tests stay cheap
(seconds) so they can gate every commit.
"""

from __future__ import annotations

import math

import torch

from tensorlbm.free_surface_zalesak import (
    advect_face_flux,
    analytic_volume_2d,
    convergence_order,
    cross_section,
    discrete_divergence_max,
    gci,
    hausdorff_distance,
    interface_thickness,
    l1_shape_error,
    liquid_volume,
    rigid_rotation_field,
    run_zalesak_case,
    shape_overlap,
    slotted_cylinder_extruded_phi_3d,
    slotted_cylinder_phi_2d,
    slotted_sphere_phi_3d,
    zalesak_step,
)

torch.set_num_threads(1)


# --------------------------------------------------------------------------- #
# 1. Geometry                                                                 #
# --------------------------------------------------------------------------- #


def test_initial_volume_matches_analytic():
    """Sub-cell integrated area matches the brute-force reference to 1 %."""
    v_ref = analytic_volume_2d()
    full_disk = math.pi * 0.15**2
    assert 0.0 < v_ref < full_disk  # the slot removes liquid
    for n in (64, 128):
        phi = slotted_cylinder_phi_2d(n)
        rel = (liquid_volume(phi) - v_ref) / v_ref
        assert abs(rel) < 1.0e-2, (n, rel)


def test_initial_shape_geometry():
    """Centroid on the symmetry axis, bounds respected, mirror symmetry."""
    n = 128
    phi = slotted_cylinder_phi_2d(n)
    assert float(phi.min()) >= 0.0 and float(phi.max()) <= 1.0
    x = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    cx = float((phi * xx).sum() / phi.sum())
    cy = float((phi * yy).sum() / phi.sum())
    assert abs(cx - 0.5) < 1e-12
    # disk spans y in [0.6, 0.9]; the slot removes mass from the upper half
    assert 0.72 < cy < 0.77
    # mirror symmetry about x = 0.5
    assert float((phi - torch.flip(phi, dims=[1])).abs().max()) < 1e-12


def test_slot_depth_matters():
    """Deeper slots remove more liquid (guards against a mis-specified slab)."""
    shallow = liquid_volume(slotted_cylinder_phi_2d(128, slot_depth=0.10))
    deep = liquid_volume(slotted_cylinder_phi_2d(128, slot_depth=0.25))
    assert deep < shallow < math.pi * 0.15**2


# --------------------------------------------------------------------------- #
# 2. Velocity field                                                           #
# --------------------------------------------------------------------------- #


def test_divergence_free_and_lattice_speed():
    """The rotation field is discretely divergence free and correctly scaled."""
    for n, dim in ((32, 2), (32, 3)):
        u = rigid_rotation_field(n, dim=dim, omega=2.0 * math.pi * n / 320)
        assert discrete_divergence_max(u) < 1.0e-12
        peak = max(float(c.abs().max()) for c in u)
        # peak lattice speed = omega * r_max, r_max <= 0.5 for the unit box
        assert 0.0 < peak <= 2.0 * math.pi * n / 320 * 0.5 + 1e-12
        if dim == 3:
            assert float(u[1].abs().max()) == 0.0  # rotation in the x-z plane
            assert float(u[2].abs().max()) > 0.0


def test_rotation_direction():
    """Counter-clockwise rotation: material above the centre moves in -x."""
    n = 32
    ux, uy = rigid_rotation_field(n, dim=2, omega=0.1)
    assert float(ux[-1, 0]) < 0.0   # top row -> ux < 0
    assert float(uy[0, -1]) > 0.0   # right column -> uy > 0


# --------------------------------------------------------------------------- #
# 3. Transport operator invariants                                            #
# --------------------------------------------------------------------------- #


def _transport(phi, u, *, steps, **kw):
    p = phi
    with torch.no_grad():
        for _ in range(steps):
            p = zalesak_step(p, u, **kw)
    return p


def test_mass_conservation_is_exact():
    """Flux form telescopes: total volume is conserved to round-off."""
    n = 48
    phi = slotted_cylinder_phi_2d(n)
    u = rigid_rotation_field(n, dim=2, omega=2.0 * math.pi * n / 480)
    v0 = liquid_volume(phi)
    for kw in ({"scheme": "upwind"}, {"scheme": "muscl"}, {"scheme": "muscl", "c_comp": 1.0}):
        p = _transport(phi, u, steps=60, **kw)
        assert abs(liquid_volume(p) - v0) / v0 < 1e-12, kw


def test_boundedness_under_strong_compression():
    """The FCT capacity limiter keeps phi inside [0, 1] even for large c_comp."""
    n = 48
    phi = slotted_cylinder_phi_2d(n)
    u = rigid_rotation_field(n, dim=2, omega=2.0 * math.pi * n / 480)
    for c in (0.5, 2.0, 8.0):
        p = _transport(phi, u, steps=80, scheme="muscl", c_comp=c)
        assert float(p.min()) > -1e-12, c
        assert float(p.max()) < 1.0 + 1e-12, c


def test_compression_sharpens_without_adding_mass():
    """Same velocity, 200 steps: compression reduces thickness, keeps volume."""
    n = 64
    phi = slotted_cylinder_phi_2d(n)
    u = rigid_rotation_field(n, dim=2, omega=2.0 * math.pi * n / 640)
    plain = _transport(phi, u, steps=200, scheme="muscl", c_comp=0.0)
    sharp = _transport(phi, u, steps=200, scheme="muscl", c_comp=1.0)
    t0 = interface_thickness(phi)["mean"]
    t_plain = interface_thickness(plain)["mean"]
    t_sharp = interface_thickness(sharp)["mean"]
    assert t_sharp < t_plain, (t_sharp, t_plain)
    assert t_sharp <= t0 + 0.25, (t_sharp, t0)
    v0 = liquid_volume(phi)
    assert abs(liquid_volume(sharp) - v0) / v0 < 1e-12


def test_uniform_advection_translates_without_loss():
    """Constant velocity: the shape is carried exactly, no spurious mass."""
    n = 64
    x = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    phi = ((xx > 0.3) & (xx < 0.6)).double()
    u = [torch.full_like(phi, 0.25), torch.zeros_like(phi)]
    p = phi
    with torch.no_grad():
        for _ in range(4):  # 4 * 0.25 = exactly one cell
            p = advect_face_flux(p, u, scheme="muscl")
    assert abs(float(p.sum()) - float(phi.sum())) < 1e-12
    # equivalence to a one-cell shift of the same profile: centroid moves 1 cell
    def centroid(f):
        return float((f * xx).sum() / f.sum())
    assert abs((centroid(p) - centroid(phi)) - 1.0 / n) < 0.1 / n
    assert float(p.min()) > -1e-12 and float(p.max()) < 1.0 + 1e-12


# --------------------------------------------------------------------------- #
# 4. Metrics                                                                  #
# --------------------------------------------------------------------------- #


def test_interface_thickness_of_known_profiles():
    """A one-cell jump has thickness ~1 cell; a wide ramp is much thicker."""
    n = 64
    x = torch.arange(n, dtype=torch.float64)
    sharp = (x >= n // 2).double().expand(n, n).contiguous()
    ramp = (x / (n - 1)).clamp(0, 1).expand(n, n).contiguous()
    t_sharp = interface_thickness(sharp)["mean"]
    t_ramp = interface_thickness(ramp)["mean"]
    assert 1.2 < t_sharp < 2.0, t_sharp  # a one-cell jump reads as ~1.6 cells
    assert t_ramp > 5.0 * t_sharp, (t_ramp, t_sharp)


def test_overlap_definitions_and_hausdorff():
    n = 64
    phi = slotted_cylinder_phi_2d(n)
    assert shape_overlap(phi, phi) == 1.0
    assert shape_overlap(phi, phi, mode="recall") == 1.0
    shifted = torch.roll(phi, 2, dims=1)
    iou = shape_overlap(phi, shifted)
    recall = shape_overlap(phi, shifted, mode="recall")
    assert 0.0 < iou < recall < 1.0
    assert hausdorff_distance(phi, phi) == 0.0
    assert hausdorff_distance(phi, shifted) >= 1.0
    assert l1_shape_error(phi, phi) == 0.0


def test_convergence_helpers():
    assert abs(convergence_order(1.0, 0.5) - 1.0) < 1e-12
    assert abs(convergence_order(1.0, 0.25) - 2.0) < 1e-12
    assert math.isnan(convergence_order(0.0, 0.1))
    g = gci(1.0, 0.25, 2.0)
    assert abs(g - 100.0 * 1.25 * 0.25 / 3.0) < 1e-12
    assert math.isnan(gci(1.0, 0.5, float("nan")))


# --------------------------------------------------------------------------- #
# 5. End-to-end smoke tests                                                   #
# --------------------------------------------------------------------------- #


def test_full_revolution_smoke_2d():
    """One revolution on a coarse grid: bounded, conservative, shape survives."""
    res = run_zalesak_case(40, u_max_lat=0.2, c_comp=1.0, scheme="muscl",
                           limiter="superbee", sample_every=10_000)
    m = res["metrics"]
    assert m["lattice"] == "D2Q9"
    assert abs(m["u_max_lattice"] - 0.2) < 0.01
    assert abs(m["volume_rel_change"]) < 1e-10
    assert m["div_max"] < 1e-12
    assert m["boundedness_violations"] == 0
    assert m["overlap_recall"] > 0.70
    assert m["thickness_mean"] < 3.0
    assert res["phi_final"].shape == (40, 40)


def test_three_d_slab_slot_mid_plane_matches_2d():
    """The slab-slot 3-D set-up: its y = cy plane is the 2-D Zalesak problem.

    Checks that (i) the mid-plane reproduces the 2-D initial condition, (ii) the
    body is genuinely three-dimensional, (iii) a D3Q19 run stays close to the
    D2Q9 run started from the same plane over a *few* steps.  The deviation grows
    with time here because the sharpening operator is geometry dependent (the
    3-D interface normal leaves the comparison plane); the exact 2-D/3-D
    equivalence is asserted in ``test_extruded_body_matches_2d_exactly``.
    """
    n = 24
    phi3 = slotted_sphere_phi_3d(n)
    phi2 = slotted_cylinder_phi_2d(n)
    sec = cross_section(phi3)
    assert abs(liquid_volume(sec) - liquid_volume(phi2)) / liquid_volume(phi2) < 0.05
    # coarse (n = 24): the cell-level mask overlaps to ~0.7, the area to < 5 %
    assert shape_overlap(sec, phi2) > 0.70
    # genuine 3-D content: the y slices have very different areas
    areas = [float(phi3[:, k].mean()) for k in range(n)]
    assert max(areas) / max(min(areas), 1e-30) > 1e3
    steps = 10
    u3 = rigid_rotation_field(n, dim=3, omega=2.0 * math.pi * n / 240)
    u2 = rigid_rotation_field(n, dim=2, omega=2.0 * math.pi * n / 240)
    p3, p2 = phi3, sec  # same initial field on the compared plane
    with torch.no_grad():
        for _ in range(steps):
            p3 = zalesak_step(p3, u3, scheme="muscl", c_comp=1.0)
            p2 = zalesak_step(p2, u2, scheme="muscl", c_comp=1.0)
    sec3 = cross_section(p3)
    rel = float((sec3 - p2).abs().mean() / p2.mean())
    assert rel < 0.20, rel
    assert abs(liquid_volume(p3) - liquid_volume(phi3)) / liquid_volume(phi3) < 1e-12


def test_extruded_body_matches_2d_exactly():
    """The extruded 3-D body must reproduce the D2Q9 run to round-off."""
    n = 24
    phi3 = slotted_cylinder_extruded_phi_3d(n)
    sec0 = cross_section(phi3)
    steps = 120
    u3 = rigid_rotation_field(n, dim=3, omega=2.0 * math.pi * n / steps)
    u2 = rigid_rotation_field(n, dim=2, omega=2.0 * math.pi * n / steps)
    p3, p2 = phi3, sec0
    with torch.no_grad():
        for _ in range(steps):
            p3 = zalesak_step(p3, u3, scheme="muscl", c_comp=1.0)
            p2 = zalesak_step(p2, u2, scheme="muscl", c_comp=1.0)
    assert float((cross_section(p3) - p2).abs().max()) < 1e-12
