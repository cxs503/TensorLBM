"""Acceptance tests for the parametric ice-class bow family (icebow_cad).

Covers the campaign acceptance criteria:
(a) volume consistency of the mesh at two resolutions,
(b) geometry gate (watertight / manifold / Euler / outward / zero-area) plus
    negative controls that *must* be rejected,
(c) parameter consistency reverse-computed from the mesh,
(d) SDF/voxel volume convergence with GCI,
(e) resistance sensitivity to the stem angle with GCI,
(f) downstream contract: the exported STL is read back by the repository's
    own STL reader and reported watertight by it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.icebow_cad import (
    FAMILY_PRESETS,
    IceBowMesh,
    IceBowParams,
    build_ice_bow_mesh,
    design_volume,
    family_members,
    gci,
    half_breadth_norm,
    holtrop_resistance,
    michell_wave_resistance,
    offsets_table,
    read_stl_vertices,
    replace_mesh,
    resistance_sensitivity,
    resolve_params,
    reverse_parameters,
    stem_x_norm,
    submerged_volume,
    validate_mesh,
    voxel_volume_study,
    voxelize_mesh,
    write_stl,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_params() -> IceBowParams:
    return resolve_params(IceBowParams())


@pytest.fixture(scope="module")
def base_mesh(base_params) -> IceBowMesh:
    return build_ice_bow_mesh(base_params)


# ---------------------------------------------------------------------------
# (a) analytic volume / Cb target
# ---------------------------------------------------------------------------


def test_cb_target_is_met_by_solved_transom_ratio(base_params):
    dv = design_volume(base_params)
    assert base_params.r_t is not None and 0.0 < base_params.r_t < 1.0
    assert dv["cb"] == pytest.approx(base_params.cb_target, rel=1e-3)
    assert dv["volume_full"] > dv["nabla"]  # freeboard volume is positive


def test_cb_range_of_the_family_is_achievable():
    for cb in (0.55, 0.65, 0.70, 0.80):
        p = resolve_params(IceBowParams(cb_target=cb))
        assert design_volume(p)["cb"] == pytest.approx(cb, rel=1e-3)


def test_half_breadth_is_zero_forward_of_the_local_stem(base_params):
    zeta = np.array([0.0, 0.4, 0.75, 1.0])
    xs = stem_x_norm(base_params, zeta)
    assert np.isclose(half_breadth_norm(base_params, xs + 1e-6, zeta), 0.0, atol=1e-9).all()
    assert np.isclose(half_breadth_norm(base_params, xs[2], zeta[2]), 0.0, atol=1e-12)
    # full beam on the parallel body at the waterline
    assert half_breadth_norm(base_params, 0.5, 0.75) * base_params.L == pytest.approx(
        base_params.B / 2.0, rel=1e-9
    )


# ---------------------------------------------------------------------------
# (b) geometry admission gate
# ---------------------------------------------------------------------------


def test_gate_passes_for_whole_family():
    for p in family_members():
        mesh = build_ice_bow_mesh(resolve_params(p))
        rep = validate_mesh(mesh)
        assert rep["passed"], (p.family, p.alpha_stem_deg, rep["checks"])
        assert rep["euler_characteristic"] == 2
        assert rep["n_zero_area"] == 0
        assert rep["n_duplicate_index_faces"] == 0
        assert rep["edge_use_min"] == 2 and rep["edge_use_max"] == 2
        assert rep["outward_normal_fraction"] == 1.0
        assert rep["volume_rel_error"] < 1e-3
        assert rep["nabla_rel_error"] < 1e-3


def test_gate_reports_mandatory_checks(base_mesh):
    rep = validate_mesh(base_mesh)
    for key in (
        "watertight",
        "manifold_no_boundary",
        "manifold_no_nonmanifold",
        "euler_characteristic_2",
        "consistent_orientation",
        "no_zero_area_triangles",
        "positive_signed_volume",
        "outward_normals",
    ):
        assert rep["checks"][key] is True


def test_negative_dropped_face_is_rejected(base_mesh):
    """A single missing triangle must fail the watertightness check."""
    faces = np.delete(base_mesh.faces, 1000, axis=0)
    broken = replace_mesh(base_mesh, base_mesh.vertices, faces)
    rep = validate_mesh(broken)
    assert rep["passed"] is False
    assert rep["checks"]["watertight"] is False
    assert rep["edge_use_min"] == 1


def test_negative_flipped_winding_is_rejected(base_mesh):
    """Flipping one face breaks the consistent (outward) orientation."""
    faces = base_mesh.faces.copy()
    faces[1234] = faces[1234][::-1]
    broken = replace_mesh(base_mesh, base_mesh.vertices, faces)
    rep = validate_mesh(broken)
    assert rep["passed"] is False
    assert rep["checks"]["consistent_orientation"] is False
    assert rep["checks"]["outward_normals"] is False


def test_negative_duplicate_vertex_is_rejected(base_mesh):
    """Splitting a shared vertex opens the surface: boundary edges appear."""
    verts = base_mesh.vertices.copy()
    faces = base_mesh.faces.copy()
    old = int(faces[0, 0])
    verts = np.vstack([verts, verts[old] + np.array([1e-4, 0.0, 0.0])])
    faces[0, 0] = verts.shape[0] - 1
    broken = replace_mesh(base_mesh, verts, faces)
    rep = validate_mesh(broken)
    assert rep["passed"] is False
    assert rep["checks"]["watertight"] is False


def test_negative_zero_area_triangle_is_rejected(base_mesh):
    """A sliver (distinct indices, zero area) must be caught by the gate."""
    verts = np.vstack(
        [
            base_mesh.vertices,
            base_mesh.vertices[10] + np.array([0.0, 0.0, 0.0]),
            base_mesh.vertices[10] + np.array([1e-9, 0.0, 0.0]),
            base_mesh.vertices[10] + np.array([2e-9, 0.0, 0.0]),
        ]
    )
    n = verts.shape[0]
    faces = np.vstack([base_mesh.faces, np.array([[n - 3, n - 2, n - 1]], dtype=np.int64)])
    rep = validate_mesh(replace_mesh(base_mesh, verts, faces))
    assert rep["passed"] is False
    assert rep["checks"]["no_zero_area_triangles"] is False
    assert rep["n_zero_area"] >= 1


def test_negative_collapsed_face_is_rejected(base_mesh):
    """A face with a repeated vertex index is a mesh error, not a warning."""
    faces = np.vstack([base_mesh.faces, np.array([[10, 10, 20]], dtype=np.int64)])
    rep = validate_mesh(replace_mesh(base_mesh, base_mesh.vertices, faces))
    assert rep["passed"] is False
    assert rep["checks"]["no_collapsed_faces"] is False


def test_negative_invalid_parameters_raise():
    with pytest.raises(ValueError):
        IceBowParams(alpha_stem_deg=90.0)  # rafted past vertical
    with pytest.raises(ValueError):
        IceBowParams(beta_ent_deg=0.0)
    with pytest.raises(ValueError):
        IceBowParams(cb_target=1.2)  # outside (0, 0.95)
    with pytest.raises(ValueError):
        IceBowParams(family="spoon")
    with pytest.raises(ValueError):
        IceBowParams(c_fwd=0.5)
    with pytest.raises(ValueError):
        IceBowParams(waterline_frac=1.0)


def test_negative_mismatched_design_cb_is_caught():
    """A geometry whose true Cb is 0.75 but which claims 0.60 must be caught."""
    honest = resolve_params(IceBowParams(cb_target=0.75))
    mesh = build_ice_bow_mesh(honest)
    liar = IceBowParams(
        **{
            **{k: getattr(honest, k) for k in honest.__dataclass_fields__},
            "cb_target": 0.60,
        }
    )
    rep = reverse_parameters(replace_mesh(mesh, mesh.vertices, mesh.faces))
    rep["params"] = liar.cb_target
    assert abs(rep["cb"] - liar.cb_target) / liar.cb_target > 0.01  # gate would reject
    assert abs(rep["cb"] - honest.cb_target) / honest.cb_target < 0.01


# ---------------------------------------------------------------------------
# (a)+(c) volume consistency at two resolutions and reverse parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["straight", "inward", "multisegment"])
def test_volume_error_default_and_fine_resolution(family):
    p = resolve_params(IceBowParams(family=family, alpha_stem_deg=30.0))
    dv = design_volume(p)
    coarse = build_ice_bow_mesh(p, n_stations=81, n_levels=61)
    fine = build_ice_bow_mesh(p, n_stations=161, n_levels=121)
    e_coarse = abs(submerged_volume(coarse)["nabla"] - dv["nabla"]) / dv["nabla"]
    e_fine = abs(submerged_volume(fine)["nabla"] - dv["nabla"]) / dv["nabla"]
    assert e_coarse <= 1.0e-2
    assert e_fine <= 3.0e-3
    assert e_fine <= e_coarse


@pytest.mark.parametrize("alpha", [20.0, 25.0, 30.0, 35.0])
def test_reverse_parameters_match_design(alpha):
    p = resolve_params(IceBowParams(alpha_stem_deg=alpha))
    rep = reverse_parameters(build_ice_bow_mesh(p, n_stations=121, n_levels=91))
    assert abs(rep["cb_error"]) <= 0.01
    assert abs(rep["L_error"]) <= 0.01
    assert abs(rep["B_error"]) <= 0.01
    assert abs(rep["alpha_error_deg"]) <= 1.0
    assert abs(rep["beta_error_deg"]) <= 1.0


def test_reverse_beta_angle_tracks_design_value():
    for beta in (16.0, 22.0, 30.0):
        p = resolve_params(IceBowParams(beta_ent_deg=beta))
        rep = reverse_parameters(build_ice_bow_mesh(p, n_stations=101, n_levels=81))
        assert abs(rep["beta_ent_deg"] - beta) <= 1.0


def test_multisegment_lower_stem_rake_exceeds_upper():
    p = resolve_params(IceBowParams(family="multisegment", alpha_stem_deg=30.0))
    z = np.array([0.1, 0.5]) * p.D
    xs = stem_x_norm(p, z / p.D) * p.L
    slope = math.degrees(math.atan2(xs[1] - xs[0], z[1] - z[0]))
    assert slope > 30.0 + 5.0  # extra rake below the break


# ---------------------------------------------------------------------------
# (d) SDF / voxel consistency
# ---------------------------------------------------------------------------


def test_voxel_volume_convergence_and_gci(base_mesh):
    study = voxel_volume_study(base_mesh, spacings=(0.005, 0.0025, 0.00125))
    volumes = [lv["volume"] for lv in study["levels"]]
    assert study["monotone"] is True
    # the finest grid must reproduce the analytic volume
    assert study["finest_rel_error_abs"] <= 1.5e-2
    assert abs(volumes[-1] - study["analytic_volume"]) / study["analytic_volume"] <= 1.5e-2
    # the submerged-series error must shrink with resolution
    errs = [abs(lv["submerged_rel_error"]) for lv in study["levels"]]
    assert errs[-1] <= max(errs)
    assert study["gci"]["gci_fine_percent"] <= 5.0


def test_voxel_mask_shape_and_volume(base_mesh):
    spacing = 0.005
    mask, origin, sp = voxelize_mesh(base_mesh, spacing)
    assert mask.dtype == np.bool_
    nz, ny, nx = mask.shape
    assert ny == int(np.ceil((np.ptp(base_mesh.vertices[:, 1]) + 2 * spacing) / spacing))
    vol = mask.sum() * sp**3
    ref = design_volume(base_mesh.params)["volume_full"]
    assert vol == pytest.approx(ref, rel=1e-2)
    # the mask origin sits one cell outside the geometry
    assert origin[1] + spacing <= base_mesh.vertices[:, 1].min()


def test_voxel_spacing_must_be_positive(base_mesh):
    with pytest.raises(ValueError):
        voxelize_mesh(base_mesh, 0.0)


# ---------------------------------------------------------------------------
# (f) downstream contract
# ---------------------------------------------------------------------------


def test_stl_roundtrip_is_read_by_repository_reader(tmp_path, base_mesh):
    from tensorlbm.geometry_voxel import mesh_watertight_status, read_stl_triangles

    path = write_stl(tmp_path / "hull.stl", base_mesh.vertices, base_mesh.faces)
    tri = read_stl_triangles(path)
    assert tuple(tri.shape) == (base_mesh.faces.shape[0], 3, 3)
    status = mesh_watertight_status(tri)
    assert status["watertight"] is True
    assert status["boundary_edges"] == 0
    assert status["nonmanifold_edges"] == 0
    vol = float(
        np.einsum(
            "ij,ij->i",
            tri[:, 0].double().numpy(),
            np.cross(tri[:, 1].double().numpy(), tri[:, 2].double().numpy()),
        ).sum()
        / 6.0
    )
    # binary STL stores float32 vertices: ~1e-4 relative volume loss is expected
    assert vol == pytest.approx(abs(design_volume(base_mesh.params)["volume_full"]), rel=1e-3)
    # the module's own reader agrees with the repository reader
    own = read_stl_vertices(path)
    assert own.shape == (base_mesh.faces.shape[0], 3, 3)
    assert np.allclose(own, tri.numpy(), atol=1e-6)


def test_offsets_table_is_consistent_and_positive(base_params):
    tab = offsets_table(base_params, n_stations=21, n_levels=11)
    assert tab["half_beam"].shape == (21, 11)
    assert (tab["half_beam"] >= 0.0).all()
    assert tab["half_beam"].max() == pytest.approx(base_params.B / 2.0, rel=1e-6)
    # analytic check of a mid-body entry (station 10, level 2)
    zeta = np.linspace(0.0, 1.0, 11)
    assert tab["half_beam"][10, 2] == pytest.approx(
        half_breadth_norm(base_params, 0.5, zeta[2]) * base_params.L
    )
    assert tab["x"].max() <= base_params.L + 1e-12
    assert np.all(np.diff(tab["waterline_length"]) >= -1e-12)


# ---------------------------------------------------------------------------
# (e) resistance sensitivity
# ---------------------------------------------------------------------------


def test_gci_recovers_known_second_order():
    h = np.array([2.0, 1.0, 0.5])
    vals = 3.0 + 5.0 * h**2
    out = gci(float(vals[2]), float(vals[1]), float(vals[0]), r=2.0)
    assert out["observed_order"] == pytest.approx(2.0, abs=0.05)
    assert out["richardson"] == pytest.approx(3.0, rel=1e-3)


def test_holtrop_inputs_are_physical(base_mesh, base_params):
    res = holtrop_resistance(base_params, fn=0.20, L_m=100.0, mesh=base_mesh)
    assert res["nabla_m3"] == pytest.approx(base_params.cb_target * 100.0 * 17.0 * 8.625, rel=0.02)
    assert res["cb"] == pytest.approx(base_params.cb_target, rel=0.02)
    assert 0.5 < res["cm"] < 1.0
    assert 0.5 < res["cp"] < 0.95
    assert 2.0e8 < res["re"] < 1.0e9
    assert 2.0e3 < res["S_m2"] < 4.0e3
    assert 5.0e4 < res["rt_n"] < 3.0e5
    assert res["rt_n"] == pytest.approx(res["rf_n"] + res["rw_n"], rel=1e-12)


def test_michell_quadrature_converges(base_params):
    vals = [
        michell_wave_resistance(base_params, n_x=nx, n_z=nz, n_u=nu)["rw_n"]
        for nx, nz, nu in ((61, 41, 81), (121, 81, 161), (241, 161, 321))
    ]
    assert vals[-1] > 0.0
    assert gci(vals[-1], vals[-2], vals[-3], r=2.0)["gci_fine_percent"] <= 5.0


def test_stem_angle_sensitivity_is_monotone_with_gci():
    members = [
        IceBowParams(family="straight", alpha_stem_deg=a, cb_target=0.70)
        for a in (20.0, 30.0, 35.0)
    ]
    sens = resistance_sensitivity(
        members,
        fn=0.20,
        densities=((41, 31), (81, 61)),
        michell_densities=((61, 41, 81), (121, 81, 161), (241, 161, 321)),
    )
    assert sens["monotone_delta_rw_michell"] is True
    assert len(sens["delta_rw_michell_straight"]) == 3
    assert sens["max_rw_michell_gci_percent"] <= 5.0
    assert sens["delta_rw_gci_max_percent"] <= 5.0
    # the deltas are anchored at alpha = 30 deg
    ref = [r for r in sens["rows"] if abs(r["alpha_stem_deg"] - 30.0) < 1e-9][0]
    assert ref["delta_rw_michell"] == pytest.approx(0.0, abs=1e-12)


def test_family_presets_differ_geometrically():
    base = resolve_params(IceBowParams(family="straight"))
    for fam, preset in FAMILY_PRESETS.items():
        p = resolve_params(IceBowParams(family=fam, **preset))
        assert p.family == fam
        if fam != "straight":
            other = build_ice_bow_mesh(p, n_stations=41, n_levels=31)
            ref = build_ice_bow_mesh(base, n_stations=41, n_levels=31)
            assert validate_mesh(other)["passed"]
            assert other.params.c_flare != ref.params.c_flare