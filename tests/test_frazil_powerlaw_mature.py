"""Tests for the mature frazil power-law rheology face.

Two speed tiers: the geometry/closure/fit tests are pure numpy and run in
milliseconds; the three shear-cell tests use a deliberately tiny cell
(``12 x 12 x 16``) and a few hundred steps, and only check the *estimator
identities* (pure Couette is exact, the two stress estimators agree, the
self-consistent fixed point is applied), never a physical coefficient.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import tensorlbm.frazil_powerlaw_mature as face
from tensorlbm.ice_suspension_rheology import closures


# ---------------------------------------------------------------------------
# Microgeometry
# ---------------------------------------------------------------------------


def test_physical_suspension_reaches_dense_concentrations():
    """RSA in physical space must realise c = 0.30 -- the previous face jammed."""
    for phi in (0.05, 0.20, 0.30):
        susp = face.physical_suspension(phi, seed=1)
        assert susp.n_crystals > 0
        assert abs(susp.solid_volume_fraction - phi) < 0.01, (phi, susp.solid_volume_fraction)
    assert face.physical_suspension(0.30, seed=1).n_crystals > \
        face.physical_suspension(0.20, seed=1).n_crystals


def test_physical_suspension_is_deterministic_in_the_seed():
    a = face.physical_suspension(0.2, seed=5)
    b = face.physical_suspension(0.2, seed=5)
    c = face.physical_suspension(0.2, seed=6)
    assert len(a.clusters) == len(b.clusters)
    assert all(np.allclose(x, y) for x, y in zip(a.clusters, b.clusters))
    assert len(a.clusters) != len(c.clusters) or not np.allclose(a.clusters[0], c.clusters[0])


def test_physical_suspension_rejects_bad_input():
    with pytest.raises(face.MatureRheologyError):
        face.physical_suspension(-0.1)
    with pytest.raises(face.MatureRheologyError):
        face.physical_suspension(0.9)
    with pytest.raises(face.MatureRheologyError):
        face.physical_suspension(0.2, family="gas")
    with pytest.raises(face.MatureRheologyError):
        face.physical_suspension(0.2, radius_fraction=0.0)


def test_empty_suspension_is_allowed():
    susp = face.physical_suspension(0.0)
    assert susp.clusters == []
    assert susp.solid_volume_fraction == 0.0


def test_voxelise_preserves_fraction_across_resolutions():
    """The same physical object must give the same ``phi`` at both resolutions."""
    susp = face.physical_suspension(0.20, seed=2)
    phis = []
    for _name, n_xy, n_z in face.RESOLUTIONS:
        parts, info = face.voxelise(susp, n_xy=n_xy, n_z=n_z)
        assert info["n_particles"] == susp.n_crystals
        phis.append(info["phi_lattice"])
        # the lattice radius must scale with the resolution
        assert info["radius_cells"] == pytest.approx(
            face.RADIUS_FRACTION * n_xy, rel=1e-6)
    assert abs(phis[0] - phis[1]) < 0.02
    assert abs(phis[0] - 0.20) < 0.02


def test_voxelise_keeps_crystals_off_the_walls():
    susp = face.physical_suspension(0.30, seed=3)
    parts, info = face.voxelise(susp, n_xy=32, n_z=36)
    for p in parts:
        rows = np.flatnonzero(p.mask.any(axis=(1, 2)))
        assert rows.min() >= 2 and rows.max() <= 33
    assert info["wall_clearance_cells"] >= 1.5


def test_voxelise_rejects_under_resolved_crystal():
    susp = face.physical_suspension(0.20, seed=1)
    with pytest.raises(face.MatureRheologyError):
        face.voxelise(susp, n_xy=6, n_z=10)


def test_voxelise_requires_room_for_the_wall_band():
    susp = face.physical_suspension(0.20, seed=1)
    with pytest.raises(face.MatureRheologyError):
        face.voxelise(susp, n_xy=32, n_z=33)


def test_flocculated_structure_is_open():
    """Aggregates trap water: the hydrodynamic fraction must exceed the solids one."""
    susp = face.physical_suspension(0.10, family="flocculated", n_primary=8, seed=4)
    assert susp.n_clusters < susp.n_crystals
    phi_h = face.hydrodynamic_volume_fraction(susp.solid_volume_fraction,
                                              n_primary=8, df=susp.df)
    assert phi_h > 2.0 * susp.solid_volume_fraction


# ---------------------------------------------------------------------------
# Closures
# ---------------------------------------------------------------------------


def test_einstein_batchelor_green_matches_the_published_series():
    assert face.einstein_batchelor_green(0.0) == pytest.approx(1.0)
    assert face.einstein_batchelor_green(0.05) == pytest.approx(1 + 2.5 * 0.05 + 6.2 * 0.0025)
    assert face.einstein_batchelor_green(0.2) == pytest.approx(1 + 0.5 + 0.248)


def test_krieger_dougherty_diverges_at_maximum_packing():
    assert face.krieger_dougherty(0.0) == pytest.approx(1.0)
    assert face.krieger_dougherty(0.2, phi_max=0.6) == pytest.approx((1 - 1 / 3) ** -1.5)
    assert face.krieger_dougherty(0.599) > 10.0
    # the dilute slopes of Einstein and KD agree to O(phi^2)
    h = 1e-6
    slope_kd = (face.krieger_dougherty(h) - 1.0) / h
    assert slope_kd == pytest.approx(2.5, rel=1e-4)


def test_hydrodynamic_volume_fraction_limits():
    assert face.hydrodynamic_volume_fraction(0.1, n_primary=1) == pytest.approx(0.1)
    assert face.hydrodynamic_volume_fraction(0.1, n_primary=8, df=2.0) == \
        pytest.approx(0.1 * math.sqrt(8))
    assert face.hydrodynamic_volume_fraction(0.1, n_primary=8, df=3.0) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Fits and intervals
# ---------------------------------------------------------------------------


def test_fit_power_law_recovers_a_synthetic_curve():
    K, n = 0.31, 0.82
    g = np.array([2.8e-5, 5.6e-5, 1.1e-4])
    tau = K * g ** n
    fit = face.fit_power_law(g, tau)
    assert fit["n"] == pytest.approx(n, rel=1e-10)
    assert fit["K"] == pytest.approx(K, rel=1e-10)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)


def test_fit_power_law_rejects_degenerate_input():
    with pytest.raises(face.MatureRheologyError):
        face.fit_power_law([1.0], [1.0])
    with pytest.raises(face.MatureRheologyError):
        face.fit_power_law([0.0, 1.0], [1.0, 2.0])


def test_bootstrap_flags_shear_thinning():
    rng = np.random.default_rng(0)
    g = np.array([2.8e-5, 5.6e-5, 1.1e-4])
    samples = []
    for _ in range(4):
        tau = 0.3 * g ** 0.8 * (1 + 0.01 * rng.standard_normal(g.size))
        samples.append({"gamma_dot": g, "tau": tau})
    boot = face.bootstrap_power_law(samples, n_boot=800)
    assert boot["n"] < 0.95
    assert boot["n_ci"][1] < 1.0
    assert boot["n_ci_excludes_one"] is True


def test_bootstrap_does_not_flag_a_newtonian_ladder():
    rng = np.random.default_rng(1)
    g = np.array([2.8e-5, 5.6e-5, 1.1e-4])
    samples = []
    for _ in range(4):
        samples.append({"gamma_dot": g, "tau": 0.2 * g * (1 + 0.01 * rng.standard_normal(g.size))})
    boot = face.bootstrap_power_law(samples, n_boot=800)
    assert boot["n"] == pytest.approx(1.0, abs=0.02)
    # a Newtonian ladder must never satisfy the shear-thinning criterion, even if
    # the (very narrow) interval happens to exclude n = 1 on a noise realisation
    assert not (boot["n"] < 0.95 and boot["n_ci_excludes_one"])


def test_bootstrap_needs_data():
    with pytest.raises(face.MatureRheologyError):
        face.bootstrap_power_law([])


def test_grid_convergence_pair_flags_a_five_percent_gate():
    same = face.grid_convergence_pair({"K": 0.3, "n": 0.9}, {"K": 0.3, "n": 0.9})
    assert same["both_below_5pct"] is True
    bad = face.grid_convergence_pair({"K": 0.3, "n": 0.9}, {"K": 0.36, "n": 0.9})
    assert bad["K_rel_diff"] == pytest.approx(0.2)
    assert bad["both_below_5pct"] is False


# ---------------------------------------------------------------------------
# Shear cell: estimator identities (tiny cell, few hundred steps)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pure_small():
    return face.pure_reference(n_xy=12, n_z=16, nu=0.2, gamma_dot=2e-3,
                               steps=260, warmup=140, sample_every=10)


def test_pure_couette_is_reproduced_by_both_estimators(pure_small):
    assert pure_small["profile_r2"] > 0.999
    assert abs(pure_small["flux_rel_err"]) < 0.02
    assert abs(pure_small["wall_rel_err"]) < 0.02


def test_per_wall_estimator_separates_the_two_walls():
    """The in-tree accumulator cancels to zero; the per-wall helper must not."""
    from tensorlbm.ice_suspension_rheology.couette import ShearCell, ShearCellConfig
    cfg = ShearCellConfig(nx=12, ny=12, nz=16, nu=0.2, u_wall=1e-2)
    cell = ShearCell(cfg, [])
    for _ in range(40):
        out = cell.step(sample=False)
    bottom = face._per_wall_exchange(cell, "bottom")
    top = face._per_wall_exchange(cell, "top")
    assert abs(bottom) > 0
    assert np.sign(bottom) == -np.sign(top)
    assert abs(abs(bottom) - abs(top)) / abs(bottom) < 0.05


def test_self_consistent_point_reports_the_correction(pure_small):
    """A pure fluid must need no kinematic correction (scale stays 1)."""
    rec = face.self_consistent_point([], n_xy=12, n_z=16, nu=0.2, gamma_dot=2e-3,
                                     steps=260, warmup=140, sample_every=10, max_iter=2)
    assert rec["scale"] == pytest.approx(1.0, abs=0.02)
    assert rec["eta_r"] == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# Acceptance logic
# ---------------------------------------------------------------------------


def _fits(**over):
    base = {
        ("dispersed", 0.05): {"eta_r": 1.12, "phi_lattice": 0.050, "n": 1.0, "K": 0.2},
        ("dispersed", 0.20): {"eta_r": 1.70, "phi_lattice": 0.200, "n": 1.0, "K": 0.22},
        ("flocculated", 0.10): {"eta_r": 1.9, "phi_lattice": 0.10, "n": 0.85, "K": 0.3,
                                "n_ci": (0.83, 0.87), "n_ci_excludes_one": True},
    }
    base.update(over)
    return base


def test_acceptance_passes_on_consistent_numbers():
    pure = [{"tau_flux": 0.2 * 1e-3, "nu_gamma_dot": 0.2 * 1e-3, "n": 1.0}]
    acc = face.evaluate_acceptance(
        pure_cases=pure, ladder_fits=_fits(),
        convergence={"x": {"both_below_5pct": True, "n_rel_diff": 0.01, "K_rel_diff": 0.01}},
        pytest_result={"n_tests": 12, "n_passed": 12})
    assert acc["all_pass"] is True, acc


def test_acceptance_flags_each_failure_mode():
    pure = [{"tau_flux": 0.2 * 1e-3, "nu_gamma_dot": 0.2 * 1e-3, "n": 1.02}]
    fits = _fits()
    fits[("dispersed", 0.05)] = {"eta_r": 1.5, "phi_lattice": 0.05, "n": 1.0, "K": 0.2}
    fits[("flocculated", 0.10)] = {"eta_r": 1.9, "phi_lattice": 0.10, "n": 1.0, "K": 0.3,
                                   "n_ci": (0.95, 1.05), "n_ci_excludes_one": False}
    acc = face.evaluate_acceptance(
        pure_cases=pure, ladder_fits=fits,
        convergence={"x": {"both_below_5pct": False, "n_rel_diff": 0.1, "K_rel_diff": 0.2}},
        pytest_result={"n_tests": 3, "n_passed": 3})
    crit = acc["criteria"]
    assert crit["c1_zero_control"]["passed"] is False
    assert crit["c2_dilute_einstein"]["passed"] is False
    assert crit["c4_shear_thinning"]["passed"] is False
    assert crit["c5_grid_convergence"]["passed"] is False
    assert crit["c6_pytest"]["passed"] is False
    assert acc["all_pass"] is False


def test_acceptance_reports_missing_cases_instead_of_passing_them():
    acc = face.evaluate_acceptance(pure_cases=[], ladder_fits={}, convergence={},
                                   pytest_result=None)
    assert acc["n_pass"] == 0
    for v in acc["criteria"].values():
        assert v["passed"] is False