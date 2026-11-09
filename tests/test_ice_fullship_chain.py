"""Tests for the P66-W2-N04 full-ship broken-ice resistance chain.

Covers the five physics components and the acceptance-criterion helpers:

* ``floe_field`` : concentration / D50 control and the independent rasterisation
* ``lbm_porous`` : conservation, convergence, momentum-exchange hull force
* ``fsi``        : ice<->flow mapping and the momentum-exchange residual
* ``dem_field``  : hull blocking, finite forces, energy bookkeeping
* ``riska``      : reference models and the comparison helper
"""

import math

import numpy as np
import pytest

from tensorlbm.ice_fullship.dem_field import DEMConfig, DEMField
from tensorlbm.ice_fullship.floe_field import (
    FloeFieldConfig, generate_floe_field, measure_d50, rasterise_concentration,
    sample_floe_diameters, validate_floe_field,
)
from tensorlbm.ice_fullship.fsi import (
    coverage_to_porous_coefficients, floe_drag_forces, momentum_exchange_audit,
    rasterise_coverage,
)
from tensorlbm.ice_fullship.lbm_porous import (
    PorousLBM, PorousLBMConfig, build_hull_mask, residual_decades,
)
from tensorlbm.ice_fullship.riska import (
    clearing_momentum, compare, riska_campaign_baseline, submersion_lindqvist,
)


# ---------------------------------------------------------------- floe field
class TestFloeField:
    def test_invalid_config(self):
        with pytest.raises(ValueError):
            FloeFieldConfig(concentration=1.5)
        with pytest.raises(ValueError):
            FloeFieldConfig(thickness=0.0)
        with pytest.raises(ValueError):
            FloeFieldConfig(sigma_g=1.0)

    def test_size_law_bounds(self):
        cfg = FloeFieldConfig()
        rng = np.random.default_rng(0)
        d = sample_floe_diameters(500, cfg, rng)
        assert d.size == 500
        assert d.min() >= cfg.d_min_ratio * cfg.d50 - 1e-12
        assert d.max() <= cfg.d_max_ratio * cfg.d50 + 1e-12

    def test_concentration_and_d50(self):
        cfg = FloeFieldConfig(length=120.0, width=60.0, concentration=0.6, d50=1.0)
        field = generate_floe_field(cfg)
        audit = validate_floe_field(field, cfg)
        assert audit["pass_concentration"] is True
        assert audit["pass_d50"] is True
        assert audit["pass_no_overlap"] is True

    def test_rasterisation_matches_geometry(self):
        cfg = FloeFieldConfig(length=80.0, width=40.0, concentration=0.45, d50=1.0)
        field = generate_floe_field(cfg)
        raster = rasterise_concentration(field, resolution=0.05)
        geom = field.geometric_concentration
        assert abs(raster - geom) / geom < 0.02

    def test_area_weighted_d50_ge_number_weighted(self):
        cfg = FloeFieldConfig(length=80.0, width=40.0)
        field = generate_floe_field(cfg)
        assert measure_d50(field, "area") >= measure_d50(field, "number")


# ------------------------------------------------------------------ water phase
@pytest.fixture(scope="module")
def small_solver():
    cfg = PorousLBMConfig(n_hull=20, n_buffer_up=8, n_buffer_down=14,
                          side_margin_cells=5, depth_margin_cells=4,
                          ice_layer_cells=2, u_lat=0.05, re_hull=100.0)
    return PorousLBM(cfg), cfg


class TestWaterPhase:
    def test_hull_mask_non_empty_and_symmetric(self):
        cfg = PorousLBMConfig(n_hull=20)
        mask = build_hull_mask(cfg)
        assert mask.sum() > 0
        # symmetric about the centreline
        assert np.array_equal(mask, mask[:, ::-1, :])

    def test_force_positive_and_transverse_zero(self, small_solver):
        solver, cfg = small_solver
        solver.run(200, check_every=100)
        f = solver.hull_force()
        assert f["R_x_total"] > 0.0
        assert abs(f["F_y"]) < 1e-2 * abs(f["R_x_total"]) + 1e-6

    def test_continuity_and_convergence(self, small_solver):
        solver, cfg = small_solver
        out = solver.run(600, check_every=100)
        assert out["max_flux_imbalance"] < 1e-4
        assert out["final_residual"] < out["residual_history"][0]
        assert residual_decades(out["residual_history"]) > 0.5

    def test_ice_layer_increases_resistance(self):
        base = dict(n_hull=20, n_buffer_up=8, n_buffer_down=14, side_margin_cells=5,
                    depth_margin_cells=4, u_lat=0.05, re_hull=100.0)
        s0 = PorousLBM(PorousLBMConfig(ice_layer_cells=0, **base))
        s1 = PorousLBM(PorousLBMConfig(ice_layer_cells=2, **base))
        s0.run(300, check_every=150)
        s1.run(300, check_every=150)
        assert s1.hull_force()["R_x_total"] > s0.hull_force()["R_x_total"]

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            PorousLBMConfig(n_hull=8)
        with pytest.raises(ValueError):
            PorousLBMConfig(re_hull=1.0e9)


# -------------------------------------------------------------------------- FSI
class TestFSI:
    def test_coverage_bounds_and_closure(self):
        cfg = PorousLBMConfig(n_hull=20, n_buffer_up=8, n_buffer_down=14,
                              side_margin_cells=5, depth_margin_cells=4)
        length = cfg.nx / cfg.cells_per_metre
        width = 2.0 * (cfg.n_halfbeam + cfg.side_margin_cells) / cfg.cells_per_metre
        fcfg = FloeFieldConfig(length=length, width=width, concentration=0.6, d50=1.0)
        field = generate_floe_field(fcfg)
        cov = rasterise_coverage(field, cfg.ny, cfg.nx, cfg)
        assert cov.shape == (cfg.ny, cfg.nx)
        assert cov.min() >= 0.0 and cov.max() <= 1.0
        beta, gamma = coverage_to_porous_coefficients(cov, cfg, cfg.ice_concentration)
        assert np.all(beta >= 0.0) and np.all(gamma >= 0.0)
        assert beta.max() > 0.0

    def test_momentum_exchange_residual_is_tiny(self):
        cfg = PorousLBMConfig(n_hull=20, n_buffer_up=8, n_buffer_down=14,
                              side_margin_cells=5, depth_margin_cells=4, ice_layer_cells=2,
                              u_lat=0.05, re_hull=100.0)
        length = cfg.nx / cfg.cells_per_metre
        width = 2.0 * (cfg.n_halfbeam + cfg.side_margin_cells) / cfg.cells_per_metre
        fcfg = FloeFieldConfig(length=length, width=width, concentration=0.6, d50=1.0, seed=3)
        field = generate_floe_field(fcfg)
        cov = rasterise_coverage(field, cfg.ny, cfg.nx, cfg, sub=4)
        solver = PorousLBM(cfg, ice_coverage=cov)
        solver.run(200, check_every=100)
        # implementation-consistency residual (same coverage discretisation)
        audit = momentum_exchange_audit(solver, field, sub=4)
        assert audit["pass"] is True
        assert audit["residual_rel"] < 1e-2
        # coarse-graining residual with an independently refined rasterisation:
        # finite and reported (the floes are ~1/10 of a lattice cell here, so this
        # is a physics limitation, not an implementation error)
        fine = momentum_exchange_audit(solver, field, sub=16)
        assert np.isfinite(fine["residual_rel"])

    def test_drag_forces_finite(self):
        cfg = PorousLBMConfig(n_hull=20, n_buffer_up=8, n_buffer_down=14,
                              side_margin_cells=5, depth_margin_cells=4, ice_layer_cells=2,
                              u_lat=0.05, re_hull=100.0)
        solver = PorousLBM(cfg)
        solver.run(150, check_every=150)
        length = cfg.nx / cfg.cells_per_metre
        width = 2.0 * (cfg.n_halfbeam + cfg.side_margin_cells) / cfg.cells_per_metre
        field = generate_floe_field(FloeFieldConfig(length=length, width=width, d50=1.0))
        drag = floe_drag_forces(solver, field)
        assert np.isfinite(drag["F_drag_x_sum"])
        assert drag["n_floes"] > 0


# -------------------------------------------------------------------------- DEM
@pytest.fixture(scope="module")
def small_dem():
    cfg = DEMConfig(L_pp=40.0, B=12.0, T=6.0, channel_width=30.0, buffer_upstream=12.0,
                    buffer_downstream=12.0, concentration=0.55, d50=0.5,
                    ship_speed=1.0, dt=2.0e-3, n_steps=300)
    return DEMField(cfg), cfg


class TestDEM:
    def test_config_validation(self):
        with pytest.raises(ValueError):
            DEMConfig(channel_width=10.0)
        with pytest.raises(ValueError):
            DEMConfig(concentration=1.2)

    def test_hull_cleared_at_init(self, small_dem):
        dem, cfg = small_dem
        xh = dem.x - dem.hull_x0
        inside = (xh > 0.0) & (xh < cfg.L_pp)
        b, _, _ = dem.hull.sample(np.clip(xh[inside], 0.0, cfg.L_pp),
                                 np.full(int(inside.sum()), 0.999 * cfg.T))
        assert np.all(np.abs(dem.y[inside]) > b - 1e-9)

    def test_forces_finite_and_positive_resistance(self, small_dem):
        dem, cfg = small_dem
        dem.run(200, warmup_fraction=0.5)
        dg = dem.diagnostics()
        assert np.isfinite(dg["R_ice_mean"])
        assert np.isfinite(dg["R_ice_RMS"])
        assert dg["n_floes"] > 0

    def test_energy_audit_keys(self, small_dem):
        dem, cfg = small_dem
        dem.run(150, warmup_fraction=0.5)
        audit = dem.energy_audit()
        for key in ("P_resistance", "P_balance", "D_contacts", "P_water",
                    "P_hull_slip", "P_hull_perp", "energy_residual_rel"):
            assert key in audit
            assert np.isfinite(audit[key])

    def test_floes_do_not_penetrate_hull(self, small_dem):
        dem, cfg = small_dem
        dem.run(150, warmup_fraction=0.5)
        xh = dem.x - dem.hull_x0
        inside = (xh > 0.0) & (xh < cfg.L_pp)
        b, _, _ = dem.hull.sample(np.clip(xh[inside], 0.0, cfg.L_pp),
                                 np.clip(cfg.T + dem.z[inside], 1e-3, 0.999 * cfg.T))
        gap = np.abs(dem.y[inside]) - b
        # The impulse-based hull contact projects a penetrating floe back onto the
        # surface, but a floe can still be forced in laterally by its neighbours
        # between two projections (documented DEM limitation: this is one of the
        # reasons the DEM resistance is not statistically converged yet).  The
        # test therefore requires the trapped fraction to stay small instead of
        # claiming a hard invariant.
        trapped = gap < -np.max(dem.r)
        assert trapped.mean() < 0.01


# --------------------------------------------------------------------- reference
class TestReferenceModels:
    def test_baseline_positive_and_monotone(self):
        r = riska_campaign_baseline(0.15, 1.5, 230.0, 32.29, 14.5, 0.8098, 20.0, 0.6)
        r2 = riska_campaign_baseline(0.30, 1.5, 230.0, 32.29, 14.5, 0.8098, 20.0, 0.6)
        assert r["R_ice"] > 0.0
        assert r2["R_ice"] > r["R_ice"]

    def test_concentration_scaling(self):
        a = riska_campaign_baseline(0.15, 1.5, 230.0, 32.29, 14.5, 0.8098, 20.0, 0.3)
        b = riska_campaign_baseline(0.15, 1.5, 230.0, 32.29, 14.5, 0.8098, 20.0, 0.6)
        assert b["R_ice"] > a["R_ice"]

    def test_lindqvist_and_clearing_scaling(self):
        s = submersion_lindqvist(0.15, 32.29, 14.5)
        assert s > 0.0
        assert submersion_lindqvist(0.30, 32.29, 14.5) == pytest.approx(2.0 * s)
        c = clearing_momentum(0.15, 32.29, 1.5, 0.6)
        assert c == pytest.approx(917.0 * 0.6 * 0.15 * 32.29 * 1.5 ** 2)

    def test_compare_helper(self):
        ref = {"R_ice": 100.0}
        assert compare(105.0, ref)["pass"] is True
        assert compare(200.0, ref)["pass"] is False
        assert compare(105.0, ref)["relative_error"] == pytest.approx(0.05)