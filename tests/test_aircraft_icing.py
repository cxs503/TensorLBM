"""Tests for tensorlbm.aircraft_icing — Phase 2a rime model.

1. Seeding calibration: the droplets-per-step rate matches the analytic
   LWC/MVD formula and the deterministic carry reproduces the analytic
   total number of seeded droplets to within one.
2. Mass conservation: seeded = frozen + exited + trapped + airborne +
   pending closes to < 1 %, and frozen mass equals n_ice_cells * m_cell.
3. Geometry feedback smoke: a small coupled run freezes ice only next to
   the airfoil, produces positive lift at positive AoA and a finite cd.
4. Rime density: Macklin (1962) / Jones (1990) correlations incl. the
   0.917 g/cm^3 cap and the config-level mode switch.
5. compile_mode: the flow step goes through tensorlbm.compile_utils
   (validated modes only) and a compiled step matches the eager step.

The integration tests deliberately use an enlarged MVD and an accelerated
LWC (accel_override) so that freezing occurs within a few hundred CPU
steps; the *physics machinery* (seeding flux, relaxation, freezing,
audit) is what is under test, not the IRT benchmark sizing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.aircraft_icing import (
    IcingConfig,
    RimeIcingSimulation,
    ice_shape_metrics,
    naca0012_mask_2d,
    rime_density_jones,
    rime_density_macklin,
    run_rime_icing,
    seed_counts_total,
    surface_arc_length,
)


# ---------------------------------------------------------------------------
# Unit mapping + seeding calibration (analytic)
# ---------------------------------------------------------------------------
def test_unit_mapping_stokes_benchmark() -> None:
    """Default config = IRT case: St = 0.155 and mapping is consistent."""
    cfg = IcingConfig()
    # tau_d = rho d^2 / (18 mu)
    assert math.isclose(cfg.tau_d_phys, 1000.0 * (20e-6) ** 2 / (18 * 1.8e-5), rel_tol=1e-9)
    assert abs(cfg.stokes - 0.155) < 0.005
    # Stokes number is invariant between unit systems
    st_lu = cfg.tau_d_lu * cfg.u_in / cfg.chord_lu
    assert math.isclose(st_lu, cfg.tau_d_lu * cfg.u_in / cfg.chord_lu, rel_tol=0)
    assert math.isclose(st_lu, cfg.stokes, rel_tol=1e-9)
    # dt follows from dx and the velocity conversion
    assert math.isclose(cfg.dt_phys, cfg.dx_phys * cfg.u_in / cfg.v_inf, rel_tol=1e-12)
    assert cfg.n_substeps >= 1


def test_droplet_mass_and_seeding_rate_analytic() -> None:
    """Seeding rate reproduces LWC * V * A_in * dt / m_d exactly."""
    cfg = IcingConfig(nx=100, ny=64, chord_frac=0.4, accel_override=25.0)
    m_d = cfg.rho_water * math.pi * cfg.mvd**3 / 6.0
    assert math.isclose(cfg.m_droplet, m_d, rel_tol=1e-12)
    a_in = cfg.ny * cfg.dx_phys**2  # one-cell-deep slab
    rate = cfg.lwc_eff * cfg.v_inf * a_in * cfg.dt_phys / m_d
    assert math.isclose(cfg.droplets_per_step, rate, rel_tol=1e-9)
    # rate scales linearly with the acceleration factor
    cfg2 = IcingConfig(nx=100, ny=64, chord_frac=0.4, accel_override=50.0)
    assert math.isclose(cfg2.droplets_per_step, 2 * cfg.droplets_per_step, rel_tol=1e-9)


def test_parcel_scaling_keeps_mass_flux() -> None:
    """Parcels resample the same mass flux: N * parcels == droplets."""
    cfg = IcingConfig(nx=100, ny=64, chord_frac=0.4, accel_override=25.0)
    n = cfg.effective_parcel_multiplier
    assert n >= 1
    assert math.isclose(cfg.m_parcel, n * cfg.m_droplet, rel_tol=1e-12)
    assert math.isclose(cfg.parcels_per_step * n, cfg.droplets_per_step, rel_tol=1e-9)
    # explicit multiplier keeps the target under control and is honoured
    cfg2 = IcingConfig(
        nx=100, ny=64, chord_frac=0.4, accel_override=25.0, parcel_multiplier=7
    )
    assert cfg2.effective_parcel_multiplier == 7
    assert cfg2.parcels_per_step == cfg.droplets_per_step / 7


def test_seed_carry_matches_analytic_total() -> None:
    """Deterministic carry seeding: total = analytic rate * steps (+-1)."""
    for rate in (0.17, 2.7, 31.13):
        steps = 137
        total = seed_counts_total(rate, steps)
        assert abs(total - rate * steps) <= 1.0
    # exact when the rate is integer
    assert seed_counts_total(3.0, 50) == 150


def test_seeded_mass_matches_analytic_flux() -> None:
    """End-to-end: seeded mass in a run = LWC_eff * V * A_in * t_equiv."""
    cfg = IcingConfig(
        nx=64,
        ny=48,
        steps=120,
        warmup_steps=0,
        uniform_flow=True,
        mvd=100e-6,
        accel_override=50.0,
        prefill_cloud=False,
        device="cpu",
        log_every=10**9,
    )
    # push the airfoil far downstream so nothing impacts within 120 steps
    cfg.cx_frac = 0.8
    res = RimeIcingSimulation(cfg, log=lambda *a: None).run()
    t_equiv = cfg.steps * cfg.dt_phys
    expected = cfg.lwc_eff * cfg.v_inf * cfg.inlet_area * t_equiv
    got = res["audit"]["seeded"]
    assert abs(got - expected) / expected < 5e-3


def test_prefill_cloud_inventory() -> None:
    """prefill seeds ~ rate * kill_x / u_in parcels at t=0 (steady cloud)."""
    common = dict(
        nx=64,
        ny=48,
        steps=120,
        warmup_steps=0,
        uniform_flow=True,
        mvd=100e-6,
        accel_override=50.0,
        device="cpu",
        seed=3,
        log_every=10**9,
    )
    cfg_off = IcingConfig(prefill_cloud=False, **common)
    cfg_on = IcingConfig(prefill_cloud=True, **common)
    r_off = RimeIcingSimulation(cfg_off, log=lambda *a: None).run()
    r_on = RimeIcingSimulation(cfg_on, log=lambda *a: None).run()
    delta = r_on["audit"]["seeded"] - r_off["audit"]["seeded"]
    # steady inventory = rate * kill_x / u_in (read the exact kill plane)
    kill_x = RimeIcingSimulation(cfg_on, log=lambda *a: None).kill_x
    analytic = cfg_on.parcels_per_step * kill_x * cfg_on.m_parcel / cfg_on.u_in
    assert delta > 0
    # solid rejection + sampling noise -> within a few percent of analytic
    assert abs(delta - analytic) / analytic < 0.05, (delta, analytic)
    # and the audit still closes with the extra inventory
    assert r_on["audit"]["closure_error"] < 1e-2


# ---------------------------------------------------------------------------
# Rime density correlations (Macklin 1962 / Jones 1990)
# ---------------------------------------------------------------------------
def test_rime_density_macklin() -> None:
    # IRT point (-10 C, 67 m/s, 20 um): R ~ 75 -> capped at solid ice
    rho_irt = rime_density_macklin(20e-6, 67.0, -8.9)
    assert rho_irt == 917.0
    # cold + slow + small: the classic fluffy-rime corner
    rho_cold = rime_density_macklin(20e-6, 10.0, -30.0)
    assert 250.0 < rho_cold < 300.0
    # warmer / faster / bigger droplets densify monotonically up to the cap
    assert rime_density_macklin(20e-6, 10.0, -20.0) > rho_cold
    assert rime_density_macklin(40e-6, 10.0, -30.0) > rho_cold
    assert rime_density_macklin(20e-6, 10.0, 0.0) == 917.0  # no subcooling -> cap


def test_rime_density_config_modes() -> None:
    cfg = IcingConfig()  # IRT defaults
    # recovery heating: T_s = -10 + 0.5*67^2/(2*1005) = -8.88 C
    assert math.isclose(cfg.t_surface_eff_c, -10.0 + 0.5 * 67.0**2 / 2010.0, rel_tol=1e-9)
    assert cfg.rho_rime_eff == 917.0  # Macklin cap at this warm/fast point
    assert cfg.rime_R_macklin > 16.0  # well above the rho = 0.917 threshold
    cfg_const = IcingConfig(rime_density_mode="const", rho_rime=300.0)
    assert cfg_const.rho_rime_eff == 300.0
    cfg_jones = IcingConfig(rime_density_mode="jones")
    assert cfg_jones.rho_rime_eff == 917.0  # Jones also caps for this case
    with pytest.raises(ValueError):
        IcingConfig(rime_density_mode="bogus").rho_rime_eff


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def test_naca_mask_symmetry_and_placement() -> None:
    # (cx, cy) anchors the *leading edge*; airfoil spans [cx, cx + chord].
    # cy = (ny-1)/2 so the flip(0) mirror is exact.
    m0 = naca0012_mask_2d(120, 80, 48.0, aoa_deg=0.0, cx=60.0, cy=39.5)
    assert bool(m0[39, 61])  # leading-edge region
    assert bool(m0[39, 100])  # aft chord (TE itself is razor-thin)
    # symmetric about cy for AoA = 0
    assert torch.equal(m0, m0.flip(0))
    n0 = int(m0.sum())
    assert 0 < n0 < 48 * 48 * 0.25  # thin body, not a blob
    # positive AoA rotates nose-up about the LE: body tilts aft-down
    m4 = naca0012_mask_2d(120, 80, 48.0, aoa_deg=8.0, cx=60.0, cy=39.5)
    cy0 = float(torch.nonzero(m0, as_tuple=True)[0].float().mean())
    cy4 = float(torch.nonzero(m4, as_tuple=True)[0].float().mean())
    assert cy4 < cy0


def test_surface_arc_length_signs() -> None:
    af = naca0012_mask_2d(160, 96, 64.0, aoa_deg=0.0, cx=60.0, cy=48.0).numpy()
    s_grid, stag, surf = surface_arc_length(af)
    assert surf.any()
    assert s_grid.shape == af.shape
    (y_stag, x_stag) = stag
    assert x_stag == int(np.nonzero(af.any(axis=0))[0].min())  # LE origin
    above = s_grid[y_stag + 5 : y_stag + 20]
    below = s_grid[y_stag - 20 : y_stag - 5]
    assert (above[above != 0] > 0).all()
    assert (below[below != 0] < 0).all()
    assert np.abs(s_grid).max() > 10.0  # arc reaches aft


def test_ice_shape_metrics_synthetic() -> None:
    ny, nx = 60, 80
    af = np.zeros((ny, nx), dtype=bool)
    af[28:32, 20:60] = True  # slab "airfoil" (rows 28..31)
    ice = af.copy()
    ice[32, 30] = True  # 1 layer above (y > y_stag = "upper")
    ice[32, 40] = True
    ice[33, 40] = True  # 2 layers
    ice[34, 40] = True  # 3 layers -> upper horn
    ice[27, 45] = True  # 1 layer below -> lower horn
    m = ice_shape_metrics(af, ice, dx_phys=0.01, chord_phys=0.4, chord_lu=40.0,
                          stag=(30, 20))
    assert m["n_ice_cells"] == 5
    assert m["upper_horn_cells"] == 3
    assert m["lower_horn_cells"] == 1
    assert math.isclose(m["horn_symmetry_pct"], 100 * (3 - 1) / 4)
    assert m["ice_max_layer"] == 3
    assert math.isclose(m["ice_area_m2"], 5 * 1e-4)


# ---------------------------------------------------------------------------
# Mass conservation (uniform flow, no LBM)
# ---------------------------------------------------------------------------
def _small_cfg(**kw) -> IcingConfig:
    base = dict(
        nx=64,
        ny=48,
        chord_frac=0.4,
        cx_frac=0.05,  # LE near the inlet: droplets arrive within ~100 steps
        kill_frac=0.5,
        u_in=0.05,
        tau=0.55,
        aoa_deg=4.0,
        steps=300,
        warmup_steps=0,
        uniform_flow=True,
        mvd=200e-6,  # heavy droplets: fewer particles for the same mass flux
        rime_density_mode="const",
        rho_rime=100.0,  # low-density rime: freezing threshold within test budget
        accel_override=6.0e4,
        parcel_multiplier=4000,  # ~39 parcels/step: exact mass flux, cheap
        device="cpu",
        seed=0,
        log_every=10**9,
    )
    base.update(kw)
    return IcingConfig(**base)


def test_mass_audit_static_flow() -> None:
    cfg = _small_cfg()
    res = run_rime_icing(cfg, log=lambda *a: None)
    a = res["audit"]
    # gate: closure < 1 %
    assert a["closure_error"] < 1e-2, a
    # ice actually froze
    assert res["metrics"]["n_ice_cells"] >= 5
    # frozen mass == n_cells * cell ice mass (exact accounting)
    n_ice = res["metrics"]["n_ice_cells"]
    assert math.isclose(a["frozen"], n_ice * cfg.m_cell_ice, rel_tol=1e-9)
    # impacts happened, growth propagated upstream of the LE face
    assert res["n_impacts"] > 0
    ys, xs = np.nonzero(res["ice_only"])
    assert xs.min() < res["metrics"]["x_le"]
    # beta normalisation: peak O(1) at the stagnation line, and the total
    # window catch equals the fully-collected-streamtube *height* which can
    # never exceed the airfoil's projected height (chord*|sin aoa| + t*cos aoa)
    b = res["beta"]
    assert len(b["beta"]) and 0.05 < float(b["beta"].max()) <= 1.5
    capture_h = float(res["beta_grid"].sum())
    proj_h = cfg.chord_lu * (abs(math.sin(math.radians(cfg.aoa_deg)))
                             + cfg.naca_t * math.cos(math.radians(cfg.aoa_deg)))
    assert 0.0 < capture_h < proj_h * 1.05, (capture_h, proj_h)


# ---------------------------------------------------------------------------
# Geometry feedback smoke (coupled BGK flow)
# ---------------------------------------------------------------------------
def test_geometry_feedback_smoke() -> None:
    cfg = _small_cfg(
        uniform_flow=False,
        cx_frac=0.3,  # flow needs inlet clearance
        kill_frac=0.4,
        aoa_deg=4.0,
        steps=500,
        warmup_steps=1000,  # circulation needs ~3 chord-flow times to develop
        accel_override=5.0e4,
        parcel_multiplier=5000,  # ~26 parcels/step
    )
    res = run_rime_icing(cfg, log=lambda *a: None)
    a = res["audit"]
    assert a["closure_error"] < 1e-2, a
    assert res["metrics"]["n_ice_cells"] >= 1
    # positive AoA -> positive lift after warmup; positive drag
    assert res["cl0"] is not None and res["cl0"] > 0.0
    assert res["cd0"] is not None and res["cd0"] > 0.0
    assert res["cd_end"] is not None and math.isfinite(res["cd_end"])
    # ice only near the airfoil: bounded layer depth, near the LE
    m = res["metrics"]
    assert m["ice_max_layer"] <= 10
    assert m["ice_x_offset_min"] > -0.3  # not far upstream of LE
    assert m["ice_x_offset_max"] < 0.6  # only the front of the airfoil
    # history recorded the coupled phase
    assert len(res["history"]["ice_cells"]) == cfg.steps
    assert res["history"]["ice_cells"][-1] == m["n_ice_cells"]


# ---------------------------------------------------------------------------
# compile_mode (shared tensorlbm.compile_utils wrapper)
# ---------------------------------------------------------------------------
def test_compile_mode_validation() -> None:
    # cudagraph-class modes are rejected by the shared whitelist
    for bad in ("reduce-overhead", "max-autotune", "cudagraphs", "bogus"):
        with pytest.raises(ValueError):
            RimeIcingSimulation(
                _small_cfg(uniform_flow=False, compile_mode=bad), log=lambda *a: None
            )
    # accepted modes construct fine (compilation itself is lazy)
    sim = RimeIcingSimulation(
        _small_cfg(uniform_flow=False, compile_mode="default"), log=lambda *a: None
    )
    assert callable(sim._step_plain)


def test_compile_mode_flow_step_equivalence() -> None:
    """A compiled flow step matches the eager flow step numerically."""
    try:
        import torch._inductor  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.skip("torch inductor backend unavailable")
    torch.manual_seed(0)
    sim_e = RimeIcingSimulation(
        _small_cfg(uniform_flow=False), log=lambda *a: None
    )
    torch.manual_seed(0)
    sim_c = RimeIcingSimulation(
        _small_cfg(uniform_flow=False, compile_mode="default"), log=lambda *a: None
    )
    for _ in range(4):
        sim_e._flow_step(False)
        sim_c._flow_step(False)
    assert torch.allclose(sim_e.f, sim_c.f, rtol=1e-4, atol=1e-7)
