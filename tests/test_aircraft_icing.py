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
    run_glaze_icing,
    run_rime_icing,
    seed_counts_total,
    surface_arc_length,
    _tvd_face_states,
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
    # trailing beta window: this test's streamtube-height gate is written
    # against the Phase 2a window semantics; this high-accel config fills
    # the LE cell within ~66 steps, leaving a pre-freeze clean window far
    # too short for parcel-lump statistics (task #84 fix 1 note).
    cfg = _small_cfg(beta_window_mode="trailing")
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
# Task #84 fix 4: pending-water decomposition + freezer leftover cascade
# ---------------------------------------------------------------------------
def test_freeze_leftover_cascade_and_pending_split() -> None:
    """#84 fix 4: stranded sub-cell water cascades off frozen cells.

    A frozen cell can never freeze again (the freeze mask excludes solid
    cells), so its post-freeze remainder and its engulfed cloud water
    would strand on the ice as permanently pending mass.  The freezer now
    cascades both to the outward fluid 4-neighbour (the _deposit_columns
    rule), and the final audit splits ``pending`` into

    * ``pending_fluid`` — sub-cell water still collecting on fluid cells
      (legitimate in-transit remainder of the exposure window), and
    * ``pending_solid`` — what the cascade could not relocate (pockets
      fully enclosed by ice).
    """
    cfg = _small_cfg(beta_window_mode="trailing")
    res = run_rime_icing(cfg, log=lambda *a: None)
    a = res["audit"]
    n_ice = res["metrics"]["n_ice_cells"]
    assert n_ice >= 5
    assert math.isclose(a["frozen"], n_ice * cfg.m_cell_ice, rel_tol=1e-9)
    # decomposition closes on the total
    assert math.isclose(
        a["pending"], a["pending_fluid"] + a["pending_solid"],
        rel_tol=1e-12, abs_tol=1e-15,
    )
    # cascade: stranded water is the minority (in this high-accel config
    # a few fully-enclosed pockets legitimately keep theirs; at production
    # sizing the cascade relocates everything and pending_solid == 0)
    assert a["pending_solid"] < a["pending_fluid"], a
    assert a["closure_error"] < 1e-2

    # unit: one freeze event relocates the full remainder outward
    sim = RimeIcingSimulation(cfg, log=lambda *a: None)
    sim.m_w.zero_()
    y0, x0 = sim.y_le, sim.x_le - 1  # fluid cell just upstream of the LE
    assert not bool(sim.solid[y0, x0])
    sim.m_w[y0, x0] = 2.4 * cfg.m_cell_ice
    before = float(sim.m_w.sum())
    n = sim._freeze()
    assert n == 1
    assert bool(sim.solid[y0, x0])
    assert float(sim.m_w[y0, x0]) == 0.0  # nothing strands on the ice
    nz = torch.nonzero(sim.m_w > 0)
    assert nz.shape[0] == 1
    yy, xx = nz[0].tolist()
    assert not bool(sim.solid[yy, xx])  # recipient is fluid -> still freezable
    assert (yy, xx) != (y0, x0)
    assert math.isclose(float(sim.m_w.sum()), before - cfg.m_cell_ice, rel_tol=1e-6)


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


# ---------------------------------------------------------------------------
# Phase 2b: Eulerian droplet field + CUMULANT collision
# ---------------------------------------------------------------------------
def _euler_cfg(**kw) -> IcingConfig:
    """Small fast Eulerian-phase config (uniform flow unless overridden).

    ``rho_rime=1e9`` keeps the freezer off so the alpha transport audit is
    clean; tests that exercise freezing override it.
    """
    base = dict(
        nx=128,
        ny=64,
        chord_frac=0.4,
        cx_frac=0.3,
        kill_frac=0.5,
        u_in=0.05,
        tau=0.55,
        aoa_deg=4.0,
        steps=500,
        warmup_steps=0,
        uniform_flow=True,
        mvd=100e-6,
        accel_override=50.0,
        rime_density_mode="const",
        rho_rime=1.0e9,  # nothing freezes: pure transport
        device="cpu",
        seed=0,
        log_every=10**9,
        droplet_phase="eulerian",
    )
    base.update(kw)
    return IcingConfig(**base)


def test_phase2b_config_validation() -> None:
    for bad_kw in (
        {"droplet_phase": "euler"},  # not a phase name
        {"collision": "mrt"},  # only bgk | cumulant
        {"collision": "bgk", "c_s": 0.1},  # SGS needs cumulant
        {"drag_law": "clift"},
        {"re_lu_target": -5.0},
        {"shadow_alpha_frac": -1.0},
    ):
        with pytest.raises(ValueError):
            IcingConfig(**bad_kw)
    # accepted combinations construct fine
    IcingConfig(collision="cumulant", c_s=0.1, re_lu_target=2.0e6, drag_law="schiller-naumann")


def test_re_lu_target_and_drag_scales() -> None:
    cfg = IcingConfig(re_lu_target=2000.0)
    assert math.isclose(cfg.tau_flow, 3.0 * cfg.u_in * cfg.chord_lu / 2000.0 + 0.5, rel_tol=1e-12)
    assert math.isclose(cfg.re_lu, 2000.0, rel_tol=1e-9)  # nu_lu follows tau_flow
    cfg_free = IcingConfig()  # no target: Phase 2a tau semantics
    assert cfg_free.tau_flow == cfg_free.tau
    # alpha_in is the accelerated cloud volume fraction (same k as LWC_eff)
    cfg_a = IcingConfig(nx=100, ny=64, accel_override=25.0)
    assert math.isclose(cfg_a.alpha_in, cfg_a.lwc_eff / cfg_a.rho_water, rel_tol=1e-12)
    # Stokes -> sn_scale exactly 0; Schiller-Naumann -> physical Re_p scale
    assert cfg_a.re_p_scale == 0.0
    cfg_sn = IcingConfig(drag_law="schiller-naumann")
    expect = cfg_sn.rho_air * (cfg_sn.dx_phys / cfg_sn.dt_phys) * cfg_sn.mvd / cfg_sn.mu_air
    assert math.isclose(cfg_sn.re_p_scale, expect, rel_tol=1e-12)
    # IRT point: |du| ~ V_inf gives Re_p ~ 100 (f_drag ~ 4.6)
    assert 80.0 < cfg_sn.u_in * cfg_sn.re_p_scale < 120.0


def test_eulerian_uniform_frontal_catch_analytic() -> None:
    """Uniform flow: deposited rate == alpha_in * u_in * H_front exactly.

    Straight trajectories make the catch a pure geometric streamtube: every
    cell whose right neighbour is solid receives alpha_in*u_in per step.
    """
    cfg = _euler_cfg()
    res = run_rime_icing(cfg, log=lambda *a: None)
    e = res["eulerian"]
    assert e is not None
    h_front = int(res["airfoil"].any(axis=1).sum())
    analytic = cfg.alpha_in * cfg.u_in * h_front  # lattice volume units / step
    measured = e["audit"]["deposited"] / cfg.mass_per_lu3 / cfg.steps
    assert abs(measured - analytic) / analytic < 1e-5, (measured, analytic)
    # streamtube identity: integrated beta == frontal projection height
    assert abs(float(e["beta_grid"].sum()) - h_front) < 0.05


def test_eulerian_mass_audit_and_positivity() -> None:
    cfg = _euler_cfg()
    res = run_rime_icing(cfg, log=lambda *a: None)
    e = res["eulerian"]
    a = e["audit"]
    # conservative FV + device-side fp64 accumulation: closes orders of
    # magnitude below the Phase 2a Lagrangian 1e-2 gate (measured ~1e-9)
    assert a["closure_error"] < 1e-6, a
    # steady state: what enters through the inlet leaves through the outlet
    assert a["outlet_out"] > 0.9 * a["inlet_in"]
    alpha = e["alpha"]
    assert float(alpha.min()) >= 0.0  # positivity preserved
    # donor-cell FV is positivity-preserving; the only overshoot is a
    # <=0.2 % boundary artefact at the outlet corner (measured), far from
    # the airfoil where beta is sampled
    assert float(alpha.max()) <= cfg.alpha_in * 1.02


def test_eulerian_freezer_interface() -> None:
    """Eulerian deposits freeze through the *same* Phase 2a freezer."""
    cfg = _euler_cfg(
        accel_override=2.0e5,
        rho_rime=100.0,  # low-density rime: cells freeze within budget
        steps=400,
    )
    res = run_rime_icing(cfg, log=lambda *a: None)
    a = res["audit"]
    e = res["eulerian"]
    n_ice = res["metrics"]["n_ice_cells"]
    assert n_ice >= 5
    # exact ledger: frozen mass == n_cells * cell ice mass (2a invariant)
    assert math.isclose(a["frozen"], n_ice * cfg.m_cell_ice, rel_tol=1e-9)
    # the alpha audit still closes with freezing + encasement
    assert e["audit"]["closure_error"] < 1e-4, e["audit"]
    assert e["audit"]["encased"] > 0.0  # trapped cloud was accounted
    # ice grows on the windward face (upstream of the LE), like Phase 2a
    ys, xs = np.nonzero(res["ice_only"])
    assert xs.min() < res["metrics"]["x_le"]


def test_eulerian_beta_bounds() -> None:
    """Beta boundary behaviour on the clean airfoil (uniform flow)."""
    cfg = _euler_cfg()
    res = run_rime_icing(cfg, log=lambda *a: None)
    e = res["eulerian"]
    b = e["beta"]
    assert len(b["beta"]) and 0.05 < float(b["beta"].max()) <= 1.5
    # uniform straight-line catch caps beta at 1 on frontal faces
    assert float(b["beta"].max()) <= 1.0 + 1e-3
    # capture height bounded by the geometric projection (2a gate)
    capture_h = float(e["beta_grid"].sum())
    proj_h = cfg.chord_lu * (abs(math.sin(math.radians(cfg.aoa_deg)))
                             + cfg.naca_t * math.cos(math.radians(cfg.aoa_deg)))
    assert 0.0 < capture_h < proj_h * 1.05, (capture_h, proj_h)
    # support is local to the leading edge (no far-field impacts)
    assert abs(b["s_over_c"]).max() < 0.5


# ---------------------------------------------------------------------------
# Task #84 fix 1: beta window on the pre-ice reference geometry
# ---------------------------------------------------------------------------
def test_beta_window_mode_validation_and_bounds() -> None:
    for bad in ("bogus", "early"):
        with pytest.raises(ValueError):
            _euler_cfg(beta_window_mode=bad)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            _euler_cfg(beta_clean_frac=bad)
    # trailing mode keeps the Phase 2a/2b semantics exactly
    cfg = _euler_cfg(beta_window_mode="trailing")
    assert cfg.beta_window_bounds == (250, 500)
    # clean mode: early window, shorter than the trailing half
    cfg = _euler_cfg()
    w0, w1 = cfg.beta_window_bounds
    assert 0 < w0 < w1 <= 500
    assert w1 - w0 <= max(1, int(cfg.beta_clean_frac * cfg.steps))


def test_beta_window_clean_fill_cap() -> None:
    """Clean window closes before any wall cell can fill with ice.

    Frontal bound on the leading-edge catch: alpha_in * u_in per step, so
    the fill accumulated over the window stays below one cell-ice mass
    and no cell turns solid inside the beta window.
    """
    cfg = _euler_cfg(rho_rime=100.0, accel_override=2.0e5, steps=400)
    w0, w1 = cfg.beta_window_bounds
    assert w1 > w0
    fill = cfg.alpha_in * cfg.u_in * (w1 - w0)  # [lu^3]
    m_cell_lu = cfg.rho_rime_eff / cfg.rho_water
    assert fill <= cfg.beta_clean_max_fill * m_cell_lu * (1.0 + 1.0 / (w1 - w0))


def test_beta_window_clean_lwc_invariance() -> None:
    """Clean-window beta is an LWC invariant (reference geometry).

    With the freezer active (rho_rime=100, high accel: cells fill within
    the run) the legacy trailing window saturates at the moving-boundary
    cap and beta_pk collapses as LWC grows; the clean window measures the
    same pre-ice collection efficiency at both LWCs.
    """
    res = {}
    for lwc in (2.5e-4, 1.0e-3):
        cfg = _euler_cfg(lwc=lwc, rho_rime=100.0, accel_override=2.0e5, steps=400)
        res[lwc] = run_rime_icing(cfg, log=lambda *a: None)
    b_lo = res[2.5e-4]["eulerian"]["beta"]
    b_hi = res[1.0e-3]["eulerian"]["beta"]
    pk_lo = float(b_lo["beta"].max())
    pk_hi = float(b_hi["beta"].max())
    assert pk_lo > 0.05 and pk_hi > 0.05
    # LWC invariance within a few percent (float rounding on the scaled
    # alpha field; the underlying dynamics are linear in alpha_in)
    assert abs(pk_hi - pk_lo) / pk_lo < 0.05, (pk_lo, pk_hi)


# ---------------------------------------------------------------------------
# Task #84 fix 2: sn_scale_factor + beta_cap_window (from #79 calibration)
# ---------------------------------------------------------------------------
def test_sn_scale_factor_decouples_drag_from_rho_air() -> None:
    """sn_scale_factor = lambda is exactly the rho_air = lambda*1.34 sweep."""
    common = dict(drag_law="schiller-naumann")
    cfg_knob = _euler_cfg(sn_scale_factor=2.4, **common)
    cfg_rho = _euler_cfg(rho_air=2.4 * 1.34, **common)
    assert math.isclose(cfg_knob.re_p_scale, cfg_rho.re_p_scale, rel_tol=1e-12)
    # default 1.0 keeps the physical Schiller-Naumann scale unchanged
    cfg_phys = _euler_cfg(**common)
    cfg_legacy = _euler_cfg(**common)
    assert cfg_phys.re_p_scale == cfg_legacy.re_p_scale
    # stokes law: knob is inert (re_p_scale == 0 switches f_drag to 1)
    assert _euler_cfg(drag_law="stokes", sn_scale_factor=5.0).re_p_scale == 0.0
    with pytest.raises(ValueError):
        _euler_cfg(sn_scale_factor=-0.1)


def test_beta_cap_window_value_and_semantics() -> None:
    """Cap formula reproduces the #79 analytic value; empty window -> inf."""
    # standard 2b production case: dx=4.17 mm, 360 s, trailing half window
    cfg = IcingConfig(
        nx=320, ny=160, steps=3000, warmup_steps=0, uniform_flow=True,
        droplet_phase="eulerian", lwc=0.5e-3, t_exposure=360.0,
        rime_density_mode="macklin", beta_window_mode="trailing",
        device="cpu", log_every=10**9,
    )
    analytic = cfg.rho_rime_eff * cfg.dx_phys / (
        cfg.lwc * cfg.v_inf * (1.0 - cfg.beta_window_frac) * cfg.t_exposure
    )
    assert math.isclose(cfg.beta_cap_window, analytic, rel_tol=1e-12)
    assert math.isclose(cfg.beta_cap_window, 0.6337, abs_tol=5e-4)  # #79 table
    # glaze whole-shot convention (frac=0 -> empty window) is cap-free
    glaze_cfg = _euler_cfg(beta_window_mode="trailing", beta_window_frac=0.0)
    assert glaze_cfg.beta_window_bounds == (500, 500)
    assert math.isinf(glaze_cfg.beta_cap_window)
    # mapping report carries both diagnostics
    rep = _euler_cfg().mapping_report()
    assert "sn_scale_factor" in rep and "beta_cap_window" in rep


# ---------------------------------------------------------------------------
# Task #84 fix 3: second-order TVD Eulerian advection (donor2)
# ---------------------------------------------------------------------------
def test_eulerian_scheme_validation() -> None:
    with pytest.raises(ValueError):
        _euler_cfg(eulerian_scheme="muscl")
    assert _euler_cfg().eulerian_scheme == "donor2"  # new default
    assert _euler_cfg(eulerian_scheme="donor").eulerian_scheme == "donor"


def test_eulerian_donor2_tvd_face_reconstruction() -> None:
    """_tvd_face_states: exact on uniform/linear data, interval-safe on any data.

    * uniform field: zero downwind difference -> face == donor state exactly
      (the donor2 upgrade is bitwise-neutral on a uniform cloud, so the
      far-field/free-stream transport is identical to the legacy scheme);
    * linear field: second order (face == exact midpoint on interior faces);
    * random field: the reconstructed face never leaves the interval spanned
      by its two cells (local TVD: no over/undershoot);
    * non-interior faces (border / solid neighbour) fall back to the plain
      donor state.
    """
    torch.manual_seed(3)
    ny, nx = 8, 16
    j = torch.arange(nx, dtype=torch.float64)[None, :].expand(ny, nx)

    def face_arrays(axis):
        shp = (ny, nx - 1) if axis == 1 else (ny - 1, nx)
        pos = torch.rand(shp) < 0.5
        interior = torch.ones(shp, dtype=torch.bool)
        return pos, interior

    # uniform: exact donor state on every face, both axes, both wind signs
    q_u = torch.full((ny, nx), 0.7)
    for axis in (0, 1):
        pos, interior = face_arrays(axis)
        f = _tvd_face_states(q_u, pos, interior, axis)
        assert torch.equal(f, torch.full_like(f, 0.7))

    # linear: exact midpoint on interior faces (2nd order); border faces
    # (index 0 / last) use the plain donor state by construction
    q_l = 0.1 * j.clone()
    pos, interior = face_arrays(1)
    f = _tvd_face_states(q_l, pos, interior, axis=1)
    mid = 0.1 * (j[:, :-1] + 0.5)
    err = (f - mid).abs()
    assert float(err[:, 1:-1].max()) < 1e-9

    # random: interval bound on interior faces
    q_r = torch.rand(ny, nx)
    for axis in (0, 1):
        pos, interior = face_arrays(axis)
        f = _tvd_face_states(q_r, pos, interior, axis)
        if axis == 1:
            lo = torch.minimum(q_r[:, :-1], q_r[:, 1:])
            hi = torch.maximum(q_r[:, :-1], q_r[:, 1:])
            sel = interior[:, 1:-1]
            fv, lov, hiv = f[:, 1:-1][sel], lo[:, 1:-1][sel], hi[:, 1:-1][sel]
        else:
            lo = torch.minimum(q_r[:-1, :], q_r[1:, :])
            hi = torch.maximum(q_r[:-1, :], q_r[1:, :])
            sel = interior[1:-1, :]
            fv, lov, hiv = f[1:-1, :][sel], lo[1:-1, :][sel], hi[1:-1, :][sel]
        assert bool((fv >= lov - 1e-12).all()) and bool((fv <= hiv + 1e-12).all())

    # non-interior faces: plain donor state regardless of the field
    pos, interior = face_arrays(1)
    interior[:, :] = False
    f = _tvd_face_states(q_r, pos, interior, axis=1)
    donor = torch.where(pos, q_r[:, :-1], q_r[:, 1:])
    assert torch.equal(f, donor)


def test_eulerian_donor2_uniform_flow_bitwise_legacy() -> None:
    """Coupled uniform-flow run on the donor2 default: safe and closed."""
    r_2 = run_rime_icing(_euler_cfg(eulerian_scheme="donor2"), log=lambda *a: None)
    a2 = r_2["eulerian"]["alpha"]
    assert float(a2.min()) >= 0.0
    assert float(a2.max()) <= r_2["eulerian"]["alpha_in"] * 1.02 + 1e-12
    assert r_2["eulerian"]["audit"]["closure_error"] < 1e-6
    assert not np.isnan(a2).any()


def test_eulerian_donor2_advection_accuracy_and_tvd() -> None:
    """Gaussian/step advection: donor2 beats donor; TVD (no over/undershoot).

    Pure advection (drag frozen, no shadow, no solid): a Gaussian and a
    step advected across the grid.  Second order keeps the peak; the
    van Leer limiter keeps the solution inside the data range.
    """
    ny, nx = 32, 64
    dev = torch.device("cpu")
    ux = torch.full((ny, nx), 0.05, device=dev)
    uy = torch.zeros((ny, nx), device=dev)
    solid = torch.zeros((ny, nx), dtype=torch.bool, device=dev)
    xg = torch.arange(nx, dtype=torch.float32, device=dev)[None, :].expand(ny, nx)
    alpha0 = torch.exp(-((xg - 20.0) / 3.0) ** 2).clone()
    tau_frozen, alpha_in, u_in = 1.0e9, 0.05, 0.05
    peaks, maxima, minima = {}, {}, {}
    for scheme, donor2 in (("donor", False), ("donor2", True)):
        alpha = alpha0.clone()
        mx, my = alpha * ux, alpha * uy
        for _ in range(200):  # advect ~10 cells
            alpha, mx, my, _imp, _bf = RimeIcingSimulation._euler_step(
                alpha, mx, my, ux, uy, solid, tau_frozen, 0.0,
                alpha_in, u_in, -1.0, donor2,
            )
        peaks[scheme] = float(alpha.max())
        maxima[scheme] = float(alpha.max())
        minima[scheme] = float(alpha.min())
    assert minima["donor2"] >= 0.0 and minima["donor"] >= 0.0
    # second order: markedly less numerical diffusion of the peak
    assert peaks["donor2"] > peaks["donor"] + 0.15, peaks
    assert peaks["donor2"] > 0.8  # peak survives 10 cells of travel
    # step profile: monotone (no oscillation, no new extrema from the limiter).
    # NOTE the small (~2.5% at birth, decaying) max-bump at the cloud edge is
    # *legacy* scheme behaviour, present with donor too: in zero-alpha cells
    # ud = mx/alpha.clamp(eps) collapses to 0, halving the edge face
    # velocity, so the front cell under-outflows.  The gate is that donor2
    # adds nothing beyond that: max(donor2) <= max(donor) + 1e-3.
    alpha_s = (xg < 30.0).float().clone()
    step_max = {}
    for scheme, donor2 in (("donor", False), ("donor2", True)):
        alpha = alpha_s.clone()
        mx, my = alpha * ux, alpha * uy
        for _ in range(120):
            alpha, mx, my, _imp, _bf = RimeIcingSimulation._euler_step(
                alpha, mx, my, ux, uy, solid, tau_frozen, 0.0,
                alpha_in, u_in, -1.0, donor2,
            )
        assert float(alpha.min()) >= -1e-8  # no undershoot
        step_max[scheme] = float(alpha.max())
    assert step_max["donor2"] <= step_max["donor"] + 1e-3, step_max
    assert step_max["donor2"] < 1.03  # bounded legacy edge bump, no ringing


def test_eulerian_shadow_region_regularisation() -> None:
    """Huge shadow threshold penalizes u_d := u_f everywhere: still stable."""
    cfg = _euler_cfg(shadow_alpha_frac=1.0e6)
    res = run_rime_icing(cfg, log=lambda *a: None)
    e = res["eulerian"]
    assert np.isfinite(e["alpha"]).all()
    assert e["audit"]["closure_error"] < 1e-6, e["audit"]
    # shadow penalty removes inertia: geometric catch only
    h_front = int(res["airfoil"].any(axis=1).sum())
    analytic = cfg.alpha_in * cfg.u_in * h_front
    measured = e["audit"]["deposited"] / cfg.mass_per_lu3 / cfg.steps
    assert abs(measured - analytic) / analytic < 1e-5


def test_eulerian_drag_law_invariance_in_uniform_flow() -> None:
    """Drag law is irrelevant when u_d == u_f: SN and Stokes agree."""
    cfg_s = _euler_cfg(drag_law="stokes")
    cfg_n = _euler_cfg(drag_law="schiller-naumann")
    r_s = run_rime_icing(cfg_s, log=lambda *a: None)
    r_n = run_rime_icing(cfg_n, log=lambda *a: None)
    h_front = int(r_s["airfoil"].any(axis=1).sum())
    analytic = cfg_s.alpha_in * cfg_s.u_in * h_front
    for r in (r_s, r_n):
        measured = r["eulerian"]["audit"]["deposited"] / cfg_s.mass_per_lu3 / cfg_s.steps
        assert abs(measured - analytic) / analytic < 1e-5


def test_eulerian_lagrangian_cross_validation_uniform() -> None:
    """Same flow trajectory: L and E capture heights agree (geometry exact).

    Sizing: the Lagrangian arm carries 1/sqrt(N_impacts) Poisson noise, so
    the config targets ~1000 window impacts (~3 % noise): leading edge near
    the inlet, high acceleration, ~640 parcels/step with a ~2.4e5 steady
    inventory (CPU-friendly).
    """
    cfg = _euler_cfg(
        droplet_phase="both",
        parcel_multiplier=50,
        accel_override=5.2e3,
        cx_frac=0.08,
        kill_frac=0.3,
        steps=300,
        nx=96,
        ny=48,
    )
    res = run_rime_icing(cfg, log=lambda *a: None)
    h_l = float(res["beta_grid"].sum())
    h_e = float(res["eulerian"]["beta_grid"].sum())
    # Eulerian is the deterministic answer; both are the geometric frontal
    # height in uniform flow
    h_front = int(res["airfoil"].any(axis=1).sum())
    assert abs(h_e - h_front) < 0.05, (h_e, h_front)
    assert abs(h_l - h_e) / h_e < 0.08, (h_l, h_e)
    assert res["audit"]["closure_error"] < 1e-2
    assert res["eulerian"]["audit"]["closure_error"] < 1e-6


def test_eulerian_step_compile_equivalence() -> None:
    """The compiled Eulerian step matches the eager step exactly."""
    try:
        import torch._inductor  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.skip("torch inductor backend unavailable")
    torch.manual_seed(0)
    common = dict(nx=64, ny=48, chord_frac=0.4, cx_frac=0.3, aoa_deg=4.0,
                  warmup_steps=0, uniform_flow=False, droplet_phase="eulerian",
                  mvd=100e-6, accel_override=1e3, rime_density_mode="const",
                  rho_rime=1.0e9, device="cpu", log_every=10**9, seed=0,
                  steps=2)
    sim_e = RimeIcingSimulation(IcingConfig(compile_mode=None, **common), log=lambda *a: None)
    sim_c = RimeIcingSimulation(IcingConfig(compile_mode="default", **common), log=lambda *a: None)
    from tensorlbm.d3q19 import macroscopic3d

    _, ux3, uy3, _ = macroscopic3d(sim_e.f)
    sim_e._init_eulerian(ux3[0], uy3[0])
    sim_c._init_eulerian(ux3[0], uy3[0])
    for _ in range(3):
        sim_e._flow_step(False)
        sim_c._flow_step(False)
        _, uxe, uye, _ = macroscopic3d(sim_e.f)
        _, uxc, uyc, _ = macroscopic3d(sim_c.f)
        sim_e._euler_advance(uxe[0], uye[0])
        sim_c._euler_advance(uxc[0], uyc[0])
    assert torch.allclose(sim_e.f, sim_c.f, rtol=1e-4, atol=1e-7)
    assert torch.allclose(sim_e.alpha, sim_c.alpha, rtol=1e-4, atol=1e-12)


def test_eulerian_closure_coupled_flow_wake() -> None:
    """Regression (Phase 2b wake-leak bug): with a real LBM flow the wake/
    shadow cells used to develop |u_d| = mx/alpha >> 1, breaking the
    donor-cell CFL bound; the positivity clamp then *created* alpha and the
    audit leaked ~5% at production scale.  The droplet-velocity cap must
    keep the field positive and the audit closed on a coupled case."""
    cfg = _euler_cfg(
        uniform_flow=False,
        warmup_steps=300,
        steps=300,
        mvd=20e-6,  # strong drag: tau_d_lu ~ O(100), like production
        accel_override=5.0e4,
        rho_rime=100.0,  # let ice actually grow + encase cloud water
        nx=96,
        ny=48,
    )
    res = run_rime_icing(cfg, log=lambda *a: None)
    e = res["eulerian"]
    a = e["audit"]
    # alpha field stays physical: non-negative; near the stagnation line
    # the droplets converge and alpha legitimately concentrates above
    # alpha_in (the 1.02 bound only holds in uniform flow), but not by
    # an unbounded amount
    alpha_np = e["alpha"]
    assert float(alpha_np.min()) >= 0.0
    assert float(alpha_np.max()) <= cfg.alpha_in * 3.0
    # mass closes: in + initial == out + deposited + encased + airborne
    assert a["closure_error"] < 1e-3, a
    assert a["deposited"] > 0.0
    assert res["audit"]["seeded"] == 0.0  # no stray Lagrangian prefill


def test_cumulant_flow_smoke() -> None:
    """CUMULANT+Smag flow step runs, stays finite, cl0 in a sane band."""
    cfg = _euler_cfg(
        uniform_flow=False,
        collision="cumulant",
        c_s=0.1,
        re_lu_target=1000.0,
        warmup_steps=300,
        steps=60,
        nx=64,
        ny=48,
    )
    assert math.isclose(cfg.tau_flow - 0.5, 3.0 * cfg.u_in * cfg.chord_lu / 1000.0, rel_tol=1e-9)
    res = run_rime_icing(cfg, log=lambda *a: None)
    assert res["cd0"] is not None and res["cd0"] > 0.0
    assert res["cl0"] is not None and 0.0 < res["cl0"] < 1.0
    assert res["eulerian"]["audit"]["closure_error"] < 1e-4


def test_cumulant_step_compile_equivalence() -> None:
    try:
        import torch._inductor  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.skip("torch inductor backend unavailable")
    torch.manual_seed(0)
    common = dict(nx=64, ny=48, chord_frac=0.4, cx_frac=0.3, aoa_deg=4.0,
                  warmup_steps=0, uniform_flow=False, collision="cumulant",
                  c_s=0.1, re_lu_target=1000.0, mvd=100e-6, accel_override=1e3,
                  rime_density_mode="const", rho_rime=1.0e9, device="cpu",
                  log_every=10**9, seed=0, steps=2, droplet_phase="lagrangian",
                  disable_droplets=True)
    sim_e = RimeIcingSimulation(IcingConfig(compile_mode=None, **common), log=lambda *a: None)
    sim_c = RimeIcingSimulation(IcingConfig(compile_mode="default", **common), log=lambda *a: None)
    for _ in range(4):
        sim_e._flow_step(False)
        sim_c._flow_step(False)
    assert torch.allclose(sim_e.f, sim_c.f, rtol=1e-4, atol=1e-7)


# ---------------------------------------------------------------------------
# Phase 3: Messinger glaze thermodynamics (energy balance + runback)
# ---------------------------------------------------------------------------
def test_saturation_vapor_pressure_branches() -> None:
    """Both Magnus branches meet at 611 Pa; ice < water below 0 C."""
    from tensorlbm.aircraft_icing import saturation_vapor_pressure_pa as esat

    assert math.isclose(esat(0.0), 610.94, rel_tol=1e-3)
    assert math.isclose(esat(0.0, over_ice=True), esat(0.0, over_ice=False))
    assert abs(esat(-10.0, over_ice=False) - 286.8) < 2.0
    assert abs(esat(-10.0, over_ice=True) - 259.9) < 2.0
    assert abs(esat(20.0) - 2334.0) < 10.0
    # auto branch: ice below zero, water at/above zero
    assert esat(-10.0) == esat(-10.0, over_ice=True)
    assert esat(10.0) == esat(10.0, over_ice=False)
    assert esat(-10.0, over_ice=True) < esat(-10.0, over_ice=False)


def test_analytic_htc_magnitude_and_decay() -> None:
    """IRT-scale stagnation h and monotone streamwise decay."""
    from tensorlbm.aircraft_icing import analytic_htc_w_m2k, analytic_tau_pa

    cfg = IcingConfig()
    s = np.array([0.0, 0.02, 0.15, 0.53, 1.06])  # m along the arc from the LE
    v = np.full_like(s, cfg.v_inf)
    h = analytic_htc_w_m2k(s, v, cfg)
    assert np.all(h > 0.0) and np.all(np.isfinite(h))
    # Frossling stagnation value for the IRT case ~ 400 W/m^2 K
    assert 250.0 < h[0] < 600.0, h[0]
    # monotone decay away from the stagnation line
    assert np.all(np.diff(h) < 0.0), h
    assert h[-1] < 0.5 * h[0]
    # wall shear for the film diagnostic: ~1% of the dynamic pressure far
    # from the LE, vanishing at the stagnation line (Hiemenz taper)
    tau = analytic_tau_pa(s, v, cfg)
    q_dyn = 0.5 * cfg.rho_air * cfg.v_inf**2
    assert np.all(tau >= 0.0) and np.all(np.isfinite(tau))
    assert tau[0] < 0.01 * q_dyn
    assert 0.001 * q_dyn < tau[-1] < 0.05 * q_dyn, tau[-1]


def test_messinger_panel_regimes_and_energy_closure() -> None:
    """Rime/glaze/warm regimes with machine-precision energy closure."""
    from tensorlbm.aircraft_icing import analytic_htc_w_m2k, messinger_panel_fluxes

    h0 = float(analytic_htc_w_m2k(np.array([0.0]), np.array([IcingConfig().v_inf]), IcingConfig())[0])
    # cold -> rime: everything freezes, surface below 0 C, no runback
    cfg = IcingConfig(t_static_c=-20.0)
    A = cfg.glaze_panel_cells * cfg.dx_phys**2
    m_imp = 0.3 * cfg.lwc * cfg.v_inf * A
    r = messinger_panel_fluxes(cfg, m_imp, 0.0, cfg.t_static_c, h0, A, cfg.v_inf)
    assert r["regime"] == "rime"
    assert r["n_f"] == 1.0
    assert r["t_s_c"] < 0.0
    assert r["m_out"] == 0.0
    assert abs(r["residual_w"]) < 1e-9 * m_imp * cfg.l_fusion
    # mild cold -> glaze: partial freezing at 0 C with runback
    cfg = IcingConfig(t_static_c=-2.0)
    r = messinger_panel_fluxes(cfg, m_imp, 0.0, cfg.t_static_c, h0, A, cfg.v_inf)
    assert r["regime"] == "glaze"
    assert 0.0 < r["n_f"] < 1.0
    assert r["t_s_c"] == 0.0
    assert r["m_out"] > 0.0
    assert abs(r["residual_w"]) < 1e-9 * m_imp * cfg.l_fusion
    # warm -> nothing freezes, film above 0 C
    cfg = IcingConfig(t_static_c=5.0)
    r = messinger_panel_fluxes(cfg, m_imp, 0.0, cfg.t_static_c, h0, A, cfg.v_inf)
    assert r["regime"] == "warm"
    assert r["n_f"] == 0.0
    assert r["t_s_c"] > 0.0
    assert r["m_ice"] == 0.0
    assert abs(r["residual_w"]) < 1e-9 * m_imp * cfg.l_fusion


def test_messinger_temperature_sweep_nf_monotone() -> None:
    """n_f decreases monotonically as the static temperature rises."""
    from tensorlbm.aircraft_icing import analytic_htc_w_m2k, messinger_panel_fluxes

    cfg0 = IcingConfig()
    h0 = float(analytic_htc_w_m2k(np.array([0.0]), np.array([cfg0.v_inf]), cfg0)[0])
    A = cfg0.glaze_panel_cells * cfg0.dx_phys**2
    m_imp = 0.3 * cfg0.lwc * cfg0.v_inf * A
    nfs = []
    for t in (-30.0, -20.0, -15.0, -10.0, -7.0, -5.0, -2.0, 0.0, 5.0):
        cfg = IcingConfig(t_static_c=t)
        r = messinger_panel_fluxes(cfg, m_imp, 0.0, cfg.t_static_c, h0, A, cfg.v_inf)
        assert 0.0 <= r["n_f"] <= 1.0
        nfs.append(r["n_f"])
    assert nfs[0] == 1.0  # deep rime
    assert nfs[-1] == 0.0  # warm: no freezing
    assert all(a >= b - 1e-12 for a, b in zip(nfs, nfs[1:])), nfs


def _glaze_panels(cfg: IcingConfig, n_side: int = 20) -> dict:
    """Synthetic both-surface panels: gaussian beta around the stagnation line."""
    from tensorlbm.aircraft_icing import analytic_htc_w_m2k

    ds = cfg.glaze_panel_cells * cfg.dx_phys
    s_up = ds * np.arange(1, n_side + 1)
    s_lo = -ds * np.arange(1, n_side + 1)
    s = np.concatenate([[0.0], s_up, s_lo])
    n = len(s)
    beta = np.exp(-(s / (6.0 * ds)) ** 2)
    area = np.full(n, ds * cfg.dx_phys)
    v_e = np.full(n, cfg.v_inf)
    tau_t = np.full(n, 50.0)  # Pa
    h = analytic_htc_w_m2k(s, v_e, cfg)
    return {
        "s_m": s,
        "A_m2": area,
        "m_imp_kg_s": beta * cfg.lwc * cfg.v_inf * area,
        "beta": beta,
        "h": h,
        "v_e": v_e,
        "tau_t": tau_t,
        "dep_y": np.zeros(0, int),
        "dep_x": np.zeros(0, int),
        "dep_p": np.zeros(0, int),
        "dep_w": np.zeros(0),
        "n_panels": n,
    }


def test_glaze_mass_conservation_full_march() -> None:
    """impacted = frozen + evaporated + runback-off-surface over the march."""
    from tensorlbm.aircraft_icing import solve_glaze_surface

    for t in (-20.0, -10.0, -5.0, -2.0, 2.0):
        cfg = IcingConfig(t_static_c=t)
        sol = solve_glaze_surface(cfg, _glaze_panels(cfg), dt=120.0)
        a = sol["audit"]
        assert 0.0 <= a["closure_error"] < 1e-9, (t, a)
        assert np.all(sol["n_f"] >= 0.0) and np.all(sol["n_f"] <= 1.0)
        assert np.all(sol["m_runback_out_kg"] >= 0.0)


def test_runback_march_structure_and_monotonicity() -> None:
    """Runback flows strictly downstream: fluxes >= 0, inflow = upstream
    outflow, stagnation outflow splits between the two surfaces."""
    from tensorlbm.aircraft_icing import solve_glaze_surface

    cfg = IcingConfig(t_static_c=-5.0)  # glaze: runback active
    sol = solve_glaze_surface(cfg, _glaze_panels(cfg), dt=120.0)
    s, m_in, m_out = sol["s_m"], sol["m_runback_in_kg"], sol["m_runback_out_kg"]
    assert np.all(m_out >= 0.0) and np.all(m_in >= 0.0)
    i_stag = int(np.argmin(np.abs(s)))
    up = np.where(s > 0)[0][np.argsort(s[s > 0])]
    lo = np.where(s < 0)[0][np.argsort(-s[s < 0])]
    # stagnation outflow splits half/half to the two surfaces
    assert math.isclose(m_in[up[0]], 0.5 * m_out[i_stag], rel_tol=1e-12)
    assert math.isclose(m_in[lo[0]], 0.5 * m_out[i_stag], rel_tol=1e-12)
    # each side marches: panel k inflow == panel k-1 outflow
    for side in (up, lo):
        for a, b in zip(side[:-1], side[1:]):
            assert math.isclose(m_in[b], m_out[a], rel_tol=1e-12, abs_tol=1e-18)
    # the film diagnostic is finite, non-negative and physically sized
    film = sol["film_m"]
    assert np.all(np.isfinite(film)) and np.all(film >= 0.0)
    wet = m_out > 0.0
    assert np.all(film[wet] > 0.0)
    assert film.max() < 1e-2  # runback films are micron-scale, never cm


def test_glaze_rime_regression_gate_vs_2a() -> None:
    """Hard gate: cold limit degenerates to the 2a behaviour exactly.

    Rime-limit condition: -30 C, 0.2 g/m^3 — the convective cooling
    capacity exceeds the collection latent heat everywhere, so every wet
    panel must solve to n_f == 1 with sub-zero T_s and no runback (at the
    baseline -20 C / 0.5 g/m^3 the stagnation is genuinely glaze, n_f
    ~ 0.65, which is the physics, not a regression).

    Task #84 fix 4 note: the conservative freezer (water cascade) removed
    the silent stranding sink this gate was implicitly calibrated on, so
    the config was moved into the regime the ledger semantics assume
    (rho_rime 100 -> 800, steps 300 -> 900: same 1800 s window, per-step
    deposit < 1 cell, stagnation cell fills over ~20 steps, no accretion
    runaway).  The voxel comparison gets a 2-cell slack (integer
    granularity at n ~ 15); the exact statement is the *mass* one, now
    asserted directly on the 2a side: frozen + pending == delivered.
    """
    rime_kw = dict(t_static_c=-30.0, lwc=2.0e-4, t_exposure=1800.0, steps=900)
    cfg = _euler_cfg(
        thermo_model="messinger",
        evap_enabled=False,
        glaze_rho_mode="const",
        rime_density_mode="const",
        rho_rime=800.0,  # #84 fix 4: per-step deposit < 1 cell (no runaway)
        **rime_kw,
    )
    accel = cfg.t_exposure / (cfg.steps * cfg.dt_phys)
    # (a) the thermo: n_f == 1, zero runback, exact mass conservation
    g = run_glaze_icing(cfg, shots=1, log=lambda *a: None)
    p = g["panels"]
    wet = g["panels"]["m_ice_kg"] + g["panels"]["m_runback_out_kg"] > 0
    assert np.allclose(p["n_f"][wet], 1.0), p["n_f"][wet]
    assert g["audit"]["runback_out"] == 0.0
    assert g["audit"]["closure_error"] < 1e-9
    assert math.isclose(g["audit"]["frozen"], g["audit"]["impacted"], rel_tol=1e-9)
    # (b) measure-only twin (identical ledger, no in-run freezing)
    twin = run_rime_icing(
        _euler_cfg(
            thermo_model="instant",
            accel_override=accel,
            **rime_kw,
        ),
        log=lambda *a: None,
    )
    led_g = float(g["audit"]["impacted"])
    led_t = twin["eulerian"]["audit"]["deposited"]
    # fp32 impact ledger (panel binning) vs fp64 device audit: ~1e-6 rel
    assert math.isclose(led_g, led_t, rel_tol=5e-5), (led_g, led_t)
    # (c) full 2a (in-run freezing) matches the glaze voxel shape
    r2a = run_rime_icing(
        _euler_cfg(
            thermo_model="instant",
            accel_override=accel,
            rho_rime=800.0,
            rime_density_mode="const",
            **rime_kw,
        ),
        log=lambda *a: None,
    )
    n_2a = int(r2a["ice_only"].sum())
    n_g = int(g["ice_only"].sum())
    # integer voxel granularity at n ~ 15: 2-cell slack
    assert abs(n_2a - n_g) <= 2, (n_2a, n_g)
    # the exact statement is mass conservation on the 2a side: everything
    # the ledger delivered (impacts + engulfed cloud) is frozen or still
    # pending as liquid -- nothing strands silently (#84 fix 4)
    ae = r2a["eulerian"]["audit"]
    delivered = ae["deposited"] + ae["encased"]
    assert math.isclose(
        r2a["audit"]["frozen"] + r2a["audit"]["pending"], delivered, rel_tol=1e-5
    ), (r2a["audit"], ae)
    assert r2a["audit"]["pending_solid"] <= 0.05 * r2a["audit"]["pending"] + 1e-15
    # (d) evaporation on (default) keeps n_f == 1 in the rime limit
    cfg2 = _euler_cfg(thermo_model="messinger", **rime_kw)
    g2 = run_glaze_icing(cfg2, shots=1, log=lambda *a: None)
    wet2 = g2["panels"]["m_ice_kg"] > 0
    assert np.allclose(g2["panels"]["n_f"][wet2], 1.0)


def test_deposit_cascade_column() -> None:
    """3.5 cell-masses on one boundary cell freeze a 3-cell column."""
    from tensorlbm.aircraft_icing import deposit_glaze_ice

    ny, nx = 8, 6
    airfoil = np.zeros((ny, nx), dtype=bool)
    airfoil[3:, :] = True  # solid floor, fluid rows 0..2
    rho = 100.0
    dx = 0.5334 / (0.4 * nx)
    m_cell = rho * dx**3
    cell_mass = np.zeros((ny, nx))
    cell_mass[2, 3] = 3.5 * m_cell
    cell_rho = np.zeros((ny, nx))
    cell_rho[2, 3] = rho
    solid, m_w = deposit_glaze_ice(airfoil, airfoil.copy(), np.zeros((ny, nx)),
                                   cell_mass, cell_rho, dx)
    ice = solid & ~airfoil
    assert ice.sum() == 3
    assert ice[2, 3] and ice[1, 3] and ice[0, 3]  # column, not lateral spread
    assert math.isclose(float(m_w.sum()), 0.5 * m_cell, rel_tol=1e-12)
    assert m_w[ice].sum() == 0.0  # frozen cells fully consumed


def test_glaze_driver_smoke_uniform() -> None:
    """End-to-end multishot glaze run on a cheap uniform-flow case."""
    cfg = _euler_cfg(
        thermo_model="messinger",
        t_static_c=-5.0,
        t_exposure=3600.0,
        steps=300,
    )
    g = run_glaze_icing(cfg, shots=2, log=lambda *a: None)
    a = g["audit"]
    assert a["closure_error"] < 1e-9, a
    assert a["frozen"] > 0.0
    assert int(g["ice_only"].sum()) > 0
    m = g["metrics"]
    assert m["n_ice_cells"] == int(g["ice_only"].sum())
    assert m["ice_max_layer"] <= 10
    assert m["stag_ice_thickness_m"] > 0.0
    p = g["panels"]
    # both regimes exercised on the synthetic-ish uniform impingement
    assert "glaze" in p["regime"] or "rime" in p["regime"]
    assert np.all(p["n_f"] >= 0.0) and np.all(p["n_f"] <= 1.0)
    assert np.all(p["thickness_m"] >= 0.0)
    assert len(g["shot_reports"]) == 2
    # ice only adjacent to the airfoil
    assert m["ice_x_offset_max"] < 0.6 and m["ice_x_offset_min"] > -0.3


def test_phase3_config_validation() -> None:
    """Phase 3 selectors validate; defaults keep the 2a/2b paths intact."""
    for bad in ("bogus", "Messinger"):
        with pytest.raises(ValueError):
            IcingConfig(thermo_model=bad)
    for bad in ("cylinder", ""):
        with pytest.raises(ValueError):
            IcingConfig(htc_mode=bad)
    for bad in ("macklin", ""):
        with pytest.raises(ValueError):
            IcingConfig(glaze_rho_mode=bad)
    for bad in (0.0, 1.5, -0.2):
        with pytest.raises(ValueError):
            IcingConfig(rh=bad)
    with pytest.raises(ValueError):
        IcingConfig(glaze_panel_cells=0.0)
    d = IcingConfig()
    assert d.thermo_model == "instant" and d.freeze_in_run is True
    assert d.glaze_rho_mode == "macklin-ts" and d.htc_mode == "analytic"
    # recovery factor defaults to sqrt(Pr)
    assert math.isclose(d.recovery_factor_eff, math.sqrt(d.prandtl_air))
    # leading-edge effective diameter: 2 * 1.1019 * t^2 * chord (NACA 4-digit)
    assert math.isclose(d.le_diameter_eff, 2 * 1.1019 * 0.12**2 * d.chord_phys, rel_tol=1e-12)
    # run_glaze_icing rejects the instant model
    with pytest.raises(ValueError):
        run_glaze_icing(IcingConfig(nx=32, ny=24), shots=1, log=lambda *a: None)
