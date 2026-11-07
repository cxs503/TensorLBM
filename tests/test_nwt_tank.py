"""End-to-end tests of the 2-D numerical wave tank driver.

These are deliberately cheap (a few seconds, ~1.7 k cells, ~1 k steps) so they
can run in CI: they check the *plumbing and invariants* of the flume, not the
accuracy acceptance (that is `benchmarks/nwt_stokes2_2d/run_acceptance.py`).

Invariants covered:

* the free-surface estimator returns exactly the still-water depth for a column
  built by the driver (no off-by-one, no compressibility bias);
* the flume geometry honours "generation zone >= 1 wavelength" and keeps every
  probe inside the fluid region;
* the wavemaker generates a wave that travels toward ``+x`` with the right
  period and a phase speed within 20 % of the closed form;
* with generation and absorption switched off the model conserves the liquid
  inventory to <1 % per period;
* two identical runs reproduce the same probe records.
"""

from __future__ import annotations

import math

import pytest
import torch

import tensorlbm.nwt as nwt
from tensorlbm.nwt import tank as tank_mod
from tensorlbm.nwt.wavemaker import column_free_surface

torch.set_num_threads(1) if torch.get_num_threads() > 4 else None

SMALL = dict(
    wavelength=24.0,
    depth=7.64,
    steepness=0.04,
    steps_per_period=344,
    n_periods=3.0,
    ramp_periods=1.0,
    analysis_periods=1.0,
    n_snapshots=3,
)


@pytest.fixture(scope="module")
def small_run():
    cfg = nwt.NWTConfig(**SMALL)
    return cfg, nwt.run_nwt(cfg)


def test_still_water_column_has_exact_depth():
    cfg = nwt.NWTConfig(**SMALL)
    geom = tank_mod.build_geometry(cfg)
    f, mass, flags, solid = tank_mod._initial_fill_and_flags(cfg, geom, torch.device("cpu"))
    V = column_free_surface(f, mass, flags)
    interior = V[1:-1]
    # float32 mass ledger -> the geometric column height is exact to ~1e-4
    assert float(interior.min()) == pytest.approx(cfg.depth, abs=2e-3)
    assert float(interior.max()) == pytest.approx(cfg.depth, abs=2e-3)


def test_geometry_respects_zone_lengths():
    cfg = nwt.NWTConfig(**SMALL)
    geom = tank_mod.build_geometry(cfg)
    lam = cfg.wavelength
    assert geom["x_generator_end"] - 1 >= lam  # generation zone >= 1 wavelength
    assert geom["sponge_len"] >= lam
    probes = [1 + int(round(cfg.generator_wavelengths * lam)) + int(round(cfg.probe_start_wavelengths * lam))]
    assert min(geom_note(cfg)) > geom["x_generator_end"]
    assert max(geom_note(cfg)) < geom["x_sponge_start"]


def geom_note(cfg):
    lam = cfg.wavelength
    gen = int(round(cfg.generator_wavelengths * lam))
    start = int(round(cfg.probe_start_wavelengths * lam))
    dx = cfg.probe_spacing_wavelengths * lam
    return [1 + gen + start + int(round(j * dx)) for j in range(cfg.n_probes)]


def test_wave_travels_downstream_with_correct_period(small_run):
    cfg, res = small_run
    eta = res.eta_probes
    assert eta.shape[0] == cfg.n_probes and eta.shape[1] > 0
    fit = nwt.fit_harmonics(eta[0], 1.0, cfg.omega)
    a = cfg.wave.amplitude
    assert fit["A1"] > 0.2 * a, f"no wave generated: A1={fit['A1']} vs a={a}"
    assert fit["A1"] < 3.0 * a, f"wave amplitude unphysical: A1={fit['A1']} vs a={a}"
    # the wave must arrive at the downstream probe later than at the upstream one
    cc = nwt.lag_from_cross_correlation(eta[0], eta[-1], 1.0)
    assert cc["lag_samples"] > 0.0
    c_meas = (res.probe_x[-1] - res.probe_x[0]) / cc["lag_time"]
    assert c_meas == pytest.approx(cfg.wave.phase_speed, rel=0.2)


def test_inventory_conserved_without_sources():
    cfg = nwt.NWTConfig(
        **{**SMALL, "generator_enabled": False, "sponge_enabled": False, "ramp_periods": 0.0, "n_periods": 1.0, "analysis_periods": 0.5}
    )
    res = nwt.run_nwt(cfg)
    drift = nwt.inventory_drift(res.inventory, cfg.steps_per_period)
    assert drift["max_relative_drift"] < 5e-3


def test_runs_are_reproducible(small_run):
    cfg, res = small_run
    cfg2 = nwt.NWTConfig(**{**SMALL, "n_periods": 1.0, "analysis_periods": 0.5})
    a = nwt.run_nwt(cfg2)
    b = nwt.run_nwt(cfg2)
    assert torch.equal(a.eta_probes, b.eta_probes)
    assert torch.equal(a.inventory, b.inventory)


def test_snapshots_and_diagnostics_present(small_run):
    cfg, res = small_run
    assert res.eta_snapshots.shape[0] == SMALL["n_snapshots"]
    assert res.eta_snapshots.shape[1] == res.geometry["nx"]
    diag = res.diagnostics
    for key in ("hydrostatic_compressibility_gh_cs2", "max_surface_velocity_ma", "second_harmonic_ratio"):
        assert math.isfinite(diag[key])
    assert diag["hydrostatic_compressibility_gh_cs2"] < 0.1
    assert diag["max_surface_velocity_ma"] < 0.05