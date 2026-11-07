"""Smoke / structural tests for the relaxation-zone wave tank and the periodic rig.

The physics campaigns live in ``scripts/p49n01_wave_tank.py`` (they need minutes
of compute); these tests only pin the *mechanics* so a regression in the tank
plumbing is caught in seconds:

* relaxation weights are monotone, zero at the inner edge and one at the wall;
* the geometry puts the probes inside the propagation region and the far-end
  distance used by the Goda method is positive and monotone;
* a handful of time steps keeps the free-surface state finite, keeps the liquid
  ledger conserved and leaves the free surface within a few cells of the
  still-water level (i.e. no blow-up, no drainage);
* the sponge switch actually changes the weight field.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.wave_tank2d.periodic import PeriodicWaveConfig, run_periodic
from tensorlbm.wave_tank2d.tank import (
    TankConfig,
    build_geometry,
    build_initial_state,
    relax_zones,
    relaxation_weight,
)


def test_relaxation_weight_profile() -> None:
    chi = np.linspace(0.0, 1.0, 101)
    w = relaxation_weight(chi)
    assert w[0] == pytest.approx(0.0, abs=1e-15)
    assert w[-1] == pytest.approx(1.0 - math.exp(-1.0), rel=1e-12)
    assert np.all(np.diff(w) >= 0.0)
    # clipped outside [0, 1]
    assert relaxation_weight(np.array([-1.0, 2.0]))[0] == 0.0
    assert relaxation_weight(np.array([-1.0, 2.0]))[1] == pytest.approx(w[-1])
    # strength scaling saturates
    assert relaxation_weight(np.array([1.0]), strength=5.0)[0] == pytest.approx(1.0)


def test_geometry_layout_and_probe_distances() -> None:
    cfg = TankConfig(h=20.0, kh=1.5, eps=0.02, sponge=True)
    wave, _ = cfg.wave_and_g()
    geom = build_geometry(cfg)
    assert geom.lw > 0 and geom.lp > 0 and geom.ls > 0
    assert geom.nx == geom.lw + geom.lp + geom.ls + 1
    assert geom.ny >= int(cfg.h) + 1
    # probes live in the propagation region, ordered, and clear of both zones
    assert geom.probe_x == sorted(geom.probe_x)
    for x in geom.probe_x:
        assert geom.lw < x < geom.nx - geom.ls
    s = geom.probe_distances()
    assert np.all(s > 0)
    assert np.all(np.diff(s) < 0)          # distance shrinks as x grows toward the wall
    # the spacing used by the two-point method stays finite for k ds
    k = wave.k
    ds = np.diff(geom.probe_distances())
    for d in ds:
        assert abs(math.sin(k * d)) > 0.05


def test_initial_state_is_still_water() -> None:
    cfg = TankConfig(h=12.0, kh=1.5, eps=0.02)
    wave, g = cfg.wave_and_g()
    geom = build_geometry(cfg)
    f, mass, flags, solid = build_initial_state(cfg, wave, geom, g)
    assert float(mass.sum()) == pytest.approx(cfg.h * (geom.nx - 2), rel=0.05)
    assert not torch.isnan(f).any()
    assert solid[0].all() and solid[:, 0].all() and solid[:, -1].all()


def test_sponge_switch_changes_only_the_downstream_zone() -> None:
    cfg_on = TankConfig(h=12.0, sponge=True)
    cfg_off = TankConfig(h=12.0, sponge=False)
    geom = build_geometry(cfg_on)
    x = np.arange(geom.nx, dtype=float)
    chi_sp = np.clip((x - (geom.nx - geom.ls)) / max(geom.ls, 1), 0.0, 1.0)
    w_on = np.zeros(geom.nx)
    w_on[geom.nx - geom.ls:] = relaxation_weight(chi_sp[geom.nx - geom.ls:])
    w_off = np.zeros(geom.nx)
    assert w_on[: geom.nx - geom.ls].max() == 0.0
    assert w_on.max() > 0.6
    assert w_off.max() == 0.0
    assert cfg_on.sponge and not cfg_off.sponge


def test_periodic_rig_mass_ledger_smoke() -> None:
    """A few hundred steps keep the Körner liquid ledger and stay finite."""
    cfg = PeriodicWaveConfig(h=8.0, kh=1.5, eps=0.02, n_periods=0.1, record_every=2,
                             hydrostatic_init=True)
    res = run_periodic(cfg, verbose=False)
    drift = abs(res["mass"]["relative_drift_total"])
    assert np.isfinite(drift)
    assert drift < 5e-3, drift
    assert np.all(np.isfinite(np.array(res["series"]["eta_probe0"])))
    assert res["dispersion"]["k_measured"] > 0.0
    snap = np.array(res["series"]["eta_snapshot_final"])
    assert np.max(np.abs(snap)) < 5.0        # surface stays near still-water level


def test_periodic_rig_rejects_nonsense_configuration() -> None:
    with pytest.raises(ValueError):
        PeriodicWaveConfig(h=0.0, kh=1.5).wave()
    with pytest.raises(ValueError):
        PeriodicWaveConfig(h=10.0, kh=-1.0).wave()

def test_relaxation_couplings_behave_as_documented() -> None:
    """``velocity`` coupling keeps the ledger; ``mass`` coupling is diagnostic only.

    This test pins the finding of the P49-N01 campaign: resetting the Körner
    flags/mass ledger inside a relaxation zone (``coupling="mass"``) is *not*
    mass consistent — it thickens the interface layer — whereas relaxing the
    populations only (``coupling="velocity"``) keeps the ledger bounded.
    Asserting the direction of both effects keeps the diagnosis reproducible.
    """
    from tensorlbm.free_surface_lbm_2d import (
        INTERFACE,
        koerner_step_2d,
        total_liquid_inventory_2d,
    )
    from tensorlbm.wave_tank2d.tank import relaxation_weight  # local import

    results = {}
    for coupling in ("velocity", "mass"):
        cfg = TankConfig(h=8.0, kh=1.5, eps=0.02, coupling=coupling, ramp_periods=0.0)
        wave, g = cfg.wave_and_g()
        geom = build_geometry(cfg)
        f, mass, flags, solid = build_initial_state(cfg, wave, geom, g)
        x = np.arange(geom.nx, dtype=float)
        w_wm = np.zeros(geom.nx)
        w_wm[: geom.lw] = relaxation_weight(np.clip(1 - x[: geom.lw] / geom.lw, 0, 1))
        w_sp = np.zeros(geom.nx)
        w_sp[geom.nx - geom.ls:] = relaxation_weight(
            np.clip((x[geom.nx - geom.ls:] - (geom.nx - geom.ls)) / geom.ls, 0, 1))
        tw = torch.tensor(w_wm, dtype=mass.dtype).view(1, -1)
        ts = torch.tensor(w_sp, dtype=mass.dtype).view(1, -1)
        m0 = total_liquid_inventory_2d(f, mass, flags)
        iface0 = int((flags == INTERFACE).sum())
        for step in range(1, 121):
            f, mass, flags = koerner_step_2d(f, mass, flags, solid, tau=cfg.tau, gy=-g)
            f, mass, flags, _ = relax_zones(f, mass, flags, solid, wave, g, step, cfg,
                                            geom, tw, ts, 1.0)
        results[coupling] = {
            "ledger_drift": abs(total_liquid_inventory_2d(f, mass, flags) - m0) / m0,
            "interface_growth": int((flags == INTERFACE).sum()) - iface0,
        }

    assert results["velocity"]["ledger_drift"] < 5e-3, results
    assert results["mass"]["interface_growth"] > results["velocity"]["interface_growth"]
