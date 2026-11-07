"""m4928 — acceptance tests for the SUBOFF tau-topic CLOUD artefacts.

Two layers:

1. **pure-function tests** (always run): the field post-processing and the GCI
   helper are exercised on synthetic data with known answers, so the analysis
   code is verified independently of the (slow) LBM run.
2. **artefact tests** (skipped when the field run has not been produced):
   cross-check the committed ``artifacts/m4928/lbm-suboff-tau-cloud`` files —
   shapes, provenance md5, table arithmetic, and the analytic/consistency
   criteria (stagnation Cp = 1, far-field u_x = U_inf, mass balance).

Run: pytest tests/test_m4928_lbm_suboff_tau_cloud.py -q
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "m4928" / "lbm-suboff-tau-cloud"
PLOT_MODULE = REPO / "benchmarks" / "lbm_suboff_tau_cloud_plot.py"


def _load_plot_module():
    spec = importlib.util.spec_from_file_location("m4928_plot", PLOT_MODULE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 1. pure functions
# --------------------------------------------------------------------------- #
def test_blasius_solution_matches_literature():
    m = _load_plot_module()
    eta, u = m.blasius_profile()
    # f'(eta) -> 1 and the wall slope f''(0) = 0.33206 (identical to 0.664/2)
    assert abs(u[-1] - 1.0) < 5e-3
    wall_slope = (u[1] - u[0]) / (eta[1] - eta[0])
    assert abs(wall_slope - 0.33206) / 0.33206 < 0.02
    # u/U = 0.99 slightly before eta = 5 (Blasius BL thickness 4.91)
    eta99 = float(np.interp(0.99, u, eta))
    assert 4.5 < eta99 < 5.3


def test_station_tables_recover_a_linear_wall_profile():
    """A Couette-like linear profile must give Cf = 2*nu*(U1/(1.5dx))/U_in^2."""
    from benchmarks.lbm_suboff_tau_cloud_run import station_tables  # noqa: PLC0415

    ny, nx = 40, 60
    dx = 0.1
    u_in = 0.05
    nu_lb = 0.0024
    solid = np.zeros((ny, nx), dtype=bool)
    solid[16:24, 20:40] = True  # slab "hull" in the middle band
    ux = np.zeros((ny, nx))
    slope = 0.02  # per cell, in u_in units of the lattice velocity
    for j in range(ny):
        ux[j, :] = slope * j
    rho = np.ones((ny, nx))
    tab = station_tables(rho, ux, np.zeros_like(ux), solid, cx_cells=30.0,
                         length_cells=20.0, u_in=u_in, nu_lb=nu_lb, dx=dx,
                         x_stations_norm=np.array([0.5]), n_profile=5)
    # upper surface cell j=24 -> u2 = slope*25 ; lower surface cell j=15 -> u2 = slope*14;
    # du/dn = u2/(1.5 dx) on each side, then the two sides are averaged.
    cf_up = 2.0 * nu_lb * (slope * 25.0 / (1.5 * dx)) / u_in**2
    cf_lo = 2.0 * nu_lb * (slope * 14.0 / (1.5 * dx)) / u_in**2
    assert tab["Cf_local_lbm"][0] == pytest.approx(0.5 * (cf_up + cf_lo), rel=1e-6)
    # delta*: sum(1 - u/u_in) * dx over the probe column, averaged over both sides
    ds_up = float(np.sum(1.0 - np.clip(slope * np.arange(24, 29), 0, 1.5)) * dx)
    ds_lo = float(np.sum(1.0 - np.clip(slope * np.arange(15, 10, -1), 0, 1.5)) * dx)
    assert tab["delta_star_m"][0] == pytest.approx(0.5 * (ds_up + ds_lo), rel=1e-6)
    # Cp uses p = rho/3 = 1/3 -> Cp = 0 exactly
    assert tab["Cp_wall"][0] == pytest.approx(0.0, abs=1e-12)


def test_gci_helper_flags_non_asymptotic_sequences():
    m = _load_plot_module()
    good = m.gci_three_level(1.0, 1.2, 1.35, r=1.5)
    assert good["applicable"] is True
    assert 0.5 <= good["p_obs"] <= 4.0
    assert good["gci_fine_pct"] > 0.0
    bad = m.gci_three_level(1.0, 1.2, 1.1, r=1.5)  # non-monotone
    assert bad["applicable"] is False
    assert "non-monotone" in bad["fallback_reason"]
    wild = m.gci_three_level(1.0, 1.0 + 1e-9, 3.0, r=1.5)  # absurd observed order
    assert wild["applicable"] is False
    assert wild["fallback_reason"]


# --------------------------------------------------------------------------- #
# 2. artefacts (skipped if the field run is absent)
# --------------------------------------------------------------------------- #
pytestmark_art = pytest.mark.skipif(not (ART / "field_cloud.json").exists(),
                                    reason="field-cloud artefacts not produced in this tree")


@pytestmark_art
def test_field_cloud_spec_is_machine_readable():
    spec = json.loads((ART / "field_cloud.json").read_text())
    for key in ("schema", "topic", "coordinates", "colour_scale", "tables", "figures",
                "checks", "provenance", "gci", "honest_notes"):
        assert key in spec, key
    assert spec["topic"] == "lbm-suboff-tau"
    assert "CLOUD" in spec["six_flags"]
    # colour bar units declared for every panel
    for panel in ("ux_over_U", "Cp", "omega_z", "cross_section_ux_over_U"):
        entry = spec["colour_scale"][panel]
        assert "unit" in entry and "range" in entry
    # every distinguished value must trace back to a field file with a md5
    for _L, prov in spec["provenance"].items():
        npz_path = REPO / prov["npz"] if not Path(prov["npz"]).is_absolute() else Path(prov["npz"])
        assert npz_path.exists(), npz_path
        h = hashlib.md5()
        with open(npz_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        assert h.hexdigest() == prov["npz_md5"]


@pytestmark_art
def test_field_arrays_are_self_consistent():
    spec = json.loads((ART / "field_cloud.json").read_text())
    for L in spec["grid_levels"]:
        npz = np.load(ART / f"fields_L{L}.npz")
        solid = npz["solid_cp"].astype(bool)
        for name in ("ux_cp", "uy_cp", "uz_cp", "Cp_cp", "vort_cp"):
            assert npz[name].shape == solid.shape, (L, name)
        assert npz["st_xL"].size == npz["st_Cp_wall"].size
        # hull occupies a contiguous streamwise band and nothing beyond the domain
        cols = np.flatnonzero(solid.any(axis=0))
        assert cols.size > 0
        assert cols.min() < cols.max()
        # probe series monotone in step, finite values present
        assert np.all(np.diff(npz["probe_step"]) > 0)
        assert np.isfinite(npz["probe_Cd_total"]).sum() > 10


@pytestmark_art
def test_analytic_field_checks_within_declared_tolerance():
    """Stagnation Cp = 1 (analytic), far-field u_x = U_inf, and mass balance."""
    spec = json.loads((ART / "field_cloud.json").read_text())
    for L in spec["grid_levels"]:
        cp_stag = spec["checks"]["Cp_stagnation"][str(L)]
        assert abs(cp_stag - 1.0) <= 0.05, (L, cp_stag)  # analytic stagnation value
        ux_inf = spec["checks"]["farfield_ux_over_U"][str(L)]
        assert abs(ux_inf - 1.0) <= 0.03, (L, ux_inf)
        mass = spec["checks"]["mass_imbalance_pct"][str(L)]
        assert mass <= 5.0, (L, mass)


@pytestmark_art
def test_table_arithmetic_reproduces_from_source_data():
    """Every error column must be arithmetically reproducible from its inputs."""
    spec = json.loads((ART / "field_cloud.json").read_text())
    with open(ART / "table1_grid_levels.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(spec["grid_levels"])
    for row in rows:
        L = int(row["resolution_cells_per_L"])
        cd, cd_ref, cd_err = (float(row["Cd_total_last100"]), float(row["Cd_ref_blasius"]),
                              float(row["Cd_err_pct"]))
        assert abs((cd - cd_ref) / cd_ref * 100.0 - cd_err) < 1e-6
        cf, cf_ref, cf_err = (float(row["Cf_local_xL0.5"]), float(row["Cf_local_blasius_xL0.5"]),
                             float(row["Cf_local_err_pct"]))
        assert abs((cf - cf_ref) / cf_ref * 100.0 - cf_err) < 1e-6
        # the CSV value must equal the value stored in the level summary
        summary = json.loads((ART / f"summary_L{L}.json").read_text())
        assert abs(summary["drag"]["cd_window_last100_samples"] - cd) < 1e-12
        assert abs(summary["field_checks"]["Cp_stagnation_final"]
                   - float(row["Cp_stagnation"])) < 1e-12

    with open(ART / "table2_stations.csv") as fh:
        st_rows = list(csv.DictReader(fh))
    assert len(st_rows) >= 10
    for row in st_rows:
        cf_lbm = float(row["Cf_local_lbm"])
        cf_bl = float(row["Cf_local_blasius"])
        err = float(row["Cf_local_err_pct"])
        assert abs((cf_lbm - cf_bl) / cf_bl * 100.0 - err) < 1e-6
        assert math.isfinite(float(row["Cp_wall"]))


@pytestmark_art
def test_gci_reports_either_value_or_fallback_reason():
    spec = json.loads((ART / "field_cloud.json").read_text())
    gci = spec["gci"]
    checked = 0
    for key, entry in gci.items():
        if not isinstance(entry, dict):
            continue
        checked += 1
        if entry.get("applicable"):
            assert entry["gci_fine_pct"] is not None
            assert entry["p_obs"] is not None
        else:
            assert entry["fallback_reason"], key
    assert checked >= 1


@pytestmark_art
def test_figures_exist_and_are_non_trivial():
    spec = json.loads((ART / "field_cloud.json").read_text())
    for name in spec["figures"].values():
        path = REPO / name
        assert path.exists(), path
        assert path.stat().st_size > 20_000, (name, path.stat().st_size)
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"