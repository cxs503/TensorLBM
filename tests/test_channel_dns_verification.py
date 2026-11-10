"""pytest for the channel-DNS verification face (spectra + gate layer).

Everything here is a *unit* test on synthetic data: the spectra must recover a
known power law and the Parseval normalisation, the resolution attestation must
classify grids correctly, and the gate layer must be fail-closed (missing or
diverged artefacts never become ``pass``).  The end-to-end DNS numbers live in
``benchmarks/p223w1_n01_channel_dns`` and the run report, not in this file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.turbulence.channel_spectra import (
    PlaneDumpMeta,
    fit_loglog_slope,
    load_plane_dump,
    plane_spectra_legs,
    plane_spectrum,
    read_plane_meta,
)
from tensorlbm.turbulence.channel_dns import (
    DNS_TARGETS,
    batch_convergence,
    cf_targets,
    dns_cells_required,
    evaluate_dns_gates,
    resolution_metrics,
    spectrum_slope_gate,
)


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------
def _synthetic_plane(nx: int, nz: int, slope: float, snapshots: int = 4, seed: int = 3):
    """Plane field whose 1-D streamwise spectrum is exactly a power law.

    Each Fourier mode gets amplitude ``k**slope`` and an independent random
    phase per snapshot, so the *time-averaged* spectrum is ``k**(2*slope)`` in
    power -- i.e. ``E(k) ~ k**(2*slope)``.  Pass ``slope=-5/6`` for ``E ~ k**-5/3``.
    """
    rng = np.random.default_rng(seed)
    k = np.fft.rfftfreq(nx, d=1.0 / nx)  # integer mode numbers
    amp = np.zeros_like(k)
    amp[1:] = k[1:] ** slope
    out = np.zeros((snapshots, nz, nx))
    for s in range(snapshots):
        phase = rng.uniform(0.0, 2.0 * math.pi, size=k.size)
        spec = amp * np.exp(1j * phase)
        line = np.fft.irfft(spec, n=nx)
        out[s] = np.tile(line, (nz, 1))
    return out


def test_plane_spectrum_parseval():
    """sum_k E(k) equals the plane-mean variance of u' (Parseval)."""
    rng = np.random.default_rng(0)
    planes = rng.normal(size=(3, 16, 32)) * 0.5 + 1.0
    k, e = plane_spectrum(planes, axis=-1)
    assert k.shape == e.shape
    varying = planes - planes.reshape(planes.shape[0], -1).mean(axis=1)[:, None, None]
    assert float(np.sum(e)) == pytest.approx(float(np.mean(varying**2)), rel=0.25)
    # the plane-mean fluctuation is removed, so the k = 0 (z-averaged) mode
    # carries only the residual of the transverse mean profile -- a few percent
    assert e[0] < 0.10 * float(np.sum(e))


def test_plane_spectrum_recovers_power_law():
    planes = _synthetic_plane(128, 8, slope=-5.0 / 6.0)
    k, e = plane_spectrum(planes, axis=-1)
    fit = fit_loglog_slope(k[1:], e[1:], k_min=4.0, k_max=32.0)
    assert fit["valid"]
    assert fit["slope"] == pytest.approx(-5.0 / 3.0, abs=0.05)
    assert fit["r2"] > 0.99


def test_fit_loglog_slope_band_is_a_priori():
    """The default band is [k_min, max(k_min + 8, 0.25 k_max)] -- no search."""
    k = np.arange(1, 65, dtype=float)
    e = k**-2.0
    fit = fit_loglog_slope(k, e)
    assert fit["k_min"] == 4.0
    assert fit["k_max"] == pytest.approx(max(12.0, 0.25 * 64.0))
    assert fit["slope"] == pytest.approx(-2.0, abs=1e-9)


def test_fit_loglog_slope_withholds_on_short_band():
    k = np.array([1.0, 2.0, 3.0])
    fit = fit_loglog_slope(k, k**-2.0, min_points=4)
    assert not fit["valid"]
    assert math.isnan(fit["slope"])


def _write_fake_dump(tmp_path: Path, rows=(3, 9), nx=16, nz=8, snapshots=2):
    meta = PlaneDumpMeta(
        nx=nx,
        nz=nz,
        rows=tuple(rows),
        snapshots=snapshots,
        steps_between=10,
        bytes_per_snapshot=len(rows) * 3 * nx * nz * 4,
        layout="test",
    )
    rng = np.random.default_rng(7)
    data = rng.normal(size=snapshots * len(rows) * 3 * nx * nz).astype("<f4")
    bin_path = tmp_path / "dump_spec.bin"
    data.tofile(bin_path)
    meta_path = tmp_path / "dump_spec.json"
    meta_path.write_text(
        json.dumps(
            {
                "layout": meta.layout,
                "nx": nx,
                "nz": nz,
                "steps_between": meta.steps_between,
                "snapshots": snapshots,
                "rows": list(rows),
                "bytes_per_snapshot": meta.bytes_per_snapshot,
            }
        ),
        encoding="utf-8",
    )
    return bin_path, meta_path, data.reshape(snapshots, len(rows), 3, nz, nx)


def test_plane_dump_roundtrip(tmp_path: Path):
    bin_path, meta_path, reference = _write_fake_dump(tmp_path)
    meta = read_plane_meta(meta_path)
    dump = load_plane_dump(bin_path, meta)
    assert sorted(dump) == [3, 9]
    for ir, row in enumerate(meta.rows):
        np.testing.assert_allclose(dump[row]["ux"][0], reference[0, ir, 0], rtol=0, atol=0)
        np.testing.assert_allclose(dump[row]["uz"][1], reference[1, ir, 2], rtol=0, atol=0)


def test_plane_dump_rejects_wrong_size(tmp_path: Path):
    bin_path, meta_path, _ = _write_fake_dump(tmp_path)
    bin_path.write_bytes(bin_path.read_bytes()[:-16])
    meta = read_plane_meta(meta_path)
    with pytest.raises(ValueError, match="metadata implies"):
        load_plane_dump(bin_path, meta)


def test_plane_spectra_legs_schema(tmp_path: Path):
    bin_path, meta_path, _ = _write_fake_dump(tmp_path, nx=32, nz=8, snapshots=2)
    meta = read_plane_meta(meta_path)
    dump = load_plane_dump(bin_path, meta)
    legs = plane_spectra_legs(dump, u_tau=0.01, nu=0.002)
    assert legs["samples"] == 2
    assert len(legs["streamwise"]) == 2
    leg = legs["streamwise"][0]
    assert leg["y_plus"] == pytest.approx((3 - 0.5) * 0.01 / 0.002)
    assert len(leg["k"]) == len(leg["E"]) == 32 // 2
    assert set(leg["fit"]) >= {"slope", "r2", "valid", "n_points"}


# ---------------------------------------------------------------------------
# resolution attestation
# ---------------------------------------------------------------------------
def test_resolution_metrics_grades():
    dns = resolution_metrics(re_tau=180.0, u_tau=0.01, ny=362, nx=1131, nz=566)
    assert dns["dy_plus"] == pytest.approx(1.0)
    assert dns["dns_resolved"] is True
    assert dns["grade"] == "dns"

    quasi = resolution_metrics(re_tau=180.0, u_tau=0.01, ny=182, nx=565, nz=283)
    assert quasi["dy_plus"] == pytest.approx(2.0)
    assert quasi["dns_resolved"] is False
    assert quasi["grade"] == "quasi_dns"

    coarse = resolution_metrics(re_tau=590.0, u_tau=0.0125, ny=122, nx=188, nz=94)
    assert coarse["dy_plus"] == pytest.approx(590.0 / 60.0)
    assert coarse["grade"] == "under_resolved"


def test_dns_cells_required_scales():
    r180 = dns_cells_required(180.0, dy_plus=1.0)
    r590 = dns_cells_required(590.0, dy_plus=1.0)
    assert r180["half_height"] == pytest.approx(180.0)
    assert r590["cells"] > 10 * r180["cells"]  # ~ (590/180)^3
    assert r180["cells"] == r180["nx"] * r180["ny"] * r180["nz"]


# ---------------------------------------------------------------------------
# gate layer (fail-closed discipline)
# ---------------------------------------------------------------------------
def test_spectrum_gate_accepts_inertial_leg():
    planes = _synthetic_plane(128, 8, slope=-5.0 / 6.0)
    dump = {17: {"ux": planes, "uy": planes * 0.0, "uz": planes * 0.0}}
    legs = plane_spectra_legs(dump, u_tau=0.01, nu=0.01)  # y+ = 16.5 .. below band
    gate = spectrum_slope_gate(legs, re_tau=180.0, y_plus_band=(10.0, 0.5))
    assert gate["valid"]
    assert gate["passed"]
    assert gate["slope"] == pytest.approx(-5.0 / 3.0, abs=0.08)


def test_spectrum_gate_rejects_wrong_slope():
    planes = _synthetic_plane(128, 8, slope=-1.0)  # E ~ k^-2
    dump = {17: {"ux": planes, "uy": planes * 0.0, "uz": planes * 0.0}}
    legs = plane_spectra_legs(dump, u_tau=0.01, nu=0.01)
    gate = spectrum_slope_gate(legs, re_tau=180.0, y_plus_band=(10.0, 0.5))
    assert gate["valid"]
    assert gate["passed"] is False


def test_spectrum_gate_withheld_without_legs():
    gate = spectrum_slope_gate(None, re_tau=180.0)
    assert gate["valid"] is False


def test_batch_convergence_spread():
    result = {
        "batch_means": [
            {"samples": 10, "mean_ux": [0.10, 0.12]},
            {"samples": 10, "mean_ux": [0.101, 0.121]},
            {"samples": 10, "mean_ux": [0.1005, 0.1205]},
            {"samples": 10, "mean_ux": [0.1002, 0.1202]},
        ]
    }
    conv = batch_convergence(result)
    assert conv["valid"]
    assert conv["n_batches"] == 4
    assert conv["bulk_velocity_spread_pct"] == pytest.approx(0.9, abs=0.15)


def test_batch_convergence_withheld_on_nan():
    result = {"batch_means": [{"samples": 1, "mean_ux": [float("nan"), 0.1]}]}
    conv = batch_convergence(result)
    assert conv["valid"] is False


def _fake_profile(**overrides):
    profile = {
        "y": np.linspace(0.5, 59.5, 60).tolist(),
        "y_plus": np.linspace(1.5, 178.5, 60).tolist(),
        "u_plus": np.linspace(1.5, 20.0, 60).tolist(),
        "u_rms_plus": [1.0] * 60,
        "v_rms_plus": [0.5] * 60,
        "w_rms_plus": [0.7] * 60,
        "uv_plus": [-0.5] * 60,
        "u_tau": {"momentum_balance": 0.0125, "viscous_sublayer": 0.0124, "log_law": 0.0125},
        "u_tau_deviations_pct": {"momentum_balance": 0.0, "viscous_sublayer": -0.8,
                                 "log_law": 0.4},
        "max_pairwise_u_tau_deviation_pct": 1.2,
        "wall_law": {"ok": True, "kappa": 0.41, "B": 5.2, "rms_residual": 0.1,
                     "window": [30.0, 53.4], "n_points": 5},
        "u_bulk": 0.2,
        "u_bulk_plus": 16.0,
        "cf": 2.0 / 16.0**2,
        "peaks": {
            "u_rms": {"y_plus": 13.0, "value": 2.7},
            "v_rms": {"y_plus": 25.0, "value": 1.0},
            "w_rms": {"y_plus": 20.0, "value": 1.4},
            "minus_uv": {"y_plus": 30.0, "value": 1.0},
        },
        "anisotropy": {"stations": 60, "u_gt_w": 60, "w_gt_v": 60, "ordering_u_w_v": 60},
        "re_tau_effective": 178.12,
        "nu": 0.0042,
        "half_height": 60.0,
    }
    profile.update(overrides)
    return profile


class _FakeMkm:
    re_tau = 178.12

    def peak(self, quantity):
        table = {
            "u_rms": {"y_plus": 13.0, "value": 2.7},
            "v_rms": {"y_plus": 25.0, "value": 1.0},
            "w_rms": {"y_plus": 20.0, "value": 1.4},
            "minus_uv": {"y_plus": 30.0, "value": 1.0},
        }
        return dict(table[quantity], quantity=quantity, y=0.0)

    def friction_coefficient(self):
        return 2.0 / 16.0**2

    def bulk_velocity_plus(self):
        return 16.0


def test_cf_targets_dean_and_mkm_agree_for_consistent_profile():
    cf = cf_targets(_fake_profile(), _FakeMkm())
    assert cf["mkm_relative_pct"] == pytest.approx(0.0, abs=1e-9)
    # C_f = 2/256 = 7.8e-3 at Re_b = 2*178*16 = 5700 -> Dean 0.073*5700^-0.25
    assert cf["cf_dean"] == pytest.approx(0.073 * cf["re_bulk"] ** -0.25)
    assert cf["dean_in_validity_range"] is False  # Re_b below Dean's 1e4 floor


def test_evaluate_dns_gates_withheld_for_missing_case():
    gates = evaluate_dns_gates({"R180_h60": None}, require=("R180_h60",))
    assert gates["study_status"] == "INCOMPLETE"
    assert gates["cases"]["R180_h60"]["status"] == "WITHHELD"
    assert all(g["passed"] is None for g in gates["gates"])


def test_evaluate_dns_gates_diverged_case_is_withheld():
    case = {"name": "R180_h60", "diverged": True, "path": "x.json", "sanitized": True}
    gates = evaluate_dns_gates({"R180_h60": case})
    assert gates["cases"]["R180_h60"]["status"] == "WITHHELD"
    assert "run diverged" in gates["gates"][0]["problems"][0]


def _fake_case(name: str, dy_plus: float = 3.0):
    profile = _fake_profile()
    resolution = resolution_metrics(
        re_tau=178.12, u_tau=0.0125, ny=int(2 * round(178.12 / dy_plus)) + 2,
        nx=188, nz=94,
    )
    return {
        "name": name,
        "diverged": False,
        "path": "x.json",
        "sanitized": False,
        "anchor": 180,
        "result": {"configuration": {"nx": 188, "ny": resolution["ny"], "nz": 94,
                                    "re_tau": 178.12, "u_tau": 0.0125},
                   "derived": {"nu": resolution["nu"], "samples": 100,
                               "statistics_eddy_turnovers": 6.0}},
        "profile": profile,
        "comparison": {
            "available": True,
            "buffer": {"ok": True, "rms_relative_pct": 2.0, "n_points": 10},
            "log": {"ok": True, "rms_relative_pct": 1.0, "n_points": 10},
        },
        "resolution": resolution,
        "spectra": None,
        "cf": cf_targets(profile, _FakeMkm()),
        "stress": {
            "available": True,
            "u_rms": {"value_relative_pct": 1.0, "y_plus_error": 0.0},
            "minus_uv": {"value_relative_pct": -2.0, "y_plus_error": 0.0},
            "v_rms": {"value_relative_pct": 3.0},
            "w_rms": {"value_relative_pct": -1.0},
            "anisotropy": {"stations": 60, "ordering_u_w_v": 60},
        },
        "convergence": {"valid": True, "bulk_velocity_spread_pct": 0.5},
        "momentum_balance": {"valid": True, "relative_rms_pct": 1.0,
                             "rms_residual": 0.01, "max_abs_residual": 0.02,
                             "n_points": 40},
        "eddy_turnovers": 6.0,
        "samples": 100,
    }


def test_evaluate_dns_gates_reports_under_resolved_fail():
    """A coarse grid must FAIL D0 even when every physics gate passes."""
    case = _fake_case("R180_h60", dy_plus=3.0)
    gates = evaluate_dns_gates({"R180_h60": case}, require=("R180_h60",))
    d0 = next(g for g in gates["gates"] if "D0" in g["gate"])
    assert d0["status"] == "fail"
    assert case["resolution"]["dns_resolved"] is False


def test_evaluate_dns_gates_pass_needs_all_gates():
    case = _fake_case("R180_dns", dy_plus=1.0)
    # the fake profile has no spectrum -> D4 withheld -> study withheld, not pass
    gates = evaluate_dns_gates({"R180_dns": case})
    assert gates["cases"]["R180_dns"]["status"] == "WITHHELD"
    # supply a synthetic -5/3 spectrum and the case becomes a full PASS
    planes = _synthetic_plane(128, 8, slope=-5.0 / 6.0)
    dump = {17: {"ux": planes, "uy": planes * 0.0, "uz": planes * 0.0}}
    # u_tau/nu = 5 -> row 17 sits at y+ = 82.5, inside the log-layer band
    case["spectra"] = plane_spectra_legs(dump, u_tau=0.0125, nu=0.0025)
    gates = evaluate_dns_gates({"R180_dns": case})
    assert gates["cases"]["R180_dns"]["status"] == "PASS"
    assert gates["study_status"] == "PASS"


def test_resolution_targets_are_documented():
    assert DNS_TARGETS["dy_plus"] <= 2.0
    assert DNS_TARGETS["dx_plus"] > DNS_TARGETS["dy_plus"]

# ---------------------------------------------------------------------------
# kernel configuration wiring (guards the --reg / spectrum CLI plumbing)
# ---------------------------------------------------------------------------
def test_les_channel_config_forwards_reg_and_spectrum_args():
    from tensorlbm.turbulence.les_channel import LesChannelConfig

    cfg = LesChannelConfig(
        nx=32, ny=18, nz=32, steps=100, warmup=20, sample=10,
        cs=0.0, regularized=0, spec_every=5, spec_rows=(3, 9),
    )
    cfg.validate()
    args = cfg.cli_args("out")
    assert args[args.index("--reg") + 1] == "0"
    assert args[args.index("--spec-every") + 1] == "5"
    assert args[args.index("--spec-rows") + 1] == "3,9"
    assert args[args.index("--cs") + 1] == "0.0"


def test_les_channel_config_rejects_bad_spectrum_rows():
    from tensorlbm.turbulence.les_channel import LesChannelConfig

    cfg = LesChannelConfig(nx=32, ny=18, nz=32, steps=100, warmup=20, sample=10,
                           spec_every=5, spec_rows=(0,))
    with pytest.raises(ValueError, match="outside the fluid rows"):
        cfg.validate()
    cfg = LesChannelConfig(nx=32, ny=18, nz=32, steps=100, warmup=20, sample=10,
                           spec_every=5)
    with pytest.raises(ValueError, match="needs at least one"):
        cfg.validate()
