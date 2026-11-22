"""Design-of-experiments tests for the RT Mach/resolution de-confounding probe.

The probe's claim is only meaningful if its arms really do hold the *continuum*
problem fixed while moving one axis.  These tests assert those invariants on the
matrix itself (not on any simulation result), and they test the analysis stage
against a synthetic dataset with a known answer, so a broken fit cannot pass by
merely executing.

Run: PYTHONPATH=src python -m pytest -q tests/test_case_pool_rt_mach_resolution.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.case_pool.rayleigh_taylor_atwood05 import (
    probe_mach_resolution as probe,  # noqa: E402
)


def _by_axis(axis: str) -> list:
    return [s for s in probe.build_matrix() if s.axis == axis]


def test_matrix_is_unique_and_every_arm_is_valid() -> None:
    specs = probe.build_matrix()
    names = [s.name for s in specs]
    assert len(names) == len(set(names)), "duplicate arm names"
    for s in specs:
        cfg = s.cfg()                     # raises on an invalid configuration
        cfg.validate()
        assert s.tau > 0.5 and s.tau_phi > 0.5
        assert s.n_steps() > 0 and s.sample_every >= 1
        g = s.groups()
        for key in ("viscous_group_G", "capillary_group_Pi", "kdelta", "ma_scale"):
            assert math.isfinite(g[key]) and g[key] > 0.0


def test_mach_axis_holds_every_group_but_mach() -> None:
    """M axis: lam fixed, G and Pi invariant, Ma must actually span an axis."""
    arms = _by_axis("M")
    assert len(arms) >= 3
    ref = arms[0].groups()
    for s in arms:
        g = s.groups()
        assert g["kdelta"] == pytest.approx(ref["kdelta"], rel=1e-12)
        assert g["viscous_group_G"] == pytest.approx(ref["viscous_group_G"], rel=1e-12)
        assert g["capillary_group_Pi"] == pytest.approx(ref["capillary_group_Pi"], rel=1e-12)
    ma = [s.groups()["ma_scale"] for s in arms]
    assert max(ma) / min(ma) >= 3.0, "Ma axis too narrow to resolve a quadratic trend"
    # monotone in g, and the reference arm (g = 2e-4) reproduces run_case's case
    assert ma == sorted(ma)
    ref_arm = next(s for s in arms if s.g == 2.0e-4)
    assert ref_arm.nu == pytest.approx(0.005, rel=1e-12)


def test_resolution_axis_holds_mach_viscosity_and_capillarity() -> None:
    """R axis: Ma, G, Pi invariant; the ``kd`` variant also holds k*delta."""
    arms = _by_axis("R")
    assert len(arms) >= 4
    ref = next(s for s in arms if s.name == "R_l096_kd").groups()
    for s in arms:
        g = s.groups()
        assert g["ma_scale"] == pytest.approx(ref["ma_scale"], rel=1e-6), s.name
        assert g["viscous_group_G"] == pytest.approx(ref["viscous_group_G"], rel=1e-6), s.name
        if s.name.endswith("kd"):
            assert g["capillary_group_Pi"] == pytest.approx(
                ref["capillary_group_Pi"], rel=1e-6), s.name
            assert g["kdelta"] == pytest.approx(ref["kdelta"], rel=1e-6), s.name
    lams = sorted({s.lam for s in arms})
    assert lams == [96, 128], "resolution axis must span more than one wavelength"
    # the Ma held on the R axis equals the reference case's Ma scale
    ma_ref = next(s for s in _by_axis("M") if s.g == 2.0e-4).groups()["ma_scale"]
    assert ref["ma_scale"] == pytest.approx(ma_ref, rel=1e-6)


def test_archived_ladder_design_is_recorded_as_confounded() -> None:
    """L axis must be *less* controlled than the R axis (that is its purpose)."""
    legs = _by_axis("L")
    assert len(legs) >= 2
    gs = [s.groups()["viscous_group_G"] for s in legs]
    pis = [s.groups()["capillary_group_Pi"] for s in legs]
    kds = [s.groups()["kdelta"] for s in legs]
    mas = [s.groups()["ma_scale"] for s in legs]
    # at least three groups move together along the archived design
    assert max(gs) / min(gs) > 1.5
    assert max(pis) / min(pis) > 1.5
    assert max(kds) / min(kds) > 1.2
    assert max(mas) / min(mas) > 1.2


def _synthetic(m_axis_pts, r_axis_pts, legacy_pts=()):
    """Build case dicts in the shape ``analyse`` consumes."""
    def mk(name, lam, g, nu, ma, err, axis, ac_width=2.0):
        spec = probe.ProbeSpec(name=name, lam=lam, g=g, nu=nu, ac_width=ac_width, axis=axis)
        return {"name": name, "lam": lam, "g": g, "nu": nu, "axis": axis,
                "ac_width": ac_width, "ma_peak": ma, "sigma_sim": 1.0,
                "sigma_viscous": 1.0, "err_vs_viscous_pct": err,
                "err_vs_inviscid_pct": err, "err_flow_uy_vs_viscous_pct": err,
                "groups": spec.groups()}
    cases = []
    for axis, pts in (("M", m_axis_pts), ("R", r_axis_pts), ("L", legacy_pts)):
        for name, lam, g, nu, ma, err, *extra in pts:
            cases.append(mk(name, lam, g, nu, ma, err, axis,
                            extra[0] if extra else 2.0))
    return cases


def test_analysis_recovers_a_known_quadratic_mach_trend() -> None:
    """Feed ``err = -2 - 400 Ma^2``: the fit must return e0 = -2, c = -400."""
    pts = [("m1", 64, 5e-5, 0.0025, 0.03, -2.0 - 400 * 0.03 ** 2),
           ("m2", 64, 2e-4, 0.005, 0.25, -2.0 - 400 * 0.25 ** 2),
           ("m3", 64, 8e-4, 0.01, 0.70, -2.0 - 400 * 0.70 ** 2)]
    res = probe.analyse(_synthetic(pts, []), legacy=[])
    fit = res["h1_mach_fit"]
    assert fit["intercept_e0_pct"] == pytest.approx(-2.0, abs=1e-9)
    assert fit["slope_c"] == pytest.approx(-400.0, rel=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)
    assert fit["h1_supported"] is True              # |c| Ma^2 > 50 % of the deficit


def test_analysis_flags_a_mach_independent_bias() -> None:
    """A flat-in-Ma residual must NOT be attributed to compressibility."""
    pts = [("m1", 64, 5e-5, 0.0025, 0.03, -10.0),
           ("m2", 64, 2e-4, 0.005, 0.25, -10.1),
           ("m3", 64, 8e-4, 0.01, 0.70, -9.9)]
    res = probe.analyse(_synthetic(pts, []), legacy=[])
    fit = res["h1_mach_fit"]
    assert abs(fit["slope_c"]) < 1.0
    assert fit["h1_supported"] is False
    assert fit["intercept_e0_pct"] == pytest.approx(-10.0, abs=0.1)


def test_analysis_reports_resolution_spread_per_variant() -> None:
    r = [("R_l096_kd", 96, 1.333e-4, 0.0075, 0.25, -3.0),
         ("R_l128_kd", 128, 1.0e-4, 0.010, 0.25, -1.0),
         ("R_l096_kd0", 96, 1.333e-4, 0.0075, 0.25, -5.0, 2.0),
         ("R_l128_kd0", 128, 1.0e-4, 0.010, 0.25, -9.0, 2.0)]
    res = probe.analyse(_synthetic([], r), legacy=[])
    h2 = res["h2_resolution"]
    assert h2["kd"]["spread_pct_points"] == pytest.approx(2.0)
    assert h2["kd0"]["spread_pct_points"] == pytest.approx(4.0)
    assert h2["kd"]["lams"] == [96, 128]
