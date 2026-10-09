"""Wake-survey drag estimator tests (synthetic, CPU-only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tensorlbm.drag_survey import (
    blasius,
    ittc_1957,
    plane_drag,
    projected_area,
    survey_dataset,
    survey_file,
    survey_point,
    wetted_area,
    write_summary,
)

RNG = np.random.default_rng(0)


def _wake_plane(n: int = 48, u_inf: float = 0.1, amp: float = 0.01, sigma: float = 4.0):
    """Uniform stream plus a centred Gaussian wake deficit, rho = 1."""
    yy, zz = np.mgrid[:n, :n]
    r2 = (yy - n / 2) ** 2 + (zz - n / 2) ** 2
    deficit = amp * np.exp(-r2 / (2.0 * sigma**2))
    ux = u_inf - deficit
    rho = np.ones_like(ux)
    return ux, rho, deficit


def test_plane_drag_recovers_top_hat_momentum_deficit() -> None:
    n, u_inf, amp = 48, 0.1, 0.01
    yy, zz = np.mgrid[:n, :n]
    inside = (yy - n / 2) ** 2 + (zz - n / 2) ** 2 <= 8.0**2
    ux = np.where(inside, u_inf - amp, u_inf)
    rho = np.ones_like(ux)

    drag, u_ref, rho_ref = plane_drag(ux, rho, ring=6)

    assert u_ref == pytest.approx(u_inf)
    assert rho_ref == pytest.approx(1.0)
    expected = amp * (u_inf - amp) * inside.sum()
    assert drag == pytest.approx(expected, rel=0.02)


def test_plane_drag_ring_reference_beats_fixed_under_far_field_shift() -> None:
    ux, rho, _ = _wake_plane()
    delta = 6.0e-4  # mass-correction drift emulation (~0.6% of u_inf)
    drag0, u0, _ = plane_drag(ux, rho)
    drag1, u1, _ = plane_drag(ux + delta, rho)

    assert u1 > u0  # ring reference tracks the shifted far field
    # residual sensitivity is first-order delta/u_eff (~0.6%) …
    assert abs(drag1 - drag0) / drag0 < 0.01
    # … while a fixed nominal reference would be ~100x worse (and can
    # flip sign at high Re) — the failure mode this estimator exists for.
    fixed0 = float(np.sum(rho * ux * (0.1 - ux)))
    fixed1 = float(np.sum(rho * (ux + delta) * (0.1 - ux - delta)))
    assert abs(fixed1 - fixed0) > 30.0 * abs(drag1 - drag0)


def test_plane_drag_rejects_oversized_ring() -> None:
    ux = np.full((6, 6), 0.1)
    with pytest.raises(ValueError, match="ring"):
        plane_drag(ux, np.ones_like(ux), ring=3)


def test_projected_and_wetted_area_of_a_block() -> None:
    mask = np.zeros((10, 10, 12), dtype=bool)
    mask[3:7, 3:7, 4:8] = True  # 4x4x4 block, strictly interior
    assert projected_area(mask) == 16  # 4x4 columns
    assert wetted_area(mask) == 96  # 6 faces x 4x4


def test_reference_lines_spot_values() -> None:
    assert ittc_1957(1.0e6) == pytest.approx(0.075 / 16.0)
    assert blasius(1.0e6) == pytest.approx(1.328e-3)
    with pytest.raises(ValueError, match="Re"):
        ittc_1957(80.0)


def _write_point(tmp_path, re: float, steps, n: int = 32) -> None:
    h5py = pytest.importorskip("h5py")
    point = tmp_path / "points" / f"re{re:.0f}"
    point.mkdir(parents=True)
    ux0, rho0, _ = _wake_plane(n=n)
    mask = np.zeros((n, n, n), dtype=bool)
    mask[:, :, 10:14] = True
    with h5py.File(point / "fields.h5", "w") as f:
        for step in steps:
            g = f.create_group(f"step_{step:06d}")
            decay = 1.0 - 0.2 * (step / max(steps))
            g.create_dataset("ux", data=np.broadcast_to(ux0 * decay, (n, n, n)))
            g.create_dataset("rho", data=np.broadcast_to(rho0, (n, n, n)))
            g.create_dataset("solid_mask", data=mask)
    (point / "status.json").write_text(json.dumps({"params": {"re": re}}))


def test_survey_file_structure_and_normalisation(tmp_path) -> None:
    pytest.importorskip("h5py")
    _write_point(tmp_path, re=100.0, steps=(10, 20))
    survey = survey_file(
        tmp_path / "points" / "re100" / "fields.h5", planes=(2, 4, 8), reference_offset=4
    )
    assert survey.grid == (32, 32, 32)
    assert survey.s_proj == 32 * 32  # mask spans full y-z for this synthetic
    assert [s.step for s in survey.snapshots] == [10, 20]
    assert all(s.c_d > 0.0 for s in survey.snapshots)
    assert survey.drift_last3 is None  # fewer than 3 snapshots
    assert survey.s_wet > 0


def test_survey_dataset_sorted_and_summary_written(tmp_path) -> None:
    pytest.importorskip("h5py")
    _write_point(tmp_path, re=200.0, steps=(10, 20, 30))
    _write_point(tmp_path, re=100.0, steps=(10, 20, 30))
    surveys = survey_dataset(tmp_path, planes=(2, 4, 8), reference_offset=4)
    assert [s.re for s in surveys] == [100.0, 200.0]
    path = write_summary(tmp_path, surveys)
    payload = json.loads(path.read_text())
    assert payload["method"].startswith("wake momentum deficit")
    assert {p["re"] for p in payload["points"]} == {100.0, 200.0}
    first = payload["points"][0]
    assert {"c_d_final", "cf_equiv_final", "s_wet", "drift_last3"} <= set(first)
    assert len(first["snapshots"]) == 3


def test_survey_point_reads_re_from_status(tmp_path) -> None:
    pytest.importorskip("h5py")
    _write_point(tmp_path, re=123.0, steps=(5,))
    survey = survey_point(tmp_path / "points" / "re123", planes=(2, 4, 8), reference_offset=4)
    assert survey.re == 123.0
