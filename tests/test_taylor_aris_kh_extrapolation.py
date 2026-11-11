"""Finite-``kH`` debt of the periodic-shear Taylor-Aris meter.

Guards the finding of ``benchmarks/pending/taylor_aris_kh_extrapolation``
(doc: ``docs/taylor_aris_kh_extrapolation_20261111.md``): the shipped single
aspect-ratio reading (``L/H = 8``, ``kH = 0.785``) is *not* the kernel error --
it is the accidental cancellation of a positive finite-wave-number bias and a
negative transverse-discretisation bias.  Three of the four gates below are
cheap re-measurements (tens of seconds); the transverse-order gate reads the
archived scan so the suite stays fast.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tensorlbm.taylor_dispersion import (  # noqa: E402
    ShearDispersionConfig,
    run_periodic_shear_dispersion,
)

ARTIFACTS = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "pending"
    / "taylor_aris_kh_extrapolation"
    / "artifacts"
    / "results.json"
)

H = 16
PE = 8.0
D = 0.1


def _err_pct(nx: int, n_steps: int = 6000) -> float:
    torch.set_num_threads(1)
    cfg = ShearDispersionConfig(ny=H, nx=nx, D=D, pe=PE, n_steps=n_steps)
    res = run_periodic_shear_dispersion(cfg)
    return 100.0 * res.rel_error


def test_shipped_aspect_ratio_understates_the_kernel_bias():
    """``L/H=8`` nearly nulls the error; ``L/H=32`` exposes the ~0.5% bias."""
    err_ship = _err_pct(8 * H)
    err_fine = _err_pct(32 * H, n_steps=4000)
    assert abs(err_ship) < 0.2, f"shipped reading unexpectedly large: {err_ship}"
    assert err_fine < -0.35, f"kH bias not exposed at L/H=32: {err_fine}"
    assert abs(err_ship) < abs(err_fine), (
        "the single-aspect-ratio reading must be a cancellation, not the truth"
    )


def test_kh_extrapolation_recovers_the_kh_zero_limit():
    """Linear-in-``(kH)^2`` two-point extrapolation -> kH->0 limit (-0.525%)."""
    err_ship = _err_pct(8 * H)
    err_fine = _err_pct(32 * H, n_steps=4000)
    k_h = {8: 2.0 * math.pi / 8.0, 32: 2.0 * math.pi / 32.0}
    x1, y1 = k_h[8] ** 2, err_ship
    x2, y2 = k_h[32] ** 2, err_fine
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    assert -0.75 < intercept < -0.35, f"kH->0 limit drifted: {intercept:.3f}%"


def test_scale_separation_guard_pins_the_kh_family():
    """The meter only has a kH->0 limit if ``H < L``; the guard must stay."""
    with pytest.raises(ValueError):
        ShearDispersionConfig(ny=32, nx=32, D=D, pe=PE)


def test_archived_transverse_bias_is_second_order_in_h():
    """Archived scan: residual at kH->0 is pure transverse error, O(1/H^2)."""
    if not ARTIFACTS.exists():  # pragma: no cover - scan not shipped
        pytest.skip("archived scan missing")
    data = json.loads(ARTIFACTS.read_text())
    fit = data["transverse_refinement_pe8_LH32"]["fit"]
    assert fit["r2"] > 0.99, f"1/H^2 law not established: R2={fit['r2']}"
    assert fit["slope"] < 0, "transverse error must shrink as H grows"
    assert abs(fit["rel_error_extrapolated_pct"]) < 0.2, (
        f"H->inf residual should vanish, got {fit['rel_error_extrapolated_pct']:.3f}%"
    )
    # the kH->0 limits must all sit inside the 3% gate
    for pe, blk in data["kH_extrapolation"].items():
        assert abs(blk["rel_error_kH0_pct"]) < 3.0, f"Pe={pe} kH->0 outside gate"
    # cancellation must grow with Pe (the whole point of the ledger entry)
    cancel = [data["kH_extrapolation"][k]["cancellation_pp"] for k in sorted(
        data["kH_extrapolation"], key=float)]
    assert cancel == sorted(cancel), f"cancellation not monotone in Pe: {cancel}"
