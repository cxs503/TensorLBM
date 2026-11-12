"""Regression: TRT (Lambda=3/16) wall placement vs BGK in body-force plane Poiseuille.

Pins the verified numbers of docs/TRT_WALL_ACCURACY_20261112.md
(benchmarks/pending/trt_wall_accuracy/result.json).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "pending" / "trt_wall_accuracy"))
pytest.importorskip("torch")
import verify_trt_wall as V  # noqa: E402


@pytest.fixture(scope="module")
def pair():
    return V.run(8, 0.80, "bgk", steps=1500), V.run(8, 0.80, "trt", steps=1500)


def test_both_schemes_reproduce_the_analytic_parabola(pair):
    bgk, trt = pair
    assert bgk["profile_L2_rel_pct"] < 3.0
    assert trt["profile_L2_rel_pct"] < 3.0


def test_trt_wall_displacement_is_smaller_than_bgk(pair):
    bgk, trt = pair
    assert abs(trt["delta_eff"]) < abs(bgk["delta_eff"]) < 0.1


def test_trt_matches_the_magic_lambda_signature_at_tau_0933():
    """Lambda_BGK = (tau-1/2)^2 equals 3/16 at tau = 1/2+sqrt(3)/4: BGK == TRT there."""
    a = V.run(8, 0.5 + 3 ** 0.5 / 4, "bgk", steps=1500)
    b = V.run(8, 0.5 + 3 ** 0.5 / 4, "trt", steps=1500)
    assert abs(a["delta_eff"] - b["delta_eff"]) < 2e-4
