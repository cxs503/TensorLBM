"""The Zou/He xval verdict must not issue PASS while a declared metric fails.

Anchor: ``XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION``.

``PREREG.md`` declares the steady-state condition ``drift < 1e-6`` next to the
primary ``dp_max < 0.15``.  Round 2 (m18793) reported ``PASS`` with drifts of
1.6e-4 ... 2.6e-3, because the driver only ever looked at ``dp_max``.  These
tests pin the corrected behaviour of ``decide_verdict``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

XVAL = (
    Path(__file__).resolve().parents[1]
    / "xval"
    / ("XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION")
)
if not (XVAL / "run_xval.py").is_file():  # pragma: no cover
    pytest.skip("xval driver not present", allow_module_level=True)

pytest.importorskip("torch")
_spec = importlib.util.spec_from_file_location("run_xval", XVAL / "run_xval.py")
rx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rx)


def test_pass_requires_steady_state() -> None:
    """dp_max inside the band but drift above 1e-6 -> no PASS."""
    v = rx.decide_verdict({"C1": 0.015}, {"C1": 1.6e-4})
    assert v["PASS_dp_max"] is True
    assert v["steady_ok"] is False
    assert v["not_steady"] == ["C1"]
    assert v["PASS"] is False


def test_pass_when_both_gates_met() -> None:
    v = rx.decide_verdict({"C1": 0.015}, {"C1": 1e-9})
    assert v["PASS"] is True
    assert v["STRETCH"] is True


def test_missing_drift_is_not_treated_as_converged() -> None:
    v = rx.decide_verdict({"C1": 0.015}, {})
    assert v["steady_ok"] is False
    assert v["PASS"] is False


def test_metric_failure_still_fails_when_steady() -> None:
    v = rx.decide_verdict({"C1": 0.2}, {"C1": 1e-9})
    assert v["PASS"] is False
    assert v["PASS_dp_max"] is False
