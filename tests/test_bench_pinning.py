"""Tests for :mod:`tensorlbm.bench_pinning` (resource pin before tuning).

The practice under test comes from the OSS cross-validation
``XVAL-LBM-MAXREGCOUNT-PINNED-BEFORE-TUNING``: pin the compute-resource budget
before tuning a kernel, record the pin with every result, and recompute a
foreign register-cap value instead of copying it.
"""

from __future__ import annotations

import json

import pytest
import torch

from tensorlbm.bench_pinning import (
    PinnedResourceViolation,
    ResourcePin,
    assert_pinned,
    maxnreg_launch_kwargs,
    pin_audit,
    pinned_resources,
    register_budget_words,
    register_cap_verdict,
)


def test_pin_rejects_invalid_fields():
    with pytest.raises(ValueError):
        ResourcePin(label="x", threads=0)
    with pytest.raises(ValueError):
        ResourcePin(label="x", threads=4, dtype="float16")
    with pytest.raises(ValueError):
        ResourcePin(label="x", threads=4, maxnreg=8)  # below the CUDA minimum
    with pytest.raises(ValueError):
        ResourcePin(label="", threads=4)
    with pytest.raises(ValueError):
        ResourcePin(label="x", threads=4, timed_steps=0)


def test_pin_sha_is_content_addressed():
    a = ResourcePin(label="bench", threads=8)
    b = ResourcePin(label="bench", threads=8)
    c = ResourcePin(label="bench", threads=4)
    assert a.sha256() == b.sha256()
    assert a.sha256() != c.sha256()
    assert len(a.sha256()) == 64


def test_pinned_resources_applies_and_restores():
    before = torch.get_num_threads()
    pin = ResourcePin(label="t", threads=3)
    with pinned_resources(pin) as audit:
        assert torch.get_num_threads() == 3
        assert audit["pin_sha256"] == pin.sha256()
        assert audit["observed"]["torch_num_threads"] == 3
        assert audit["observed"]["maxnreg_pinned"] is False
        assert_pinned(pin)
    assert torch.get_num_threads() == before


def test_assert_pinned_detects_drift():
    pin = ResourcePin(label="t", threads=2)
    with pinned_resources(pin):
        torch.set_num_threads(5)  # a measurement-side drift, the thing we forbid
        with pytest.raises(PinnedResourceViolation):
            assert_pinned(pin)
    torch.set_num_threads(1)


def test_audit_carries_extra_numbers():
    pin = ResourcePin(label="t", threads=2)
    rec = pin_audit(pin, step_time_s=0.5, collision_impl="bgk")
    assert rec["step_time_s"] == 0.5
    assert rec["collision_impl"] == "bgk"
    assert json.loads(json.dumps(rec))["pin"]["threads"] == 2


def test_maxnreg_launch_kwargs_is_the_gpu_spelling_of_the_pin():
    assert maxnreg_launch_kwargs(ResourcePin(label="t", threads=1)) == {}
    assert maxnreg_launch_kwargs(ResourcePin(label="t", threads=1, maxnreg=64)) == {"maxnreg": 64}


def test_register_budget_words():
    assert register_budget_words(19, "fp32") == 19
    assert register_budget_words(19, "fp64") == 38
    assert register_budget_words(0, "fp64") == 0
    with pytest.raises(ValueError):
        register_budget_words(-1)
    with pytest.raises(ValueError):
        register_budget_words(4, "fp128")


def test_register_cap_verdict_recomputes_a_foreign_cap():
    # D2Q9 fused BGK (9 f + 9 feq + 9 cu + 5 scalars = 32 live values) in fp32
    # fits under the OSS cap of 64 words; the same kernel in fp64 does not.
    assert register_cap_verdict(32, "fp32", 64)["fits"] is True
    v64 = register_cap_verdict(32, "fp64", 64)
    assert v64["fits"] is False
    assert v64["required_words"] == 64
    assert v64["overflow_words"] == 16  # 64 + 16 reserve - 64
    # D3Q19 fused (19 f + 19 feq + 19 cu + 5) overflows even in fp32.
    assert register_cap_verdict(62, "fp32", 64)["fits"] is False
    assert register_cap_verdict(62, "fp64", 64)["required_with_reserve"] == 140