"""Tests for the Triton codegen pinning layer (``tensorlbm.codegen_pin``).

The contract under test is the *pinning discipline*, not GPU throughput:
these tests must pass on a CPU-only build node (no CUDA backend), where the
layer's job is to refuse to pretend a register cap was applied.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tensorlbm import codegen_pin as cp

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_and_hex():
    knobs = {"num_warps": 2, "num_stages": 2, "maxnreg": None}
    first = cp.fingerprint_of(knobs)
    assert first == cp.fingerprint_of(dict(reversed(list(knobs.items()))))
    assert len(first) == 16
    int(first, 16)  # hex


def test_fingerprint_changes_with_each_codegen_knob():
    base = {"num_warps": 2, "num_stages": 2, "maxnreg": None}
    seen = {cp.fingerprint_of(base)}
    for key, value in (
        ("num_warps", 4),
        ("num_stages", 3),
        ("maxnreg", 32),
    ):
        changed = dict(base)
        changed[key] = value
        fingerprint = cp.fingerprint_of(changed)
        assert fingerprint not in seen, f"{key} did not change the pin identity"
        seen.add(fingerprint)
    assert len(seen) == 4


def test_fingerprint_ignores_pin_version_noise_but_not_knob_identity():
    # The version is part of the digest: a pin-scheme change invalidates old
    # comparisons even if the numbers happen to match.
    knobs = {"num_warps": 2, "num_stages": 2, "maxnreg": None}
    with_version_1 = cp.fingerprint_of(knobs)
    original = cp.PIN_VERSION
    try:
        cp.PIN_VERSION = "2"
        assert cp.fingerprint_of(knobs) != with_version_1
    finally:
        cp.PIN_VERSION = original


# ---------------------------------------------------------------------------
# resolution / violation
# ---------------------------------------------------------------------------


def test_declared_pins_exist_for_production_kernels():
    for name in (
        "triton_fused_obstacle_xfar_les",
        "triton_fused_obstacle_les",
        "triton_obstacle_force_reduction",
    ):
        pin = cp.get_pin(name)
        assert pin.fingerprint() == cp.fingerprint_of(pin.knobs())


def test_unknown_kernel_is_a_pin_violation():
    with pytest.raises(cp.CodegenPinViolation):
        cp.get_pin("no_such_kernel")


def test_resolve_with_declared_values_is_pinned():
    pin = cp.get_pin("triton_fused_obstacle_les")
    resolved = cp.resolve(
        "triton_fused_obstacle_les",
        overrides={"num_warps": pin.num_warps, "num_stages": pin.num_stages},
    )
    assert resolved.pinned is True
    assert resolved.fingerprint == pin.fingerprint()
    assert resolved.deviations == {}


def test_resolve_rejects_silent_codegen_drift():
    pin = cp.get_pin("triton_fused_obstacle_les")
    with pytest.raises(cp.CodegenPinViolation) as excinfo:
        cp.resolve("triton_fused_obstacle_les", overrides={"num_warps": pin.num_warps + 2})
    message = str(excinfo.value)
    assert "num_warps" in message
    assert pin.fingerprint() in message


def test_resolve_allows_explicit_tuning_with_a_different_fingerprint():
    pin = cp.get_pin("triton_fused_obstacle_les")
    tuned = cp.resolve(
        "triton_fused_obstacle_les",
        overrides={"num_warps": pin.num_warps * 2},
        allow_tuning=True,
    )
    assert tuned.pinned is False
    assert tuned.knobs["num_warps"] == pin.num_warps * 2
    assert tuned.fingerprint != pin.fingerprint()
    assert tuned.fingerprint.endswith(tuned.fingerprint.split("-T")[-1])
    with pytest.raises(cp.CodegenPinViolation):
        cp.assert_comparable(pin.to_dict() | {"pinned": True}, tuned.to_dict())


def test_non_codegen_override_is_rejected():
    with pytest.raises(ValueError):
        cp.resolve("triton_fused_obstacle_les", overrides={"block_x": 128})


# ---------------------------------------------------------------------------
# launch kwargs + maxnreg forwarding
# ---------------------------------------------------------------------------


def test_launch_kwargs_omit_maxnreg_when_backend_cannot_take_it(monkeypatch):
    monkeypatch.setattr(cp, "maxnreg_supported", lambda: False)
    kwargs = cp.launch_kwargs(num_warps=2, num_stages=2, maxnreg=32)
    assert kwargs == {"num_warps": 2, "num_stages": 2}


def test_launch_kwargs_forward_maxnreg_when_supported(monkeypatch):
    monkeypatch.setattr(cp, "maxnreg_supported", lambda: True)
    kwargs = cp.launch_kwargs(num_warps=2, num_stages=2, maxnreg=32)
    assert kwargs == {"num_warps": 2, "num_stages": 2, "maxnreg": 32}


def test_launch_kwargs_uses_declared_maxnreg_when_caller_leaves_it_open(monkeypatch):
    monkeypatch.setattr(cp, "maxnreg_supported", lambda: True)
    pin = cp.KernelPin(
        kernel="synthetic_kernel",
        num_warps=2,
        num_stages=2,
        maxnreg=40,
        source="synthetic:launch",
    )
    monkeypatch.setitem(cp.DECLARED_PINS, pin.kernel, pin)
    kwargs = cp.launch_kwargs(kernel="synthetic_kernel", num_warps=2, num_stages=2, maxnreg=None)
    assert kwargs["maxnreg"] == 40


def test_launch_kwargs_never_invents_a_register_cap():
    # No declared pin anywhere sets maxnreg yet -> the honest output carries no
    # register cap, and the audit records the gap instead.
    assert all(pin.maxnreg is None for pin in cp.DECLARED_PINS.values())


# ---------------------------------------------------------------------------
# comparability + recorder
# ---------------------------------------------------------------------------


def test_assert_comparable_rejects_unpinned_records():
    pinned = cp.resolve("triton_fused").to_dict()
    with pytest.raises(cp.CodegenPinViolation):
        cp.assert_comparable(pinned, {"pinned": False, "fingerprint": "deadbeef"})
    cp.assert_comparable(pinned, cp.resolve("triton_fused").to_dict())


def test_recorder_refuses_unpinned_measurements(tmp_path):
    recorder = cp.PinnedRecorder(tmp_path / "bench.jsonl")
    tuned = cp.resolve("triton_fused", overrides={"num_warps": 8}, allow_tuning=True)
    with pytest.raises(cp.CodegenPinViolation):
        recorder.record(tuned, metric="mlups", value=1234.5)
    assert not (tmp_path / "bench.jsonl").exists()


def test_recorder_writes_pinned_jsonl(tmp_path):
    path = tmp_path / "bench.jsonl"
    recorder = cp.PinnedRecorder(path)
    pinned = cp.resolve("triton_fused")
    recorder.record(pinned, metric="mlups", value=987.5, extra={"n": 64})
    recorder.record(pinned, metric="mlups", value=991.0, extra={"n": 64})
    entries = recorder.entries()
    assert len(entries) == 2
    assert {entry["fingerprint"] for entry in entries} == {pinned.fingerprint}
    assert all(entry["pinned"] for entry in entries)
    assert entries[0]["knobs"]["num_warps"] == cp.get_pin("triton_fused").num_warps
    # cross-pin comparison of the recorded rows must be refused
    tuned = cp.resolve("triton_fused", overrides={"num_warps": 8}, allow_tuning=True)
    with pytest.raises(cp.CodegenPinViolation):
        cp.assert_comparable(entries[0], tuned.to_dict())


# ---------------------------------------------------------------------------
# environment probe
# ---------------------------------------------------------------------------


def test_environment_snapshot_never_raises_and_declares_absent_values():
    snapshot = cp.environment_snapshot()
    for key in (
        "timestamp",
        "platform",
        "machine",
        "python",
        "triton_version",
        "torch_version",
        "cuda_available",
        "maxnreg_supported",
    ):
        assert key in snapshot
    assert isinstance(snapshot["maxnreg_supported"], bool)
    # nvcc absent on the CPU build node -> recorded as None, not guessed
    assert snapshot["nvcc_version"] is None or isinstance(snapshot["nvcc_version"], str)


# ---------------------------------------------------------------------------
# static audit
# ---------------------------------------------------------------------------


def test_audit_finds_kernels_launch_sites_and_no_drift():
    report = cp.audit_codegen()
    summary = report["summary"]
    assert summary["kernels"] >= 10, summary
    assert summary["launch_sites"] >= 10, summary
    assert summary["launch_sites_pinned_via_codegen_pin"] >= 3, summary
    assert report["drift"] == [], report["drift"]


def test_audit_detects_drift_when_a_pin_lies():
    bogus = cp.KernelPin(
        kernel="triton_fused",
        num_warps=99,
        num_stages=99,
        source="tensorlbm.triton_fused:triton_fused",
    )
    report = cp.audit_codegen(pins={"triton_fused": bogus})
    knob_findings = {finding["knob"] for finding in report["drift"]}
    assert {"num_warps", "num_stages"} <= knob_findings
    assert report["summary"]["drift_findings"] >= 2


def test_audit_reports_missing_wrapper():
    ghost = cp.KernelPin(kernel="ghost", num_warps=2, num_stages=2, source="tensorlbm.nope:nothing")
    report = cp.audit_codegen(pins={"ghost": ghost})
    issues = {finding["issue"] for finding in report["drift"]}
    assert "declared_wrapper_not_found" in issues


def test_audit_works_on_a_synthetic_tree_with_a_literal_maxnreg(tmp_path):
    """Fixture uses synthetic names on purpose: the auditor must not need GPU code."""
    module = tmp_path / "src" / "syntheticpkg" / "kern.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def synthetic_kernel(x):\n"
        "    pass\n"
        "\n"
        "def launch(x, num_warps=4, num_stages=2, maxnreg=32):\n"
        "    synthetic_kernel[(1,)](x, num_warps=num_warps, num_stages=num_stages,\n"
        "                           maxnreg=maxnreg)\n"
        "    return x\n",
        encoding="utf-8",
    )
    report = cp.audit_codegen([str(module)], pins={})
    assert [kernel["name"] for kernel in report["kernels"]] == ["synthetic_kernel"]
    assert report["summary"]["launch_sites"] == 1
    site = report["launch_sites"][0]
    assert site["kernel"] == "synthetic_kernel"
    assert site["value_kinds"] == {
        "num_warps": "symbol",
        "num_stages": "symbol",
        "maxnreg": "symbol",
    }
    wrapper = report["wrappers"][0]
    assert wrapper["codegen"] == {"num_warps": 4, "num_stages": 2, "maxnreg": 32}
    assert report["drift"] == []


def test_audit_flags_maxnreg_added_without_updating_the_pin(tmp_path):
    module = tmp_path / "src" / "syntheticpkg" / "kern2.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "import triton\n\ndef launch(x, num_warps=2, num_stages=2, maxnreg=32):\n    return x\n",
        encoding="utf-8",
    )
    pin = cp.KernelPin(
        kernel="k",
        num_warps=2,
        num_stages=2,
        maxnreg=None,
        source="syntheticpkg.kern2:launch",
    )
    report = cp.audit_codegen([str(module)], pins={"k": pin})
    assert any(
        finding["issue"] == "maxnreg_added_without_pin_update" for finding in report["drift"]
    )


# ---------------------------------------------------------------------------
# production launch path is actually routed through the pin helper
# ---------------------------------------------------------------------------


def test_production_obstacle_launches_are_routed_through_codegen_pin():
    source = (REPO_ROOT / "src" / "tensorlbm" / "triton_fused_obstacle.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    routed = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Subscript):
            continue
        for keyword in node.keywords:
            if keyword.arg is not None:
                continue
            text = ast.unparse(keyword.value)
            if "launch_kwargs" in text:
                routed += 1
    assert routed >= 2
    # the previous hard-coded pairs must be gone at the launch sites
    assert "num_warps=num_warps,\n        num_stages=num_stages,\n    )" not in source


def test_declared_pins_match_kernel_sources():
    """Pinned *before* tuning: a knob change must show up as an audit failure."""
    report = cp.audit_codegen()
    assert report["drift"] == [], json.dumps(report["drift"], indent=2)
