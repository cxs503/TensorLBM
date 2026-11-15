"""Round 4: the pointwise gate's dtype noise floor and detection limit.

Round 3 froze the D3Q19 pointwise bundle and gated the in-repo kernels at
``eps_inf <= 1e-10`` in fp64.  Two claims were asserted without measurement:
(a) that the gate is insensitive to reduced precision, and (b) that the gate
would catch "any" regression.  This test reads
``xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/tolerance/tolerance_policy.json``
(produced by ``tolerance/run_tolerance_audit.py``) and enforces the parts that
must stay true, plus one *recomputed* point so the policy cannot drift away
from the data it claims to describe.

Guard philosophy: the shipped fp64 threshold (1e-10) must stay far above the
fp64 floor and far below the fp32 floor -- the second half is what forbids
copying 1e-10 onto an fp32 execution path, which would be a false failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "xval" / "XVAL-LBM-POINTWISE-REFERENCE-REGRESSION"
TOL = ANCHOR / "tolerance"
BUNDLE = ANCHOR / "d3q19" / "refdata" / "pointwise_d3q19_v1.npz"
POLICY_F = TOL / "tolerance_policy.json"

pytestmark = pytest.mark.skipif(not (BUNDLE.exists() and POLICY_F.exists()),
                               reason="frozen D3Q19 bundle / tolerance policy missing")

sys.path.insert(0, str(TOL))
import precision_emulation as pe  # noqa: E402

SHIPPED_GATE = 1e-10  # tests/test_xval_pointwise_d3q19_refdata.py
TAU = 0.7
STEPS = 30


@pytest.fixture(scope="module")
def policy():
    return json.loads(POLICY_F.read_text())


@pytest.fixture(scope="module")
def refs():
    import numpy as np  # noqa: PLC0415

    with np.load(BUNDLE) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# 1. the shipped gate is admissible in the precision it is used in
# ---------------------------------------------------------------------------
def test_fp64_floor_is_far_below_shipped_gate(policy, refs):
    floor = policy["fp64"]["noise_floor_eps_inf"]
    assert 0.0 < floor < 1e-12, floor
    assert SHIPPED_GATE >= 1000.0 * floor, (SHIPPED_GATE, floor)
    assert policy["fp64"]["admissible_gate_eps_inf"] == SHIPPED_GATE


def test_fp64_reference_step_reproduces_frozen_bundle(refs):
    """Control: the re-derived fp64 step is the round-3 reference (not a lookalike)."""
    traj = pe.run(refs["f0"], TAU, STEPS, "fp64", None)
    e1 = pe.core.compare(traj[0], refs["ref_t1"])["eps_inf"]
    e30 = pe.core.compare(traj[-1], refs["ref_t30"])["eps_inf"]
    assert e1 <= 1e-13, e1
    assert e30 <= 1e-13, e30


# ---------------------------------------------------------------------------
# 2. the fp64 gate is NOT admissible on a reduced-precision path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode,lo,hi", [
    ("fp32", 1e-7, 1e-4),            # true fp32 arithmetic
    ("fp64+fp32_store", 1e-9, 1e-5),  # fp64 arithmetic, fp32 memory
    ("fp64+bf16_store", 1e-3, 1e-1),
])
def test_reduced_precision_floor_brackets(policy, mode, lo, hi):
    floor = policy[mode]["noise_floor_eps_inf"]
    assert lo <= floor <= hi, (mode, floor)


def test_shipped_gate_would_false_alarm_on_fp32(policy):
    """The 1e-10 gate is fp64-only: an fp32 path sits ~4 decades above it."""
    floor32 = policy["fp32"]["noise_floor_eps_inf"]
    assert floor32 > SHIPPED_GATE, (floor32, SHIPPED_GATE)
    assert floor32 / SHIPPED_GATE > 1e3


def test_reduced_precision_is_measured_inside_the_loop(policy, refs):
    """Kills the round-1 tautology (fp32 cast of two already-equal fp64 arrays).

    The old row claimed ``eps_scaled_inf_fp32_rounding == 0.0``; applying the
    precision per step must give a strictly positive, bracketable deviation.
    """
    traj = pe.run(refs["f0"], TAU, STEPS, "fp32", None)
    eps = pe.core.compare(traj[-1], refs["ref_t30"])["eps_inf"]
    assert eps > 1e-9, eps                      # not zero, not cast-only
    assert eps < 1e-3, eps
    fp32_rows = json.loads((ANCHOR / "summary.json").read_text()).get("fp32_sensitivity", [])
    if fp32_rows:  # superseded round-1 rows, kept as a documented correction
        assert all(r["eps_scaled_inf_fp32_rounding"] == 0.0 for r in fp32_rows)


# ---------------------------------------------------------------------------
# 3. the policy is consistent with its own data, and enforced conservatively
# ---------------------------------------------------------------------------
def test_admissible_gates_clear_their_floor(policy):
    for mode, p in policy.items():
        if "admissible_gate_eps_inf" not in p:
            continue
        gate = p["admissible_gate_eps_inf"]
        floor = p["noise_floor_eps_inf"]
        assert gate >= 10.0 * floor, (mode, gate, floor)


def test_detection_floor_ordering(policy):
    """fp64 resolves far smaller tau drifts than fp32 -- the honest ranking."""
    fp64_d = policy["fp64"]["detectable_tau_rel"]
    fp32_d = policy["fp32"]["detectable_tau_rel_margin10"]
    assert fp64_d is not None and fp64_d <= 1e-7, fp64_d
    assert fp32_d is not None and fp32_d >= 1e-3, fp32_d
    assert fp32_d / fp64_d >= 1e3, (fp32_d, fp64_d)


def test_sensitivity_is_monotone_and_signal_grows_with_regression():
    sens = json.loads((TOL / "summary.json").read_text())["sensitivity"]
    for mode, s in sens.items():
        rows = sorted(s["rows"], key=lambda r: r["delta_rel"])
        small = next(r["eps_inf"] for r in rows if r["delta_rel"] == 1e-4)
        big = next(r["eps_inf"] for r in rows if r["delta_rel"] == 1e-2)
        assert big > small, (mode, small, big)
        assert s["snr_at_1pct_tau"] > 10.0, (mode, s["snr_at_1pct_tau"])


def test_policy_matches_recomputation(policy, refs):
    """One policy point recomputed from scratch: the JSON must describe reality."""
    traj = pe.run(refs["f0"], TAU, STEPS, "fp32", None)
    eps = pe.core.compare(traj[-1], refs["ref_t30"])["eps_inf"]
    recorded = [r["eps_inf_t30"] for r in
                json.loads((TOL / "summary.json").read_text())["table"]
                if r["mode"] == "fp32" and r["field"] == "seed0" and r["tau"] == TAU]
    assert recorded, "policy table has no fp32/seed0 row"
    assert abs(eps - recorded[0]) <= 1e-9 + 1e-3 * abs(recorded[0]), (eps, recorded[0])
    floor = policy["fp32"]["noise_floor_eps_inf"]
    assert eps <= 3.0 * floor, (eps, floor)

# ---------------------------------------------------------------------------
# 4. the emulation really computes in the reduced precision, and the floor is
#    quoted for the trajectory length it was measured at
# ---------------------------------------------------------------------------
def test_fp32_path_is_fp32_arithmetic(refs):
    d = pe.step_intermediate_dtypes(refs["f0"], TAU, "fp32")
    assert set(d.values()) == {"float32"}, d
    d64 = pe.step_intermediate_dtypes(refs["f0"], TAU, "fp64")
    assert set(d64.values()) == {"float64"}, d64


def test_fp32_floor_grows_with_trajectory_length():
    ext = json.loads((TOL / "summary.json").read_text())["step_length_dependence"]
    fp32 = ext["fp32"]
    assert fp32["120"] > fp32["30"] > 0.0, fp32
    assert fp32["120"] / fp32["30"] < 10.0, fp32  # slow (sqrt-N-like) growth
