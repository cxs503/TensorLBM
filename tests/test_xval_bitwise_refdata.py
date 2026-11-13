"""Bitwise reference-data regression gate for the D2Q9 kernels.

These tests are the *integration caller* of
``benchmarks/xval_bitwise_refdata/harness.py``: the repo test suite drives the
real solver primitives and asserts two things:

1. the committed golden refdata is reproduced **bit for bit** (byte digest), and
2. the gate itself is not dead -- a 1-ulp tamper or a 1e-7 tau bump must be
   reported as a regression, and an environment mismatch must degrade to an
   explicit ulp budget instead of a false red.

If a kernel in ``src/tensorlbm/{d2q9,solver}.py`` is changed on purpose, the
golden data must be re-frozen deliberately::

    PYTHONPATH=src python3 benchmarks/xval_bitwise_refdata/run.py gen
    PYTHONPATH=src python3 benchmarks/xval_bitwise_refdata/run.py check

and the new ``result.json`` committed in the same change.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
HARNESS_DIR = REPO / "benchmarks" / "xval_bitwise_refdata"
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

import harness  # noqa: E402


@pytest.fixture(scope="module")
def golden() -> dict:
    return harness.load_golden()


# --------------------------------------------------------------------------- #
# 1. the gate is green on the committed golden data
# --------------------------------------------------------------------------- #
def test_golden_refdata_is_reproduced_bitwise(golden):
    reg = harness.verify(golden)
    assert reg["env_match"], f"environment/code drifted from golden: {reg['env_diffs']}"
    assert reg["n_regression"] == 0
    assert reg["n_bitwise_ok"] == reg["n_cases"] == len(harness.CASES)
    assert reg["passed"] is True
    # every case must carry a 64-hex sha256 digest, not a placeholder
    for name, case in reg["cases"].items():
        assert len(case["digest"]) == 64, name
        assert case["digest"] == case["golden_digest"], name
        assert case["invariants"]["finite"] is True, name


def test_committed_result_json_matches_current_kernels():
    """result.json must be produced by the kernels currently in the tree."""
    res = json.loads((HARNESS_DIR / "result.json").read_text())
    assert res["regression"]["code"] == harness.code_identity(), (
        "kernels changed since result.json was frozen -- re-run "
        "`run.py gen && run.py check` and commit both artifacts"
    )
    assert res["regression"]["passed"] is True
    assert res["benchmark"]["passed"] is True
    assert res["passed"] is True


# --------------------------------------------------------------------------- #
# 2. negative controls: the gate must be able to say "fail"
# --------------------------------------------------------------------------- #
def test_one_ulp_tamper_is_detected(golden):
    name = "chain_trt_50_24"
    f = harness.run_case(name)
    ref = harness.decode(golden["cases"][name]["payload"])
    bad = harness.tamper(f, at=(0, 5, 5), n_ulp=1)

    assert harness.digest(bad) != harness.digest(ref)
    um = harness.ulp_metrics(bad, ref)
    assert um["n_diff"] == 1
    assert um["max_ulp"] == 1.0
    assert um["first_mismatch"]["index"] == (0, 5, 5)


def _drifted_runner(case_name: str, at=(0, 2, 2), n_ulp: int = 1):
    """Simulated 1-ulp kernel drift on exactly one case."""

    def runner(name):
        f = harness.run_case(name)
        return harness.tamper(f, at=at, n_ulp=n_ulp) if name == case_name else f

    return runner


def test_verify_flags_regression_on_kernel_drift(golden):
    """Env matches and one population moves by 1 ulp -> hard red."""
    name = "bgk_step_16"
    reg = harness.verify(golden, runner=_drifted_runner(name))
    assert reg["passed"] is False
    assert reg["n_regression"] == 1
    assert reg["cases"][name]["verdict"] == "REGRESSION"
    assert reg["cases"][name]["ulp"]["max_ulp"] == 1.0


def test_verify_rejects_hand_edited_golden(golden):
    """A golden payload that no longer matches its own digest must be refused."""
    broken = copy.deepcopy(golden)
    name = "bgk_step_16"
    ref = harness.decode(broken["cases"][name]["payload"])
    broken["cases"][name]["payload"] = harness.encode(harness.tamper(ref, at=(0, 2, 2)))

    reg = harness.verify(broken)
    assert reg["passed"] is False
    assert reg["cases"][name]["verdict"] == "GOLDEN_INCONSISTENT"


def test_env_mismatch_degrades_to_ulp_budget(golden):
    """Unknown environment -> ulp budget instead of a false bitwise red."""
    simulated = copy.deepcopy(golden)
    simulated["env"]["torch"] = "0.0.0-not-the-golden-torch"
    reg = harness.verify(simulated)
    assert reg["env_match"] is False
    assert reg["ulp_budget"] > 0.0
    assert any("torch" in d for d in reg["env_diffs"])
    # unchanged generator + same machine: digest still matches bitwise
    assert reg["n_bitwise_ok"] == reg["n_cases"]

    # ... and 1 ulp of unknown drift on an unknown platform is ULP_DRIFT,
    # i.e. reported but not a false red (honest degradation, not silent pass)
    name = "bgk_step_16"
    reg2 = harness.verify(simulated, runner=_drifted_runner(name))
    assert reg2["cases"][name]["verdict"] == "ULP_DRIFT"
    assert reg2["cases"][name]["ulp"]["max_ulp"] == 1.0
    assert reg2["passed"] is True


def test_tau_bump_is_detected(golden):
    neg = harness.negative_controls(golden)
    assert neg["both_detected"] is True
    assert neg["tau_bump"]["digest_detects"] is True
    assert neg["tau_bump"]["max_ulp"] > 1.0


# --------------------------------------------------------------------------- #
# 3. analytic invariants of the primitives (independent of golden data)
# --------------------------------------------------------------------------- #
def test_feq_conserves_mass_and_momentum():
    ny, nx = 16, 16
    rho0, ux0, uy0 = harness.int_fields(ny, nx, torch.float64)
    f = harness.feq_field(ny, nx)
    rho, ux, uy = harness.solver.macroscopic(f)
    # sum_q f_eq = rho and sum_q f_eq c_q = rho u hold analytically; only
    # round-off (~1e-16) may remain
    assert torch.allclose(rho, rho0, atol=1e-14, rtol=0.0)
    assert torch.allclose(ux, ux0, atol=1e-14, rtol=0.0)
    assert torch.allclose(uy, uy0, atol=1e-14, rtol=0.0)
    assert float(f.min()) > 0.0


def test_stream_is_mass_conserving_permutation():
    f = harness.feq_field(16, 16)
    g = harness.solver.stream(f)
    assert float(g.sum()) == pytest.approx(float(f.sum()), abs=1e-13)
    # periodic streaming only moves values between cells: the multiset is identical
    assert torch.equal(torch.sort(f.flatten()).values, torch.sort(g.flatten()).values)


def test_collision_conserves_mass():
    """float64 mass must be conserved to round-off, not to 1e-8.

    ``collide_bgk_fused`` used to cast the float32 weight table into float64
    (sum = 1 + 7.45e-9), biasing rho by +7.45e-9/tau per step.  This test is the
    guard that the fix uses dtype-exact weights.
    """
    f = harness.feq_field(16, 16)
    m0 = float(f.sum())
    for name, fn in (
        ("collide_bgk", lambda x: harness.solver.collide_bgk(x, 0.8)),
        ("collide_trt", lambda x: harness.solver.collide_trt(x, 0.8)),
        ("collide_mrt", lambda x: harness.solver.collide_mrt(x, 0.8)),
        ("collide_bgk_fused", lambda x: harness.solver.collide_bgk_fused(x, 0.8)),
        ("collide_rlbm", lambda x: harness.solver.collide_rlbm(x, 0.8)),
    ):
        assert float(fn(f).sum()) == pytest.approx(m0, rel=1e-14), name


def test_fused_chain_has_no_linear_mass_ramp():
    """200 fused steps: drift must stay at round-off (was 1.86e-6 before the fix)."""
    f0 = harness.feq_field(16, 16)
    m0 = float(f0.sum())
    f = f0
    for _ in range(200):
        f = harness.solver.stream(harness.solver.collide_bgk_fused(f, 0.8))
    drift = abs(float(f.sum()) - m0) / m0
    assert drift < 1e-12, f"linear mass ramp reintroduced: {drift:.3e}"


def test_float32_weight_table_is_unchanged_by_the_fix():
    """The fix must be a no-op for float32 (same bit pattern as the old table)."""
    from tensorlbm import d2q9

    assert torch.equal(d2q9._w_for(torch.float32, torch.device("cpu")), d2q9.W)


# --------------------------------------------------------------------------- #
# 4. determinism across processes (the property the whole lock relies on)
# --------------------------------------------------------------------------- #
def test_digest_reproducible_in_a_fresh_process():
    code = (
        "import sys;"
        f"sys.path.insert(0, {str(HARNESS_DIR)!r});"
        f"sys.path.insert(0, {str(REPO / 'src')!r});"
        "import harness;"
        "print(harness.digest(harness.run_case('bgk_step_16')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == harness.digest(harness.run_case("bgk_step_16"))


# --------------------------------------------------------------------------- #
# 5. the physical benchmark behind the lock
# --------------------------------------------------------------------------- #
def test_tgv_benchmark_below_acceptance_tolerance():
    cfg = {
        "levels": [
            {"ny": 24, "nx": 24, "tau": 0.8, "steps": 56},
            {"ny": 32, "nx": 32, "tau": 0.8, "steps": 100},
        ],
        "tau_points": [{"ny": 32, "nx": 32, "tau": 0.9, "steps": 75}],
        "tolerance_pct": 15.0,
    }
    bench = harness.benchmark(cfg)
    assert bench["passed"] is True
    errs = [lv["amp_err_pct_final"] for lv in bench["levels"]]
    assert max(errs) < 1.0, errs  # owner accuracy bar is 15%; TRT TGV is ~0.1-0.8%
    assert errs[0] > errs[1] > 2.0 * errs[1] or errs[0] > errs[1]  # resolution helps
    for lv in bench["levels"] + bench["tau_points"]:
        # weakly compressible: rho spread O(Ma^2) ~ 2.5e-3 plus coarse-grid
        # start-up transient; 2e-2 is a loose sanity ceiling, not a fit
        assert lv["max_rho_spread"] < 2e-2
        assert lv["field_l2_pct_final"] < 2.0
