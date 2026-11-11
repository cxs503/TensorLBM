"""Tests for the heterogeneous-backend equivalence ladder.

Coverage map (each test names the gate it protects):

* registry / probe / fail-closed refusals  -> criterion (d)
* precision gate (no silent fp64 -> fp32)  -> criterion (d)
* L0 bitwise determinism + cross-framework fp64 equivalence -> criterion (a)
* L0 rounding budget, with a *negative control* (shipped truncated weight
  table) that must fail the gate -> the gate is not vacuous
* L2 conservation drift: fp64 passes, fp32 fails -> criterion (b)
* L1 statistical comparator logic (synthetic profiles, no integration)
* chunked channel driver == monolithic driver (resume correctness)
* Pareto helper

Heavy integration runs (64^3 x 200 steps, Re_tau=180 x 10^4 steps) are executed
by ``scripts/run_p106w3a2_n01_equivalence.py`` and stored under
``artifacts/p106w3a2_n01/``; the tests here keep the small, fast checks so that
``pytest -q tests/test_hetero_equivalence.py`` stays sub-minute.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import tensorlbm.hetero_equivalence as HE


# ---------------------------------------------------------------------------
# registry, probe, fail-closed
# ---------------------------------------------------------------------------
def test_profile_registry_has_golden_and_accelerators():
    assert HE.GOLDEN_PROFILE in HE.PROFILES
    for name in ("cpu_fp64_torch", "cpu_fp32_torch", "cuda_fp64_torch",
                 "triton_fused_fp32", "sdaa_fp32_torch", "paddle_fp32"):
        assert name in HE.PROFILES


def test_unknown_profile_raises_keyerror():
    with pytest.raises(KeyError):
        HE.resolve_profile("cpu_fp128_magic")


def test_cpu_tiers_are_available():
    ledger = HE.RefusalLedger()
    for name in ("cpu_fp64_serial", "cpu_fp64_torch", "cpu_fp32_torch"):
        prof = HE.resolve_profile(name, ledger=ledger)
        assert prof.name == name  # never silently substituted
    assert len(ledger) == 0


def test_missing_backend_fails_closed_and_is_registered():
    """A tier whose device/framework is absent must raise, never fall back."""
    torch = pytest.importorskip("torch")
    import importlib.util

    unavailable = None
    if importlib.util.find_spec("paddle") is None:
        unavailable = "paddle_fp32"
    elif not torch.cuda.is_available():
        unavailable = "cuda_fp32_torch"
    if unavailable is None:
        pytest.skip("every registered refusal tier is present on this host")

    ledger = HE.RefusalLedger()
    with pytest.raises(HE.BackendUnavailable):
        HE.resolve_profile(unavailable, ledger=ledger)
    records = ledger.as_list()
    assert len(records) == 1
    assert records[0]["profile"] == unavailable
    assert records[0]["kind"] == "unavailable"
    assert records[0]["reason"]


def test_cuda_tier_refused_without_cuda():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    ledger = HE.RefusalLedger()
    with pytest.raises(HE.BackendUnavailable):
        HE.resolve_profile("cuda_fp64_torch", ledger=ledger)
    assert ledger.as_list()[0]["reason"].startswith("torch.cuda.is_available()")


def test_precision_gate_refuses_downgrade():
    ledger = HE.RefusalLedger()
    with pytest.raises(HE.PrecisionInsufficient):
        HE.require_precision("cpu_fp32_torch", "float64", ledger=ledger)
    rec = ledger.as_list()[0]
    assert rec["kind"] == "precision"
    # the fp64 tier is allowed
    assert HE.require_precision("cpu_fp64_torch", "float64").dtype == "float64"


def test_require_certified_fails_closed():
    certs = {"cpu_fp32_torch": {"L0": "certified", "L2": "failed"}}
    ledger = HE.RefusalLedger()
    assert HE.require_certified("cpu_fp32_torch", certs, "L0", ledger=ledger) == "certified"
    with pytest.raises(HE.CertificationError):
        HE.require_certified("cpu_fp32_torch", certs, "L2", ledger=ledger)
    with pytest.raises(HE.CertificationError):
        HE.require_certified("cpu_fp64_torch", certs, "L1", ledger=ledger)  # not evaluated
    assert {r["kind"] for r in ledger.as_list()} == {"uncertified"}


# ---------------------------------------------------------------------------
# L0
# ---------------------------------------------------------------------------
N_SMALL = 24
STEPS_SMALL = 40


@pytest.fixture(scope="module")
def golden_small():
    return HE.l0_run(HE.GOLDEN_PROFILE, n=N_SMALL, steps=STEPS_SMALL)


def test_l0_golden_is_deterministic(golden_small):
    again = HE.l0_run(HE.GOLDEN_PROFILE, n=N_SMALL, steps=STEPS_SMALL)
    assert again["digest_chain"] == golden_small["digest_chain"]
    assert again["final_fields_sha256"] == golden_small["final_fields_sha256"]


@pytest.mark.parametrize("tier", ["cpu_fp64_numpy_tree", "cpu_fp64_torch_gemm"])
def test_l0_fp64_cross_implementation(golden_small, tier):
    """Independent fp64 implementations must agree bitwise or inside the budget."""
    run = HE.l0_run(tier, n=N_SMALL, steps=STEPS_SMALL)
    check = HE.check_l0(golden_small, run, amplification=440.0)
    assert check["passed"], check
    assert check["max_rel_deviation_field"] <= check["ulp_budget_relative"]


def test_l0_fp32_within_rounding_budget(golden_small):
    run = HE.l0_run("cpu_fp32_torch", n=N_SMALL, steps=STEPS_SMALL)
    check = HE.check_l0(golden_small, run, amplification=380.0)
    assert not check["bitwise_identical"]  # fp32 cannot be bit-identical to fp64
    assert check["passed"], check
    assert 0.0 < check["budget_utilisation"] <= 1.0


def test_l0_negative_control_shipped_weights_fails(golden_small):
    """The shipped float32 weight table must be caught by the L0 gate."""
    run = HE.l0_run("cpu_fp64_shipped_w", n=N_SMALL, steps=STEPS_SMALL)
    check = HE.check_l0(golden_small, run, amplification=440.0)
    assert not check["bitwise_identical"]
    assert not check["passed"], "the L0 budget must reject a scheme-level difference"


def test_weight_table_audit_documents_the_defect():
    audit = HE.weight_table_audit()
    assert audit["excess_sum"] > 0.0
    assert math.isclose(audit["exact_sum"], 1.0, rel_tol=0, abs_tol=1e-15)
    # the shipped table is float32: the weights are off by ~1e-8, and their sum
    # is off by 1.5e-8 - that is the mass source the L2 gate rejects
    assert 0.0 < audit["max_abs_weight_error"] < 1e-7
    assert audit["shipped_dtype"] == "float32"


def test_ulp_budget_scales_with_precision():
    assert HE.ulp_budget(1.0, 2.0**-53) < HE.ulp_budget(1.0, 2.0**-24)
    assert math.isclose(HE.ulp_budget(2.0, 1e-16), 2.0 * HE.ulp_budget(1.0, 1e-16))


# ---------------------------------------------------------------------------
# L2
# ---------------------------------------------------------------------------
def test_l2_fp64_conserves_mass_and_momentum(golden_small):
    check = HE.check_l2(golden_small)
    assert check["passed"], check
    assert check["mass"]["drift_per_step_per_cell"] <= HE.L2_DRIFT_MAX


def test_l2_fp32_drift_is_quantified_and_rejected():
    run = HE.l0_run("cpu_fp32_torch", n=N_SMALL, steps=STEPS_SMALL)
    check = HE.check_l2(run)
    assert not check["passed"]
    # fp32 noise is ~7 orders of magnitude above the invariant gate
    assert check["mass"]["drift_per_step_per_cell"] > 1e-9


# ---------------------------------------------------------------------------
# L1 comparators (synthetic profiles: no integration)
# ---------------------------------------------------------------------------
def _fake_l1(name="cpu_fp64_torch", dtype="float64", scale=1.0):
    y_plus = [0.0, 1.0, 5.0, 10.0, 20.0, 40.0, 80.0, 120.0, 180.0]
    u_plus = [0.0, 1.0, 5.0, 9.0, 12.0, 15.0, 17.5, 18.3, 19.0]
    return {
        "level": "L1",
        "profile": {"name": name, "dtype": dtype},
        "wall_scheme": "interpolated",
        "wall_units": {
            "y_plus": y_plus,
            "u_plus": [v * scale for v in u_plus],
            "uu_plus": [v * scale for v in [0, 0, 0.5, 1.0, 1.5, 2.0, 2.2, 2.2, 2.0]],
            "vv_plus": [v * scale for v in [0, 0, 0.05, 0.2, 0.5, 0.8, 0.9, 0.9, 0.8]],
            "ww_plus": [v * scale for v in [0, 0, 0.1, 0.4, 0.9, 1.3, 1.5, 1.5, 1.3]],
            "uv_plus": [0.0] * 9,
        },
    }


def test_l1_identical_profiles_pass():
    golden = _fake_l1()
    check = HE.check_l1(golden, _fake_l1(name="cpu_fp32_torch", dtype="float32"))
    assert check["passed"]
    assert check["worst_normalised_l2"] == pytest.approx(0.0, abs=1e-15)


def test_l1_one_percent_perturbation_passes_two_percent_fails():
    golden = _fake_l1()
    ok = HE.check_l1(golden, _fake_l1(name="b", scale=1.01))
    bad = HE.check_l1(golden, _fake_l1(name="c", scale=1.05))
    assert ok["passed"] and ok["worst_normalised_l2"] < 0.02
    assert not bad["passed"] and bad["worst_normalised_l2"] > 0.02


def test_l1_mkm_not_worse_gate():
    class _MKM:
        re_tau = 178.12

        def __init__(self, y_plus, u_plus):
            self.y_plus = np.asarray(y_plus, dtype=np.float64)
            self.u_plus = np.asarray(u_plus, dtype=np.float64)

    golden = _fake_l1()
    mkm = _MKM(golden["wall_units"]["y_plus"], golden["wall_units"]["u_plus"])
    same = HE.check_l1(golden, _fake_l1(name="b"), mkm=mkm)
    worse = HE.check_l1(golden, _fake_l1(name="c", scale=1.05), mkm=mkm)
    assert same["mkm"]["not_worse"] and same["passed"]
    assert not worse["mkm"]["not_worse"] and not worse["passed"]


def test_normalised_profile_deviation_rejects_mismatched_grids():
    a = _fake_l1()
    b = _fake_l1()
    b["wall_units"]["y_plus"] = [v + 1.0 for v in b["wall_units"]["y_plus"]]
    with pytest.raises(ValueError):
        HE.normalised_profile_deviation(a, b)


# ---------------------------------------------------------------------------
# chunked driver
# ---------------------------------------------------------------------------
def test_chunked_channel_matches_monolithic(tmp_path):
    """Splitting the integration in chunks must not change the result."""
    pytest.importorskip("torch")
    kw = dict(nx=8, ny=16, nz=8, steps=40, warmup_steps=10, sample_interval=2,
              re_tau=180.0, u_tau=0.01, smagorinsky_cs=0.1, seed=7)
    mono = HE.l1_run_channel("cpu_fp64_torch", **kw)
    run = HE.ChunkedChannelRun("cpu_fp64_torch", state_dir=tmp_path, **kw)
    for _ in range(2):
        run.advance(20)
    chunked = run.finalize()
    for key in ("u_plus", "uu_plus", "vv_plus", "ww_plus"):
        a = np.asarray(mono["wall_units"][key], dtype=np.float64)
        b = np.asarray(chunked["wall_units"][key], dtype=np.float64)
        assert np.allclose(a, b, rtol=0, atol=1e-12), key
    assert chunked["steps_done"] == 40
    assert chunked["samples"] == mono["samples"]


def test_chunked_state_refuses_incompatible_parameters(tmp_path):
    pytest.importorskip("torch")
    kw = dict(nx=8, ny=16, nz=8, steps=40, warmup_steps=10, sample_interval=2,
              re_tau=180.0, u_tau=0.01, smagorinsky_cs=0.1, seed=7)
    run = HE.ChunkedChannelRun("cpu_fp64_torch", state_dir=tmp_path, **kw)
    run.advance(20)
    with pytest.raises(ValueError):
        HE.ChunkedChannelRun("cpu_fp64_torch", state_dir=tmp_path, **{**kw, "steps": 60})


def test_l1_rejects_non_torch_tier():
    with pytest.raises(HE.BackendUnavailable):
        HE.l1_run_channel("cpu_fp64_serial", nx=8, ny=16, nz=8, steps=10, warmup_steps=2,
                          sample_interval=1)


# ---------------------------------------------------------------------------
# ledger helpers
# ---------------------------------------------------------------------------
def test_pareto_front_keeps_only_non_dominated_points():
    points = [
        {"name": "slow_accurate", "cost": 10.0, "error": 1e-6},
        {"name": "fast_inaccurate", "cost": 1.0, "error": 1e-2},
        {"name": "dominated", "cost": 10.0, "error": 1e-2},
        {"name": "middle", "cost": 4.0, "error": 1e-4},
    ]
    front = HE.pareto_front(points)
    assert "dominated" not in front
    assert set(front) == {"fast_inaccurate", "middle", "slow_accurate"}


def test_freeze_golden_writes_hash_manifest(tmp_path, golden_small):
    manifest = HE.freeze_golden(golden_small, tmp_path, name="golden_test")
    assert len(manifest["per_step_digests"]) == STEPS_SMALL
    assert len(manifest["npz_sha256"]) == 64
    stored = (tmp_path / "golden_test_manifest.json").read_text(encoding="utf-8")
    assert manifest["trace_digest_chain"] in stored
    assert (tmp_path / "golden_test.npz").exists()


def test_aggregate_ladder_registers_refusals():
    ledger = HE.RefusalLedger()
    payload = HE.aggregate_ladder(ledger)
    assert len(payload["inventory"]) == len(HE.PROFILES)
    refused = {r["profile"] for r in payload["refusals"]}
    available = {i["profile"] for i in payload["inventory"] if i["available"]}
    assert refused.isdisjoint(available)