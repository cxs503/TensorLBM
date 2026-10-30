"""Self-checks for the SDAA sweep artifact of case sphere_re100_d3q27.

These tests do not touch the solver: they verify that the machine output in
``artifacts/sdaa_sphere_re100_d3q27.json`` is internally consistent, i.e. that
every judgment number can be re-derived from the recorded primitives and that
each gate verdict follows from the frozen threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CASE = REPO / "benchmarks" / "verified" / "sphere_re100_d3q27"
ART = REPO / "artifacts" / "sdaa_sphere_re100_d3q27.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    if not ART.exists():
        pytest.skip(f"{ART} absent (run scripts/sdaa_sweep_sphere_re100_d3q27.py)")
    return json.loads(ART.read_text())


def test_device_evidence_present(payload: dict) -> None:
    line = payload["device_evidence"]["device_evidence_line"]
    assert "torch.sdaa.get_device_name(0)" in line
    assert payload["device_evidence"]["sdaa_device_name_0"]
    # a CPU/CUDA run must not be able to masquerade as an SDAA run
    assert payload["device_evidence"]["torch_device_used"].startswith("sdaa")
    assert payload["device_evidence"]["probe_matches_cpu"] is True


def test_cd_ref_matches_the_pinned_archive(payload: dict) -> None:
    arch = json.loads(
        (CASE / "diag" / "remote_pull" / "verified__sphere_re100" / "result.json").read_text()
    )
    assert payload["reference"]["cd_ref"] == arch["reference"]["cd_ref"]
    assert all(v["match"] for v in payload["reference"]["sha256_pins"].values())


def test_err_and_gate_verdicts_are_self_consistent(payload: dict) -> None:
    cd_ref = payload["reference"]["cd_ref"]
    for tier in payload["tiers"]:
        if tier.get("status") != "RAN":
            continue
        err = abs(tier["cd_window"] - cd_ref) / cd_ref * 100.0
        assert err == pytest.approx(tier["err_pct"], rel=0, abs=1e-9)
        g = tier["g3"]
        assert g["drift_ok"] == (g["drift_pct"] < 0.3)
        assert g["cv_ok"] == (g["cv_closure"] is not None and g["cv_closure"] <= 1e-2)
        assert g["mass_ok"] == (abs(g["mass_drift_ppm"]) <= 100)
        assert tier["err_ok"] == (tier["err_pct"] <= 3.0)
        # Re held fixed at 100 by the tau rule
        assert abs(tier["re_eff"] - 100.0) <= 1e-6


def test_gate_table_matches_tiers_and_overall(payload: dict) -> None:
    rows = {r["gate"]: r for r in payload["gate_table"]}
    for tier in payload["tiers"]:
        if tier.get("status") != "RAN":
            continue
        d = tier["meta"]["D"]
        assert rows[f"G1.accuracy.D{d}"]["verdict"] == (
            "PASS" if tier["err_ok"] else "FAIL")
        assert rows[f"G3.D{d}.drift"]["verdict"] == (
            "PASS" if tier["g3"]["drift_ok"] else "FAIL")
    applicable = [r for r in payload["gate_table"] if r["verdict"] != "NOT_APPLICABLE"]
    expected = "PASS" if all(r["verdict"] == "PASS" for r in applicable) else "FAIL"
    assert payload["overall_verdict"] == expected


def test_monotone_row_uses_two_tiers(payload: dict) -> None:
    ran = [t for t in payload["tiers"] if t.get("status") == "RAN"]
    row = next(r for r in payload["gate_table"] if r["gate"] == "G2.monotone")
    if len(ran) == 2:
        assert row["verdict"] == (
            "PASS" if ran[1]["err_pct"] < ran[0]["err_pct"] else "FAIL")
    else:
        assert row["verdict"] == "NOT_APPLICABLE"


def test_tier_histories_exist_and_are_consistent(payload: dict) -> None:
    for tier in payload["tiers"]:
        if tier.get("status") != "RAN":
            continue
        tj = REPO / tier["tier_json"]
        assert tj.exists()
        meta = json.loads(tj.read_text())["meta"]
        assert meta["domain_lu"] == tier["meta"]["domain_lu"]
        assert meta["tau"] == pytest.approx(tier["meta"]["tau"], abs=0.0)
        assert meta["steps"] == tier["steps_used"]