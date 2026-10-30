"""Self-consistency tests for the SDAA sweep artifact of ``stokes_sphere_dg``.

These tests never re-run the (14.9 min) simulation.  They check that every
judged number in ``artifacts/sdaa_stokes_sphere_dg.json`` can be *recomputed*
from the raw series that the same file stores, i.e. that no judgment value was
hand-copied, and that the device evidence line is present.
"""

from __future__ import annotations

import json
import math
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(REPO, "artifacts", "sdaa_stokes_sphere_dg.json")
CASE = os.path.join(REPO, "benchmarks", "verified", "stokes_sphere_dg")


@pytest.fixture(scope="module")
def art():
    if not os.path.exists(ART):
        pytest.skip("sweep artifact missing (run scripts/sdaa_sweep_stokes_sphere_dg.py)")
    with open(ART) as fh:
        return json.load(fh)


def _tail20(xs):
    k = max(1, int(math.ceil(0.2 * len(xs))))
    return xs[len(xs) - k :]


def _tail40(xs):
    k = max(1, int(math.ceil(0.4 * len(xs))))
    return xs[len(xs) - k :]


def test_device_evidence_present(art):
    ev = art["device_evidence"]
    line = ev["device_evidence_line"]
    assert line.startswith("torch.sdaa.get_device_name(0) = ")
    assert line.split("=", 1)[1].strip()  # a real device name, not empty
    assert ev["device"] == "sdaa:0"
    assert ev["cuda_available"] is False, "a CUDA/CPU path would invalidate the evidence"
    assert ev["fp64_real"] is True, "float64 must be real on the accelerator"


def test_gauge_is_the_case_gauge(art):
    g = art["gauge_source"]
    import hashlib

    h = hashlib.md5(open(os.path.join(CASE, "reference_table.json"), "rb").read()).hexdigest()
    assert g["reference_table_md5"] == h == g["pinned_md5"]
    assert g["md5_match"] is True
    # thresholds must be the case's own limits, not relaxed copies
    assert g["limits"]["ERR_LIM"] == pytest.approx(0.03)
    assert g["limits"]["S0"] == pytest.approx(1e-6)
    assert g["limits"]["S1"] == pytest.approx(1e-3)
    assert g["limits"]["S2"] == pytest.approx(1e-3)
    assert g["limits"]["S3"] == pytest.approx(1e-2)


def test_no_cpu_numbers_and_full_step_count(art):
    for t in art["tiers"]:
        assert t["steps_run"] == t["steps_full"], (
            "steps must follow the case step table (no shortened/CPU-style run)"
        )
        assert t["status"] == "completed"
        for s in t["series"]:
            for key in ("U_sup", "K", "F_ME", "mass"):
                assert math.isfinite(s[key])
        assert t["geometry"]["geometry_gate_pass"] is True
        assert t["geometry"]["n_wrapped"] == 0


def test_judged_numbers_recompute_from_series(art):
    for t in art["tiers"]:
        v = t["verdict"]
        series = t["series"]
        Ks = [s["K"] for s in series]
        Us = [s["U_sup"] for s in series]
        F_ledger = t["estimators"]["F_ledger"]

        k_sim = sum(_tail20(Ks)) / len(_tail20(Ks))
        assert v["K_sim"] == pytest.approx(k_sim, rel=1e-12)

        err = k_sim / v["K_ref"] - 1.0
        assert v["err"] == pytest.approx(err, rel=1e-10)
        assert v["err_pct"] == pytest.approx(err * 100.0, rel=1e-10)

        u20 = sum(_tail20(Us)) / len(_tail20(Us))
        u40 = sum(_tail40(Us)) / len(_tail40(Us))
        assert v["U_sup_tail20"] == pytest.approx(u20, rel=1e-12)
        assert v["S2_rel"] == pytest.approx(abs(u20 - u40) / abs(u20), rel=1e-10)

        s0 = max(s["mass_drift"] for s in series)
        assert v["S0_max_mass_drift"] == pytest.approx(s0, rel=1e-10)

        rel = [abs(s["F_ME"] - F_ledger) / F_ledger for s in _tail20(series)]
        assert v["S3_mean_rel"] == pytest.approx(sum(rel) / len(rel), rel=1e-10)


def test_gates_use_the_case_thresholds(art):
    for g in art["gate_table"]:
        lim = g.get("limit")
        if lim is not None:
            passed_by_hand = (abs(g["value"]) <= abs(lim)) if "MAIN" in g["gate"] else (
                g["value"] <= lim
            )
            assert g["pass"] is bool(passed_by_hand), g
    assert art["n_gates"] == len(art["gate_table"])
    assert art["n_gates_fail"] == sum(1 for g in art["gate_table"] if not g["pass"])
    if art["n_gates_fail"]:
        assert art["overall_verdict"] == "FAIL"
    else:
        assert art["overall_verdict"] in ("PASS", "PARTIAL_PASS")


def test_reproduces_nvidia_archive(art):
    """The SDAA run must reproduce the archived NVIDIA verdict for the same tier."""
    for t in art["tiers"]:
        ar = t.get("nvidia_archive", {})
        if not ar.get("found"):
            continue
        assert t["steps_run"] == ar["archive_steps"]
        assert t["verdict"]["K_sim"] == pytest.approx(ar["archive_K_sim"], rel=1e-9)


def test_coverage_is_disclosed(art):
    """Partial coverage must be reported, never silently upgraded to PASS."""
    cov = art["coverage"]
    assert cov["tiers_reference"] == 12
    if cov["tiers_executed"] < 12:
        assert art["overall_verdict"] == "PARTIAL_PASS"
        assert any(v["convergence_gate"] == "NOT_EVALUATED"
                   for v in art["convergence"].values())