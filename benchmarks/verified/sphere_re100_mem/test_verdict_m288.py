"""Guards for the m288 verdict closure (pytest).

The verdict file must be *recomputable*: every judged number is rederived here
from the raw per-sample force histories with the unchanged case README rule
(last 20% window mean, |err| <= 3% vs the locked Schiller-Naumann reference,
|drift| < 0.3%, steps >= 8000, tau = 3*u_lb*D/Re + 0.5).  Any drift of the
judging rule, of the reference value or of the reported numbers fails here.

Run:  python -m pytest benchmarks/verified/sphere_re100_mem/test_verdict_m288.py -q
"""

from __future__ import annotations

import json
import os

import pytest

BASE = os.path.dirname(os.path.abspath(__file__))
W8A = os.path.join(BASE, "..", "sphere_re100", "w8a")
VERDICT = os.path.join(BASE, "verdict_m288.json")

CD_REF = 24.0 / 100.0 * (1.0 + 0.15 * 100.0**0.687)
RAW = {
    "T1_d40big_12k": os.path.join(W8A, "t1_d40big_12k.json"),
    "T2_d60big_24k": os.path.join(W8A, "t2_d60big_24k.json"),
}


@pytest.fixture(scope="module")
def verdict():
    if not os.path.exists(VERDICT):
        pytest.skip("verdict_m288.json not built yet")
    with open(VERDICT) as fh:
        return json.load(fh)


def _window(hist, frac=0.2):
    vals = [h["cd"] for h in hist]
    w = max(1, int(len(vals) * frac))
    tail, prev = vals[-w:], vals[-2 * w: -w]
    mean = sum(tail) / len(tail)
    mprev = sum(prev) / len(prev)
    return w, mean, (mean - mprev) / mprev * 100.0


def test_reference_is_the_locked_schiller_naumann(verdict):
    assert verdict["reference"]["cd_ref"] == pytest.approx(CD_REF, abs=0, rel=0)
    assert verdict["reference"]["gate_pct"] == 3.0
    assert verdict["reference"]["locked_before_runs"] is True


def test_two_tiers_present(verdict):
    assert [t["tier"] for t in verdict["tiers"]] == list(RAW)


@pytest.mark.parametrize("tier", list(RAW))
def test_tier_numbers_recompute_from_raw_history(verdict, tier):
    t = next(x for x in verdict["tiers"] if x["tier"] == tier)
    with open(RAW[tier]) as fh:
        d = json.load(fh)
    w, mean, drift = _window(d["history"])
    assert t["window"]["n_window"] == w
    assert t["window"]["cd_mean"] == pytest.approx(mean, rel=0, abs=1e-15)
    assert t["window"]["drift_pct"] == pytest.approx(drift, rel=1e-12, abs=1e-12)
    err = (mean - CD_REF) / CD_REF * 100.0
    assert t["err_pct"] == pytest.approx(err, rel=0, abs=1e-12)
    assert t["gate_ok"] == (abs(err) <= 3.0)
    assert t["steady_ok"] == (
        abs(drift) < 0.3 and d["meta"]["steps"] >= 8000 and t["tau_ok"]
    )
    assert t["verdict"] == ("PASS" if (t["gate_ok"] and t["steady_ok"]) else "FAIL")


@pytest.mark.parametrize("tier", list(RAW))
def test_ci_brackets_the_mean(verdict, tier):
    t = next(x for x in verdict["tiers"] if x["tier"] == tier)
    lo, hi = t["window"]["ci95"]
    assert lo < t["window"]["cd_mean"] < hi
    assert t["window"]["ci95_halfwidth"] == pytest.approx(
        (hi - lo) / 2.0, rel=1e-12, abs=1e-15
    )
    # 95% CI must be far inside the gate margin for the verdict to be robust
    assert t["window"]["ci95_halfwidth"] / CD_REF * 100.0 < 3.0 - abs(t["err_pct"])


def test_verdict_matches_ladder(verdict):
    ladder = verdict["ladder"]
    ok = all(t["gate_ok"] and t["steady_ok"] for t in verdict["tiers"])
    mono = abs(verdict["tiers"][0]["err_pct"]) > abs(verdict["tiers"][1]["err_pct"])
    assert ladder["all_tiers_within_gate_and_steady"] == ok
    assert ladder["abs_err_strictly_decreasing"] == mono
    assert ladder["verdict"] == ("PASS" if (ok and mono) else "FAIL")
    assert verdict["verdict"] == "VERDICT: " + ladder["verdict"]
    assert verdict["verdict_origin"].startswith("NO_VERDICT ->")


def test_verdict_uses_the_readme_window_only(verdict):
    """The gate is judged on the pre-registered last-20% window, nothing else."""
    for t in verdict["tiers"]:
        n_total = t["window"]["n_samples_total"]
        assert t["window"]["n_window"] == max(1, int(n_total * 0.2)), t["tier"]
        # the judged number must be the one recomputed for the README window
        key = str(t["window"]["n_window"])
        assert key in t["window_sensitivity"], (t["tier"], key)
        assert abs(t["window_sensitivity"][key]["err_pct"] - t["err_pct"]) < 1e-9


def test_long_windows_are_documented_as_boundary(verdict):
    """Longer windows (which include the initial transient) must NOT be hidden:
    they are reported, and the verdict JSON must say the window choice matters."""
    over = [
        (t["tier"], w, rec["err_pct"])
        for t in verdict["tiers"]
        for w, rec in t["window_sensitivity"].items()
        if abs(rec["err_pct"]) > 3.0
    ]
    if over:  # honest reporting: boundary exists and is disclosed
        assert any("window" in s.lower() for s in verdict["limits"]), over


def test_device_evidence_file_is_consistent(verdict):
    p = os.path.join(BASE, "device_evidence_m288.json")
    if not os.path.exists(p):
        pytest.skip("device_evidence_m288.json not built yet")
    with open(p) as fh:
        ev = json.load(fh)
    dev = ev["environment"]
    assert "sdaa" in dev["real_device"]
    assert "/dev/tcaicard" in dev["sdaa_0_name"]
    assert dev["cuda_is_available"] is False
    # the verdict must never claim device provenance where there is none
    for t in verdict["tiers"]:
        if not t["device"]["sdaa_evidence_ok"]:
            assert t["device"]["device_evidence_source"].startswith("archived artifact only")