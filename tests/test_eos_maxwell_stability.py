"""Gates for the EOS-Maxwell construction stability probe (benchmarks/eos_maxwell).

These tests are the *cheap*, CI-sized half of the probe: they assert the
construction machinery and the measurement/reporting contract, not the
production campaign (which is an explicit cluster job, see
``benchmarks/eos_maxwell/README.md``).

Covered contracts
-----------------
1. **Construction is real and self-consistent** — the Maxwell equal-area solver
   returns a pair whose relative area residual is at round-off, the pair lies
   below the psi-clamp density, and bisection hits the requested ratio.
2. **Monotonicity of the construction** — the coexistence ratio decreases with
   ``T_r`` (the property bisection relies on).
3. **Independent cross-check** — the same construction at ``T_r = 0.55``
   reproduces the Maxwell lock embedded in the *verified* benchmark
   ``benchmarks/verified/laplace_pr_eos`` (123.2232) to 1e-3 relative.  This is
   a cross-implementation check, not a self-consistency check.
4. **Periodic-safe planar control** — the flat-interface initialisation is
   periodic (a single step, liquid-below/vapour-above, is *not*: it puts a
   one-cell full density jump across the periodic seam and inflates every force
   diagnostic).
5. **Honest-fail contract** — a construct that is known to be beyond the
   stability edge returns a structured failure record (onset step + mechanism)
   instead of raising, and does not silently produce numbers.
6. **Consumer anchors exist** — the artefacts named in the consumer map
   (``benchmarks/eos_maxwell/consumers.json``) are present in the tree.
7. **Viscosity lever contract** — the tau arm exists, its decision rule is
   predeclared, and the pure post-processor that applies it returns each of the
   four predeclared branches (unit-tested on synthetic records, no marching).
8. **Artefact traceability** — if the committed ``out/results.json`` is present,
   its recorded code hashes equal the hashes of the files in the tree, and the
   split-job provenance is intact.  A stale table then fails loudly instead of
   being quoted.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parent / "benchmarks" / "eos_maxwell"

if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import probe_lib as P  # noqa: E402

# ---------------------------------------------------------------------------
# 1-2. construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [10.0, 100.0])
def test_maxwell_construction_hits_target(target):
    mw = P.maxwell_at_ratio(target, eos="pr")
    assert mw["found"] is True
    assert mw["within_target_tol"] is True
    assert abs(mw["ratio"] - target) / target <= P.PREREG["ratio_target_tol"]
    # equal-area residual is at round-off relative to p_sat * (v_v - v_l)
    assert mw["area_residual_rel"] < P.PREREG["maxwell_area_tol"]
    # the liquid density must stay below the psi-clamp density
    assert 0.0 < mw["rho_vapour"] < mw["rho_liquid"] < mw["rho_cross"]
    assert mw["p_sat"] > 0.0


def test_coexistence_ratio_monotone_in_temperature():
    ratios = [P.maxwell_at_tr("pr", tr)["ratio"] for tr in (0.85, 0.60, 0.45)]
    assert all(r["found"] for r in [P.maxwell_at_tr("pr", tr) for tr in (0.85, 0.60, 0.45)])
    assert ratios[0] < ratios[1] < ratios[2], ratios


# ---------------------------------------------------------------------------
# 3. cross-check against the verified benchmark lock
# ---------------------------------------------------------------------------


def test_matches_verified_laplace_pr_eos_lock():
    """The construction reproduces the frozen lock of laplace_pr_eos (T_r=0.55)."""
    run_py = _HERE.parent / "benchmarks" / "verified" / "laplace_pr_eos" / "run.py"
    if not run_py.exists():  # pragma: no cover - benchmark moved/renamed
        pytest.skip("laplace_pr_eos benchmark not present")
    spec = importlib.util.spec_from_file_location("laplace_pr_eos_run", run_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    lock = mod.MAXWELL_LOCK["pr"]["table"][0]
    assert lock["T_r"] == pytest.approx(0.55)
    ours = P.maxwell_at_tr("pr", lock["T_r"])
    assert ours["found"] is True
    assert ours["ratio"] == pytest.approx(lock["ratio"], rel=1e-3)
    assert ours["rho_liquid"] == pytest.approx(lock["rho_l"], rel=1e-3)
    assert ours["rho_vapour"] == pytest.approx(lock["rho_v"], rel=1e-2)


# ---------------------------------------------------------------------------
# 4. planar control initialisation
# ---------------------------------------------------------------------------


def test_planar_control_is_periodic_and_not_force_inflated():
    """Guard for the periodic-seam defect found in this probe's own first draft.

    A single step (liquid below, vapour above) joins rho_l to rho_v across one
    cell at the periodic boundary; the resulting max|F| was ~6x the droplet
    value at the same interface width.  The slab construction must have no seam.
    """
    from tensorlbm.multiphase import make_psi_eos, sc_single_component_force

    mw = P.maxwell_at_tr("pr", 0.8265)  # ratio ~10, cheap
    rho_slab, _ = P.planar_field(64, 64, mw["rho_liquid"], mw["rho_vapour"], torch.device("cpu"), 4.0)
    seam = float((rho_slab[0] - rho_slab[-1]).abs().max())
    assert seam < 1e-2, f"periodic seam jump {seam:.3e}"

    rho_drop, _ = P.droplet_field(64, 12.0, mw["rho_liquid"], mw["rho_vapour"], torch.device("cpu"), 4.0)
    psi = make_psi_eos("pr", mw["T"])
    fx_s, fy_s = sc_single_component_force(rho_slab, 1.0, psi, 0.0, 0.0, None, None, "standard")
    fx_d, fy_d = sc_single_component_force(rho_drop, 1.0, psi, 0.0, 0.0, None, None, "standard")
    f_slab = float(torch.sqrt(fx_s**2 + fy_s**2).max())
    f_drop = float(torch.sqrt(fx_d**2 + fy_d**2).max())
    assert f_slab < 2.0 * f_drop, (f_slab, f_drop)

    liquid, vapour = P.planar_regions(64, 4.0)
    assert float(rho_slab[liquid].mean()) > 5.0 * float(rho_slab[vapour].mean())


# ---------------------------------------------------------------------------
# 5. measurement + honest-fail contract
# ---------------------------------------------------------------------------


def test_stable_tier_smoke_reports_all_measurement_columns():
    """A short march below the stability edge returns a full measurement record."""
    mw = P.maxwell_at_ratio(10.0, eos="pr")
    spec = P.RunSpec(
        tag="pytest_smoke_ratio10",
        eos="pr",
        target_ratio=10.0,
        L=48,
        R=12.0,
        steps=1200,
        sample=100,
        forcing="edm",
        device="cpu",
        width=6.0,
    )
    rec = P.run_probe(spec, mw)
    assert rec["stable"] is True, rec["error"]
    assert rec["steps_run"] == 1200
    for key in ("ratio", "ratio_error", "ma_max", "u_max", "rho_max_field", "mass_rate_per_step"):
        assert rec[key] is not None, key
    assert rec["ratio_error"] < P.PREREG["ratio_error_max"]
    assert rec["mass_rate_per_step"] < P.PREREG["mass_drift_rate_max"]
    assert rec["rho_max_field"] < rec["rho_cross"]
    assert rec["radial_profile"], "radial density profile missing"
    prof = rec["radial_profile"]
    assert prof[0]["rho_mean"] > prof[-1]["rho_mean"]  # liquid core above vapour
    verdict = P.classify(rec)
    assert verdict["cls"] in ("PASS", "MARGINAL", "FAIL")
    assert isinstance(verdict["reason"], str)


def test_honest_fail_returns_structured_record():
    """Ratio 1000 is beyond the stability edge: collapse is a *result*, not a crash."""
    mw = P.maxwell_at_ratio(1000.0, eos="pr")
    spec = P.RunSpec(
        tag="pytest_collapse_ratio1000",
        eos="pr",
        target_ratio=1000.0,
        L=48,
        R=12.0,
        steps=400,
        sample=50,
        forcing="edm",
        device="cpu",
    )
    rec = P.run_probe(spec, mw)  # must not raise
    assert rec["stable"] is False
    assert isinstance(rec["collapse_step"], int) and rec["collapse_step"] >= 1
    assert rec["collapse_step"] <= 400
    assert rec["error"] and ("non-finite" in rec["error"] or "blow-up" in rec["error"] or "clamp" in rec["error"])
    # no derived numbers are reported for a collapsed run
    assert "ratio" not in rec or rec.get("ratio") is None
    assert P.classify(rec)["cls"] == "FAIL"


# ---------------------------------------------------------------------------
# 6. consumer anchors
# ---------------------------------------------------------------------------


def test_consumer_anchors_exist():
    cmap = json.loads((_BENCH / "consumers.json").read_text(encoding="utf-8"))
    repo = _HERE.parent
    assert len(cmap["consumers"]) >= 3
    for c in cmap["consumers"]:
        for anchor in c["anchors"]:
            path = repo / anchor["path"]
            assert path.exists(), anchor["path"]
            if anchor.get("contains"):
                text = path.read_text(encoding="utf-8", errors="ignore")
                assert anchor["contains"] in text, (anchor["path"], anchor["contains"])

# ---------------------------------------------------------------------------
# 7. viscosity (tau) lever arm contract
# ---------------------------------------------------------------------------


def _tau_row(ratio, tau, ma, err, stable=True, collapse_step=None):
    return dict(
        tag=f"synthetic_ratio{ratio:g}_tau{tau:g}",
        target_ratio=ratio,
        stable=stable,
        steps_run=20000 if stable else collapse_step,
        collapse_step=collapse_step,
        ma_max=None if not stable else ma,
        ratio_error=None if not stable else err,
        spec=dict(tau=tau, forcing="edm"),
        verdict=dict(cls="FAIL" if not stable else "MARGINAL", reason="synthetic"),
    )


def test_tau_arm_is_wired_and_predeclared(monkeypatch):
    """The tau arm must exist in the driver and its rule must be predeclared."""
    import run as R  # driver module (no side effects under __main__ guard)

    assert "tau" in R.ALL_MODES
    pre = P.PREREG["tau_arm"]
    assert pre["grid"]["tau"], "tau grid not predeclared"
    assert set(pre["grid"]["target_ratios"]) == set(P.PREREG["target_ratios"])
    assert "decision_rule" in pre and "question" in pre

    # verdict block carries the arm with the columns the consumers read
    v = R._verdict({"tiers": [], "controls": [], "tau": [_tau_row(10.0, 0.8, 0.04, 0.10)]})
    assert len(v["tau"]) == 1
    row = v["tau"][0]
    assert row["tau"] == 0.8 and row["ma_max"] == 0.04 and row["ratio_error"] == 0.10


@pytest.mark.parametrize(
    "ma,err,expected",
    [
        (0.01, 0.05, "usable_at_some_tau"),  # both gates clearable by tau
        (0.01, 0.50, "spurious_tunable_ratio_structural"),  # current tunable, ratio not
        (0.45, 0.05, "ratio_held_currents_above_pass_gate"),  # ratio held, currents not
        (0.45, 0.50, "dead_at_all_tau"),  # neither gate reachable
    ],
)
def test_tau_lever_decision_rule_branches(ma, err, expected):
    rows = [
        _tau_row(10.0, 0.6, ma * 1.2, err * 1.2),
        _tau_row(10.0, 2.0, ma, err),  # best point: defines both minima
        _tau_row(100.0, 1.0, 0.2, 1.0),  # a different ratio must not leak in
    ]
    summary = P.tau_lever_summary(rows)
    assert summary["10"]["decision"] == expected
    assert summary["10"]["ma_min"] == pytest.approx(ma)
    assert summary["10"]["ratio_error_min"] == pytest.approx(err)
    assert summary["10"]["ma_ok"] is (ma <= P.PREREG["spurious_pass_ma"])
    assert summary["10"]["ratio_ok"] is (err <= P.PREREG["ratio_error_max"])
    # every predeclared ratio is reported, even when the arm has no rows at all
    assert "1000" in summary and summary["1000"]["decision"] == "arm_not_run"


def test_tau_lever_flags_runs_that_are_not_relaxation_resolved():
    """A low-tau run that clears the Ma gate only because it is not relaxed.

    Predeclared caveat (PREREG['tau_arm']['caveat_rule']): the hold length is
    kept at the tier value, so at small tau the viscous timescale R**2/nu can
    exceed it; such a run cannot be positive evidence for a tunable Ma gate.
    """
    slow = P.stability_budget(tau=0.6)["timescales"]["viscous_droplet"]
    fast = P.stability_budget(tau=2.0)["timescales"]["viscous_droplet"]
    assert slow > fast, (slow, fast)
    assert P.stability_budget(tau=0.6)["hold_over_slowest"] < 2.0 < P.stability_budget(tau=2.0)["hold_over_slowest"]

    rows = [
        _tau_row(10.0, 0.6, 0.001, 0.40),  # tiny Ma but not relaxation-resolved
        _tau_row(10.0, 2.0, 0.20, 0.40),
    ]
    for r in rows:
        r["spec"].update(R=20.0, L=96, width=4.0)
    summary = P.tau_lever_summary(rows)["10"]
    assert summary["not_relaxation_resolved"] == [0.6]
    assert summary["ma_ok"] is True  # the raw (predeclared) gate looks cleared
    assert summary["ma_ok_relaxed"] is False
    assert summary["decision"] == "ma_pass_only_in_unrelaxed_runs"


def test_hold_length_is_justified_by_the_slowest_timescale():
    """N is not a round number: it must cover the slowest linear mode several times."""
    bud = P.stability_budget(R=20.0, tau=1.0)
    assert bud["slowest"] == "viscous_droplet"
    assert bud["hold_over_slowest"] >= 3.0, bud
    assert bud["nu"] == pytest.approx((1.0 / 3.0) * 0.5)


def test_tau_lever_ignores_collapsed_runs_as_gate_evidence():
    rows = [
        _tau_row(1000.0, 0.6, None, None, stable=False, collapse_step=17),
        _tau_row(1000.0, 2.0, None, None, stable=False, collapse_step=23),
    ]
    summary = P.tau_lever_summary(rows)
    assert summary["1000"]["n_collapsed"] == 2
    assert summary["1000"]["decision"] == "no_stable_run"
    assert summary["1000"]["ma_ok"] is False and summary["1000"]["ratio_ok"] is False


# ---------------------------------------------------------------------------
# 8. artefact traceability (stale-table gate)
# ---------------------------------------------------------------------------


def test_results_artefact_matches_tree_code_hashes():
    """A published results.json must be reproducible from *this* tree.

    The probe stamps the sha256 of probe_lib.py and run.py into the artefact.
    If either file is edited without re-running the campaign, the tables are no
    longer traceable to the code; this gate makes that state fail instead of
    being silently quoted by a downstream consumer.
    """
    import hashlib

    rj = _BENCH / "out" / "results.json"
    if not rj.exists():  # pragma: no cover - artefact is a campaign output
        pytest.skip("campaign artefact not present in this tree")
    res = json.loads(rj.read_text(encoding="utf-8"))
    env = res["environment"]
    lib_sha = hashlib.sha256((_BENCH / "probe_lib.py").read_bytes()).hexdigest()
    drv_sha = hashlib.sha256((_BENCH / "run.py").read_bytes()).hexdigest()
    assert env["probe_lib_sha256"] == lib_sha, "out/results.json is stale: re-run the campaign"
    assert res["driver_sha256"] == drv_sha, "out/results.json is stale: re-run the campaign"

    # every run record carries the code hashes it was produced with
    for arm in ("tiers", "controls", "boundary", "widths", "longhold", "schemes", "repeats", "tau"):
        for r in res.get(arm, []):
            assert r["code"]["probe_lib_sha256"] == lib_sha, (arm, r["tag"])
            assert r["code"]["driver_sha256"] == drv_sha, (arm, r["tag"])

    # split-job provenance: sources are named and cover every arm that is present
    prov = res.get("provenance")
    if prov:
        assert prov["sources"], "merged artefact without provenance sources"
        covered = set()
        for s in prov["sources"]:
            covered |= set(s["arms"])
        present = {a for a in ("tiers", "controls", "boundary", "widths", "longhold", "schemes", "repeats", "tau")
                   if res.get(a)}
        assert present <= covered, (present - covered)

    # the verdict block is a deterministic function of the arms
    import run as R
    assert res["verdict"] == R._verdict(dict(res)), "stored verdict disagrees with the driver rule"


# ---------------------------------------------------------------------------
# 9. criterion artefact + merge tool contract
# ---------------------------------------------------------------------------


def test_criterion_artefact_matches_the_artefact_it_describes():
    """CRITERION.md is generated; its decisions must equal the tau-arm summary."""
    rj = _BENCH / "out" / "results.json"
    cj = _BENCH / "CRITERION.md"
    if not (rj.exists() and cj.exists()):  # pragma: no cover
        pytest.skip("criterion artefact not present in this tree")
    res = json.loads(rj.read_text(encoding="utf-8"))
    text = cj.read_text(encoding="utf-8")
    summary = P.tau_lever_summary(res.get("tau", []))
    for ratio in res["prereg"]["target_ratios"]:
        dec = summary[f"{ratio:g}"]["decision"]
        assert dec in text, f"CRITERION.md does not report decision {dec} for ratio {ratio:g}"
    # the gates quoted in the criterion file are the predeclared ones
    assert f"{res['prereg']['ratio_error_max']:.0%}" in text


def test_merge_tool_refuses_artefacts_from_different_code_states(tmp_path):
    """A merged table must describe one code state (no silent concatenation)."""
    import merge_runs as M

    def doc(sha, tag):
        return {
            "benchmark": "eos_maxwell_droplet_stability_probe",
            "eos": "pr",
            "prereg": P.PREREG,
            "environment": {"probe_lib_sha256": sha, "machine": "m", "threads": 1},
            "driver_sha256": "d",
            "maxwell_tiers": {"10": {"found": True, "ratio": 10.0, "wall_s": hash(tag) % 7}},
            "generated_utc": "t",
            "wall_s_total": 1.0,
            "tiers": [
                {
                    "tag": tag,
                    "target_ratio": 10.0,
                    "stable": True,
                    "steps_run": 20000,
                    "collapse_step": None,
                    "ma_max": 0.01,
                    "ratio_error": 0.01,
                    "spec": {"forcing": "edm", "tau": 1.0, "width": 4.0, "threads": 1},
                    "verdict": {"cls": "PASS", "reason": "synthetic"},
                }
            ],
        }

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(doc("sha_a", "run_a")), encoding="utf-8")
    b.write_text(json.dumps(doc("sha_b", "run_b")), encoding="utf-8")
    with pytest.raises(SystemExit):
        M.merge([a, b])

    b.write_text(json.dumps(doc("sha_a", "run_b")), encoding="utf-8")
    # the per-job bisection *timing* differs by construction; it is not physics and
    # must not block a merge (only the construction values may differ -> then refuse)
    merged = M.merge([a, b])
    phys = doc("sha_a", "run_c")
    phys["maxwell_tiers"]["10"]["ratio"] = 10.5
    c = tmp_path / "c.json"
    c.write_text(json.dumps(phys), encoding="utf-8")
    with pytest.raises(SystemExit):
        M.merge([a, c])
    assert [r["tag"] for r in merged["tiers"]] == ["run_a", "run_b"]
    assert merged["provenance"]["sources"], "provenance lost on merge"
    assert merged["verdict"]["tiers_pass"] == 2

    # duplicate tags are refused (arms are append-only, one run == one record)
    with pytest.raises(SystemExit):
        M.merge([a, a])
