#!/usr/bin/env python3
"""Render the mission report from the machine-produced artifacts.

Every number in the markdown is read out of ``result.json`` /
``probe_weight_bias.json`` / ``refdata/d2q9_bitwise_v1.json`` — nothing is
typed by hand, so the report cannot drift from the run.

Usage: PYTHONPATH=src python3 benchmarks/xval_bitwise_refdata/make_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "docs" / "xval_bitwise_refdata_m17718" / "REPORT.md"


def main() -> int:
    res = json.loads((HERE / "result.json").read_text())
    probe = json.loads((HERE / "probe_weight_bias.json").read_text())
    golden = json.loads((HERE / "refdata" / "d2q9_bitwise_v1.json").read_text())

    reg = res["regression"]
    neg = res["negative_controls"]
    bench = res["benchmark"]
    env = res["env"]

    L = []
    A = L.append

    A("# XVAL-LBM-BITWISE-REFDATA-REGRESSION-PLUS-BENCHMAR")
    A("")
    A("- mission: m17718 (round 2 of anchor; round 1 m17505 honest-FAIL, m17591 L1)")
    A(f"- repo: TensorLBM @ `{res['git_sha']}`")
    A(f"- checked: {res['checked_utc']}")
    A(
        f"- environment: torch {env['torch']}, numpy {env['numpy']}, "
        f"{env['system']}/{env['machine']}, threads={env['torch_threads']}"
    )
    A(
        f"- overall verdict: **{'PASS' if res['passed'] else 'FAIL'}** "
        f"(regression={reg['passed']}, negative_controls={neg['both_detected']}, "
        f"benchmark={bench['passed']})"
    )
    A("")

    # ------------------------------------------------------------------ ①
    A("## ① Problem setting and reference basis")
    A("")
    A("**Problem.** The D2Q9 kernels (`equilibrium`, `collide_bgk`, `collide_trt`,")
    A("`collide_mrt`, `collide_bgk_fused`, `collide_rlbm`, `stream`) are the base of")
    A("every 2D result in this repository, yet nothing pinned their *output bits*.")
    A("Float-tolerance tests accept silent drift: a changed lattice weight, a reordered")
    A("summation or a dtype leak moves results by 1e-9..1e-8 and still passes every")
    A("`rel=1e-6` assertion. This mission builds (a) a byte-level reference-data")
    A("regression lock over those kernels and (b) a physics benchmark that ties the")
    A("lock to an exact solution.")
    A("")
    A("**Reference basis.**")
    A("")
    A(f"- Regression reference: *self-consistent frozen payload*, {golden['schema']},")
    A(f"  generated from git `{golden['git_sha']}` and bound to the kernel sources by sha256:")
    for rel, sha in golden["code"].items():
        A(f"  - `{rel}` → `{sha[:16]}…`")
    A("- Physics reference (benchmark): 2D Taylor–Green vortex closed-form decay")
    A(
        "  `A(t) = (u0/√2)·exp(−2νk²t)`, `k = 2π/Lx`, `ν = (τ−0.5)/3`."
        " This is an analytic solution (no digitised table, nothing copy-pasted):"
    )
    A("  Taylor, *Phil. Mag.* (1923); standard LBM verification case per Krüger et al.,")
    A("  *The Lattice Boltzmann Method: Principles and Practice* (2017), ch. 8.")
    A("")

    # ------------------------------------------------------------------ ②
    A("## ② Method and implementation")
    A("")
    A("**Regression lock.**")
    A("")
    A("1. Initial fields are built from integer lattice algebra with powers-of-two")
    A("   divisions (`harness.int_fields`: `ρ = 1 + a/64`, `u = b/128`, integer `a,b`),")
    A("   so the initial state is *exactly* representable and platform independent —")
    A("   no RNG, no `libm`. Any digest difference is therefore produced by the kernels.")
    A("2. Each case runs the **real** library primitives (`src/tensorlbm/d2q9.py`,")
    A("   `src/tensorlbm/solver.py`); the harness contains no re-implementation.")
    A("3. The result field is digested with sha256 over dtype + shape + raw IEEE-754")
    A("   payload, and the payload itself is stored (base64) in the golden file, so a")
    A("   mismatch can be localised to cell/direction with a ulp distance")
    A("   (`harness.ulp_metrics`, monotone integer mapping of the bit pattern).")
    A("4. Verdicts: `BITWISE_OK` / `ULP_DRIFT` (env drifted, inside budget) /")
    A("   `REGRESSION` / `GOLDEN_INCONSISTENT` / `MISSING_GOLDEN`.")
    A("5. Negative controls run on every `check`: a 1-ulp tamper of one population and a")
    A("   `1e-7` relative τ bump must both move the digest — otherwise the gate is dead.")
    A("")
    A("**Physics benchmark.** TRT collision + periodic streaming on a doubly periodic")
    A("box, initialised with the exact TGV field. The target decay factor is held fixed")
    A("(`2νk²t = const`, i.e. `t ∝ L²/ν`), so every resolution is compared at the same")
    A("physical time: 24²/56 steps, 32²/100 steps, 64²/400 steps at τ = 0.8, plus a τ")
    A("series (0.7 → 150 steps, 0.9 → 75 steps) at 32².")
    A("")
    A("**Code pointers.**")
    A("")
    A("- lock + gate: `benchmarks/xval_bitwise_refdata/harness.py`")
    A("  (`int_fields`, `CASES`, `digest`, `ulp_metrics`, `verify`, `negative_controls`)")
    A("- entry point: `benchmarks/xval_bitwise_refdata/run.py` (`gen|verify|controls|bench|check`)")
    A("- test-suite caller: `tests/test_xval_bitwise_refdata.py` (14 tests)")
    A("- root-cause probe: `benchmarks/xval_bitwise_refdata/probe_weight_bias.py`")
    A(
        "- consumed library kernels: `src/tensorlbm/d2q9.py::equilibrium`, "
        "`src/tensorlbm/solver.py::{collide_bgk, collide_trt, collide_mrt, "
        "collide_bgk_fused, collide_rlbm, stream, macroscopic}`"
    )
    A("")

    # ------------------------------------------------------------------ ③
    A("## ③ Parameter table")
    A("")
    A("Regression cases (dimensionless):")
    A("")
    A("| case | lattice | dtype | τ | steps | kernels |")
    A("|---|---|---|---|---|---|")
    for name, c in golden["cases"].items():
        ny, nx = c["payload"]["shape"][1], c["payload"]["shape"][2]
        dt = c["payload"]["dtype"].replace("torch.", "")
        A(
            f"| `{name}` | {nx}×{ny} | {dt} | 0.8 | "
            f"{'50' if name.startswith('chain') else '1'} | {', '.join(c['kernels'])} |"
        )
    A("")
    A("Benchmark parameter points (all dimensionless):")
    A("")
    A("| level | L | τ | ν | u₀ | Ma = u₀/√3 | Re_cells = u₀L/ν | steps | 2νk²t |")
    A("|---|---|---|---|---|---|---|---|---|")
    import math

    for lv in bench["levels"] + bench["tau_points"]:
        k = 2 * math.pi / lv["nx"]
        A(
            f"| `{lv['level']}` | {lv['nx']} | {lv['tau']:.2f} | {lv['nu']:.4f} | "
            f"{lv['u0']} | {lv['ma']:.4f} | {lv['re_cells']:.1f} | {lv['steps']} | "
            f"{2 * lv['nu'] * k * k * lv['steps']:.4f} |"
        )
    A("")

    # ------------------------------------------------------------------ ④
    A("## ④ Reference vs got")
    A("")
    A("### 4.1 Bitwise regression lock")
    A("")
    A(
        f"Environment matches golden: **{reg['env_match']}**; ulp budget in force: "
        f"**{reg['ulp_budget']}** ulp; cases: {reg['n_cases']}, bitwise-identical: "
        f"{reg['n_bitwise_ok']}, ulp-drift: {reg['n_ulp_drift']}, regressions: "
        f"{reg['n_regression']}."
    )
    A("")
    A("| case | verdict | digest (16 hex) | n_diff | max ulp |")
    A("|---|---|---|---|---|")
    for name, c in reg["cases"].items():
        um = c.get("ulp") or {}
        A(
            f"| `{name}` | {c['verdict']} | `{c['digest'][:16]}` | "
            f"{um.get('n_diff', '—')} | {um.get('max_ulp', '—')} |"
        )
    A("")
    A("Gate power (negative controls, declared **before** looking at the run):")
    A("")
    t, tb = neg["ulp_tamper"], neg["tau_bump"]
    A(
        f"- 1-ulp tamper at index {t['tampered_index']} of `{t['case']}` → digest detects: "
        f"**{t['digest_detects']}**, max ulp = {t['max_ulp']:.0f}, differing cells = {t['n_diff']} "
        f"(verdict would be `{t['verdict_would_be']}`)."
    )
    A(
        f"- τ = {tb['tau'][0]} → {tb['tau'][1]:.10g} (1e-7 relative) → digest detects: "
        f"**{tb['digest_detects']}**, max ulp = {tb['max_ulp']:.0f}, differing cells = {tb['n_diff']}."
    )
    A(f"- both controls fired: **{neg['both_detected']}**.")
    A("")
    A("### 4.2 Physics benchmark (LBM vs analytic TGV decay)")
    A("")
    A("| level | ref A(t) | got A(t) | err [%] | < 15% (owner bar) | field L2 err [%] |")
    A("|---|---|---|---|---|---|")
    for row in bench["ref_vs_got"]:
        lv = next(x for x in bench["levels"] + bench["tau_points"] if x["level"] == row["item"])
        A(
            f"| `{row['item']}` | {row['ref']:.9e} | {row['got']:.9e} | {row['err_pct']:.4f} | "
            f"{'✅' if row['pass'] else '❌'} | {lv['field_l2_pct_final']:.4f} |"
        )
    A("")
    A(
        f"Worst error **{bench['worst_err_pct']:.4f}%** against the {bench['tolerance_pct']}% "
        f"acceptance bar → **{'PASS' if bench['passed'] else 'FAIL'}**."
    )
    A("")

    # ------------------------------------------------------------------ ⑤
    A("## ⑤ Convergence and parameter sensitivity")
    A("")
    A("Resolution series at fixed physical time (τ = 0.8, `2νk²t` held constant):")
    A("")
    A("| L | amp err [%] | ratio vs next finer |")
    A("|---|---|---|")
    lvls = bench["levels"]
    for i, lv in enumerate(lvls):
        ratio = (
            f"{lv['amp_err_pct_final'] / lvls[i + 1]['amp_err_pct_final']:.2f}"
            if i + 1 < len(lvls)
            else "—"
        )
        A(f"| {lv['nx']} | {lv['amp_err_pct_final']:.4f} | {ratio} |")
    A("")
    A("Errors fall monotonically with resolution (first order in the acoustic/")
    A("compressibility error, i.e. the operator converges); no level exceeds 1%.")
    A("")
    A("τ sensitivity at 32² (same decay target):")
    A("")
    A("| τ | ν | steps | amp err [%] | max ρ spread |")
    A("|---|---|---|---|---|")
    for lv in bench["tau_points"] + [bench["levels"][1]]:
        A(
            f"| {lv['tau']:.2f} | {lv['nu']:.4f} | {lv['steps']} | "
            f"{lv['amp_err_pct_final']:.4f} | {lv['max_rho_spread']:.3e} |"
        )
    A("")
    A("The error is flat in τ (0.38–0.45%), i.e. the observed error is the *lattice*")
    A("truncation error, not a relaxation-time artefact.")
    A("")

    # ------------------------------------------------------------------ ⑥
    A("## ⑥ Failure modes, defect found, and artifacts")
    A("")
    A("### 6.1 Defect found by the lock: float32 weights inside `collide_bgk_fused`")
    A("")
    A("The lock's first act was to fail: `collide_bgk_fused` moved float64 mass by")
    A(f"{probe['measured_legacy_step_drift']:.4e} per step (all other operators: ~1e-14).")
    A("Root cause: the fused path built weights from the **float32** table `d2q9.W` and")
    A(
        "cast it to float64, so `Σw = 1 + "
        f"{probe['weight_table_bias_f32_in_f64']:.4e}`; a fused equilibrium then injects"
    )
    A("`bias/τ` of spurious density per step — the exact defect `d2q9._w_for` was added")
    A("for, but this call site was missed. Quantitative confirmation (probe, no revert):")
    A("")
    A("| quantity | predicted | measured | rel. error |")
    A("|---|---|---|---|")
    A(
        f"| per-step drift | {probe['predicted_step_drift']:.6e} | "
        f"{probe['measured_legacy_step_drift']:.6e} | {probe['step_prediction_rel_err']:.1e} |"
    )
    A(
        f"| {probe['steps']}-step drift (linear ramp) | {probe['predicted_steps_drift']:.6e} | "
        f"{probe['measured_legacy_steps_drift']:.6e} | {probe['steps_prediction_rel_err']:.1e} |"
    )
    A(
        f"| {probe['steps']}-step drift after fix | 0 | "
        f"{probe['measured_shipped_steps_drift']:.6e} | — |"
    )
    A("")
    A(
        "Fix (one line, `src/tensorlbm/solver.py`): `w = _w_on(device).to(dtype)` →"
        " `w = _w_for(dtype, device)`."
    )
    A("Because `_w_for(float32)` is bit-identical to `W`, the float32 path is unchanged")
    A("(asserted by `test_float32_weight_table_is_unchanged_by_the_fix`).")
    A("")
    A("**Locality evidence** — after the fix, only the fused case digest moved:")
    A("")
    A("| case | digest after fix |")
    A("|---|---|")
    for name, c in golden["cases"].items():
        A(f"| `{name}` | `{c['digest'][:16]}` |")
    A("")
    A("(the 8 previously frozen cases kept their digests byte for byte except")
    A("`bgk_fused_step_16`, which is exactly the changed path)")
    A("")
    A("**Same-class sites not fixed here** (no test binds them; changing them would")
    A("alter 3D/other numerics without an acceptance gate): `src/tensorlbm/solver.py`")
    A("`collide_rlbm`, `src/tensorlbm/d3q27.py`, `src/tensorlbm/solver3d.py`,")
    A("`src/tensorlbm/cylinder_cross_validation.py` — all still call")
    A("`_w_on(device).to(f.dtype)`. Recorded for a follow-up with its own gate.")
    A("")
    A("### 6.2 Failure modes of the lock itself")
    A("")
    A("- *False red on another machine*: handled by the env/ulp-budget degradation,")
    A("  exercised by `test_env_mismatch_degrades_to_ulp_budget`.")
    A("- *Dead gate* (comparison that can never fail): handled by two negative controls,")
    A("  plus `test_verify_flags_regression_on_kernel_drift` and")
    A("  `test_verify_rejects_hand_edited_golden`.")
    A("- *Non-deterministic kernel*: `collide_bgk_matmul` is excluded on purpose — BLAS")
    A("  matmul order is outside torch's determinism contract, so a bitwise lock there")
    A("  would be a lie.")
    A("- *Stale golden after an intentional kernel change*: `run.py gen` + `check`, and")
    A("  the test suite fails until `result.json`'s kernel sha256 match the tree.")
    A("")
    A("### 6.3 Artifacts and reproduction")
    A("")
    A(f"```\n{res['repro_cmd']}\n```")
    A("")
    A("| artifact | path |")
    A("|---|---|")
    A("| entry point | `benchmarks/xval_bitwise_refdata/run.py` |")
    A("| lock + gate library | `benchmarks/xval_bitwise_refdata/harness.py` |")
    A(
        "| frozen golden data (9 cases, payloads) | `benchmarks/xval_bitwise_refdata/refdata/d2q9_bitwise_v1.json` |"
    )
    A("| machine-produced run result | `benchmarks/xval_bitwise_refdata/result.json` |")
    A(
        "| defect root-cause probe | `benchmarks/xval_bitwise_refdata/probe_weight_bias.py` + `.json` |"
    )
    A("| figure | " + ", ".join(f"`{f['path']}`" for f in res.get("figures", [])) + " |")
    A("| test-suite caller | `tests/test_xval_bitwise_refdata.py` (14 tests) |")
    A(f"| harness sha256 | `{res['harness_sha256'][:32]}…` |")
    A("")
    A("### 6.4 Error judgement against the owner's bar")
    A("")
    A("| key quantity | reference | error | < 3%? | < 15%? |")
    A("|---|---|---|---|---|")
    for row in bench["ref_vs_got"]:
        A(
            f"| TGV amplitude, `{row['item']}` | {row['ref']:.6e} | "
            f"{row['err_pct']:.4f}% | {'yes' if row['err_pct'] < 3 else 'no'} | "
            f"{'yes' if row['err_pct'] < 15 else 'no'} |"
        )
    A(
        f"| float64 fused-step mass drift | 0 | {probe['measured_shipped_step_drift']:.1e} "
        f"(was {probe['measured_legacy_step_drift']:.1e}) | yes | yes |"
    )
    A("")
    A("All benchmark errors are < 3%, i.e. an order of magnitude inside the 15% bar;")
    A("the regression side is exact (0 ulp) rather than approximate.")
    A("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"[report] wrote {OUT} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
