#!/usr/bin/env python3
"""Build the board-audit artifacts for mission m18101.

Reads only committed/rerun numbers (no hand-copied values):
  * the audited mission's own summary.json (committed artifact)
  * the auditor's independent rerun of the slab arm (artifacts/m18101_rerun_slab_cases.json)
  * the auditor's independent rerun of the analytic arms, stored inline below after
    being printed by run_xval.arm_* in the auditor session.

Writes:
  artifacts/m18101_audit_evidence.json   -> the exact JSON inserted into tensor_maturity_verdicts
  docs/audit_m18101/fig_reproduction.png -> committed vs reproduced parity + invariant margins
  verdict_m18151.sql                     -> idempotent INSERT
  docs/AUDIT_m18101_VERDICT.md           -> review record
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = pathlib.Path("/root/tp-agent/work/m18101")
XV = SRC / "xval/XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE"
ANCHOR = "XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE"
SCOPE = f"m18101 {ANCHOR}"
AUDIT_MID = 18151
TOL = 15.0

committed = json.loads((XV / "artifacts/summary.json").read_text())
rerun_slab = json.loads((ROOT / "artifacts/m18101_rerun_slab_cases.json").read_text())

# auditor rerun of the analytic arms (run_xval.arm_* , same interpreter, PYTHONPATH=work/m18101/src)
rerun = {
    "modes_axis": 0.09968367713501558,
    "modes_diag": 0.07747629793383312,
    "refine_order": 1.9991225659715155,
    "momentum_std": 1.8289300537817463e-17,
    "momentum_edm": 1.8320223632233457e-19,
    "oss_identity": 7.450580655744572e-09,
    "uniform_nil": 0.0,
    "symmetry_rot90": 7.329216021954404e-16,
    "symmetry_transpose": 7.057770047308658e-16,
    "mutation_min": 0.12451637816845536,
}

c_cases = committed["C_slab"]["cases"]
pairs = []
for c, r in zip(c_cases, rerun_slab):
    assert (c["n"], c["G12"]) == (r["n"], r["G12"])
    pairs.append(
        {
            "case": f"n={c['n']} G12={c['G12']}",
            "std_committed_pct": 100 * c["standard"]["err_vs_spectral"],
            "std_rerun_pct": 100 * r["standard"]["err_vs_spectral"],
            "edm_committed_pct": 100 * c["edm"]["err_vs_spectral"],
            "edm_rerun_pct": 100 * r["edm"]["err_vs_spectral"],
            "steps_committed": c["steps_run"],
            "steps_rerun": r["steps_run"],
            "converged_committed": c["converged"],
            "converged_rerun": r["converged"],
        }
    )

a = committed["A_mode_sweep"]
b = committed["B_refinement"]
d = committed["D_momentum"]
e = committed["E_invariants"]
f = committed["F_mutation"]
v = committed["verdicts"]

# ---------- figure ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))

labels = [p["case"] + "\nstandard" for p in pairs] + [p["case"] + "\nedm" for p in pairs]
cm = [p["std_committed_pct"] for p in pairs] + [p["edm_committed_pct"] for p in pairs]
rr = [p["std_rerun_pct"] for p in pairs] + [p["edm_rerun_pct"] for p in pairs]
labels += ["mode axis\n(>=8 cell)", "mode diag\n(>=8 cell)"]
cm += [100 * a["worst_err_axis_resolved"], 100 * a["worst_err_diagonal_resolved"]]
rr += [100 * rerun["modes_axis"], 100 * rerun["modes_diag"]]
y = np.arange(len(labels))[::-1]
ax1.barh(y + 0.18, cm, height=0.34, color="#9aa5b1", label="committed summary.json")
ax1.barh(y - 0.18, rr, height=0.34, color="#2b6cb0", label="auditor rerun")
ax1.axvline(TOL, color="#c53030", ls="--", lw=1.2, label="15% acceptance")
for yi, val in zip(y, rr):
    ax1.text(val + 0.25, yi - 0.18, f"{val:.2f}%", va="center", fontsize=7)
ax1.set_yticks(y)
ax1.set_yticklabels(labels, fontsize=7.5)
ax1.set_xlabel("relative error vs reference [%]")
ax1.set_xlim(0, 16.5)
ax1.set_title("(a) Error metrics: committed vs independently reproduced")
ax1.legend(fontsize=7.5, loc="lower right")
ax1.grid(axis="x", alpha=0.25)

res = [
    ("kernel == OSS stencil", e["oss_identity"], 1e-6),
    ("kernel == closed form S(k)", a["closed_form_identity_max"], 1e-6),
    ("D2Q9 rot90 equivariance", e["symmetry_rot90"], 1e-12),
    ("D2Q9 transpose equivariance", e["symmetry_transpose"], 1e-12),
    ("net momentum std", d["standard"]["defect_rel"], 1e-13),
    ("net momentum edm", d["edm"]["defect_rel"], 1e-13),
    ("uniform field force", 1e-20, 1e-14),  # exact 0, floored for log axis
]
names = [r[0] for r in res]
vals = np.array([max(r[1], 1e-20) for r in res])
lims = np.array([r[2] for r in res])
yy = np.arange(len(res))[::-1]
ax2.barh(yy + 0.18, vals, height=0.34, color="#2b6cb0", label="measured residual")
ax2.barh(yy - 0.18, lims, height=0.34, color="#e2b007", label="predeclared limit")
for yi, (nm, val) in zip(yy, zip(names, vals)):
    txt = "exact 0" if val <= 1e-20 else f"{val:.2e}"
    ax2.text(val * 1.6, yi + 0.18, txt, va="center", fontsize=7)
ax2.set_xscale("log")
ax2.set_yticks(yy)
ax2.set_yticklabels(names, fontsize=8)
ax2.set_xlabel("relative residual (log scale; exact zero floored at 1e-20)")
ax2.set_title("(b) Invariants: measured residual vs predeclared limit")
ax2.legend(fontsize=7.5, loc="lower right")
ax2.grid(axis="x", alpha=0.25)
fig.tight_layout()
figpath = ROOT / "docs/audit_m18101/fig_reproduction.png"
fig.savefig(figpath, dpi=150)
plt.close(fig)

# ---------- evidence ----------
evidence = {
    "verdict": "L4 confirmed (with two non-blocking process/limitation notes)",
    "audited_mission": 18101,
    "audited_commit": "f2f53bd5 (TensorLBM, branch agent/m18101; not yet merged to /root/tensor-ecosystem/TensorLBM main)",
    "impl_files": [
        f"{SRC}/src/tensorlbm/multiphase.py",
        f"{XV}/run_xval.py",
        f"{XV}/xval_lib.py",
        f"{XV}/make_report.py",
        f"{SRC}/tests/test_xval_mcmp_cross_force.py",
    ],
    "impl_pointers": {
        "kernel": "src/tensorlbm/multiphase.py:534 sc_two_component_force (standard/edm)",
        "solver_call": "src/tensorlbm/multiphase.py:788 collide_sc_two_component calls sc_two_component_force",
        "slab_harness": "src/tensorlbm/mcmp_verification.py:670 simulate_planar_slab (time-marches collide+stream)",
        "references": "xval/.../xval_lib.py:77 oss_force (forward gather), :129 symbol_ratio S(k), :105/:116 spectral_grad/spectral_force",
    },
    "entry": f"{XV}/run_xval.py",
    "run_artifacts": [
        f"{XV}/artifacts/summary.json",
        f"{XV}/artifacts/slab_cases.jsonl",
        f"{XV}/artifacts/fields.npz",
        f"{XV}/artifacts/fig_xval.png",
        f"{XV}/REPORT.md",
        f"{ROOT}/artifacts/m18101_rerun_slab_cases.json",
        f"{ROOT}/docs/audit_m18101/fig_reproduction.png",
    ],
    "integration_callers": [
        "collide_sc_two_component: production D2Q9 MCMP path, src/tensorlbm/multiphase.py:788 -> sc_two_component_force; "
        "callers src/tensorlbm/cases/two_layer_poiseuille.py:43 and src/tensorlbm/mcmp_verification.py:670 "
        "(simulate_planar_slab -> collide_sc_two_component -> stream, 2250-4500 steps, real relaxation)",
    ],
    "benchmark": {
        "anchor": ANCHOR,
        "acceptance": "<15% (owner standard; predeclared in xval/.../predeclared.json before results)",
        "tensor_benchmark_cases_row": "NONE (SELECT ... WHERE anchor_slug=... returned 0 rows) - tolerance lives only in "
                                      "predeclared.json + tensor_xval_cases id=8 (self-declared, status settled)",
        "got_vs_ref": "slab vs spectral gradient: standard 3.654 / 3.488 / 7.012 %, edm 3.537 / 3.384 / 6.430 %; "
                      "analytic modes: axis 9.968 %, diag 7.748 %; refinement order 1.999; ALL <15% PASS",
        "invariants": {
            "oss_stencil_identity": e["oss_identity"],
            "closed_form_identity": a["closed_form_identity_max"],
            "uniform_nil": e["uniform_nil"],
            "symmetry": max(e["symmetry_rot90"], e["symmetry_transpose"]),
            "momentum_standard": d["standard"]["defect_rel"],
            "momentum_edm": d["edm"]["defect_rel"],
        },
        "falsifiability": {"gate": "6/6 planted defects >=15%", "achieved": "5/6", "min": f["min_mutation_err"],
                           "honest": True, "note": "M5 psi=sqrt(rho) 12.45% -> declared FAIL, threshold not moved"},
        "independent_reproduction": pairs,
    },
    "repro_cmd": "cd /root/tp-agent/work/m18101 && PYTHONPATH=src python -m pytest tests/test_xval_mcmp_cross_force.py -q  "
                 "-> 9 passed in 6.08s  ;  PYTHONPATH=src python xval/XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE/run_xval.py "
                 "--slab-case 64:-2.5:4000 -> standard 3.654%, edm 3.537% (auditor rerun reproduced all 3 slab cases and all 6 analytic arms)",
    "tests": [
        f"{SRC}/tests/test_xval_mcmp_cross_force.py",
        "9 passed (auditor rerun, 6.08 s): kernel==OSS, closed form, uniform nil, rot90/transpose, refinement order, "
        "momentum, mutation >=5/6, summary has >=3 slab cases",
    ],
    "level_boundary": {
        "L2": "met: real kernel + independent NumPy references + runnable driver",
        "L3": "met: exercises the production D2Q9 MCMP path (collide_sc_two_component -> sc_two_component_force), "
              "time-marched through stream() for 2250-4500 steps",
        "L4": "met: complete cases (3 relaxed slabs + 6 analytic arms) + error analysis + bit-level reproducible artifacts",
        "L5": "not met: no new solver capability; external literature anchor cited but not machine-verifiable offline "
              "(no vendored OSS source, no hash), and page-level citations deliberately omitted",
    },
    "checks_run": [
        "read mission row 18101, platform report 16985 (6 sections, 1 embedded figure), tensor_xval_cases id=8, oss_solutions id=430",
        "pytest tests/test_xval_mcmp_cross_force.py -q -> 9 passed in 6.08s (matches claim)",
        "re-ran arm_mode_sweep/arm_symbol_refinement/arm_momentum/arm_invariants/arm_mutation -> bit-identical to summary.json",
        "re-ran slab arm n=64 G=-2.5, n=96 G=-2.5, n=96 G=-3.0 on the production solver -> errors within 0.001 pp of committed",
        "checked kernel call site and production callers in src/tensorlbm/",
        "checked tensor_benchmark_cases for the anchor -> 0 rows",
        "dry-ran maturity_gate.judge(4, evidence, 'TensorLBM', 18151) -> (4, [])",
    ],
    "checks_not_run": [
        "equivalence against a compiled palabos/listLBM binary (offline; OSS stencil is a hand transcription in xval_lib.py)",
        "D3Q19/D3Q27 and wall_mode='dry'/'mirror' branches (explicitly out of scope in REPORT.md)",
        "loong64 cluster rerun (workstation CPU only)",
    ],
    "independence_limitations": [
        "OSS 'reference' is an independent NumPy transcription of the published stencil, not upstream executable code; "
        "no vendored source/hash -> the identity test proves agreement with the transcription.",
        "The stronger independent anchors are analytic: closed-form S(k) and FFT-exact gradient; these do not depend on any transcription.",
        "The C-arm errors are evaluated against the exact gradient of the *instantaneous* density field, so they measure "
        "operator truncation, not model-level surface tension - stated in REPORT.md and platform report 16985.",
    ],
    "findings": [
        {"severity": "non-blocking", "finding": "tensor_benchmark_cases has no row for this anchor (0 rows): the platform "
         "cannot machine-adjudicate the tolerance; acceptance rests on predeclared.json + tensor_xval_cases id=8.",
         "evidence": "psql SELECT case_name,metric,tolerance ... -> (0 rows)"},
        {"severity": "non-blocking", "finding": "Two of three committed slab cases were truncated at 3000 steps "
         "(converged=false), while the report's own repro_cmd uses 4000/3000/3000. Auditor rerun at 4000/4500 steps "
         "changes the errors by <=0.001 pp (G=-3.0 then converges) -> numbers are robust, metadata is inconsistent.",
         "evidence": "committed steps 2250/3000/3000 vs rerun 2250/4000/4500; std 3.488 vs 3.487 %, 7.012 vs 7.012 %"},
        {"severity": "non-blocking", "finding": "Falsifiability arm plants defects in the NumPy reference, not in the kernel; "
         "it validates the metric's sensitivity, not kernel defect detection. Claimed as such (5/6, honest FAIL).",
         "evidence": "run_xval.py:335-388 uses L.oss_force with mutated weights/directions"},
        {"severity": "none", "finding": "No overclaim found: headline 7.01/6.43 %, 9.97/7.75 %, 5/6 mutation and gate_pass=false "
         "all match artifacts bit-for-bit.", "evidence": "summary.json verdicts + auditor rerun"},
    ],
    "report_six_sections": "tensor_reports id=16985: ①..⑥ present, 1 embedded figure, explicit '<3%: NO / <15%: PASS' answer",
    "audit_independence": "auditor mission 18151; no edits to m18101 worktree; reruns executed from /tmp scratch copy against work/m18101/src",
}
(ROOT / "artifacts/m18101_audit_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=1))

# ---------- SQL ----------
sql = f"""-- board audit verdict for m18101 (auditor mission {AUDIT_MID})
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', '{SCOPE}', {AUDIT_MID}, 4, $json${json.dumps(evidence, ensure_ascii=False)}$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id,
      level = EXCLUDED.level,
      evidence = EXCLUDED.evidence,
      gate_state = 'pending',
      gate_level = NULL,
      gate_note = NULL;
"""
(ROOT / "verdict_m18151.sql").write_text(sql)
print("wrote", figpath)
print("evidence bytes", len(json.dumps(evidence, ensure_ascii=False)))
print("parity", json.dumps(pairs, indent=1))