"""m20976 board audit: independent re-run of the m20893 R2 ladder rungs.

Imports the *delivered* resolution_study.py verbatim (extracted from commit
578410df6, branch agent/m20893) and re-runs selected rungs on this box, so the
recorded KPI maxima can be compared with a fresh execution rather than with the
report text.

Checks
  C1  a=16, T<=4.0  full window  -> report 28.16 (Z_cell) / 27.39 (Z_sub.05)
  C2  a=16, T<=1.5  R1 window    -> R1 frozen artefact 22.3 %
  C3  a=32, T<=1.5  R1 window    -> R1 frozen artefact 13.806 %
  C4  a=48, T<=4.0               -> artefact r2_cj3032_maxima 28.05 (report
                                    table 3.1 prints 79.37, which is the OSS
                                    port row -> transcription check)
  C5  a=64, T<=4.0               -> artefact 42.86 (non-monotone claim)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "repro"))
sys.path.insert(0, str(HERE.parent / "src"))

import resolution_study as R  # noqa: E402

OUT = HERE / "evidence"
OUT.mkdir(parents=True, exist_ok=True)

ART = json.loads((HERE / "repro" / "r2_cj3032_maxima.json").read_text())
ART_ROWS = {r["label"]: r for r in ART["rows"]}

CASES = [
    ("C1", 16, 4.0, "tensorlbm_koerner_a16_tau1"),
    ("C2", 16, 1.5, "tensorlbm_koerner_a16_tau1"),
    ("C3", 32, 1.5, "tensorlbm_koerner_a32_tau1"),
    ("C4", 48, 4.0, "tensorlbm_koerner_a48_tau1"),
    ("C5", 64, 4.0, "tensorlbm_koerner_a64_tau1"),
]


def run(check, a, t_max, label):
    t0 = time.time()
    res = R.run_tensorlbm(a, 1.0, 5e-5, t_max, 0.25)
    res = R.eval_run(R.eval_run(res, "mm"), "ritter")
    row = {
        "check": check,
        "a": a,
        "t_max_T": t_max,
        "artifact_label": label,
        "steps": res["steps"],
        "u_max": round(res["u_max"], 4),
        "inventory_drift_max_pct": res.get("inventory_drift_max_pct"),
        "Z_cell_max_err_pct": res["vs_mm"]["Z_cell"]["max_abs_rel_err_pct"],
        "Z_sub05_max_err_pct": res["vs_mm"]["Z_sub0.05"]["max_abs_rel_err_pct"],
        "Z_sub05_ritter_pct": res["vs_ritter"]["Z_sub0.05"]["max_abs_rel_err_pct"],
        "artifact_Z_cell": ART_ROWS[label]["Z_cell_err_pct"],
        "artifact_Z_sub05": ART_ROWS[label]["Z_sub05_err_pct"],
        "artifact_u_max": ART_ROWS[label]["u_max"],
        "artifact_drift_pct": ART_ROWS[label]["inventory_drift_pct"],
        "wall_s_audit": round(time.time() - t0, 2),
        "artifact_wall_s": ART_ROWS[label]["wall_s"],
        "T_at_last_sample": res["series"][-1]["T"] if res.get("series") else None,
        "n_samples": len(res.get("series") or []),
    }
    print(json.dumps(row, indent=1), flush=True)
    return row


rows = [run(*c) for c in CASES]
(OUT / "repro_check.json").write_text(json.dumps(
    {"source": "audit_m20976/verify_r2.py, delivered resolution_study.py from "
               "commit 578410df6 (branch agent/m20893)",
     "host": "loongarch64 local worktree, torch CPU, torch.set_num_threads(1)",
     "rows": rows}, indent=1))
print("WROTE", OUT / "repro_check.json")