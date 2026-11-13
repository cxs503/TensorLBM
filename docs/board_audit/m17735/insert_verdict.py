#!/usr/bin/env python3
"""m17735 落表：m17716 交差核验结论写入 tensor_maturity_verdicts（board-audit 席位）。

evidence 见同目录 evidence.py（gate 契约格式；指针全部来自本单实测，m17716 工作树 head=4846dcdd）。
基准裁判：tensor_benchmark_cases 本单前 0 行（referee row absent）→ 按先例（id=47 CATHODIC-SPHERE、
id=55 RIGID-LID「registered by audit ... because the referee row was absent」）以本单独立复跑数字补注册。
幂等：benchmark 行 NOT EXISTS 才插；verdict 按 (repo,scope,verdict_by) upsert 并重置 gate_state='pending'。
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
from evidence import LEVEL, evidence  # noqa: E402

payload = json.dumps(evidence, ensure_ascii=False)

reg_sql = (
    "INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status, passed_at) "
    "SELECT 'XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION', "
    "'waLBerla equal-level border stream correction transplant (D2Q9): F-H prolongation scale calibration + substitution operator invariants, 64^2/128^2 coarse + 2:1 fine block, TGV U0=0.04 + shear, tau_c=0.6/tau_f=0.7', "
    "'analytic Taylor-Green / shear non-equilibrium field (program-generated in analytic_fields_2d, zero hand-copy) + waLBerla documented practice (oss_solutions id=283); waLBerla body NOT executed (no network/GPL) - WITHHELD for OSS run output', "
    "'F-H prolongation scale calibration: relative L2 of reconstructed non-equilibrium vs exact non-equilibrium at optimal scale (tgv_n64_p32/tgv_n128_p64/shear_n64_p32); operator invariants: uniform-field inertness, prepare/apply round trip, mass-flux defect of substitution', "
    "'<15% (owner standard) per KPI; inertness/round-trip exact 0; measured max 0.0984% (tgv_n64_p32), 12/12 rows pass; auditor m17735 re-ran repro_cmd bit-identically and registered this row because it was absent', "
    "'TensorLBM', 'passed', now() "
    "WHERE NOT EXISTS (SELECT 1 FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION') "
    "RETURNING id"
)

verdict_sql = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES "
    "('TensorLBM', 'm17716 XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION', 17735, {lvl}, $json${ev}$json$::jsonb, 'board-audit') "
    "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, evidence=EXCLUDED.evidence, "
    "mission_id=EXCLUDED.mission_id, gate_state='pending', gate_level=NULL, gate_note=NULL "
    "RETURNING id, level, gate_state"
).format(lvl=LEVEL, ev=payload)

if __name__ == "__main__":
    sql = reg_sql + ";\n" + verdict_sql + ";\n"
    fd, path = tempfile.mkstemp(suffix=".sql", dir="/tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(sql)
    os.chmod(path, 0o644)
    r = subprocess.run(["su", "postgres", "-c", f"psql -d tensorplatform -t -A -F '|' -f {path}"],
                       capture_output=True, text=True)
    print("rc", r.returncode)
    print("stdout:", r.stdout.strip())
    print("stderr:", r.stderr.strip()[:800])
    (HERE / "db_write_output.txt").write_text(
        f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}\n", encoding="utf-8")
