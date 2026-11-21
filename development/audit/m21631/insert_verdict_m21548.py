#!/usr/bin/env python3
"""m21631 board-audit — land the maturity verdict for m21548 (TensorLBM, CASE-BACKSTEP-FVM-RE100).

Reads ``verdict_evidence_m21548.json`` (next to this script), writes exactly one row into
``tensor_maturity_verdicts`` and prints back the adjudication the platform's maturity_gate
writes asynchronously (gate_state / gate_level / gate_note).

Idempotent: any previous board-audit row for the same (scope, mission_id) is deleted first,
so a re-audit cannot stack duplicates.

Run:  python3 audit/m21631/insert_verdict_m21548.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = "tensorplatform"
REPO = "TensorLBM"
SCOPE = "m21548 CASE-BACKSTEP-FVM-RE100"
MISSION_ID = 21631
LEVEL = 1
VERDICT_BY = "board-audit"


def psql(sql: str) -> str:
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    r = subprocess.run(
        ["runuser", "-u", "postgres", "--", "psql", "-d", DB, "-tA", "-F", "|", "-c", sql],
        capture_output=True, text=True, env=env,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:600])
    return r.stdout


def main() -> None:
    ev = json.loads((HERE / "verdict_evidence_m21548.json").read_text())
    assert ev.get("level") == LEVEL, f"evidence says level={ev.get('level')} but script writes {LEVEL}"
    payload = json.dumps(ev, ensure_ascii=False, default=float).replace("'", "''")
    psql("DELETE FROM tensor_maturity_verdicts "
         f"WHERE verdict_by='{VERDICT_BY}' AND mission_id={MISSION_ID} AND scope='{SCOPE}';")
    psql("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
         f"VALUES ('{REPO}', '{SCOPE}', {MISSION_ID}, {LEVEL}, '{payload}'::jsonb, '{VERDICT_BY}');")
    print(psql(
        "SELECT id||' | '||repo||' | '||scope||' | mission_id='||mission_id||' | level='||level||"
        "' | by='||verdict_by||' | gate_state='||coalesce(gate_state,'(pending)')||"
        "' | gate_level='||coalesce(gate_level::text,'(pending)')||"
        "' | created='||created_at "
        "FROM tensor_maturity_verdicts "
        f"WHERE verdict_by='{VERDICT_BY}' AND mission_id={MISSION_ID} AND scope='{SCOPE}' "
        "ORDER BY id DESC LIMIT 1;"))


if __name__ == "__main__":
    main()