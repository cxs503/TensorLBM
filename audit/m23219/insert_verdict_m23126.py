#!/usr/bin/env python3
"""m23219 board-audit — land the maturity verdict for m23126 (TensorLBM).

Reads ``verdict_evidence_m23126.json`` (written next to this script), inserts one
row into ``tensor_maturity_verdicts`` and prints the gate adjudication that the
platform's maturity_gate writes back (gate_state / gate_level / gate_note).

Idempotent: re-running deletes the previous board-audit row for the same
(scope, mission_id) before inserting, so a re-audit cannot stack duplicates.

Run:  python3 audit/m23219/insert_verdict_m23126.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = "tensorplatform"
REPO = "TensorLBM"
SCOPE = "m23126 XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD"
MISSION_ID = 23219          # this audit task
AUDITED_MISSION = 23126     # mission under review
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
    ev = json.loads((HERE / "verdict_evidence_m23126.json").read_text())
    ev["audited_mission"] = AUDITED_MISSION
    level = int(ev["level"])
    payload = json.dumps(ev, ensure_ascii=False, default=float).replace("'", "''")
    psql(
        "DELETE FROM tensor_maturity_verdicts "
        f"WHERE verdict_by='{VERDICT_BY}' AND mission_id={MISSION_ID} AND scope='{SCOPE}';"
    )
    psql(
        "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
        f"VALUES ('{REPO}', '{SCOPE}', {MISSION_ID}, {level}, '{payload}'::jsonb, '{VERDICT_BY}');"
    )
    q = (
        "SELECT id||' | '||repo||' | '||scope||' | mission='||mission_id||' | level='||level||"
        "' | by='||verdict_by||' | gate_state='||coalesce(gate_state,'(pending)')||"
        "' | gate_level='||coalesce(gate_level::text,'(pending)')||' | note='||"
        "coalesce(gate_note::text,'(pending)')||' | created='||created_at "
        "FROM tensor_maturity_verdicts "
        f"WHERE verdict_by='{VERDICT_BY}' AND mission_id={MISSION_ID} AND scope='{SCOPE}' "
        "ORDER BY id DESC LIMIT 1;"
    )
    print(psql(q))


if __name__ == "__main__":
    main()