#!/usr/bin/env python3
"""Read-only dry run of the platform's own judge (maturity_gate.JUDGE_SCHEME) on verdict #797.

Does NOT write to the DB and does NOT perform any gate action (no spawn, no state change):
it only imports maturity_gate to reuse `judge()` / `link_original()` verbatim, so the audit
report can state what the gate *will* decide, instead of guessing.

Output: artifacts/m19633_audit/gate_predict.json
"""
from __future__ import annotations

import importlib.util
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(REPO, "artifacts", "m19633_audit")
VERDICT_SCOPE = "m19633 XVAL-LBM-ROUND2-METHOD-SELECTION-3D"

spec = importlib.util.spec_from_file_location("maturity_gate", "/root/tp-agent/maturity_gate.py")
mg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mg)

row = mg.q(f"SELECT id, level, evidence::text FROM tensor_maturity_verdicts "
           f"WHERE repo='TensorLBM' AND scope='{VERDICT_SCOPE}' AND verdict_by='board-audit'")
vid, claimed, evtxt = int(row[0][0]), int(row[0][1]), row[0][2]
ev = json.loads(evtxt)
gl, why = mg.judge(claimed, ev, "TensorLBM", 19698)
orig = mg.link_original(VERDICT_SCOPE, ev, "TensorLBM")
ost = mg.q(f"SELECT status FROM tensor_dev_missions WHERE id={orig}") if orig else []

out = {
    "verdict_id": vid,
    "claimed_level": claimed,
    "gate_level_predict": gl,
    "gate_fail_reasons": why,
    "predicted_gate_state": ("reopened + rework spawned" if gl <= 1 and ost and ost[0][0] == "succeeded"
                             else "see maturity_gate policy"),
    "linked_original_mission": orig,
    "original_mission_status": ost[0][0] if ost else None,
    "note": ("dry run only: the live gate skips audit missions that are not yet terminal, so the "
             "authoritative decision lands after mission 19698 finishes"),
    "scheme_source": "/root/tp-agent/maturity_gate.py::SCHEME + judge()",
}
os.makedirs(ART, exist_ok=True)
json.dump(out, open(os.path.join(ART, "gate_predict.json"), "w"), indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2, ensure_ascii=False))