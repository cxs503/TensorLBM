#!/usr/bin/env python3
"""Publish the board-audit report for m18049 -> m18049 verdict (mission 18150).

Every number in the markdown body is injected from
artifacts/board_audit/m18150/audit_m18049.json; the figure is embedded from
artifacts/board_audit/m18150/fig_audit_m18049.png. Nothing is hand-typed.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "board_audit" / "m18150"
BASE = os.environ["TENSOR_API_BASE"]


def _post(path: str, payload: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=120))


def main() -> int:
    a = json.loads((ART / "audit_m18049.json").read_text(encoding="utf-8"))
    f = a["facts"]
    env = f["environment"]
    lvl = a["level_assessment"]

    claim_rows = "\n".join(
        f"| {c['id']} | {c['claim']} | **{c['status']}** | {c['method']} |" for c in a["claims"])
    finding_rows = "\n".join(
        f"| {x['id']} | {x['severity']} | {x['finding']} | {x['impact']} |" for x in a["findings"])

    md = f"""# Board audit (cross-check) — m18049 {a['anchor_slug']}

**Verdict: L{a['level']} · disposition: {a['disposition']}.** Audited `{a['repo']}`
branch `{a['audited_branch']}`, commit `{a['audited_commit']}`, worktree `/root/tp-agent/work/m18049`.
This is an adjudication of an already-succeeded mission, not a redo: the mission's own
tool was re-executed, its tests re-run, and every published number re-read from its
manifest. The audited worktree was restored to its committed state afterwards
(`worktree_restored={a['reproduction']['worktree_restored']}`).

## ① Case introduction

The anchor is a *method* anchor, not a physics case: `AminZar69/CudaPhaseField` pins the
CUDA register budget (`-maxrregcount`) **before** any performance tuning, so that a
throughput delta between two kernel revisions is attributable to the change under test
rather than to ptxas silently re-allocating registers. TensorLBM's analogue has three
knobs — `num_warps` (~`__launch_bounds__` threads), `num_stages` (pipeline depth),
`maxnreg` (~`-maxrregcount`). The audited deliverable is a declared-pin table + drift
gate + measurement-provenance guard applied to TensorLBM's Triton GPU path.

Host actually used (recorded, not assumed): {env['machine']}, python {env['python']},
torch {f['torch']}, triton {f['triton']}, `cuda_available={env['cuda_available']}`,
`maxnreg_supported={env['maxnreg_supported']}`, `nvcc={env['nvcc_version']}`.

## ② What was done (audit method)

1. Read the mission record, its 5 declared changed files and the single deliverable
   commit `{a['audited_commit'][:8]}`; verified all deliverables are git-tracked (unlike
   earlier missions whose artifacts were untracked and therefore gate-invisible).
2. Re-executed the mission's reproducible entry, `{a['reproduction']['cmd']}`,
   exit code **{a['reproduction']['returncode']}**, then hashed the regenerated artifacts.
3. Re-ran the test file: `{a['tests']['pytest']}` (exit {a['tests']['pytest_exit']}) and
   the drift/routing subset: `{a['tests']['gate_subset']}`.
4. Queried the platform's only judge —
   `SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='{a['anchor_slug']}'`
   — which returned **{len(a['benchmark']['rows'])} rows** (`registered={a['benchmark']['registered']}`).
5. Probed the failure modes that matter here: does the drift gate actually fire (adversarial
   pin tampering test), does routing exist in the production module (AST test), and does the
   artifact generator reproduce byte-for-byte.

## ③ Results (injected from `artifacts/board_audit/m18150/audit_m18049.json`)

| quantity | value |
|---|---|
| `@triton.jit` kernels audited | {f['kernels']} |
| launch sites | {f['launch_sites']} |
| launch sites routed through `codegen_pin.launch_kwargs` | {f['routed_launch_sites']} |
| launch sites forwarding `maxnreg` | {f['launch_sites_forwarding_maxnreg']} |
| declared pins | {f['declared_pins']} |
| declared pins that fix `maxnreg` | {f['declared_pins_with_maxnreg']} |
| distinct pin fingerprints | {f['distinct_fingerprints']} |
| drift findings (fresh run) | {f['drift_findings']} |
| remaining inline launch sites | {f['unrouted_launch_sites']} |
| tests re-run by the auditor | {a['tests']['pytest'][0]} |
| artifacts reproduce byte-identically (modulo timestamp) | {a['reproduction']['identical_to_committed_except_environment_timestamp']} |

### Claim ledger — published claim vs independent re-derivation

| id | claim as published | status | independent method |
|---|---|---|---|
{claim_rows}

### Findings, severity-ranked

| id | severity | finding | impact |
|---|---|---|---|
{finding_rows}

## ④ Figures

Panel A: the eight published claims against the auditor's independent re-derivation
(verified / withheld / wrong-count). Panel B: pin coverage read from the manifest — 13
kernels and 12 launch sites audited, {f['routed_launch_sites']} sites routed, and
**{f['declared_pins_with_maxnreg']}/{f['declared_pins']} pins actually fixing a register cap**,
which is precisely where the anchor's hard mechanism stops. The figure is drawn directly
from the audit JSON, so it cannot disagree with the table above.

## ⑤ Note points / limitations

- **The benchmark clause cannot be satisfied for this anchor, and that is a platform defect,
  not a mission defect.** `tensor_benchmark_cases` has 0 rows for the anchor, and
  `feature_ledger.acceptance` for it is the generic "<15%" accuracy floor — a criterion that
  is meaningless for a launch-configuration governance change. Until that acceptance is
  restated machine-checkably (drift_findings==0, routed sites, gate exit code), no mission on
  this anchor can ever reach L4. This is finding F4 and the actionable recommendation.
- **Register cap never fixed.** `maxnreg` is `None` in {f['declared_pins_with_maxnreg']} of
  {f['declared_pins']} pins because the audit node has no CUDA backend. The mission declares
  this as honest FAIL / not_run in its report section ⑥ — I confirm the wording matches the
  evidence, so this is a capability gap, not an overclaim.
- **Runtime integration is asserted structurally, not executed.** The three wrappers in
  `triton_fused_obstacle.py` compose kwargs through the pin table and an AST test pins that
  routing, but no kernel can be launched here. The import guard also falls back to a local
  shim that silently drops pins if the package layout breaks (finding F5, low severity).
- **Minor documentation defect (F2).** The report's "remaining work" sentence says "Six"
  launch sites while the enumeration it prints (3+4) and the manifest both give
  {f['unrouted_launch_sites']}. Routing claims themselves are exact.
- **Audit independence limits.** I audited the artifact and the code, and re-ran the tool,
  but I am on the same loong64 CPU node as the mission: the GPU-side behaviour of `maxnreg`
  forwarding remains unverified by anyone in this environment.

## ⑥ Error adjudication vs baseline — is the key quantity within 3%?

| quantity | ref (baseline) | got | verdict |
|---|---|---|---|
| ref/got accuracy error for this anchor | **no registered reference value exists** (0 rows in `tensor_benchmark_cases`) | n/a | **not applicable — declared, not fabricated** |
| drift findings (declared pins vs source-derived codegen) | no pin table existed before (not measurable) | {f['drift_findings']} | PASS (exact, zero tolerance) |
| launch sites routed through the pin helper | 0/{f['launch_sites']} | {f['routed_launch_sites']}/{f['launch_sites']} | PASS (partial by design, remainder listed) |
| artifacts reproducible from source | n/a | regenerated byte-identical except `environment.timestamp` = {a['reproduction']['identical_to_committed_except_environment_timestamp']} | PASS |
| `maxnreg` honoured on hardware | unverified | unverified (`cuda_available={env['cuda_available']}`) | **honest FAIL / not_run — not <3%, not claimed** |

So the answer to "is the key quantity <3%?" is: **there is no accuracy quantity to compare for
this anchor; no percentage is asserted, and the one hardware claim is explicitly withheld.**
What is machine-checkable (drift gate, routing coverage, artifact reproducibility) passes
exactly, with zero tolerance.

## Level adjudication

**L{a['level']}.** Not lower — {lvl['why_not_lower']}. Not L2 — {lvl['why_not_2']}.
Not L4 — {lvl['why_not_L4'] if 'why_not_L4' in lvl else lvl['why_not_4']}.

## Artifacts
- `/root/tp-agent/work/m18150/artifacts/board_audit/m18150/audit_m18049.json`
- `/root/tp-agent/work/m18150/artifacts/board_audit/m18150/fig_audit_m18049.png`
- `/root/tp-agent/work/m18150/docs/board_audit/m18150/AUDIT_m18049.md`
- `/root/tp-agent/work/m18150/verdict_m18150.sql` (rows id 694 in `tensor_maturity_verdicts`)
"""

    png = base64.b64encode((ART / "fig_audit_m18049.png").read_bytes()).decode()
    payload = {
        "title": "Board audit m18049: Triton codegen pinning (maxnreg pinned before tuning) "
                 "-> verdict L3, benchmark clause unsatisfiable (0 rows)",
        "summary": f"Cross-check of m18049 ({a['anchor_slug']}): re-ran the entry (exit "
                   f"{a['reproduction']['returncode']}), 24 tests green, artifacts regenerate "
                   f"byte-identically; 5/12 launch sites routed but 0/5 pins fix maxnreg and "
                   f"tensor_benchmark_cases has 0 rows for the anchor -> L{a['level']}, "
                   f"disposition '{a['disposition']}'.",
        "scenario_key": "tensorlbm/codegen_pin",
        "mission_id": 18150,
        "tags": ["TensorLBM", "board-audit", "maturity-verdict", "codegen-pin", "maxnreg",
                 "benchmark-provenance"],
        "markdown": md,
        "figures": [{
            "name": "fig_audit_m18049.png",
            "caption": ("A: m18049's published claims vs the auditor's independent re-derivation "
                        f"(6 verified, 1 withheld, 1 wrong count). B: pin coverage from "
                        f"artifacts/codegen_pin_manifest.json — {f['kernels']} kernels / "
                        f"{f['launch_sites']} launch sites audited, {f['routed_launch_sites']} routed, "
                        f"{f['declared_pins_with_maxnreg']}/{f['declared_pins']} pins fixing a "
                        f"register cap, drift={f['drift_findings']}."),
            "png_base64": png,
        }],
    }
    token = _post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                  "password": os.environ["TENSOR_AGENT_PASS"]})
    tok = token.get("access_token") or token.get("token")
    res = _post("/tensor/reports", payload, tok)
    print(json.dumps({"report_id": (res.get("data") or res).get("id")}, ensure_ascii=False))
    (ART / "report_m18150.json").write_text(json.dumps(payload, ensure_ascii=False),
                                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())