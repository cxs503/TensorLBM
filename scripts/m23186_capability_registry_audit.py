#!/usr/bin/env python
"""Audit the capability registry against code, tests, evidence and prose.

Findings (each row carries a disposition):

``OVERCLAIM_WIRING``      wired=True but no non-test module imports the entry
``OVERCLAIM_TEST``        tested=True but the cited test file never mentions the entry
``OVERCLAIM_VERIFIED``    verified=True without an ANALYTIC/XVALUE grade + error band
``EVIDENCE_ARTIFACT_DRIFT`` declared max_error_pct disagrees with the run artifact
``DOC_CLAIM_CONFLICT``    prose claims production/validated for a non-mature capability
``REGISTRY_PROBLEM``      any error/warning emitted by the static scan

Usage: PYTHONPATH=src python scripts/m23186_capability_registry_audit.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "tensorlbm"))
sys.path.insert(0, str(ROOT / "src"))

import registry as reg  # noqa: E402

PKG = ROOT / "src" / "tensorlbm"
ARTIFACT = ROOT / "docs" / "m23186" / "collision_family_verification.json"

DISPOSITION = {
    "OVERCLAIM_WIRING": "改码（接线）或改标 wired=False 后在案",
    "OVERCLAIM_TEST": "改码（补测试）或改标 tested=False",
    "OVERCLAIM_VERIFIED": "改标（verified=False）或补 ANALYTIC/XVALUE 误差带",
    "EVIDENCE_ARTIFACT_DRIFT": "改文（把 evidence 对齐实测工件）",
    "DOC_CLAIM_CONFLICT": "改文（README/文档改口径）并在案",
    "REGISTRY_PROBLEM": "改码或改文，见 detail",
}

#: Findings adjudicated by mission m23186 as correct-as-is (在案), with reasons.
#: They stay in the report so the next auditor sees the decision, not a silence.
ACCEPTED = {
    ("DOC_CLAIM_CONFLICT", "lbm.collision.mrt.d3q27",
     "d3q27_composition_evidence_r1.md"):
        "该行说明的是 D3Q27 composition 的 *范围*（R1 已验证 D3Q19/MRT 组合），"
        "按行触发扫描器；registry 的 d3q27 MRT 标 developing 是有意的（无 D3Q27 解析误差带），不改文",
    ("DOC_CLAIM_CONFLICT", "lbm.collision.mrt.d2q9",
     "d3q27_composition_evidence_r1.md"):
        "同上：跨句上下文误判，不构成对 D2Q9 MRT 的成熟度断言",
    ("DOC_CLAIM_CONFLICT", "lbm.amr.reflux_restrict",
     "d3q27_composition_evidence_r1.md"):
        "同一 composition 范围表述的按行触发；README 的生产级措辞已改文",
    ("DOC_CLAIM_CONFLICT", "lbm.amr.reflux_restrict",
     "report_m2463.md"):
        "历史报告归档件按纪律不改写；README 的生产级措辞已改文（见本次改动）",
}

# prose keywords -> capability ids they would be claiming something about
CLAIM_KEYWORDS = {
    r"\bMRT\b": ["lbm.collision.mrt.d3q19", "lbm.collision.mrt.d3q27", "lbm.collision.mrt.d2q9"],
    r"\bKBC\b": ["lbm.collision.kbc.d3q19", "lbm.collision.kbc.d3q27"],
    r"cascaded|级联|中心矩": ["lbm.collision.cascaded.d3q19", "lbm.collision.cascaded.d3q27",
                          "lbm.collision.cascaded.d3q27.second_order"],
    r"\bFWH\b": ["lbm.acoustics.fwh_farfield"],
    r"\bAMR\b|自适应网格": ["lbm.amr.reflux_restrict"],
    r"[Tt]riton": ["lbm.backend.triton.suboff_step"],
    r"Shan-Chen|多相": ["lbm.multiphase.shan_chen_force.d3q19"],
}
PRODUCTION_WORDS = re.compile(
    r"生产可用|production[ -]ready|validated|已验证|14 of 14|全部可用|fully supported", re.I)


def importers(entry: str, own_module_file: str) -> list[str]:
    """Non-test modules under src/ that reference ``entry`` (excl. contracts/audits)."""
    if not entry:
        return []
    out = subprocess.run(
        ["grep", "-rl", "-E", rf"\b{re.escape(entry)}\b", str(PKG)],
        capture_output=True, text=True)
    hits = []
    for line in out.stdout.splitlines():
        path = Path(line)
        if str(path) == own_module_file:
            continue
        name = path.name
        if "contract" in name or "audit" in name or name == "registry.py":
            continue
        hits.append(str(path.relative_to(ROOT)))
    return sorted(hits)


def main() -> int:
    caps, problems = reg.scan_static(PKG, repo_root=ROOT)
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8")) if ARTIFACT.is_file() else {}
    measured = {f"lbm.collision.{f['collision']}.d3q19": f["max_error_pct"]
                for f in artifact.get("families", [])}

    rows: list[dict] = []

    def add(code: str, cid: str | None, detail: str, doc: str | None = None) -> None:
        doc_name = (doc or "").split(":")[0]
        reason = ACCEPTED.get((code, cid, doc_name))
        rows.append({"code": code, "capability_id": cid, "detail": detail,
                     "doc": doc,
                     "disposition": ("在案（已裁定：" + reason + "）") if reason
                     else DISPOSITION.get(code, "在案"),
                     "accepted": bool(reason)})

    for prob in problems:
        add("REGISTRY_PROBLEM", prob.get("capability_id"),
            f"[{prob['severity']}] {prob['code']}: {prob['detail']}")

    for cap in caps:
        own = reg._module_file(PKG, cap.module_path, "tensorlbm")
        own_str = str(own) if own else ""
        if cap.wired:
            users = importers(cap.entry or "", own_str)
            if not users:
                add("OVERCLAIM_WIRING", cap.id,
                    f"wired=True 但 src/ 内无生产模块引用 {cap.entry}（仅被契约/审计/测试引用）")
        if cap.tested:
            tfile = ROOT / (cap.test_path or "")
            if not cap.test_path or not tfile.is_file():
                add("OVERCLAIM_TEST", cap.id, f"tested=True 但 test_path 不存在: {cap.test_path}")
            else:
                text = tfile.read_text(encoding="utf-8", errors="ignore")
                token = cap.entry or cap.module_path.rsplit(".", 1)[-1]
                if token not in text and cap.id not in text:
                    add("OVERCLAIM_TEST", cap.id,
                        f"tested=True 但 {cap.test_path} 未提及 {token}")
        if cap.verified:
            grade = str(cap.evidence.get("grade", ""))
            if grade not in reg.VERIFYING_GRADES:
                add("OVERCLAIM_VERIFIED", cap.id, f"verified=True 但 grade={grade!r}")
            elif not isinstance(cap.evidence.get("max_error_pct"), (int, float)):
                add("OVERCLAIM_VERIFIED", cap.id, "verified=True 但无 max_error_pct 误差带")
        if cap.id in measured:
            declared = float(cap.evidence.get("max_error_pct", float("nan")))
            got = float(measured[cap.id])
            if abs(declared - got) > 5e-3:
                add("EVIDENCE_ARTIFACT_DRIFT", cap.id,
                    f"evidence.max_error_pct={declared} vs 工件实测 {got:.4f}")

    # prose ↔ registry
    docs = [ROOT / "README.md", ROOT / "README.zh-CN.md"] + sorted(
        p for p in (ROOT / "docs").glob("*.md"))
    by_id = {c.id: c for c in caps}
    for doc in docs:
        if not doc.is_file():
            continue
        for lineno, line in enumerate(doc.read_text(encoding="utf-8", errors="ignore")
                                      .splitlines(), 1):
            if not PRODUCTION_WORDS.search(line):
                continue
            for pattern, ids in CLAIM_KEYWORDS.items():
                if not re.search(pattern, line):
                    continue
                for cid in ids:
                    cap = by_id.get(cid)
                    if cap is None:
                        add("DOC_CLAIM_CONFLICT", cid,
                            f"文档断言生产可用，registry 无此能力", doc=f"{doc.name}:{lineno}")
                    elif cap.maturity != "mature":
                        add("DOC_CLAIM_CONFLICT", cid,
                            f"文档断言生产可用/已验证，registry maturity={cap.maturity}"
                            f"（wired={cap.wired} verified={cap.verified}）",
                            doc=f"{doc.name}:{lineno}")

    summary = {
        "capabilities": len(caps),
        "mature": sum(1 for c in caps if c.maturity == "mature"),
        "developing": sum(1 for c in caps if c.maturity == "developing"),
        "planned": sum(1 for c in caps if c.maturity == "planned"),
        "wired": sum(1 for c in caps if c.wired),
        "verified": sum(1 for c in caps if c.verified),
        "findings": len(rows),
        "open_findings": sum(1 for r in rows if not r["accepted"]),
        "by_code": {},
    }
    for row in rows:
        summary["by_code"][row["code"]] = summary["by_code"].get(row["code"], 0) + 1

    (ROOT / "docs" / "m23186").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "m23186" / "registry_audit.json").write_text(
        json.dumps({"summary": summary, "findings": rows,
                    "capabilities": [c.to_payload() for c in caps]},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# TensorLBM capability registry audit", "",
             f"- capabilities: {summary['capabilities']} (mature {summary['mature']} / "
             f"developing {summary['developing']} / planned {summary['planned']})",
             f"- wired {summary['wired']}, verified {summary['verified']}",
             f"- findings: {summary['findings']} {summary['by_code']}", "",
             "| code | capability | detail | doc | disposition |",
             "|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['code']} | {row['capability_id']} | {row['detail']} | "
                     f"{row['doc'] or ''} | {row['disposition']} |")
    if not rows:
        lines.append("| — | — | 无未处置虚标 | — | — |")
    (ROOT / "docs" / "m23186" / "registry_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    for row in rows[:15]:
        print(f"  {row['code']:24s} {row['capability_id']:42s} {row['detail'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
