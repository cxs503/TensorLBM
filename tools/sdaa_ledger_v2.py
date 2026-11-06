#!/usr/bin/env python3
"""SDAA 工件台账 v2 —— 归一抽取 + 冲突重裁 + 只读静态视图。

契约（v2）
---------
* 一条 entry = 一份工件实例（磁盘上的 ``artifacts/**/sdaa*.json``）。
  同一内容在多棵 mission worktree 里反复出现时，用 ``content_sha256`` +
  ``duplicate_of`` 显式标记副本关系；唯一内容视图见 ``entries_canonical``。
* 每条 entry **必须**含 ``evidence_level in {REAL,MOCK,STUB,WITHHELD,NOT_SUPPORTED}``。
* 缺失字段一律 ``null`` + ``missing.<field>`` 写原因；**不做默认填充**
  （``stats.default_fill_count`` 恒 0，脚本自检断言）。
* 冲突重裁：``conflict`` / ``conflict_reason`` 逐条写明「工件自述终态」与
  「派生证据等级 / 历史文档终态」的矛盾；只重裁不重跑，需要重跑的证据缺口
  登记进 ``needs_rerun[]``。
* 只读视图：静态 HTML，零后端、零外链。

用法::

    python3 tools/sdaa_ledger_v2.py               # 全语料（本树 + 全部 mission worktree）
    python3 tools/sdaa_ledger_v2.py --strict-local # 仅本工作树 artifacts/
    python3 tools/sdaa_ledger_v2.py --check        # 只跑机器判据自检
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import html
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LEVELS = ("REAL", "MOCK", "STUB", "WITHHELD", "NOT_SUPPORTED")
LEVEL_DOC = {
    "REAL": "真实算法：有 SDAA 设备证据 + 程序化算出的量化结果",
    "MOCK": "硬编码/伪造标记（mock|synthetic|fake|hardcoded 等）",
    "STUB": "空壳：只有结构/状态字段，无结果类量化指标",
    "WITHHELD": "证据不足：解析失败 / 缺设备证据 / SDAA 无有效数值 / 报告类工件",
    "NOT_SUPPORTED": "工件显式声明不支持 / 未实现",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKTREE_GLOB = "/root/tp-agent/work/*/artifacts"
# 台账自身产物（artifacts/sdaa_ledger_v2.json/.html 及其副本）不进语料
EXCLUDE_RE = re.compile(r"^sdaa_ledger")

RE_SDAA_DEVICE = re.compile(
    r"tcaicard|sdaa:\s*\d|\"sdaa_available\"\s*:\s*true|sdaa_device_name", re.I)
RE_CPU_ONLY = re.compile(r"\"device\"\s*:\s*\"cpu\"|cpu_baseline|cpu-only|not sdaa", re.I)
RE_FOREIGN_DEVICE = re.compile(r"\"device\"\s*:\s*\"(cuda|cpu)|cuda:\d|nvidia gpu|torch\.cuda", re.I)
RE_MOCK = re.compile(r"\b(mock|mocked|synthetic|hard[-_ ]?coded|fake|fabricated|"
                     r"placeholder[_ ]?value|dummy[_ ]?data)\b", re.I)
RE_UNSUPPORTED = re.compile(r"\b(not[_ -]?supported|unsupported|not[_ -]?implemented|"
                            r"unimplemented|nyi)\b|不支持|未实现", re.I)
RE_NOEVIDENCE = re.compile(r"no[-_ ]?evidence|no valid (?:sdaa )?(?:result|value|run)|"
                           r"cannot produce|未产生任何有效|无有效数值|无任何有效|无法产生任何有效", re.I)
RE_STUB_STATE = re.compile(r"\b(todo|not_run|not attempted|in_progress|pending|empty|stub)\b", re.I)
RE_PASS_WORD = re.compile(r"\b(pass|passed|ok|success|succeeded)\b", re.I)
RE_FAIL_WORD = re.compile(r"\b(fail|failed|failure|honest[-_ ]?fail)\b", re.I)

CONFIG_KEYS = {
    "steps", "n_steps", "min_steps", "max_steps", "nx", "ny", "nz", "nt", "n", "grid", "grids",
    "grid_H_list", "H", "H_list", "H_eff", "D", "R", "R_eff", "L", "L_cells", "tau", "u0", "U0",
    "re", "Re", "ma", "Ma", "timeout_s", "wall_clock_limit_s", "wall_time_s", "wall_clock_s",
    "elapsed_s", "elapsed_total_s", "version", "mission_id", "round", "count", "device_count",
    "sdaa_count", "capacity_gib", "already_allocated_gib", "free_gib", "reserved_gib",
    "tried_to_allocate_gib", "total_memory_gib", "alpha", "alpha_list", "omega", "dt", "dx",
}
RESULT_KEY_HINTS = (
    "err", "error", "rel", "l2", "linf", "cd", "cl", "nu", "nusselt", "st", "strouhal", "amp",
    "amplitude", "decrement", "decay", "convergence", "residual", "mean", "max", "min", "rate",
    "slope", "flux", "permeability", "drag", "lift", "moment", "pressure", "u_center", "u_peak",
    "v_peak", "period", "phase", "value", "measured", "computed", "analytic", "pass_pct",
)
CLAIM_KEYS = ("verdict", "status", "mission_status", "overall_pass", "all_pass", "verified",
              "passed", "gate")
HISTORY_RE_OVERALL = re.compile(r"Overall:?\s*(?:[❌✅✔✗]\s*)?\**\s*(PASS|FAIL|HONEST[-_ ]?FAIL)", re.I)
HISTORY_RE_BOLD = re.compile(r"本验证\s*\**\s*(PASS|FAIL|HONEST[-_ ]?FAIL)", re.I)
HISTORY_RE_ANY = re.compile(r"\b(PASS|FAIL|HONEST[-_ ]?FAIL)\b")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:n]


def iter_numeric(obj: Any, prefix: str = "", depth: int = 0) -> Iterable[Tuple[str, float]]:
    if depth > 8 or isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield prefix, float(obj)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_numeric(v, f"{prefix}.{k}" if prefix else str(k), depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from iter_numeric(v, f"{prefix}[{i}]", depth + 1)


def result_metrics(flat: List[Tuple[str, float]], limit: int = 24) -> Dict[str, float]:
    """挑出结果类量化指标（排掉配置数字），保持路径顺序。"""
    out: Dict[str, float] = {}
    for path, val in flat:
        leaf = re.split(r"[.\[]", path)[-1] or path
        if leaf in CONFIG_KEYS:
            continue
        low = path.lower()
        if any(h in low for h in RESULT_KEY_HINTS):
            out[path] = val
            if len(out) >= limit:
                break
    return out


def count_judgements(obj: Any, acc: Dict[str, int], depth: int = 0) -> None:
    """统计工件自带判据的 PASS/FAIL 条数。"""
    if depth > 8:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if isinstance(v, bool) and kl in ("pass", "passed", "success", "ok", "all_pass", "within_gate"):
                acc["pass" if v else "fail"] += 1
            elif isinstance(v, str):
                up = v.strip().upper()
                if up in ("PASS", "PASSED") or "✅" in v:
                    acc["pass"] += 1
                if up.startswith("FAIL") or "❌" in v:
                    acc["fail"] += 1
            count_judgements(v, acc, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            count_judgements(v, acc, depth + 1)


def find_claim(obj: Any) -> Dict[str, Any]:
    """工件自述终态：强字段 > 弱字段（status/mission_status/gate）。

    只把 verdict=*/overall_pass/all_pass/verified/overall.pass 这类**终态字段**当强声明；
    status 里出现 "ok"/"success"/"done" 之类只作弱声明（不能当 PASS 用）。
    """
    strong: List[Tuple[str, Any]] = []
    weak: List[Tuple[str, Any]] = []
    STRONG_KEYS = ("verdict", "overall_pass", "all_pass", "verified", "passed")
    WEAK_KEYS = ("status", "mission_status", "gate", "overall")

    def walk(o: Any, depth: int = 0) -> None:
        if depth > 5:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                kl = str(k).lower()
                if kl in STRONG_KEYS and isinstance(v, (str, bool)):
                    strong.append((str(k), v))
                elif kl in WEAK_KEYS:
                    if isinstance(v, (str, bool)):
                        weak.append((str(k), v))
                    elif isinstance(v, dict):
                        for kk in ("pass", "passed", "all_pass", "within_gate", "verdict", "status"):
                            if isinstance(v.get(kk), (str, bool)):
                                strong.append((f"{k}.{kk}", v[kk]))
                walk(v, depth + 1)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, depth + 1)

    walk(obj)
    for strength, pool in (("strong", strong), ("weak", weak)):
        for key, val in pool:
            if isinstance(val, bool):
                return {"norm": "PASS" if val else "FAIL", "raw": str(val), "field": key,
                        "strength": strength}
            sval = str(val).strip()
            up = sval.upper()
            if RE_FAIL_WORD.search(up):
                return {"norm": "FAIL", "raw": sval, "field": key, "strength": strength}
            if up in ("PASS", "PASSED", "OK", "SUCCESS", "SUCCEEDED", "DONE", "COMPLETE"):
                norm = "PASS" if up in ("PASS", "PASSED") else "WEAK_PASS"
                return {"norm": norm, "raw": sval, "field": key, "strength": strength}
    return {"norm": None, "raw": None, "field": None, "strength": None}


def detect_kind(doc: Any, text: str, gates: Any, wall: Any) -> Tuple[str, str]:
    """工件类型：report / case_definition / run_evidence / unknown（附判定理由）。"""
    if isinstance(doc, dict) and {"title", "markdown"} <= set(doc):
        return "report", "title+markdown，平台报告类工件"
    has_dev = bool(RE_SDAA_DEVICE.search(text)) or bool(RE_CPU_ONLY.search(text)) or \
        bool(RE_FOREIGN_DEVICE.search(text))
    if gates is not None or has_dev or wall is not None:
        return "run_evidence", "含设备/判据/墙钟，属运行证据"
    keys = set(doc) if isinstance(doc, dict) else set()
    if keys & {"config", "lattice", "collision", "geometry", "boundary", "tiers", "grid",
               "grids", "mode", "mode_rule", "driving"}:
        return "case_definition", "含 case 定义/配置字段但无设备、无判据、无墙钟"
    return "unknown", "结构不可辨识"


def detect_case(doc: Any, filename: str) -> Tuple[Optional[str], str, Optional[str]]:
    if isinstance(doc, dict):
        for key in ("case", "benchmark", "case_name", "case_dir", "case_path"):
            val = doc.get(key)
            if isinstance(val, str) and val.strip():
                return os.path.basename(val.strip().rstrip("/")), f"field:{key}", None
            if isinstance(val, dict) and isinstance(val.get("name"), str):
                return val["name"], f"field:{key}.name", None
        if "title" in doc and "markdown" in doc:
            return None, "none", "报告类工件（title/markdown），不含 case 字段"
    stem = re.sub(r"^sdaa_", "", Path(filename).stem)
    stem = re.sub(r"_(report|evidence|result|diag|diagnosis|v\d+)$", "", stem)
    if stem:
        return stem, "filename", None
    return None, "none", "工件与文件名均无 case 标识"


def load_history(pairs: List[Tuple[str, str]]) -> Dict[str, Dict[str, Any]]:
    """从 docs/sdaa_sweep/<case>.md 抽历史终态（只读文本证据）。"""
    hist: Dict[str, Dict[str, Any]] = {}
    for case, text in pairs:
        verdict = None
        m = HISTORY_RE_OVERALL.search(text) or HISTORY_RE_BOLD.search(text)
        if m:
            verdict = m.group(1).upper().replace(" ", "").replace("-", "_")
            if verdict.startswith("FAIL"):
                verdict = "FAIL"
        else:
            counts = Counter(x.upper().replace("-", "_") for x in HISTORY_RE_ANY.findall(text[-4000:]))
            if counts:
                verdict = "PASS" if counts.get("PASS", 0) > counts.get("FAIL", 0) else "FAIL"
        if verdict is None:
            continue
        cur = hist.setdefault(case, {"verdicts": Counter(), "sources": []})
        cur["verdicts"][verdict] += 1
        cur["sources"].append(f"docs/sdaa_sweep/{case}.md")
    out: Dict[str, Dict[str, Any]] = {}
    for case, info in hist.items():
        vh = info["verdicts"]
        npass = vh.get("PASS", 0)
        nfail = sum(v for k, v in vh.items() if k != "PASS")
        out[case] = {
            "terminal": "PASS" if npass > 0 and npass >= nfail else "FAIL",
            "verdict_histogram": dict(vh),
            "sources": info["sources"][:3],
        }
    return out


def load_history_docs(root: Path) -> Dict[str, Dict[str, Any]]:
    pairs: List[Tuple[str, str]] = []
    for f in sorted((root / "docs" / "sdaa_sweep").glob("*.md")):
        try:
            pairs.append((f.stem, f.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    return load_history(pairs)

# ---------------------------------------------------------------- 单件抽取 + 定级 + 重裁
def adjudicate_entry(doc: Any, text: str, parse_ok: bool, parse_error: Optional[str],
                     rel_path: str, history: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """归一抽取一份工件；缺失字段一律 null + missing_reason（不填默认值）。"""
    missing: Dict[str, str] = {}
    sig: Dict[str, Any] = {}
    case, case_source, case_reason = detect_case(doc, Path(rel_path).name)

    if not parse_ok:
        sig.update(device_evidence=False, numeric_leaves=0, result_metrics=0, mock_marker=False,
                   unsupported_marker=False, no_evidence_marker=False, stub_state_marker=False,
                   report_like=False, claimed_terminal=None)
        out = {
            "case": case, "case_source": case_source, "evidence_kind": "unknown",
            "evidence_level": "WITHHELD",
            "evidence_reason": f"JSON 解析失败（{parse_error}），无法派生证据等级",
            "evidence_signals": sig, "device": None, "device_kind": "none",
            "verdict_claimed": None, "verdict_claimed_raw": None, "verdict_claim_strength": None,
            "gate_pass": None, "gate_fail": None, "numeric_metrics": None, "wall_clock_s": None,
            "mission_id": None, "reference": None, "timestamp": None, "conflict": None,
            "conflict_reason": None, "needs_rerun": False, "rerun_reason": None,
            "missing": {"parse": parse_error or "parse_error"},
        }
        if case_reason:
            out["missing"]["case"] = case_reason
        return out

    # ---- 字段抽取（缺则 null + 原因）
    def first_key(keys: Tuple[str, ...]) -> Any:
        if isinstance(doc, dict):
            for k in keys:
                if k in doc and doc[k] is not None:
                    return doc[k]
        return None

    device = first_key(("device_evidence", "device", "hardware"))
    gates = first_key(("gates", "criteria", "criteria_gates", "judgment", "judgments", "arms",
                       "arm_gates", "criteria_gate"))
    wall = first_key(("wall_clock_s", "wall_clock_total_s", "elapsed_s", "elapsed_total_s",
                      "wall_clock_sec", "total_wall_s", "wall_clock_total_sec"))
    mission_id = first_key(("mission_id", "mission"))
    ts = first_key(("timestamp", "generated", "generated_utc", "generated_at"))
    ref = first_key(("reference", "references", "reference_data_summary", "analytic", "theory"))

    kind, kind_reason = detect_kind(doc, text, gates, wall)
    is_report = kind == "report"
    flat = list(iter_numeric(doc))
    all_metrics = {} if is_report else result_metrics(flat, limit=10 ** 6)
    metrics = dict(list(all_metrics.items())[:8])  # 台账每件只留前 8 项样例指标
    acc = {"pass": 0, "fail": 0}
    count_judgements(doc, acc)

    has_device = bool(RE_SDAA_DEVICE.search(text))
    cpu_marker = bool(RE_CPU_ONLY.search(text))
    foreign = bool(RE_FOREIGN_DEVICE.search(text))
    mock_marker = bool(RE_MOCK.search(text))
    unsupported_marker = bool(RE_UNSUPPORTED.search(text))
    noevidence_marker = bool(RE_NOEVIDENCE.search(text))
    stub_marker = bool(RE_STUB_STATE.search(text))
    claim = {"norm": None, "raw": None, "field": None, "strength": None} if is_report else find_claim(doc)
    claim_norm, claim_raw, claim_field = claim["norm"], claim["raw"], claim["field"]

    if has_device:
        dev_kind = "sdaa"
    elif re.search(r"cuda|nvidia", text, re.I):
        dev_kind = "cuda"
    elif cpu_marker:
        dev_kind = "cpu"
    else:
        dev_kind = "none"

    sig.update(device_evidence=has_device, device_kind=dev_kind, cpu_only_marker=cpu_marker,
               foreign_device_marker=foreign, numeric_leaves=len(flat),
               result_metrics=len(all_metrics), gate_pass=acc["pass"], gate_fail=acc["fail"],
               mock_marker=mock_marker, unsupported_marker=unsupported_marker,
               no_evidence_marker=noevidence_marker, stub_state_marker=stub_marker,
               report_like=is_report, artifact_kind=kind, artifact_kind_reason=kind_reason,
               claimed_terminal=claim_norm, claim_strength=claim["strength"])

    # ---- 证据定级（顺序即优先级）
    if unsupported_marker and len(all_metrics) < 3:
        level, reason = "NOT_SUPPORTED", "工件显式声明不支持/未实现"
    elif mock_marker:
        level, reason = "MOCK", "含硬编码/伪造标记（mock|synthetic|hardcoded|fake|placeholder|dummy|fabricated）"
    elif is_report:
        level, reason = "WITHHELD", "报告类工件（title/markdown），无原始数值证据"
    elif kind == "case_definition":
        level, reason = "WITHHELD", f"案例定义/配置类工件（{kind_reason}），本身不承载运行证据"
    elif noevidence_marker:
        level, reason = "WITHHELD", "工件自述 no-evidence / 未产生任何有效数值"
    elif len(all_metrics) < 3:
        if stub_marker or claim_norm is None:
            level = "STUB"
            reason = f"无结果类量化指标（数值叶子 {len(flat)}，结果类 {len(all_metrics)}），仅结构/状态字段"
        else:
            level = "WITHHELD"
            reason = f"有终态声明但无结果类量化指标（结果类 {len(all_metrics)}）"
    elif not has_device:
        level = "WITHHELD"
        if dev_kind == "cuda":
            reason = "设备归属为非 SDAA（CUDA/NVIDIA），工件却以 sdaa_ 前缀归档，SDAA 证据不成立"
        elif dev_kind == "cpu":
            reason = "仅 CPU 结果，无 SDAA 设备证据"
        else:
            reason = "缺 SDAA 设备证据（无 tcaicard / sdaa:<id> / sdaa_available=true），设备归属不明"
    else:
        level = "REAL"
        reason = f"SDAA 设备证据 + {len(all_metrics)} 项程序化量化结果"

    if device is None:
        missing["device"] = "工件无 device_evidence/device/hardware 字段"
    if gates is None:
        missing["criteria_gates"] = "工件无 gates/criteria/judgment 判据块"
    if claim_norm is None:
        missing["verdict_claimed"] = (
            "工件无 verdict/overall_pass/all_pass/verified 强终态字段"
            + (f"（仅有弱字段 {claim_raw}）" if claim_raw else ""))
    if not all_metrics:
        missing["numeric_metrics"] = (
            "案例定义类工件本就不含结果数值" if kind == "case_definition"
            else "无结果类量化指标（误差/系数/收敛量）")
    if wall is None:
        missing["wall_clock_s"] = "工件无墙钟字段"
    if mission_id is None:
        missing["mission_id"] = "工件无 mission_id"
    if ts is None:
        missing["timestamp"] = "工件无 timestamp/generated"
    if ref is None:
        missing["reference"] = "工件无 reference/analytic 基准块"
    if case_reason:
        missing["case"] = case_reason

    gate_pass = None if (acc["pass"] == 0 and acc["fail"] == 0) else acc["fail"] == 0

    # ---- 冲突重裁（只重裁不重跑）
    hist = history.get(case) if case else None
    hist_terminal = hist["terminal"] if hist else None
    hist_sources = (hist or {}).get("sources", [])
    conflict: Optional[Dict[str, Any]] = None
    conflict_reason: Optional[str] = None
    needs_rerun = False
    rerun_reason: Optional[str] = None

    def mark(kind_c: str, why: str, rerun: bool, claimed: Optional[str] = None,
             extra: Optional[Dict[str, Any]] = None) -> None:
        nonlocal conflict, conflict_reason, needs_rerun
        conflict = {"type": kind_c, "claimed": claimed if claimed is not None else claim_raw,
                    "claim_strength": claim["strength"], "derived_level": level,
                    "historical_terminal": hist_terminal, "historical_sources": hist_sources}
        if extra:
            conflict.update(extra)
        conflict_reason = why
        needs_rerun = rerun

    if claim_norm == "PASS" and level != "REAL":
        mark("claim_unsupported",
             f"工件自述 PASS（{claim_field}={claim_raw}），但派生证据等级为 {level}：{reason}；"
             f"PASS 无 SDAA 数值支撑，重裁为 {level}", True)
    elif claim_norm == "PASS" and acc["fail"] > 0:
        mark("self_inconsistent",
             f"工件自述 PASS（{claim_field}={claim_raw}），但内部判据含 {acc['fail']} 条 FAIL / "
             f"{acc['pass']} 条 PASS，自述终态与判据明细矛盾（重裁取明细：gate_pass=False）", False,
             extra={"gate_fail_count": acc["fail"], "gate_pass_count": acc["pass"]})
    elif claim_norm == "FAIL" and level == "REAL" and acc["fail"] == 0 and acc["pass"] > 0:
        mark("claim_understated",
             f"工件自述 FAIL（{claim_field}={claim_raw}），但 {acc['pass']} 条判据全 PASS 且数值完整"
             f"（{level}），终态与明细矛盾，需重裁", True)
    elif claim_norm is not None and hist_terminal is not None and claim_norm != hist_terminal:
        mark("history_drift",
             f"工件终态 {claim_norm}（{claim_field}={claim_raw}，{'强' if claim['strength'] == 'strong' else '弱'}字段）"
             f"与历史文档终态 {hist_terminal}（{', '.join(hist_sources)}）不一致，历史终态需以最新工件重裁", False)
    elif claim_norm == "WEAK_PASS" and level != "REAL":
        mark("weak_claim_unsupported",
             f"工件弱终态字段 {claim_field}={claim_raw}（非 verdict/overall_pass 类强声明），"
             f"而派生等级为 {level}：{reason}；不据此判 PASS", False)
    elif claim_norm == "WEAK_PASS" and hist_terminal == "FAIL":
        mark("weak_status_vs_history",
             f"工件弱终态字段 {claim_field}={claim_raw}（成功态）与历史文档终态 FAIL"
             f"（{', '.join(hist_sources)}）不一致，属摘要级工件与历史结论的漂移", False)
    elif claim_norm is None and kind == "run_evidence" and hist_terminal == "PASS" and level != "REAL":
        mark("history_conflict",
             f"历史文档终态 PASS（{', '.join(hist_sources)}），但最新运行证据工件未携带可判 PASS 的证据"
             f"（{level}：{reason}）", True, claimed="PASS(history)")

    if not needs_rerun and level in ("MOCK", "STUB", "WITHHELD") and case \
            and kind in ("run_evidence", "unknown"):
        needs_rerun = True
        rerun_reason = f"证据缺口（{level}）：{reason}"

    return {
        "case": case, "case_source": case_source, "evidence_kind": kind,
        "evidence_level": level, "evidence_reason": reason, "evidence_signals": sig,
        "device": device, "device_kind": dev_kind,
        "verdict_claimed": claim_norm, "verdict_claimed_raw": claim_raw,
        "verdict_claim_strength": claim["strength"],
        "gate_pass": gate_pass, "gate_fail": acc["fail"],
        "numeric_metrics": metrics or None,
        "wall_clock_s": wall if wall is None or isinstance(wall, (int, float, str)) else (
            json.dumps(wall, ensure_ascii=False)[:200]),
        "mission_id": mission_id, "reference": ref, "timestamp": ts,
        "conflict": conflict, "conflict_reason": conflict_reason,
        "needs_rerun": needs_rerun, "rerun_reason": rerun_reason, "missing": missing,
    }


def rerun_action(entry: Dict[str, Any]) -> str:
    lvl = entry["evidence_level"]
    if lvl == "MOCK":
        return "删除硬编码/伪造数值，上卡重跑该案例并以程序输出重生成工件"
    if lvl == "STUB":
        return "补齐量化结果与判据后重跑（当前工件只有结构/状态字段）"
    if lvl == "NOT_SUPPORTED":
        return "换可用后端或声明不支持，勿以 PASS 结案"
    if lvl == "WITHHELD":
        return "补齐 SDAA 设备证据行 + 有效数值后重跑（含 OOM/超时需改分辨率或换卡）"
    return "重跑该案例以复核终态（数值齐全但终态与明细矛盾）"


# ---------------------------------------------------------------- 语料发现与构建
def discover(root: Path, worktree_glob: str, strict_local: bool) -> Tuple[List[Tuple[str, Path, str]], Dict[str, Any]]:
    files: List[Tuple[str, Path, str]] = []
    roots: List[Tuple[str, Path]] = []
    art = root / "artifacts"
    if art.is_dir():
        roots.append(("self", art))
    if not strict_local:
        for d in sorted(glob.glob(worktree_glob)):
            p = Path(d)
            if p.is_dir() and p.resolve() != art.resolve():
                roots.append((p.parent.name, p))

    seen = set()
    for wid, base in roots:
        for f in sorted(base.rglob("sdaa*.json")):
            if not f.is_file():
                continue
            rp = f.resolve()
            if rp in seen or EXCLUDE_RE.search(rp.name):
                continue  # 台账产物自身不得进入语料
            seen.add(rp)
            files.append((wid, rp, str(rp.relative_to(base.resolve()))))

    outside = sorted(set(glob.glob("/tmp/sdaa*.json")) - {str(f[1]) for f in files})
    corpus = {
        "roots": [{"worktree": wid, "path": str(base)} for wid, base in roots],
        "worktree_count": len(roots),
        "outside_scope_instances": len(outside),
        "outside_scope_note": "不在任何 mission worktree artifacts/ 内的同名副本（/tmp 临时产物等），"
                              "不计入台账口径",
        "pattern": "artifacts/**/sdaa*.json",
    }
    return files, corpus


def build(root: Path, worktree_glob: str, strict_local: bool) -> Dict[str, Any]:
    files, corpus = discover(root, worktree_glob, strict_local)
    history = load_history_docs(root)

    entries: List[Dict[str, Any]] = []
    by_hash: Dict[str, List[str]] = defaultdict(list)
    parse_fail: List[Dict[str, str]] = []
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for idx, (wid, path, rel) in enumerate(files):
        try:
            raw = path.read_bytes()
        except OSError as exc:
            parse_fail.append({"artifact": str(path), "error": f"OSError: {exc}"})
            continue
        digest = sha256_bytes(raw)
        text = raw.decode("utf-8-sig", errors="replace")
        try:
            doc = json.loads(text)
            parse_ok, perr = True, None
        except Exception as exc:  # noqa: BLE001 —— 逐件容错
            doc, parse_ok, perr = None, False, f"{type(exc).__name__}: {exc}"[:200]
            parse_fail.append({"artifact": str(path), "error": perr})

        body = adjudicate_entry(doc, text, parse_ok, perr, rel, history)
        entry = {
            "entry_id": f"e{idx:05d}-{short_hash(wid + '/' + rel)}",
            "worktree": wid,
            "artifact_rel": rel,
            "artifact_path": str(path),
            "artifact_sha256": digest,
            "artifact_bytes": len(raw),
            "artifact_mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(
                timespec="seconds"),
            "parse_ok": parse_ok,
            "parse_error": perr,
            "duplicate_of": None,
            **body,
        }
        by_hash[digest].append(entry["entry_id"])
        entries.append(entry)

    # 副本归一：canonical 优先本工作树实例，其次路径最短
    id2entry = {e["entry_id"]: e for e in entries}
    canon_of_hash: Dict[str, str] = {}
    for digest, ids in by_hash.items():
        best = min((id2entry[i] for i in ids),
                   key=lambda e: (e["worktree"] != "self", len(e["artifact_path"]), e["artifact_path"]))
        canon_of_hash[digest] = best["entry_id"]
    for e in entries:
        canon = canon_of_hash[e["artifact_sha256"]]
        if e["entry_id"] != canon:
            e["duplicate_of"] = canon

    canonical = [e for e in entries if e["duplicate_of"] is None]

    # 案例级复核：同案例已有 REAL（canonical）证据时，非证据载体工件不产生
    # “历史 PASS 无证据”冲突，也不重复登记重跑（避免案例级假阳性）。
    real_rel: Dict[str, str] = {}
    for e in sorted(canonical, key=lambda x: (x["evidence_level"] != "REAL", x["artifact_rel"])):
        if e["case"] and e["evidence_level"] == "REAL":
            real_rel.setdefault(e["case"], e["artifact_rel"])
    for e in entries:
        case = e["case"]
        if not case or case not in real_rel:
            continue
        if e["conflict"] and e["conflict"]["type"] == "history_conflict":
            e["conflict"]["type"] = "non_evidence_artifact"
            e["conflict"]["superseded_by"] = real_rel[case]
            e["conflict_reason"] = (
                f"{e['conflict_reason']}；同案例另有 REAL 证据工件 {real_rel[case]} 承载判据，"
                f"本件为非证据载体，不据此登记重跑")
            e["needs_rerun"] = False
            e["rerun_reason"] = f"已抑制：本件非证据载体，同案例 REAL 工件为 {real_rel[case]}"
        elif e["needs_rerun"] and e["conflict"] is None and e["evidence_level"] in ("STUB", "WITHHELD"):
            e["needs_rerun"] = False
            e["rerun_reason"] = (
                f"已抑制：证据缺口由同案例 REAL 工件 {real_rel[case]} 覆盖，本件不重复登记重跑；"
                f"原缺口：{e['rerun_reason'] or '无'}")

    dist = Counter(e["evidence_level"] for e in entries)
    dist_canon = Counter(e["evidence_level"] for e in canonical)
    conflicts = [e for e in entries if e["conflict"]]
    conflicts_canon = [e for e in canonical if e["conflict"]]
    needs_rerun = [{
        "entry_id": e["entry_id"], "case": e["case"], "artifact_rel": e["artifact_rel"],
        "worktree": e["worktree"], "conflict_type": (e["conflict"] or {}).get("type"),
        "trigger": "conflict" if e["conflict"] and e["conflict_reason"] else "evidence_gap",
        "derived_level": e["evidence_level"],
        "reason": e["conflict_reason"] or e["rerun_reason"],
        "suggested_action": rerun_action(e),
    } for e in entries if e["needs_rerun"]]
    needs_rerun_canon = [r for r in needs_rerun
                         if id2entry[r["entry_id"]]["duplicate_of"] is None]

    missing_hist = Counter()
    for e in entries:
        for k in e["missing"]:
            missing_hist[k] += 1

    case_acc: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"instances": 0, "levels": Counter(), "needs_rerun": 0})
    for e in canonical:
        key = e["case"] or "<unknown>"
        case_acc[key]["instances"] += 1
        case_acc[key]["levels"][e["evidence_level"]] += 1
        case_acc[key]["needs_rerun"] += int(bool(e["needs_rerun"]))
    case_rows = []
    for name, info in sorted(case_acc.items()):
        top = max(info["levels"].items(), key=lambda kv: kv[1])
        case_rows.append({"case": name, "instances": info["instances"],
                          "dominant_level": top[0], "levels": dict(info["levels"]),
                          "needs_rerun": info["needs_rerun"]})

    default_fill = 0
    for e in entries:  # 契约自检：凡空字段必须落到 missing 里，绝不填默认值
        for field in ("case", "device", "numeric_metrics", "verdict_claimed", "wall_clock_s"):
            if e.get(field) is None and field not in e["missing"]:
                default_fill += 1
    assert default_fill == 0, f"契约破坏：{default_fill} 个空字段缺少 missing_reason"

    stats = {
        "entries": len(entries),
        "unique_artifacts": len(by_hash),
        "duplicate_instances": len(entries) - len(by_hash),
        "duplication_factor": round(len(entries) / max(1, len(by_hash)), 2),
        "parse_ok": round(sum(1 for e in entries if e["parse_ok"]) / max(1, len(entries)), 4),
        "parse_fail_count": len(parse_fail),
        "default_fill_count": default_fill,
        "cases": len(case_rows),
        "evidence_level_instances": {k: dist.get(k, 0) for k in LEVELS},
        "evidence_level_canonical": {k: dist_canon.get(k, 0) for k in LEVELS},
        "entries_without_evidence_level": sum(1 for e in entries if e.get("evidence_level") not in LEVELS),
        "conflict_entries": len(conflicts),
        "conflict_entries_canonical": len(conflicts_canon),
        "conflict_entries_missing_reason": sum(1 for e in conflicts if not e["conflict_reason"]),
        "needs_rerun_entries": len(needs_rerun),
        "needs_rerun_canonical": len(needs_rerun_canon),
        "cases_with_real_evidence": len(real_rel),
        "evidence_kind_instances": dict(Counter(e["evidence_kind"] for e in entries)),
        "device_kind_instances": dict(Counter(e["device_kind"] for e in entries)),
        "claimed_terminal_instances": dict(Counter(str(e["verdict_claimed"]) for e in entries)),
        "conflict_type_instances": dict(Counter(e["conflict"]["type"] for e in conflicts)),
        "missing_field_histogram": dict(missing_hist.most_common()),
        "generated_utc": generated,
        "tool": "tools/sdaa_ledger_v2.py",
    }

    return {
        "schema": "sdaa_ledger_v2",
        "generated_utc": generated,
        "corpus": corpus,
        "policy": {
            "entry_granularity": "一份工件实例（artifacts/**/sdaa*.json 文件）",
            "duplicate_handling": "同 sha256 的实例标 duplicate_of，canonical 见 entries_canonical",
            "levels": LEVEL_DOC,
            "missing_fields": "null + missing.<field>，不做默认填充（default_fill_count 恒 0）",
            "conflict_policy": "只重裁不重跑；需重跑的证据缺口登记在 needs_rerun",
            "history_source": "docs/sdaa_sweep/*.md 的历史终态文本",
        },
        "stats": stats,
        "cases": case_rows,
        "needs_rerun": needs_rerun,
        "parse_failures": parse_fail,
        "entries": entries,
        "entries_canonical": [e["entry_id"] for e in canonical],
    }


# ---------------------------------------------------------------- 只读静态 HTML
def _bar(frac: float, color: str, width: int = 220) -> str:
    w = max(2, int(round(frac * width)))
    return (f'<span class="bar"><span class="fill" style="width:{w}px;background:{color}">'
            f'</span></span>')


LEVEL_COLOR = {"REAL": "#1f8a4c", "MOCK": "#c0392b", "STUB": "#8e44ad",
               "WITHHELD": "#b9770e", "NOT_SUPPORTED": "#5d6d7e"}


def render_html(ledger: Dict[str, Any]) -> str:
    st = ledger["stats"]
    cases = ledger["cases"]
    entries = ledger["entries"]
    canon_ids = set(ledger["entries_canonical"])
    canon = [e for e in entries if e["entry_id"] in canon_ids]
    conflicts = [e for e in entries if e["conflict"]]
    esc = html.escape

    def dist_table(dist: Dict[str, int], total: int, caption: str) -> str:
        rows = []
        for lvl in LEVELS:
            n = dist.get(lvl, 0)
            rows.append(
                f"<tr><td><span class='tag' style='background:{LEVEL_COLOR[lvl]}'>{lvl}</span></td>"
                f"<td class='num'>{n}</td><td class='num'>{n / max(1, total) * 100:.1f}%</td>"
                f"<td>{_bar(n / max(1, total), LEVEL_COLOR[lvl])}</td>"
                f"<td class='dim'>{esc(LEVEL_DOC[lvl])}</td></tr>")
        return (f"<table><caption>{esc(caption)}（n={total}）</caption><thead><tr><th>等级</th>"
                f"<th>条目</th><th>占比</th><th></th><th>口径</th></tr></thead><tbody>"
                + "".join(rows) + "</tbody></table>")

    case_rows = []
    for row in cases:
        levels = "".join(
            f"<span class='tag' style='background:{LEVEL_COLOR.get(k, '#666')}'>{k}:{v}</span> "
            for k, v in sorted(row["levels"].items(), key=lambda kv: -kv[1]))
        case_rows.append(
            f"<tr><td><code>{esc(str(row['case']))}</code></td><td class='num'>{row['instances']}</td>"
            f"<td><span class='tag' style='background:{LEVEL_COLOR.get(row['dominant_level'], '#666')}'>"
            f"{row['dominant_level']}</span> {levels}</td>"
            f"<td class='num'>{row['needs_rerun']}</td></tr>")

    conflict_rows = []
    for e in conflicts:
        c = e["conflict"]
        conflict_rows.append(
            f"<tr><td><code>{esc(str(e['case']))}</code></td>"
            f"<td><code class='dim'>{esc(e['artifact_rel'])}</code>"
            f"<div class='dim'>worktree={esc(e['worktree'])} · sha={esc(e['artifact_sha256'][:10])}</div></td>"
            f"<td><span class='tag' style='background:{LEVEL_COLOR.get(e['evidence_level'], '#666')}'>"
            f"{e['evidence_level']}</span></td><td>{esc(c['type'])}</td>"
            f"<td>{esc(str(c.get('claimed')))}</td>"
            f"<td>{esc(str(c.get('historical_terminal')))}</td>"
            f"<td class='why'>{esc(e['conflict_reason'] or '')}</td>"
            f"<td class='num'>{'是' if e['needs_rerun'] else '否'}</td></tr>")

    rerun_rows = []
    for r in ledger["needs_rerun"]:
        rerun_rows.append(
            f"<tr><td><code>{esc(str(r['case']))}</code></td>"
            f"<td><code class='dim'>{esc(r['artifact_rel'])}</code></td>"
            f"<td>{esc(str(r['conflict_type']))}</td><td>{esc(r['suggested_action'])}</td></tr>")

    missing_rows = [f"<tr><td><code>{esc(k)}</code></td><td class='num'>{v}</td>"
                    f"<td class='num'>{v / max(1, st['entries']) * 100:.1f}%</td></tr>"
                    for k, v in st["missing_field_histogram"].items()]

    sample_rows = []
    for e in canon[:200]:
        metrics = e["numeric_metrics"] or {}
        mtxt = ", ".join(f"{k}={v:.4g}" for k, v in list(metrics.items())[:3]) or "—"
        sample_rows.append(
            f"<tr data-level='{e['evidence_level']}'><td><code>{esc(str(e['case']))}</code></td>"
            f"<td><code class='dim'>{esc(e['artifact_rel'])}</code></td>"
            f"<td><span class='tag' style='background:{LEVEL_COLOR.get(e['evidence_level'], '#666')}'>"
            f"{e['evidence_level']}</span></td>"
            f"<td>{esc(str(e['verdict_claimed']))}</td>"
            f"<td>{esc(str(e['gate_pass']))}</td>"
            f"<td class='num'>{e['evidence_signals']['result_metrics']}</td>"
            f"<td class='mono'>{esc(mtxt)}</td>"
            f"<td class='why'>{esc(e['evidence_reason'])}</td></tr>")

    corpus = ledger["corpus"]
    root_lines = "".join(f"<li><code>{esc(r['worktree'])}</code> · {esc(r['path'])}</li>"
                         for r in corpus["roots"][:12])
    more = len(corpus["roots"]) - 12
    root_lines += f"<li class='dim'>… 另有 {more} 棵 worktree</li>" if more > 0 else ""

    total = st["entries"]
    summary = (f"台账共 {total} 条工件实例，覆盖 {corpus['worktree_count']} 棵 worktree；"
               f"按内容哈希归一后唯一工件 {st['unique_artifacts']} 份"
               f"（副本放大 {st['duplication_factor']}×）。parse_ok={st['parse_ok']:.2%}，"
               f"缺失字段默认填充 {st['default_fill_count']} 处（null+missing_reason 口径），"
               f"冲突 {st['conflict_entries']} 条、登记重跑 {st['needs_rerun_entries']} 条。")

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>SDAA 工件台账 v2</title>
<style>
:root{{--fg:#1b2631;--dim:#6b7a86;--bg:#f6f8f9;--line:#d7dee3}}
*{{box-sizing:border-box}}
body{{margin:0;padding:28px 32px 64px;font:14px/1.6 -apple-system,"Segoe UI","Noto Sans CJK SC",sans-serif;
color:var(--fg);background:var(--bg)}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:32px 0 10px;border-left:4px solid #1f8a4c;
padding-left:9px}} h3{{font-size:15px;margin:18px 0 8px}}
.sub{{color:var(--dim);margin:0 0 16px}}
.kpis{{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0 4px}}
.kpi{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:132px}}
.kpi b{{display:block;font-size:20px}} .kpi span{{color:var(--dim);font-size:12px}}
table{{border-collapse:collapse;width:100%;background:#fff;margin:8px 0 14px;font-size:13px}}
caption{{text-align:left;color:var(--dim);padding:6px 0;font-size:12px}}
th,td{{border:1px solid var(--line);padding:6px 9px;vertical-align:top;text-align:left}}
th{{background:#eef2f4;font-weight:600}} .num{{text-align:right;white-space:nowrap}}
td.why{{max-width:420px}} .mono{{font-family:ui-monospace,Menlo,monospace;font-size:12px;max-width:320px;
overflow-wrap:anywhere}} .dim{{color:var(--dim)}} code{{font-family:ui-monospace,Menlo,monospace;font-size:12px}}
.tag{{display:inline-block;color:#fff;border-radius:4px;padding:0 6px;font-size:11px;line-height:17px}}
.bar{{display:inline-block;width:220px;height:11px;background:#e6ebee;border-radius:6px;overflow:hidden;
vertical-align:middle}} .fill{{display:block;height:11px}}
.note{{background:#fff8e6;border:1px solid #f0d58c;border-radius:8px;padding:10px 14px;margin:12px 0}}
footer{{margin-top:36px;color:var(--dim);font-size:12px}}
</style></head><body>
<h1>SDAA 工件台账 v2</h1>
<p class="sub">归一抽取 · evidence_level 定级 · 冲突重裁（只重裁不重跑） · 只读静态视图 ·
生成于 {esc(st['generated_utc'])} · 工具 <code>{esc(st['tool'])}</code></p>

<div class="kpis">
 <div class="kpi"><b>{total}</b><span>工件条目（实例）</span></div>
 <div class="kpi"><b>{st['unique_artifacts']}</b><span>唯一内容工件</span></div>
 <div class="kpi"><b>{st['parse_ok']:.1%}</b><span>parse_ok</span></div>
 <div class="kpi"><b>{st['cases']}</b><span>案例数（唯一内容）</span></div>
 <div class="kpi"><b>{st['conflict_entries']}</b><span>冲突条目</span></div>
 <div class="kpi"><b>{st['needs_rerun_entries']}</b><span>登记重跑</span></div>
 <div class="kpi"><b>{st['default_fill_count']}</b><span>默认填充计数</span></div>
</div>
<p>{esc(summary)}</p>

<div class="note"><b>口径提示</b>：本轮语料里同内容工件被多棵 mission worktree 反复检出
（{total} 份实例 = {st['unique_artifacts']} 份唯一内容，放大 {st['duplication_factor']}×）。
台账因此同时给两个视图：<b>实例视图</b>（<code>entries</code>，磁盘上真实存在的每一份文件）与
<b>唯一内容视图</b>（<code>entries_canonical</code>，按 sha256 去重后的 canonical 实例）。
“约 3132 份”的既有口径是副本放大的结果，实质唯一工件远少于该数——这是本轮审计的首要发现。</div>

<h2>1. 语料与口径</h2>
<table><tbody>
<tr><th>发现模式</th><td><code>{esc(corpus['pattern'])}</code></td></tr>
<tr><th>worktree 根数</th><td>{corpus['worktree_count']}</td></tr>
<tr><th>实例 / 唯一</th><td>{total} / {st['unique_artifacts']}（副本 {st['duplicate_instances']} 份）</td></tr>
<tr><th>解析失败</th><td>{st['parse_fail_count']}</td></tr>
<tr><th>口径外副本</th><td>{corpus['outside_scope_instances']}（{esc(corpus['outside_scope_note'])}）</td></tr>
</tbody></table>
<h3>语料根（只读）</h3><ul class="dim">{root_lines}</ul>
<h3>定级口径</h3>
<table><thead><tr><th>等级</th><th>判据</th></tr></thead><tbody>
{''.join(f"<tr><td><span class='tag' style='background:{LEVEL_COLOR[l]}'>{l}</span></td><td>{esc(d)}</td></tr>" for l, d in LEVEL_DOC.items())}
</tbody></table>

<h2>2. 工件类型 / 设备归属 / 冲突类型</h2>
<table><thead><tr><th>维度</th><th>分布</th></tr></thead><tbody>
<tr><th>工件类型 evidence_kind</th><td>{esc(' · '.join(f'{k}={v}' for k, v in st['evidence_kind_instances'].items()))}</td></tr>
<tr><th>设备归属 device_kind</th><td>{esc(' · '.join(f'{k}={v}' for k, v in st['device_kind_instances'].items()))}</td></tr>
<tr><th>自述终态</th><td>{esc(' · '.join(f'{k}={v}' for k, v in st['claimed_terminal_instances'].items()))}</td></tr>
<tr><th>冲突类型</th><td>{esc(' · '.join(f'{k}={v}' for k, v in st['conflict_type_instances'].items()))}</td></tr>
<tr><th>有 REAL 证据的案例</th><td>{st['cases_with_real_evidence']} / {st['cases']}</td></tr>
</tbody></table>

<h2>3. evidence_level 分布</h2>
{dist_table(st['evidence_level_instances'], total, '实例视图（entries）')}
{dist_table(st['evidence_level_canonical'], st['unique_artifacts'], '唯一内容视图（entries_canonical）')}

<h2>4. 案例台账（唯一内容视图）</h2>
<table><thead><tr><th>案例</th><th>唯一工件</th><th>等级构成</th><th>需重跑</th></tr></thead>
<tbody>{''.join(case_rows)}</tbody></table>

<h2>5. 冲突重裁（{len(conflicts)} 条实例 / {st['conflict_entries_canonical']} 条唯一内容）</h2>
<p class="dim">冲突 = 工件自述终态（verdict/status/overall_pass…）或历史文档终态（docs/sdaa_sweep/*.md）
与台账派生的证据等级矛盾。只重裁不重跑：证据缺口登记进第 6 节 needs_rerun 队列。</p>
<table><thead><tr><th>案例</th><th>工件</th><th>派生等级</th><th>冲突类型</th><th>自述</th>
<th>历史终态</th><th>conflict_reason</th><th>需重跑</th></tr></thead>
<tbody>{''.join(conflict_rows) or '<tr><td colspan=8 class=dim>无</td></tr>'}</tbody></table>

<h2>6. needs_rerun 队列（{len(ledger['needs_rerun'])} 条实例 / {st['needs_rerun_canonical']} 条唯一内容）</h2>
<table><thead><tr><th>案例</th><th>工件</th><th>冲突类型</th><th>建议动作</th></tr></thead>
<tbody>{''.join(rerun_rows) or '<tr><td colspan=4 class=dim>无</td></tr>'}</tbody></table>

<h2>7. 缺失字段（null + missing_reason，未做默认填充）</h2>
<table><thead><tr><th>字段</th><th>条目数</th><th>占比</th></tr></thead>
<tbody>{''.join(missing_rows)}</tbody></table>

<h2>8. 唯一内容工件明细（前 {min(200, st['unique_artifacts'])} 条）</h2>
<table><thead><tr><th>案例</th><th>工件</th><th>等级</th><th>自述终态</th><th>判据全过</th>
<th>结果类指标数</th><th>样例指标</th><th>定级理由</th></tr></thead>
<tbody>{''.join(sample_rows)}</tbody></table>

<footer>静态只读视图，由 <code>tools/sdaa_ledger_v2.py</code> 生成；不调用平台后端。
数据源 <code>artifacts/sdaa_ledger_v2.json</code>。</footer>
</body></html>
"""


# ---------------------------------------------------------------- CLI / 自检
def self_check(ledger: Dict[str, Any], html_path: Path) -> Tuple[bool, List[str]]:
    st = ledger["stats"]
    checks: List[Tuple[str, bool, str]] = [
        ("entries>=3000", st["entries"] >= 3000, f"entries={st['entries']}"),
        ("parse_ok>=0.90", st["parse_ok"] >= 0.90, f"parse_ok={st['parse_ok']}"),
        ("every entry has evidence_level",
         st["entries_without_evidence_level"] == 0,
         f"violations={st['entries_without_evidence_level']}"),
        ("conflicts have conflict_reason", st["conflict_entries_missing_reason"] == 0,
         f"missing={st['conflict_entries_missing_reason']} / conflicts={st['conflict_entries']}"),
        ("default_fill_count==0", st["default_fill_count"] == 0,
         f"default_fill_count={st['default_fill_count']}"),
        ("html exists & is html", html_path.is_file() and
         html_path.read_text(encoding="utf-8", errors="replace").lstrip().lower().startswith("<!doctype html"),
         f"path={html_path}"),
    ]
    ok = all(c[1] for c in checks)
    return ok, [f"[{'PASS' if c[1] else 'FAIL'}] {c[0]} ({c[2]})" for c in checks]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="SDAA 工件台账 v2")
    ap.add_argument("--root", default=str(REPO_ROOT), help="工作树根（默认脚本所在仓）")
    ap.add_argument("--worktree-glob", default=DEFAULT_WORKTREE_GLOB, help="其他 mission worktree artifacts 通配")
    ap.add_argument("--strict-local", action="store_true", help="只用本工作树 artifacts/")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-html", default=None)
    ap.add_argument("--check", action="store_true", help="只对现有台账跑机器判据自检")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    out_json = Path(args.out_json) if args.out_json else root / "artifacts" / "sdaa_ledger_v2.json"
    out_html = Path(args.out_html) if args.out_html else root / "artifacts" / "sdaa_ledger_v2.html"

    if args.check:
        if not out_json.is_file():
            print(f"[FAIL] ledger json missing: {out_json}")
            return 1
        ledger = json.loads(out_json.read_text(encoding="utf-8"))
        ok, lines = self_check(ledger, out_html)
    else:
        ledger = build(root, args.worktree_glob, args.strict_local)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(ledger, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        out_html.write_text(render_html(ledger), encoding="utf-8")
        ok, lines = self_check(ledger, out_html)
        if not args.quiet:
            st = ledger["stats"]
            print(f"ledger: {out_json} ({out_json.stat().st_size/1e6:.2f} MB)")
            print(f"html  : {out_html} ({out_html.stat().st_size/1e6:.2f} MB)")
            print(f"entries={st['entries']} unique={st['unique_artifacts']} "
                  f"parse_ok={st['parse_ok']} cases={st['cases']}")
            print("levels(instances):", st["evidence_level_instances"])
            print("levels(canonical):", st["evidence_level_canonical"])
            print(f"conflicts={st['conflict_entries']} (canonical {st['conflict_entries_canonical']}) "
                  f"needs_rerun={st['needs_rerun_entries']}")
            print("missing:", st["missing_field_histogram"])
    for line in lines:
        print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
