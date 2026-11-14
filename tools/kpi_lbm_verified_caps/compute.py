#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯函数算核：把 collect 的快照行算成 KPI 现值、敏感度、缺口。

本模块不碰数据库、不读环境，全部函数对输入行是纯映射——便于用合成行做 selftest
（证明分子/分母真参与计算、去重与窗口边界真生效），也便于 oracle 用另一条路径对拍。
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from . import spec

_MISSION_PREFIX = re.compile(r"^m[0-9]+\s+")


def parse_ts(s: str):
    """'2026-11-13T17:28:13.130081' 或带 +00 → aware datetime(UTC)；空→None。"""
    if not s:
        return None
    s = s.strip().replace(" ", "T")
    if s.endswith("+00"):
        s = s[:-3] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso_week_start(now: datetime) -> datetime:
    midnight = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight - timedelta(days=midnight.weekday())


def normalize_anchor(scope: str, evidence_anchor: str = "") -> str:
    """锚名：优先 evidence->>'anchor_slug'，否则去掉 scope 的 'm<id> ' 前导。"""
    if evidence_anchor and evidence_anchor.strip():
        return evidence_anchor.strip()
    return _MISSION_PREFIX.sub("", (scope or "").strip()).strip()


def physics_exclusions(anchor: str) -> list[str]:
    up = (anchor or "").upper()
    return [k for k in spec.PHYSICS_EXCLUDE_KEYWORDS if k in up]


def is_physics(anchor: str) -> bool:
    return not physics_exclusions(anchor)


def _level(v) -> int:
    try:
        return int(str(v).strip() or -1)
    except ValueError:
        return -1


def credited(row: dict) -> bool:
    """单行 verdict 是否构成入账（gate confirmed 且 L3+）。"""
    return row.get("gate_state") == spec.CREDIT_GATE_STATE and _level(row.get("gate_level")) >= spec.CREDIT_MIN_LEVEL


def anchor_roster(rows: list[dict]) -> dict:
    """按 anchor 聚合窗口内 verdict 行 → {anchor: {...}}（同锚多行只留 max level 与首次确认时刻）。"""
    roster: dict[str, dict] = {}
    for r in rows:
        a = normalize_anchor(r.get("scope", ""), r.get("evidence_anchor", ""))
        if not a:
            continue
        rec = roster.setdefault(
            a,
            {
                "anchor": a,
                "n_verdict_rows": 0,
                "states": {},
                "levels": [],
                "credited_rows": [],
                "first_confirmed_at": None,
                "max_level": -1,
                "verdict_ids": [],
                "producer_missions": set(),
                "verdict_missions": set(),
                "all_rows": [],
            },
        )
        rec["n_verdict_rows"] += 1
        st = r.get("gate_state") or ""
        rec["states"][st] = rec["states"].get(st, 0) + 1
        rec["verdict_ids"].append(int(r["id"]))
        if r.get("mission_id"):
            rec["verdict_missions"].add(int(r["mission_id"]))
        lvl = _level(r.get("gate_level"))
        rec["all_rows"].append((parse_ts(r.get("created_at", "")) or datetime.min.replace(tzinfo=timezone.utc), int(r["id"]), st, lvl))
        if lvl >= 0:
            rec["levels"].append(lvl)
        if credited(r):
            ts = parse_ts(r.get("created_at", ""))
            rec["credited_rows"].append({"verdict_id": int(r["id"]), "level": lvl, "created_at": ts})
            if lvl > rec["max_level"]:
                rec["max_level"] = lvl
            if ts and (rec["first_confirmed_at"] is None or ts < rec["first_confirmed_at"]):
                rec["first_confirmed_at"] = ts
            aud = (r.get("audited_mission") or "").strip()
            rec["producer_missions"].add(int(aud) if aud else int(r["mission_id"] or 0))
    for rec in roster.values():
        rec["credit"] = rec["first_confirmed_at"] is not None
        # 最新一行 verdict 才是该锚的当前判级：若之后被 reopened 或降级，则本轮不成立（防「先确认后推翻」）
        last = max(rec["all_rows"], key=lambda t: (t[0], t[1])) if rec["all_rows"] else None
        rec["latest_state"] = last[2] if last else ""
        rec["latest_level"] = last[3] if last else -1
        rec["latest_verdict_id"] = last[1] if last else None
        rec["invalidated_by_later_verdict"] = False
        if rec["credit"] and not (rec["latest_state"] == spec.CREDIT_GATE_STATE and rec["latest_level"] >= spec.CREDIT_MIN_LEVEL):
            rec["credit"] = False
            rec["invalidated_by_later_verdict"] = True
        rec["levels"] = sorted(rec["levels"])
        rec["producer_missions"] = sorted(x for x in rec["producer_missions"] if x)
        rec["verdict_missions"] = sorted(rec["verdict_missions"])
        rec["physics"] = is_physics(rec["anchor"])
        rec["physics_exclusion_hits"] = physics_exclusions(rec["anchor"])
        rec["first_confirmed_at"] = rec["first_confirmed_at"].isoformat() if rec["first_confirmed_at"] else None
    return roster


def _in_window(iso_ts: str | None, start: datetime, end: datetime) -> bool:
    dt = parse_ts(iso_ts) if iso_ts else None
    return bool(dt and start <= dt <= end)


def window_metrics(roster: dict, missions: dict, start: datetime, end: datetime) -> dict:
    """给定窗口 [start, end]，按锚计新增 L3+ 物理能力。missions={id: finished_at iso}。"""
    in_win = {a: r for a, r in roster.items() if r.get("credit") and _in_window(r["first_confirmed_at"], start, end)}
    invalidated = sorted(a for a, r in roster.items() if r.get("invalidated_by_later_verdict"))
    all_anchors = sorted(in_win)
    physics = sorted(a for a in all_anchors if in_win[a]["physics"])
    excluded = sorted(set(all_anchors) - set(physics))

    strict, loose_only = [], []
    for a in physics:
        r = in_win[a]
        prod_in = any(_in_window(missions.get(str(m)) or "", start, end) for m in r["producer_missions"])
        (strict if prod_in else loose_only).append(a)

    lvl_hist: dict[str, int] = {}
    for a in physics:
        lvl_hist[str(in_win[a]["max_level"])] = lvl_hist.get(str(in_win[a]["max_level"]), 0) + 1

    return {
        "window_start_utc": start.isoformat(),
        "window_end_utc": end.isoformat(),
        "current_primary": len(physics),
        "current_all_anchors": len(all_anchors),
        "current_strict": len(strict),
        "current_loose": len(physics),
        "loose_only_anchors": loose_only,
        "physics_excluded_anchors": excluded,
        "invalidated_by_later_verdict": invalidated,
        "level_histogram": dict(sorted(lvl_hist.items())),
        "n_level4plus": sum(v for k, v in lvl_hist.items() if int(k) >= 4),
        "anchors": [
            {
                "anchor": a,
                "max_level": in_win[a]["max_level"],
                "first_confirmed_at": in_win[a]["first_confirmed_at"],
                "producer_missions": in_win[a]["producer_missions"],
                "producer_finished_at": [missions.get(str(m)) for m in in_win[a]["producer_missions"]],
                "verdict_ids": in_win[a]["verdict_ids"],
                "verdict_rows_in_window": in_win[a]["n_verdict_rows"],
                "states": in_win[a]["states"],
                "latest_state": in_win[a]["latest_state"],
                "latest_level": in_win[a]["latest_level"],
                "latest_verdict_id": in_win[a]["latest_verdict_id"],
                "physics": in_win[a]["physics"],
            }
            for a in all_anchors
        ],
    }


def referee_coverage(physics_anchors: list[str], cases: list[dict]) -> dict:
    by_anchor: dict[str, list[dict]] = {}
    for c in cases:
        by_anchor.setdefault(c.get("anchor", ""), []).append(c)
    covered = {
        a: [
            {"case_name": c.get("case_name", ""), "metric": c.get("metric", ""), "tolerance": c.get("tolerance", ""), "status": c.get("status", "")}
            for c in by_anchor.get(a, [])
        ]
        for a in physics_anchors
    }
    n_cov = sum(1 for a in physics_anchors if covered[a])
    return {
        "n_anchors": len(physics_anchors),
        "n_with_referee_rows": n_cov,
        "coverage_ratio": round(n_cov / len(physics_anchors), 4) if physics_anchors else None,
        "per_anchor": covered,
        "uncovered_anchors": [a for a in physics_anchors if not covered[a]],
    }


def conversion(roster: dict, start: datetime, end: datetime) -> dict:
    """本周被审计的全部锚（任何状态）→ 其中 bump 到 confirmed L3+ 的比例（转化率诊断）。"""
    seen = [r for r in roster.values() if r["n_verdict_rows"] > 0]
    confirmed_any = [r for r in seen if r["credit"] and _in_window(r["first_confirmed_at"], start, end)]
    reopened = [r for r in seen if r["states"].get("reopened", 0) > 0]
    below = [r for r in confirmed_any if not r["physics"]]
    return {
        "n_assets_with_verdicts": len(seen),
        "n_confirmed_l3p_any_domain": len(confirmed_any),
        "n_confirmed_l3p_physics": len(confirmed_any) - len(below),
        "n_with_reopened_rows": len(reopened),
        "ratio": round((len(confirmed_any) - len(below)) / len(seen), 4) if seen else None,
    }


def summarize(snap: dict) -> dict:
    """快照 → 全部计量指标（primary + sensitivity + 诊断）。"""
    from . import collect  # 局部导入避免循环

    rows = collect.block(snap, "Q3_verdicts_window")
    missions = {r["id"]: r.get("finished_at", "") for r in collect.block(snap, "Q6_missions_ref_by_verdicts")}
    cases = collect.block(snap, "Q7_referee_cases_lbm")

    wins = snap.get("windows") or {}
    end = parse_ts(wins.get("window_end_utc", snap["window_end_utc"]))
    primary_start = parse_ts(wins.get("primary_start_utc", snap["window_start_utc"]))
    rolling_start = parse_ts(wins.get("rolling_start_utc", snap["window_start_utc"]))

    roster = anchor_roster(rows)
    primary = window_metrics(roster, missions, primary_start, end)
    rolling = window_metrics(roster, missions, rolling_start, end)

    physics_anchors = [a["anchor"] for a in primary["anchors"] if a["physics"]]
    refc = referee_coverage(physics_anchors, cases)

    return {
        "spec": spec.spec_dict(),
        "spec_sha256": spec.spec_sha256(),
        "snapshot_sha256": snap["snapshot_sha256"],
        "collected_at_utc": snap["window_end_utc"],
        "primary": primary,
        "rolling_7d": rolling,
        "referee_coverage": refc,
        "conversion": conversion(roster, primary_start, end),
        "roster_size": len(roster),
        "prior_confirmed_l3p_rows": len(collect.block(snap, "Q4_verdicts_prior_confirmed_l3p")),
    }