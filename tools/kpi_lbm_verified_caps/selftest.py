#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""口径自检：用合成行证明「分子/分母真参与计算」、去重与窗口边界真生效。

每个 case 都是可判真假的最小形态（不是跑一遍代码就算过）：
  S1  空集               → current_primary = 0
  S2  一条 confirmed L3   → 1
  S3  reopened L3         → 0（reopened = gate 未确认）
  S4  confirmed L2        → 0（门槛 L3）
  S5  同锚两行 confirmed  → 1（去重）+ max_level 取高
  S6  窗口外一行 confirmed → 0（窗口边界真生效）
  S7  管道锚(TEMPLATE)    → primary 0 / all_anchors 1（物理锚判定真生效）
  S8  产出任务在窗口外     → strict 0 / loose 1
  S9  首次确认时刻取最早一条（与行序无关）
  S10 ISO 周起点 = 周一 00:00 UTC
  S11 normalize_anchor 优先 evidence.anchor_slug
  S12 只读护栏拒绝写语句
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from . import collect, compute

GPU = timezone.utc
T0 = datetime(2026, 11, 9, 0, 0, tzinfo=GPU)  # 周一
T1 = datetime(2026, 11, 14, 1, 0, tzinfo=GPU)  # 采集时刻


def _row(vid, anchor, level, state="confirmed", ts=None, mission=1, audited="", scope=None):
    return {
        "id": str(vid),
        "mission_id": str(mission),
        "gate_state": state,
        "gate_level": str(level) if level is not None else "",
        "scope": scope if scope is not None else f"m{mission} {anchor}",
        "created_at": (ts or (T0 + timedelta(days=1))).strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "evidence_anchor": anchor,
        "audited_mission": audited,
        "verdict_summary": "",
    }


def _metrics(rows, missions=None, start=T0, end=T1):
    roster = compute.anchor_roster(rows)
    return compute.window_metrics(roster, missions or {}, start, end)


def run() -> dict:
    cases = []

    def check(name, cond, detail):
        cases.append({"case": name, "pass": bool(cond), "detail": detail})

    m = _metrics([])
    check("S1_empty_is_zero", m["current_primary"] == 0, f"primary={m['current_primary']}")

    m = _metrics([_row(1, "XVAL-LBM-A", 3)])
    check("S2_one_confirmed_l3", m["current_primary"] == 1, f"primary={m['current_primary']}")

    m = _metrics([_row(1, "XVAL-LBM-A", 3, state="reopened")])
    check("S3_reopened_not_credited", m["current_primary"] == 0, f"primary={m['current_primary']}")

    m = _metrics([_row(1, "XVAL-LBM-A", 2)])
    check("S4_l2_below_floor", m["current_primary"] == 0, f"primary={m['current_primary']}")

    m = _metrics([_row(1, "XVAL-LBM-A", 3), _row(2, "XVAL-LBM-A", 4)])
    check(
        "S5_dedupe_by_anchor",
        m["current_primary"] == 1 and m["anchors"][0]["max_level"] == 4,
        f"primary={m['current_primary']} max_level={m['anchors'][0]['max_level'] if m['anchors'] else None}",
    )

    m = _metrics([_row(1, "XVAL-LBM-A", 3, ts=T0 - timedelta(hours=1))])
    check("S6_window_boundary", m["current_primary"] == 0, f"primary={m['current_primary']}")

    m = _metrics([_row(1, "XVAL-LBM-YML-TEMPLATE-PIPE", 4)])
    check(
        "S7_plumbing_excluded",
        m["current_primary"] == 0 and m["current_all_anchors"] == 1,
        f"primary={m['current_primary']} all={m['current_all_anchors']}",
    )

    rows = [_row(1, "XVAL-LBM-A", 4, mission=10, audited="10")]
    m = _metrics(rows, missions={"10": (T0 - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S.%f")})
    check(
        "S8_strict_excludes_old_delivery",
        m["current_strict"] == 0 and m["current_loose"] == 1,
        f"strict={m['current_strict']} loose={m['current_loose']}",
    )

    early = T0 + timedelta(days=1, hours=1)
    late = T0 + timedelta(days=2)
    rows = [_row(2, "XVAL-LBM-A", 3, ts=late), _row(1, "XVAL-LBM-A", 3, ts=early)]
    m = _metrics(rows)
    got = m["anchors"][0]["first_confirmed_at"]
    check("S9_first_confirmed_is_earliest", got.startswith(early.strftime("%Y-%m-%dT%H:%M:%S")), f"first={got}")

    ws = compute.iso_week_start(datetime(2026, 11, 14, 1, 0, tzinfo=GPU))
    check("S10_iso_week_monday", ws == T0, f"week_start={ws.isoformat()}")

    check(
        "S11_anchor_from_evidence",
        compute.normalize_anchor("m123 CASE-X", "CASE-Y") == "CASE-Y"
        and compute.normalize_anchor("m123 CASE-X", "") == "CASE-X",
        "evidence_slug preferred, else scope stripped",
    )

    rejected = False
    try:
        collect._assert_readonly("INSERT INTO feature_ledger VALUES (1)")
    except collect.ReadOnlyViolation:
        rejected = True
    ok_select = True
    try:
        collect._assert_readonly("SELECT 1")
        collect._assert_readonly("WITH a AS (SELECT 1) SELECT * FROM a")
    except collect.ReadOnlyViolation:
        ok_select = False
    check("S12_readonly_guard", rejected and ok_select, f"insert_rejected={rejected} select_allowed={ok_select}")

    n_fail = sum(1 for c in cases if not c["pass"])
    return {"cases": cases, "n_cases": len(cases), "n_fail": n_fail, "fails": [c for c in cases if not c["pass"]]}


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=1))