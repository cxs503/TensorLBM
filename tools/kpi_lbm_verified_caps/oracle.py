#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立双路径复算（oracle）：用一条单 CTE SQL 直接算 primary / strict，与 python 算核对拍。

python 路径：collect 行 → anchor_roster 聚合 → window_metrics
oracle 路径：数据库内 GROUP BY + NOT EXISTS，不经过任何 python 聚合逻辑

两条路径结果必须逐位相等（不一致即 exit 2，视为口径实现有缺陷）。
"""
from __future__ import annotations

import json
import subprocess

from . import collect, spec

_EXCL_ARRAY = ",".join("'%" + k + "%'" for k in spec.PHYSICS_EXCLUDE_KEYWORDS)


def _sql_for(start: str, end: str) -> str:
    repo = spec.REPO
    return f"""with v as (
  select id,
         mission_id,
         gate_level,
         created_at,
         coalesce(nullif(evidence->>'anchor_slug',''), regexp_replace(scope,'^m[0-9]+ ','')) as anchor,
         coalesce(substring(evidence->>'audited_mission' from '^[0-9]{{1,9}}$')::bigint, mission_id) as aud
    from tensor_maturity_verdicts
   where repo='{repo}'
     and gate_state='{spec.CREDIT_GATE_STATE}'
     and gate_level>={spec.CREDIT_MIN_LEVEL}
     and created_at between '{start}'::timestamptz and '{end}'::timestamptz
), phys as (
  select anchor, max(gate_level) as max_level, min(created_at) as first_at, array_agg(distinct aud) as auds
    from v
   where upper(anchor) not like all (array[{_EXCL_ARRAY}])
   group by anchor
)
select (select count(*) from phys) as n_primary,
       (select count(*) from phys p
         where exists (select 1
                         from unnest(p.auds) as a(mid)
                         join tensor_dev_missions m on m.id = a.mid
                        where m.finished_at between '{start}'::timestamptz and '{end}'::timestamptz)) as n_strict,
       (select count(*) from (select distinct anchor from v) s) as n_all_anchors,
       (select count(*) from v) as n_rows"""


def oracle_counts(start: str, end: str) -> dict:
    sql = _sql_for(start, end)
    rows = collect._q(sql, names=["n_primary", "n_strict", "n_all_anchors", "n_rows"])
    r = rows[0]
    return {
        "n_primary": int(r["n_primary"]),
        "n_strict": int(r["n_strict"]),
        "n_all_anchors": int(r["n_all_anchors"]),
        "n_rows": int(r["n_rows"]),
        "sql": " ".join(sql.split()),
    }


def compare(summary: dict) -> dict:
    prim = summary["primary"]
    oc = oracle_counts(prim["window_start_utc"], prim["window_end_utc"])
    got = {
        "n_primary": prim["current_primary"],
        "n_all_anchors": prim["current_all_anchors"],
        "n_strict": prim["current_strict"],
    }
    diffs = {k: {"oracle": oc[k], "python": got[k]} for k in got if oc[k] != got[k]}
    return {
        "oracle": oc,
        "python": got,
        "match": not diffs,
        "diffs": diffs,
        "exit_code": 0 if not diffs else 2,
    }


if __name__ == "__main__":  # pragma: no cover
    import sys

    s = json.load(sys.stdin)
    print(json.dumps(compare(s), ensure_ascii=False, indent=1))