#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读取数层：把 spec 需要的每一段数据用一条显式 SQL 取回，SQL 原文随快照落盘。

纪律：
  * 每条语句必须是 SELECT/WITH（`_assert_readonly` 硬拦截），本模块不含任何写操作；
  * 不信任任何缓存台账，只取平台主库 tensorplatform 的原始行；
  * 每段结果按 (query_id, sql, n_rows, sha256(canonical rows)) 记账，便于双路径复算与审计。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Iterable

from . import spec

PG_BIN = "/usr/bin/psql"
PG_DB = os.environ.get("TP_PG_DB", "tensorplatform")
PG_ENV = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
_SEP = "\x01"


class ReadOnlyViolation(RuntimeError):
    pass


def _assert_readonly(sql: str) -> None:
    head = " ".join(sql.strip().lstrip("(").split()[:1]).upper()
    if head not in ("SELECT", "WITH"):
        raise ReadOnlyViolation(f"non-read-only statement rejected: {head!r}")


def _psql(sql: str) -> list[list[str]]:
    _assert_readonly(sql)
    proc = subprocess.run(
        ["runuser", "-u", "postgres", "--", PG_BIN, "-d", PG_DB, "-tA", "-F", _SEP, "-c", sql],
        capture_output=True,
        text=True,
        env=PG_ENV,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"psql failed rc={proc.returncode}: {proc.stderr.strip()[:400]}")
    rows = [ln.split(_SEP) for ln in proc.stdout.splitlines() if ln.strip() != ""]
    return rows


def _q(sql: str, names: list[str] | None = None) -> list[dict]:
    """执行并返回 dict 行（列名用 AS 显式给出，顺序即列顺序；复杂 CTE 可显式传 names）。"""
    raw = _psql(sql)
    cols = names or _col_names(sql)
    return [dict(zip(cols, r)) for r in raw]


def _col_names(sql: str) -> list[str]:
    """从 SQL 的顶层 select 列表解析列名（本模块 SQL 均为扁平简单列表，用 AS 显式命名）。"""
    body = sql.strip()
    low = body.lower()
    i = low.index("select")
    j = low.index("from")
    if low[:i].strip().startswith("with"):
        # WITH ... 形态：取最后一个顶层 select
        depth = 0
        last = None
        for k, ch in enumerate(body[: low.index("from")]):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif depth == 0 and body[k : k + 6].lower() == "select":
                last = k
        i = last
    cols = body[i + 6 : j]
    out = []
    depth = 0
    cur = ""
    for ch in cols:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    names = []
    for c in out:
        c = c.strip()
        if " as " in c.lower():
            names.append(c.lower().rsplit(" as ", 1)[1].strip().strip('"'))
        else:
            names.append(c.split(".")[-1].strip().strip('"'))
    return names


def build_queries(window_start: str, window_end: str) -> list[tuple[str, str, list[str]]]:
    """返回 [(query_id, sql, column_names)]。窗口用 ISO 8601 UTC 字面量注入（只读，无参数绑定需求）。

    列名一律显式给出：SQL 内联的 substring(... from '...') 会让「解析 select 列表」式取列名失准，
    显式列名是这里唯一可靠的做法（也是审计可核的）。
    """
    repo = spec.REPO
    return [
        (
            "Q1_ledger_bounds_repo",
            f"""select count(* ) as n_rows,
       coalesce(to_char(min(created_at),'YYYY-MM-DD HH24:MI:SS'),'') as min_created,
       coalesce(to_char(max(created_at),'YYYY-MM-DD HH24:MI:SS'),'') as max_created,
       coalesce(to_char(min(created_at),'YYYY-MM-DD'),'') as min_day
  from tensor_maturity_verdicts
 where repo='{repo}'""",
            ["n_rows", "min_created", "max_created", "min_day"],
        ),
        (
            "Q2_ledger_bounds_all_repos",
            """select count(*) as n_rows,
       coalesce(to_char(min(created_at),'YYYY-MM-DD'),'') as min_day,
       coalesce(to_char(max(created_at),'YYYY-MM-DD'),'') as max_day
  from tensor_maturity_verdicts""",
            ["n_rows", "min_day", "max_day"],
        ),
        (
            "Q3_verdicts_window",
            f"""select id::text as id,
       mission_id::text as mission_id,
       gate_state as gate_state,
       coalesce(gate_level::text,'') as gate_level,
       scope as scope,
       to_char(created_at,'YYYY-MM-DD"T"HH24:MI:SS.US') as created_at,
       coalesce(evidence->>'anchor_slug','') as evidence_anchor,
       coalesce(substring(evidence->>'audited_mission' from '^[0-9]{{1,9}}$'),'') as audited_mission,
       coalesce(evidence->>'verdict_summary','') as verdict_summary
  from tensor_maturity_verdicts
 where repo='{repo}'
   and created_at >= '{window_start}'::timestamptz
   and created_at <= '{window_end}'::timestamptz
 order by created_at, id""",
            ["id", "mission_id", "gate_state", "gate_level", "scope", "created_at", "evidence_anchor", "audited_mission", "verdict_summary"],
        ),
        (
            "Q4_verdicts_prior_confirmed_l3p",
            f"""select regexp_replace(scope,'^m[0-9]+ ','') as anchor,
       count(*) as n_rows,
       max(gate_level) as max_level
  from tensor_maturity_verdicts
 where repo='{repo}' and gate_state='confirmed' and gate_level>=3
   and created_at < '{window_start}'::timestamptz
 group by 1 order by 1""",
            ["anchor", "n_rows", "max_level"],
        ),
        (
            "Q5_missions_finished_window",
            f"""select id::text as id, status as status,
       coalesce(to_char(finished_at,'YYYY-MM-DD"T"HH24:MI:SS.US'),'') as finished_at
  from tensor_dev_missions
 where repo='{repo}' and finished_at >= '{window_start}'::timestamptz
   and finished_at <= '{window_end}'::timestamptz
 order by id""",
            ["id", "status", "finished_at"],
        ),
        (
            "Q6_missions_ref_by_verdicts",
            f"""select id::text as id, status as status,
       coalesce(to_char(finished_at,'YYYY-MM-DD"T"HH24:MI:SS.US'),'') as finished_at
  from tensor_dev_missions
 where id in (
   select mission_id from tensor_maturity_verdicts
    where repo='{repo}' and created_at >= '{window_start}'::timestamptz
   union
   select substring(evidence->>'audited_mission' from '^[0-9]{{1,9}}$')::bigint from tensor_maturity_verdicts
    where repo='{repo}' and created_at >= '{window_start}'::timestamptz
 )
 order by id""",
            ["id", "status", "finished_at"],
        ),
        (
            "Q7_referee_cases_lbm",
            f"""select anchor_slug as anchor, case_name as case_name, metric as metric,
       coalesce(tolerance::text,'') as tolerance, coalesce(status,'') as status
  from tensor_benchmark_cases
 where repo='{repo}' order by anchor_slug, case_name""",
            ["anchor", "case_name", "metric", "tolerance", "status"],
        ),
        (
            "Q8_referee_rows_for_kpi_anchor",
            f"""select count(*) as n_rows
  from tensor_benchmark_cases
 where anchor_slug='{spec.ANCHOR_SLUG}'""",
            ["n_rows"],
        ),
        (
            "Q9_feature_ledger_kpi_row",
            f"""select id::text as id, slug as slug, coalesce(status,'') as status,
       coalesce(to_char(done_at,'YYYY-MM-DD'),'') as done_at,
       coalesce(evidence,'') as evidence
  from feature_ledger where slug='{spec.ANCHOR_SLUG}'""",
            ["id", "slug", "status", "done_at", "evidence"],
        ),
        (
            "Q10_mission_status_window",
            f"""select status as status, count(*) as n_rows
  from tensor_dev_missions
 where repo='{repo}' and finished_at >= '{window_start}'::timestamptz
   and finished_at <= '{window_end}'::timestamptz
 group by 1 order by 2 desc""",
            ["status", "n_rows"],
        ),
    ]


def collect(window_start: str, window_end: str, windows: dict | None = None) -> dict:
    """执行全部只读 SQL，返回 snapshot（含 SQL 原文、行、指纹）。

    window_start = 采集起点（= min(ISO 周起点, now-7d)，保证主口径与敏感度口径都被覆盖）；
    windows      = 口径窗口字典（primary/rolling），原样入快照以保证可复算。
    """
    blocks = []
    for qid, sql, names in build_queries(window_start, window_end):
        rows = _q(sql, names)
        canon = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        blocks.append(
            {
                "query_id": qid,
                "sql": " ".join(sql.split()),
                "n_rows": len(rows),
                "rows_sha256": hashlib.sha256(canon.encode()).hexdigest(),
                "rows": rows,
            }
        )
    snap = {
        "repo": spec.REPO,
        "anchor_slug": spec.ANCHOR_SLUG,
        "spec_version": spec.SPEC_VERSION,
        "spec_sha256": spec.spec_sha256(),
        "window_start_utc": window_start,
        "window_end_utc": window_end,
        "windows": windows or {},
        "collector": "tools.kpi_lbm_verified_caps.collect (read-only SELECT only)",
        "blocks": blocks,
    }
    blob = json.dumps(snap, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    snap["snapshot_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    return snap


def block(snap: dict, qid: str) -> list[dict]:
    for b in snap["blocks"]:
        if b["query_id"] == qid:
            return b["rows"]
    raise KeyError(qid)


def first(snap: dict, qid: str) -> dict:
    rows = block(snap, qid)
    return rows[0] if rows else {}


def assert_all_readonly(snap: dict) -> None:
    for b in snap["blocks"]:
        _assert_readonly(b["sql"])


def iterate_sql(snap: dict) -> Iterable[tuple[str, str]]:
    for b in snap["blocks"]:
        yield b["query_id"], b["sql"]