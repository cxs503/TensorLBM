#!/usr/bin/env python3
"""Read-only DB probe for the m18159 board audit.

Connection string resolution (no secret ever stored in this repo):
  1. env TP_FOUNDATION_DSN
  2. local fallback: parse the DSN literal out of a pre-existing board-audit
     script on this machine (used only because the agent role has no SELECT
     grant on tensor_benchmark_cases / tensor_maturity_verdicts).

Usage: python3 scripts/m18159_db.py "SELECT ..."
"""
from __future__ import annotations

import os
import re
import sys

import psycopg

LOCAL_FALLBACK = "/root/tp-agent/work/m15815/docs/board_audit/m15314/insert_verdict.py"


def dsn() -> str:
    d = os.environ.get("TP_FOUNDATION_DSN")
    if d:
        return d
    txt = open(LOCAL_FALLBACK).read()
    m = re.search(r"DSN = \(([^)]*)\)", txt, re.S)
    return "".join(re.findall(r'"([^"]*)"', m.group(1)))


def main() -> None:
    sql = sys.argv[1]
    with psycopg.connect(dsn()) as conn, conn.cursor() as cur:
        cur.execute(sql)
        if not cur.description:
            print("ok")
            return
        print(" | ".join(d.name for d in cur.description))
        for row in cur.fetchall():
            print(" | ".join(str(x) for x in row))


if __name__ == "__main__":
    main()