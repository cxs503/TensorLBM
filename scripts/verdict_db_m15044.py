"""Read/insert tensor_maturity_verdicts for the m15044 board-audit mission.

Credentials come from the platform .env (never inlined on a command line).
Usage:
  python3 scripts/verdict_db_m15044.py schema
  python3 scripts/verdict_db_m15044.py read <scope_like>
  python3 scripts/verdict_db_m15044.py insert <verdict.json>
"""
from __future__ import annotations

import json
import re
import sys

ENV = "/root/tensor-ecosystem/TensorPlatform/.env"


def dsn():
    url = None
    for line in open(ENV):
        if line.startswith("DATABASE_URL="):
            url = line.split("=", 1)[1].strip()
    assert url, "DATABASE_URL not found"
    m = re.match(r"postgresql\+\w+://([^:]+):([^@]+)@([^/]+)/(\S+)", url)
    user, pw, host, db = m.groups()
    db = db.split("?")[0]
    host, _, port = host.partition(":")
    return dict(host=host or "127.0.0.1", port=int(port or 5432),
                user=user, password=pw, dbname=db)


def connect():
    import psycopg
    return psycopg.connect(**dsn(), autocommit=True)


def schema():
    with connect() as conn:
        for row in conn.execute(
                "SELECT column_name, data_type FROM information_schema.columns"
                " WHERE table_name = 'tensor_maturity_verdicts' ORDER BY ordinal_position"):
            print(row)


def read(scope_like):
    with connect() as conn:
        cur = conn.execute(
            "SELECT id, repo, scope, mission_id, level, verdict_by, gate_state,"
            " gate_level, gate_note FROM tensor_maturity_verdicts"
            " WHERE scope ILIKE %s ORDER BY id DESC LIMIT 10", (f"%{scope_like}%",))
        for row in cur.fetchall():
            print(json.dumps(list(row), ensure_ascii=False)[:800])


def insert(path):
    payload = json.load(open(path))
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO tensor_maturity_verdicts"
            " (repo, scope, mission_id, level, evidence, verdict_by)"
            " VALUES (%s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (repo, scope, verdict_by) DO UPDATE"
            " SET level = EXCLUDED.level, evidence = EXCLUDED.evidence,"
            " mission_id = EXCLUDED.mission_id"
            " RETURNING id, repo, scope, mission_id, level, verdict_by, gate_state",
            (payload["repo"], payload["scope"], payload["mission_id"],
             payload["level"], json.dumps(payload["evidence"], ensure_ascii=False),
             payload["verdict_by"]))
        print(json.dumps(cur.fetchone(), ensure_ascii=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "schema":
        schema()
    elif cmd == "read":
        read(sys.argv[2] if len(sys.argv) > 2 else "")
    elif cmd == "insert":
        insert(sys.argv[2])
    else:
        raise SystemExit("unknown command")