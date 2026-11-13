"""Board-audit DB helper for m17855 (read benchmark cases / missions, insert verdict).

Credentials from the platform .env (never inlined on a command line).
Usage:
  python3 scripts/dbq_m17855.py bench <anchor_slug>
  python3 scripts/dbq_m17855.py mission <mission_id>
  python3 scripts/dbq_m17855.py reports <mission_id>
  python3 scripts/dbq_m17855.py insert <verdict.json>
  python3 scripts/dbq_m17855.py read <scope_like>
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


def bench(slug):
    with connect() as conn:
        cur = conn.execute(
            "SELECT case_name, metric, tolerance, anchor_slug FROM tensor_benchmark_cases"
            " WHERE anchor_slug = %s", (slug,))
        cols = [c.name for c in cur.description]
        for row in cur.fetchall():
            print(json.dumps(dict(zip(cols, row)), ensure_ascii=False, default=str))


def mission(mid):
    with connect() as conn:
        cur = conn.execute(
            "SELECT column_name FROM information_schema.columns"
            " WHERE table_name='missions' AND column_name IN"
            " ('id','status','repo','title','goal','result','verdict','branch','parent_id','created_at')")
        cols = [r[0] for r in cur.fetchall()]
        q = f"SELECT {','.join(cols)} FROM missions WHERE id=%s"
        cur = conn.execute(q, (mid,))
        for row in cur.fetchall():
            print(json.dumps(dict(zip(cols, [str(x)[:4000] for x in row])), ensure_ascii=False))


def reports(mid):
    with connect() as conn:
        cur = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='tensor_reports'")
        cols = [r[0] for r in cur.fetchall()]
        print("cols:", cols, file=sys.stderr)
        cur = conn.execute(
            "SELECT id, title, mission_id, created_at FROM tensor_reports"
            " WHERE mission_id=%s ORDER BY id", (mid,))
        for row in cur.fetchall():
            print(json.dumps([str(x)[:300] for x in row], ensure_ascii=False))


def read(scope_like):
    with connect() as conn:
        cur = conn.execute(
            "SELECT id, repo, scope, mission_id, level, verdict_by, gate_state,"
            " gate_level, gate_note FROM tensor_maturity_verdicts"
            " WHERE scope ILIKE %s ORDER BY id DESC LIMIT 10", (f"%{scope_like}%",))
        for row in cur.fetchall():
            print(json.dumps(list(row), ensure_ascii=False, default=str)[:1500])


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
        print(json.dumps(cur.fetchone(), ensure_ascii=False, default=str))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "bench":
        bench(sys.argv[2])
    elif cmd == "mission":
        mission(int(sys.argv[2]))
    elif cmd == "reports":
        reports(int(sys.argv[2]))
    elif cmd == "insert":
        insert(sys.argv[2])
    elif cmd == "read":
        read(sys.argv[2] if len(sys.argv) > 2 else "")
    else:
        sys.exit(f"unknown cmd {cmd}")