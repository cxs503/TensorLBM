"""m18311 board audit — register the machine-referee rows and the maturity verdict.

Idempotent: benchmark rows are keyed by (anchor_slug, case_name); the verdict row is
keyed by (repo, scope, verdict_by) and updated in place.

Run: python3 docs/board_audit/m18311/insert_rows.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import psycopg

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPRO = json.loads((HERE / "audit_repro.json").read_text())
ROWS = json.loads((HERE / "benchmark_rows.json").read_text())
EVIDENCE = json.loads((HERE / "verdict_evidence.json").read_text())

DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    with psycopg.connect(DSN) as c:
        cur = c.cursor()
        for r in ROWS:
            cur.execute(
                "SELECT id FROM tensor_benchmark_cases WHERE anchor_slug=%s AND case_name=%s",
                (r["anchor_slug"], r["case_name"]))
            hit = cur.fetchone()
            if hit:
                print("benchmark row exists id=", hit[0], r["case_name"][:50])
                continue
            if a.dry_run:
                print("WOULD INSERT benchmark:", r["case_name"][:60], r["status"])
                continue
            cur.execute(
                "INSERT INTO tensor_benchmark_cases "
                "(anchor_slug, case_name, ref_source, metric, tolerance, repo, status, passed_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s, CASE WHEN %s='passed' THEN now() ELSE NULL END) "
                "RETURNING id",
                (r["anchor_slug"], r["case_name"], r["ref_source"], r["metric"],
                 r["tolerance"], r["repo"], r["status"], r["status"]))
            print("INSERTED benchmark id=", cur.fetchone()[0], r["status"], r["case_name"][:45])

        scope = EVIDENCE["scope"]
        cur.execute("SELECT id, level FROM tensor_maturity_verdicts "
                    "WHERE repo=%s AND scope=%s AND verdict_by=%s",
                    (EVIDENCE["repo"], scope, EVIDENCE["verdict_by"]))
        hit = cur.fetchone()
        ev = json.dumps(EVIDENCE["evidence"], ensure_ascii=False)
        if a.dry_run:
            print("WOULD upsert verdict", EVIDENCE["repo"], scope, "L", EVIDENCE["level"])
            return
        if hit:
            cur.execute("UPDATE tensor_maturity_verdicts SET level=%s, mission_id=%s, evidence=%s::jsonb "
                        "WHERE id=%s",
                        (EVIDENCE["level"], EVIDENCE["mission_id"], ev, hit[0]))
            print("UPDATED verdict id=", hit[0], "-> L", EVIDENCE["level"])
        else:
            cur.execute("INSERT INTO tensor_maturity_verdicts "
                        "(repo, scope, mission_id, level, evidence, verdict_by) "
                        "VALUES (%s,%s,%s,%s,%s::jsonb,%s) RETURNING id",
                        (EVIDENCE["repo"], scope, EVIDENCE["mission_id"], EVIDENCE["level"],
                         ev, EVIDENCE["verdict_by"]))
            print("INSERTED verdict id=", cur.fetchone()[0])
        c.commit()


if __name__ == "__main__":
    main()