"""开工三查：knowledge_base / negative_knowledge_ledger / oss_solutions（只读）。"""
import os

import psycopg

DB = os.environ.get("TP_PG_DB", "tensorplatform")


def q(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def trunc(v, n=200):
    return str(v).replace("\n", " ")[:n]


with psycopg.connect(f"dbname={DB} user=postgres", connect_timeout=8) as conn:
    cur = conn.cursor()

    print("== 1. knowledge_base (LBM|cylinder|strouhal|vortex|karman) ==")
    rows = q(
        cur,
        """select id, kb_key, ktype, domain, title, content
           from knowledge_base
           where (coalesce(title,'')||' '||coalesce(content,'')||' '||coalesce(domain,'')||' '||coalesce(kb_key,''))
                 ~* '(lbm|lattice.?boltzmann|cylinder|strouhal|vortex|karman|re=100|re100)'
           order by id desc limit 12""",
    )
    for r in rows:
        print(f"- [{r['id']}] {r['kb_key']} ({r['ktype']}/{r['domain']}) {trunc(r['title'],80)} :: {trunc(r['content'],160)}")
    print("count:", len(rows))

    print("\n== 2. negative_knowledge_ledger (TensorLBM) ==")
    rows2 = q(
        cur,
        """select id, title, hypothesis, failure_reason, failure_kind, keywords, domain_tags, status
           from negative_knowledge_ledger
           where (coalesce(title,'')||' '||coalesce(hypothesis,'')||' '||coalesce(failure_reason,'')
                  ||' '||coalesce(keywords::text,'')||' '||coalesce(domain_tags::text,'')) ~* 'tensorlbm'
           order by id desc limit 25""",
    )
    for r in rows2:
        print(f"- [{r['id']}] {trunc(r['title'],90)} | {r['failure_kind']} | {trunc(r['failure_reason'],140)} | kw={trunc(r['keywords'],60)}")
    print("count:", len(rows2))

    print("\n== 3. oss_solutions (LBM|cylinder|vortex|strouhal) ==")
    rows3 = q(
        cur,
        """select id, repo, feature_slug, tag, method, conventions
           from oss_solutions
           where (coalesce(feature_slug,'')||' '||coalesce(tag,'')||' '||coalesce(method,'')
                  ||' '||coalesce(conventions,'')||' '||coalesce(repo,''))
                 ~* '(lbm|lattice.?boltzmann|cylinder|vortex|strouhal|karman)'
           order by id desc limit 15""",
    )
    for r in rows3:
        print(f"- [{r['id']}] {r['repo']}/{r['feature_slug']} tag={r['tag']} :: {trunc(r['method'],120)}")
    print("count:", len(rows3))