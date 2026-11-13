import psycopg, json

con = psycopg.connect("dbname=tensorplatform user=postgres", connect_timeout=8)
cur = con.cursor()

print("=== oss_inventory: multiphase / eos / maxwell / ratio related ===")
cur.execute("""select id, repo, tier, focus, status, methods, multiphase, cache_nodes, cache_commit
               from oss_inventory
               where repo ilike %s or focus ilike %s or methods ilike %s or multiphase ilike %s
                  or summary ilike %s or task_key ilike %s
               order by id""",
            ("%lbmpy%", "%maxwell%", "%maxwell%", "%eos%", "%maxwell%", "%ratio%"))
c = [d[0] for d in cur.description]
for row in cur.fetchall():
    print(json.dumps(dict(zip(c, row)), ensure_ascii=False, default=str)[:700])

print("=== oss_solutions existing for this feature ===")
cur.execute("select id, repo, feature_slug, tag, left(method,200) from oss_solutions where feature_slug ilike %s", ("%MAXWELL%",))
c = [d[0] for d in cur.description]
for row in cur.fetchall():
    print(json.dumps(dict(zip(c, row)), ensure_ascii=False, default=str)[:800])

print("=== knowledge_base: eos/maxwell/coexistence/density ===")
cur.execute("""select kb_key, ktype, domain, title, left(content,300) from knowledge_base
               where title ilike %s or content ilike %s or kb_key ilike %s
               order by id desc limit 20""",
            ("%coexistence%", "%maxwell%", "%eos%"))
c = [d[0] for d in cur.description]
for row in cur.fetchall():
    print(json.dumps(dict(zip(c, row)), ensure_ascii=False, default=str)[:600])
