import psycopg, json

con = psycopg.connect("dbname=tensorplatform user=postgres", connect_timeout=8)
cur = con.cursor()

print("=== 1. negative_knowledge_ledger keywords=TensorLBM ===")
cur.execute("select * from negative_knowledge_ledger where keywords like %s order by id desc limit 15", ("%TensorLBM%",))
cols = [d[0] for d in cur.description]
for row in cur.fetchall():
    d = dict(zip(cols, row))
    print(json.dumps(d, ensure_ascii=False, default=str)[:1500])
    print("---")

print("=== 2. oss_inventory feature_slug ILIKE %RATIO% ===")
cur.execute("select * from oss_inventory where feature_slug ilike %s", ("%RATIO%",))
cols = [d[0] for d in cur.description]
for row in cur.fetchall():
    d = dict(zip(cols, row))
    print(json.dumps(d, ensure_ascii=False, default=str)[:1200])
    print("---")

print("=== 3. tensor_xval_cases anchor ===")
cur.execute("select * from tensor_xval_cases where anchor ilike %s", ("%MAXWELL%",))
cols = [d[0] for d in cur.description]
for row in cur.fetchall():
    d = dict(zip(cols, row))
    print(json.dumps(d, ensure_ascii=False, default=str)[:2000])
    print("---")
