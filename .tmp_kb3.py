import psycopg, json

con = psycopg.connect("dbname=tensorplatform user=postgres", connect_timeout=8)
cur = con.cursor()

def cols_of(t):
    cur.execute("select column_name from information_schema.columns where table_name=%s order by ordinal_position", (t,))
    return [r[0] for r in cur.fetchall()]

for t in ["oss_inventory", "tensor_xval_cases", "negative_knowledge_ledger", "oss_solutions", "knowledge_base"]:
    print(t, "->", cols_of(t))

print("=== oss_inventory sample ===")
cur.execute("select * from oss_inventory limit 3")
c = [d[0] for d in cur.description]
for row in cur.fetchall():
    print(json.dumps(dict(zip(c, row)), ensure_ascii=False, default=str)[:800])

print("=== xval cases maxwell ===")
cur.execute("select * from tensor_xval_cases where anchor ilike %s or anchor ilike %s", ("%MAXWELL%", "%DENSITY-RATIO%"))
c = [d[0] for d in cur.description]
for row in cur.fetchall():
    print(json.dumps(dict(zip(c, row)), ensure_ascii=False, default=str)[:2500])
