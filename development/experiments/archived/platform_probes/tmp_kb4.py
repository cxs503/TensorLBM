import psycopg, json

con = psycopg.connect("dbname=tensorplatform user=postgres", connect_timeout=8)
cur = con.cursor()

print("=== xval cases maxwell ===")
cur.execute("select * from tensor_xval_cases where anchor_slug ilike %s", ("%MAXWELL%",))
c = [d[0] for d in cur.description]
for row in cur.fetchall():
    print(json.dumps(dict(zip(c, row)), ensure_ascii=False, default=str)[:3000])

print("=== mission 18132 ===")
cur.execute("select * from tensor_dev_missions where id=18132")
c = [d[0] for d in cur.description]
for row in cur.fetchall():
    d = dict(zip(c, row))
    for k, v in d.items():
        print(k, ":", str(v)[:1200])

print("=== mission 18225 (this) ===")
cur.execute("select * from tensor_dev_missions where id=18225")
c = [d[0] for d in cur.description]
for row in cur.fetchall():
    d = dict(zip(c, row))
    for k, v in d.items():
        print(k, ":", str(v)[:1500])
