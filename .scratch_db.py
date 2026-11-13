import sqlite3, sys, json
db = sqlite3.connect("/root/tp-agent/campaign.db")
db.row_factory = sqlite3.Row
c = db.cursor()
kw = ["TensorLBM", "lbm", "LBM"]
print("### capability/kb 同域")
for r in c.execute("SELECT id,ktype,domain,title,substr(content,1,220) AS c FROM knowledge_base WHERE status='active' AND (domain LIKE '%LBM%' OR title LIKE '%LBM%' OR title LIKE '%coexist%' OR title LIKE '%calibrat%') ORDER BY id DESC LIMIT 25"):
    print(dict(r))
print("\n### negative_knowledge_ledger")
try:
    for r in c.execute("SELECT * FROM negative_knowledge_ledger ORDER BY id DESC LIMIT 15"):
        d = dict(r)
        print({k: (str(v)[:150]) for k, v in d.items()})
except Exception as e:
    print("ERR", e)
print("\n### oss_solutions 同域")
try:
    cols = [x[1] for x in c.execute("PRAGMA table_info(oss_solutions)")]
    print("cols:", cols)
    for r in c.execute("SELECT * FROM oss_solutions ORDER BY id DESC LIMIT 40"):
        d = dict(r)
        print({k: str(v)[:120] for k, v in d.items()})
except Exception as e:
    print("ERR", e)