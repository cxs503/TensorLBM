import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import call, login

st, tok, _ = login()
st, d = call("tensor/missions?limit=500", token=tok)
items = d.get("items", [])
print("n missions", len(items))
for m in items:
    if "P88W1-N01" in (m.get("goal") or ""):
        print(m["id"], m.get("status"), m.get("claimed_by"), m.get("created_at"), m.get("finished_at"), (m.get("result") or {}).get("summary", "")[:300])