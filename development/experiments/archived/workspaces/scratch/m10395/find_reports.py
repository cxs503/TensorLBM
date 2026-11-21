import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import call, login

st, tok, _ = login()
st, d = call("tensor/reports?limit=200", token=tok)
items = d.get("items", d if isinstance(d, list) else [])
hits = [r for r in items if "P88W1" in json.dumps(r, ensure_ascii=False) or "slosh" in json.dumps(r, ensure_ascii=False).lower()]
for r in hits:
    print(r["id"], r.get("mission_id"), r.get("created_at"), r["title"])
    print("   ", r.get("summary", "")[:600].replace("\n", " "))
print("total items", len(items), "hits", len(hits))