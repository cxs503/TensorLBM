import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import call, login

st, tok, _ = login()
for q in ["tensor/reports?mission_id=10395", "tensor/reports?mission_id=10395&limit=50", "tensor/reports?offset=100&limit=100"]:
    st, d = call(q, token=tok)
    if isinstance(d, dict) and "items" in d:
        print(q, "->", len(d["items"]), d.get("total"))
        for r in d["items"][:5]:
            print("   ", r["id"], r.get("mission_id"), r["title"][:80])
    else:
        print(q, "->", st, str(d)[:300])