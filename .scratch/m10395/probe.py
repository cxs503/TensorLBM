import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import call, login

st, tok, _ = login()
cands = ["tensor/missions/10395", "missions/10395", "tensor/reports?page=2&limit=100",
         "tensor/missions?limit=200", "openapi.json"]
for q in cands:
    st, d = call(q, token=tok)
    s = json.dumps(d, ensure_ascii=False) if not isinstance(d, str) else d
    print("==", q, st, len(s))
    if q == "openapi.json":
        paths = d.get("paths", {})
        print([p for p in paths if "report" in p or "mission" in p][:40])
    else:
        print(s[:1500])