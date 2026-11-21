import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import call, login

st, tok, _ = login()
for q in ["openapi.json", "docs", "campaigns", "tensor/campaigns", "tensor/campaign/nodes?limit=5",
          "tensor/campaign_nodes?limit=5", "tensor/sessions/7125/events?limit=20"]:
    st, d = call(q, token=tok)
    s = json.dumps(d, ensure_ascii=False) if not isinstance(d, str) else d
    print("==", q, st, s[:400])
    print()