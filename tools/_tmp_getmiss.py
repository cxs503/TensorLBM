import os, json, urllib.request

base = os.environ["TENSOR_API_BASE"].rstrip("/")


def req(path, data=None, token=None):
    body = None if data is None else json.dumps(data).encode()
    r = urllib.request.Request(base + path, data=body)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(r, timeout=30) as f:
        return json.loads(f.read().decode())


tok = req("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                          "password": os.environ["TENSOR_AGENT_PASS"]})["access_token"]
open("/tmp/tok.txt", "w").write(tok)
for p in ["/tensor/missions/6036", "/campaigns/nodes?mission_id=6036",
          "/campaigns/plans"]:
    try:
        m = req(p, token=tok)
        print("==", p, "->", json.dumps(m, ensure_ascii=False)[:2500])
    except Exception as e:
        print("==", p, "ERR", e)