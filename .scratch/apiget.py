import json, os, sys, urllib.request, urllib.error

BASE = os.environ["TENSOR_API_BASE"]


def req(path, data=None, token=None, method=None):
    url = BASE.rstrip("/") + "/" + path.lstrip("/")
    body = None if data is None else json.dumps(data).encode()
    r = urllib.request.Request(url, data=body, method=method or ("POST" if body else "GET"))
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(r, timeout=60) as f:
            return f.status, json.loads(f.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:800]


def login():
    st, d = req("auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                               "password": os.environ["TENSOR_AGENT_PASS"]})
    tok = d.get("access_token") or d.get("token")
    open("/tmp/tok.txt", "w").write(tok)
    return tok


if __name__ == "__main__":
    mode = sys.argv[1]
    tok = login() if mode != "nol" else open("/tmp/tok.txt").read().strip()
    if mode in ("reports", "nol"):
        st, d = req("tensor/reports?limit=20", token=tok)
        items = d.get("items") if isinstance(d, dict) else d
        print("status", st, "n=", len(items) if isinstance(items, list) else d)
        for r in (items or [])[:20]:
            print(r.get("id"), "|", str(r.get("title"))[:70], "|", str(r.get("created_at"))[:19])