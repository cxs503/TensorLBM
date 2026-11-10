import json, os, urllib.request, urllib.error

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")


def call(path, method="GET", body=None, token=None):
    url = BASE.rstrip("/") + "/" + path.lstrip("/")
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw.decode(errors="replace")
    try:
        return 200, json.loads(raw)
    except Exception:
        return 200, raw.decode(errors="replace")


def login():
    st, d = call("auth/login", "POST", {"username": os.environ.get("TENSOR_AGENT_USER", "eco_admin"),
                                        "password": os.environ.get("TENSOR_AGENT_PASS", "")})
    tok = d.get("access_token") or d.get("token") if isinstance(d, dict) else None
    return st, tok, d


if __name__ == "__main__":
    st, tok, d = login()
    print(st, (tok or "")[:16])
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tok.txt"), "w").write(tok or "")
    st2, d2 = call("tensor/reports?limit=20", token=tok)
    print(json.dumps(d2, ensure_ascii=False)[:4000])