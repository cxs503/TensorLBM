import json, os, pathlib, urllib.request

base = os.environ["TENSOR_API_BASE"].rstrip("/")
login = json.dumps({
    "username": os.environ["TENSOR_AGENT_USER"],
    "password": os.environ["TENSOR_AGENT_PASS"],
}).encode()


def post(path, payload):
    req = urllib.request.Request(base + path, data=payload,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


tok = post("/auth/login", login)
token = tok.get("access_token") or tok.get("token")
if not token:
    raise SystemExit(f"no token: {list(tok)[:6]}")

req = urllib.request.Request(
    base + "/tensor/reports",
    data=pathlib.Path("/tmp/report.json").read_bytes(),
    headers={"Content-Type": "application/json",
             "Authorization": "Bearer " + token},
)
with urllib.request.urlopen(req, timeout=300) as r:
    out = json.loads(r.read().decode())
print("reports POST ok:", {k: out[k] for k in list(out)[:6]})