"""Probe the tensor API for the login route (no secrets printed)."""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def req(path, data=None, form=False, method="POST"):
    url = f"{BASE}/{path.lstrip('/')}"
    headers = {}
    if data is not None:
        if form:
            data = urllib.parse.urlencode(data).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            data = json.dumps(data).encode()
            headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=10) as fh:
            return fh.status, fh.read().decode()[:160]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:160]
    except Exception as e:
        return -1, repr(e)[:160]


cands = [
    "login", "token", "auth", "auth/token", "api/v1/login", "api/v1/token",
    "api/v1/auth/token", "api/login", "api/token", "agent/login", "agents/login",
    "tensor/agents/login", "tensor/agent/login", "v1/login", "user/login",
    "users/login", "api/v1/auth/sign-in", "auth/signin", "signin",
    "tensor/reports", "reports", "api/v1/tensor/reports",
]
creds = {"username": USER, "password": PASS}
for c in cands:
    for form in (False, True):
        st, body = req(c, creds, form=form)
        tag = "form" if form else "json"
        print(f"{c:28s} {tag:4s} -> {st} {body[:90]}")
        if st != 401 or "UNAUTHORIZED" not in body:
            break