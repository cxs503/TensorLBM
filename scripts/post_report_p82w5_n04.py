"""Post the P82W5-N04 report to the Tensor API (token from env, no secrets on CLI)."""
import json
import os
import sys

import urllib.error
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post_json(url, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer %s" % token)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, r.read().decode()[:4000]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()[:4000]
    except Exception as exc:                    # noqa: BLE001
        return None, str(exc)


def get_token():
    """``POST /auth/login`` with a JSON body ``{username, password}``."""
    st, txt = post_json(BASE + "/auth/login",
                        {"username": USER, "password": PASS})
    if st == 200:
        try:
            tok = json.loads(txt).get("access_token")
        except Exception:                      # noqa: BLE001
            tok = None
        if tok:
            print("login ok")
            return tok
    print("login failed", st, txt[:200])
    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/report_p82w5.json"
    payload = json.load(open(path))
    tok = get_token()
    if tok is None:
        print("NO TOKEN")
        return 1
    out = os.path.join(os.path.dirname(path), "report_post_result.json")
    st, txt = post_json(BASE + "/tensor/reports", payload, tok)
    print("POST /tensor/reports ->", st, txt[:300])
    with open(out, "w") as fh:
        json.dump({"status": st, "body": txt}, fh, indent=2)
    return 0 if st in (200, 201) else 2


if __name__ == "__main__":
    sys.exit(main())