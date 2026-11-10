"""Post the P162W1-N05 report payload to the Tensor platform report API.

The payload is ``artifacts/p162w1_n05/report.json`` (produced by
``scripts/p162w1_n05_aggregate.py``): title / summary / scenario_key /
mission_id / tags / markdown / figures[{name, caption, png_base64}].

Usage
-----
    TENSOR_AGENT_USER=... TENSOR_AGENT_PASS=... PYTHONPATH=src \
        python scripts/p162w1_n05_post_report.py
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def main() -> int:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]
    login = _post(f"{base}/auth/login", {"username": user, "password": password})
    token = login["access_token"]
    payload_path = Path("artifacts/p162w1_n05/report.json")
    payload = json.loads(payload_path.read_text())
    payload.setdefault("mission_id", 9074)
    print(f"posting report: {payload['title']} "
          f"({len(payload.get('figures', []))} figures, "
          f"{len(payload.get('markdown', ''))} chars)")
    try:
        out = _post(f"{base}/tensor/reports", payload, token)
        print("ok:", json.dumps(out)[:400])
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        print("HTTPError", exc.code, exc.read()[:600])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())