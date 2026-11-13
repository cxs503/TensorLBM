#!/usr/bin/env python3
"""W26 / m17569 — publish the mission report and commit the deliverables.

Runs in-process so that no credential ever appears on a shell command line:
  1. log in to the platform API (TENSOR_AGENT_USER / TENSOR_AGENT_PASS from env),
  2. POST /tensor/reports with the payload built by build_report.py,
  3. copy the exact payload into the artifact directory (proof of what was sent),
  4. git add + commit only the W26 deliverable tree.

Nothing is pushed: the platform pushes accepted branches itself.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path("/root/tp-agent/work/m17569")
ART = REPO / "papers" / "artifacts" / "w26" / "lbm-suboff-tau"
PAYLOAD = Path("/tmp/report_w26_m17569.json")


def post_json(url: str, body: dict, token: str | None = None, form: bool = False):
    data = json.dumps(body).encode() if not form else urllib.parse.urlencode(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json" if not form else
                   "application/x-www-form-urlencoded")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.read().decode()[:500]


def login(base: str, user: str, pw: str) -> str | None:
    attempts = [
        (f"{base}/auth/login", {"username": user, "password": pw}, False),
        (f"{base}/auth/token", {"username": user, "password": pw}, False),
        (f"{base}/token", {"username": user, "password": pw}, True),
        (f"{base}/auth/login", {"username": user, "password": pw}, True),
    ]
    for url, body, form in attempts:
        try:
            st, txt = post_json(url, body, form=form)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            print(f"[login] {url} -> {e}")
            continue
        try:
            d = json.loads(txt)
        except json.JSONDecodeError:
            print(f"[login] {url} -> {st} non-json")
            continue
        tok = d.get("access_token") or d.get("token") or (d.get("data") or {}).get("access_token")
        if tok:
            print(f"[login] ok via {url}")
            return tok
    return None


def main() -> None:
    base = os.environ.get("TENSOR_API_BASE", "").rstrip("/")
    user = os.environ.get("TENSOR_AGENT_USER", "")
    pw = os.environ.get("TENSOR_AGENT_PASS", "")
    if not PAYLOAD.exists():
        sys.exit("[publish] payload missing; run build_report.py first")
    payload = json.loads(PAYLOAD.read_text())
    ok = False
    tok = login(base, user, pw) if (base and user and pw) else None
    if tok:
        try:
            st, txt = post_json(f"{base}/tensor/reports", payload, token=tok)
            print(f"[publish] POST /tensor/reports -> {st} {txt[:300]}")
            ok = st in (200, 201, 202)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            print(f"[publish] POST failed: {e}")
    else:
        print("[publish] login unavailable; report NOT published")

    shutil.copy2(PAYLOAD, ART / "report_payload.json")
    print(f"[publish] payload copied into the artifact tree; published={ok}")

    subprocess.run(["git", "add", str(ART.relative_to(REPO))], cwd=REPO, check=True)
    msg = (
        "W26 SUBOFF friction anchor re-registration + fixed-Re grid GCI and (tau,N) sign surface\n\n"
        "- anchor register: 17 full-precision DTRC rows, 34/34 verbatim quotes machine-verified\n"
        "  as substrings of the archived transcriptions, ITTC-1957 recompute max |residual|\n"
        "  5.49e-15 (criterion 1e-9); measured C_F is NOT in the repository -> honest WITHHELD\n"
        "  (the recorded (1+k) is identically CT/CF_ITTC57, i.e. definitional)\n"
        "- fixed lattice Re=1000 three-grid ladder {16,24,36} (r=1.5, tau follows the grid):\n"
        "  C_T = 0.0434/0.0306/0.0422 -> oscillatory divergence, observed order and GCI\n"
        "  undefined -> criterion 3 honestly FAILs at the case Re; at the stated Re=1.2e7 it is\n"
        "  WITHHELD with a quantified reachability table\n"
        "- (tau,N) sign surface, 10 cells: tau <= 0.50015 diverges, no stable negative\n"
        "  window-mean drag in the scanned domain; the engine's AUTO wall treatment switches to\n"
        "  the wall-function route at the high-Re end, disclosed per cell\n"
        "- pytest: 11 tests re-derive every reported number from the raw npz/time series\n"
    )
    subprocess.run(["git", "-c", "user.name=tp-agent", "-c", "user.email=tp-agent@local",
                    "commit", "-m", msg], cwd=REPO, check=False)
    out = subprocess.run(["git", "status", "--short"], cwd=REPO, capture_output=True, text=True)
    print("[git] status:\n" + out.stdout[:800])


if __name__ == "__main__":
    main()