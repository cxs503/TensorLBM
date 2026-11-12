#!/usr/bin/env python3
"""Finalise CASE-LBM-RB-ONSET-RA1708: wait for the run, figures, report, platform post.

Detached helper (survives the calling shell):
  python3 tools/ra1708_finalize.py > runs/ra1708_finalize.log 2>&1
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path("/root/tp-agent/work/m16209")
CASE = REPO / "benchmarks/case_pool/rayleigh_benard_onset"
RUN = REPO / "runs/case_rayleigh_benard_onset"


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)


def wait_for_summary(timeout_s: int = 3600) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if (RUN / "summary.json").exists():
            log("summary.json present")
            return True
        time.sleep(20)
    log("TIMEOUT waiting for summary.json")
    return False


def sh(cmd: list[str], timeout: int = 1800) -> int:
    log("RUN " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(REPO), env={**os.environ, "PYTHONPATH": "src"},
                       capture_output=True, text=True, timeout=timeout)
    log(f"rc={r.returncode} out={r.stdout[-1500:]} err={r.stderr[-1500:]}")
    return r.returncode


def copy_artifacts() -> None:
    dst = CASE / "artifacts"
    dst.mkdir(parents=True, exist_ok=True)
    for f in RUN.glob("*.json"):
        shutil.copy2(f, dst / f.name)
    figdir = RUN / "figures"
    if figdir.exists():
        (dst / "figures").mkdir(exist_ok=True)
        for f in figdir.glob("*.png"):
            shutil.copy2(f, dst / "figures" / f.name)
    npz = RUN / "fields_snapshot.npz"
    if npz.exists():
        shutil.copy2(npz, dst / npz.name)
    log(f"copied artifacts -> {dst}")


def post_report(payload_path: Path) -> None:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    user = os.environ.get("TENSOR_AGENT_USER", "")
    pwd = os.environ.get("TENSOR_AGENT_PASS", "")

    def call(url: str, data=None, token: str | None = None, method: str | None = None):
        body = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method=method or ("POST" if body else "GET"))
        req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode())

    token = None
    for path in ("/auth/login", "/login"):
        for body in ({"username": user, "password": pwd},
                     {"email": user, "password": pwd}):
            try:
                st, resp = call(base + path, body)
                log(f"login {path} -> {st} keys={list(resp)[:8]}")
                data = resp.get("data", resp) if isinstance(resp, dict) else {}
                token = (data.get("access_token") or data.get("token")
                         or resp.get("access_token"))
                if token:
                    break
            except Exception as exc:  # noqa: BLE001
                log(f"login {path} failed: {exc}")
        if token:
            break
    if not token:
        log("NO TOKEN — platform report NOT sent")
        return
    payload = json.loads(payload_path.read_text())
    log(f"payload markdown {len(payload.get('markdown',''))} chars, "
        f"{len(payload.get('figures', []))} figures, "
        f"{sum(len(f['png_base64']) for f in payload.get('figures', []))/1e6:.2f} MB b64")
    try:
        st, resp = call(base + "/tensor/reports", payload, token=token)
        log(f"POST /tensor/reports -> {st} {json.dumps(resp)[:400]}")
        data = resp.get("data", resp) if isinstance(resp, dict) else {}
        rid = data.get("id") or data.get("report_id") or resp.get("id")
        if rid:
            try:
                st2, r2 = call(base + "/tensor/missions/16209/link-report",
                               {"report_id": rid}, token=token)
                log(f"link-report {rid} -> {st2} {json.dumps(r2)[:300]}")
            except Exception as exc:  # noqa: BLE001
                log(f"link-report failed: {exc}")
    except urllib.error.HTTPError as exc:
        log(f"POST failed HTTP {exc.code}: {exc.read()[:800]!r}")


def main() -> int:
    log("finalize start")
    if not wait_for_summary():
        return 1
    sh([sys.executable, "-m", "pytest", "tests/test_case_pool_rayleigh_benard_onset.py", "-q"])
    sh([sys.executable, str(CASE / "make_figures.py"), "--run-dir", str(RUN), "--asym"],
       timeout=2400)
    sh([sys.executable, str(CASE / "make_report.py"), "--run-dir", str(RUN),
        "--out", str(CASE / "REPORT.md"), "--platform-json", "/tmp/report.json"])
    copy_artifacts()
    post_report(Path("/tmp/report.json"))
    log("finalize done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())