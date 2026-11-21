#!/usr/bin/env python3
"""Publish the RELX1-PHASETAU-R2 report to the platform API."""
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def call(path, payload, method="POST", token=None):
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = "Bearer " + token
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(), headers=h, method=method)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main():
    tok = call("/auth/login", {"username": USER, "password": PASS})["access_token"]
    report = json.load(open("/tmp/phasetau_report.json"))
    out = call("/tensor/reports", report, token=tok)
    print(json.dumps(out, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
