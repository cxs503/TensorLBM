"""Enumerate platform routes related to bench/benchmark to locate the referee-table reader."""
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
req = urllib.request.Request(f"{BASE}/openapi.json")
spec = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
hits = [p for p in spec.get("paths", {}) if "bench" in p or "case" in p or "anchor" in p or "maturity" in p]
for p in sorted(hits):
    print(p, sorted(spec["paths"][p].keys()))