#!/usr/bin/env python3
"""Fetch the OSS reference sources (zero hand-copy provenance) for the
XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT anchor.

Only the files that the transplant actually cites are fetched, so the ref/
directory stays small; each file gets a sha256 recorded in provenance.json.
"""
import hashlib
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ref", "sailfish_src")
REPO = "sailfish-team/sailfish"
REFS = ["main", "master"]
FILES = [
    "LICENSE",
    "README.md",
    "sailfish/lb_binary.py",
    "sailfish/sym_equilibrium.py",
    "sailfish/templates/finite_difference_optimized.mako",
    "sailfish/sym.py",
]


def fetch(path: str):
    for ref in REFS:
        url = f"https://raw.githubusercontent.com/{REPO}/{ref}/{path}"
        try:
            with urllib.request.urlopen(url, timeout=60) as f:
                return ref, f.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"{path}: {last}")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    prov = {"repo": REPO, "retrieved_via": "raw.githubusercontent.com", "files": {}}
    for path in FILES:
        try:
            ref, blob = fetch(path)
        except Exception as exc:  # noqa: BLE001
            print(f"MISS {path}: {exc}")
            continue
        dst = os.path.join(OUT, path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(blob)
        prov["files"][path] = {
            "ref": ref,
            "bytes": len(blob),
            "lines": blob.count(b"\n"),
            "sha256": hashlib.sha256(blob).hexdigest(),
        }
        print(f"OK {path} ref={ref} lines={blob.count(chr(10).encode())} sha256={prov['files'][path]['sha256'][:16]}")
    with open(os.path.join(HERE, "ref", "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())