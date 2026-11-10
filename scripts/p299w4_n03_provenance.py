#!/usr/bin/env python
"""Write a provenance manifest (sha256 + run metadata) for the p299w4-n03 artefacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "p299w4_n03_alm"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> int:
    files = sorted(p for p in ART.rglob("*") if p.is_file() and p.suffix in {".json", ".png", ".md"})
    manifest = {
        "node": "p299w4-n03-lbm-alm-propeller",
        "mission_id": 11071,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_rev": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "files": [
            {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in files
        ],
    }
    out = ART / "PROVENANCE.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
