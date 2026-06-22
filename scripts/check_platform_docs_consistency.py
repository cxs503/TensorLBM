#!/usr/bin/env python3
"""Lightweight consistency checks between app behavior and platform docs."""
from __future__ import annotations

import re
from pathlib import Path


def _fail(msg: str) -> None:
    print(f"[docs-consistency] ERROR: {msg}")
    raise SystemExit(1)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    preprocess = (repo / "app/backend/routers/preprocess.py").read_text(encoding="utf-8")
    manual = (repo / "docs/platform_user_manual.md").read_text(encoding="utf-8")

    m = re.search(r'TENSORLBM_MAX_UPLOAD_MB",\s*"(\d+)"', preprocess)
    if not m:
        _fail("Could not locate TENSORLBM_MAX_UPLOAD_MB default in preprocess router.")
    default_mb = int(m.group(1))

    if "TENSORLBM_MAX_UPLOAD_MB" not in manual:
        _fail("platform_user_manual.md must document TENSORLBM_MAX_UPLOAD_MB.")
    if "大小未限制" in manual:
        _fail("platform_user_manual.md still claims STL upload is unlimited.")
    if str(default_mb) not in manual:
        _fail(f"platform_user_manual.md should mention the default limit ({default_mb}MB).")

    print("[docs-consistency] OK")


if __name__ == "__main__":
    main()
