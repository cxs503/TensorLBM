#!/usr/bin/env python3
"""Validate that en.json and zh.json have identical key sets.

Usage:
    python platform/i18n/check_keys.py

Exit code 0 = all keys match; non-zero = mismatch (prints details).
Can be called from CI.
"""
import json
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_STATIC_I18N = _HERE.parent / "frontend" / "static" / "i18n"


def flatten(d: dict, prefix: str = "") -> set:
    """Return a flat set of dot-notation keys for a (possibly nested) dict."""
    keys: set = set()
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            keys |= flatten(v, full)
        else:
            keys.add(full)
    return keys


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    en_path = _STATIC_I18N / "en.json"
    zh_path = _STATIC_I18N / "zh.json"

    for p in (en_path, zh_path):
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            return 2

    en_keys = flatten(load_json(en_path))
    zh_keys = flatten(load_json(zh_path))

    missing_zh = sorted(en_keys - zh_keys)
    extra_zh = sorted(zh_keys - en_keys)

    ok = True
    if missing_zh:
        ok = False
        print("Keys present in en.json but MISSING from zh.json:")
        for k in missing_zh:
            print(f"  {k}")
    if extra_zh:
        ok = False
        print("Keys present in zh.json but MISSING from en.json:")
        for k in extra_zh:
            print(f"  {k}")

    if ok:
        print(f"OK – both locale files share {len(en_keys)} keys.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
