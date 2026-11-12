#!/usr/bin/env python3
"""Backwards-compatible entry point for the GAP-LBM-INTL-BENCH audit.

The implementation now lives in the installable package
(``tensorlbm.audit.intl_bench``) so it can be unit-tested and called from the
public API; this module is only a shim kept so older runbooks / citations that
point at ``tools/audit/m15617_intl_bench_audit.py`` keep working.

    python tools/audit/m15617_intl_bench_audit.py --repo-root . --out DIR
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src"))

from tensorlbm.audit.intl_bench import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
