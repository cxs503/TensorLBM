"""Shard dispatcher: map a 1-based shard index to one (config, arm) ladder case.

Used by the platform cluster queue::

    python run_shard.py --index 3/12 \
        --outroot xval/XVAL-LBM-RASTER-OBSTACLE-CORNER-CLASS-BOUNCE-BACK/runs

The ladder is enumerated deterministically (config order of ``CONFIGS`` x
``("oss", "mask")``), so shard k always runs the same case.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def ladder():
    from run_xval import ARMS, CONFIGS

    return [(cfg, arm) for cfg in CONFIGS for arm in ARMS]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="i/n (1-based)")
    ap.add_argument("--outroot", default=str(HERE / "runs"))
    a = ap.parse_args(argv)
    i, n = (int(v) for v in a.index.split("/"))
    cases = ladder()
    if n != len(cases):
        print(f"WARNING: shard count {n} != ladder size {len(cases)}", file=sys.stderr)
    cfg, arm = cases[i - 1]
    outdir = Path(a.outroot) / f"{cfg}_{arm}"
    print(f"[shard {i}/{n}] config={cfg} arm={arm} -> {outdir}", flush=True)
    try:
        from run_xval import run_case

        run_case(cfg, arm, outdir)
        (outdir / "status.json").write_text(json.dumps({"status": "ok", "config": cfg, "arm": arm}))
        return 0
    except Exception as exc:  # honest failure, recorded not hidden
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "status.json").write_text(
            json.dumps({"status": "failed", "config": cfg, "arm": arm,
                        "error": repr(exc), "traceback": traceback.format_exc()})
        )
        print(f"[shard {i}/{n}] FAILED: {exc!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())