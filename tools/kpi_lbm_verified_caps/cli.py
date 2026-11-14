#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI：selftest / measure / verify（退出码 0=ok，2=对拍或不自检通过，1=运行错误）。

用法：
  python3 -m tools.kpi_lbm_verified_caps selftest
  python3 -m tools.kpi_lbm_verified_caps measure  --out runs/kpi_lbm_verified_caps
  python3 -m tools.kpi_lbm_verified_caps verify   --out runs/kpi_lbm_verified_caps
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import collect, compute, oracle, report, selftest, spec


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def run_measure(out: Path) -> dict:
    now = _now()
    primary_start = compute.iso_week_start(now)
    rolling_start = now - timedelta(days=7)
    collect_start = min(primary_start, rolling_start)
    windows = {
        "primary_start_utc": primary_start.isoformat(),
        "rolling_start_utc": rolling_start.isoformat(),
        "window_end_utc": now.isoformat(),
    }
    snap = collect.collect(collect_start.isoformat(), now.isoformat(), windows)
    collect.assert_all_readonly(snap)

    summary = compute.summarize(snap)
    oracle_cmp = oracle.compare(summary)
    st = selftest.run()

    ledger_bounds = collect.first(snap, "Q1_ledger_bounds_repo")
    ledger_all = collect.first(snap, "Q2_ledger_bounds_all_repos")
    referee_kpi_rows = int((collect.first(snap, "Q8_referee_rows_for_kpi_anchor") or {}).get("n_rows") or 0)
    fl = collect.first(snap, "Q9_feature_ledger_kpi_row") or {}

    result = report.build(summary, ledger_all, referee_kpi_rows, oracle_cmp, st)
    result["feature_ledger_row"] = fl
    result["ledger_bounds_repo"] = ledger_bounds
    result["oracle_comparison"] = oracle_cmp
    result["selftest"] = st
    result["window_partial"] = True

    out.mkdir(parents=True, exist_ok=True)
    (out / "snapshot.json").write_text(json.dumps(snap, ensure_ascii=False, indent=1))
    report.write_tsv(summary, out / "data" / "anchors.tsv")

    figs = report.figures({**result, "_primary": summary["primary"]})
    md_src = {
        **summary,
        "_snapshot_blocks": snap["blocks"],
        "_ledger_bounds": ledger_all,
    }
    md = report.markdown(result, md_src, figs, oracle_cmp, st)
    (out / "KPI-LBM-VERIFIED-CAPS.md").write_text(md)
    result["figures"] = figs
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def run_verify(out: Path) -> int:
    res = json.loads((out / "result.json").read_text())
    snap = json.loads((out / "snapshot.json").read_text())
    # 1) spec 指纹自洽
    if res["spec_sha256"] != spec.spec_sha256():
        print(f"spec_sha256 mismatch: {res['spec_sha256']} != {spec.spec_sha256()}")
        return 2
    # 2) 快照 → 重算与落盘 result 一致（同口径两次独立运行）
    summary = compute.summarize(snap)
    if summary["primary"]["current_primary"] != res["current"]:
        print(f"recompute mismatch: {summary['primary']['current_primary']} != {res['current']}")
        return 2
    # 3) oracle 单 CTE 对拍
    cmp = oracle.compare(summary)
    print(json.dumps({"oracle": cmp, "recompute_current": summary["primary"]["current_primary"]}, ensure_ascii=False))
    return 0 if cmp["match"] else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kpi_lbm_verified_caps")
    ap.add_argument("cmd", choices=["selftest", "measure", "verify"])
    ap.add_argument("--out", default="runs/kpi_lbm_verified_caps")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        r = selftest.run()
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return 0 if r["n_fail"] == 0 else 2
    if a.cmd == "measure":
        res = run_measure(Path(a.out))
        print(
            json.dumps(
                {
                    "current": res["current"],
                    "unit": res["unit"],
                    "target": res["target"],
                    "gap": res["gap"],
                    "oracle_match": res["verification"]["oracle_match"],
                    "selftest_n_fail": res["verification"]["selftest_n_fail"],
                    "out": a.out,
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        return 0
    return run_verify(Path(a.out))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())