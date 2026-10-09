#!/usr/bin/env python3
"""Big-grid Musker vs log-law test launcher — 8 parallel jobs on SDAA 8-15.

Tests whether Musker wall law prevents fine grid collapse by being continuous
across all y+ (viscous + buffer + log-law), unlike log-law which fails when
y+ enters the buffer layer on fine grids.

Card allocation:
   8: SUBOFF bare_hull 320³, Cs=0.05, wall_law="musker", 3000 steps
   9: SUBOFF bare_hull 320³, Cs=0.05, wall_law="log",    3000 steps (baseline)
  10: KVLCC2 ship 256³,      Cs=0.05, wall_law="musker", 3000 steps
  11: KVLCC2 ship 256³,      Cs=0.05, wall_law="log",    3000 steps (baseline)
  12: SUBOFF bare_hull 256³, Cs=0.05, wall_law="musker", 3000 steps
  13: SUBOFF bare_hull 256³, Cs=0.05, wall_law="log",    3000 steps (baseline)
  14: SUBOFF bare_hull 384³, Cs=0.05, wall_law="musker", 2000 steps (may OOM)
  15: SUBOFF bare_hull 384³, Cs=0.05, wall_law="log",    2000 steps (may OOM)

All: D3Q19 MRT+Smag, Re=2e6, farfield, sliding window=500.
Output to /tmp/biggrid_results.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKER = Path(__file__).parent / "biggrid_worker.py"
SRC = Path(__file__).parent / "src"
LOG_DIR = Path("/tmp/biggrid_logs")
OUTPUT_FILE = Path("/tmp/biggrid_results.json")
LOG_DIR.mkdir(exist_ok=True)

JOBS = [
    # (card_id, case_name, wall_law, label)
    (8,  "suboff_320", "musker", "SUBOFF 320³ musker"),
    (9,  "suboff_320", "log",    "SUBOFF 320³ log"),
    (10, "kvlcc2_256", "musker", "KVLCC2 256³ musker"),
    (11, "kvlcc2_256", "log",    "KVLCC2 256³ log"),
    (12, "suboff_256", "musker", "SUBOFF 256³ musker"),
    (13, "suboff_256", "log",    "SUBOFF 256³ log"),
    (14, "suboff_384", "musker", "SUBOFF 384³ musker"),
    (15, "suboff_384", "log",    "SUBOFF 384³ log"),
]


def main():
    print("=" * 90)
    print("BIG-GRID MUSKER vs LOG-LAW TEST — 8 JOBS ON SDAA 8-15")
    print("  Key question: Does Musker prevent fine grid collapse?")
    print("  D3Q19 MRT+Smag + wallfn + farfield, Re=2e6, sliding-window 500")
    print("  Musker is continuous across all y+ (viscous+buffer+log)")
    print("  Log-law fails when first off-wall cell enters buffer on fine grids")
    print("=" * 90)
    print()

    # Launch all workers
    procs = {}
    for card_id, case_name, wall_law, label in JOBS:
        log_file = LOG_DIR / f"biggrid_{card_id:02d}_{case_name}_{wall_law}.log"
        cmd = [
            sys.executable, "-u", str(WORKER),
            str(card_id), case_name, wall_law,
        ]
        env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONUNBUFFERED": "1"}

        with open(log_file, "w") as log_fp:
            proc = subprocess.Popen(
                cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT,
            )
        procs[(card_id, case_name, wall_law)] = (proc, log_file, label)
        print(f"  [{card_id:02d}] {label:35s} PID={proc.pid}  log={log_file}")

    print(f"\n  All {len(procs)} jobs launched. Waiting for completion...\n")

    t_start = time.time()
    while True:
        done = sum(1 for proc, _, _ in procs.values() if proc.poll() is not None)
        elapsed = time.time() - t_start
        print(f"  [{elapsed:.0f}s] {done}/{len(procs)} completed", flush=True)
        if done == len(procs):
            break
        time.sleep(60)

    total_elapsed = time.time() - t_start
    print(f"\n  All jobs finished in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)\n")

    # Collect results from log files (last JSON line)
    results = {}
    print("=" * 95)
    print("RESULTS SUMMARY")
    print("=" * 95)
    print(f"\n{'Case':<30} {'Ct_fric':>10} {'Ct_pres':>10} {'Ct_total':>10} {'Error%':>8} {'Finite':>7} {'Time':>8}")
    print("-" * 85)

    for (card_id, case_name, wall_law), (proc, log_file, label) in procs.items():
        rc = proc.returncode
        key = f"{case_name}_{wall_law}"

        try:
            log_text = log_file.read_text()
            lines = log_text.strip().splitlines()
            result = None
            for line in reversed(lines):
                try:
                    candidate = json.loads(line)
                    if isinstance(candidate, dict) and "case" in candidate:
                        result = candidate
                        break
                except (json.JSONDecodeError, ValueError):
                    continue

            if result is None:
                result = {
                    "case": key, "card_id": card_id, "wall_law": wall_law,
                    "_status": "PARSE_FAILED", "_rc": rc,
                    "_log_tail": "\n".join(lines[-10:]) if lines else "",
                }
            else:
                result["_rc"] = rc
                result["card_id"] = card_id
                result["wall_law"] = wall_law

        except FileNotFoundError:
            result = {"case": key, "card_id": card_id, "wall_law": wall_law, "_status": "LOG_MISSING"}

        results[key] = result

        status = result.get("_status", "OK")
        if status != "OK":
            print(f"  {label:<30s} {'n/a':>10} {'n/a':>10} {'n/a':>10} {'n/a':>8} {'n/a':>7} {'n/a':>8}")
        else:
            cf = result.get("Ct_fric", 0)
            cp = result.get("Ct_pres", 0)
            ct = result.get("Ct_total", 0)
            err = result.get("error_pct", 0)
            fin = "YES" if result.get("finite", False) else "NO"
            t_s = result.get("wall_time_s", 0)
            t_str = f"{t_s:.0f}s" if t_s else "?"
            print(f"  {label:<30s} {cf:>10.5f} {cp:>10.5f} {ct:>10.5f} {err:>7.1f}% {fin:>7} {t_str:>8}")

    print("-" * 85)
    print()

    # ─── Musker vs Log-law comparison ──────────────────────────────────────
    print("=" * 95)
    print("MUSKER vs LOG-LAW COMPARISON")
    print("=" * 95)

    pairs = [
        ("suboff_320", "SUBOFF 320³"),
        ("kvlcc2_256", "KVLCC2 256³"),
        ("suboff_256", "SUBOFF 256³"),
        ("suboff_384", "SUBOFF 384³"),
    ]

    print(f"\n{'Grid':<20} {'Wall Law':<10} {'Ct_total':>10} {'Error%':>8} {'Finite':>7} {'Time':>8} {'Verdict':>20}")
    print("-" * 85)

    for case_name, label in pairs:
        for wl in ("musker", "log"):
            key = f"{case_name}_{wl}"
            r = results.get(key, {})
            if r.get("_status") == "OK":
                ct = r.get("Ct_total", 0)
                err = r.get("error_pct", 0)
                fin = "YES" if r.get("finite", False) else "NO"
                t_s = r.get("wall_time_s", 0)
                t_str = f"{t_s:.0f}s" if t_s else "?"
                nan_step = r.get("nan_at_step", -1)
                verdict = "STABLE" if (fin and ct > 0 and err < 200) else "COLLAPSED"
                if nan_step > 0:
                    verdict += f" @step{nan_step}"
                print(f"  {label:<20s} {wl:<10s} {ct:>10.5f} {err:>7.1f}% {fin:>7} {t_str:>8} {verdict:>20}")
            else:
                print(f"  {label:<20s} {wl:<10s} {'n/a':>10} {'n/a':>8} {'NO':>7} {'?':>8} {'FAILED':>20}")

        # Print Musker vs Log comparison
        musker_key = f"{case_name}_musker"
        log_key = f"{case_name}_log"
        mr = results.get(musker_key, {})
        lr = results.get(log_key, {})
        if mr.get("_status") == "OK" and lr.get("_status") == "OK":
            m_ct = mr.get("Ct_total", 0)
            l_ct = lr.get("Ct_total", 0)
            m_fin = mr.get("finite", False)
            l_fin = lr.get("finite", False)
            m_err = mr.get("error_pct", float("inf"))
            l_err = lr.get("error_pct", float("inf"))
            if m_fin and l_fin and m_ct > 0 and l_ct > 0:
                if abs(m_err) < abs(l_err):
                    winner = "MUSKER wins ✓"
                elif abs(l_err) < abs(m_err):
                    winner = "LOG wins"
                else:
                    winner = "TIE"
                print(f"         {'':>10} ΔCt={m_ct-l_ct:+.5f}, Musker err={m_err:.1f}% vs Log err={l_err:.1f}% → {winner}")
            elif m_fin and not l_fin:
                print(f"         {'':>10} LOG COLLAPSED, MUSKER SURVIVED ✓ — confirms hypothesis!")
            elif not m_fin and l_fin:
                print(f"         {'':>10} MUSKER COLLAPSED, LOG SURVIVED — unexpected!")
            elif not m_fin and not l_fin:
                print(f"         {'':>10} BOTH COLLAPSED — grid too fine for both laws")
        print()

    # ─── Key verdict ──────────────────────────────────────────────────────
    print("=" * 95)
    print("KEY VERDICT: Does Musker prevent fine grid collapse?")
    print("=" * 95)

    musker_collapsed = []
    log_collapsed = []
    musker_ok = []
    log_ok = []

    for case_name, label in pairs:
        mr = results.get(f"{case_name}_musker", {})
        lr = results.get(f"{case_name}_log", {})
        if mr.get("_status") == "OK":
            if mr.get("finite") and mr.get("Ct_total", 0) > 0:
                musker_ok.append(case_name)
            else:
                musker_collapsed.append(case_name)
        if lr.get("_status") == "OK":
            if lr.get("finite") and lr.get("Ct_total", 0) > 0:
                log_ok.append(case_name)
            else:
                log_collapsed.append(case_name)

    print(f"\n  Musker stable on: {musker_ok if musker_ok else 'NONE'}")
    print(f"  Musker collapsed: {musker_collapsed if musker_collapsed else 'NONE'}")
    print(f"  Log stable on:    {log_ok if log_ok else 'NONE'}")
    print(f"  Log collapsed:    {log_collapsed if log_collapsed else 'NONE'}")

    only_musker = set(musker_ok) - set(log_ok)
    only_log = set(log_ok) - set(musker_ok)

    if only_musker:
        print(f"\n  ✓ MUSKER survived on grids where LOG collapsed: {only_musker}")
        print(f"    HYPOTHESIS CONFIRMED: Musker prevents fine grid collapse.")
    elif only_log:
        print(f"\n  ✗ Unexpected: LOG survived on grids where MUSKER collapsed: {only_log}")
    elif musker_collapsed and log_collapsed:
        common = set(musker_collapsed) & set(log_collapsed)
        if common:
            print(f"\n  ⚠ Both collapsed on: {common} — grid may be beyond memory limits")
    else:
        print(f"\n  Both wall laws stable on all tested grids — grid not fine enough for buffer-layer entry.")

    print()

    # ─── Write output ─────────────────────────────────────────────────────
    output = {
        "title": "Big-Grid Musker vs Log-Law Test",
        "description": "Tests whether Musker wall law (continuous across all y+) prevents fine grid collapse that plagues log-law on fine grids",
        "hypothesis": "Log-law fails when first off-wall cell enters buffer layer (y+~5-30). Musker is continuous and should survive.",
        "total_wall_time_s": total_elapsed,
        "jobs": len(JOBS),
        "cards_used": "SDAA 8-15",
        "grids_tested": ["256³", "320³", "384³"],
        "wall_laws_tested": ["musker", "log"],
        "musker_stable": musker_ok,
        "musker_collapsed": musker_collapsed,
        "log_stable": log_ok,
        "log_collapsed": log_collapsed,
        "musker_saved": list(only_musker),
        "results": results,
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=str))
    print(f"  Results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
