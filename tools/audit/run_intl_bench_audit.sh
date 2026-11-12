#!/usr/bin/env bash
# Reproducible runner for the GAP-LBM-INTL-BENCH audit (m15912).
#
#   tools/audit/run_intl_bench_audit.sh [repo_root] [out_dir]
#
# Produces, in <out_dir>: module_maturity.json/.csv, compete_score.csv,
# maturity.sql, summary.json, run.log, manifest.sha256, repro.txt.
# Exit code is the audit's own (0 = every upstream citation resolved,
# 3 = an upstream checkout is missing, see citation_failures in the JSON).
set -uo pipefail
REPO_ROOT=${1:-.}
OUT=${2:-artifacts/intl_bench_v1/run1}
cd "$REPO_ROOT" || exit 2
mkdir -p "$OUT"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 -m tensorlbm.audit --repo-root . --out "$OUT" > "$OUT/summary.json" 2> "$OUT/run.log"
RC=$?
END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{
  echo "cmd: PYTHONPATH=src python3 -m tensorlbm.audit --repo-root . --out $OUT"
  echo "repo_head: $(git rev-parse HEAD 2>/dev/null || echo n/a)"
  echo "started_utc: $START"
  echo "finished_utc: $END"
  echo "exit_code: $RC"
  echo "python: $(python3 -V 2>&1)"
} > "$OUT/repro.txt"
( cd "$OUT" && sha256sum module_maturity.json module_maturity.csv compete_score.csv maturity.sql \
    level_distribution.png compete_radar.png summary.json 2>/dev/null > manifest.sha256 )
echo "audit exit=$RC -> $OUT"
exit $RC