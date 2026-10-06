#!/usr/bin/env bash
# Non-interactive watcher: block until all re200 tier outputs (json or oom.json)
# exist, then print a one-shot summary. Bounded by WATCH_MAX seconds.
cd "$(dirname "$0")"
WATCH_MAX=${WATCH_MAX:-18000}   # 5h
t0=$(date +%s)
TIERS="re200_D16_12k re200_D20_12k re200_D24_12k re200_D28_12k re200_D30_12k"
while true; do
  done_all=1
  for t in $TIERS; do
    if [[ -f "$t.json" || -f "$t.json.oom.json" ]]; then :; else done_all=0; fi
  done
  now=$(date +%s)
  el=$((now - t0))
  echo "[watch ${el}s] done_all=$done_all"
  for t in $TIERS; do
    if [[ -f "$t.json" ]]; then
      st=$(grep -oE '"steps": [0-9]+' "$t.json" | head -1)
      echo "  $t -> OK $st"
    elif [[ -f "$t.json.oom.json" ]]; then
      echo "  $t -> OOM"
    else
      echo "  $t -> running"
    fi
  done
  if [[ "$done_all" == "1" ]]; then echo "ALL_TIERS_DONE"; break; fi
  if (( el > WATCH_MAX )); then echo "WATCH_TIMEOUT"; break; fi
  sleep 60
done