#!/usr/bin/env bash
# Independent single-card acoustic campaign.  No distributed/multi-card solver is used.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STAMP=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT="$ROOT/validation_logs/acoustic_sdaa_campaign_$STAMP"
mkdir -p "$OUT"

# Format: card|case|nx|ny|tau|velocity|steps
# Trailing-edge grids are all at/above the geometry/microphone-safe minimum
# (the current solver's sponge and r=70 upper microphone require ny >= 242).
CASES=(
  '8|te_base|312|242|0.550|0.100|2400'
  '9|te_tau056|320|244|0.560|0.100|2600'
  '10|te_tau058|328|246|0.580|0.100|2800'
  '11|te_u009|336|248|0.550|0.090|3000'
  '12|te_u011|344|250|0.550|0.110|3200'
  '13|te_tau054|352|252|0.540|0.100|3400'
  '14|te_tau057_u095|360|254|0.570|0.095|3600'
  '15|te_tau060_u105|368|256|0.600|0.105|3800'
  '16|rossiter_base|400|100|0.550|0.100|1800'
  '17|rossiter_tau056|408|104|0.560|0.100|1950'
  '18|rossiter_tau058|416|108|0.580|0.100|2100'
  '19|rossiter_u009|424|112|0.550|0.090|2250'
  '20|rossiter_u011|432|116|0.550|0.110|2400'
  '21|rossiter_tau054|440|120|0.540|0.100|2550'
  '22|rossiter_tau057_u095|448|124|0.570|0.095|2700'
  '23|rossiter_tau060_u105|456|128|0.600|0.105|2850'
)

printf 'utc_start,card,case,nx,ny,tau,u,steps,command\n' > "$OUT/manifest.csv"
printf 'utc_start,utc_end,elapsed_s,card,case,exit_code,log\n' > "$OUT/status.csv"

run_case() {
  local spec="$1" start end rc card case nx ny tau u steps log cmd
  IFS='|' read -r card case nx ny tau u steps <<< "$spec"
  log="$OUT/${card}_${case}.log"
  start=$(date -u +%FT%TZ)
  if [[ "$case" == te_* ]]; then
    cmd=(timeout --foreground 720s python "$ROOT/examples/benchmark_trailing_edge_noise.py" --device "sdaa:$card" --nx "$nx" --ny "$ny" --tau "$tau" --u-in "$u" --steps "$steps" --log-every 400)
  else
    cmd=(timeout --foreground 720s python "$ROOT/examples/benchmark_rossiter_cavity.py" --device "sdaa:$card" --nx "$nx" --ny "$ny" --tau "$tau" --U "$u" --steps "$steps" --log-every 300)
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%q\n' "$start" "$card" "$case" "$nx" "$ny" "$tau" "$u" "$steps" "${cmd[*]}" >> "$OUT/manifest.csv"
  set +e
  (cd "$ROOT" && "${cmd[@]}") >"$log" 2>&1
  rc=$?
  set -e
  end=$(date -u +%FT%TZ)
  local elapsed
  elapsed=$(( $(date -ud "$end" +%s) - $(date -ud "$start" +%s) ))
  printf '%s,%s,%s,%s,%s,%s,%s\n' "$start" "$end" "$elapsed" "$card" "$case" "$rc" "$log" >> "$OUT/status.csv"
  return 0
}

for spec in "${CASES[@]}"; do run_case "$spec" & done
wait

python - "$OUT" <<'PY'
import csv, pathlib, re, sys
out = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader((out / 'status.csv').open()))
summary = ['# SDAA independent acoustic campaign', '', '|card|case|exit|elapsed s|reported metric|reported status|', '|---:|---|---:|---:|---|---|']
for row in sorted(rows, key=lambda r:int(r['card'])):
    text = pathlib.Path(row['log']).read_text(errors='replace')
    if row['case'].startswith('te_'):
        metric = re.search(r'主峰: f=([0-9.eE+-]+)\s+St=([0-9.eE+-]+)', text)
        metric_s = ('f=%s, St=%s' % metric.groups()) if metric else 'not produced'
        status = re.findall(r'\b(PASS|FAIL)\b.*尾缘自噪声', text)
    else:
        metric = re.search(r'最佳匹配: 模态\s+(\d+).*?误差\s+=\s+([0-9.]+)%', text, re.S)
        metric_s = ('mode=%s, error=%s%%' % metric.groups()) if metric else 'not produced'
        status = re.findall(r'状态\s+:\s+.*?\b(PASS|FAIL)\b', text)
    summary.append('|%s|%s|%s|%s|%s|%s|' % (row['card'], row['case'], row['exit_code'], row['elapsed_s'], metric_s, status[-1] if status else 'not produced'))
(out / 'SUMMARY.md').write_text('\n'.join(summary) + '\n')
PY
printf 'Campaign artifacts: %s\n' "$OUT"
