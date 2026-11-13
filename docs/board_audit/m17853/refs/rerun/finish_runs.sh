#!/bin/sh
cd "$(dirname "$0")"
set +e
for pw in -0.50 0.50 0.70; do
  d=$(python3 -c "print('runs/port_psi%+.2f_xi3.0'%$pw)")
  [ -f "$d/result.json" ] || python3 run_xval.py port --psi-wall $pw --xi 3.0
done
for xi in 2.0 4.0; do
  for pw in 0.00 0.35; do
    t="port_psi${pw}_xi${xi}"
    [ -f "runs/$t/result.json" ] || python3 run_xval.py port --psi-wall $pw --xi $xi --tag "$t"
  done
done
python3 oss_runner.py 0.00 32 48 100 10 > runs/oss_probe_100.txt 2>&1
python3 oss_runner.py 0.00 32 48 200 10 > runs/oss_probe_200.txt 2>&1
python3 run_xval.py summarize > runs/summary_stdout.txt 2>&1
python3 make_figures.py > runs/figures_stdout.txt 2>&1
echo ALL_DONE
