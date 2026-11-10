#!/usr/bin/env bash
# Local driver for the IN130-N03 campaign (thread-pinned; the control plane is
# heavily loaded, so the pricey cylinder sweep is replaced by a divergence probe).
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src
OUT=artifacts/IN130-N03
mkdir -p "$OUT/raw"
C="python scripts/run_weak_compressibility_campaign.py"
LOG=logs/in130n03
mkdir -p "$LOG"

$C --task poiseuille --out "$OUT" --max-steps 80000 > "$LOG/poiseuille.log" 2>&1
echo "poiseuille done" >> "$LOG/status.log"

$C --task pressure --out "$OUT" --pressure-steps 20000 > "$LOG/pressure.log" 2>&1
echo "pressure done" >> "$LOG/status.log"

$C --task pressure --out "$OUT/conv" --pressure-steps 40000 > "$LOG/pressure_conv.log" 2>&1
echo "pressure_conv done" >> "$LOG/status.log"

$C --task acoustic --out "$OUT" --acoustic-steps 20000 > "$LOG/acoustic.log" 2>&1
echo "acoustic done" >> "$LOG/status.log"

# cylinder: documented divergence probe at the low-Ma corner (cheap: they blow
# up within a few hundred steps) + one best-effort run at the highest stable tau
$C --task cylinder --machs 0.01 0.05 0.1 --out "$OUT" --transient-tu 5 --max-tu 40 \
    --min-periods 3 --sample-interval 4 > "$LOG/cyl_low.log" 2>&1
echo "cyl_low done" >> "$LOG/status.log"

$C --task cylinder --machs 0.3 --out "$OUT" --transient-tu 8 --max-tu 25 \
    --min-periods 3 --sample-interval 4 > "$LOG/cyl_high.log" 2>&1
echo "cyl_high done" >> "$LOG/status.log"

$C --task aggregate --out "$OUT" > "$LOG/aggregate.log" 2>&1
echo "aggregate rc=$?" >> "$LOG/status.log"
python scripts/plot_weak_compressibility.py --artifacts "$OUT" > "$LOG/plots.log" 2>&1
echo "plots done" >> "$LOG/status.log"
python scripts/build_weak_compressibility_report.py --artifacts "$OUT" \
    --docs-out docs/reports/IN130-N03-weak-compressibility.md \
    --payload-out artifacts/IN130-N03/report_payload.json > "$LOG/report.log" 2>&1
echo "report done" >> "$LOG/status.log"
echo ALL-DONE >> "$LOG/status.log"