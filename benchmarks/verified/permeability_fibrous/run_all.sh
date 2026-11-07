#!/bin/bash
# ladder driver (CPU, sequential, logged)
set -x
cd "$(dirname "$0")"
OUT=out
mkdir -p $OUT
for g in 9 13 19 28 40; do
  for c in hex_vf0.70 hex_vf0.75; do
    python3 -u run.py case --case $c --grid $g --closure on  --out-dir $OUT
  done
done
for g in 16 24 36 54; do
  python3 -u run.py case --case sq_vf0.40 --grid $g --closure on --out-dir $OUT
done
for c in hex_vf0.70 hex_vf0.75; do
  for g in 9 19 40; do
    python3 -u run.py case --case $c --grid $g --closure off --out-dir $OUT
  done
done
for g in 16 36 54; do
  python3 -u run.py case --case sq_vf0.40 --grid $g --closure off --out-dir $OUT
done
for c in sq_vf0.05 sq_vf0.10; do
  python3 -u run.py case --case $c --grid 24 --closure on  --out-dir $OUT
  python3 -u run.py case --case $c --grid 24 --closure off --out-dir $OUT
done
echo DONE_ALL
