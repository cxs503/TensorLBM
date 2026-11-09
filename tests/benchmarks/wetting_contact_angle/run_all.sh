#!/usr/bin/env bash
# Fan the contact-angle cases out over independent processes (one case each).
#   bash run_all.sh <stage>        stage ∈ {a, b1, b2}
# Cases are written to results/cases/; assemble.py turns them into the artifact.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
CASES="$HERE/results/cases"
mkdir -p "$CASES"
STAGE="${1:-a}"
PAR="${PAR:-5}"
THREADS="${THREADS:-6}"
PY="${PYTHON:-python}"

run() { "$PY" "$HERE/run_case_cli.py" --threads "$THREADS" "$@"; }

case "$STAGE" in
  a)
    for g in coarse fine; do
      for th in 70 90 110; do
        echo "--theta $th --grid $g --law young --out $CASES/a_${g}_${th}.json"
      done
    done
    ;;
  b1)
    # calibration scan on the coarse grid (theta only sizes the initial cap)
    for fw in 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.80; do
      echo "--theta 0 --grid coarse --law scan --fw $fw --out $CASES/b1_${fw}.json"
    done
    ;;
  b2)
    # hold-out: calibrated prescription on the fine grid; args come from assemble.py --plan
    cat "$CASES/b2_jobs.txt"
    ;;
  *)
    echo "unknown stage $STAGE" >&2; exit 2 ;;
esac | xargs -P "$PAR" -I{} sh -c "cd '$HERE' && $PY '$HERE/run_case_cli.py' --threads $THREADS {}"

echo "== stage $STAGE done"