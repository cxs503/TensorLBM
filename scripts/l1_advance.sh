#!/usr/bin/env bash
# Advance the resumable Re_tau=180 channel run for one backend tier.
#
# The wall time of a single shell command is capped in this environment, so the
# chunked driver is called repeatedly inside one command until either the run is
# complete or the wall-clock budget is spent.  Each call checkpoints the
# population field and the running statistics, so progress is never lost and the
# loop can simply be re-issued.
#
#   scripts/l1_advance.sh <profile> <chunk_steps> <budget_seconds> [extra args]
#
# Extra args are forwarded to the driver (grid, smagorinsky, wall scheme, ...).
set -u
PROFILE=${1:?profile required}
CHUNK=${2:-500}
BUDGET=${3:-100}
shift 3 || true
PY=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
START=$(date +%s)
while :; do
  $PY "$HERE/scripts/run_p106w3a2_n01_equivalence.py" --stage l1 --profile "$PROFILE" \
      --advance "$CHUNK" "$@" || exit 1
  NOW=$(date +%s)
  DONE=$($PY - "$HERE" "$PROFILE" "$@" <<'PY'
import json, sys
from pathlib import Path
here, profile = sys.argv[1], sys.argv[2]
args = sys.argv[3:]
wall = "interpolated"
tag = "full"
for i, a in enumerate(args):
    if a == "--wall-scheme":
        wall = args[i + 1]
    if a == "--tag":
        tag = args[i + 1]
state = Path(here) / "artifacts" / "p106w3a2_n01" / "l1" / f"state_{profile}_{wall}_{tag}" / "state.pt"
import torch
print(int(torch.load(state, map_location="cpu", weights_only=False)["step_count"]))
PY
)
  echo "steps_done=${DONE} elapsed=$((NOW - START))s"
  if [ "$DONE" -ge 10000 ]; then echo "complete"; break; fi
  if [ $((NOW - START)) -ge "$BUDGET" ]; then echo "budget spent"; break; fi
done