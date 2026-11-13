#!/usr/bin/env bash
# XVAL-LBM-METHOD-SELECTION-FIRST-EVIDENCE — resolution ladder on cluster nodes.
# Rungs: D=14 and D=20 for A1 (MRT+sponge), A2 (BGK, OSS selection), A4 (MRT, no sponge);
# D=14 failure probes for A3 (TRT magic Lambda=3/16) and A5 (TRT tau_minus=2.0).
# Predeclared in predeclared_v3.json. Each arm runs to completion (no wall-clock chunking
# needed: the cluster job budget is far larger than one rung).
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=xval/benchmark_method_selection
mkdir -p "$ROOT/runs" "$ROOT/logs"
python -c "import torch, numpy; print('env', torch.__version__, numpy.__version__, torch.get_num_threads())"

arm() {  # arm D threads
  python "$ROOT/run_arm.py" --arm "$1" --D "$2" --threads "$3" > "$ROOT/logs/$1_D$2.log" 2>&1
  echo "[ladder] $1_D$2 exit=$? $(date -u +%FT%TZ)"
}

arm A2 14 4 &
arm A1 14 4 &
arm A4 14 4 &
arm A3 14 2 &
arm A5 14 2 &
wait
arm A2 20 4 &
arm A1 20 8 &
arm A4 20 8 &
wait

# single JSON bundle of every ladder rung so the numbers can be read back from the job log
python - <<'PY'
import json, pathlib
root = pathlib.Path("xval/benchmark_method_selection/runs")
bundle = {}
paths = []
for pat in ("*_D1[4-9]/result.json", "*_D2[0-9]/result.json",
            "*_D1[4-9]/fail.json", "*_D2[0-9]/fail.json"):
    paths += sorted(root.glob(pat))
for p in paths:
    bundle[p.parent.name] = json.loads(p.read_text())
print("LADDER_BUNDLE_BEGIN")
print(json.dumps(bundle, indent=1))
print("LADDER_BUNDLE_END")
PY
echo "[ladder] all done $(date -u +%FT%TZ)"