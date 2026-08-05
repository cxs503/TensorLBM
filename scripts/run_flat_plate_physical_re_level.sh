#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 L256|L384|L512 PHYSICAL_GPU_UUID [RESULT_DIR]" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage
length=${1#L}
gpu=$2
result_dir=${3:-results/flat_plate_physical_re}
[[ $gpu == GPU-* ]] || usage

case "$length" in
  256)
    nx=512; ny=128; steps=32000; warmup=16000; ramp=2048
    sponge=24; cv=6; exchange=3; report=1024; wall_diagnostic=64
    checkpoint_interval=4096; statistics=16000
    ;;
  384)
    nx=768; ny=192; steps=48000; warmup=24000; ramp=3072
    sponge=36; cv=9; exchange=4.5; report=1536; wall_diagnostic=96
    checkpoint_interval=6144; statistics=24000
    ;;
  512)
    nx=1024; ny=256; steps=64000; warmup=32000; ramp=4096
    sponge=48; cv=12; exchange=6; report=2048; wall_diagnostic=128
    checkpoint_interval=8192; statistics=32000
    ;;
  *) usage ;;
esac

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
mkdir -p "$result_dir"
python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${TENSORLBM_PREFLIGHT_ONLY:-0} == 1 ]]; then
  exec "$python" -c 'import tensorlbm; print(tensorlbm.__file__)'
fi

physical_re=13213381.41322709
stem="$result_dir/flat-plate-v6-physical-re13p213m-l${length}-${steps}"
exec 9>"$stem.lock"
flock 9
if [[ -f "$stem.json" ]]; then
  "$python" - "$stem.json" "$length" "$steps" "$physical_re" <<'PY'
import json
import math
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
configuration = result.get("configuration", {})
observations = result.get("result", {})
valid = (
    result.get("schema") == "tensorlbm-flat-plate-wall-model-v4"
    and int(configuration.get("plate_length", -1)) == int(sys.argv[2])
    and int(configuration.get("steps", -1)) == int(sys.argv[3])
    and math.isclose(
        float(configuration.get("reynolds", math.nan)),
        float(sys.argv[4]),
        rel_tol=1.0e-14,
    )
    and observations.get("finite") is True
)
if not valid:
    raise SystemExit("existing physical-Re flat-plate result is incompatible")
PY
  exit 0
fi
resume=()
if [[ -f "$stem.ckpt" ]]; then
  resume=(--resume)
fi

export CUDA_VISIBLE_DEVICES=$gpu
exec "$python" examples/flat_plate_wall_model_validate.py \
  --device cuda:0 --nx "$nx" --ny "$ny" --nz 3 \
  --plate-length "$length" --plate-start-fraction 0.20 \
  --reynolds "$physical_re" --resolved-reynolds 20000 \
  --lattice-speed 0.06 \
  --steps "$steps" --warmup-steps "$warmup" --ramp-steps "$ramp" \
  --statistics-window-steps "$statistics" \
  --sponge-width "$sponge" --sponge-strength 0.2 --cv-margin "$cv" \
  --wall-law musker --stress-exchange-distance "$exchange" --cs-smag 0.05 \
  --report-interval "$report" --wall-diagnostic-interval "$wall_diagnostic" \
  --checkpoint-interval "$checkpoint_interval" --checkpoint "$stem.ckpt" \
  --output "$stem.json" "${resume[@]}"
