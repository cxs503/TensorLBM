#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
  echo "usage: $0 RESULT_DIR" >&2
  exit 2
fi
result_dir=$1
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=${TENSORLBM_PYTHON:-/home/wxsc/anaconda3/envs/ftw-env/bin/python}
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
inputs=(
  "$result_dir/sphere-v9-natural-kbc-corrected-bfl-no-inlet-r9-7200.json"
  "$result_dir/sphere-v9-natural-kbc-corrected-bfl-no-inlet-r12-9600.json"
  "$result_dir/sphere-v9-natural-kbc-corrected-bfl-no-inlet-r15-12000.json"
)
for input in "${inputs[@]}"; do
  [[ -f $input ]] || {
    echo "missing corrected-BFL sphere member: $input" >&2
    exit 2
  }
done
exec "$python" "$root/examples/sphere_grid_convergence_assess.py" \
  "${inputs[@]}" \
  --output "$result_dir/sphere-v9-natural-kbc-corrected-bfl-r9-r12-r15-convergence.json"
