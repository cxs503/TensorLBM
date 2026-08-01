#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 L90|L120|L150 PHYSICAL_GPU [RESULT_DIR] [WAIT_FOR_PID]" >&2
  echo "       SUBOFF_HULL_TYPE=bare_hull|full (default: bare_hull)" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 4 ]] || usage
level=${1#L}
gpu=$2
result_dir=${3:-results/amr_campaign_20260801}
wait_for_pid=${4:-}
hull_type=${SUBOFF_HULL_TYPE:-bare_hull}
if [[ "$hull_type" != bare_hull && "$hull_type" != full ]]; then
  echo "SUBOFF_HULL_TYPE must be bare_hull or full" >&2
  exit 2
fi

case "$level" in
  90)
    nx=450; cross=90; outer_wall=6; outer_wake=75
    surface=30; steps=12000; warmup=4500; report=375
    ramp=3000; statistics=7500; wall_diagnostic=60; sponge=18
    checkpoint_interval=750
    ;;
  120)
    nx=600; cross=120; outer_wall=8; outer_wake=100
    surface=40; steps=16000; warmup=6000; report=500
    ramp=4000; statistics=10000; wall_diagnostic=80; sponge=24
    checkpoint_interval=1000
    ;;
  150)
    nx=750; cross=150; outer_wall=10; outer_wake=125
    surface=50; steps=20000; warmup=7500; report=625
    ramp=5000; statistics=12500; wall_diagnostic=100; sponge=30
    checkpoint_interval=1250
    ;;
  *) usage ;;
esac

if [[ -n "$wait_for_pid" ]]; then
  [[ "$wait_for_pid" =~ ^[0-9]+$ ]] || usage
  while kill -0 "$wait_for_pid" 2>/dev/null; do
    sleep 20
  done
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
mkdir -p "$result_dir"
if [[ -n ${TENSORLBM_PYTHON:-} ]]; then
  python=$TENSORLBM_PYTHON
elif [[ -x $root/.venv/bin/python ]]; then
  python=$root/.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
  python=$(command -v python3)
else
  echo "no Python interpreter found; set TENSORLBM_PYTHON" >&2
  exit 127
fi
if [[ ! -x $python ]]; then
  echo "TensorLBM Python is not executable: $python" >&2
  exit 126
fi
export PYTHONPATH="$root/src:$root/examples${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${TENSORLBM_PREFLIGHT_ONLY:-0} == 1 ]]; then
  exec "$python" -c 'import tensorlbm; print(tensorlbm.__file__)'
fi

variant=
if [[ "$hull_type" == full ]]; then
  variant=-aff8
fi
campaign_generation=${TENSORLBM_CAMPAIGN_GENERATION:-v3}
if [[ ! $campaign_generation =~ ^v[0-9]+$ ]]; then
  echo "TENSORLBM_CAMPAIGN_GENERATION must look like v3 or v4" >&2
  exit 2
fi
checkpoint="$result_dir/suboff-nested-${campaign_generation}${variant}-equivalent-l${level}-${steps%000}k.ckpt"
output="$result_dir/suboff-nested-${campaign_generation}${variant}-equivalent-l${level}-${steps%000}k.json"
resume=()
if [[ -f "$checkpoint" ]]; then
  resume=(--resume)
fi
restriction_filter=()
if [[ ${TENSORLBM_REGULARIZE_RESTRICTION:-0} == 1 ]]; then
  restriction_filter=(--regularize-restriction)
fi
prolongation_filter=()
if [[ ${TENSORLBM_REGULARIZE_PROLONGATION:-0} == 1 ]]; then
  prolongation_filter=(--regularize-prolongation)
fi
ghost_interpolation=${TENSORLBM_GHOST_INTERPOLATION:-injection}
if [[ $ghost_interpolation != injection && $ghost_interpolation != trilinear ]]; then
  echo "TENSORLBM_GHOST_INTERPOLATION must be injection or trilinear" >&2
  exit 2
fi
reflux_correction_stencil=${TENSORLBM_REFLUX_CORRECTION_STENCIL:-exterior_cells}
if [[ $reflux_correction_stencil != exterior_cells && $reflux_correction_stencil != crossing_links ]]; then
  echo "TENSORLBM_REFLUX_CORRECTION_STENCIL must be exterior_cells or crossing_links" >&2
  exit 2
fi
transfer_positivity=()
if [[ ${TENSORLBM_ENFORCE_TRANSFER_POSITIVITY:-0} == 1 ]]; then
  transfer_positivity=(--enforce-transfer-positivity)
fi
inner_wall_margin=${TENSORLBM_INNER_WALL_MARGIN:-4}
inner_wake_cells=${TENSORLBM_INNER_WAKE_CELLS:-8}
cv_margin=${TENSORLBM_CV_MARGIN:-4}
aux_cv_margins=${TENSORLBM_AUX_CV_MARGINS:-2,6}
resolved_reynolds_start=${TENSORLBM_RESOLVED_REYNOLDS_START:-0}
viscosity_ramp_start=${TENSORLBM_VISCOSITY_RAMP_START_STEP:-0}
viscosity_ramp_end=${TENSORLBM_VISCOSITY_RAMP_END_STEP:-0}
health_interval=${TENSORLBM_HEALTH_INTERVAL:-$report}
interface_filter_width=${TENSORLBM_INTERFACE_FILTER_WIDTH:-0}
interface_filter_strength=${TENSORLBM_INTERFACE_FILTER_STRENGTH:-0}
maximum_reflux_applied_correction_fraction=${TENSORLBM_MAXIMUM_REFLUX_APPLIED_CORRECTION_FRACTION:-0.001}
cs_smag=${TENSORLBM_CS_SMAG:-0.05}
wale_cw=${TENSORLBM_WALE_CW:-0.5}
vreman_cv=${TENSORLBM_VREMAN_CV:-0.025}
collision_model=${TENSORLBM_COLLISION_MODEL:-cumulant_smagorinsky}
case "$collision_model" in
  cumulant_smagorinsky|cumulant_wale|cumulant_vreman|entropic_kbc|natural_kbc) ;;
  *) echo "unsupported TENSORLBM_COLLISION_MODEL: $collision_model" >&2; exit 2 ;;
esac
stress_exchange_distance=${TENSORLBM_STRESS_EXCHANGE_DISTANCE:-1}
wall_ramp_options=()
if [[ -n ${TENSORLBM_WALL_NORMAL_RAMP_STEPS:-} ]]; then
  wall_ramp_options+=(--wall-normal-ramp-steps "$TENSORLBM_WALL_NORMAL_RAMP_STEPS")
fi
if [[ -n ${TENSORLBM_WALL_SHEAR_RAMP_STEPS:-} ]]; then
  wall_ramp_options+=(--wall-shear-ramp-steps "$TENSORLBM_WALL_SHEAR_RAMP_STEPS")
fi

export CUDA_VISIBLE_DEVICES=$gpu
exec "$python" examples/suboff_nested_static_amr_smoke.py \
  --device cuda:0 --hull-type "$hull_type" --speed-knots 5.92 \
  --nx "$nx" --ny "$cross" --nz "$cross" --hull-length "$level" \
  --center-x-fraction 0.3 \
  --outer-wall-margin "$outer_wall" --outer-wake-cells "$outer_wake" \
  --inner-wall-margin "$inner_wall_margin" \
  --inner-wake-cells "$inner_wake_cells" \
  --cv-margin "$cv_margin" --aux-cv-margins "$aux_cv_margins" \
  --surface-force-interval "$surface" \
  --steps "$steps" --warmup-steps "$warmup" \
  --statistics-window-steps "$statistics" --ramp-steps "$ramp" \
  "${wall_ramp_options[@]}" \
  --report-interval "$report" --wall-diagnostic-interval "$wall_diagnostic" \
  --health-interval "$health_interval" --maximum-health-speed 0.3 \
  --maximum-reflux-applied-correction-fraction "$maximum_reflux_applied_correction_fraction" \
  --minimum-convective-times 8 \
  --minimum-target-reynolds-convective-times 7.5 \
  --minimum-statistics-convective-times 5 \
  --lattice-speed 0.06 --resolved-reynolds 100000 \
  --resolved-reynolds-start "$resolved_reynolds_start" \
  --viscosity-ramp-start-step "$viscosity_ramp_start" \
  --viscosity-ramp-end-step "$viscosity_ramp_end" \
  --collision-model "$collision_model" \
  --cs-smag "$cs_smag" --wale-cw "$wale_cw" --vreman-cv "$vreman_cv" \
  --wall-law musker --stress-exchange-distance "$stress_exchange_distance" \
  --sponge-width "$sponge" --sponge-strength 0.3 \
  --far-field-mode non_equilibrium_extrapolation \
  --memory-bytes-per-cell 742 \
  --ghost-interpolation "$ghost_interpolation" \
  --reflux-correction-stencil "$reflux_correction_stencil" \
  --interface-filter-width "$interface_filter_width" \
  --interface-filter-strength "$interface_filter_strength" \
  --checkpoint "$checkpoint" --checkpoint-interval "$checkpoint_interval" \
  --output "$output" "${restriction_filter[@]}" \
  "${prolongation_filter[@]}" "${transfer_positivity[@]}" "${resume[@]}"
