#!/bin/bash
# Historical m23309 report submission utility for a local TensorPlatform service.
set -euo pipefail

report=${1:-/tmp/report.json}
: "${TENSOR_REPORT_TOKEN:?Set TENSOR_REPORT_TOKEN locally before use}"
if [[ ! -f "$report" || ! -r "$report" ]]; then
    printf 'Report is not readable: %s\n' "$report" >&2
    exit 1
fi

printf 'Authorization: Bearer %s\n' "$TENSOR_REPORT_TOKEN" |
    curl --fail-with-body --silent --show-error --request POST \
        'http://127.0.0.1:8010/api/v1/tensor/reports' \
        --header @- --header 'Content-Type: application/json' \
        --data-binary "@$report"
