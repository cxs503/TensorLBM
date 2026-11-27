#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

python tests/test_repository_layout.py -v
python tests/test_repository_report_migration.py -v
python tools/check_repository_layout.py
python tests/test_source_authority.py -v
python tools/check_source_authority.py
python tests/test_benchmark_manuals.py -v
python tools/check_benchmark_manuals.py
python tests/test_benchmark_registry.py -v
python tools/benchmark_registry.py
python benchmarks/status_ledger.py --check --check-artifacts
