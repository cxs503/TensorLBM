#!/usr/bin/env bash
# Build the palabos-based reference tool for XVAL-LBM-TAYLORGREEN.
#
# The palabos library itself is built from the upstream git mirror
# (https://gitlab.com/unigespc/palabos) into $PLB_BUILD; see docs/XVAL_TAYLORGREEN.md
# for the exact commands (library-only CMake project, no MPI, no HDF5, C++20).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLB_ROOT="${PLB_ROOT:-/tmp/palabos}"
PLB_BUILD="${PLB_BUILD:-/tmp/plbbuild}"
OUT="${1:-$HERE/bin}"
mkdir -p "$OUT"

g++ -O3 -DNDEBUG -std=c++20 -DPLB_USE_POSIX \
    -I"$PLB_ROOT/src" -I"$PLB_ROOT/externalLibraries" -I"$PLB_ROOT/externalLibraries/Eigen3" \
    "$HERE/tg2d_palabos.cpp" "$PLB_BUILD/libpalabos.a" -o "$OUT/tg2d_palabos"
echo "built $OUT/tg2d_palabos"
