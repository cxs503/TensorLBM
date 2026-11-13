#!/usr/bin/env bash
# Fetch, build and run the OSS reference used by this XVAL (zero hand-copying).
#
#   lbm-principles-practice/code @ 6e2c592fdc3592c14dfd52f860fc1ceea930bcb0
#   chapter8/film_inamuro.cpp  (Krüger et al., "The Lattice Boltzmann Method:
#   Principles and Practice", Springer 2017; MIT licence)
#
# Usage:  ./reproduce_oss.sh [work_dir]      (default: /tmp/oss_lbm)
set -euo pipefail
OSS_REPO="https://github.com/lbm-principles-practice/code"
OSS_COMMIT="6e2c592fdc3592c14dfd52f860fc1ceea930bcb0"
WORK="${1:-/tmp/oss_lbm}"
if [ ! -d "$WORK/.git" ]; then
  git clone "$OSS_REPO" "$WORK"
fi
git -C "$WORK" fetch --depth 1 origin "$OSS_COMMIT"
git -C "$WORK" checkout -q "$OSS_COMMIT"
g++ -O2 -o "$WORK/film_inamuro" "$WORK/chapter8/film_inamuro.cpp"
mkdir -p "$WORK/run/film_inamuro"
( cd "$WORK/run" && "../film_inamuro" )      # ~110 s for N=50000, writes film_inamuro/film_inamuro<step>.dat
echo "reference fields in $WORK/run/film_inamuro/"
