#!/usr/bin/env bash
# release.sh — bump version, build, and upload TensorLBM to PyPI.
#
# Usage:
#   ./release.sh <new-version>       # e.g.  ./release.sh 0.4.0
#
# Prerequisites:
#   pip install build twine
#   A valid ~/.pypirc or TWINE_PASSWORD env var (API token) set.
#
# The script is intentionally conservative:
#   1. Checks the working tree is clean.
#   2. Bumps _version.py.
#   3. Creates a commit and an annotated tag.
#   4. Builds sdist + wheel with python -m build.
#   5. Runs a quick smoke test on the wheel.
#   6. Uploads to PyPI with twine (requires confirmation).
#   7. Reminds you to push the tag.
#
# Set DRY_RUN=1 to skip the git commit/tag and PyPI upload.

set -euo pipefail

VERSION_FILE="src/tensorlbm/_version.py"
NEW_VERSION="${1:-}"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ -z "$NEW_VERSION" ]]; then
  echo "Usage: $0 <new-version>  (e.g. $0 0.4.0)" >&2
  exit 1
fi

if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(\.post[0-9]+|\.dev[0-9]+|[ab][0-9]+)?$ ]]; then
  echo "ERROR: '$NEW_VERSION' does not look like a PEP 440 version." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: Working tree is not clean. Commit or stash your changes first." >&2
  exit 1
fi

DRY_RUN="${DRY_RUN:-0}"

# ── Show current version ──────────────────────────────────────────────────────
CURRENT=$(python -c "from tensorlbm._version import __version__; print(__version__)" 2>/dev/null || echo "unknown")
echo "Current version : $CURRENT"
echo "New version     : $NEW_VERSION"
echo ""
read -r -p "Proceed? [y/N] " confirm
[[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

# ── Bump version file ─────────────────────────────────────────────────────────
sed -i "s/__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" "$VERSION_FILE"
echo "Updated $VERSION_FILE → $NEW_VERSION"

# ── Commit and tag ────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == "0" ]]; then
  git add "$VERSION_FILE"
  git commit -m "chore(release): bump version to $NEW_VERSION"
  git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"
  echo "Created commit and tag v$NEW_VERSION"
else
  echo "[DRY_RUN] Skipping git commit and tag."
fi

# ── Build ─────────────────────────────────────────────────────────────────────
rm -rf dist/
python -m build
echo ""
echo "Built artifacts:"
ls -lh dist/

# ── Smoke test ────────────────────────────────────────────────────────────────
WHEEL=$(ls dist/*.whl | head -1)
python -m pip install "$WHEEL" --force-reinstall --quiet
INSTALLED=$(python -c "import tensorlbm; print(tensorlbm.__version__)")
if [[ "$INSTALLED" != "$NEW_VERSION" ]]; then
  echo "ERROR: Installed version '$INSTALLED' != expected '$NEW_VERSION'" >&2
  exit 1
fi
echo "Smoke test passed: tensorlbm $INSTALLED installs cleanly."

# ── Upload ────────────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == "0" ]]; then
  echo ""
  read -r -p "Upload to PyPI now? [y/N] " upload_confirm
  if [[ "$upload_confirm" =~ ^[Yy]$ ]]; then
    python -m twine upload dist/*
    echo ""
    echo "✓ Released tensorlbm $NEW_VERSION to PyPI."
    echo ""
    echo "Next steps:"
    echo "  git push origin main"
    echo "  git push origin v$NEW_VERSION"
  else
    echo "Upload skipped. Run: python -m twine upload dist/*"
  fi
else
  echo "[DRY_RUN] Skipping twine upload."
  echo "Built artifacts are in dist/ — upload manually with: python -m twine upload dist/*"
fi
