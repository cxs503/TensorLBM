# Release process (PyPI)

How a TensorLBM version gets from `main` to <https://pypi.org/project/tensorlbm/>.

## Versioning

- **Single source of truth**: `src/tensorlbm/_version.py`
  (`__version__ = "0.3.0"`). `pyproject.toml` reads it via
  `[tool.setuptools.dynamic] version = {attr = "tensorlbm._version.__version__"}`,
  so the wheel, the sdist, `importlib.metadata`, and `tensorlbm.__version__`
  all agree without any build-time templating.
- **Scheme**: Semantic Versioning (`MAJOR.MINOR.PATCH`, see CHANGELOG.md).
  Pre-1.0, breaking API changes bump MINOR.
- **CHANGELOG.md**: changes accumulate under `[Unreleased]`; at release time
  that section is renamed to the new version with a date, and a fresh empty
  `[Unreleased]` is added on top.
- `CITATION.cff` carries its own `version:` / `date-released:` fields — update
  them in the same release commit.

## Recommended path: tag push → publish.yml

`.github/workflows/publish.yml` triggers on pushes of tags matching `v*`:

1. **build job** — checks out the tag, builds sdist + wheel with
   `python -m build`, runs `python -m twine check dist/*`, smoke-installs the
   wheel (`pip install dist/*.whl --no-deps` + version assert), uploads
   `dist/` as an artifact.
2. **publish job** — waits on `build`, downloads the artifact, and publishes
   to PyPI with [`pypa/gh-action-pypi-publish`](https://github.com/pypa/gh-action-pypi-publish)
   using **trusted publishing (OIDC)** — no API token stored in repo secrets.

Release steps:

```bash
# 1. Everything intended for the release is merged to main and CI is green
# 2. Bump the version and roll the changelog (one commit)
$EDITOR src/tensorlbm/_version.py CHANGELOG.md CITATION.cff
git commit -am "chore(release): bump version to X.Y.Z"
# 3. Tag and push (annotated tag, name must match vX.Y.Z)
git tag -a vX.Y.Z -m "TensorLBM X.Y.Z"
git push origin main vX.Y.Z
# 4. publish.yml builds, twine-checks, and uploads; watch:
#    https://github.com/cxs503/TensorLBM/actions/workflows/publish.yml
```

### One-time setup (maintainer, web UI only)

- **PyPI trusted publisher**: on pypi.org → *Publishing* → add a GitHub
  publisher with owner `cxs503`, repo `TensorLBM`, workflow `publish.yml`,
  environment `pypi-release`. (Do this before the first release; the project
  must exist on PyPI, which the first trusted-publish run creates.)
- **GitHub environment**: repo *Settings → Environments →* create
  `pypi-release` (optionally add required reviewers for a publish gate).
- Token alternative: create a PyPI API token and use it via the commented
  `password: ${{ secrets.PYPI_API_TOKEN }}` block in `publish.yml`.

## Alternative path: manual `release.sh`

`./release.sh X.Y.Z` automates the local equivalent: verifies a clean tree,
bumps `_version.py`, commits, tags, builds sdist + wheel, smoke-tests the
wheel, and uploads with `twine upload` (needs `~/.pypirc` or `TWINE_PASSWORD`;
`DRY_RUN=1` skips git and upload). Prefer the tag-push flow — it keeps
credentials out of maintainer machines entirely.

## Local verification (what CI checks, on your box)

```bash
python -m pip install -U build twine
python -m build                        # dist/tensorlbm-X.Y.Z.{tar.gz,whl}
python -m twine check dist/*           # metadata renders on PyPI
python -m venv /tmp/venv-check && /tmp/venv-check/bin/pip install dist/*.whl
/tmp/venv-check/bin/python -c "import tensorlbm; print(tensorlbm.__version__)"
```

Optional extras published with the package: `pip install tensorlbm[io]`
(h5py/pyyaml for dataset & config IO) and `tensorlbm[fused]` (Triton fused
kernels; GPU only). Neither is required for `import tensorlbm`.
