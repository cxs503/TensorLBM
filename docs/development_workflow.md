# TensorLBM Development Workflow

This is the canonical developer entrypoint. Commands below run from the repository
root unless noted; no personal machine path is required. File ownership and output
retention follow the [repository layout policy](architecture/repository_layout.md).

## 1) Environment setup

```bash
# In your TensorLBM checkout:
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Only install the web platform dependencies if working on it:

```bash
pip install -r app/requirements.txt
```

Current Python CI installs from `pyproject.toml`, not from `uv.lock`. Record the
resolved package versions when producing numerical evidence; a successful install
alone does not establish reproducibility or access to a GPU.

## 2) Choose the check matching your change

| Scope | Command | What it does not prove |
|---|---|---|
| File placement | `python tools/check_repository_layout.py` | Link validity, numerical correctness or completed migration |
| Governance tooling | `python tests/test_repository_layout.py -v` and `python tests/test_audit_import_graph.py` | Runtime import behavior |
| Lint / format | `ruff check src tests examples benchmarks` and `ruff format --check src tests examples benchmarks` | Numerical correctness; existing unrelated debt can still fail |
| Fast library suite | `pytest -q -m "not slow" --timeout=600` | GPU/long-campaign acceptance; fast scope still has environment-dependent tests |
| Focused regression | `pytest -q tests/test_<module>.py` | Whole-library acceptance |
| Full library suite | `pytest -q --timeout=1800` | Successful GPU execution when hardware is absent |
| Platform suite | `pytest -q app/tests` (requires platform dependencies) | A built/served frontend unless explicitly tested |
| Core typing | `mypy src/tensorlbm/d2q9.py src/tensorlbm/d3q19.py src/tensorlbm/solver.py src/tensorlbm/utils.py --ignore-missing-imports` | Full package type safety |
| Full typing | `mypy src/tensorlbm --ignore-missing-imports` | This is currently advisory in CI |

`pytest` uses the checkout's `src/` through `pyproject.toml`. The default test path
is `tests/`; run `app/tests` explicitly. Report passes, failures, skips, xfails,
XPASS and collection exclusions separately. A green run with quarantined tests
is not a claim that those defects are fixed. Do not change physical thresholds
or quarantine entries just to make a file move pass.

New `tools/` code is outside the existing broad Ruff command: lint changed tooling
explicitly, e.g. `ruff check tools/check_repository_layout.py`.

Frontend checks (Node/npm required):

```bash
(cd frontend-vue && npm ci && npm run build)
```

This builds ignored `frontend-vue/dist/`; do not commit that directory. The command
is a local verification entry point, not a claim that a frontend build CI job has
already been added. Use the generated build when testing serving behavior.

## 3) Representative examples

```bash
python examples/cylinder_flow.py --nx 64 --ny 24 --radius 4 --n-steps 20 \
  --output-interval 10 --run-name smoke --overwrite
python examples/sphere_flow_3d.py --nx 60 --ny 30 --nz 30 --n-steps 50
```

These are smoke runs, not physics validation campaigns. Check the example's
`--help` and document its configuration before making an accuracy claim.

## 4) Start the web platform

From the repository root:

```bash
bash app/start.sh
```

The launcher installs app dependencies and starts the development server. It does
not build the Vue frontend. See [app documentation](../app/README.md) for deployment.

## 5) Outputs and evidence

Use ignored `outputs/` for new simulation runs and `artifacts/runs/` for local
maintenance diagnostics. Track only curated evidence under the appropriate
benchmark or `evidence/<topic>/`, with commands, environment, source revision,
precision/device, results and limitations. Do not duplicate the same raw run in
`docs/`, `agent_work/` and `benchmarks/`.

Canonical step images use `flow_step_XXXXXX.png`. Selected workflows may also
write legacy `snapshot_XXXXXX.png` aliases; new code should use `flow_step_*.png`.

Before a relocation, fill in [the migration manifest](architecture/migration_manifest.csv),
search all consumers and relative paths, and define a compatibility/verification
plan. For small layout-only patches, preserve numerical code and compare relevant
checks before and after the move.
