# TensorLBM Development Workflow

This is the canonical developer entrypoint for local setup, checks, and platform startup.

## 1) Environment setup

```bash
cd /tmp/workspace/cxs503/TensorLBM
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install -r platform/requirements.txt
```

## 2) Core quality checks

```bash
ruff check src tests examples benchmarks
PYTHONPATH=src pytest -q
mypy src/tensorlbm --ignore-missing-imports
```

## 3) Run representative examples

```bash
PYTHONPATH=src python examples/cylinder_flow.py
PYTHONPATH=src python examples/sphere_flow_3d.py --nx 60 --ny 30 --nz 30 --n-steps 50
```

## 4) Start the web platform

```bash
cd /tmp/workspace/cxs503/TensorLBM/platform
bash start.sh
```

## 5) Output artifact naming convention

Canonical step images use:

```text
flow_step_XXXXXX.png
```

For migration compatibility, selected workflows may also write legacy aliases:

```text
snapshot_XXXXXX.png
```

New code should treat `flow_step_*.png` as the primary naming scheme.

