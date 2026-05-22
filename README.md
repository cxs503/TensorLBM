# TensorLBM

A minimal PyTorch implementation of a 2D D2Q9 Lattice Boltzmann Method (LBM) demo for flow around a cylinder.

## What is included

- D2Q9 lattice constants and utilities (`src/tensorlbm/d2q9.py`)
- Macroscopic recovery (`rho`, `ux`, `uy`)
- BGK/SRT collision step
- Streaming step
- Cylinder obstacle mask generation
- Simple bounce-back handling for top/bottom walls and the cylinder obstacle
- Runnable cylinder-flow example that saves a visualization image

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the cylinder-flow demo

```bash
PYTHONPATH=src python examples/cylinder_flow.py
```

By default, the script runs on CPU and writes outputs under `outputs/` in a per-run folder.

Common runtime options:

- `--nx`, `--ny`: grid size
- `--steps`: number of simulation steps
- `--output-interval`: interval for writing images
- `--log-interval`: interval for diagnostics logging
- `--radius`, `--cx`, `--cy`: cylinder geometry
- `--re` or `--tau`: Reynolds-number-based setup or direct relaxation parameter
- `--output-root`, `--run-name`: output location and run naming

Example commands:

```bash
PYTHONPATH=src python examples/cylinder_flow.py --steps 300 --output-interval 100
PYTHONPATH=src python examples/cylinder_flow.py --nx 400 --ny 120 --radius 14 --re 120 --run-name re120_case
```

## Output

Each run creates a directory under `outputs/`, for example:

- `outputs/cylinder_nx320_ny100_steps1200_YYYYMMDD-HHMMSS/`
  - `run_metadata.json`
  - `flow_step_000200.png`
  - `flow_step_000400.png`
  - `...`
  - `flow_step_001200.png`

`run_metadata.json` stores the resolved run configuration and derived values (including `tau` and cylinder center).
