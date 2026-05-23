# TensorLBM

TensorLBM is a CPU-first PyTorch Lattice Boltzmann Method platform focused on **reproducible research experiments** with clear extension points.

## What TensorLBM provides

- A small, explicit public API in `src/tensorlbm/__init__.py`
- **D2Q9** and **D3Q19** lattice primitives (`equilibrium`, `macroscopic`, lattice constants)
- **BGK** and **MRT** collision operators for both 2D and 3D
- Boundary conditions: bounce-back, **Zou/He** inlet-velocity and outlet-pressure BCs
- Momentum-exchange force diagnostics (drag/lift) for the 2D cylinder
- A 2D cylinder-flow runner with CLI, Strouhal-number extraction, structured outputs, and diagnostics
- A 3D sphere-flow runner with CLI and structured outputs
- Batch Reynolds-number parameter scan (`examples/param_scan.py`)
- Automated tests and CI

## Engineering principles

1. **Stable API first**: keep public exports intentional and backward-compatible.
2. **Composable solver core**: isolate lattice math, solver stepping, and boundary logic.
3. **Reproducible runs**: parameterized CLI + deterministic run folder layout + metadata snapshot.
4. **Fast feedback loops**: smoke tests and CI on push/PR.
5. **CPU-first defaults, GPU-ready shape**: default to CPU, but keep interfaces ready for device scaling.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

## Run the cylinder-flow example

Default run:

```bash
PYTHONPATH=src python examples/cylinder_flow.py
```

Small smoke run:

```bash
PYTHONPATH=src python examples/cylinder_flow.py \
  --nx 64 --ny 24 --radius 4 --n-steps 20 --output-interval 10 \
  --run-name smoke --overwrite
```

Useful options:

- `--nx`, `--ny`: grid size
- `--u-in`, `--re`, `--radius`: flow and geometry parameters
- `--n-steps`, `--output-interval`: runtime and output cadence
- `--output-root`, `--run-name`, `--overwrite`: output organization
- `--device`: `cpu` (default), `cuda`, or `mps` (Apple Silicon)

## Run the sphere-flow example (3D)

```bash
PYTHONPATH=src python examples/sphere_flow_3d.py \
  --nx 60 --ny 30 --nz 30 --radius 4 --n-steps 50 --output-interval 25 \
  --run-name smoke --overwrite
```

## Batch parameter scan

```bash
PYTHONPATH=src python examples/param_scan.py \
  --re 20 40 80 100 --nx 160 --ny 60 --n-steps 2000 --output-interval 100
```

## Output organization

Each run writes into:

```text
outputs/
  cylinder_flow/
    <run-name>/
      run_metadata.json
      forces.csv
      flow_step_000200.png
      ...
  sphere_flow/
    <run-name>/
      run_metadata.json
      flow_step_000100.png
      ...
```

`run_metadata.json` includes configuration, derived physical parameters (`nu`, `tau`), runtime info, and diagnostics history.

## Tests and checks

Run tests:

```bash
PYTHONPATH=src pytest -q
```

GitHub Actions runs the same test command on every push and pull request.
