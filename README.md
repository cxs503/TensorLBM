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

By default, the script runs on CPU and writes output images to `outputs/`.

## Run tests

```bash
PYTHONPATH=src pytest -q
```

## Output

After the run, the example saves:

- `outputs/cylinder_flow_final.png` (velocity magnitude + vorticity)

You can adjust simulation parameters at the top of `examples/cylinder_flow.py` (grid size, Reynolds number, inlet velocity, steps, etc.).

## Current scope and limitations

- Scope: a lightweight D2Q9/BGK cylinder-flow example with CPU-first defaults.
- The boundary treatment is intentionally simple (equilibrium inlet, zero-gradient outlet, bounce-back walls/obstacle).
- This project is currently aimed at clarity and extension, not high-fidelity validation or performance benchmarking.
