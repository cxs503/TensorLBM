# TensorLBM

TensorLBM is a lightweight starter scaffold for experimenting with
Lattice Boltzmann Method (LBM) simulations using **PyTorch** tensors.

This repository currently provides:
- a clean `src/` package layout,
- a minimal D2Q9 lattice implementation,
- a short CPU simulation loop,
- and a runnable example script.

## Quickstart

### 1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run the minimal example

```bash
PYTHONPATH=src python examples/minimal_d2q9.py
```

You should see a short run complete and print summary statistics.

## Project structure

```text
.
├── examples/
│   └── minimal_d2q9.py
├── requirements.txt
└── src/
    └── tensorlbm/
        ├── __init__.py
        ├── constants.py
        ├── lattice.py
        └── simulation.py
```

## Notes

- Scope is intentionally modest for an initial bootstrap.
- The current implementation is CPU-first and designed to be easy to extend.