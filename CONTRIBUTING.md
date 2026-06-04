# Contributing to TensorLBM

Thank you for your interest in contributing!  This document describes the project conventions, development workflow, and guidelines for adding new solvers or benchmarks.

---

## Table of Contents

1. [Development Setup](#1-development-setup)
2. [Coding Conventions](#2-coding-conventions)
3. [Testing](#3-testing)
4. [Adding a New Solver or Benchmark](#4-adding-a-new-solver-or-benchmark)
5. [Pull Request Workflow](#5-pull-request-workflow)

---

## 1. Development Setup

Canonical workflow: see `docs/development_workflow.md`.

```bash
# Clone the repository
git clone https://github.com/cxs503/TensorLBM.git
cd TensorLBM

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install in editable mode with all dev dependencies
pip install -e ".[dev]"
pip install -r platform/requirements.txt
```

Minimum requirements: **Python ≥ 3.11**, **PyTorch ≥ 2.0**.

---

## 2. Coding Conventions

### Style

- Line length: **100 characters** (enforced by `ruff`).
- Target Python: **3.11+** syntax.
- Use `from __future__ import annotations` in every source file.
- Use `typing.Protocol` for extension points (see `protocols.py`).

### Type annotations

- All public functions must have complete type annotations.
- Use `mypy --strict` as the standard (run via `mypy src/tensorlbm`).
- Avoid `Any` in public API; use specific types or `TypeVar`.

### Documentation

- Every public function, class, and module must have a Google-style docstring.
- Module-level docstrings should describe the physics, list all exported symbols, and cite relevant references.
- Reference format: `Author (year) Journal volume page` (no DOI needed).

### Imports

- Standard library first, then third-party (torch, numpy, matplotlib), then local.
- Separate import groups with a blank line.
- Keep `TYPE_CHECKING` guard for heavy imports only used in annotations.

### Naming

| Concept | Convention | Example |
|---------|-----------|---------|
| Distribution tensor | `f` (D2Q9), `f3d` or `f` (D3Q19/27) | `f: torch.Tensor` |
| Lattice direction index | `i` or `q` | `for i in range(9)` |
| Density / velocity | `rho`, `ux`, `uy`, `uz` | — |
| Relaxation time | `tau` | `tau: float = 0.7` |
| Solid mask | `*_mask` | `wall_mask`, `obstacle_mask` |
| Config dataclass | `*Config` | `CylinderFlowConfig` |
| Runner function | `run_*` | `run_cylinder_flow` |
| Canonical step image | `flow_step_XXXXXX.png` | `flow_step_000500.png` |

---

## 3. Testing

Run the full test suite:

```bash
PYTHONPATH=src pytest -q
```

Run with coverage:

```bash
PYTHONPATH=src pytest -q --cov=tensorlbm --cov-report=term-missing
```

Run type checking:

```bash
mypy src/tensorlbm --ignore-missing-imports
```

Run the linter:

```bash
ruff check src tests examples benchmarks
```

### Test conventions

- Every new function must have at least one test covering:
  1. **Shape** — output tensor has the expected shape.
  2. **Conservation** — mass and momentum are conserved (for collision operators).
  3. **Fixed-point** — equilibrium is unchanged by collision.
  4. **Finite output** — no NaN or Inf values.
- Physics benchmarks should include a convergence or accuracy test comparing against an analytical solution or reference data.
- Test files go in `tests/` and are named `test_<module>.py`.
- Do **not** remove or comment out existing tests.

---

## 4. Adding a New Solver or Benchmark

### Checklist

- [ ] Create `src/tensorlbm/<name>.py` with a module-level docstring.
- [ ] Export all public symbols in the module's `__all__` list.
- [ ] Add imports to `src/tensorlbm/__init__.py` and update `__all__`.
- [ ] Write tests in `tests/test_<name>.py` covering the checklist above.
- [ ] Add an example script to `examples/<name>.py` with a `--help` CLI.
- [ ] Update `CHANGELOG.md` under the `[Unreleased]` section.
- [ ] Update `docs/software_manual.md` if the new module introduces a new benchmark or engineering application.

### Collision operator pattern

A new collider should have the signature:

```python
def collide_mymodel(f: torch.Tensor, tau: float, **kwargs: float) -> torch.Tensor:
    ...
```

It must satisfy `CollisionOperator` from `protocols.py` (mass and momentum conserved, identity at equilibrium).

### Config / runner pattern

Follow the existing pattern:

```python
@dataclass(frozen=True)
class MyConfig:
    nx: int = 128
    ...
    output_root: Path = Path("outputs")
    run_name: str | None = None

    def validate(self) -> None: ...
    def resolved_run_name(self) -> str: ...
    def save(self, path: str | Path) -> Path: ...
    @classmethod
    def load(cls, path: str | Path) -> MyConfig: ...

def run_my_simulation(config: MyConfig) -> Path: ...
```

---

## 5. Pull Request Workflow

1. **Fork** the repository and create a feature branch: `git checkout -b feature/my-feature`.
2. Make your changes, following the coding conventions above.
3. Ensure all tests pass and linting is clean.
4. Open a pull request against `main` with a clear title and description.
5. Reference relevant issues in the description.
6. Respond to review comments promptly.

For bug fixes, please include a regression test that would have caught the bug.
