# TensorLBM Examples Framework

## Problem: The "Example-Per-Main" Antipattern

TensorLBM inherited a common scientific computing antipattern from upstream
(listLBM by sorush-khajepor): **100+ standalone scripts** in `examples/`, each
with duplicated `argparse`/`main()` boilerplate:

```python
# examples/lid_driven_cavity.py
def build_parser(): ...
def main():
    args = build_parser().parse_args()
    config = LidDrivenCavityConfig(...)
    run_lid_driven_cavity(config)
if __name__ == "__main__":
    main()

# examples/poiseuille.py — same pattern, different params
# examples/cylinder_flow.py — same pattern, different params
# ... 100+ more
```

**Problems:**
- No way to run all examples programmatically (for CI/verification)
- Duplicated CLI boilerplate across every file
- No central registry — discovering examples requires `ls examples/`
- Hard to test examples as library code
- No standard output format for downstream analysis

## Solution: Structured Example Framework

### Architecture

```
tensorlbm/examples/
├── __init__.py          # Package entry, auto-imports examples
├── base.py              # Example ABC + ExampleResult dataclass
├── registry.py          # Central registry (list/get/run/run_all)
├── runner.py            # Unified CLI entry point
└── lid_driven_cavity.py # First migrated example
```

### Usage

**Programmatic:**
```python
from tensorlbm.examples import registry

# List all examples
print(registry.list_examples())  # ['lid_driven_cavity', ...]

# Run one example
result = registry.run('lid_driven_cavity', n=65, re=100)
print(result.metrics)  # {'l2_u': 0.012, 'l2_v': 0.015, ...}

# Run all examples (for CI)
results = registry.run_all(n_steps=100, device='cpu')
```

**CLI:**
```bash
# List examples
python -m tensorlbm.examples.runner --list

# Run one example
python -m tensorlbm.examples.runner lid_driven_cavity --n 65 --re 100

# Run all examples
python -m tensorlbm.examples.runner --run-all --n-steps 100

# Machine-readable output
python -m tensorlbm.examples.runner --list --json
```

### Adding a New Example

1. Create `tensorlbm/examples/my_example.py`:
```python
from .base import Example, ExampleResult
from . import registry

class MyExample(Example):
    @property
    def name(self): return "my_example"

    def configure(self):
        return {"param1": 42, "param2": "default"}

    def run(self, params):
        # ... do work ...
        return ExampleResult(
            name=self.name, params=params,
            elapsed_s=elapsed, success=True,
            metrics={"key_metric": value},
        )

registry.register(MyExample())
```

2. Import in `tensorlbm/examples/__init__.py`:
```python
from . import my_example  # noqa: F401
```

3. Done — the example is now available via registry and CLI.

### Design Decisions

| Decision | Rationale |
|---|---|
| ABC base class | Enforces consistent interface across all examples |
| Auto-registration on import | Zero-config discovery; `import tensorlbm.examples` populates registry |
| `ExampleResult` dataclass | Standard output format for CI, analysis, and reporting |
| Separate `runner.py` module | Single CLI entry point replaces 100+ `if __name__` blocks |
| Backward compatible | Old `examples/*.py` scripts still work; migration is incremental |

### Migration Status

| Example | Migrated | Notes |
|---|---|---|
| lid_driven_cavity | ✅ | First migrated example |
| poiseuille_2d | ⏳ | Next candidate |
| ... (100+) | ⏳ | Incremental migration |

### Stability Notes

LBM simulations require `tau > 0.5` for stability. The relationship is:
```
tau = nu / cs² + 0.5 = (u_lid * L / Re) / (1/3) + 0.5
```

For Re=100, u_lid=0.1:
- n=33 → tau≈0.60 (marginal, may diverge)
- n=65 → tau≈0.69 (stable)
- n=129 → tau≈0.88 (very stable)

The example defaults use n=65, re=100 for a balance of speed and stability.
