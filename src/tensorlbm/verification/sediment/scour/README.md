# Sediment Transport & Scour Verification

This directory contains the verification suite for TensorLBM's sediment transport and scour capabilities.

## Overview

The verification suite validates six machine-checkable criteria (C1-C6) against classical theory and experimental data:

- **C1**: Shields incipient motion threshold vs Soulsby & Whitehouse (1991)
- **C2**: Rouse equilibrium suspended-sediment profile (three independent solvers)
- **C3**: Exner mass conservation (bed volume + coupled exchange)
- **C4**: Negative controls (no-motion gate, zero-settling uniformity, Exner off)
- **C5**: Single-pile equilibrium scour depth (Melville & Chiew band)
- **C6**: JSON contract for ship-platform UKC integration

## Quick Start

```python
from tensorlbm.verification.sediment.scour import (
    shields_case, rouse_case, pile_scour_case
)

# Run Shields threshold verification
c1 = shields_case(nx=16, h_cells=64, steps=60000)
print(f"θ_cr deviation: {c1['theta_cr_deviation']:.2%}")

# Run Rouse profile verification
c2 = rouse_case(nx=16, h_cells=64, steps=40000)
print(f"Rouse deviation: {c2['deviation']:.4f}")

# Run pile scour verification
c3 = pile_scour_case(cells_per_D=12, n_relax=6000, n_cycle=30)
print(f"S/D ratio: {c3['z_s_over_D']:.2f}")
```

## Running the Full Suite

```bash
# Run all verification cases and generate artifacts
python -m tensorlbm.verification.sediment.scour.runner \
    --out artifacts/sediment_scour

# Run tests
pytest tests/test_sediment_scour.py -v
```

## Module Structure

- `sediment.py`: Physics kernels (Shields, Rouse, transport, Exner)
- `lattice.py`: D2Q9 channel solver for hydraulics
- `cases.py`: Three verification cases (Shields, Rouse, pile scour)
- `runner.py`: End-to-end runner with artifact generation

## Bibliographic Anchors

- Shields (1936): Original incipient motion experiments
- Rouse (1937): Equilibrium suspended-sediment profile
- Soulsby & Whitehouse (1991): Shields curve fit
- Melville & Chiew (1999): Pile scour depth band
- Sumer & Fredsoe (2002): Scour mechanics textbook

## Known Limitations

**Model-form limitation (honest disclosure)**: The pile scour case uses a depth-averaged 2D model, which cannot resolve the horseshoe vortex / downflow that dominates 3D pile scour. The case reports both `depth_feedback=True` and `False` variants to make this sensitivity visible.

## Test Results

All 17 tests pass:
- 13 non-slow tests: ~40s
- 4 slow tests: ~130s

See `tests/test_sediment_scour.py` for details.
