# Ship Cabin Radiation Enclosure Benchmark

## Physical Setup

Three-surface enclosure representing a ship cabin:
- **Deck (floor)**: 4m × 6m, ε=0.9, T=300K, solar absorptance=0.6
- **Bulkhead (wall)**: 4m × 3m, ε=0.8, T=295K, solar absorptance=0.5
- **Overhead (ceiling)**: 6m × 3m, ε=0.85, T=298K, solar absorptance=0.4

**Solar load**: 800 W/m² from 45° elevation angle
**Sky radiation**: Berdahl-Martin model with T_ambient=295K, T_dewpoint=285K

## View Factors

View factors computed using analytical solutions and normalized to satisfy:
- Summation rule: Σ_j F_ij = 1
- Reciprocity: A_i F_ij = A_j F_ji

## Results

See `result.json` for complete results including:
- Net radiation flux per surface [W/m²]
- Solar and sky radiation contributions [W/m²]
- 8-12 μm band radiance for IR signature [W/(m²·sr)]
- Energy conservation residual

## Running

```bash
PYTHONPATH=src python benchmarks/verified/radiation_enclosure_ship/run.py
```

## Verification

- View factor summation and reciprocity checks
- Energy conservation: |Σ q_i A_i| / Σ|q_i A_i|
- Gauss-Seidel vs direct solve consistency
