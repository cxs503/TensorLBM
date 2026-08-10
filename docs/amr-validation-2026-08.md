# Nested-AMR Sphere & SUBOFF Validation (2026-08)

Evidence-gated validation of block-structured nested AMR on sphere flow
(Re=100) and DARPA SUBOFF (physical Re=1.32e7). All numbers are measured on
RTX 3090 (local) and Wuxi Supercomputing Center 5×RTX 3090.

## TL;DR

- **3-level nested body-fitted shell AMR** reaches **5.16% drag error** on the
  Re=100 sphere using **0.6% of the uniform-grid memory** (uniform R16
  reference: 4.45%).
- **Checkpoint/resume** is bit-identical to an uninterrupted run (segmented
  8000-step run == single run, Cd 1.1481 both).
- **Multi-GPU distribution** (L1/L2/L3 across 2 GPUs) is bit-identical to
  single-GPU (Cd 1.821415 both) at **2.3× speedup**.
- **D3Q27 lattice** support (cumulant + cascaded) added for sphere and SUBOFF,
  including the first full-stencil D3Q27 Bouzidi BFL implementation.
- High-Re SUBOFF (tau=0.500007) **requires LES** (WALE/Smagorinsky); pure
  laminar cumulant diverges — deployment must pass `--collision-model
  mrt_les`/`cumulant_smagorinsky`.

## Results matrix (sphere Re=100, Schiller-Naumann Cd=1.0917, 8000 steps)

| Scheme | Cd | Error | Memory saving | Notes |
|---|---|---|---|---|
| Uniform R16 (320×224×224) | 1.1403 | 4.45% | 0% | accuracy baseline |
| **L3 3-level shell (D3Q19)** | **1.1481** | **5.16%** | **99.4%** | cumulant |
| L3 3-level shell (D3Q27) | 1.1483 | 5.18% | 99.4% | 27-channels ≈ 19 |
| L2 combo shell (cumulant) | 1.1576 | 6.03% | 97.0% | |
| L2 combo shell (cascaded) | 1.1583 | 6.10% | 97.0% | |
| Single shell (route 1) | 1.1783 | 7.93% | 84.1% | |
| Cellwise adaptive | 8.03 | 636% | — | failed on GPU |

## Key findings

1. **Shell margin was a dead parameter** in `HullProximityRegion.expand_mask()`
   (single 3×3×3 dilation regardless of margin). Fixed: true N-pass dilation.
   All pre-fix "shell" runs were effectively 1-cell shells (combo error
   18.8% → 6.03% after the fix).
2. **Blunt bodies need volumetric fine region around the whole surface**:
   shell-only refinement works for slender bodies (SUBOFF), but for spheres
   the L1 shell must be thick (16 cells) or accuracy degrades.
3. **Wall function only on the finest level** (L3): applying it on multiple
   levels double-counts wall stress.
4. **D3Q27 SUBOFF BFL needed wall_model_slip semantics** (u - act·(u·n)n,
   tangential slip preserved). The initial `(1-act)·u` full-frame following
   pinned the wall to zero at activation 1 → -3200N spurious drag.

## Modules

| Module | Purpose |
|---|---|
| `src/tensorlbm/sphere_amr_common.py` | Lattice-dispatchable sphere AMR assembly (D3Q19/27, cumulant/cascaded) |
| `src/tensorlbm/amr_checkpoint.py` | Crash-resistant save/resume with config-signature gating |
| `src/tensorlbm/amr_shell_planning.py` | Body-fitted shell + wake → BoxRegion planning |
| `src/tensorlbm/evidence_io.py` | Evidence JSON output helpers |
| `src/tensorlbm/interpolated_bc_suboff_d3q27.py` | D3Q27 SUBOFF BFL q-field (bisection raycast) |

## Runners

- `examples/amr_sphere_shell_l3_validate.py` — 3-level sphere, `--devices`
  for multi-GPU, `--checkpoint/--resume`, `--lattice`, `--collision`
- `examples/suboff_shell_l3_validate.py` — 3-level SUBOFF, LES dispatch,
  wall function on L3 only
- `examples/amr_sphere_shell_validate.py` / `nested_shell` / `cellwise` /
  `shell_l2` — route 1/2/3 + combo variants

## Evidence

- `docs/evidence/amr-sphere-all-routes-r1.json` (10 runs, all GPU)
- `docs/evidence/amr-sphere-drag-validation-r1.json`

## Reproduce

```bash
cd /data/TensorLBM
PYTHONPATH=src .venv/bin/python examples/amr_sphere_shell_l3_validate.py \
  --device cuda:0 --nx 192 --ny 128 --nz 128 --radius 8 --reynolds 100 \
  --steps 8000 --warmup-steps 5000 --ramp-steps 500 \
  --shell-margin 16 --wake-cells 45 --l2-margin 8 --wall-margin 2 \
  --ghost-interpolation trilinear --collision cumulant \
  --output outputs/amr-sphere-l3-8k.json
```
