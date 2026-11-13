# XVAL — Inamuro wet-node / anti-bounce-back vs the OSS reference

**Anchor**: `XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE`
**Implementation under test**: `src/tensorlbm/boundary/inamuro_wetnode.py`
**OSS reference (zero hand-copying)**: `lbm-principles-practice/code` @
`6e2c592fdc3592c14dfd52f860fc1ceea930bcb0` — `chapter8/film_inamuro.cpp`
(executed binary), `chapter5/couette_wetnode.m`, `chapter5/poiseuille_wetnode.m`
(source read for the wall rules). MIT licence; code of Krüger, Kusumaatmaja,
Kuzmin, Shardt, Silva & Viggen, *The Lattice Boltzmann Method: Principles and
Practice*, Springer 2017.

## Reproduce

```bash
# 1. fetch + build + run the reference binary (~110 s at N=50000)
benchmarks/verified/inamuro_wetnode_xval/reproduce_oss.sh /tmp/oss_lbm

# 2. TensorLBM side + comparison + figures
PYTHONPATH=src python3 benchmarks/verified/inamuro_wetnode_xval/run.py \
    --part film --steps 50000 --devsteps 5000 \
    --oss-ref /tmp/oss_lbm --out artifacts/m17603

# 3. unit/property tests of the operator
PYTHONPATH=src python3 -m pytest tests/test_inamuro_wetnode.py -q
```

## What is compared

| part | ref | got | metric |
|---|---|---|---|
| film (scalar Dirichlet + zero-gradient + corner average) | concentration field written by the upstream C++ binary at steps 100/1000/10000/50000 | `inamuro_wetnode_scalar` / `inamuro_zero_gradient` / `inamuro_corner_average` driven through the same problem | relative L2 and Linf, field-to-field |
| velocity (Dirichlet) | analytic Couette | `inamuro_wetnode_velocity` | **not_run** — see `docs/XVAL_INAMURO_WETNODE_M17603.md` (open driver defect); operator exactness is covered by the tests |

Acceptance: relative L2 < 15 % (owner precision standard).

Artifacts: `artifacts/m17603/xval_inamuro_wetnode.json`,
`film_fields.npz`, `film_ref_<step>.dat`, `fig1_film_field_xval.png`.
