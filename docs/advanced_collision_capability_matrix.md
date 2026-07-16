# 3-D advanced collision capability matrix

`tensorlbm.advanced_collision_contract` is the lattice-neutral public contract
for advanced 3-D collision selection.  A cell marked **available** has an
executable kernel exposed by `collide_advanced_3d`; a similarly named routine
elsewhere is not enough.

| Lattice | MRT | CM / cascaded | entropic KBC |
|---|---|---|---|
| D3Q19 | **available**: `solver3d.collide_mrt3d`, 19×19 explicit transform | **WITHHELD_NO_D3Q19_CM_KERNEL**: no standalone validated kernel | **WITHHELD_NO_D3Q19_KBC_KERNEL**: no standalone entropy-solved kernel |
| D3Q27 | **available**: `d3q27.collide_mrt27`, full-rank 27×27 Gram–Schmidt transform/inverse | **WITHHELD_NO_D3Q27_CM_KERNEL**: `advanced_collision.collide_cascaded_d3q27` is only second-order regularized reconstruction; it explicitly does not compute/relax higher central moments | **WITHHELD_NO_D3Q27_KBC_KERNEL**: `advanced_collision.collide_kbc_d3q27` uses caller-selected blending and does not solve the entropy condition |

## Executable interface

```python
from tensorlbm import collide_advanced_3d, collision_capability_matrix

f_post = collide_advanced_3d("D3Q27", "MRT", f, tau=0.8, s_e=1.19)
matrix = collision_capability_matrix()
```

`MRT` is currently executable on both lattices. `CM` (also alias `cascaded`)
and `KBC` (also alias `entropic_kbc`) intentionally raise
`CollisionKernelWithheldError` with the listed `WITHHELD_*` reason until a
validated full implementation is added.  The dispatcher checks the population
direction dimension (`19` or `27`), four-dimensional population shape, and
`tau > 0.5`; it never routes D3Q27 requests through a D3Q19 kernel.
