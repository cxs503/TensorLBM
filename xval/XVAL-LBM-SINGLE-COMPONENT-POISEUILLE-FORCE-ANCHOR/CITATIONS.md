# OSS source citations (zero hand-copy)

Source: `ProjectPhysX/FluidX3D`, branch `master`, fetched 2026-11-13 through the GitHub
contents API (raw.githubusercontent.com is blocked in this sandbox), default build
configuration in `src/defines.hpp` (`#define SRT` at line 10, `D2Q9` and `TRT` commented out,
`FORCE_FIELD` commented, `VOLUME_FORCE` required for the volume-force block).

| file | bytes | sha256 |
|---|---|---|
| src/kernel.cpp | 203478 | 66fd44a63bfccd6d78881000162b99fe0295b851f89a5383825097bc5c7fb016 |
| src/lbm.cpp | 83381 | 12ee951dc8f19420c72ff79c8d0a29d1024fee6cf30ad5fdf13420a182dac623 |
| src/defines.hpp | 7226 | 8a2e1641e0280eca193a612f447d6955f271ce1c95bfd856ed65df8dd72df863 |

Every rule transplanted into `poiseuille_kernels.py` cites its line range:

- `kernel.cpp:1124-1135` `calculate_forcing_terms` — `uF = -1/3 (u.F)`,
  `Fin[i] = 9 w_i [(c_i.F)(c_i.u + 1/3) + uF]`, i.e. `= w_i[3(c_i-u).F + 9(c_i.u)(c_i.F)]`.
- `kernel.cpp:1589-1598` VOLUME_FORCE block — `rho2 = 0.5/rhon`, `u += F*rho2` clamped to
  `±def_c`, then the forcing terms and `f_eq` are built from the shifted velocity.
- `kernel.cpp:1640-1648` SRT — `c_tau = fma(w,-0.5,1)` (`= 1-1/(2tau)`), `Fin *= c_tau`,
  `f_post = (1-w) f + w f_eq + Fin`.
- `kernel.cpp:1036-1059` `calculate_f_eq` — second-order equilibrium in DDF-shifted storage.
- `kernel.cpp:1094-1106` `calculate_rho_u` — `rho = 1 + sum(f)`, `rho u = sum(c f)`.
- `kernel.cpp:1360-1373` `load_f`/`store_f` — Esoteric-Pull (`j = t%2 ? i : (i%2? i+1 : i-1)`),
  solid cells never store (`kernel.cpp:1911-1913`), which completes the halfway bounce-back
  implicitly.
- `kernel.cpp:1615` — `w = 1/(3 nu + 1/2)`, i.e. `nu = (1/w - 1/2)/3`.
- `lbm.cpp:207-232` — `update_force_field` / `object_force` (momentum exchange on `TYPE_S`),
  the OSS force-diagnostic path; not needed for the volume-force anchor.

Only short quotations for verification are reproduced above (no OSS source file is vendored
into this repository); the anchor re-implements the cited rules in `poiseuille_kernels.py`.
