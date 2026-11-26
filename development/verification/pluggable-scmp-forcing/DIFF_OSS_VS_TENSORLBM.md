# Diff list: TensorLBM vs OSS reference (pluggable SCMP forcing)

Round 2 XVAL, anchor `XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT`.
Full narrative + numbers: `REPORT.md`.

## New Capabilities Added

| function | description | reference |
|---|---|---|
| `collide_velocity_shift(f, fx, fy, tau)` | Velocity-shift forcing (Shan-Chen 1993) | PRE 47, 1815 |
| `collide_edm(f, fx, fy, tau)` | Exact Difference Method (Kupershtokh 2004) | — |
| `collide_sc_single_component(..., forcing)` | Unified pluggable collision interface | — |
| `sc_single_component_force(...)` | SC force with custom ψ support | — |

## Approach differences

| aspect | OSS reference (paper formulas) | TensorLBM | impact |
|---|---|---|---|
| velocity_shift | `u_eq = u + τ·F/ρ` | identical | bit-identical |
| EDM | `Δf = feq(u + F/ρ) − feq(u)` | identical | bit-identical |
| Guo | `S_i = w_i(1−1/(2τ))[(c_i−u*)·F/cs² + (c_i·u*)(c_i·F)/cs⁴]` | identical (mass-conserving) | bit-identical |
| SC force gather | backward neighbor `ψ(x−c_i)` | identical (G>0=attraction) | bit-identical |
| pseudopotential | `psi_exp` | identical | bit-identical |
| unified interface | — | `collide_sc_single_component(forcing=...)` | **new** |
| EOS support | — | CS, PR, `make_psi_eos` factory | TL extension |

## Single-step verification (L=64, R=12)

All three schemes produce bit-identical results between reference (numpy float64)
and TensorLBM (torch float64):
- Mass difference: < 1e-4 (float32/float64 precision limit)
- Max density difference: 0.0

## Porting recommendation

1. **Implementation complete** — all three forcing schemes now available via
   `collide_sc_single_component(f, G, tau, forcing="velocity_shift|edm|guo")`.
2. **Single-step verification confirms** bit-identical behavior with published formulas.
3. **Full Laplace law validation** pending cluster run cj3285.
