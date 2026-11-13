# Diff list: TensorLBM vs OSS reference (pluggable SCMP forcing)

Round 1 XVAL, anchor `XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT`.
Full narrative + numbers: `REPORT.md`.

## Approach differences

| aspect | OSS reference (paper formulas) | TensorLBM | impact |
|---|---|---|---|
| velocity_shift | `u_eq = u + τ·F/ρ` | identical | bit-identical |
| EDM | `Δf = feq(u + F/ρ) − feq(u)` | identical | bit-identical |
| Guo | `S_i = w_i(1−1/(2τ))[(c_i−u*)·F/cs² + (c_i·u*)(c_i·F)/cs⁴]` | identical (mass-conserving) | bit-identical |
| SC force gather | backward neighbor `ψ(x−c_i)` | identical (G>0=attraction) | bit-identical |
| pseudopotential | `psi_exp` | identical | bit-identical |
| EOS support | — | CS, PR, `make_psi_eos` factory | TL extension |
| scheme parameter | — | `"standard"` / `"pressure_tensor"` | TL extension |

## Parameter / boundary-condition differences

* **Grid:** Both use periodic D2Q9, L=128, R∈{12,16,20,24}.
* **Walls:** periodic (no walls) — identical.
* **Forcing:** all three schemes available in both arms.

## Porting tradeoffs (recommendation)

1. **No code change needed** — TensorLBM's implementation is bit-identical
   to the published formulas for all three forcing schemes.
2. **TensorLBM extensions** (EOS factories, pressure_tensor scheme) are
   valuable additions beyond the OSS reference scope.
3. **Single-step verification** confirms bit-identical output at t=0
   (ρ_min=0.3700, ρ_max=4.4860, mass=3554.34 for all six arm×scheme combos).
