# Air lubrication (PAL/BDR) verification frontier

Module: `src/tensorlbm/air_lubrication_turbulent.py` (frontier layer on top of
`src/tensorlbm/air_lubrication.py`). Reproduce everything with

    python scripts/run_air_lubrication_frontier.py
    python -m pytest tests/test_air_lubrication.py -q

Artefacts land in `artifacts/air_lubrication/{result.json,gates_verified.json,figures}`.

## Measured limitation: 2D LBM does not sustain a turbulent boundary layer

`ChannelAirLubrication2D(nx=64, ny=32, u_mean=0.06, nu=1.4e-3)` with Smagorinsky
SGS, 5 % initial noise and constant-flow-rate control gives, after 12 000 steps,
`Re_tau = 66` and `C_f = 4.64e-3`, i.e. exactly the laminar plane-Poiseuille
value `12/Re_b`. 2D turbulence has no vortex stretching, so the near-wall cycle
cannot self-sustain. The turbulent carrier is therefore solved as a wall-normal
boundary-value problem in wall units (verified against the law of the wall and
Kármán-Schoenherr), and the lattice is used only where it is honest: laminar
stratified channel, interface stress continuity, species and momentum
conservation.

## What is claimed, and how strongly

| item | grade | evidence |
|---|---|---|
| turbulent carrier `C_f(Re_theta=1000..3000)` | REAL | vs Kármán-Schoenherr ≤ 1.6 %; log-law kappa 0.428 |
| `Delta C_f(alpha_w, delta_b/delta)` | REAL numerics + MODEL closure | viscosity stratification + documented bubble turbulence damping, reported separately |
| two-layer exact solution, interface momentum | REAL | closed-form, layer balances 0 to round-off |
| interface momentum / species mass conservation | REAL | ≤ 1.6e-16 / 1.8e-12 residual with mutation negative controls |
| bubble capture on a curved hull bottom | REAL | Lagrangian drift, capture → 0 for millimetre bubbles |
| `Q_air` optimum | REAL | compressor power vs saved power, interior maximum |
| point-by-point agreement with Madavan et al. (1985) | WITHHELD | no digitised table offline; only shape/integral checks |
| microbubble `Delta C_f(alpha_w)` vs PAL sandwich law within 20 % | **FAIL** | model 21-38 % vs law 60-92 % at `alpha_w = 0.3`; the law has no viscous-sublayer exclusion |
| thick air-layer `Delta C_f` vs PAL law within 20 % (`h/delta >= 0.2`, `Re_theta >= 2000`) | PASS | max 15.2 % error; the law is regime-limited, not wrong |

## Physical constraint that dominates the answer

Bubbles cannot occupy the viscous sublayer, so the void fraction is zero below
`y+ = 30`. Without that exclusion the mixture would replace sublayer fluid and
the drag reduction would be over-predicted by roughly a factor of three
(94 % instead of 29 % at `alpha_w = 0.3`). This is a physical constraint, not a
tuning knob, and it is the reason the model disagrees with the classical PAL
sandwich law while landing in the measured microbubble band (20-30 %).
