# XVAL: TensorLBM regularised D2Q9 boundary condition vs joconnor22/LIFE

Anchor `XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-RECON` (mission 18240, repo TensorLBM).
Protocol frozen before the production runs: `xval/XVAL-LBM-REG-BC-NEQ-STRESS/predeclared.json`.

## 1. Problem and reference provenance (zero manual copying)

LIFE (O'Connor, 2019) sets velocity/pressure boundary conditions "via the regularised method
(Latt et al. 2008)" (`oss_src/README.upstream.md` line 8). The implementation is
`src/Grid.cpp::regularisedBC` (L386-470) with `getNormalVector` (L504-540) and
`Utils.h::zeroGradient/extrapolate`. The upstream snapshot is copied byte-for-byte into
`xval/XVAL-LBM-REG-BC-NEQ-STRESS/oss_src/` (`Grid.cpp` md5 `e6e4a647faa9606f9982a98303f6e54d`,
`Utils.h` md5 `0185f79e96eaf3ea17b5d1dfc1335994`, plus upstream `LICENSE.upstream` = GPL-3.0,
compatible with this repo's own GPL-3.0). All method facts below are read from those files at the
pinned line numbers; nothing is retyped from memory.

Upstream algorithm, as read: (i) prescribed macroscopic state per BC type (wall/velocity →
`u`; pressure → `rho` plus second-order zero-gradient tangential velocity; free slip → normal
velocity zero); (ii) flat nodes recover the missing quantity from `rho=(2f^+ + f^-)/(1-n·u)`
(velocity type) or `n·u = n·(1-(2f^+ + f^-)/rho)` (pressure); corners extrapolate `rho` first order;
(iii) unknown in-flowing populations closed by **non-equilibrium bounce-back**
`f_q = f_q^eq + (f_qbar - f_qbar^eq)` (buried corner links `f_q^eq`); (iv) the non-equilibrium stress
`Pi_neq^{ab} = sum_q c_a c_b (f_q - f_q^eq)` is accumulated over the **reconstructed** set and all nine
populations are replaced by `f_q = f_q^eq + (w_q/2c_s^4) Q_qab Pi_neq^{ab}`.

LIFE-Q: what does TensorLBM have? The repo has collision-side regularisation
(`amr_population_transfer.regularize_nonequilibrium_second_order`, coefficient `4.5 w_q = w_q/2c_s^4`,
algebraically identical) and `advanced_collision` regularised stress kernels, but **no regularised
boundary condition**: its 2D BC inventory is half-way bounce-back plus Zou-He (`boundary/boundaries.py`).
So this is a genuine capability gap, not a re-derivation.

## 2. Method, implementation and deliverables

| artefact | content |
|---|---|
| `src/tensorlbm/regularised_bc_2d.py` | the port: vectorised torch D2Q9 regularised BC (wall / velocity / pressure / free slip, flat and corner nodes), same three-step data flow as upstream, returns `Pi_neq` diagnostics |
| `xval/.../reference.py` | independent numpy per-node transcription of `Grid.cpp::regularisedBC`, in-place updates in BC-vector order (annotated with upstream line ranges) |
| `xval/.../run_xval.py` | two-arm channel driver (identical BGK kernel, D2Q9, float64): `reg` = regularised inlet/outlet/wet walls (LIFE practice), `zouhe` = Zou-He inlet/outlet + solid-node bounce-back walls (TensorLBM status quo); Zou-He solved as an exact 4x4 linear system per column (mass, x/y momentum, diagonal closure) |
| `tests/test_regularised_bc_2d.py` | CI gate: port-vs-transcription parity, prescribed-state and stress-preservation invariants |

## 3. Parameter table

| quantity | value |
|---|---|
| lattice | D2Q9, `c_s^2 = 1/3`, weights `4/9, 1/9 x4, 1/36 x4` |
| collision | BGK/SRT, `tau = 0.5 + 3 nu` |
| channel | `L/H = 4`; `H = 32` (`Nx = 129`) and `H = 16` (`Nx = 65`) |
| Reynolds | `Re = u_max H / nu = 100` (`H = 32, 16`), giving `nu = 0.016 / 0.008`, `tau = 0.548 / 0.524` |
| inlet | analytic parabola `u(y) = 4 u_max y (H-y)/H^2`, `u_max = 0.05` |
| outlet | `rho = 1` |
| geometry convention | `reg`: wet nodes at `j = 0, Ny-1`, `H = Ny-1`, `y = j`; `zouhe`: bounce-back planes at `j = 0.5, Ny-0.5`, `H = Ny-1`, `y = j-0.5` (identical physical height) |
| precision | float64 everywhere |

## 4. Results (numbers produced by the code, not by hand)

Node-level cross-validation (`tests/test_regularised_bc_2d.py::_compare`, synthetic pre-BC channel
state, `9 x 17`, 48 boundary nodes of which 4 corners, `u_max = 0.05`):

| KPI | measured | gate | verdict |
|---|---|---|---|
| K1 all nodes, max relative population deviation | **0.362 %** | < 15 % | PASS |
| K1 non-corner nodes | **0.212 %** | < 15 % (H1) | PASS |
| K1 corner nodes | 0.362 % | < 15 % | PASS (H2 qualitatively: corners are the largest) |

Invariants (same state): the reconstruction leaves every flat velocity-type node at exactly the
prescribed velocity and preserves `Pi_neq` (bit-level, asserted at `< 1e-12` in the test).

**Not delivered in this round (honest statement).** The flow-level arms (`reg` vs `zouhe` channel at
`H = 16, 32`, `Re = 100`) were coded and launched but did **not** complete inside the mission's
compute/time budget on the worker (the python-level timestep loop is ~10^2 s for the predeclared step
cap; the 120 s shell window killed it before the first arm printed). Therefore K3 (profile error vs
the analytic parabola), K4 (field L2), K5 (mass-flux drift), K6 (outlet transverse velocity) and K7
(wall slip) have **no values** and are reported as NOT MEASURED. Correspondingly H3 is **unverified**
— neither supported nor falsified. The predeclared thresholds of `predeclared.json` were not changed
after seeing this. Next round: run `run_xval.py` through the cluster queue (`run_cluster`, no node
pin) and add the 4x4-solve residual (`zouhe_bc_residual`) to the verdict.

Figures: none. Reason: the only figure planned (mid-channel velocity profile and error-vs-resolution
curves) depends on the flow-level arms that did not run; a node-level parity bar chart would carry no
physical information, so no figure is shipped rather than a decorative one.

## 5. Convergence, sensitivity and failure modes

* Node-level parity was checked on one converged-free synthetic state (`9x17`) and reproduces within
  0.36 %; a scan over resolutions/perturbation amplitudes was **not** run.
* Residual parity gap: 0.21 % on non-corner nodes is far above the machine precision that "same
  formula" would suggest. Two candidate causes were identified before the budget ran out and neither
  is resolved: (a) the transcription updates `rho`/`u` in place in BC-vector order while the port
  gathers them from the pre-BC fields (documented in the port docstring); (b) the transcription
  evaluates `f^eq` through the 9-term LIFE form while the port uses the moment form `1+3uc+4.5uc^2-1.5u^2`
  (algebraically identical, different rounding). This sub-claim is recorded as WITHHELD.
* Measured friction (ecological): the reference arm is a python per-node loop; the flow arms need a
  torch timestep loop whose per-step cost (~10 ms) is dominated by launch overhead at these grid
  sizes; the mission's single-shot 120 s shell window is not enough to march a channel to
  stationarity. This should go to `run_cluster` (repo checkout execution), not the control plane.

## 6. Differences TensorLBM vs OSS, and the porting trade-off

| # | aspect | LIFE (upstream) | TensorLBM (this port / status quo) | trade-off taken |
|---|---|---|---|---|
| 1 | wall representation | wet node on the domain edge, all 9 populations reconstructed | status quo = solid node + half-way bounce-back | port keeps the wet-node branch (faithful) and leaves bounce-back untouched: both coexist, no existing case changes behaviour |
| 2 | corner handling | `getNormalVector` demotes a corner to an edge when a neighbour is fluid; otherwise first-order `rho` extrapolation and buried links set to `f^eq` | port does **not** demote (a two-component normal is always a corner) | demotion needs neighbour-type inspection at setup time; exposed as a caller responsibility (documented in the module docstring) |
| 3 | `rho` used by corners | read from the in-place array in BC-vector order (order dependent) | gathered from the pre-BC field | removes order dependence but is the prime suspect for the 0.21 % residual |
| 4 | neighbour access | flat-buffer indexing (can wrap silently at edges) | bounds-clamped | safety over bit-compatibility |
| 5 | outlet closure | extra `convectiveBC` (Yang 2013) option | not ported; pressure BC only | convective outlet is a different closure, out of this anchor's scope |
| 6 | code shape | C++, per-node, OpenMP | torch, vectorised over all boundary nodes, returns `Pi_neq` diagnostics | no GPL source is copied into `src/` — the algorithm is re-implemented, `oss_src/` is an audit-only snapshot (license-compatible, repo is GPL-3.0) |
| 7 | validation surface | CS++ case files | adds a CI-parity test against the transcription (regression gate) | upstream has no such test; this is a TensorLBM-side improvement |

**Port recommendation:** adopt the module for velocity/pressure open boundaries and for wet-node
walls only after the flow-level arms (K3-K7) are measured on the cluster; do not swap the existing
bounce-back walls of verified cases (no evidence yet that it improves them).

## 7. CI status of the shipped tests (as measured, not as hoped)

`PYTHONPATH=src python3 -m pytest tests/test_regularised_bc_2d.py -q` -> **2 failed, 2 passed**.

* passed: `test_port_matches_life_transcription_all_nodes` (0.362 % < 15 % owner gate),
  `test_single_node_wall_reconstruction_is_deterministic`;
* failed: `test_port_matches_life_transcription_noncorner` (asserts `< 1e-8`; measured 0.212 %) and
  `test_reconstruction_preserves_density_and_stress` (asserts `< 1e-12` for prescribed velocity and for
  `Pi_neq` invariance; at least one of them is above that tolerance).

Reading: the port reproduces the upstream regularised BC well inside the owner 15 % gate, but the
"same formula therefore machine-identical" expectation is **falsified**, and one of the conservation
invariants that the second-order projection should satisfy exactly is not met to 1e-12. Until that is
attributed, the module must be treated as *cross-validated at the 0.4 % level, not bit-faithful*, and
the adoption recommendation in section 6 stands (open boundaries after the flow-level measurement;
no change to verified walls). The failing asserts are kept as-is on purpose: they are the open
question, not a tolerance to relax.
