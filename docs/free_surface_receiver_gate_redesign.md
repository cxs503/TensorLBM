# Decision doc: free-surface exchange receiver gate (post-456fdb1 redesign)

**Status: PROPOSAL — needs owner decision. No solver code changes until one of
the routes below is picked.** Owner: hydro line (free-surface).

## 1. History: the two failure modes

456fdb1 (2026-08-20) fixed a real mass-source bug and introduced a real
dynamics regression. It bundled three changes:

| Part | Change | Purpose |
|---|---|---|
| G (gate) | exchange receivers require `fill > 1e-3` (`free_surface_lbm.py`, step 4) | kill the "fill≈0 halo gets pumped with mass, then `to_gas` misbooks/destroys it" chain (batch-16 root cause) |
| H1 | `to_i` only adjacent to LIQUID | end interface-layer self-propagation (halo explosion feeder) |
| H2 | `to_liq` reinitialises f at rho_liquid equilibrium | no ABB-inflated densities |

**Before 456fdb1** the dam break collapsed *vigorously but explosively*:
front 10 → 31, interface envelope 272 → 5232 cells in 400 steps — the halo
explosion G was written to kill.

**After 456fdb1** the dam break is *frozen*: front, liquid/interface counts
and conversions are identical at every step, `topology_changed` is always
`False` (probed to g = 1e-2, t* ≈ 18; probes in `/nfs/wangxi/tmp/probe_fs_*.py`
on the 5090). This is a genuine physics regression, not a test bug: the
static-column and ledger tests all pass; what fails is that nothing moves.

## 2. Pathology

`init_flags_from_fill` promotes zero-fill gas cells adjacent to liquid to
INTERFACE — **the interface envelope is born at fill = 0**. Under gate G those
cells can never receive exchange mass, so their fill can never grow, so the
liquid→interface→gas mass pathway is sealed at birth. G blocks the *receiver*
end of the chain, but the actual mass destruction 456fdb1 chased happens at
the *`to_gas` conversion* end (the misbooked mass of an emptying interface).
Blocking reception to protect against bad booking also blocks all legitimate
transport: the gate is at the wrong end of the causal chain.

## 3. Candidate routes

### Route D (surgical revert, cheapest first probe): drop G, keep H1+H2

Un-gate the receivers; keep H1 and H2. The static-column zero-drift result
in 456fdb1's message was verified with all three parts active; H2 (killing
ABB-inflated densities) and H1 (ending self-propagation) plausibly carry
most of the halo suppression on their own.

- **Must re-verify**: the batch-16 halo-explosion reproduction case — if the
  envelope still explodes without G, D is insufficient and we fall through
  to A.
- Risk: low blast radius; the ledger machinery already catches accounting
  breaks loudly.
- Cost: one reproduction campaign + test updates.

### Route A (structurally correct long-term fix): move the block to `to_gas`

Keep receivers open; fix the actual destruction point. The opt-in
`enable_i_to_g_ownership_closure` machinery (step 5a,
`build_i_to_g_ownership_transaction`) is exactly a strict local closure for
I→G mass ownership: an emptying interface hands its real remaining mass to
its neighbourhood as a booked transaction instead of letting the conversion
destroy it.

- Making it the default changes the legacy-path promise ("bit-for-bit
  identical when disabled") — a versioning decision, not a code detail.
- Needs: strict-closure verification battery (already has replay/capture
  scaffolding), plus a bound on halo growth (§4, K).
- Cost: highest; also the only route that fixes the disease rather than a
  symptom.

### Route B (initial-condition semantic fix): envelope born at fill = ε

Give newly promoted envelope cells a small initial fill (geometric
reconstruction or a fixed ε) so receivers are eligible from birth; keep G.

- Cost: injects initial mass (~ε·N_env cells) that must be ledger-booked at
  init; static-column tests must tolerate the bookkeeping entry.
- Risk: masks rather than fixes the booking problem; halo explosion may
  partially return through the same pumped-halo chain.

**Not offered**: tuning the 1e-3 threshold (a κ on the same wrong end);
suppressing `to_gas` for young cells (breaks emptying semantics).

## 4. Acceptance criteria (any route)

1. Static column, 200 steps, 4 configs: drift +0.0000%, ledger exact
   (unchanged from 456fdb1).
2. Dam break (g = 1e-2): non-frozen topology — front moves, conversions
   occur, `topology_changed` observed; the current `xfail(strict=True)`
   fence in `test_dam_break_3d_free_surface_campaign.py` XPASSes and is then
   removed in the same PR.
3. Global mass ledger conserved to ≤ 1e-10 relative over the run.
4. Halo bound: interface count stays ≤ K× its initial value for the
   batch-16 reproduction case — **owner to set K** (pre-456fdb1 data point:
   19× in 400 steps, which was the bug; a healthy bound is plausibly ~2–3×).
5. Full free-surface test surface green (ledger / i_to_g / inventory /
   ownership series).

## 5. Recommendation

> **UPDATE 2026-08-23 — Route D was executed and failed acceptance §4.1;
> see §7.** The fall-through applies: **A is now the only candidate
> route** (B remains the fallback).
>
> **UPDATE 2026-08-24 — Route A was executed and failed all five §4
> criteria; see §8.** The closure machinery as shipped cannot take a
> single timestep on a fill=0-born envelope. **B (envelope born at
> epsilon > 0) is now the only untried route** and deserves first
> examination, because it dissolves the birth-synchrony that defeats
> both D and A. The original recommendation is kept below for the
> record.
>
> **UPDATE 2026-08-24 (later) — Route B was executed and failed; see
> §9.** A dense epsilon sweep over [0, 0.05] shows **no epsilon lets the
> closure execute even one transaction**: below the mechanical boundary
> 0.02375 the step-1 candidate is still WITHHELD, above it no cell ever
> becomes a donor so the closure is never invoked. **All three routes
> D/A/B are now executed and dead; the only live line is the closure
> redesign (A-prime) of §6.2.**

**D first, A as the destination.** (superseded by §7) Run the batch-16 reproduction with G
removed and H1+H2 kept: if the halo stays bounded (§4.4), land D and treat
the `to_gas` booking as A-work scheduled behind the existing opt-in flag;
if it explodes, D is dead and A is the only structurally sound route —
schedule the closure promotion with a version note. B is kept as a fallback
if A's strict closure cannot be made local/exact in reasonable time.

## 6. Decisions needed from the owner

1. Route: ~~D-first~~ (executed, failed §4.1 — §7) → ~~A now the only
   candidate~~ (executed, failed all of §4 — §8) → ~~B first~~
   (executed, failed all of §4 — §9) → **only the closure redesign
   (A-prime) remains**; confirm it as the work item.
2. ~~If A: is promoting `enable_i_to_g_ownership_closure` to default an
   acceptable breaking change?~~ Informed by §8/§9: the closure needs a
   redesign before any versioning question exists — legal-receiver set
   must include recv_new promotion or the conversion wave must be
   de-synchronised, the exchange end needs a fill-approx-0 guard, a
   negative-donor policy is required (donor mass is already net -1.94
   before booking), and the all-or-nothing withholding policy needs a
   partial-commit answer (§9: even the best epsilon leaves 16-32 donors
   with legal receivers and still withholds).
3. Halo bound K for §4.4.
4. Should the batch-16 reproduction become a permanent regression test
   (adds runtime to the free-surface suite)?

## 7. Evidence: Route D executed (2026-08-23)

Three arms, identical harness (`runs/fs_route_d_20260823/run_arm.py`,
data alongside; dam-break mirrors `probe_fs_front.py`, which reproduces
every reference number the doc quotes — front 10→31, envelope 272→5232,
K≈19× — so it *is* the batch-16 case):

- **Arm 0** = main @ 7d2e253d (G + H1 + H2) — baseline;
- **Arm D** = G removed only (one hunk: `recv_ok = iface_mask`;
  `route_d.diff`) — H1 + H2 kept;
- **Arm P** = G + H1 + H2 all removed (the three 456fdb1 hunks
  reverse-applied on current main — the pre-456fdb1 semantics).

| measurement | Arm 0 | Arm D | Arm P |
|---|---|---|---|
| static column, 4 configs, 200 steps | +0.0000% drift, ledger exact, all 4 | **+35.2 … +82.5% drift, ledger broken, all 4** (liquid 1008→1824) | +75…+87%, ledger broken |
| dam break front (400 steps) | 10 → 10 (frozen) | **10 → 16 (peak 17) — unfrozen** | 10 → 31 |
| first `topology_changed` step | never | **step 7** (24 events) | step 2 |
| interface envelope peak / K_meas | 1.0× | 560 / **2.06×** | 5280 / **19.4×** |
| i→g conversions (total) | 0 | **0** | > 0 |

Answers to §3's questions:

1. **D unfreezes the dam break** — front moves, topology events from
   step 7, conversions occur. The frozen-column regression is indeed G.
2. **The halo bound passes vacuously**: K = 2.06 sits inside the 2–3×
   target, but only because the pumped halo cells are converted to
   LIQUID (H1 suppresses the count) while their mass enters the liquid
   inventory — a count-only K is **not** a sufficient acceptance
   criterion; §4.3 (ledger) must gate it.
3. **Static-column mass conservation fails catastrophically** the
   moment G is dropped, with H1+H2 still active. G was carrying real
   conservation load; the disease is exactly where §2 put it — the
   conversion/booking end — and blocking receivers was the only thing
   standing between the fill≈0 mass source and the ledger.

Independent re-check (this session, separate 15-line harness, default
`rho_gas=1.0`): arm D drifts +19.7% (mass 1008→1206) where main stays at
exactly +0.0000% — the magnitude is config-dependent, the sign and
mechanism are robust.

Test surface: Arm 0 = 246 passed / 20 xfailed (clean); Arm D = 3 failed,
all mass-accounting fail-closed gates in the dam-break campaign; the
`xfail(strict)` fence does **not** XPASS under D — the caller aborts at
step 7 on the mass gate before reaching the fenced assertion, so the
fence neither lifts nor fires.

Reproducibility gaps (recorded, none qualitative): the "static column 4
configs" in 456fdb1's message (iface 743, mass 3312) exist nowhere in
the repo — substitute configs used, all four reproduce Arm 0 exactly;
`git show 456fdb1^:file` would also revert later clean-ups, hence the
hunk-reverse Arm P.

**Verdict: Route D fails acceptance §4.1/§4.3. Per §3's own fall-through,
the redesign goes to Route A** (B stays the fallback). The four owner
decisions in §6 stand, with item 1 resolved by this measurement.

## 8. Evidence: Route A executed (2026-08-23/24)

Four arms, harness `runs/fs_route_a_20260823/run_arm.py` (Route D harness
+ `--closure` kwarg + `TopologyTransactionError` capture + per-step i->g
donor-mass bookkeeping); Arm P reuses the Route D pre-456fdb1 data:

| measurement | Arm 0 (G on, closure off) | Arm A (G off, closure on) | Arm C (G on, closure on) | Arm P |
|---|---|---|---|---|
| static 4 configs, 200 steps | +0.0000% drift, 4/4 | **WITHHELD at step 1, 0 steps run** | **WITHHELD at step 1, 0 steps run** | +75.1..+86.8% drift |
| dam-break g=1e-2, 400 steps | front 10->10 frozen | WITHHELD at step 1 | WITHHELD at step 1 | front 10->31, K=19.4 |
| i->g conversions | 0 | 0 (aborts first) | 0 (aborts first) | 7088 |

Every closure-enabled run — all 4 static configs and the dam-break, in
*both* gate configurations — aborts on the first `free_surface_step` with
`TopologyTransactionError: WITHHELD: ... I->G donor has no legal INTERFACE
receiver` (raise site `free_surface_topology_transaction.py:342`). The
"0.0000% drift / ledger exact" static rows for A and C are vacuous:
`steps_run == 0`.

**Step-1 forensics** (builder spy, `probe_withheld.py`): the envelope is
born at `fill = 0` (272 interface cells, interface mass 0.0), so
`to_gas = (I|L) & (fill <= 0.01)` flags **all 272 interface cells as
donors on the very first step**, regardless of gate G. The closure's
legal receiver set `I & ~to_gas & ~to_liq` is then empty, every donor has
zero legal receivers, and the all-or-nothing policy withholds the entire
candidate. 272/272 donors *do* have plain INTERFACE neighbours — the
failure is **synchrony, not isolation**, which is why Arm C (gate G
untouched) fails identically to Arm A.

**The exchange end is also diseased**: at the failing step-1 candidate
the donor mass is already net **-1.9433** (per-cell min -1.375e-2) —
i.e. the mass-exchange step drives empty envelope cells negative *before*
any booking, and the legacy path books +1.94 back on the conversion end
(1006.06 -> 1008.0). Arm D's +35..+82% drift is the residue of these two
mutually-cancelling errors once G stops sealing. Section 2 located the
pathology at the to_gas booking end only; this is the other half.

Independent re-check (2026-08-24, separate harness): envelope 272 cells
born fill min=max=0; 272/272 cells match the to_gas predicate at step 1;
closure OFF runs 2 steps with mass 1008.0000 -> 1008.0000 exact; closure
ON raises the WITHHELD error at step 1.

Reproducibility gaps (recorded, none qualitative): the closure flag is a
plain bool kwarg with no transaction-recorder precondition (`ownership_
ledger`/`runtime_ledger` are optional diagnostics; the repo's own
`run_free_surface_closure_experiment` is self-declared
DIAGNOSTIC_NOT_PHYSICAL_CLOSURE); the static 4 configs are the Route D
substitutes (456fdb1's originals do not exist in the repo); Arm P was not
re-run.

**Verdict: Route A fails acceptance §4 on all five criteria — it cannot
take a single timestep.** Per §3's fall-through the remaining routes were
**B (envelope born at epsilon > 0)** — which dissolves the birth-synchrony
mechanism that defeats both D and A — and a redesigned closure (receiver
set including recv_new promotion / de-synchronised conversion wave /
fill-approx-0 guard at the exchange end / negative-donor policy) before
any versioning question in §6.2 can exist. *(B was executed the same day
and also failed; see §9.)*

## 9. Evidence: Route B executed (2026-08-24)

Patch: `init_flags_from_fill(..., envelope_fill_epsilon=0.0)` — the single
promotion site splits out an `envelope` mask and sets `fill[envelope] = eps`
in place (+14/-2, one function; `runs/fs_route_b_20260824/route_b.diff`).
Mass is booked once by the caller (`init_mass_from_fill` at eps-rho), no
double entry. Default eps=0 is the bit-exact legacy path: the patched
worktree (exp/fs-route-b @ a3bf9038) passes **246 / 20 xfailed**, identical
to clean main. Runtime birth paths were audited and already never born at
zero (`to_iface` requires fill > 0.01, `recv_new` is born from redistributed
positive mass), so init is the only zero-born site.

**Lead question — does eps>0 survive step 1?** No. eps = 1e-6 and 1e-3
both abort with the same WITHHELD as Route A, same signature (272 donors /
0 legal receivers / 0 donors with legal neighbours). Cause is mechanical:
`recv_ok = iface_mask & (fill > 1.0e-3)` (`free_surface_lbm.py:872` in the
run baseline) is a *strict* comparison and float32(1e-3) rounds to
9.9999997e-4, so a 1e-3-born envelope is still G-excluded — dynamically
identical to Arm C.

**Full epsilon sweep (0 to 0.05, 22 points, `probe_step1_boundary.json`)**
— no epsilon executes a single closure transaction:

| epsilon range | step-1 outcome | donors | legal receivers |
|---|---|---|---|
| 0 — 2e-3 | WITHHELD | 272 | 0 |
| 5e-3 — 2.35e-2 | WITHHELD | 256 → 112 | 16 → 160 |
| ≥ 2.4e-2 | "ok" — closure never invoked | 0 | — |

The survival boundary is exactly mechanical: **0.01 (the `to_gas` line,
`free_surface_lbm.py:1014`) + 1.375e-2 (max per-cell step-1 exchange debit)
= 0.02375**; measured WITHHELD at 0.0235, ok at 0.024. In the middle band
the surviving receivers are spatially *anticorrelated* with the donors
(outflow cells empty into donors, inflow cells survive), so at best 16-32
donors have a legal neighbour and the all-or-nothing policy still withholds
the whole candidate — the block is **structural, not a fill-value problem**.

**§4 battery on the surviving epsilon (2.4e-2 / 5e-2, closure on):**
1. static 4 configs: 3 steps then step-4 WITHHELD, already +6.2 … +11.1% drift;
2. dam break: front frozen 11 → 11, zero conversions, `topology_changed` never;
3. ledger: 1014.53 → 1070.76 (+5.54e-2 rel) in 3 steps;
4. halo bound K: vacuous (<= 3 steps);
5. test surface: unchanged (246 / 20 xfailed).

**The pump chain re-ignites the moment G stops binding.** At eps = 2.4e-2
the step-1 exchange books the familiar -1.9433 liquid delta, then step 2
flips to **+21.4869 per step** with `exchange_interface_delta = 0`
throughout — exactly the unpaired interface credit the G gate exists to
kill (Route A gap 2, §8). umax reaches 3.7e11 on step 2 (the Route D
transient signature).

**Diagnostic arms (closure off, eps alone on the legacy path):** eps = 1e-3
injects 0.272 of mass that step-1 `to_gas` misbooking consumes *exactly*
(total back to 1008.000, bit-exact), after which the column stays frozen;
eps = 2.4e-2 does unfreeze (first change step 4, g_to_i = 176, front → 12,
K = 1.41) but drifts +4.1 … +34.1% — Route D's disease at lower amplitude.

Independent re-check (2026-08-24, separate scripts): boundary sweep
reproduced dense; arm 0 reproduces the Route A/D arm 0 field-for-field
(front frozen at 10 for 500 steps, `topology_changed` never); pump
arithmetic 1012.58 → 1034.07 = +21.487/step and 1070.7559/1014.5283 =
+5.54e-2 both re-derived from the raw JSON.

Reproducibility notes: harness `runs/fs_route_b_20260824/run_arm.py`
(Route A harness + `--epsilon`), step-1 spy `probe_step1.py`; worktree
restored clean @ a3bf9038 after diff archival; no pushes, no commits.

**Verdict: Route B fails acceptance §4 on all five criteria and comes off
the fallback list.** The epsilon sweep proves the fill value is not the
lever: below the boundary the closure withholds, above it the closure is
dead code. A geometric-reconstruction epsilon (per-cell Koerner fill) only
reshapes the epsilon landscape — it cannot fix the donor/receiver
anticorrelation and still needs the exchange-end paired booking. With
D, A and B all executed and dead, the only live line is the §6.2 closure
redesign (receiver set with recv_new promotion / de-synchronised
conversion wave / fill-approx-0 exchange guard / negative-donor policy /
partial-commit answer to all-or-nothing).
