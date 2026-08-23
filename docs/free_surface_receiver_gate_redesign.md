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
> route** (B remains the fallback). The original recommendation is kept
> below for the record.

**D first, A as the destination.** (superseded by §7) Run the batch-16 reproduction with G
removed and H1+H2 kept: if the halo stays bounded (§4.4), land D and treat
the `to_gas` booking as A-work scheduled behind the existing opt-in flag;
if it explodes, D is dead and A is the only structurally sound route —
schedule the closure promotion with a version note. B is kept as a fallback
if A's strict closure cannot be made local/exact in reasonable time.

## 6. Decisions needed from the owner

1. Route: ~~D-first~~ (executed, failed §4.1 — §7) → **A now the only
   candidate**; B fallback still allowed?
2. If A: is promoting `enable_i_to_g_ownership_closure` to default an
   acceptable breaking change, and in which minor version?
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
