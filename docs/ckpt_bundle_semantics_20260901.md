# ckpt_bundle API — semantic friction: options and recommendations (2026-09-01)

- Status: **design / decision document** — prepares owner decisions; changes
  no code. File:line refs are `src/tensorlbm/ai/…` at e84a528f; numbers come
  from `rehearsal.json` (opened 2026-09-01) unless attributed otherwise.
- Provenance: the API friction register of the 2026-08-31 production rehearsal
  (PR [#269](https://github.com/cxs503/TensorLBM/pull/269) comment; machine
  copy `friction` list in
  `/nfs/wangxi/runs/ckpt_bundle_rehearsal_20260831/rehearsal.json`), semantic
  items only. The 4 mechanical items are closed by the #269 follow-up da0299c5
  + #271 (merged, main e84a528f) and are referenced, not re-proposed.

## 0. Register provenance and numbering

The rehearsal register carries 12 entries: 4 mechanical (closed), 7 semantic
(open — this document), 1 positive note. The PR-comment condensation numbered
the semantic entries 5–9, folding three concerns into item 9:

| here | PR #269 comment | rehearsal.json `friction` |
|---|---|---|
| S5 | item 5 | `[4]` |
| S6 | item 6 | `[5]` |
| S7 | item 7 | `[6]` |
| S8 | item 8 | `[7]` |
| S9a / S9b / S9c | item 9 (three clauses) | `[8]` / `[9]` / `[10]` |
| §8 fragments | — (not in the PR comment) | `[3]` norm validation, `[2]` from_pool |

## 1. S5 — `predict` serves ONE shared field/sdf for all N cond rows

**Friction (as logged).** "`predict` accepts exactly ONE reference field and
ONE sdf for all N cond rows; a per-row-geometry corpus must call predict per
row (batch-of-1 latents) or switch to `predict_batch` with counts=1 — the
batch-composition/latency tradeoff is invisible from the signature."

**Current.** `ckpt_bundle.py:682` — `predict(self, fields, sdf, cond)`: the
shared sdf is coerced once and expanded N-fold (`:718`), fields likewise
(`:717`); `predict_batch` (`ckpt_bundle.py:734`) is the per-row path via
`counts` grouping (`:776-778`). The tradeoff is now *documented* (docstring
`:699-706`, closed as mechanical item 4) but stays invisible at the signature
level. Measured size (`rehearsal.json`): per-row batch-1 latents vs the
shared-design batch reach `max_rel` 1.81e-5 (ts2) / 1.59e-5 (ts4) over 250
rows (238 / 201 rows > 1e-6); one query costs ~19.7 ms median (ts2 19.682 ms,
ts4 19.673 ms, n = 30) while the 381-row x 10-member batch runs in 0.601 s /
0.593 s.

**Options.** **A** keep-as-is — the docstring paragraph is the contract
(compat: none) · **B** additive — explicit alias `predict_shared_design(fields,
sdf, cond)` calling `predict`, making the assumption readable at the call site
(compat: none, new name) · **C** breaking — per-row sdfs behind a flag
(compat: high; risks the bit-exact path for no measured need).

**Recommendation: A.** The mechanical-item-4 docstring already carries the
numbers and the rule "compose batches of a shared design rather than looping
per row"; `predict_batch` covers per-row. Take B only if a third call pattern
appears. Migration: none.

## 2. S6 — silent 2-arg -> 3-arg `predict` widening; no capability probe

**Friction (as logged).** "`predict(fields, sdf, cond)` silently widens the
base `ModelEnsembleBackend.predict(fields, cond)`; a polymorphic caller
passing the old 2-arg shape gets a TypeError, not a helpful error, and there
is no runtime capability probe (isinstance check needed) to distinguish
backends."

**Current.** Base signature `inference_service.py:541`; override
`ckpt_bundle.py:682` with the deliberate-widening comment `:680-681`. The
hazard is concrete: `DragSurrogateService.predict` dispatches on
`isinstance(self.backend, ModelEnsembleBackend)` (`inference_service.py:1394`)
then calls the 2-arg form `self.backend.predict(field, cond)`
(`inference_service.py:1405`); `PerMemberEnsembleBackend` *is* a
`ModelEnsembleBackend` (`ckpt_bundle.py:515`), so it passes the gate and dies
with `TypeError: missing 1 required positional argument: 'cond'` (`cond`
already consumed as `sdf`). The class docstring flags it (`:538-540`: "not a
drop-in for `DragSurrogateService`"). Only probe today: the `kind` string
("model" vs "per-member-model", `inference_service.py:535-536` /
`ckpt_bundle.py:648-650`).

**Options.** **A** keep — `kind` + docstring; misuse stays a confusing
TypeError (compat: none) · **B** additive — capability property (e.g.
`needs_sdf`, `False` on the base, `True` here) plus a named guard in
`DragSurrogateService.predict` (compat: none; converts an existing crash into
an error naming the fix) · **C** breaking — required keyword `sdf=` on the
override (compat: breaks every current per-member caller).

**Recommendation: B.** Cheap, additive, matches the repo convention that
error messages name the remediation; the guard only fires on code that would
have crashed anyway. Migration: none for callers; two lines in the service.

## 3. S7 — 4-D sdf ambiguity: `(C,D,H,W)` unbatched vs `(G,D,H,W)` batched

**Friction (as logged).** "`_sdf_tensor` accepts 4-D with TWO meanings
(`(C,D,H,W)` unbatched vs `(G,D,H,W)` batched, disambiguated only by the
internal `batched=` flag); `(G,C,D,H,W)` and `(G,1,C,D,H,W)` are both accepted
batched — spent time in source to confirm `(381,32,32,64)` was legal for
`predict_batch`."

**Current.** `ckpt_bundle.py:657-678` (`_sdf_tensor`): unbatched ndim==4 is
read as channels (`:672-673`), batched ndim==4 as geometries (`:660-662`),
batched ndim==6 `(G,1,C,D,H,W)` is squeezed (`:663-664`). The coerced channel
count is never checked against the encoder `in_ch` on the unbatched path
(wrong shapes fail later inside the conv); the batched path checks G against
the field batch (`:774-775`). For the production pool both readings coincide
(`encoder.in_ch = 1`, `rehearsal.json` `arch_ts2`/`arch_ts4`) — why the
ambiguity never bit the rehearsal.

**Options.** **A** keep — rely on `in_ch=1` + docstring shapes (`:748-749`)
(compat: none) · **B** additive — validate the coerced channel axis against
encoder `in_ch` in `_sdf_tensor` and document both 4-D readings (compat:
well-formed inputs unchanged; malformed inputs error earlier) · **C**
breaking — accept only 5-D forms (compat: breaks the documented `(D,H,W)` /
`(G,D,H,W)` call patterns).

**Recommendation: B.** Closes the ambiguity where it can actually disagree
(C != in_ch) at near-zero cost, without removing any accepted shape.
Migration: none (new validation only raises on inputs that would have failed
downstream anyway).

## 4. S8 — mixed `LoadedMember` + tuple lists rejected

**Friction (as logged).** "with raw `(enc, body)` tuples requires `norms=`
but a MIXED list (some `LoadedMember`, some tuples) rejects the bundled
sidecars of the `LoadedMember` half (`all_bundled` flips to False globally) —
per-member norm fallback is not attempted."

**Current.** `ckpt_bundle.py:556-575`: `all_bundled` is one flag flipped by
any tuple member (`:565-566`); `norms=None` then raises even though every
`LoadedMember` carries a complete sidecar (`:571-575`). Explicit `norms`
override the bundled stats wholesale — a `Mapping` broadcasts to all members
(`:577-578`), a sequence must match the member count and replaces the
sidecars (`:580-582`). A mixed ensemble cannot use bundled norms for one half
and explicit norms for the other.

**Options.** **A** keep — the error names the rule; mixed callers pass full
`norms` (compat: none) · **B** additive — per-member fallback: bundled
sidecar for `LoadedMember` members, explicit block otherwise; raise only for
a tuple member lacking one; explicit `norms` keep today's override semantics
(compat: none — today-mixed lists raised or duplicated norms; after B they
just work) · **C** breaking — only `LoadedMember` members (compat: removes
the hand-built-pair path the tests exercise).

**Recommendation: B.** Pure additive fallback; deletes a dead-end error
without changing any value the backend produces for existing inputs.
Migration: none required; mixed-list callers can drop duplicated `norms`.

## 5. S9a — `meta["member"]` magic key

**Friction (as logged).** "`member_labels()` keys off `meta['member']`;
`save_member_bundle` gives no schema hint for meta, so default labels degrade
to m0..m9 unless the caller knows the magic key."

**Current.** `ckpt_bundle.py:652-655` reads `m.get("member", f"m{i}")` — the
same convention as the single-model format (`inference_service.py:538-539`).
`save_member_bundle` copies `meta` verbatim (`:385`) and its docstring
(`:354-367`) never mentions the reserved key; bare-pair loads start from
`meta = {}` (`:497`) and therefore label `m0..m9`. Explicit escape hatch
exists: `labels=` (`:549`, `:631`).

**Options.** **A** keep — `labels=` is the path (compat: none) · **B**
additive docs — document the reserved `member` key in the
`save_member_bundle` docstring and the module bundle-format section (compat:
none) · **C** breaking — default `meta` to `{"member": <stem>}` at save time
(compat: changes the bytes of every re-written bundle for a cosmetic win).

**Recommendation: B.** The convention is shared with the base format and
already has an explicit override; it only needs to be discoverable.
Migration: none (docstring-only).

## 6. S9b — wrapper-vs-trunk pair choice

**Friction (as logged).** "`LoadedMember` bundles the pair as (stage1 =
`SupervisedSDFEncoder` wrapper, model = `TwoStageCondFNODrag`) and the
ensemble stores `(item.stage1, item.model)`; the constructor ALSO accepts a
bare trunk — which encoder object to pass is only answerable from source (the
identity check `body.encoder is trunk` silently protects, but you only know
to care after reading it)."

**Current.** `LoadedMember` carries `stage1` (wrapper) + `model`
(`ckpt_bundle.py:407-408`); the constructor resolves it to
`(item.stage1, item.model)` (`:559`) but accepts either the wrapper or the
bare trunk on the tuple path — `:601` — guarded by the identity check
`:602-607`. At serving the encoder slot is used only for checks and `.eval()`
(`:589-607`, `:627-628`): `predict` drives `self._models` alone (`:721`) and
the latent is recomputed inside `TwoStageCondFNODrag.forward`
(`sdf_two_stage.py:219`), because `_build_member_pair` constructs the body
around the wrapper's trunk object (`ckpt_bundle.py:266-267`).

**Options.** **A** keep — both forms work, the identity check protects
(compat: none) · **B** additive docs — name both accepted forms in the class
docstring, state that the served latent always comes from `body.encoder`
(the slot is a consistency check); optionally store the unwrapped trunk
internally (compat: none) · **C** breaking — trunk only (compat: contradicts
the `LoadedMember` layout itself — the primary path).

**Recommendation: B (docs-first).** The dual acceptance costs nothing at
runtime and C contradicts the primary path; the fix is making the contract
readable. Migration: none.

## 7. S9c — cond-column semantics out-of-band

**Friction (as logged).** "cond column semantics (param columns = FIRST
n_cols of the corpus cond matrix: `[log10 re, log10 uin(, log10 sail, log10
fin)]`; latent rides after) is out-of-band knowledge documented in the sanity
driver, not in the API docstrings."

**Current.** The docstrings pin the *shape* contract — `p_mean`/`p_std` cover
the param columns only, latent columns ride raw (`ckpt_bundle.py:61-65`,
`:364-366`, `:528-531`, enforced `:612-620`; `[p | z]` concatenation happens
inside the body forward, `sdf_two_stage.py:210`/`:222`) — but never name the
physical columns. Those live in `drag_cond.py:197-203` (`COND_V3_CHANNEL_NAMES`
= `log10_re`, `log10_u_in`, `log10_sail_scale`, `log10_fin_scale` + geometry
block); the arms take a different prefix width (ts2 `param_dim` 2, ts4 4,
`rehearsal.json` `arch_ts2`/`arch_ts4`), and nothing in the bundle records
which prefix a member expects beyond the `param_dim` integer.

**Options.** **A** keep — `param_dim` + the sanity-driver notes carry the
meaning (compat: none) · **B** additive docs + optional provenance key — a
docstring paragraph naming the expected prefix (first 2 / first 4 corpus cond
columns, per `COND_V3_CHANNEL_NAMES`) and, optionally, a `meta["cond_columns"]`
sidecar key written by callers, not enforced by loaders (compat: none) ·
**C** breaking — require `cond` as a named mapping (compat: high; fights the
numpy boundary every call).

**Recommendation: B.** The knowledge already exists in-repo (`drag_cond.py`);
the gap is one docstring paragraph plus an optional, non-enforced key.
Migration: none (docs; `meta["cond_columns"]` is opt-in at write time).

## 8. Register fragments outside the PR-comment list (open, not covered by #271)

Two rehearsal-register clauses map to neither the PR-comment mechanical items
nor #271 and remain present at e84a528f:

- **Save-time norm validation asymmetric** (`friction[3]`):
  `save_member_bundle` checks key *presence* only (`ckpt_bundle.py:373-375`);
  the `p_mean`/`p_std` width check lives in `PerMemberEnsembleBackend.__init__`
  (`:612-620`), so a padded 34-column norm saves fine and only fails at
  ensemble build. **A** keep / **B** additive save-time width check against
  the (already inferred/validated) arch `param_dim`. **Recommendation B** —
  the arch block is in hand at save time; the error moves to the writer and
  no valid bundle changes.
- **No pool-migration helper** (`friction[2]`, second clause): no
  `from_pool(dir)` glob helper; the shard-tag convention was hand-rolled in
  the rehearsal driver. **A** keep (the 20-bundle migration is done) / **B**
  additive helper. **Recommendation A** — no consumer today; revisit if a
  second legacy pool is migrated.

## 9. Cross-reference: 2026-08-30 serving engineering gaps

Three of the four serving engineering gaps recorded on 2026-08-30 are closed
by #269 (merged): bare-checkpoint sidecar (`ckpt_bundle` format v1), the
stage-2 loader documentation (`.fno` remap contract in the module docstring),
and per-seed (encoder, body) pairs (`PerMemberEnsembleBackend`). The fourth —
new-geometry reference field + SDF generator (= L2) — remains owner-gated.

## 10. Summary

| item | friction (short) | recommendation | compat cost | change |
|---|---|---|---|---|
| S5 | one shared field/sdf for all N rows; tradeoff invisible in signature | **A keep** (docs closed by #271) | none | none |
| S6 | silent 2-arg -> 3-arg widening; no capability probe | **B additive** `needs_sdf` + named service guard | none | small |
| S7 | 4-D sdf = channels (predict) vs geometries (predict_batch) | **B additive** `in_ch` check in `_sdf_tensor` | none (errors earlier) | small |
| S8 | mixed `LoadedMember` + tuple lists reject bundled norms | **B additive** per-member norm fallback | none | small |
| S9a | `meta["member"]` magic key undocumented | **B docs** reserved-key docstring | none | trivial |
| S9b | wrapper vs trunk in the pair slot | **B docs** (+ optional normalise) | none | trivial |
| S9c | cond-column identity out-of-band | **B docs** + optional `meta["cond_columns"]` | none | trivial |
| §8a | save-time norm width not validated | **B additive** width check in `save_member_bundle` | none (errors earlier) | trivial |
| §8b | no `from_pool` helper | **A keep** (no consumer today) | none | none |

All recommendations are additive or documentation-only; none touches the
bit-exact serving arithmetic (`predict` / `predict_batch` bodies stay
untouched) or the on-disk bundle format (version stays 1).

## 11. Decisions requested from owner

1. **S5** — accept keep-as-is (A), or ask for an explicit
   `predict_shared_design` alias (B)?
2. **S6** — approve the additive capability probe `needs_sdf` + named guard
   in `DragSurrogateService.predict` (B), or keep the TypeError (A)?
3. **S7** — approve the additive `in_ch` validation in `_sdf_tensor` (B), or
   leave shapes docstring-only (A)?
4. **S8** — approve per-member norm fallback for mixed lists (B), or keep the
   all-or-nothing rule (A)?
5. **S9a–S9c** — approve the three documentation patches (B each); for S9c
   also decide whether the optional `meta["cond_columns"]` key should be
   written at the next pool re-bundle.
6. **§8a** — approve the save-time norm-width check (B)?
7. **§8b** — confirm no `from_pool` helper is wanted now (A).
8. **Sequencing** — none of the above blocks the current v6qx serving stack;
   S6–S8 + §8a fit one small additive PR, S9a/S9b/S9c + the S7 docstring
   rider are a docs-only PR. Confirm that split (or merge them).
