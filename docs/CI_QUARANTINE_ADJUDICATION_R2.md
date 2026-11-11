# CI quarantine adjudication R2 — the free-surface `i_to_g` ledger family

Anchor: **NEW: CI 隔离测试裁决台账**（`tests/ci_quarantine.txt` 现有 103 个 node id，conftest
把它们一律改写成 non-strict `xfail`，因此"隔离中的测试仍在真失败"这件事本身没有任何判决记录）。

EDIT_WHITELIST: `tools/`, `tests/`, `docs/`, `artifacts/`

## 1. Why this round exists

`tests/conftest.py` turns every quarantined node id into `pytest.mark.xfail`, so a
quarantined test that still fails is reported as **xfail**, not failure. The list has
grown to **103 node ids / 30 files** and nothing records whether an entry is a real
defect, a drifted fixture, or an obsolete test. `tools/quarantine_adjudicate.py` (new,
unit-tested in `tests/test_quarantine_adjudicate.py`) runs one node id per subprocess with
`--runxfail` (threads pinned to 1) and classifies the outcome:

| verdict | meaning |
|---|---|
| `PASS_NOW` | passes today → unquarantine candidate |
| `EMPTY_EVIDENCE` | assertion shows an empty evidence/diagnostic payload |
| `API_DRIFT` | moved/renamed interface (Import/Attribute/Key/Type/Name error) |
| `NUMERIC` | tolerance/magnitude assertion |
| `THRESHOLD` | count/ordering assertion |
| `ASSERT` | any other assertion |
| `ERROR` / `TIMEOUT` | collection error / exceeded per-test budget |

The tool only *reports*: it never edits `ci_quarantine.txt`. Promotion of a test is a
separate reviewed step, because a test that passes in isolation may still be
order-dependent under the full suite (that is exactly what batch 4 quarantines).

## 2. This round's scope (bounded honestly)

Adjudicated: **13 of the 16 free-surface node ids** (batch 2/4/5 entries from
`test_free_surface_conversion_density_audit`, `..._inventory_reconciliation`,
`..._i_to_g_exact_ledger`, `..._i_to_g_failure_policy`, `..._i_to_g_ownership_closure`,
`..._topology_mutation_replay_contract`). Ledger:
`artifacts/ci_quarantine/adjudication_free_surface_r1.json`, generated
2026-11-11T11:20:01Z, 114.94 s wall, machine-rendered table in
`docs/ci_quarantine_free_surface_r1.md` (zero hand-copied numbers).

**Not adjudicated (not_run):** the 3 `test_free_surface_topology_runtime_campaign.py`
entries (100-step float32 campaigns; the first sweep exceeded its 300 s per-test budget —
`TIMEOUT`, which the harness records rather than masking) and the other **87** node ids
(batch 4/5b/6 and the 35 `test_suboff_nested_static_amr_smoke` + 13
`test_regress_wall_multi_gpu` + 6 `test_multi_device_sdaa_common` entries, which need GPU /
SDAA / multi-rank hardware this host does not provide). Their disposition stays
**unadjudicated**, not passed.

Result of the 13: **0 `PASS_NOW`, 0 `API_DRIFT`, 2 `EMPTY_EVIDENCE`, 1 `NUMERIC`,
1 `THRESHOLD`, 9 `ASSERT`** — the family genuinely still fails, so nothing is promoted.

## 3. Root cause of the failing family

The five ledger tests all assert that `run_free_surface_closure_experiment()` case
`B_forced_conversion_deterministic` (step index 2) or `C_dam_break_style_tiny_dynamic_topology`
(step index 3) contains an **`I_TO_G`** conversion record, e.g.
`assert audit.i_to_g_count > 0` → `assert 0 > 0`.

The conversion rule in `src/tensorlbm/free_surface_lbm.py` (Körner step 5) is:

```
to_liq = interface & (fill > 1.0 + kappa)                      # kappa = 1e-2
to_gas = interface & (fill < -kappa)
to_gas |= interface & ~has_liquid_neighbour & ((fill < kappa_f) | ~has_interface_neighbour)
```

i.e. an INTERFACE cell only becomes GAS if it is **detached from LIQUID** and thin
(`kappa_f = 0.1`), or carries negative mass. In cases B/C every INTERFACE cell keeps a
LIQUID neighbour for all 3 resp. 10 steps, and the fill never goes negative, so the
`I_TO_G` branch is never entered. Measured event inventory over both cases' steps:
`I_TO_L` events occur only at **step index 0**, and `I_TO_G` count is **0** — see
`tests/test_free_surface_i_to_g_conversion_probe.py::test_closure_experiment_b_and_c_emit_no_i_to_g_conversion_events`.

So the family is **fixture drift**, not (as the batch note feared) an obviously broken
evidence path. The reachability proof is the second new test:
`tests/test_free_surface_i_to_g_conversion_probe.py::test_isolated_low_fill_interface_converts_to_gas_with_evidence`
builds a 6³ state with exactly one INTERFACE cell (`fill = 0.05`) and **no LIQUID cell
anywhere**, steps it, and gets:

* `flags[centre] == GAS`, `mass[centre] == 0.0`;
* exactly one `conversion_evidence` record with `flag_before = 2 (I)`, `flag_after = 0 (G)`;
* `build_conversion_density_audit(...)` → `i_to_g_count == 1`, `i_to_l_count == 0`,
  classification `I_TO_G`, status `DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE`.

That is the *first* passing, non-quarantined evidence that the I→G ledger path emits
per-cell records and that the cold audit classifies them correctly. It does **not**
validate I→G physics (no analytic solution for a one-cell state); it converts an
unanswerable claim ("the I→G ledger is unverified") into a decided one.

## 4. Two findings that are more than fixture drift

1. `test_free_surface_i_to_g_ownership_closure.py::
   test_opt_in_closure_experiment_reports_b_and_c_as_failed_diagnostics` fails with
   `assert 'WITHHELD' == 'FAILED_DIAGNOSTIC'`: the diagnostic vocabulary written by the
   solver (`WITHHELD…`) and the vocabulary the test expects
   (`FAILED_DIAGNOSTIC`) have drifted apart. This is a **contract mismatch**, not a
   numeric drift — one of the two names is stale and nothing else in the suite pins the
   vocabulary (all its siblings are quarantined). Recorded as a defect candidate; not
   fixed here because choosing the surviving name is an ownership decision for the
   free-surface lane, not a mechanical repair.
2. `test_free_surface_inventory_reconciliation.py` asserts `0.0 > 0.0`
   (`THRESHOLD`) and `0.0 != 0.0 ± 2e-7` (`NUMERIC`): the "pre-topology" and
   "conversion" attribution buckets are both exactly zero on the current engine, i.e. the
   attribution split the test was written for no longer exists. Same family root cause.

## 5. Disposition

* Keep all 13 quarantined (**no promotion**; the harness found no `PASS_NOW`).
* Follow-up A (owner: free-surface lane): re-derive the B/C fixtures so they actually
  detach an INTERFACE cell from LIQUID, then re-adjudicate the 5 ledger tests.
* Follow-up B: reconcile `WITHHELD*` vs `FAILED_DIAGNOSTIC` vocabulary and pin it with a
  non-quarantined contract test.
* Follow-up C: run `tools/quarantine_adjudicate.py` on the remaining 87 entries on a host
  with GPU/SDAA/multi-rank access; the tool is ready and its output is auditable.

## 6. Limitations

* 13/103 entries adjudicated; 90 are unknown, not passed.
* Verdicts are single-process (threads = 1) runs; batch-4 entries are quarantined
  precisely because they are order-dependent, so a `PASS_NOW` here would **not** be
  sufficient for promotion.
* Signatures are truncated to 240 chars in the ledger; full logs are not retained, so a
  verdict can be re-derived but not replayed from the artifact alone — rerun the harness.

## 7. Reproduce

```bash
python3 -m pytest tests/test_quarantine_adjudicate.py -q          # 11 passed
python3 -m pytest tests/test_free_surface_i_to_g_conversion_probe.py -q   # 2 passed
python3 tools/quarantine_adjudicate.py --file free_surface --timeout 90 \
  --out artifacts/ci_quarantine/adjudication_free_surface_r1.json \
  --markdown docs/ci_quarantine_free_surface_r1.md
```