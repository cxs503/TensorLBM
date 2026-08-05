# I→G Strict-Failure Cold Campaign Policy R1

## Boundary

`tensorlbm.free_surface_i_to_g_failure_policy` is a standalone cold campaign
adapter. It is deliberately not imported from `tensorlbm.__init__`,
`free_surface_lbm`, `hull_free_surface`, or `dam_break`; no default solver or
production hull caller gains fallback behavior.

The policy module does not import or call `free_surface_step`. A campaign caller
provides a strict step callback, an optional explicit legacy callback, and a
deep snapshot/equality/fingerprint triple for its complete state tuple. This
keeps numerical execution, the exact float32 I→G gate, and policy reporting
separate.

## Policy contract

`IToGStrictFailurePolicy.RAISE` is the default. All policies, including
`RAISE`, require callable `snapshot_state`, `states_equal`, and
`fingerprint_state`. A strict callback runs only on an independent snapshot.
On `TopologyTransactionError`, the wrapper first verifies the original
committed prestate and two pre-attempt snapshots against the stored immutable
fingerprint, then immediately propagates that same exception unchanged. It
does not construct diagnostics or require replay evidence on the `RAISE` path.

`STOP_AND_REPORT` stops immediately. The strict callback receives only a
detached proposal snapshot, and the retained committed state plus both saved
snapshots are exact-compared to the independent immutable baseline and its
pre-attempt fingerprint before reporting. `states_equal` must return the
literal `bool`; `fingerprint_state` must return a stable immutable whole-state
value (for TensorLBM the test adapter uses dtype/shape/raw tensor bytes). This
detects shallow wrapper snapshots that share tensors, without intentionally
mutating caller state. It returns the last successfully committed state,
with `committed_steps` and `attempted_steps` stored separately. The rejected
step never becomes committed. Its `IToGFailureDiagnostic` requires
`STRICT_FAILURE_REPLAYED_EXACT` from the existing strict replay and embeds the
existing `audit_failed_i_to_g_residual` result; unavailable/unreplayable
failure evidence fails closed rather than producing a STOP report.

`SKIP_EXPERIMENTAL_PROPOSAL` requires all three explicit inputs:

1. `policy=SKIP_EXPERIMENTAL_PROPOSAL`;
2. `allow_experimental_fallback=True`;
3. `legacy_step`, `snapshot_state`, `states_equal`, and `fingerprint_state`
   callbacks.

Before strict invocation, the wrapper creates independent committed and
immutable-baseline snapshots and stores the original fingerprint. Every adapter
snapshot is immediately `copy.deepcopy`-isolated before it is exposed to either
callback. Therefore a new shallow wrapper that shares a list, tensor, or other
mutable child cannot give either callback a path back to the retained campaign
state. States that cannot be deeply copied fail closed before callback entry.

On strict failure it verifies that the rejected attempt did not alter the
original prestate or either snapshot, then calls `legacy_step` once on a fresh
deep-isolated snapshot of the committed snapshot. Before publishing any legacy
candidate—and also when legacy raises—it revalidates the original prestate,
committed snapshot, immutable baseline, and fallback-input fingerprints. Any
change raises `RuntimeError`; no report is returned. A successful legacy
candidate is itself deep-copied before becoming the next campaign state, so the
callback cannot retain an alias to the published campaign state. It never uses
a partial strict candidate. The report is always `status=WITHHELD`,
`fallback_status=FALLBACK_NOT_PHYSICAL`, and
`physical_closure_claim=False`. It is a legacy continuation experiment, not a
physical closure result.

### Generic isolation limit

This is a Python callback boundary, not a sandbox. The policy requires that the
entire state graph support faithful `copy.deepcopy`; otherwise it refuses to
invoke the callback. This protects ordinary nested Python containers and
PyTorch tensors from shallow-wrapper aliasing. A custom object may implement a
malicious or defective `__deepcopy__`, hold external process/global mutable
state, or mutate data reachable outside the declared state graph. No generic
in-process policy can prove or undo those effects. Such adapters are outside
this contract and must be replaced with a trustworthy deep-copyable state
representation (or the fallback must remain unavailable).

The ledger explicitly contains:

- `attempted_steps`: every strict proposal attempt;
- `committed_steps`: only state transitions retained by the campaign;
- `fallback_steps`: committed legacy restarts after a rejected strict proposal.

A campaign may meet additional strict failures after a legacy continuation;
those are independently attempted and independently marked as fallback steps.
No rejected state is reused or silently promoted.

## Evidence and claim limits

The failure diagnostic uses the pre-existing trusted strict failure capture.
For the real B/C step-3 rejection this exposes:

- `STRICT_FAILURE_REPLAYED_EXACT` for the original rejection;
- `WITHHELD_NOT_REPRESENTABLE` for the detached residual screen;
- no topology candidate or physical closure claim.

The adapter does not relax any gate, correct global mass, change float32
arithmetic, or integrate a residual proposal into the solver.

## Verified R1 contracts

`tests/test_free_surface_i_to_g_failure_policy.py` covers:

- synthetic strict RAISE without evidence: original exception identity is
  preserved and a mutating callback cannot alter input state;
- shallow tensor-sharing snapshots fail closed through fingerprint mutation;
- shallow wrapper/list/tensor legacy mutation is run only against a wrapper
  deep-copy; fallback-input mutation is detected after both normal return and
  callback exception, while original/prestate snapshots remain unchanged;
- missing isolation callbacks are rejected for every policy;
- STOP_AND_REPORT with two committed steps then failure evidence;
- explicit fallback restart from an equal prestate and
  `FALLBACK_NOT_PHYSICAL` marking;
- real B/C strict paths: two commits then step-3 failure;
- real B/C legacy continuations: `WITHHELD` plus one or more explicitly marked
  fallback steps, never a physical closure result;
- no module import from package defaults and no hull/dam-break/solver import by
  the cold policy module.

Run:

```text
python -m py_compile src/tensorlbm/free_surface_i_to_g_failure_policy.py
pytest -q tests/test_free_surface_i_to_g_failure_policy.py \
  tests/test_free_surface_topology_mutation_replay_contract.py \
  tests/test_free_surface_i_to_g_ownership_closure.py
```
