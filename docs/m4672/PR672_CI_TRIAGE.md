# PR #672 / branch `agent/m4654` — CI triage and recovery record

Scope: mission m4672 ("fix red PR #672, round 1"). This note records what the red
actually was, what was changed, and how the claim was verified. It is deliberately
narrow: the PR under repair adds exactly one documentation file, so no solver code
was touched.

## 1. What the red was

Head of the PR at triage time: `47c6c2fc222a44dc0b4c8d2176972342feaabac8`
(message `m4654: matured mission deliverables（6 个文件被门禁滤除）`).

The failing list reported by the merge pipeline is 14 check-runs, i.e. **two** full
7-job workflow runs on the same commit:

| run id | event | created (UTC) | finished (UTC) | conclusion |
|---|---|---|---|---|
| 42145081432 | `push` | 2026-11-06T01:37:36Z | 2026-11-06T02:04:42Z | cancelled |
| 42145626670 | `pull_request` | 2026-11-06T01:42:36Z | 2026-11-06T02:04:39Z | cancelled |

Inspection of the jobs API shows that **every job in both runs ended `cancelled`
with `steps: []` (zero steps executed, no runner ever assigned)**, e.g.

```
lint            completed cancelled  runner=(none)  steps=0
repository-layout completed cancelled runner=(none) steps=0
test (3.11)     completed cancelled  runner=(none)  steps=0
test (3.12)     completed cancelled  runner=(none)  steps=0
platform-test   completed cancelled  runner=(none)  steps=0
architecture-audit completed cancelled runner=(none) steps=0
package-build   completed cancelled  runner=(none)  steps=0
```

So the red is **not** a lint / test / layout verdict — no gate ever produced one.
The two runs sat queued for 22–27 min and were then cancelled inside the same
~10 s window (02:04:28–02:06:04Z) together with 18 other TensorLBM branch runs that
had been queued since 01:04Z, while `main`'s push run 42145612061 stayed queued
with no runner assigned. That signature (mass simultaneous cancellation of queued
jobs, hosts never assigned) is a runner-capacity / queue-eviction event on the
account side, not a property of this branch.

## 2. Consequence for the merge gate, and the only in-repo fix available

A `cancelled` conclusion is sticky per commit SHA: as long as the PR head stays
`47c6c2fc`, the seven check names keep reading `cancelled` and the repository RED
rule (`failing-names ⊆ main-failing-names`) cannot fire, because `cancelled ≠
test (3.11)`. The only in-repo action that can clear this state is **a new head
commit**, which forces GitHub to create a fresh, un-cancelled run.

That is exactly what this mission delivers: the triage document plus a new commit
on the branch, re-triggering CI. No workflow, solver or test file was modified —
there was no executed gate to repair, and editing CI config to work around a
capacity outage would change behaviour for every other branch in the repository.

Pre-existing context (not caused by this PR, not touched here): the last completed
`main` runs (41995371431 @ 866438ac, 42008133677 @ 866438ac) fail only the two
`test (3.11)` / `test (3.12)` jobs. Those names are in `main`-failing set, so a
fresh run of this PR whose only red is those two names satisfies the
"adds no new red" pass-through rule.

## 3. Local reproduction of every gate the workflow runs

Because CI executed nothing, the workflow's own steps were run locally on the
current `main` tip plus this PR's content (`1ef76442` + worktree):

```
### repository-layout            Repository layout OK (5137 candidate paths; historical exceptions retained).
### tests/test_repository_layout.py                 Ran 12 tests ... OK
### tests/test_repository_report_migration.py      Ran  6 tests ... OK
### tests/test_audit_import_graph.py               Ran  5 tests ... OK
### scripts/check_platform_docs_consistency.py     [docs-consistency] OK
### pytest -k "layout or migration or provenance or docs_consistency or repository"
    43 passed, 2 skipped, 81 subtests passed in 44.48s
```

`lint` and `mypy core gate` are unaffected by construction: the diff contains no
`.py` file (`git diff --name-status <main>..HEAD` → one added `.md`).

## 4. Honest limitations

* The **full** fast suite (`pytest -m "not slow"`, 6794 tests) was started locally
  twice but the runner of this environment terminated the session at ~30 % of
  progress (out-of-tool process reaping); no complete-suite verdict is claimed
  here. The bounded subset above is the evidence actually obtained. A complete
  suite verdict will come from the re-triggered CI run itself.
* Whether the re-triggered run gets a host is outside this repository's control;
  if it is cancelled again, the cause is the same capacity outage and the record
  above is the artefact to point at.
* No numerical result, field plot or error bar is applicable: the diff is
  documentation only, so section ⑥ of the report standard ("key quantity vs
  baseline error") is declared not applicable rather than silently omitted.