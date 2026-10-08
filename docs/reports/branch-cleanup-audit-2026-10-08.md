# Branch cleanup audit — 2026-10-08

**Audit snapshot:** 2026-10-08 22:47 UTC  
**Repository:** `cxs503/TensorLBM`  
**Detailed inventory:** [branch-cleanup-audit-2026-10-08.csv](branch-cleanup-audit-2026-10-08.csv)

## Result

| Measure | Result |
|---|---:|
| Remote branches audited | 352 |
| Open PRs | 1 (#328) |
| Closed PRs in the inventory | 334 |
| A — verified safe to consider deleting | 0 |
| B — requires maintainer review | 332 |
| C — must retain | 20 |
| Deletions performed | 0 |
| Estimated cleanup now | 0 |

The `main` branch SHA at the snapshot was `07ee2f6eb8ea7a5212b5d3e60b90276cd12595bc`.
The `copilot/merge-branch-into-main` head is retained for open PR #328
(head SHA `c8faee69d9d5c2835a0797bcd7a6d42f378538ca`). No branch was deleted,
no PR was closed, and `main` was not modified by this audit.

## Method and findings

- Enumerated all 352 remote `refs/heads/*`; branch SHA and last committer timestamp
  are recorded in the CSV. Compared each ref with `origin/main` using
  `git rev-list --count origin/main..origin/<branch>`.
- Eighteen non-default branch tips have commits not reachable from `main` (159
  branch-relative ahead commits in total; counts overlap across branches). Those
  branches are C and must be preserved.
- The complete PR listing contained 335 extant PRs: #328 is open and the other
  334 are closed. All closed-list entries had `merged_at` populated. The bulk
  listing's `merged` field contradicted that metadata, so merged status is
  reported as “closed; merged_at populated” rather than treating the boolean as
  authoritative. Detailed PR reads spot-checked #10, #100, #200, #318, #329,
  #335, and #336 as merged. #328 is confirmed open and unmerged.
- Inspected the repository workflows: CI has unfiltered push and pull-request
  triggers; publish is tag-triggered (`v*`); Copilot setup is manually/path
  triggered. No per-branch allowlist was found in these workflow files.
- The branch listing reported `protected=false` for every branch. This does not
  rule out organization/repository rulesets or dependencies outside the visible
  workflow files.

## Classification

- **A (0):** None. The available evidence does not establish that any branch is
  free of maintainer, external automation, or unfinished-work dependencies.
- **B (332):** Branch tips have zero commits outside `main`, but still require
  owner confirmation that their branch names/refs are not used by external
  workflows or ongoing work. Recommendation: retain pending review.
- **C (20):** `main`; the 18 non-default branches with commits not reachable
  from `main`; and `copilot/cleanup-merged-branches`, which is the active branch
  for this audit. Recommendation: retain.

The CSV identifies PRs by head branch, and also records open PRs that target a
branch. For closed PRs, the merge-status caveat above applies.

## Pending confirmation

Before any cleanup, a repository administrator should confirm whether B branches
are referenced by repository/organization rulesets, deployments, external
automation, or active work not represented by an open PR. No executable cleanup
script or workflow was created in this audit phase. If cleanup is approved, the
next phase should produce and display a complete deletion list, then recheck
protection, open PRs, and commits against `main` for each branch immediately
before deletion; failed or uncertain checks must skip the branch.
