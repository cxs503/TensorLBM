"""Pipeline end-to-end smoke marker for mission m4630.

Deliberately minimal and dependency-free: it carries a single immutable
constant so the test suite can prove that an agent-generated file round-trips
through the branch/PR/review pipeline unchanged.  It lives under ``tests/``
because the repository layout policy (``docs/architecture/repository_layout_policy.json``)
does not permit new files at the repository root.
"""

PIPELINE_SMOKE_M4630 = "4630"