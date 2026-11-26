# Staged-CI gate study

- [Published study](../../../docs/XVAL-GITLAB-CI-STAGED-GATES.md)
- [Maintained design](../../../docs/ci/gitlab_ci_staged_gates.md)
- [Historical summary](summary.json)
- [Preregistered design](predeclared.json)
- [Upstream manifest](oss_refs/manifest.json)
- [Active staged-gate test](../../../tests/test_ci_staged_gates_xval.py)

From the root of a separate checkout, run the analysis:

```sh
PYTHONPATH=. python -m development.verification.gitlab_ci_staged_gates.run_xval
```

The analysis reuses the committed OSS references; it may rewrite the local
manifest and summary. The manifest paths identify the relocated copies of the same upstream
bytes; the migration ledger records the original manifest. Existing gate semantics, job policy, and physical acceptance rules
remain in their respective maintained configuration and code.
