# Security Policy

## Supported Versions

TensorLBM is under active development. Security fixes are applied to the latest
release line and to `main`. Older tagged releases are not guaranteed to receive
back-ports.

| Version | Supported |
|---------|-----------|
| `main`  | ✅        |
| latest released tag | ✅ |
| older tags | ❌ |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, report privately via GitHub's
[private vulnerability reporting](https://github.com/cxs503/TensorLBM/security/advisories/new)
or email the maintainer listed in `pyproject.toml` (`authors`).

Include:

- a description of the vulnerability and its impact,
- steps to reproduce (preferably a minimal script),
- affected version(s) / commit,
- any suggested mitigation.

We aim to acknowledge reports within **5 business days** and to provide a
remediation plan within **30 days**, depending on severity.

## Scope Notes

- TensorLBM ships research-grade numerical code. Model weights, datasets, and
  run artifacts are distributed via Release assets / Hugging Face Hub / Git LFS,
  never as executable code in-tree.
- The `app/` FastAPI platform and `frontend-vue/` are separate deployable
  components; their own dependency and secret-handling policies apply at
  deployment time. Do not commit secrets, tokens, or credentials to the repo.
- The solver library (`src/tensorlbm`) must not depend on application, web, or
  agent-orchestration code; report any reverse-dependency as a structural
  vulnerability.
