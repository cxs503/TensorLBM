# Maintained workflow scripts

Use this directory for reusable simulation, data preparation, visualization and
developer workflow entry points. Repository-only static tools belong in `tools/`;
one-off research belongs in `experiments/<topic>/`.

New or substantially changed scripts should document purpose, dependencies,
invocation, inputs, output directory and hardware needs. Provide `--help` when
applicable, use configurable paths, and avoid `/root/...` or private checkout paths.
Running a script successfully does not mean its physical result is accepted.

Existing scripts are not all certified by this convention. Validate a script before
advertising it as maintained; do not bulk rename legacy diagnostics. Follow the
[development workflow](../docs/development_workflow.md) and
[layout policy](../docs/architecture/repository_layout.md).
