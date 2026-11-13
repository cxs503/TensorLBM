"""CLI entry point: ``python -m tensorlbm.xval`` runs the cross-validation benchmark."""

from tensorlbm.xval.xval_benchmark import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
