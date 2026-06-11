"""TensorLBM Platform test package.

These tests exercise the FastAPI B/S platform end-to-end with the
``starlette.testclient`` (``fastapi.testclient.TestClient``).  They are
fully self-contained: no network access is required and every simulation
job is configured with very small grids and very few time steps so that
the full suite finishes in a few minutes on a CPU-only machine.

Run from the repository root with::

    pytest platform/tests -q

or from the ``platform`` directory with::

    pytest tests -q
"""
