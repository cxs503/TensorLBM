"""TensorLBM examples framework.

Replaces the "example-per-main" antipattern (100+ standalone scripts with
duplicated argparse/main boilerplate) with a structured, testable, and
composable example system.

Quick start::

    from tensorlbm.examples import registry
    registry.list_examples()
    registry.run('lid_driven_cavity', nx=64, re=400)

CLI::

    python -m tensorlbm.examples.runner --list
    python -m tensorlbm.examples.runner lid_driven_cavity --nx 64
"""

from .base import Example, ExampleResult
from . import registry

# Import examples to trigger auto-registration
from . import lid_driven_cavity  # noqa: F401

__all__ = [
    "Example",
    "ExampleResult",
    "registry",
]
