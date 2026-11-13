"""Example registry for TensorLBM.

Central catalogue of all examples.  New examples register themselves by
calling ``register()`` — typically at module import time.

Usage::

    from tensorlbm.examples import registry
    registry.list_examples()          # → ['lid_driven_cavity', 'poiseuille', ...]
    registry.get('lid_driven_cavity') # → Example instance
    registry.run('lid_driven_cavity', nx=64, re=400)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from .base import Example

_registry: Dict[str, Example] = {}


def register(example: Example) -> None:
    """Register an example instance.

    Parameters
    ----------
    example : Example
        The example to register.  ``example.name`` must be unique.
    """
    if example.name in _registry:
        raise ValueError(f"example '{example.name}' already registered")
    _registry[example.name] = example


def get(name: str) -> Optional[Example]:
    """Look up an example by name."""
    return _registry.get(name)


def list_examples() -> List[str]:
    """Return sorted list of registered example names."""
    return sorted(_registry.keys())


def run(name: str, **overrides) -> "ExampleResult":
    """Run an example by name with optional parameter overrides.

    Parameters
    ----------
    name : str
        Registered example name.
    **overrides : dict
        Key-value pairs that override the example's default parameters.

    Returns
    -------
    ExampleResult
    """
    example = get(name)
    if example is None:
        raise KeyError(f"unknown example '{name}'; available: {list_examples()}")
    params = example.configure()
    params.update(overrides)
    return example.run(params)


def run_all(**global_overrides) -> Dict[str, "ExampleResult"]:
    """Run every registered example.

    Parameters
    ----------
    **global_overrides : dict
        Applied to every example (e.g. ``device='cpu'``, ``n_steps=10``).

    Returns
    -------
    dict[name, ExampleResult]
    """
    results = {}
    for name in list_examples():
        results[name] = run(name, **global_overrides)
    return results
