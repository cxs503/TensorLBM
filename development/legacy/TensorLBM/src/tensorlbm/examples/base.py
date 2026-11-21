"""Base class for TensorLBM examples.

Every example inherits from :class:`Example` and implements:
- ``configure()`` → returns a dict of parameters (used for CLI defaults and programmatic calls)
- ``run(params)`` → executes the simulation and returns an ``ExampleResult``

This replaces the "example-per-main" antipattern (100+ standalone scripts with
duplicated argparse/main boilerplate) with a single registry + unified runner.
"""

from __future__ import annotations

import abc
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


@dataclasses.dataclass
class ExampleResult:
    """Outcome of running an example."""
    name: str
    params: Dict[str, Any]
    elapsed_s: float
    success: bool
    artifacts: Dict[str, str] = dataclasses.field(default_factory=dict)
    metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
    error: Optional[str] = None


class Example(abc.ABC):
    """Abstract base for all TensorLBM examples."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short slug, e.g. ``lid_driven_cavity``."""

    @property
    def description(self) -> str:
        """One-line description (override for richer docs)."""
        return self.__doc__ or ""

    @abc.abstractmethod
    def configure(self) -> Dict[str, Any]:
        """Return default parameters as a dict.

        Keys become CLI arguments (``--key``) and programmatic overrides.
        """

    @abc.abstractmethod
    def run(self, params: Dict[str, Any]) -> ExampleResult:
        """Execute the example with the given parameters."""

    def add_cli_args(self, parser) -> None:
        """Add argparse arguments from ``configure()`` defaults.

        Override for custom argument types / help text.
        """
        defaults = self.configure()
        for key, val in defaults.items():
            arg_name = f"--{key.replace('_', '-')}"
            if isinstance(val, bool):
                parser.add_argument(arg_name, action="store_true", default=val)
            elif isinstance(val, int):
                parser.add_argument(arg_name, type=int, default=val)
            elif isinstance(val, float):
                parser.add_argument(arg_name, type=float, default=val)
            else:
                parser.add_argument(arg_name, default=val)

    def run_from_cli(self, args) -> ExampleResult:
        """Run with parsed argparse namespace → dict."""
        params = self.configure()
        params.update({k: v for k, v in vars(args).items() if v is not None})
        return self.run(params)
