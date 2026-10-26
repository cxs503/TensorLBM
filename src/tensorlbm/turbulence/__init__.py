"""Turbulence models and backwards-compatible LES module exports.

The former ``tensorlbm.turbulence`` module is now ``.turbulence`` inside
this package. Delegate attributes lazily, including private helpers used by
existing callers, while keeping bare package imports independent of torch.
"""

from importlib import import_module as _import_module
from typing import Any

__all__ = [
    "collide_smagorinsky_bgk",
    "collide_smagorinsky_mrt",
    "collide_smagorinsky_bgk3d",
    "collide_smagorinsky_mrt3d",
    "collide_smagorinsky_bgk27",
    "collide_smagorinsky_mrt27",
    "collide_dynamic_smagorinsky_bgk",
    "collide_dynamic_smagorinsky_bgk3d",
    "collide_dynamic_smagorinsky_mrt3d",
    "collide_dynamic_smagorinsky_bgk27",
    "collide_dynamic_smagorinsky_mrt27",
    "collide_wale_bgk",
    "collide_wale_bgk3d",
    "collide_wale_bgk27",
    "collide_vreman_bgk",
    "collide_vreman_bgk3d",
    "collide_vreman_bgk27",
]


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    implementation = _import_module(f"{__name__}.turbulence")
    return getattr(implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
