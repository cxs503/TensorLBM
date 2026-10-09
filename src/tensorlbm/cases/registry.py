"""Name-keyed case registry (lettuce ``_flow_by_name`` pattern, MIT).

lettuce registers its flows in a plain ``Dict[str, Tuple[Type[ExtFlow],
Type[Stencil]]]`` so the CLI can construct any flow by name
(``lettuce/ext/_flows/_flow_by_name.py``).  TensorLBM's equivalent keeps
one registry key per case class so the platform layer (and the upcoming
``scan_runner``) can enumerate and instantiate cases without importing
worker scripts.  Changes versus lettuce: duplicate names are rejected,
unregistration is supported for tests, and factory metadata (default
parameters, lattice, collision) is exposed for enumeration.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from .base import CaseBase

__all__ = [
    "register_case",
    "get_case",
    "has_case",
    "list_cases",
    "unregister_case",
    "case_registry",
]

CaseT = TypeVar("CaseT", bound=type)

#: name → case class (lettuce ``flow_by_name`` equivalent).
case_registry: dict[str, type[CaseBase]] = {}


def register_case(
    name: str | None = None,
) -> Callable[[CaseT], CaseT]:
    """Class decorator registering a :class:`~tensorlbm.cases.CaseBase`.

    ::

        @register_case("cavity")
        class LidCavityCase(CaseBase): ...

    The registry key defaults to the class's ``name`` attribute.  Raises
    ``ValueError`` on duplicate names — one key must mean one case.
    """

    def decorator(cls: CaseT) -> CaseT:
        key = name if name is not None else getattr(cls, "name", "")
        if not key or not isinstance(key, str):
            raise ValueError(
                f"case {cls.__name__} must set a non-empty 'name' or register_case(name=...)"
            )
        if key in case_registry:
            existing = case_registry[key].__name__
            raise ValueError(
                f"case name {key!r} is already registered by {existing}; case names must be unique"
            )
        case_registry[key] = cls  # type: ignore[assignment]
        return cls

    return decorator


def get_case(
    name: str,
    *,
    resolution: int | tuple[int, int, int] | None = None,
    re: float | None = None,
    **kwargs: Any,
) -> CaseBase:
    """Instantiate a registered case by name.

    Args:
        name: registry key (see :func:`list_cases`).
        resolution: forwarded to the case constructor.
        re: Reynolds number (each case defines its own default).
        **kwargs: case-specific parameters (``u_lid``, ``u_in``, …).

    Raises:
        KeyError: unknown name (listing the available names).
    """
    try:
        cls = case_registry[name]
    except KeyError:
        available = ", ".join(sorted(case_registry)) or "<empty>"
        raise KeyError(f"unknown case {name!r}; registered cases: {available}") from None
    if resolution is not None and "resolution" not in kwargs:
        kwargs["resolution"] = resolution
    if re is not None and "re" not in kwargs:
        kwargs["re"] = re
    return cls(**kwargs)


def has_case(name: str) -> bool:
    """Return whether *name* is registered."""
    return name in case_registry


def unregister_case(name: str) -> type[CaseBase]:
    """Remove *name* from the registry (returns the removed class)."""
    try:
        return case_registry.pop(name)
    except KeyError:
        raise KeyError(f"unknown case {name!r}") from None


def list_cases() -> list[dict[str, Any]]:
    """Enumerate registered cases with factory metadata.

    Returns one dict per case with ``name``, ``lattice``, ``collision``,
    ``description`` and ``default_params`` — the surface the platform
    layer and ``scan_runner`` consume to build UIs / DoE plans without
    importing worker scripts.
    """
    out: list[dict[str, Any]] = []
    for name, cls in sorted(case_registry.items()):
        defaults = getattr(cls, "default_params", None)
        if callable(defaults):
            defaults = defaults()
        out.append(
            {
                "name": name,
                "class": cls.__name__,
                "lattice": getattr(cls, "lattice", "D3Q19"),
                "collision": getattr(cls, "collision", "bgk"),
                "description": getattr(cls, "description", ""),
                "default_params": dict(defaults) if defaults else {},
            }
        )
    return out
