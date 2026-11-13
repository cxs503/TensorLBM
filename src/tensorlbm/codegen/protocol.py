"""Component protocol for template-driven LBM kernel generation (D2Q9 baseline).

Purpose
-------
Every LBM "component" (equilibrium, collision operator, ...) has to declare, in a
machine-checkable way, *which lattice it lives on* and *which transport
coefficients it claims to realise*.  A code generator can then emit a
specialised (unrolled, constants baked in) kernel from a text template and the
declaration, and the declaration itself becomes the cross-validation anchor:
the measured viscosity of the generated kernel must equal the viscosity the
component declared, otherwise the protocol/implementation pair is inconsistent.

Method provenance (see ``docs/codegen_component_protocol.md`` for the survey)
-----------------------------------------------------------------------------
* pylbm ``pylbm/stencil.py`` + ``pylbm/generator/generator.py::Generator.add_routine``
  (local copy ``~/oss-cache/pylbm``, MIT): a *Stencil* object holding
  ``v``/``w``/``lambda_`` drives routine generation; the generator keeps a
  routine registry and emits code from symbolic expressions.  Our
  ``StencilSpec`` + ``ComponentSpec`` + registry mirror that structure.
* lettuce ``lettuce/cuda_native/_default_code_gen.py`` (via ``oss_solutions``
  id=119, ``feature_slug`` this anchor): per-component code templates are
  instantiated for a given stencil, which is exactly the "component protocol ->
  template -> kernel" pipeline implemented here in pure Python/NumPy.
* lbmpy (AST-based kernel generation) informed the "emit statements, do not
  interpret parameters at run time" rule only; no code is copied.

Nothing here is domain specific: no ship, no benchmark name, no case data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

PROTOCOL_VERSION = "1.0"

#: Transport-coefficient model used by the whole package: lattice units,
#: ``nu = (1/s_plus - 1/2) / 3`` for the shear mode.
CS2 = 1.0 / 3.0

_FAMILY_REQUIRED_PARAMS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "BGK": ("tau",),
        "TRT": ("tau", "magic"),
    }
)


class StencilViolation(ValueError):
    """Raised when a stencil does not satisfy the D2Q9 lattice invariants."""


class ProtocolViolation(ValueError):
    """Raised when a component does not satisfy the component protocol."""

    def __init__(self, component: str, reasons: Iterable[str]) -> None:
        self.component = component
        self.reasons = tuple(reasons)
        super().__init__(f"{component}: " + "; ".join(self.reasons))


@dataclass(frozen=True)
class StencilSpec:
    """A discrete velocity set with its weights (pylbm ``Stencil`` analogue).

    Only the attributes a generator needs are kept: velocities, weights, the
    opposite-direction permutation and the human name.
    """

    name: str
    ex: tuple[int, ...]
    ey: tuple[int, ...]
    w: tuple[float, ...]
    opposite: tuple[int, ...]

    @property
    def q(self) -> int:
        return len(self.ex)

    # ---------------------------------------------------------------- checks
    def violations(self, tol: float = 1e-12) -> list[str]:
        """Return the list of broken lattice invariants (empty == valid).

        The invariants are the ones a D2Q9 equilibrium expansion actually
        relies on: the zero direction first, an involutive opposite map,
        normalised weights, zero first moment, isotropic second moment
        ``sum w e_a e_b = cs2 delta_ab`` and the fourth-moment pair
        ``sum w e_x^4 = 3 cs2^2 = 1/3`` and ``sum w e_x^2 e_y^2 = cs2^2 = 1/9``.
        """
        out: list[str] = []
        q = self.q
        if not (len(self.ey) == len(self.w) == len(self.opposite) == q):
            out.append("ragged stencil: ex/ey/w/opposite length mismatch")
            return out
        if self.opposite[0] != 0 or self.ex[0] != 0 or self.ey[0] != 0:
            out.append("direction 0 must be the rest direction")
        for i, j in enumerate(self.opposite):
            if not 0 <= j < q:
                out.append(f"opposite[{i}]={j} out of range")
                continue
            if self.opposite[j] != i:
                out.append(f"opposite map not involutive at i={i}")
            if (self.ex[i], self.ey[i]) != (-self.ex[j], -self.ey[j]):
                out.append(f"opposite[{i}] is not -e_{i}")
            if abs(self.w[i] - self.w[j]) > tol:
                out.append(f"weights of opposite pair {i}/{j} differ")
        if abs(sum(self.w) - 1.0) > tol:
            out.append(f"sum(w)={sum(self.w)!r} != 1")
        s1x = sum(wi * exi for wi, exi in zip(self.w, self.ex))
        s1y = sum(wi * eyi for wi, eyi in zip(self.w, self.ey))
        if abs(s1x) > tol or abs(s1y) > tol:
            out.append("first moment is not zero")
        s2xx = sum(wi * exi * exi for wi, exi in zip(self.w, self.ex))
        s2yy = sum(wi * eyi * eyi for wi, eyi in zip(self.w, self.ey))
        s2xy = sum(wi * exi * eyi for wi, exi, eyi in zip(self.w, self.ex, self.ey))
        if abs(s2xx - CS2) > tol or abs(s2yy - CS2) > tol:
            out.append(f"second moment != cs2={CS2!r}")
        if abs(s2xy) > tol:
            out.append("second moment off-diagonal != 0")
        s4xxxx = sum(wi * exi**4 for wi, exi in zip(self.w, self.ex))
        s4xxyy = sum(wi * exi * exi * eyi * eyi for wi, exi, eyi in zip(self.w, self.ex, self.ey))
        if abs(s4xxxx - 3.0 * CS2**2) > tol:
            out.append("fourth moment sum(w e_x^4) != 3 cs2^2")
        if abs(s4xxyy - CS2**2) > tol:
            out.append("fourth moment sum(w e_x^2 e_y^2) != cs2^2")
        return out

    def validate(self) -> None:
        bad = self.violations()
        if bad:
            raise StencilViolation(f"{self.name}: " + "; ".join(bad))

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "ex": list(self.ex),
            "ey": list(self.ey),
            "w": list(self.w),
            "opposite": list(self.opposite),
        }


D2Q9 = StencilSpec(
    name="D2Q9",
    ex=(0, 1, 0, -1, 0, 1, -1, -1, 1),
    ey=(0, 0, 1, 0, -1, 1, 1, -1, -1),
    w=(4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36),
    opposite=(0, 3, 4, 1, 2, 7, 8, 5, 6),
)
D2Q9.validate()


@dataclass(frozen=True)
class ComponentSpec:
    """What a component declares about itself (the protocol payload)."""

    name: str
    family: str
    params: tuple[tuple[str, float], ...]

    @classmethod
    def make(cls, name: str, family: str, **params: float) -> "ComponentSpec":
        return cls(name=name, family=family, params=tuple(sorted(params.items())))

    def param(self, key: str) -> float:
        try:
            return dict(self.params)[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ProtocolViolation(self.name, [f"missing param {key!r}"]) from exc

    @property
    def params_dict(self) -> dict[str, float]:
        return dict(self.params)

    def violations(self) -> list[str]:
        out: list[str] = []
        if not self.name:
            out.append("empty component name")
        if self.family not in _FAMILY_REQUIRED_PARAMS:
            out.append(f"unknown family {self.family!r}")
            return out
        have = dict(self.params)
        for key in _FAMILY_REQUIRED_PARAMS[self.family]:
            if key not in have:
                out.append(f"family {self.family} requires param {key!r}")
        tau = have.get("tau")
        if tau is not None and not tau > 0.5:
            out.append(f"tau={tau!r} must exceed 0.5 for a positive viscosity")
        magic = have.get("magic")
        if magic is not None and not 0.0 < magic < 1.0:
            out.append(f"magic={magic!r} outside (0, 1)")
        return out

    def validate(self) -> None:
        bad = self.violations()
        if bad:
            raise ProtocolViolation(self.name, bad)

    def as_dict(self) -> dict:
        return {"name": self.name, "family": self.family, "params": self.params_dict}


@runtime_checkable
class CollisionComponent(Protocol):
    """Structural contract every collision component must satisfy.

    ``s_plus``/``s_minus`` are the even/odd relaxation rates actually used by
    the emitted code, ``transport_coefficients`` is what the component claims,
    and ``emit_collision`` returns the statements of the post-collision block
    (``p<i>`` from ``f``, ``feq_plus_<i>`` and ``feq_minus_<i>``).
    """

    spec: ComponentSpec

    def s_plus(self) -> float: ...

    def s_minus(self) -> float: ...

    def transport_coefficients(self) -> Mapping[str, float]: ...

    def emit_collision(self, stencil: StencilSpec) -> Sequence[str]: ...


def validate_component(
    component: object, stencil: StencilSpec = D2Q9, tol: float = 1e-12
) -> dict:
    """Check a component against the protocol; raise :class:`ProtocolViolation`.

    Beyond structural conformance this enforces the *self-consistency*
    invariant that makes generated-code cross-validation meaningful:

    * ``1/s_plus`` must reproduce the shear viscosity the component declares
      (``nu = (1/s_plus - 1/2)/3``);
    * for TRT, ``magic = (1/s_minus - 1/2) * (1/s_plus - 1/2)``;
    * both rates must lie inside the stable interval ``(0, 2)``;
    * the emitted statements must be non-empty, placeholder-free and mention
      every post-collision time level ``p<i>``.
    """
    reasons: list[str] = []
    display = type(component).__name__
    if not isinstance(component, CollisionComponent):
        return _fail(display, ["does not satisfy the CollisionComponent protocol"])
    spec = component.spec
    if not isinstance(spec, ComponentSpec):
        return _fail(display, [f"spec is {type(spec).__name__}, not ComponentSpec"])
    reasons.extend(spec.violations())
    if reasons:
        return _fail(display, reasons)

    stencil.validate()
    sp, sm = float(component.s_plus()), float(component.s_minus())
    if not 0.0 < sp < 2.0:
        reasons.append(f"s_plus={sp!r} outside (0, 2)")
    if not 0.0 < sm < 2.0:
        reasons.append(f"s_minus={sm!r} outside (0, 2)")

    declared = dict(component.transport_coefficients())
    if "nu" not in declared:
        reasons.append("transport_coefficients() does not declare 'nu'")
    else:
        implied = (1.0 / sp - 0.5) * CS2 if sp != 0.0 else float("inf")
        if abs(implied - float(declared["nu"])) > tol:
            reasons.append(
                f"declared nu={declared['nu']!r} != nu implied by s_plus={sp!r} ({implied!r})"
            )
    if spec.family == "TRT":
        magic_implied = (1.0 / sm - 0.5) * (1.0 / sp - 0.5)
        if abs(magic_implied - spec.param("magic")) > tol:
            reasons.append(
                f"magic={spec.param('magic')!r} != (1/s_minus-1/2)(1/s_plus-1/2)"
                f"={magic_implied!r}"
            )

    lines = list(component.emit_collision(stencil))
    if not lines:
        reasons.append("emit_collision() returned no statements")
    else:
        joined = "\n".join(lines)
        if "{" in joined or "}" in joined:
            reasons.append("emit_collision() left an unresolved template placeholder")
        for i in range(stencil.q):
            if f"p{i} =" not in joined:
                reasons.append(f"emit_collision() does not define p{i}")
    if reasons:
        return _fail(display, reasons)
    return {
        "component": display,
        "name": spec.name,
        "family": spec.family,
        "params": spec.params_dict,
        "s_plus": sp,
        "s_minus": sm,
        "transport": declared,
        "stencil": stencil.name,
        "protocol_version": PROTOCOL_VERSION,
        "statements": len(lines),
    }


def _fail(display: str, reasons: list[str]) -> None:
    raise ProtocolViolation(display, reasons)


# ----------------------------------------------------------------- registry
_REGISTRY: dict[str, type] = {}


def register_component(cls: type) -> type:
    """Register a component class; the class is validated at instantiation.

    The registry is what the generator looks components up by, mirroring
    pylbm's routine registry: a name -> implementation map that the code
    generator resolves without importing anything case specific.
    """
    annotated = "spec" in getattr(cls, "__annotations__", {})
    if not (annotated or hasattr(cls, "spec")):
        raise ProtocolViolation(
            cls.__name__, ["registered class declares neither a 'spec' annotation nor attribute"]
        )
    key = getattr(cls, "registry_key", cls.__name__)
    if key in _REGISTRY:
        raise ProtocolViolation(cls.__name__, [f"registry key {key!r} already registered"])
    _REGISTRY[key] = cls
    return cls


def available_components() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_component(key: str) -> type:
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"unknown component {key!r}; have {available_components()}") from exc


__all__ = [
    "CS2",
    "CollisionComponent",
    "ComponentSpec",
    "D2Q9",
    "PROTOCOL_VERSION",
    "ProtocolViolation",
    "StencilSpec",
    "StencilViolation",
    "available_components",
    "get_component",
    "register_component",
    "validate_component",
]