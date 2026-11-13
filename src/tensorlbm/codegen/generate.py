"""Emit a specialised LBM kernel from a component declaration + a template.

Pipeline (structure borrowed from pylbm's ``Generator.add_routine`` and
lettuce's per-component code generation, see ``protocol.py`` docstring):

1. validate the component against :class:`~tensorlbm.codegen.protocol.CollisionComponent`
   (structural conformance *and* the declared-vs-implied transport invariant);
2. ask the component for its post-collision statements, ask the generator for
   the unrolled moment/equilibrium/streaming statements (stencil constants baked
   as literals, no run-time loop over directions);
3. render :data:`~tensorlbm.codegen.templates.MODULE_TEMPLATE`, hash the source,
   ``exec`` it into a fresh module and hand back a :class:`GeneratedKernel`.

The kernel is pure NumPy and operates on ``f`` of shape ``(q, ny, nx)`` with
periodic boundaries in both directions (the classic 2D Taylor-Green setup).
"""

from __future__ import annotations

import hashlib
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .components import BGKComponent, TRTComponent  # noqa: F401  (registers components)
from .protocol import (
    PROTOCOL_VERSION,
    D2Q9,
    CollisionComponent,
    ComponentSpec,
    StencilSpec,
    validate_component,
)
from .templates import MODULE_TEMPLATE, MODULE_TEMPLATE_NAME, render


def _lit(x: float) -> str:
    return repr(float(x))


def _cu_expr(ex: int, ey: int) -> str:
    terms: list[str] = []
    if ex:
        terms.append("ux" if ex > 0 else "(-ux)")
    if ey:
        terms.append("uy" if ey > 0 else "(-uy)")
    if not terms:
        return "0.0"
    if len(terms) == 1:
        return terms[0]
    return "(" + " + ".join(terms) + ")"


def _moment_expr(fname: str, coeffs: tuple[int, ...], q: int) -> str:
    terms: list[str] = []
    for i in range(q):
        c = coeffs[i]
        if c == 0:
            continue
        terms.append(f"+ {fname}[{i}]" if c > 0 else f"- {fname}[{i}]")
    body = " ".join(terms)
    if body.startswith("+ "):
        body = body[2:]
    return f"({body}) / rho"


def _moments_block(stencil: StencilSpec) -> str:
    q = stencil.q
    rho = " + ".join(f"f[{i}]" for i in range(q))
    lines = [
        f"    rho = {rho}",
        f"    ux = {_moment_expr('f', stencil.ex, q)}",
        f"    uy = {_moment_expr('f', stencil.ey, q)}",
        "    usq = 1.5*(ux*ux + uy*uy)",
    ]
    return "\n".join(lines)


def _equilibrium_block(stencil: StencilSpec) -> str:
    """Unrolled discrete equilibrium, split into symmetric/antisymmetric parts.

    ``feq_plus_i = w_i rho (1 + 1.5 (e_i.u)^2 - 1.5 u^2)``
    ``feq_minus_i = 3 w_i rho (e_i.u)``
    so that ``feq_plus_i + feq_minus_i`` is the standard second-order
    equilibrium and TRT can relax the two parts with different rates.
    """
    lines: list[str] = []
    for i in range(1, stencil.q):
        expr = _cu_expr(stencil.ex[i], stencil.ey[i])
        lines.append(f"    cu_{i} = {expr}")
    for i in range(stencil.q):
        w = _lit(stencil.w[i])
        cu = f"cu_{i}" if i else "0.0"
        lines.append(
            f"    feq_plus_{i} = {w}*rho*(1.0 + 4.5*({cu})*({cu}) - usq)"
        )
        lines.append(f"    feq_minus_{i} = {w}*rho*3.0*({cu})")
    return "\n".join(lines)


def _stream_block(stencil: StencilSpec) -> str:
    """Pull streaming with periodic wrap: ``f_i(x) = f_i^post(x - e_i)``."""
    lines = ["    f_new = np.empty_like(f_post)", "    f_new[0] = f_post[0]"]
    for i in range(1, stencil.q):
        lines.append(
            f"    f_new[{i}] = np.roll(f_post[{i}], {stencil.ey[i]}, axis=0)"
            if stencil.ex[i] == 0
            else (
                f"    f_new[{i}] = np.roll(f_post[{i}], {stencil.ex[i]}, axis=1)"
                if stencil.ey[i] == 0
                else f"    f_new[{i}] = np.roll(f_post[{i}], ({stencil.ey[i]}, {stencil.ex[i]}), axis=(0, 1))"
            )
        )
    lines.append("    return f_new")
    return "\n".join(lines)


def build_source(
    component: object, stencil: StencilSpec = D2Q9
) -> tuple[str, dict]:
    """Return ``(source, provenance)`` for ``component`` on ``stencil``.

    No execution happens here, which makes the emitter side-effect free and
    unit-testable (determinism of the sha256 is a test, not a claim).
    """
    report = validate_component(component, stencil)
    spec: ComponentSpec = component.spec  # type: ignore[attr-defined]
    collision_lines = list(component.emit_collision(stencil))  # type: ignore[attr-defined]
    subs = {
        "component_name": spec.name,
        "family": spec.family,
        "stencil_name": stencil.name,
        "params": ", ".join(f"{k}={_lit(v)}" for k, v in spec.params),
        "protocol_version": PROTOCOL_VERSION,
        "template_name": MODULE_TEMPLATE_NAME,
        "q": stencil.q,
        "ex": tuple(int(v) for v in stencil.ex),
        "ey": tuple(int(v) for v in stencil.ey),
        "w": tuple(_lit(v) for v in stencil.w),
        "s_plus": _lit(component.s_plus()),  # type: ignore[attr-defined]
        "s_minus": _lit(component.s_minus()),  # type: ignore[attr-defined]
        "moments_block": _moments_block(stencil),
        "moments_block_no_f": "    usq = 1.5*(ux*ux + uy*uy)",
        "equilibrium_block": _equilibrium_block(stencil),
        "collision_block": "\n".join(collision_lines),
        "post_stack": ", ".join(f"p{i}" for i in range(stencil.q)),
        "stream_block": _stream_block(stencil),
        "eq_stack": ", ".join(
            f"feq_plus_{i} + feq_minus_{i}" for i in range(stencil.q)
        ),
    }
    # ``w`` must render as a float tuple literal, not a string tuple.
    subs["w"] = "(" + ", ".join(_lit(v) for v in stencil.w) + ")"
    source = render(MODULE_TEMPLATE, subs, name=MODULE_TEMPLATE_NAME)
    provenance = {
        "component": report,
        "template_name": MODULE_TEMPLATE_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "stencil": stencil.as_dict(),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_lines": source.count("\n") + 1,
    }
    return source, provenance


@dataclass(frozen=True)
class GeneratedKernel:
    """A compiled generated kernel plus the declaration it was built from."""

    spec: ComponentSpec
    stencil: StencilSpec
    source: str
    sha256: str
    module: types.ModuleType
    provenance: Mapping[str, Any]

    # -- population-level API ------------------------------------------------
    def step(self, f: np.ndarray) -> np.ndarray:
        return self.module.step(f)

    def collide(self, f: np.ndarray) -> np.ndarray:
        return self.module.collide(f)

    def stream(self, f_post: np.ndarray) -> np.ndarray:
        return self.module.stream(f_post)

    def moments(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.module.moments(f)

    def equilibrium(
        self, rho: np.ndarray, ux: np.ndarray, uy: np.ndarray
    ) -> np.ndarray:
        return self.module.equilibrium(rho, ux, uy)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(self.source)
        return p


def generate_kernel(component: object, stencil: StencilSpec = D2Q9) -> GeneratedKernel:
    """Validate, emit, hash and exec a kernel for ``component``."""
    source, provenance = build_source(component, stencil)
    sha = provenance["source_sha256"]
    module = types.ModuleType(f"tensorlbm_generated_{sha[:12]}")
    module.__dict__["__builtins__"] = __builtins__
    exec(compile(source, f"<generated {sha[:12]}>", "exec"), module.__dict__)
    for attr in ("step", "collide", "stream", "moments", "equilibrium"):
        if not callable(getattr(module, attr, None)):
            raise RuntimeError(f"generated module is missing {attr}()")
    return GeneratedKernel(
        spec=component.spec,  # type: ignore[attr-defined]
        stencil=stencil,
        source=source,
        sha256=sha,
        module=module,
        provenance=provenance,
    )


def generate_kernel_for(name: str, **params: float) -> GeneratedKernel:
    """Convenience path: registry key + params -> kernel."""
    from .protocol import get_component

    cls = get_component(name)
    return generate_kernel(cls(**params))


__all__ = [
    "GeneratedKernel",
    "build_source",
    "generate_kernel",
    "generate_kernel_for",
]