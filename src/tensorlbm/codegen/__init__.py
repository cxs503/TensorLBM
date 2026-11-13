"""Template-driven, protocol-checked LBM kernel generation (D2Q9 baseline).

Public surface::

    from tensorlbm.codegen import BGKComponent, TRTComponent, generate_kernel

    kernel = generate_kernel(BGKComponent(tau=0.56))
    kernel.sha256          # provenance of the emitted source
    kernel.provenance      # component declaration + template + stencil
    f = kernel.step(f)     # (9, ny, nx) periodic numpy kernel

The generated kernel is cross-validated against the independent
:mod:`tensorlbm.codegen.reference` implementation and against the analytic
two-dimensional Taylor-Green decay (see ``xval/component_protocol_codegen``).
"""

from __future__ import annotations

from .components import BGKComponent, TRTComponent
from .generate import GeneratedKernel, build_source, generate_kernel, generate_kernel_for
from .protocol import (
    PROTOCOL_VERSION,
    D2Q9,
    CollisionComponent,
    ComponentSpec,
    ProtocolViolation,
    StencilSpec,
    StencilViolation,
    available_components,
    get_component,
    register_component,
    validate_component,
)
from .reference import (
    reference_collide,
    reference_equilibrium,
    reference_moments,
    reference_step,
    reference_stream,
    shear_viscosity,
)

__all__ = [
    "BGKComponent",
    "CollisionComponent",
    "ComponentSpec",
    "D2Q9",
    "GeneratedKernel",
    "PROTOCOL_VERSION",
    "ProtocolViolation",
    "StencilSpec",
    "StencilViolation",
    "TRTComponent",
    "available_components",
    "build_source",
    "generate_kernel",
    "generate_kernel_for",
    "get_component",
    "reference_collide",
    "reference_equilibrium",
    "reference_moments",
    "reference_step",
    "reference_stream",
    "register_component",
    "shear_viscosity",
    "validate_component",
]