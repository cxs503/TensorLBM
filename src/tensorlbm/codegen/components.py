"""Collision components that satisfy :mod:`tensorlbm.codegen.protocol`.

Two families ship with the generator:

``BGKComponent``
    single relaxation time, ``s_plus = s_minus = 1/tau``.
``TRTComponent``
    two relaxation times (Ginzburg 2008, "magic parameter" formulation):
    ``s_plus = 1/tau`` and ``s_minus`` chosen from ``magic`` through
    ``magic = (1/s_minus - 1/2) (1/s_plus - 1/2)``, i.e.
    ``s_minus = 1 / (magic/(1/s_plus - 1/2) + 1/2)``.
    Equilibrium is split into its symmetric/antisymmetric parts about the
    opposite direction, so the emitted code relaxes ``feq_plus``/``feq_minus``
    separately.

Both components emit *statements*, not numbers: the constants are baked into
the text by :mod:`tensorlbm.codegen.generate`, which is what makes the output a
specialised kernel rather than a parametrised loop.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .protocol import CS2, ComponentSpec, StencilSpec, register_component


def _fmt(x: float) -> str:
    """Round-trip exact float literal (repr of a float is exact)."""
    return repr(float(x))


@register_component
class BGKComponent:
    """Single-relaxation-time (BGK) collision component."""

    registry_key = "BGK"
    spec: ComponentSpec

    def __init__(self, tau: float, name: str | None = None) -> None:
        self.tau = float(tau)
        self.spec = ComponentSpec.make(name or "bgk", "BGK", tau=self.tau)
        self.spec.validate()

    def s_plus(self) -> float:
        return 1.0 / self.tau

    def s_minus(self) -> float:
        return 1.0 / self.tau

    def transport_coefficients(self) -> Mapping[str, float]:
        return {
            "nu": (self.tau - 0.5) * CS2,
            "zeta": (self.tau - 0.5) * CS2,
            "magic": (self.tau - 0.5) ** 2,
        }

    def emit_collision(self, stencil: StencilSpec) -> Sequence[str]:
        sp = _fmt(self.s_plus())
        return [
            f"    p{i} = f[{i}] - {sp}*(f[{i}] - (feq_plus_{i} + feq_minus_{i}))"
            for i in range(stencil.q)
        ]


@register_component
class TRTComponent:
    """Two-relaxation-time collision component (Ginzburg 2008)."""

    registry_key = "TRT"
    spec: ComponentSpec

    def __init__(self, tau: float, magic: float = 3.0 / 16.0, name: str | None = None) -> None:
        self.tau = float(tau)
        self.magic = float(magic)
        self.spec = ComponentSpec.make(
            name or "trt", "TRT", tau=self.tau, magic=self.magic
        )
        self.spec.validate()

    def s_plus(self) -> float:
        return 1.0 / self.tau

    def s_minus(self) -> float:
        lam = 1.0 / self.s_plus() - 0.5
        return 1.0 / (self.magic / lam + 0.5)

    def transport_coefficients(self) -> Mapping[str, float]:
        return {
            "nu": (self.tau - 0.5) * CS2,
            "zeta": (1.0 / self.s_minus() - 0.5) * CS2,
            "magic": self.magic,
        }

    def emit_collision(self, stencil: StencilSpec) -> Sequence[str]:
        sp, sm = _fmt(self.s_plus()), _fmt(self.s_minus())
        lines: list[str] = []
        for i in range(1, stencil.q):
            if i > stencil.opposite[i]:
                continue
            j = stencil.opposite[i]
            lines.append(f"    fp_{i} = 0.5*(f[{i}] + f[{j}])")
            lines.append(f"    fm_{i} = 0.5*(f[{i}] - f[{j}])")
        lines.append("    fp_0 = f[0]")
        lines.append("    fm_0 = 0.0")
        for i in range(stencil.q):
            j = stencil.opposite[i]
            pair = min(i, j)
            if i == j:
                lines.append(
                    f"    p{i} = fp_0*(1.0 - {sp}) + {sp}*feq_plus_{i}"
                    f" + fm_0*(1.0 - {sm}) + {sm}*feq_minus_{i}"
                )
            else:
                # fm_j = -fm_i for an opposite pair: the antisymmetric part of
                # the population flips sign under i -> opposite(i).
                sign = "" if i < j else "-"
                lines.append(
                    f"    p{i} = fp_{pair}*(1.0 - {sp}) + {sp}*feq_plus_{i}"
                    f" + {sign}fm_{pair}*(1.0 - {sm}) + {sm}*feq_minus_{i}"
                )
        return lines


__all__ = ["BGKComponent", "TRTComponent"]