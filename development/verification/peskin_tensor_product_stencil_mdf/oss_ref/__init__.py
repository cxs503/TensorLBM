"""Vendored haimingz/vivsim immersed-boundary reference (2-D), numpy-shimmed.

Exports mirror ``vivsim/ib/__init__.py`` at upstream commit fb84b12 so that the
comparison driver calls the OSS API surface unchanged.
"""

from .geometry import get_area, get_ds
from .kernels import kernel_cosine_4pt, kernel_peskin_3pt, kernel_peskin_4pt
from .mdf import multi_direct_forcing
from .stencil import get_ib_stencil, interpolate, spread

__all__ = [
    "get_area",
    "get_ds",
    "kernel_cosine_4pt",
    "kernel_peskin_3pt",
    "kernel_peskin_4pt",
    "multi_direct_forcing",
    "get_ib_stencil",
    "interpolate",
    "spread",
]
