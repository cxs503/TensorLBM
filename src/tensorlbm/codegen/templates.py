"""Text templates a generated kernel is rendered from.

The templates are deliberately plain ``str`` with ``{name}`` fields and no
templating dependency: :func:`render` uses :class:`string.Formatter` to check
that every field was supplied *and* that nothing else was left behind, so a
template edit that silently drops a placeholder fails loudly in the test suite
instead of producing a kernel that imports but does nothing.

Template shape follows lettuce ``cuda_native/_template.py`` /
``_default_code_gen.py`` (component blocks concatenated into one kernel) and
pylbm's routine emission, at NumPy level.
"""

from __future__ import annotations

import re
import string
from typing import Mapping

from .protocol import PROTOCOL_VERSION

#: Name recorded in provenance.
MODULE_TEMPLATE_NAME = "module.d2q9.v1"

MODULE_TEMPLATE = '''"""Generated LBM kernel -- produced by tensorlbm.codegen, do not edit.

component   : {component_name}
family      : {family}
stencil     : {stencil_name}
params      : {params}
protocol    : {protocol_version}
template    : {template_name}
generated_by: tensorlbm.codegen.generate
"""

from __future__ import annotations

import numpy as np

Q = {q}
EX = {ex}
EY = {ey}
W = {w}
S_PLUS = {s_plus}
S_MINUS = {s_minus}
FAMILY = {family!r}
COMPONENT = {component_name!r}
PROTOCOL_VERSION = {protocol_version!r}


def moments(f):
{moments_block}
    return rho, ux, uy


def collide(f):
{moments_block}
{equilibrium_block}
{collision_block}
    return np.stack([{post_stack}])


def stream(f_post):
{stream_block}


def step(f):
    return stream(collide(f))


def equilibrium(rho, ux, uy):
{moments_block_no_f}
{equilibrium_block}
    return np.stack([{eq_stack}])
'''

_PLACEHOLDER_RE = re.compile(r"\{[a-z_][a-z0-9_]*\}")

_FIELDS = frozenset(
    {
        "component_name",
        "family",
        "stencil_name",
        "params",
        "protocol_version",
        "template_name",
        "q",
        "ex",
        "ey",
        "w",
        "s_plus",
        "s_minus",
        "moments_block",
        "moments_block_no_f",
        "equilibrium_block",
        "collision_block",
        "post_stack",
        "stream_block",
        "eq_stack",
    }
)


def template_fields(template: str = MODULE_TEMPLATE) -> frozenset[str]:
    names: set[str] = set()
    for _literal, field_name, _spec, _conv in string.Formatter().parse(template):
        if field_name is not None:
            names.add(field_name)
    return frozenset(names)


def render(template: str, subs: Mapping[str, object], *, name: str = "<template>") -> str:
    """Strictly format ``template`` with ``subs``.

    Raises ``KeyError`` for a missing field, ``ValueError`` for an unknown one
    or for any residual ``{``/``}`` in the output (a sign of a broken field
    name), so a template/emitter mismatch can never pass silently.
    """
    declared = template_fields(template)
    missing = declared - set(subs)
    extra = set(subs) - declared
    if missing:
        raise KeyError(f"{name}: template fields not supplied: {sorted(missing)}")
    if extra:
        raise ValueError(f"{name}: substitutions not used by template: {sorted(extra)}")
    out = template.format(**subs)
    residual = _PLACEHOLDER_RE.search(out)
    if residual is not None:
        raise ValueError(
            f"{name}: unresolved placeholder {residual.group(0)!r} in rendered source"
        )
    return out


__all__ = [
    "MODULE_TEMPLATE",
    "MODULE_TEMPLATE_NAME",
    "PROTOCOL_VERSION",
    "render",
    "template_fields",
]