# -*- coding: utf-8 -*-
"""Keep docs/software_manual.md API examples in sync with the real public API.

The manual's chapter 7 ("Public API Reference") is copy-paste material.  During
a 2026-10 refactor several constructors/functions were renamed (e.g.
``LBMUnitConverter(L_ref=...)`` -> ``LBMUnitConverter(re=..., l_phys=...)``),
leaving examples that fail immediately with ``TypeError: unexpected keyword
argument``.

This test parses every fenced Python block in the manual, resolves the names
imported from ``tensorlbm``, and checks that each keyword argument actually
exists in the live signature.  It is a static check: no simulations are run,
no GPU/data files are needed, and it costs milliseconds in CI.

A failure means either the documentation is stale (fix the docs) or a public
API was renamed without updating the manual (fix the docs too).
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re

import pytest

import tensorlbm

MANUAL = pathlib.Path(__file__).resolve().parents[1] / "docs" / "software_manual.md"


def _python_blocks(text: str):
    """Yield (ordinal, source) for fenced python/py blocks."""
    for i, m in enumerate(re.finditer(r"```(?:python|py)\s*\n(.*?)```", text, re.S), 1):
        yield i, m.group(1)


def _imported_tensorlbm_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tensorlbm"):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("tensorlbm"):
                    names.add(alias.asname or alias.name.split(".")[0])
    return names


def test_manual_api_keywords_match_live_signatures():
    """Every keyword used with a tensorlbm API in the manual must exist."""
    assert MANUAL.is_file(), f"manual not found: {MANUAL}"
    text = MANUAL.read_text(encoding="utf-8")

    mismatches: list[str] = []
    checked = 0
    for ordinal, source in _python_blocks(text):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Some blocks are illustrative fragments (shape tables, JSON
            # samples, source excerpts), not runnable code.  Skip them.
            continue
        imported = _imported_tensorlbm_names(tree)
        if not imported:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            name = node.func.id
            if name not in imported:
                continue
            obj = getattr(tensorlbm, name, None)
            if obj is None or not callable(obj):
                continue
            try:
                sig = inspect.signature(obj)
            except (TypeError, ValueError):
                continue
            params = sig.parameters
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
                continue
            for kw in node.keywords:
                if kw.arg is None:  # **kwargs unpacking
                    continue
                if kw.arg not in params:
                    mismatches.append(
                        f"block #{ordinal}: {name}({kw.arg}=...) -> not in "
                        f"{name}{sig}"
                    )
                checked += 1

            # Also catch required arguments that are simply omitted, e.g.
            # ``wigley_hull_mask(..., draft=...)`` without the mandatory
            # ``device`` argument.  Keyword-name checks alone miss these.
            positional = [
                pname for pname, p in params.items()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            covered = set(positional[: len(node.args)])
            covered.update(kw.arg for kw in node.keywords if kw.arg is not None)
            required = [
                pname for pname, p in params.items()
                if p.default is inspect.Parameter.empty
                and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                               inspect.Parameter.POSITIONAL_OR_KEYWORD,
                               inspect.Parameter.KEYWORD_ONLY)
            ]
            missing = [pname for pname in required if pname not in covered]
            if missing:
                mismatches.append(
                    f"block #{ordinal}: {name}(...) -> missing required "
                    f"argument(s) {', '.join(missing)} in {name}{sig}"
                )

    assert checked > 0, "no API calls checked; parser may be stale"
    assert not mismatches, (
        "software_manual.md uses keyword arguments that no longer exist:\n  - "
        + "\n  - ".join(mismatches)
    )


@pytest.mark.parametrize(
    "name",
    [
        "LBMUnitConverter",
        "wigley_hull_mask",
        "AdaptationSchedule",
        "run_ai_dns_pipeline",
        "CHTConfig",
        "FWHSurface",
        "auto_decompose",
        "MultiGPUSolver2D",
    ],
)
def test_documented_api_is_still_exported(name):
    """Guard against the manual referencing APIs removed from the public surface."""
    assert hasattr(tensorlbm, name), f"tensorlbm.{name} is no longer exported"
