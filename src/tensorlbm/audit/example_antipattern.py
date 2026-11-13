"""Machine audit of the *example-per-main* anti-pattern.

The anti-pattern (as practised by ``sorush-khajepor/listLBM`` and, at a larger
scale, by TensorLBM's own ``examples/``): every example is a stand-alone
``main`` -- its own argument parsing, its own lattice construction, its own
time loop, its own post-processing.  Consequences are measurable, so the audit
measures them instead of asserting them:

``M1 standalone mains``
    how many example files carry their own ``if __name__ == "__main__"``
    entry, parse their own CLI and drive their own time loop.
``M2 duplicated setup blocks``
    structural duplication: every top-level statement and every function body
    is normalised (identifier names and literal values erased) and hashed; a
    block whose hash appears in two or more files is a duplicated block.
    ``dup_loc`` is the number of source lines inside such blocks and
    ``dup_loc_ratio`` its share of the scanned lines.
``M3 thin-main share``
    how many examples delegate the run to a library-level case runtime
    (``tensorlbm.example_harness``) instead of re-implementing it.

The audit is a *tool*, not a claim: it emits JSON (plus a human summary) and
never grades anything by itself.  It also ships a discrimination control -- see
``tests/test_example_antipattern_audit.py`` -- because a detector that flags
everything is worthless.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

__all__ = ["audit_tree", "format_summary", "main"]

# statements shorter than this are noise (imports, single calls, docstrings)
_MIN_BLOCK_LINES = 3
# names whose presence means "this file drives its own lattice loop"
_LOOP_KERNELS = {"stream", "stream3d", "roll", "collide_bgk", "collide_trt", "collide_mrt"}
_THIN_ENTRY_MARKERS = {"run_case", "get_case", "example_harness"}


@dataclass
class FileReport:
    path: str
    loc: int = 0
    has_main: bool = False
    parses_argv: bool = False
    own_step_loop: bool = False
    thin_main: bool = False
    block_hashes: list[str] = field(default_factory=list)
    block_spans: list[tuple[int, int]] = field(default_factory=list)
    block_driver: list[bool] = field(default_factory=list)
    error: str | None = None


def _normalise(node: ast.AST) -> str:
    """Structure-only AST dump: identifier names and literals erased."""

    class _Norm(ast.NodeTransformer):
        def visit_Name(self, n: ast.Name) -> ast.AST:  # noqa: N802
            return ast.copy_location(ast.Name(id="_v", ctx=n.ctx), n)

        def visit_Attribute(self, n: ast.Attribute) -> ast.AST:  # noqa: N802
            return ast.copy_location(
                ast.Attribute(value=self.visit(n.value), attr=n.attr, ctx=n.ctx), n
            )

        def visit_Constant(self, n: ast.Constant) -> ast.AST:  # noqa: N802
            tag = type(n.value).__name__
            return ast.copy_location(ast.Constant(value=f"<{tag}>"), n)

        def visit_arg(self, n: ast.arg) -> ast.AST:
            return ast.copy_location(ast.arg(arg="_a", annotation=None), n)

        def visit_FunctionDef(self, n):  # noqa: N802
            return ast.copy_location(
                ast.FunctionDef(
                    name="_f",
                    args=self.visit(n.args),
                    body=[self.visit(b) for b in n.body],
                    decorator_list=[],
                    returns=None,
                    type_comment=None,
                ),
                n,
            )

        def visit_AsyncFunctionDef(self, n):  # noqa: N802
            return self.visit_FunctionDef(n)

    return ast.dump(_Norm().visit(ast.parse(ast.unparse(node))), annotate_fields=True)


def _block_span(node: ast.AST) -> int:
    end = getattr(node, "end_lineno", None) or getattr(node, "lineno", 0)
    return max(int(end) - int(getattr(node, "lineno", end)) + 1, 1)


def _scan_file(path: str) -> FileReport:
    rep = FileReport(path=path)
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
        tree = ast.parse(src)
    except (OSError, SyntaxError) as exc:  # pragma: no cover - unreadable file
        rep.error = f"{type(exc).__name__}: {exc}"
        return rep
    rep.loc = src.count("\n") + (0 if src.endswith("\n") else 1)

    names: set[str] = set()
    harness_import = False
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and "example_harness" in n.module:
            harness_import = True
        if isinstance(n, ast.Import) and any("example_harness" in a.name for a in n.names):
            harness_import = True
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, ast.ImportFrom) and n.module:
            names.update(n.module.split("."))
        elif isinstance(n, ast.Import):
            for a in n.names:
                names.update(a.name.split("."))

    for node in tree.body:
        if isinstance(node, ast.If) and "__main__" in ast.dump(node.test):
            rep.has_main = True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = ast.dump(node)
            if "argparse" in text or "sys" in text and "argv" in src:
                rep.parses_argv = True
    if "argparse" in src or "sys.argv" in src or "parse_args" in src:
        rep.parses_argv = True
    rep.own_step_loop = bool(_LOOP_KERNELS & names) or ("def step" in src or "def _step" in src)
    # a thin main must actually import the shared runtime -- name collisions
    # (a local helper that happens to be called run_case) are not evidence
    rep.thin_main = harness_import

    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        span = _block_span(node)
        if span < _MIN_BLOCK_LINES:
            continue
        try:
            norm = _normalise(node)
        except Exception:  # pragma: no cover - exotic node
            continue
        rep.block_hashes.append(hashlib.sha256(norm.encode()).hexdigest()[:16])
        start = int(getattr(node, "lineno", 0))
        rep.block_spans.append((start, start + span - 1))
        rep.block_driver.append(_is_driver_block(node))
    return rep


def _is_driver_block(node: ast.AST) -> bool:
    """True if the block re-implements part of a solver driver.

    A duplicated *time loop* (or a duplicated streaming/collision call) is the
    expensive kind of copy: it drifts from the library silently.  A duplicated
    ``print``/``argparse`` stanza is noise.  Only the former is counted as a
    duplicated driver block.
    """
    for sub in ast.walk(node):
        if isinstance(sub, (ast.For, ast.While)):
            return True
        if isinstance(sub, ast.Call):
            f = sub.func
            name = getattr(f, "id", None) or getattr(f, "attr", None)
            if name in _LOOP_KERNELS:
                return True
    return False


def _iter_py(root: str, recursive: bool) -> Iterator[str]:
    if os.path.isfile(root):
        yield root
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)
        if not recursive:
            break


def audit_tree(root: str, recursive: bool = False) -> dict[str, Any]:
    """Audit ``root`` (a directory of example scripts, or a single file)."""
    files = [_scan_file(p) for p in _iter_py(root, recursive)]
    ok = [f for f in files if f.error is None]

    seen: dict[str, int] = {}
    for f in ok:
        for h in f.block_hashes:
            seen[h] = seen.get(h, 0) + 1
    dup_hashes = {h for h, n in seen.items() if n >= 2}

    dup_lines: set[tuple[str, int]] = set()
    dup_driver_lines: set[tuple[str, int]] = set()
    dup_blocks = 0
    dup_driver_blocks = 0
    for f in ok:
        for h, (lo, hi), drv in zip(f.block_hashes, f.block_spans, f.block_driver):
            if h not in dup_hashes:
                continue
            dup_blocks += 1
            for ln in range(lo, hi + 1):
                dup_lines.add((f.path, ln))
                if drv:
                    dup_driver_lines.add((f.path, ln))
            if drv:
                dup_driver_blocks += 1

    dup_loc = len(dup_lines)
    dup_driver_loc = len(dup_driver_lines)
    total_loc = sum(f.loc for f in ok) or 1
    report = {
        "root": os.path.abspath(root),
        "recursive": recursive,
        "python_files": len(files),
        "parsed_files": len(ok),
        "parse_errors": [{"path": f.path, "error": f.error} for f in files if f.error],
        "total_loc": total_loc,
        "M1_standalone_mains": {
            "n_files_with_main": sum(f.has_main for f in ok),
            "n_files_parsing_argv": sum(f.parses_argv for f in ok),
            "n_files_own_step_loop": sum(f.own_step_loop for f in ok),
            "share_with_main": round(sum(f.has_main for f in ok) / len(ok), 4) if ok else 0.0,
        },
        "M2_duplication": {
            "n_distinct_blocks": len(seen),
            "n_duplicated_blocks": len(dup_hashes),
            "n_dup_block_instances": dup_blocks,
            "dup_loc": dup_loc,
            "dup_loc_ratio": round(dup_loc / total_loc, 4),
            "n_dup_driver_blocks": dup_driver_blocks,
            "dup_driver_loc": dup_driver_loc,
            "dup_driver_loc_ratio": round(dup_driver_loc / total_loc, 4),
            "min_block_lines": _MIN_BLOCK_LINES,
        },
        "M3_thin_mains": {
            "n_files_delegating_to_harness": sum(f.thin_main for f in ok),
            "share": round(sum(f.thin_main for f in ok) / len(ok), 4) if ok else 0.0,
        },
        "files": [
            {
                "path": os.path.relpath(f.path, os.path.abspath(root) if os.path.isdir(root) else os.path.dirname(os.path.abspath(root))),
                "loc": f.loc,
                "has_main": f.has_main,
                "parses_argv": f.parses_argv,
                "own_step_loop": f.own_step_loop,
                "thin_main": f.thin_main,
                "n_blocks": len(f.block_hashes),
                "n_dup_blocks": sum(1 for h in f.block_hashes if h in dup_hashes),
            }
            for f in ok
        ],
    }
    return report


def format_summary(report: dict[str, Any]) -> str:
    m1 = report["M1_standalone_mains"]
    m2 = report["M2_duplication"]
    m3 = report["M3_thin_mains"]
    lines = [
        f"root={report['root']} files={report['parsed_files']}/{report['python_files']} loc={report['total_loc']}",
        "M1 standalone mains: "
        f"main={m1['n_files_with_main']} ({m1['share_with_main']:.1%}) "
        f"argv={m1['n_files_parsing_argv']} own_step_loop={m1['n_files_own_step_loop']}",
        "M2 duplication: "
        f"blocks={m2['n_distinct_blocks']} duplicated={m2['n_duplicated_blocks']} "
        f"dup_loc={m2['dup_loc']} ({m2['dup_loc_ratio']:.1%} of lines) "
        f"driver_blocks={m2['n_dup_driver_blocks']} "
        f"driver_loc={m2['dup_driver_loc']} ({m2['dup_driver_loc_ratio']:.1%})",
        "M3 thin mains: "
        f"{m3['n_files_delegating_to_harness']} ({m3['share']:.1%}) delegate to example_harness",
    ]
    if report["parse_errors"]:
        lines.append(f"parse errors: {len(report['parse_errors'])}")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m tensorlbm.audit example-antipattern",
        description="Measure the example-per-main anti-pattern in a tree of example scripts.",
    )
    ap.add_argument("--root", default="examples", help="directory (or file) to scan")
    ap.add_argument("--recursive", action="store_true", help="recurse into sub-directories")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument("--quiet", action="store_true")
    ns = ap.parse_args(list(argv) if argv is not None else None)

    report = audit_tree(ns.root, recursive=ns.recursive)
    if ns.out:
        os.makedirs(os.path.dirname(os.path.abspath(ns.out)), exist_ok=True)
        with open(ns.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    if not ns.quiet:
        print(format_summary(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())