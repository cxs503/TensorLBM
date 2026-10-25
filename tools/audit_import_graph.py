#!/usr/bin/env python3
"""Static AST audit of Python imports in selected repository layers.

Example:
  python tools/audit_import_graph.py --root src/tensorlbm=tensorlbm --root app/backend=app.backend
"""
from __future__ import annotations
import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

def parse_roots(values: list[str], repo: Path) -> list[tuple[Path, str]]:
    roots = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"root must be PATH=PACKAGE_PREFIX: {value!r}")
        path_text, prefix = value.split("=", 1)
        path = (repo / path_text).resolve()
        if not path.is_dir():
            raise ValueError(f"root directory does not exist: {path_text}")
        roots.append((path, prefix.strip(". ")))
    return roots

def module_for(path: Path, root: Path, prefix: str) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(part for part in (prefix, ".".join(parts)) if part)

def resolve_relative(current: str, is_package: bool, level: int, target: str | None) -> str | None:
    package = current.split(".") if is_package else current.split(".")[:-1]
    remove = level - 1
    if remove > len(package):
        return None
    base = package[:len(package) - remove]
    if target:
        base.extend(target.split("."))
    return ".".join(base) or None

def literal_module(call: ast.Call) -> str | None:
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    return None

def scan(roots: list[tuple[Path, str]]) -> dict[str, Any]:
    files: dict[str, Path] = {}
    file_modules: dict[Path, str] = {}
    for root, prefix in roots:
        for path in sorted(root.rglob("*.py")):
            module = module_for(path, root, prefix)
            files[module] = path
            file_modules[path] = module
    known = set(files)
    for name in list(known):
        parts = name.split(".")
        for i in range(1, len(parts)):
            known.add(".".join(parts[:i]))
    edges: set[tuple[str, str]] = set()
    imports: list[dict[str, Any]] = []
    dynamic: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def record(source: str, target: str, line: int, kind: str, raw: str) -> None:
        internal = target in known or any(name.startswith(target + ".") for name in known)
        imports.append({"source": source, "target": target, "line": line, "kind": kind,
                        "raw": raw, "internal": internal})
        if internal and target in files:
            edges.add((source, target))

    for path, source in sorted(file_modules.items(), key=lambda item: item[1]):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            errors.append({"module": source, "path": str(path), "error": str(exc)})
            continue
        is_package = path.name == "__init__.py"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    record(source, alias.name, node.lineno, "import", alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    target = resolve_relative(source, is_package, node.level, node.module)
                    if target:
                        record(source, target, node.lineno, "from-relative", "." * node.level + (node.module or ""))
                        for alias in node.names:
                            child = target + "." + alias.name
                            if alias.name != "*" and child in files:
                                record(source, child, node.lineno, "from-relative-member", "." * node.level + (node.module or "") + ":" + alias.name)
                    else:
                        imports.append({"source": source, "target": None, "line": node.lineno,
                                        "kind": "from-relative-unresolved", "raw": "." * node.level + (node.module or ""),
                                        "internal": None})
                elif node.module:
                    record(source, node.module, node.lineno, "from", node.module)
                    for alias in node.names:
                        child = node.module + "." + alias.name
                        if alias.name != "*" and child in files:
                            record(source, child, node.lineno, "from-member", node.module + ":" + alias.name)
            elif isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else (
                    f"{func.value.id}.{func.attr}" if isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name) else "")
                if name in {"__import__", "importlib.import_module", "import_module",
                            "importlib.util.find_spec", "find_spec"}:
                    target = literal_module(node)
                    dynamic.append({"source": source, "line": node.lineno, "call": name,
                                    "literal_module": target,
                                    "status": "literal-target" if target else "dynamic-target-unknown"})
                    if target:
                        record(source, target, node.lineno, "dynamic-literal", target)

    adjacency: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        adjacency[source].add(target)
    index = 0
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    cycles: list[list[str]] = []
    def visit(node: str) -> None:
        nonlocal index
        indices[node] = low[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for nxt in sorted(adjacency[node]):
            if nxt not in indices:
                visit(nxt)
                low[node] = min(low[node], low[nxt])
            elif nxt in on_stack:
                low[node] = min(low[node], indices[nxt])
        if low[node] == indices[node]:
            component = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1 or node in adjacency[node]:
                cycles.append(sorted(component))
    for module in sorted(files):
        if module not in indices:
            visit(module)
    cross_layer = []
    for source, target in sorted(edges):
        sl = next((p for _, p in roots if source == p or source.startswith(p + ".")), "other")
        tl = next((p for _, p in roots if target == p or target.startswith(p + ".")), "other")
        if sl != tl:
            cross_layer.append({"source": source, "target": target, "source_layer": sl, "target_layer": tl})
    return {"schema_version": 1, "scope": [{"path": str(p), "module_prefix": x} for p, x in roots],
            "module_count": len(files),
            "modules": [{"module": n, "path": str(p)} for n, p in sorted(files.items())],
            "edges": [{"source": s, "target": t} for s, t in sorted(edges)],
            "imports": sorted(imports, key=lambda x: (x["source"], x["line"], x["target"] or "")),
            "dynamic_imports": sorted(dynamic, key=lambda x: (x["source"], x["line"])),
            "cycles": sorted(cycles), "cross_layer_edges": cross_layer, "parse_errors": errors,
            "notes": ["Only statically resolved imports between discovered modules become graph edges.",
                      "Dynamic imports are uncertainty signals, not proof of runtime success.",
                      "Namespace package prefixes are treated as internal; third-party resolution is not attempted.",
                      "Static analysis alone does not prove runtime importability or boundary correctness."]}

def markdown_report(data: dict[str, Any]) -> str:
    lines = ["# Static Python Import Audit", "",
             f"- Discovered modules: **{data['module_count']}**",
             f"- Internal module edges: **{len(data['edges'])}**",
             f"- Dynamic import sites: **{len(data['dynamic_imports'])}**",
             f"- Import cycles: **{len(data['cycles'])}**",
             f"- Cross-layer edges: **{len(data['cross_layer_edges'])}**",
             f"- Parse/read errors: **{len(data['parse_errors'])}**", "",
             "> Static evidence only. Dynamic imports and runtime behavior need separate validation.", "",
             "## Import cycles", ""]
    lines.extend(["- " + " <-> ".join(c) for c in data["cycles"]] or ["- None detected."])
    lines += ["", "## Cross-layer edges", ""]
    lines.extend([f"- {e['source']} -> {e['target']} ({e['source_layer']} -> {e['target_layer']})"
                  for e in data["cross_layer_edges"]] or ["- None detected."])
    lines += ["", "## Dynamic imports", ""]
    lines.extend([f"- {d['source']}:{d['line']} {d['call']} target={d['literal_module'] or 'unknown'} ({d['status']})"
                  for d in data["dynamic_imports"]] or ["- None detected."])
    if data["parse_errors"]:
        lines += ["", "## Parse/read errors", ""]
        lines.extend([f"- {e['path']}: {e['error']}" for e in data["parse_errors"]])
    lines += ["", "## Scope", ""]
    lines.extend([f"- {x['path']} -> {x['module_prefix']}" for x in data["scope"]])
    return "\n".join(lines) + "\n"

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--root", action="append", default=[], metavar="PATH=PREFIX")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    specs = args.root or ["src/tensorlbm=tensorlbm", "app/backend=app.backend",
                          "tests=tests", "app/tests=app.tests"]
    data = scan(parse_roots(specs, repo))
    rendered = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        out = args.json_out if args.json_out.is_absolute() else repo / args.json_out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
    summary = markdown_report(data)
    if args.markdown_out:
        out = args.markdown_out if args.markdown_out.is_absolute() else repo / args.markdown_out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(summary, encoding="utf-8")
    print(summary, end="")
    return 1 if data["parse_errors"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
