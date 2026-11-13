"""Structural cross-validation of the "example-per-main" antipattern (XVAL).

Motivation (anchor XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN)
--------------------------------------------------------
The OSS inventory lists ``sorush-khajepor/listLBM`` as an example of the
*example-per-main* antipattern: every example is a standalone program unit
with its own ``program main``/``if __name__ == "__main__"`` block, so the
shared parts (license header, I/O scaffolding, parameter block, dispatch)
are copied per example and the choice of *which* example runs is a build-file
edit rather than a command-line argument.

This module measures that signature **mechanically** on any source tree, so a
claim such as "TensorLBM does / does not repeat the antipattern" is backed by
numbers instead of by reading a few files.  Nothing here imports the examples:
all inspection is static (AST for Python, regex on declarations for Fortran).

Metrics (all deterministic, no network, no solver execution)
------------------------------------------------------------
``n_files``               example source files considered.
``n_entry_units``         files carrying their own executable entry point.
``entry_ratio``           n_entry_units / n_files  (1.0 == every example is
                          its own program unit, the antipattern by construction).
``nonblank_lines``        physical non-blank lines across those files.
``dup_line_ratio``        share of non-blank lines that belong to a ``block_len``
                          -line n-gram occurring verbatim in >= 2 files; this is
                          the copier-output score (license headers, boilerplate).
``cli_units`` / ``cli_adoption``  example units that parse command-line options
                          (argparse in Python, ``get_command_argument`` in
                          Fortran) -> parameters reachable without editing code.
``hardcoded_param_ratio`` Fortran only: share of integer/real declarations with a
                          literal initialiser (``integer :: xlen=400``), i.e.
                          parameters baked into the source.
``import_side_effect_ratio`` Python only: share of units that execute calls at
                          module import time (not defs/classes/constants),
                          i.e. work that cannot be composed.
``registry_files`` / ``dispatcher_scripts``  programmatic example index / single
                          dispatch entry present in the tree.
``build_selector``        build-file lines that pick *one* example (evidence of
                          "edit the build to change the case").

Usage
-----
    python tools/xval/example_main_audit.py --root examples --json
    python tools/xval/example_main_audit.py --compare-tree /path/to/listLBM
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PY_SUFFIXES = {".py"}
F95_SUFFIXES = {".f90", ".f95", ".f03", ".F90"}

# Candidate files that would make the example set *programmatically* listable.
REGISTRY_CANDIDATES = (
    "examples/index.json",
    "examples/registry.json",
    "examples/REGISTRY.md",
    "examples/INDEX.md",
    "examples/manifest.json",
)
DISPATCHER_CANDIDATES = (
    "tools/example_runner.py",
    "tools/xval/example_runner.py",
    "examples/run.py",
    "examples/run_example.py",
    "examples/dispatch.py",
)
BUILD_CANDIDATES = ("CMakeLists.txt", "Makefile", "makefile", "pyproject.toml", "setup.cfg")

_NONBLANK = re.compile(r"\S")
_F95_PARAM_DECL = re.compile(
    r"::\s*(?:[A-Za-z_][\w(),:=\s]*?)", re.IGNORECASE
)
_F95_LITERAL_INIT = re.compile(
    r"::[^!\n]*?[A-Za-z_]\w*\s*(?:\([^)\n]*\))?\s*=\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[deE][-+]?\d+)?_?\w*"
)
_F95_PROGRAM = re.compile(r"^\s*program\s+([A-Za-z_]\w*)", re.IGNORECASE | re.MULTILINE)
_F95_CLI = re.compile(
    r"get_command_argument|command_argument_count|iargc|getarg", re.IGNORECASE
)
_F95_PARAMETERS_STMT = re.compile(r"::\s*[A-Za-z_]|\bparameter\b", re.IGNORECASE)
_BUILD_SELECTOR = re.compile(
    r"(example_path|EXAMPLE_PATH|EXAMPLES?\s*=|--example\b|example\s*:=)", re.MULTILINE
)


# --------------------------------------------------------------------------- #
# per-file scanners
# --------------------------------------------------------------------------- #
@dataclass
class FileInfo:
    """Static facts about one candidate example file."""

    path: str
    suffix: str
    nonblank_lines: int
    has_entry: bool = False
    entry_kind: str = ""
    has_cli: bool = False
    cli_flags: list[str] = field(default_factory=list)
    import_side_effects: int = 0
    module_statements: int = 0
    hardcoded_params: int = 0
    param_declarations: int = 0
    summary: str = ""
    sha256: str = ""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _nonblank_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if _NONBLANK.search(line))


def _python_file_info(path: Path, text: str) -> FileInfo:
    info = FileInfo(
        path=str(path),
        suffix=path.suffix,
        nonblank_lines=_nonblank_count(text),
        sha256=hashlib.sha256(text.encode("utf-8", "replace")).hexdigest(),
    )
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return info

    doc = ast.get_docstring(tree) or ""
    info.summary = doc.strip().splitlines()[0].strip() if doc.strip() else ""

    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_main_guard(node):
            info.has_entry = True
            info.entry_kind = "main_guard"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "argparse":
                    info.has_cli = True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "argparse":
            info.has_cli = True
        if isinstance(node, ast.Attribute) and node.attr in {"argv", "parse_args", "parse_known_args"}:
            info.has_cli = True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"add_argument", "add_argument_group"}:
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if arg.value.startswith("-"):
                            info.cli_flags.append(arg.value)
    info.cli_flags = sorted(set(info.cli_flags))

    # Module-level executable statements (work done on import).
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Assign, ast.AnnAssign, ast.Expr)):
            if isinstance(stmt, ast.Expr) and _is_docstring(stmt):
                continue
            info.module_statements += 1
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                info.import_side_effects += 1
            continue
        if isinstance(stmt, ast.If) and _is_main_guard(stmt):
            continue
        info.module_statements += 1
    return info


def _is_docstring(stmt: ast.stmt) -> bool:
    return isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    left, right = test.left, test.comparators[0]
    names = []
    for side in (left, right):
        if isinstance(side, ast.Name):
            names.append(side.id)
        elif isinstance(side, ast.Constant) and isinstance(side.value, str):
            names.append(side.value)
    return "__name__" in names and "__main__" in names


def python_file_info(path: Path) -> dict[str, object]:
    """Public single-file entry point for the Python static facts."""
    path = Path(path)
    return asdict(_python_file_info(path, _read(path)))


def fortran_file_info(path: Path) -> dict[str, object]:
    """Public single-file entry point for the Fortran static facts."""
    path = Path(path)
    return asdict(_fortran_file_info(path, _read(path)))


def _fortran_file_info(path: Path, text: str) -> FileInfo:
    info = FileInfo(
        path=str(path),
        suffix=path.suffix,
        nonblank_lines=_nonblank_count(text),
        sha256=hashlib.sha256(text.encode("utf-8", "replace")).hexdigest(),
    )
    match = _F95_PROGRAM.search(text)
    if match:
        info.has_entry = True
        info.entry_kind = f"program {match.group(1)}"
    if _F95_CLI.search(text):
        info.has_cli = True
    # parameter declarations with a literal initialiser, e.g. integer :: xlen=400
    decl_lines = [
        line for line in text.splitlines()
        if _F95_PARAMETERS_STMT.search(line) and "::" in line
        and not line.lstrip().startswith("!")
    ]
    info.param_declarations = len(decl_lines)
    info.hardcoded_params = sum(1 for line in decl_lines if _F95_LITERAL_INIT.search(line))
    # first comment line that is not part of the licence block
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("!") and "License" not in stripped and "Copyright" not in stripped:
            info.summary = stripped.lstrip("! ").strip()
            break
    return info


# --------------------------------------------------------------------------- #
# tree scanner
# --------------------------------------------------------------------------- #
def _iter_files(root: Path, suffixes: set[str]) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes)


def duplicate_line_ratio(texts: list[str], block_len: int = 3) -> tuple[float, int, int]:
    """Share of non-blank lines inside an n-gram shared by >= 2 files.

    Returns ``(ratio, dup_lines, nonblank_lines)``.  A line counts once even if
    several shared n-grams cover it.
    """
    if block_len < 1:
        raise ValueError("block_len must be >= 1")
    grams: dict[tuple[str, ...], int] = {}
    per_file: list[list[str]] = []
    for text in texts:
        lines = [line.strip() for line in text.splitlines() if _NONBLANK.search(line)]
        per_file.append(lines)
        seen: set[tuple[str, ...]] = set()
        for i in range(len(lines) - block_len + 1):
            gram = tuple(lines[i:i + block_len])
            if gram not in seen:
                seen.add(gram)
                grams[gram] = grams.get(gram, 0) + 1
    shared = {gram for gram, df in grams.items() if df >= 2}
    total = 0
    dup = 0
    for lines in per_file:
        covered = [False] * len(lines)
        for i in range(len(lines) - block_len + 1):
            if tuple(lines[i:i + block_len]) in shared:
                for j in range(i, i + block_len):
                    covered[j] = True
        total += len(lines)
        dup += sum(covered)
    return (dup / total if total else 0.0), dup, total


def _build_selectors(root: Path) -> list[dict[str, object]]:
    """Build-file lines that pick *one* example (listLBM: ``set(example_path ...)``).

    Build files usually live at the project root, not inside the example
    directory, so the two parent levels above ``root`` are searched too.
    """
    found: list[dict[str, object]] = []
    seen: set[Path] = set()
    searches: list[tuple[Path, list[Path]]] = []
    for depth, base in enumerate([root, root.parent, root.parent.parent]):
        cands: list[Path] = []
        for name in BUILD_CANDIDATES:
            # rglob only inside the audited tree; ancestor levels cost a directory
            # walk over sibling checkouts and yield nothing but latency.
            cands.extend(sorted(base.rglob(name)) if depth == 0 else [base / name])
        searches.append((base, cands))
    for base, candidates in searches:
        if True:
            for path in candidates:
                if not path.is_file() or path in seen:
                    continue
                seen.add(path)
                text = _read(path)
                hits = _BUILD_SELECTOR.findall(text)
                if not hits:
                    continue
                try:
                    rel = str(path.relative_to(base))
                except ValueError:  # pragma: no cover - defensive
                    rel = str(path)
                found.append(
                    {
                        "path": rel,
                        "matches": len(hits),
                        "distinct_example_refs": len(set(re.findall(r"examples/[\w./-]+", text))),
                    }
                )
    return found


def scan_tree(
    root: Path,
    *,
    block_len: int = 3,
    suffixes: set[str] | None = None,
    relative_to: Path | None = None,
    ignore: tuple[str, ...] = (),
) -> dict[str, object]:
    """Static example-per-main metrics for one source tree."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    rel_base = Path(relative_to) if relative_to else root
    suffixes = suffixes or (PY_SUFFIXES | F95_SUFFIXES)
    files = _iter_files(root, suffixes)
    infos: list[FileInfo] = []
    texts: list[str] = []
    for path in files:
        text = _read(path)
        texts.append(text)
        info = (_python_file_info if path.suffix in PY_SUFFIXES else _fortran_file_info)(path, text)
        info.path = str(_rel(path, rel_base))
        infos.append(info)

    entry_units = [i for i in infos if i.has_entry]
    cli_units = [i for i in entry_units if i.has_cli]
    dup_ratio, dup_lines, nonblank = duplicate_line_ratio(texts, block_len=block_len)

    py_units = [i for i in infos if i.suffix in PY_SUFFIXES and i.has_entry]
    f95_units = [i for i in infos if i.suffix in F95_SUFFIXES]
    hardcoded_ratio = None
    if f95_units:
        decls = sum(i.param_declarations for i in f95_units)
        hard = sum(i.hardcoded_params for i in f95_units)
        hardcoded_ratio = hard / decls if decls else 0.0

    ignored = {item.replace("\\", "/") for item in ignore}
    registry = [
        c for c in REGISTRY_CANDIDATES
        if c not in ignored and ((Path(rel_base) / c).is_file() or (root / c).is_file())
    ]
    registry += [str(p) for p in sorted(root.rglob("registry*.json")) if p.is_file()][:3]
    dispatchers = [c for c in DISPATCHER_CANDIDATES
                   if c not in ignored and _exists_upwards(Path(rel_base), c)]

    return {
        "root": str(root),
        "block_len": block_len,
        "n_files": len(infos),
        "n_entry_units": len(entry_units),
        "entry_ratio": len(entry_units) / len(infos) if infos else 0.0,
        "entry_kinds": sorted({i.entry_kind.split()[0] for i in entry_units}),
        "nonblank_lines": nonblank,
        "dup_lines": dup_lines,
        "dup_line_ratio": dup_ratio,
        "cli_units": len(cli_units),
        "cli_adoption": len(cli_units) / len(entry_units) if entry_units else 0.0,
        "hardcoded_param_ratio": hardcoded_ratio,
        "import_side_effect_ratio": (
            sum(1 for i in py_units if i.import_side_effects) / len(py_units) if py_units else None
        ),
        "registry_files": sorted(set(registry)),
        "dispatcher_scripts": sorted(set(dispatchers)),
        "has_programmatic_index": bool(registry or dispatchers),
        "build_selectors": _build_selectors(root),
        "units": [asdict(i) for i in infos],
    }


def _rel(path: Path, base: Path) -> Path:
    try:
        return path.resolve().relative_to(base.resolve())
    except ValueError:
        return path


def _exists_upwards(base: Path, rel: str) -> bool:
    for parent in [base, *base.parents]:
        if (parent / rel).is_file():
            return True
    return False


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _summary_line(m: dict[str, object]) -> str:
    return (
        f"{m['root']}: files={m['n_files']} entries={m['n_entry_units']} "
        f"entry_ratio={m['entry_ratio']:.3f} dup_line_ratio={m['dup_line_ratio']:.3f} "
        f"cli_adoption={m['cli_adoption']:.3f} index={m['has_programmatic_index']}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(REPO_ROOT / "examples"),
                    help="tree to audit (default: this repo's examples/)")
    ap.add_argument("--compare-tree", action="append", default=[],
                    help="additional tree to audit with the same metrics (repeatable)")
    ap.add_argument("--block-len", type=int, default=3, help="boilerplate n-gram length")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--units", action="store_true", help="include per-file units in JSON")
    ap.add_argument("--out", default=None, help="write JSON report to this path")
    args = ap.parse_args(argv)

    roots = [Path(args.root), *[Path(p) for p in args.compare_tree]]
    reports = [scan_tree(r, block_len=args.block_len) for r in roots]
    for report in reports:
        if not args.units:
            report.pop("units")
        print(_summary_line(report) if not args.json else "")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(reports, indent=2, default=str) + "\n")
    if args.json:
        print(json.dumps(reports, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())