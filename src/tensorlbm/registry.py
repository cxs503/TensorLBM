"""Microservice-style capability registry for TensorLBM.

Motivation
----------
TensorLBM advertises a lot of numerics (collision kernels, multiphase, free
surface, acoustics, icing, AMR, backends, force sampling, differentiable
calibration).  Historically the inventory lived in prose (README tables,
``docs/*capability*`` documents, ad-hoc JSON contracts).  Prose cannot be
checked, so capabilities drifted: a kernel could be renamed, mocked, or never
wired into a solver while the document kept claiming it.

This module turns the inventory into code that registers itself at import time,
the same way a FastAPI router registers its routes:

.. code-block:: python

    from .registry import capability

    @capability(
        "lbm.collision.bgk.d3q19",
        kind="collision",
        maturity="mature",
        wired=True,
        tested=True,
        verified=True,
        test_path="tests/test_lbm_verification.py",
        evidence={"grade": "MECHANICS", "note": "..."},
    )
    def collide_bgk3d(f, tau):
        ...

Nothing else is needed: importing the module populates the process-wide
:data:`REGISTRY` singleton.  The platform (TensorPlatform ``8010``) exposes
``/api/v1/tensor/capabilities`` and accepts a dump at ``/sync``; its own gate
rejects ``maturity="mature"`` unless ``verified`` *and* ``wired`` are both true,
so the same discipline is enforced here at registration time
(:class:`MaturityDisciplineError`) — a false "mature" fails at import instead of
silently reaching the dashboard.

Two dump modes (the CLI contract used by TensorPlatform's ``capability_sync``
daemon is *stdout is one JSON object* with at least ``source_commit`` and
``capabilities``)::

    # import mode: needs the environment (torch), sees runtime state
    python -m tensorlbm.registry

    # static AST mode: imports nothing from the package, needs no torch
    python src/tensorlbm/registry.py --static

The static mode is deliberately runnable *by file path*: ``python -m
tensorlbm.registry`` imports the package ``__init__`` (which imports torch), so
a torch-free host must call the file directly.  The module itself only imports
the standard library, and :mod:`tensorlbm.registry` is therefore safe for
modules deep in the package to import.

Honesty rules encoded here (see :func:`validate_capability`):

``mature``
    requires ``wired and verified`` (mirrors the server-side 422 gate) and is
    *warned about* when ``tested``/``test_path`` are missing.
``planned``
    requires an evidence anchor naming the roadmap row
    (``evidence["feature_ledger"]``), because "we plan to build it" is only
    meaningful if it is booked somewhere.
``endpoint``
    may only be declared by a wired capability, and must use a recognised
    scheme (``cli:`` / ``route:`` / ``api:``), because TensorPhysics consumes it
    as a callable entry point.

Pointer resolution (both modes): every ``module_path``, ``entry`` and
``test_path`` is resolved against the repository.  A capability that names a
module, function or test file that does not exist is reported as a *problem*,
not silently dropped — that is exactly the drift the old prose documents could
not detect.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping

__all__ = [
    "CAPABILITY_KINDS",
    "Capability",
    "DuplicateCapabilityError",
    "MATURITIES",
    "MaturityDisciplineError",
    "REGISTRY",
    "Registry",
    "capability",
    "collect_problems",
    "dump",
    "main",
    "scan_static",
    "validate_capability",
]

# --------------------------------------------------------------------------- #
# Vocabularies
# --------------------------------------------------------------------------- #

#: Maturity ladder.  Mirrors the server constraint
#: ``ck_capreg_maturity CHECK (maturity = ANY ('mature','developing','planned'))``.
MATURITIES: tuple[str, ...] = ("mature", "developing", "planned")

#: Coarse capability families.  Free-form strings are allowed (the platform
#: stores ``kind`` as text and indexes it), but these are the names in use so a
#: dashboard can group without string archaeology.
CAPABILITY_KINDS: tuple[str, ...] = (
    "collision",
    "boundary",
    "multiphase",
    "free_surface",
    "acoustics",
    "icing",
    "amr",
    "backend",
    "force",
    "differentiable",
    "turbulence",
    "solver",
    "module",
)

#: Evidence grades used in ``evidence["grade"]``.  New values must be added here
#: so the audit script can reason about them; ``MOCK``/``NONE`` can never back a
#: ``verified`` claim.
EVIDENCE_GRADES: tuple[str, ...] = (
    "ANALYTIC",     # closed-form / manufactured solution
    "XVALUE",       # external reference values (benchmark, experiment, literature)
    "MECHANICS",    # conservation/symmetry/limit test only
    "MOCK",         # hard-coded or random stand-in
    "NONE",         # no evidence
)

#: Grades strong enough to justify ``verified=True``.
VERIFYING_GRADES: tuple[str, ...] = ("ANALYTIC", "XVALUE")

ENDPOINT_SCHEMES: tuple[str, ...] = ("cli:", "route:", "api:")

PKG_NAME = "tensorlbm"


class MaturityDisciplineError(ValueError):
    """A capability claims more maturity than its flags and evidence support."""


class DuplicateCapabilityError(RuntimeError):
    """Two registrations used the same capability id."""


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Capability:
    """One registry row.

    Field names are identical to the platform payload (``tensor_capability_registry``
    columns) so a dump can be POSTed to ``/tensor/capabilities/sync`` unchanged.
    """

    id: str
    kind: str = "module"
    maturity: str = "developing"
    module_path: str = ""
    entry: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    wired: bool = False
    tested: bool = False
    verified: bool = False
    test_path: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    endpoint: str | None = None

    # provenance, not sent to the platform (the platform records its own
    # source_commit for the whole sync)
    source: str | None = None

    def requalify(self, **changes: Any) -> "Capability":
        """Return a copy with ``changes`` applied (used by the static scanner)."""
        data = self.to_payload()
        data.update(changes)
        data.pop("source", None)
        return Capability(source=self.source, **data)

    def dotted(self) -> str:
        """``module_path.entry`` — what the drift auditor resolves."""
        return f"{self.module_path}.{self.entry}" if self.entry else self.module_path

    def to_payload(self) -> dict[str, Any]:
        """JSON payload exactly as the platform expects it."""
        return {
            "id": self.id,
            "kind": self.kind,
            "maturity": self.maturity,
            "module_path": self.module_path,
            "entry": self.entry,
            "params": dict(self.params),
            "wired": bool(self.wired),
            "tested": bool(self.tested),
            "verified": bool(self.verified),
            "test_path": self.test_path,
            "evidence": dict(self.evidence),
            "endpoint": self.endpoint,
        }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_capability(cap: Capability) -> list[dict[str, Any]]:
    """Return problems with one capability (empty list == acceptable).

    ``error`` problems block registration; ``warning`` problems are disclosed.
    The rules are the local mirror of the platform gate plus the honesty rules
    stated in the module docstring.
    """
    problems: list[dict[str, Any]] = []

    def add(severity: str, code: str, detail: str) -> None:
        problems.append(
            {"severity": severity, "code": code, "capability_id": cap.id,
             "detail": detail, "source": cap.source}
        )

    if not cap.id or len(cap.id) < 2:
        add("error", "ID_TOO_SHORT", "capability id must have at least 2 characters")
    if "." not in cap.id:
        add("warning", "ID_NOT_NAMESPACED", "capability id has no '.' namespace component")
    if cap.maturity not in MATURITIES:
        add("error", "UNKNOWN_MATURITY", f"maturity={cap.maturity!r} not in {MATURITIES}")
    if not cap.module_path:
        add("error", "MODULE_PATH_MISSING", "module_path is required")
    if not cap.entry:
        add("warning", "ENTRY_MISSING", "no entry point named; capability is a bare module claim")

    grade = str(cap.evidence.get("grade", "") or "")
    if grade and grade not in EVIDENCE_GRADES:
        add("error", "UNKNOWN_EVIDENCE_GRADE",
            f"evidence.grade={grade!r} not in {EVIDENCE_GRADES}")

    # server-side gate mirrored locally: mature => verified and wired
    if cap.maturity == "mature":
        if not cap.wired:
            add("error", "MATURE_NOT_WIRED",
                "maturity='mature' requires wired=True (would be rejected with 422)")
        if not cap.verified:
            add("error", "MATURE_NOT_VERIFIED",
                "maturity='mature' requires verified=True (would be rejected with 422)")
        if not cap.tested:
            add("warning", "MATURE_NOT_TESTED", "maturity='mature' without tested=True")
        if not cap.test_path:
            add("warning", "MATURE_NO_TEST_PATH", "maturity='mature' without test_path")
    elif cap.maturity == "planned":
        ledger = cap.evidence.get("feature_ledger")
        if not isinstance(ledger, str) or not ledger.strip():
            add("error", "PLANNED_WITHOUT_LEDGER",
                "maturity='planned' requires evidence.feature_ledger (roadmap anchor)")
        if cap.verified:
            add("error", "PLANNED_BUT_VERIFIED",
                "maturity='planned' cannot be verified=True; promote to mature instead")

    # verified must rest on a verifying grade when a grade is declared
    if cap.verified and grade and grade not in VERIFYING_GRADES:
        add("error", "VERIFIED_ON_WEAK_EVIDENCE",
            f"verified=True with evidence.grade={grade!r} (needs one of {VERIFYING_GRADES})")
    if cap.verified and grade in VERIFYING_GRADES and "max_error_pct" not in cap.evidence:
        add("warning", "VERIFIED_WITHOUT_MAX_ERROR",
            "verified=True but evidence.max_error_pct absent — error band undisclosed")

    # endpoint is a callable contract for TensorPhysics
    if cap.endpoint:
        if not any(cap.endpoint.startswith(s) for s in ENDPOINT_SCHEMES):
            add("error", "ENDPOINT_BAD_SCHEME",
                f"endpoint={cap.endpoint!r} must start with one of {ENDPOINT_SCHEMES}")
        if not cap.wired:
            add("error", "ENDPOINT_WITHOUT_WIRING",
                "endpoint declared for a capability that is not wired")
    elif cap.wired and cap.kind == "solver":
        add("warning", "WIRED_SOLVER_WITHOUT_ENDPOINT",
            "wired solver capability without an endpoint TensorPhysics could call")

    if cap.tested and not cap.test_path:
        add("warning", "TESTED_WITHOUT_TEST_PATH", "tested=True but no test_path")
    return problems


# --------------------------------------------------------------------------- #
# Registry singleton
# --------------------------------------------------------------------------- #


class Registry:
    """Process-wide capability table (microservice self-registration target)."""

    def __init__(self) -> None:
        self._caps: MutableMapping[str, Capability] = {}
        self.problems: list[dict[str, Any]] = []
        self.skipped: list[dict[str, Any]] = []

    # -- registration ------------------------------------------------------ #
    def register(self, cap: Capability, *, strict: bool = True) -> Capability:
        problems = validate_capability(cap)
        self.problems.extend(problems)
        blocking = [p for p in problems if p["severity"] == "error"]
        if blocking:
            codes = ", ".join(sorted({p["code"] for p in blocking}))
            if strict:
                raise MaturityDisciplineError(
                    f"capability {cap.id!r} violates registry discipline: {codes}"
                )
            self.skipped.append({"id": cap.id, "codes": sorted({p["code"] for p in blocking})})
            return cap
        if cap.id in self._caps:
            raise DuplicateCapabilityError(
                f"capability id {cap.id!r} registered twice "
                f"({self._caps[cap.id].source} and {cap.source})"
            )
        self._caps[cap.id] = cap
        return cap

    # -- inspection -------------------------------------------------------- #
    def __contains__(self, cid: object) -> bool:
        return cid in self._caps

    def __len__(self) -> int:
        return len(self._caps)

    def __iter__(self) -> Iterator[Capability]:
        return iter(self.capabilities())

    def get(self, cid: str) -> Capability:
        return self._caps[cid]

    def capabilities(self) -> list[Capability]:
        return [self._caps[k] for k in sorted(self._caps)]

    def ids(self) -> list[str]:
        return sorted(self._caps)

    def clear(self) -> None:
        """Drop all registrations (test helper; production code never calls it)."""
        self._caps.clear()
        self.problems.clear()
        self.skipped.clear()

    def summary(self) -> dict[str, Any]:
        by_maturity = {m: 0 for m in MATURITIES}
        by_kind: dict[str, int] = {}
        for cap in self._caps.values():
            by_maturity[cap.maturity] = by_maturity.get(cap.maturity, 0) + 1
            by_kind[cap.kind] = by_kind.get(cap.kind, 0) + 1
        return {
            "count": len(self._caps),
            "by_maturity": by_maturity,
            "by_kind": dict(sorted(by_kind.items())),
            "wired": sum(1 for c in self._caps.values() if c.wired),
            "tested": sum(1 for c in self._caps.values() if c.tested),
            "verified": sum(1 for c in self._caps.values() if c.verified),
            "with_endpoint": sum(1 for c in self._caps.values() if c.endpoint),
        }

    def dump(self, *, source_commit: str | None = None, mode: str = "import",
             repo: str = "TensorLBM") -> dict[str, Any]:
        capabilities = [c.to_payload() for c in self.capabilities()]
        return {
            "repo": repo,
            "mode": mode,
            "source_commit": source_commit or "unknown",
            "count": len(capabilities),
            "summary": self.summary(),
            "capabilities": capabilities,
            "problems": list(self.problems),
        }


#: The singleton every ``@capability`` decorator feeds.
REGISTRY = Registry()


def capability(
    id: str,
    kind: str = "module",
    maturity: str = "developing",
    module_path: str | None = None,
    entry: str | None = None,
    params: Mapping[str, Any] | None = None,
    wired: bool = False,
    tested: bool = False,
    verified: bool = False,
    test_path: str | None = None,
    evidence: Mapping[str, Any] | None = None,
    endpoint: str | None = None,
    *,
    registry: Registry | None = None,
    strict: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register the decorated function/class as a capability at import time.

    ``module_path`` and ``entry`` default to the *decorated object's* module and
    qualified name, so the common case needs no boilerplate — and the static AST
    scanner derives exactly the same two values from the source file, which makes
    import-mode and static-mode dumps comparable.
    """
    target = registry if registry is not None else REGISTRY

    def decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
        mod = module_path or getattr(obj, "__module__", "") or ""
        name = entry or getattr(obj, "__name__", "") or ""
        cap = Capability(
            id=id,
            kind=kind,
            maturity=maturity,
            module_path=mod,
            entry=name or None,
            params=dict(params or {}),
            wired=bool(wired),
            tested=bool(tested),
            verified=bool(verified),
            test_path=test_path,
            evidence=dict(evidence or {}),
            endpoint=endpoint,
            source=f"{mod}.{name}",
        )
        target.register(cap, strict=strict)
        try:
            setattr(obj, "__capability__", cap)
        except (AttributeError, TypeError):  # exotic callables
            pass
        return obj

    return decorator


# --------------------------------------------------------------------------- #
# Static AST scanning (no import, no torch)
# --------------------------------------------------------------------------- #

_ROLE_SEVERITY = {"error": "error", "warning": "warning"}


def _repo_root(start: Path) -> Path:
    """Nearest ancestor holding ``pyproject.toml`` / ``.git`` (else ``start``)."""
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return start


def _git_commit(root: Path) -> str:
    env = os.environ.get("TENSOR_SOURCE_COMMIT") or os.environ.get("GIT_COMMIT")
    if env:
        return env.strip()
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _module_file(pkg_root: Path, module_path: str, package: str) -> Path | None:
    """``tensorlbm.a.b`` -> ``<pkg_root>/tensorlbm/a/b.py`` (or ``b/__init__.py``)."""
    parts = module_path.split(".")
    if not parts or parts[0] != package:
        # a foreign module path cannot be resolved inside this repository
        return None
    rel = parts[1:]
    base = pkg_root.joinpath(*rel) if rel else pkg_root
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _defined_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _class_members(tree: ast.AST) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ClassDef):
            out[node.name] = {
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return out


def _resolve_entry(tree: ast.AST, entry: str) -> bool:
    names = _defined_names(tree)
    head, _, rest = entry.partition(".")
    if head not in names:
        return False
    if not rest:
        return True
    members = _class_members(tree).get(head)
    return bool(members) and rest.split(".")[0] in members


def _decorator_kwargs(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
                      ) -> list[tuple[dict[str, Any], int, list[str]]]:
    """Literal ``@capability(...)`` arguments of one def/class node.

    Returns ``(kwargs, lineno, nonliteral_arg_names)``.
    """
    found: list[tuple[dict[str, Any], int, list[str]]] = []
    for dec in node.decorator_list:
        func = dec.func if isinstance(dec, ast.Call) else dec
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "capability":
            continue
        if not isinstance(dec, ast.Call):
            continue
        kwargs: dict[str, Any] = {}
        nonliteral: list[str] = []
        for arg in dec.args:
            nonliteral.append("<positional>")
        for kw in dec.keywords:
            if kw.arg is None:
                nonliteral.append("**kwargs")
                continue
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                nonliteral.append(kw.arg)
        found.append((kwargs, dec.lineno, nonliteral))
    return found


def scan_static(
    pkg_root: Path | str | None = None,
    *,
    package: str = PKG_NAME,
    repo_root: Path | str | None = None,
    timeout_files: int | None = None,
) -> tuple[list[Capability], list[dict[str, Any]]]:
    """Recover the registry from source by AST parsing alone.

    Imports nothing from the package: no ``__init__`` execution, no torch, no
    CUDA.  Every pointer (module / entry / test) is resolved against the working
    tree, so the returned ``problems`` are the honest drift report.
    """
    pkg_root_path = Path(pkg_root) if pkg_root is not None else Path(__file__).resolve().parent
    root = Path(repo_root) if repo_root is not None else _repo_root(pkg_root_path.parent)
    caps: list[Capability] = []
    problems: list[dict[str, Any]] = []

    py_files = sorted(
        p for p in pkg_root_path.rglob("*.py")
        if "__pycache__" not in p.parts
    )
    if timeout_files is not None:
        py_files = py_files[:timeout_files]

    seen: dict[str, str] = {}
    for path in py_files:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            problems.append({
                "severity": "warning", "code": "FILE_UNPARSEABLE", "capability_id": None,
                "detail": f"{exc}", "file": str(path.relative_to(root)) if root in path.parents
                else str(path),
            })
            continue
        rel = path.relative_to(pkg_root_path)
        inferred_module = ".".join([package, *rel.with_suffix("").parts]).replace(".__init__", "")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for kwargs, lineno, nonliteral in _decorator_kwargs(node):
                where = {"file": str(path.relative_to(root)), "line": lineno}
                cap_id = kwargs.get("id")
                if not isinstance(cap_id, str) or len(cap_id) < 2:
                    problems.append({
                        "severity": "error", "code": "STATIC_ID_MISSING", "capability_id": cap_id,
                        "detail": "capability decorator without a literal 2+ char id", **where,
                    })
                    continue
                if nonliteral:
                    problems.append({
                        "severity": "error", "code": "STATIC_NONLITERAL_ARG",
                        "capability_id": cap_id,
                        "detail": f"non-literal decorator argument(s): {sorted(set(nonliteral))} "
                                  "— static mode cannot evaluate them",
                        **where,
                    })
                    continue
                cap = Capability(
                    id=cap_id,
                    kind=kwargs.get("kind", "module"),
                    maturity=kwargs.get("maturity", "developing"),
                    module_path=kwargs.get("module_path")
                    or (inferred_module if node.name != "__init__" else package),
                    entry=(kwargs["entry"] if "entry" in kwargs else node.name),
                    params=kwargs.get("params", {}) or {},
                    wired=bool(kwargs.get("wired", False)),
                    tested=bool(kwargs.get("tested", False)),
                    verified=bool(kwargs.get("verified", False)),
                    test_path=kwargs.get("test_path"),
                    evidence=kwargs.get("evidence", {}) or {},
                    endpoint=kwargs.get("endpoint"),
                    source=f"{str(path.relative_to(root))}:{lineno}",
                )
                # pointers must resolve inside the working tree
                if _module_file(pkg_root_path, cap.module_path, package) is None:
                    problems.append({
                        "severity": "error", "code": "MODULE_UNRESOLVED",
                        "capability_id": cap.id,
                        "detail": f"module_path={cap.module_path!r} does not exist under "
                                  f"{pkg_root_path.name}/", **where,
                    })
                elif cap.entry:
                    mpath = _module_file(pkg_root_path, cap.module_path, package)
                    try:
                        mtree = ast.parse(mpath.read_text(encoding="utf-8"))
                    except (OSError, SyntaxError, UnicodeDecodeError):
                        mtree = None
                    if mtree is not None and not _resolve_entry(mtree, cap.entry):
                        problems.append({
                            "severity": "error", "code": "ENTRY_UNRESOLVED",
                            "capability_id": cap.id,
                            "detail": f"entry={cap.entry!r} not defined in "
                                      f"{mpath.relative_to(root)}", **where,
                        })
                if cap.test_path:
                    tfile = root / cap.test_path
                    if not tfile.is_file():
                        problems.append({
                            "severity": "error", "code": "TEST_FILE_MISSING",
                            "capability_id": cap.id,
                            "detail": f"test_path={cap.test_path!r} does not exist", **where,
                        })
                    else:
                        try:
                            ttree = ast.parse(tfile.read_text(encoding="utf-8"))
                        except (OSError, SyntaxError, UnicodeDecodeError):
                            ttree = None
                        if ttree is not None and not _test_names(ttree):
                            problems.append({
                                "severity": "error", "code": "TEST_FILE_EMPTY",
                                "capability_id": cap.id,
                                "detail": f"test_path={cap.test_path!r} defines no test_* function",
                                **where,
                            })
                elif cap.tested:
                    problems.append({
                        "severity": "warning", "code": "TESTED_WITHOUT_TEST_PATH",
                        "capability_id": cap.id,
                        "detail": "tested=True but no test_path to re-run", **where,
                    })
                for prob in validate_capability(cap):
                    prob.setdefault("file", where["file"])
                    prob.setdefault("line", lineno)
                    problems.append(prob)
                if cap.id in seen:
                    problems.append({
                        "severity": "error", "code": "STATIC_DUPLICATE_ID",
                        "capability_id": cap.id,
                        "detail": f"id already declared at {seen[cap.id]}", **where,
                    })
                    continue
                seen[cap.id] = f"{where['file']}:{lineno}"
                caps.append(cap)
    caps.sort(key=lambda c: c.id)
    return caps, problems


def _test_names(tree: ast.AST) -> set[str]:
    return {
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")
    }


# --------------------------------------------------------------------------- #
# Import mode
# --------------------------------------------------------------------------- #

#: Modules are imported only if the static scan found a ``@capability`` in them:
#: importing the whole package to find decorations would be slow and would drag
#: optional backends (triton/paddle/mindspore) into the process.
def _is_capability_decorator(dec: ast.AST) -> bool:
    func = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(func, ast.Attribute):
        return func.attr == "capability"
    return isinstance(func, ast.Name) and func.id == "capability"


def _modules_with_decorators(pkg_root: Path, package: str = PKG_NAME) -> list[str]:
    """Dotted module paths that contain at least one ``@capability`` decorator."""
    mods: list[str] = []
    for path in sorted(pkg_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        decorated = any(
            _is_capability_decorator(dec)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for dec in node.decorator_list
        )
        if not decorated:
            continue
        parts = [package, *path.relative_to(pkg_root).with_suffix("").parts]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        mods.append(".".join(parts))
    return mods


def import_registry(
    modules: Iterable[str] | None = None,
    *,
    pkg_root: Path | None = None,
    registry: Registry | None = None,
) -> tuple[Registry, dict[str, Any]]:
    """Import the decoration-bearing modules and return the populated registry.

    Requires a runnable environment (torch etc.).  Import failures are recorded
    instead of raised, so one broken optional backend cannot blank the whole
    inventory.
    """
    import importlib

    target = registry if registry is not None else REGISTRY
    root = pkg_root or Path(__file__).resolve().parent
    # Prefer *this* working tree over any other ``tensorlbm`` already on
    # ``sys.path`` (an editable install pointing at a sibling checkout would
    # otherwise register that checkout's inventory).
    parent = str(root.parent)
    if sys.path[:1] != [parent]:
        sys.path.insert(0, parent)
    shadow = sys.modules.get(PKG_NAME)
    shadow_path = getattr(shadow, "__file__", None)
    mods = list(modules) if modules is not None else _modules_with_decorators(root)
    failures: list[dict[str, Any]] = []
    for name in mods:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - depends on environment
            failures.append({"module": name, "error": f"{type(exc).__name__}: {exc}"})
    meta: dict[str, Any] = {"imported": len(mods) - len(failures), "failed": failures,
                            "modules": mods}
    if shadow_path and str(root) not in str(shadow_path):
        meta["shadowed_package"] = {
            "requested": str(root),
            "already_imported": str(shadow_path),
            "note": "the process had a different tensorlbm; counts may be polluted",
        }
    return target, meta


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _emit(payload: Mapping[str, Any], out: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n", encoding="utf-8")
    sys.stdout.write(text + "\n")


def _summary_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    caps = payload.get("capabilities") or []
    by_maturity = {m: 0 for m in MATURITIES}
    by_kind: dict[str, int] = {}
    for cap in caps:
        by_maturity[cap.get("maturity", "developing")] = by_maturity.get(
            cap.get("maturity", "developing"), 0) + 1
        by_kind[cap.get("kind", "module")] = by_kind.get(cap.get("kind", "module"), 0) + 1
    return {
        "count": len(caps),
        "by_maturity": by_maturity,
        "by_kind": dict(sorted(by_kind.items())),
        "wired": sum(1 for c in caps if c.get("wired")),
        "tested": sum(1 for c in caps if c.get("tested")),
        "verified": sum(1 for c in caps if c.get("verified")),
        "with_endpoint": sum(1 for c in caps if c.get("endpoint")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tensorlbm.registry",
        description="Dump the TensorLBM capability registry (import or static AST mode).",
    )
    parser.add_argument("--static", action="store_true",
                        help="AST scan only: does not import the package, needs no torch")
    parser.add_argument("--package-root", default=None,
                        help="path to the tensorlbm package directory "
                             "(default: directory of this file)")
    parser.add_argument("--repo-root", default=None,
                        help="repository root used to resolve test_path (default: auto)")
    parser.add_argument("--out", default=None, help="also write the JSON payload here")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when any error-severity problem is found")
    parser.add_argument("--parity", action="store_true",
                        help="import mode: also run the static scan and report differences")
    args = parser.parse_args(argv)

    pkg_root = Path(args.package_root).resolve() if args.package_root else Path(__file__).resolve().parent
    repo_root = Path(args.repo_root).resolve() if args.repo_root else _repo_root(pkg_root.parent)
    commit = _git_commit(repo_root)

    if args.static:
        caps, problems = scan_static(pkg_root, repo_root=repo_root)
        payload: dict[str, Any] = {
            "repo": "TensorLBM",
            "mode": "static",
            "source_commit": commit,
            "package_root": str(pkg_root),
            "count": len(caps),
            "capabilities": [c.to_payload() for c in caps],
            "problems": problems,
        }
        payload["summary"] = _summary_from_payload(payload)
    else:
        # Import mode needs the *package* instance of this module: when the CLI
        # is executed as a file, ``__main__`` and ``tensorlbm.registry`` are two
        # distinct modules, and the decorated modules register into the latter.
        host = sys.modules.get(f"{__package__}.registry") if __package__ else None
        if host is None:
            import importlib

            if str(pkg_root.parent) not in sys.path:
                sys.path.insert(0, str(pkg_root.parent))
            host = importlib.import_module(f"{PKG_NAME}.registry")
        registry, meta = host.import_registry(pkg_root=pkg_root)
        payload = registry.dump(source_commit=commit, mode="import")
        payload["imports"] = meta
        if args.parity:
            caps, problems = scan_static(pkg_root, repo_root=repo_root)
            static = {c.id: c.to_payload() for c in caps}
            runtime = {c["id"]: c for c in payload["capabilities"]}
            missing = sorted(set(static) - set(runtime))
            extra = sorted(set(runtime) - set(static))
            differing = {}
            for cid in sorted(set(static) & set(runtime)):
                diff = {k: [static[cid][k], runtime[cid].get(k)]
                        for k in static[cid] if static[cid][k] != runtime[cid].get(k)}
                if diff:
                    differing[cid] = diff
            payload["parity"] = {
                "static_only": missing,
                "runtime_only": extra,
                "field_mismatch": differing,
                "static_problems": problems,
                "ok": not (missing or extra or differing),
            }

    errors = [p for p in payload.get("problems", []) if p.get("severity") == "error"]
    payload["error_count"] = len(errors)
    _emit(payload, args.out)
    if args.strict and errors:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())