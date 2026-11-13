"""Provenance for the OSS side of the comparison, plus nox-session resolution.

Two jobs:

1. ``fetch()`` downloads the upstream CI definition and its nox session file,
   records ``url``/``sha256``/``fetched_at`` in ``oss_refs/manifest.json`` and
   keeps the bytes under ``oss_refs/`` so the comparison is reproducible
   offline.  Refetching is idempotent (content-addressed); a network failure
   falls back to the cached copy and says so in the manifest, it never silently
   substitutes something else.

2. ``resolve_noxfile()`` reads what each ``nox -s <session>`` actually does
   (the GitLab job only names the session).  The session bodies are parsed with
   ``ast`` -- the commands and their arguments are read from the file, not
   transcribed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OSS_DIR = HERE / "oss_refs"

UPSTREAM = {
    # The lbmpy project moved to i10git (FAU); the gitlab.com mirror redirects
    # to a login page, so the canonical upstream is used.
    "gitlab_ci": "https://i10git.cs.fau.de/pycodegen/lbmpy/-/raw/master/.gitlab-ci.yml",
    "noxfile": "https://i10git.cs.fau.de/pycodegen/lbmpy/-/raw/master/noxfile.py",
    "pyproject": "https://i10git.cs.fau.de/pycodegen/lbmpy/-/raw/master/pyproject.toml",
}
LOCAL = {
    "gitlab_ci": OSS_DIR / "lbmpy.gitlab-ci.yml",
    "noxfile": OSS_DIR / "lbmpy.noxfile.py",
    "pyproject": OSS_DIR / "lbmpy.pyproject.toml",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(refresh: bool = False, timeout: int = 60) -> dict[str, Any]:
    """Refresh the OSS references; returns the manifest."""
    manifest: dict[str, Any] = {}
    for key, url in UPSTREAM.items():
        path = LOCAL[key]
        if path.exists() and not refresh:
            data = path.read_bytes()
            manifest[key] = {
                "url": url,
                "path": str(path.relative_to(HERE.parents[1])),
                "sha256": _sha(data),
                "source": "cache",
            }
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tensorlbm-xval/1.0"})
            data = urllib.request.urlopen(req, timeout=timeout).read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            manifest[key] = {
                "url": url,
                "path": str(path.relative_to(HERE.parents[1])),
                "sha256": _sha(data),
                "source": "network",
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        except Exception as exc:  # noqa: BLE001 - network is optional by design
            if path.exists():
                data = path.read_bytes()
                manifest[key] = {
                    "url": url,
                    "path": str(path.relative_to(HERE.parents[1])),
                    "sha256": _sha(data),
                    "source": "cache-after-network-failure",
                    "network_error": str(exc)[:160],
                }
            else:
                manifest[key] = {"url": url, "error": str(exc)[:160]}
    (OSS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# --------------------------------------------------------------------------- #
# nox session resolution
# --------------------------------------------------------------------------- #
def parse_nox_sessions(path: str | Path) -> dict[str, dict[str, Any]]:
    """Extract ``session.run(...)`` command lists per ``@nox.session`` function."""
    tree = ast.parse(Path(path).read_text())
    sessions: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        is_session = any(
            (isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "session")
            or (isinstance(d, ast.Attribute) and d.attr == "session")
            for d in node.decorator_list
        )
        if not is_session:
            continue
        calls: list[list[str]] = []
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call) or not isinstance(sub.func, ast.Attribute):
                continue
            if sub.func.attr != "run":
                continue
            args = []
            for a in sub.args:
                if isinstance(a, ast.Constant):
                    args.append(str(a.value))
                elif isinstance(a, ast.Name):
                    args.append(a.id)
                elif isinstance(a, ast.Attribute):
                    args.append(a.attr)
                elif isinstance(a, ast.Call):
                    fn = getattr(a.func, "attr", getattr(a.func, "id", "call"))
                    inner = [str(x.value) if isinstance(x, ast.Constant) else "?" for x in a.args]
                    args.append(f"{fn}({','.join(inner)})")
                else:
                    args.append("?")
            calls.append(args)
        sessions[node.name] = {
            "python": _session_python(node),
            "calls": calls,
            "docstring": (ast.get_docstring(node) or "").splitlines()[:1],
        }  # type: ignore[index]
    return sessions


def _session_python(node: ast.FunctionDef) -> list[str] | None:
    for d in node.decorator_list:
        if isinstance(d, ast.Call):
            for kw in d.keywords:
                if kw.arg == "python":
                    if isinstance(kw.value, ast.Constant):
                        return [str(kw.value.value)]
                    if isinstance(kw.value, (ast.List, ast.Tuple)):
                        return [str(e.value) for e in kw.value.elts if isinstance(e, ast.Constant)]
    return None


def _base_session(name: str, sessions: dict[str, Any]) -> str:
    """Map a parametrized session id (``testsuite-3.13(cpu)``) to its function."""
    if name in sessions:
        return name
    import re as _re

    stripped = _re.sub(r"(-\d+\.\d+)?(\(.*\))?$", "", name)
    if stripped in sessions:
        return stripped
    for cand in sessions:
        if name.startswith(cand):
            return cand
    return name


def resolve_noxfile(spec: dict[str, Any], noxfile: str | Path) -> dict[str, Any]:
    """Rewrite each ``nox -s <session>`` job with the session's real commands.

    Returns a small report of what was resolved so the mapping is auditable.
    """
    sessions = parse_nox_sessions(noxfile)
    resolved: dict[str, Any] = {}
    for job in spec["jobs"]:
        sel = job["selection"]
        session = sel.get("nox_session")
        if not session:
            continue
        body = sessions.get(_base_session(session, sessions))
        if body is None:
            resolved[session] = {"found": False, "declared_sessions": sorted(sessions)}
            continue
        pytest_call = next((c for c in body["calls"] if c and c[0] == "pytest"), None)
        commands = [" ".join(c) for c in body["calls"]]
        if pytest_call:
            sel["kind"] = "pytest"
            if "-m" in pytest_call:
                sel["marker_expr"] = pytest_call[pytest_call.index("-m") + 1]
            if "-n" in pytest_call:
                sel["parallelism"] = {
                    "kind": "xdist",
                    "workers": pytest_call[pytest_call.index("-n") + 1],
                }
            sel["coverage"] = "--cov" in pytest_call
        elif any(
            "coverage" in (c[0] if c else "") or (c and c[0] == "coverage") for c in body["calls"]
        ):
            sel["kind"] = "report"
        elif any(c and c[0] in ("flake8", "mypy", "ruff") for c in body["calls"]):
            sel["kind"] = "lint"
            sel["scope"] = [
                a for c in body["calls"] for a in c[1:] if a.startswith(("src", "tests"))
            ]
        elif any(c and c[0] == "python" for c in body["calls"]):
            sel["kind"] = "targeted"
        sel["session_commands"] = commands
        resolved[session] = {
            "found": True,
            "pytest_args": pytest_call,
            "commands": commands,
            "python": body["python"],
        }
    return resolved


def nox_gate_inventory(path: str | Path) -> dict[str, Any]:
    """Sessions *declared* in the noxfile vs sessions *invoked* by the CI file."""
    sessions = parse_nox_sessions(path)
    return {
        "declared_sessions": sorted(sessions),
        "session_python_matrix": {k: v["python"] for k, v in sessions.items()},
    }
