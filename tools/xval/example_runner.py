"""Single dispatch entry for the TensorLBM example set (soft remedy for the
example-per-main antipattern).

Why this exists
---------------
``example_main_audit.py`` measures the *example-per-main* signature: N example
files, each carrying its own ``if __name__ == "__main__"`` block, no
programmatic index, and the case selection living in a build file.  The
TensorLBM tree scores ``entry_ratio = 0.94`` on that metric (see
``docs/XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN.md``) but is **not** rewritten by
this tool: rewriting 189 validated examples would touch physics code for a
pure packaging concern.

Instead the antipattern is removed at the *dispatch* layer, exactly the part
the OSS reference (``listLBM``: ``set(example_path ../examples/.../main.f90)``
in ``CMakeLists.txt``) gets wrong:

* ``--list`` builds a machine-readable index (path, id, summary, entry kind,
  CLI flags, sha256) from static AST inspection -- no import of examples, so
  no solver work and no side effects;
* ``--run ID`` executes the selected example as a subprocess with a fixed
  environment (``PYTHONPATH=<repo>/src`` first) and writes a manifest JSON
  with the file sha256, command, exit code, duration and stdout digest --
  the provenance the platform's evidence discipline requires;
* ``--audit`` re-runs the antipattern metrics on one or more trees so the
  before/after claim is reproducible.

Nothing in this module changes an example's behaviour: the subprocess call is
the same interpreter + file that a user would type by hand.

Usage
-----
    python tools/xval/example_runner.py --list
    python tools/xval/example_runner.py --list --json --out artifacts/.../index.json
    python tools/xval/example_runner.py --run minimal_d2q9 -- --nx 64
    python tools/xval/example_runner.py --audit --compare-tree /path/to/listLBM
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):  # executed as a script: make sibling import work
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from example_main_audit import python_file_info, scan_tree  # type: ignore[import-not-found]
else:  # pragma: no cover - package import path used by tests
    from .example_main_audit import python_file_info, scan_tree

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "1.0"


@dataclass
class ExampleEntry:
    """One discoverable example unit."""

    id: str
    path: str
    summary: str
    has_entry: bool
    entry_kind: str
    cli_flags: list[str] = field(default_factory=list)
    nonblank_lines: int = 0
    sha256: str = ""
    run_hint: str = ""


def _short_id(rel: Path) -> str:
    """``examples/foo/bar.py`` -> ``foo/bar``; ``examples/foo.py`` -> ``foo``."""
    parts = rel.with_suffix("").parts
    if parts and parts[0] == "examples":
        parts = parts[1:]
    return "/".join(parts)


def _run_hint(doc: str) -> str:
    """First ``python ... examples/...`` command inside the module docstring."""
    for line in doc.splitlines():
        text = line.strip().lstrip(":").strip()
        if "examples/" in text and "python" in text:
            return text
    return ""


def build_index(examples_dir: Path, root: Path) -> list[ExampleEntry]:
    """Static index of every ``*.py`` under ``examples_dir`` (sorted by id)."""
    examples_dir = Path(examples_dir)
    entries: list[ExampleEntry] = []
    for path in sorted(examples_dir.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        rel = path.resolve().relative_to(Path(root).resolve())
        text = path.read_text(encoding="utf-8", errors="replace")
        unit = python_file_info(path)
        entries.append(
            ExampleEntry(
                id=_short_id(rel),
                path=str(rel),
                summary=str(unit["summary"]),
                has_entry=bool(unit["has_entry"]),
                entry_kind=str(unit["entry_kind"]),
                cli_flags=list(unit["cli_flags"]),
                nonblank_lines=int(unit["nonblank_lines"]),
                sha256=str(unit["sha256"]),
                run_hint=_run_hint(text[:4000]),
            )
        )
    return entries


def resolve(entries: list[ExampleEntry], target: str) -> ExampleEntry:
    """Resolve ``target`` by id or by path (exact, then unique suffix match)."""
    target = target.replace("\\", "/")
    for entry in entries:
        if entry.id == target or entry.path == target:
            return entry
    matches = [e for e in entries if e.id.endswith("/" + target) or e.path.endswith("/" + target)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"no example matches {target!r}")
    raise KeyError(f"ambiguous example {target!r}: {[m.id for m in matches]}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_example(
    entry: ExampleEntry,
    *,
    root: Path,
    extra_args: list[str] | None = None,
    timeout: float = 1800.0,
    manifest: Path | None = None,
    python: str | None = None,
    keep_stdout: bool = False,
) -> dict[str, object]:
    """Execute one example in a subprocess and (optionally) write a manifest."""
    root = Path(root)
    python = python or sys.executable
    path = root / entry.path
    env = dict(os.environ)
    src = str(root / "src")
    parts = [src, *[p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    cmd = [python, entry.path, *(extra_args or [])]
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True,
                          timeout=timeout, check=False)
    duration = time.perf_counter() - t0
    record = {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/xval/example_runner.py",
        "example_id": entry.id,
        "example_path": entry.path,
        "example_sha256": entry.sha256,
        "source_sha256_actual": _sha256(path.read_bytes()),
        "cmd": cmd,
        "cmd_shell": " ".join(shlex.quote(c) for c in cmd),
        "cwd": str(root),
        "pythonpath": env["PYTHONPATH"],
        "python": python,
        "rc": proc.returncode,
        "duration_s": round(duration, 3),
        "stdout_sha256": _sha256(proc.stdout.encode()),
        "stderr_sha256": _sha256(proc.stderr.encode()),
        "stdout_lines": len(proc.stdout.splitlines()),
        "stdout_tail": proc.stdout.splitlines()[-15:],
        "stderr_tail": proc.stderr.splitlines()[-15:],
        "started_at": started.isoformat().replace("+00:00", "Z"),
        "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if keep_stdout:
        # Full text is returned for A/B comparisons; the on-disk manifest keeps
        # only the digest + tail so artefacts stay small.
        record["stdout"] = proc.stdout
        record["stderr"] = proc.stderr
    if manifest:
        manifest = Path(manifest)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        slim = {k: v for k, v in record.items() if k not in {"stdout", "stderr"}}
        manifest.write_text(json.dumps(slim, indent=2) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TensorLBM example dispatcher (single entry point)")
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--examples-dir", default=None, help="default: <root>/examples")
    ap.add_argument("--list", action="store_true", help="print the static example index")
    ap.add_argument("--run", metavar="ID", default=None, help="run one example by id")
    ap.add_argument("--audit", action="store_true", help="print antipattern metrics")
    ap.add_argument("--compare-tree", action="append", default=[])
    ap.add_argument("--block-len", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None, help="write JSON output to this path")
    ap.add_argument("--manifest", default=None, help="run manifest path (default: artifacts/...)")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="arguments after '--' are forwarded to the example")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    examples_dir = Path(args.examples_dir) if args.examples_dir else root / "examples"
    entries = build_index(examples_dir, root)

    if args.audit:
        trees = [examples_dir, *[Path(p) for p in args.compare_tree]]
        reports = [scan_tree(t, block_len=args.block_len) for t in trees]
        for report in reports:
            report.pop("units", None)
        payload: object = reports
        if args.json or args.out:
            text = json.dumps(payload, indent=2)
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(text + "\n")
            if args.json:
                print(text)
        else:
            for r in reports:
                print(
                    f"{r['root']}: files={r['n_files']} entries={r['n_entry_units']} "
                    f"entry_ratio={r['entry_ratio']:.3f} dup={r['dup_line_ratio']:.3f} "
                    f"cli={r['cli_adoption']:.3f} index={r['has_programmatic_index']} "
                    f"build_selectors={len(r['build_selectors'])}"
                )
        return 0

    if args.run:
        entry = resolve(entries, args.run)
        rest = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
        manifest = None
        if not args.no_manifest:
            manifest = (
                Path(args.manifest)
                if args.manifest
                else root / "artifacts" / "xval_example_per_main" / "runs"
                / f"{entry.id.replace('/', '_')}_{int(time.time())}.json"
            )
        record = run_example(entry, root=root, extra_args=rest, timeout=args.timeout,
                             manifest=manifest)
        if args.json:
            print(json.dumps(record, indent=2))
        else:
            print(f"[example_runner] {entry.id}: rc={record['rc']} "
                  f"duration={record['duration_s']}s stdout_sha256={record['stdout_sha256'][:16]}")
            tail = record["stdout_tail"]
            if tail:
                print("\n".join(str(line) for line in tail))
            manifest_path = manifest if manifest else None
            if manifest_path:
                print(f"[example_runner] manifest -> {manifest_path}")
        return int(record["rc"])

    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/xval/example_runner.py",
        "root": str(root),
        "examples_dir": str(examples_dir),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_entries": len(entries),
        "n_runnable": sum(1 for e in entries if e.has_entry),
        "n_cli": sum(1 for e in entries if e.cli_flags),
        "entries": [asdict(e) for e in entries],
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    if args.json:
        print(text)
    else:
        print(f"{len(entries)} examples ({payload['n_runnable']} runnable, "
              f"{payload['n_cli']} with CLI flags) under {examples_dir}")
        for entry in entries:
            flag = "cli" if entry.cli_flags else "   "
            entry_mark = "main" if entry.has_entry else "----"
            print(f"  {entry_mark} {flag}  {entry.id:<58} {entry.summary[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())