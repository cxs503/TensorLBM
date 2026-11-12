"""Run ledger: provenance-carrying artifacts for every executed matrix case.

A verification matrix is only worth what its *runs* can prove.  The frozen
``result.json`` files under ``benchmarks/`` say what a case measured; they do
not say **when, where, from which commit and by which command** the numbers
were produced.  This module records exactly that for every execution started
through ``runner.py run``:

::

    artifacts/validation_matrix/runs/
    ├── index.jsonl                                  append-only run index
    └── <UTC-stamp>__<case_id>/
        ├── manifest.json                            machine-readable provenance
        ├── run.log                                  full stdout+stderr of the repro command
        └── status.json                              the machine verdict of that run

The manifest binds four things that are otherwise easy to hand-wave:

1. **source identity** — git sha (and dirty flag of the tree that ran),
   ``matrix.json``/``references.json`` sha256, python/torch versions;
2. **execution identity** — exact ``repro_cmd``, argv, hostname, CPU count,
   env knobs that change numerics (``OMP_NUM_THREADS``, ``TORCH_NUM_THREADS``,
   ``PYTHONHASHSEED``), start/finish UTC and wall time;
3. **outcome identity** — return code, artifact path + sha256 + size, the
   three-level status and the per-metric verdict of *that* run;
4. **chain identity** — the evidence chain of the judged artifact, so a run can
   be re-verified later even if the artifact is regenerated.

Nothing here invents or upgrades a verdict: the ledger copies whatever
``criteria.judge_case`` decided.  A failed run is recorded as a failed run.

CLI (also reachable as ``python -m benchmarks.validation_matrix.runledger``)::

    python benchmarks/validation_matrix/runledger.py list [--case CASE] [--json]
    python benchmarks/validation_matrix/runledger.py verify            # sha256 drift check
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
LEDGER_ROOT = REPO_ROOT / "artifacts" / "validation_matrix" / "runs"
INDEX = LEDGER_ROOT / "index.jsonl"
SCHEMA = "tensorlbm.validation_matrix.run_ledger/v1"


# --------------------------------------------------------------------------- #
# identity helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_identity() -> dict:
    """Commit + dirty state of the tree that is about to be executed."""
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": sha,
        "dirty": None if status is None else bool(status),
        "describe": _git("describe", "--always", "--dirty"),
    }


def host_identity() -> dict:
    """Machine identity incl. the thread knobs that change LBM numerics."""
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "env": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS",
            "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES",
        ) if os.environ.get(k) is not None},
    }
    try:  # torch is optional for the pure-rule tests
        import torch  # noqa: F401

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch is normally present
        info["torch"] = None
    return info


def _rel(path: Path) -> str:
    """Repo-relative when inside the tree, absolute otherwise (tmp ledgers)."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(path).resolve())


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _stamp(ts: _dt.datetime) -> str:
    return ts.strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #
def record_run(case: dict, run: dict, verdict: dict,
               ledger_root: Path | None = None,
               extra: dict | None = None) -> Path:
    """Write one run directory + index line; return the directory.

    ``case`` is the ``matrix.json`` registry entry, ``run`` the dict returned by
    ``runner.run_repro`` (cmd/returncode/elapsed/stdout/stderr), ``verdict`` the
    dict returned by ``runner.evaluate_case``.  No field is derived from a
    human-readable claim: every value is copied from those three objects or
    measured here.
    """
    root = Path(ledger_root) if ledger_root is not None else LEDGER_ROOT
    finished = _utcnow()
    started = finished - _dt.timedelta(seconds=float(run.get("elapsed_s") or 0.0))
    run_id = f"{_stamp(finished)}__{case['id']}"
    out_dir = root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log_text = (
        f"$ {run.get('cmd')}\n"
        f"# cwd={REPO_ROOT}\n"
        f"# returncode={run.get('returncode')} elapsed_s={run.get('elapsed_s'):.3f}\n"
        "----- stdout -----\n"
        f"{run.get('stdout') or run.get('stdout_tail') or ''}\n"
        "----- stderr -----\n"
        f"{run.get('stderr') or run.get('stderr_tail') or ''}\n"
    )
    (out_dir / "run.log").write_text(log_text, encoding="utf-8")
    (out_dir / "status.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    artifact_rel = verdict.get("artifact")
    artifact_path = REPO_ROOT / artifact_rel if artifact_rel else None
    manifest = {
        "schema": SCHEMA,
        "run_id": run_id,
        "case": case["id"],
        "slot": case.get("slot"),
        "title": case.get("title"),
        "matrix_version": verdict.get("matrix_version"),
        "repro_cmd": run.get("cmd"),
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_utc": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_s": round(float(run.get("elapsed_s") or 0.0), 3),
        "returncode": run.get("returncode"),
        "repo": {
            **git_identity(),
            "matrix_sha256": sha256_file(HERE / "matrix.json"),
            "references_sha256": sha256_file(HERE / "references.json"),
        },
        "host": host_identity(),
        "artifact": {
            "path": artifact_rel,
            "exists": bool(artifact_path and artifact_path.exists()),
            "sha256": verdict.get("artifact_sha256")
            or (sha256_file(artifact_path) if artifact_path and artifact_path.exists() else None),
            "bytes": verdict.get("artifact_bytes"),
        },
        "verdict": {
            "status": verdict.get("status"),
            "machine_pass": verdict.get("machine_pass"),
            "reason": verdict.get("reason"),
            "metrics": [
                {
                    "metric": m.get("metric"),
                    "reference": m.get("reference"),
                    "tol_pct": m.get("tol_pct"),
                    "max_abs_err_pct": m.get("max_abs_err_pct"),
                    "pass": m.get("pass"),
                    "levels": [
                        {k: lvl.get(k) for k in ("level", "value", "err_pct", "pass", "gated")}
                        for lvl in m.get("levels", [])
                    ],
                }
                for m in verdict.get("metrics", [])
            ],
            "convergence": verdict.get("convergence", []),
        },
        "evidence_chain": verdict.get("evidence_chain", []),
    }
    if extra:
        manifest["extra"] = extra
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.jsonl"
    index_row = {
        "run_id": run_id,
        "case": case["id"],
        "finished_utc": manifest["finished_utc"],
        "elapsed_s": manifest["elapsed_s"],
        "returncode": manifest["returncode"],
        "status": manifest["verdict"]["status"],
        "machine_pass": manifest["verdict"]["machine_pass"],
        "artifact_sha256": manifest["artifact"]["sha256"],
        "manifest_sha256": sha256_file(out_dir / "manifest.json"),
        "manifest": _rel(out_dir / "manifest.json"),
    }
    with open(index_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(index_row, ensure_ascii=False) + "\n")
    return out_dir


def load_index(ledger_root: Path | None = None) -> list[dict]:
    index = (Path(ledger_root) / "index.jsonl") if ledger_root is not None else INDEX
    if not index.exists():
        return []
    rows = []
    for line in index.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def verify_ledger(ledger_root: Path | None = None) -> dict:
    """Re-hash every recorded manifest; report drift (tamper / stale evidence)."""
    rows = load_index(ledger_root)
    checks = []
    for row in rows:
        path = Path(row["manifest"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        entry = {"run_id": row["run_id"], "manifest": row["manifest"]}
        if not path.exists():
            entry.update(ok=False, reason="manifest missing")
        else:
            digest = sha256_file(path)
            entry.update(
                ok=digest == row.get("manifest_sha256"),
                sha256=digest,
                recorded_sha256=row.get("manifest_sha256"),
            )
            if not entry["ok"]:
                entry["reason"] = "manifest sha256 drifted from index"
        checks.append(entry)
    return {
        "n_runs": len(rows),
        "n_ok": sum(1 for c in checks if c["ok"]),
        "pass": all(c["ok"] for c in checks),
        "runs": checks,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cmd_list(args) -> int:
    rows = load_index()
    if args.case:
        rows = [r for r in rows if r["case"] == args.case]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    if not rows:
        print("no runs recorded yet (run `runner.py run --case <id>`)")
        return 0
    print(f"{len(rows)} recorded run(s), ledger {LEDGER_ROOT.relative_to(REPO_ROOT)}")
    for r in rows:
        print(f"  {r['finished_utc']}  {r['case']:<28} rc={r['returncode']} "
              f"status={r['status']:<7} {r['elapsed_s']:.1f}s  {r['manifest']}")
    return 0


def cmd_verify(args) -> int:
    report = verify_ledger()
    print(f"{report['n_ok']}/{report['n_runs']} ledger manifests re-hashed clean")
    for c in report["runs"]:
        if not c["ok"]:
            print(f"  FAIL {c['run_id']}: {c.get('reason')}")
    return 0 if report["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list", help="list recorded runs")
    p.add_argument("--case")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)
    sub.add_parser("verify", help="re-hash the ledger manifests").set_defaults(func=cmd_verify)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())