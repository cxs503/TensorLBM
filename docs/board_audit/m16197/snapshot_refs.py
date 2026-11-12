#!/usr/bin/env python3
"""Snapshot the audited m16140 artifacts into this audit tree, with sha256 provenance.

The audited worktree (/root/tp-agent/work/m16140) is subject to recycling, so the
evidence must survive it: every file below is a verbatim byte copy, hashed, and its
source commit recorded. Nothing is edited after copying.
"""
import hashlib
import json
import shutil
from pathlib import Path

SRC = Path("/root/tp-agent/work/m16140/benchmarks/case_pool/sc_sphere_array_drag")
REF = Path(__file__).resolve().parent / "refs"
REF.mkdir(parents=True, exist_ok=True)

FILES = [
    "run.py",
    "REPORT.md",
    "NOTES.md",
    "out/case_phi0.125_N16.json",
    "out/case_phi0.125_N32.json",
    "out/case_phi0.45_N32.json",
    "out/result.json",
    "figures/k_vs_N.png",
    "figures/error_vs_N.png",
    "figures/field_slice.png",
]

rec = []
for rel in FILES:
    s = SRC / rel
    if not s.exists():
        rec.append({"path": rel, "status": "MISSING"})
        continue
    dst = REF / rel.replace("/", "__")
    shutil.copyfile(s, dst)
    h = hashlib.sha256(dst.read_bytes()).hexdigest()
    rec.append({"path": rel, "snapshot": dst.name, "bytes": dst.stat().st_size, "sha256": h})

prov = {
    "source_worktree": str(SRC),
    "source_commit": "c0a3e6c3661da3eda07fbbff2872a2cc098f6f79",
    "source_commit_subject": "m16140: CASE-LBM-CUBIC-ARRAY-DRAG case-pool entry (TensorLBM)",
    "tracked_in_delivered_commit": [r["path"] for r in rec if r.get("path") != "out/case_phi0.45_N32.json"],
    "untracked_in_delivered_commit": ["out/case_phi0.45_N32.json"],
    "files": rec,
    "note": "byte copies, hashed at copy time; case_phi0.45_N32.json was git-untracked in m16140 (finding F3)",
}
(REF / "provenance.json").write_text(json.dumps(prov, indent=1))
print("snapshotted", len([r for r in rec if r.get("sha256")]), "files")
for r in rec:
    print(" ", r["path"], r.get("bytes"), (r.get("sha256") or "")[:12])