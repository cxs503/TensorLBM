#!/usr/bin/env python3
"""Extract the TCLB reference snippets that this XVAL rests on ("zero hand-copy").

Every claim in ``docs/XVAL-LBM-PER-MARGIN-ARRAYS-POINTER-SHIFT-STREAMING.md``
about the OSS implementation is backed by a byte-exact excerpt taken from the
upstream checkout by *this script* -- no source text is retyped by hand.

Source tree: a shallow checkout of ``https://github.com/CFD-GO/TCLB.git``
(branch ``master``, commit recorded in ``.git/refs/heads/master``).  Only the
files listed in ``SNIPPETS`` are read; each one is hashed (sha256) so the
report can be checked against the same revision.

Usage::

    python benchmarks/xval_tclb_pointer_shift/extract_oss_ref.py [--src /tmp/TCLB]

Writes ``benchmarks/xval_tclb_pointer_shift/oss_ref/REF.json`` plus one
``.txt`` excerpt per snippet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

DEFAULT_SRC = Path("/tmp/TCLB")
OUT_DIR = Path(__file__).resolve().parent / "oss_ref"

#: (key, file, start-anchor regex, end-anchor regex, claim it backs)
SNIPPETS: list[tuple[str, str, str, str, str]] = [
    (
        "ftabs_per_margin_arrays",
        "src/Global.h.Rt",
        r"/// Main data structure for the lattice",
        r"#define INFO_H 1",
        "populations are stored as one array per margin block (per-margin arrays)",
    ),
    (
        "margin_table",
        "src/conf.R",
        r"^Margin = data\.frame\(",
        r"^Margin=rows\(Margin\)",
        "the 27 margin blocks are the 3x3x3 combinations of side/low/mid/high spans",
    ),
    (
        "offsets_block_addressing",
        "src/conf.R",
        r"^offsets = function\(\) \{",
        r"^\s*list\(Fields=ret, MarginSizes=MarginNSize \* size\)",
        "block offsets are linear offsets inside a per-margin array (offset.p / get_offsets / put_offsets)",
    ),
    (
        "stencil_derived_extent",
        "src/conf.R",
        r"^AddField = function\(",
        r"adjoint=adjoint,",
        "each field's spatial extent is derived from the stencil, not hand-written",
    ),
    (
        "move_pocket_pointer_shift",
        "src/LatticeAccess.inc.cpp.Rt",
        r"^  move\.pocket = function\(i,offset, MContext\) \{",
        r"^  range_int=function\(x\)",
        "a block transition is a pointer increment (pocket.blockN += delta), not an index recomputation",
    ),
    (
        "field_access_uses_pocket",
        "src/LatticeAccess.inc.cpp.Rt",
        r"^  field\.access\.one = function\(d,m,off,access\) \{",
        r"^  load\.field = function",
        "every population read/write indexes a moved pocket with a constant per-block offset",
    ),
    (
        "pop_push_templates",
        "src/LatticeAccess.inc.cpp.Rt",
        r"::pop<\?%s s\$suffix \?>\(N & node\) const",
        r"::push<\?%s s\$suffix \?>\(N & node\) const",
        "streaming reads (pop) and writes (push) are emitted per stage from the block/pocket machinery",
    ),
    (
        "iterate_border_interior_split",
        "src/Lattice.cu.Rt",
        r"DEBUG_PROF_PUSH\(\"<\?%s stage\$name \?>\"\);",
        r"container->RunInterior< Primal, NoGlobals, <\?%s stage\$name \?> > \(kernelStream\); break;",
        "each stage runs a border kernel, then an MPI margin exchange, then an interior kernel",
    ),
    (
        "mpi_margin_exchange",
        "src/Lattice.cu.Rt",
        r"^inline void Lattice::MPIStream_A\(\)",
        r"^inline void Lattice::MPIStream_B\(int tag\)",
        "only non-empty margin blocks are exchanged across ranks",
    ),
]

#: files hashed for provenance even when no excerpt is taken from them
HASH_ONLY = ["README.md", ".git/refs/heads/master", ".git/packed-refs", "makefile"]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def extract(path: Path, start_re: str, end_re: str) -> dict:
    text = path.read_text(errors="replace")
    m0 = re.search(start_re, text, flags=re.MULTILINE)
    if m0 is None:
        raise SystemExit(f"start anchor not found in {path}: {start_re}")
    m1 = re.search(end_re, text[m0.start():], flags=re.MULTILINE)
    end = m0.start() + (m1.end() if m1 else len(text) - m0.start())
    excerpt = text[m0.start():end]
    return {
        "file": str(path),
        "start_line": line_of(text, m0.start()),
        "end_line": line_of(text, end),
        "n_lines": excerpt.count("\n") + 1,
        "excerpt": excerpt,
        "excerpt_sha256": sha256(excerpt.encode()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    args = ap.parse_args()
    src: Path = args.src
    if not (src / "src" / "conf.R").exists():
        raise SystemExit(f"not a TCLB checkout: {src}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    revision = (src / ".git" / "refs" / "heads" / "master").read_text().strip()

    files: dict[str, dict] = {}
    for rel in sorted({s[1] for s in SNIPPETS} | set(HASH_ONLY)):
        p = src / rel
        if not p.exists():
            continue
        raw = p.read_bytes()
        files[rel] = {"sha256": sha256(raw), "bytes": len(raw)}

    snippets = []
    for key, rel, start_re, end_re, claim in SNIPPETS:
        info = extract(src / rel, start_re, end_re)
        info = {k: v for k, v in info.items() if k != "excerpt"}
        info.update(key=key, rel_path=rel, claim=claim)
        # keep the excerpt text in a separate .txt so REF.json stays readable
        ex = extract(src / rel, start_re, end_re)
        (OUT_DIR / f"{key}.txt").write_text(
            f"# {rel}:{ex['start_line']}-{ex['end_line']}  sha256(excerpt)={ex['excerpt_sha256']}\n"
            + ex["excerpt"]
        )
        snippets.append(info)

    ref = {
        "oss": "CFD-GO/TCLB",
        "remote": "https://github.com/CFD-GO/TCLB.git",
        "branch": "master",
        "commit": revision,
        "checkout": str(src),
        "checkout_git_note": (
            "shallow clone: `.git` object store is incomplete (`git log` fails with "
            "'bad object HEAD'), so provenance is pinned by the ref file content and "
            "per-file sha256 instead of a git object id"
        ),
        "tclb_runnable_here": False,
        "tclb_runnable_reason": (
            "code generation and build need R + Rscript (src/*.Rt are R templates) and an "
            "MPI/CUDA toolchain; this container has neither R nor network access"
        ),
        "extracted_by": "benchmarks/xval_tclb_pointer_shift/extract_oss_ref.py",
        "files": files,
        "snippets": snippets,
    }
    (OUT_DIR / "REF.json").write_text(json.dumps(ref, indent=2) + "\n")
    print(json.dumps({"commit": revision, "files": len(files), "snippets": len(snippets)}, indent=2))
    for name, info in files.items():
        print(f"  {name:34s} {info['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())