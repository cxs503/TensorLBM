#!/usr/bin/env python3
"""Extract the constant lattice tables from upstream LatFlow sources (零手抄).

The cross-validation reference is
``loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow`` (``LatFlow``).
Running it is not possible in this environment (TensorFlow 1.x graph/session
API, ``xrange``), so the reference numbers used by the cross-validation are
produced by a *verbatim transplant* of upstream's streaming/bounce formulation
onto torch.  To keep that transplant honest -- and to keep every table a
machine-extracted fact rather than a transcription -- this script parses the
upstream files and stores the tables as JSON next to their provenance
(sha256 of every source file, fetch URL, fetch timestamp).

Upstream files can be supplied as a checkout path, as a tarball, or downloaded
on first use (``--download``); the recorded sha256 makes the extraction
reproducible even if the network is unavailable later.

Usage
-----
    python benchmarks/xval_conv_as_streaming/extract_oss_tables.py \
        --src /tmp/oss_lbm/Lattice-Boltzmann-fluid-flow-in-Tensorflow-master
    python benchmarks/xval_conv_as_streaming/extract_oss_tables.py --download
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import tarfile
import tempfile
import urllib.request

import numpy as np
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF_DIR = HERE / "oss_ref"

REPO = "loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow"
TARBALL_URL = f"https://codeload.github.com/{REPO}/tar.gz/refs/heads/master"
LATTICES = ("D2Q9", "D3Q15", "D3Q19")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _balanced(text: str, start: int, open_ch: str = "[", close_ch: str = "]") -> str:
    """Return the bracketed slice beginning at ``start`` (which must be the opener)."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("unbalanced brackets")


def _eval_numeric(src: str):
    """Evaluate a pure-numeric Python literal expression (``4./9.``, ``1./18.`` ...).

    ``ast.literal_eval`` rejects the binary operators upstream writes its
    weights with, so walk the AST ourselves and allow only numbers, unary
    signs, ``+ - * /`` and (nested) list/tuple literals.
    """
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            val = walk(node.operand)
            return -val if isinstance(node.op, ast.USub) else val
        if isinstance(node, ast.BinOp):
            lhs, rhs = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return lhs + rhs
            if isinstance(node.op, ast.Sub):
                return lhs - rhs
            if isinstance(node.op, ast.Mult):
                return lhs * rhs
            if isinstance(node.op, ast.Div):
                return lhs / rhs
        if isinstance(node, (ast.List, ast.Tuple)):
            return [walk(e) for e in node.elts]
        raise ValueError(f"unsupported node {type(node).__name__} in numeric table")

    return walk(ast.parse(src, mode="eval"))


def _constant_list(text: str, name: str) -> list:
    """Parse ``NAME = tf.constant(<list>, dtype=...)`` (or ``np.array``)."""
    m = re.search(rf"{name}\s*=\s*(?:tf\.constant|np\.array)\(\s*", text)
    if not m:
        raise ValueError(f"{name} assignment not found")
    start = text.index("[", m.end())
    return _eval_numeric(_balanced(text, start))


def _stream_table(text: str, n: int) -> list:
    """Build the STREAM table from ``STREAM[i, j, ..., q, q] = 1.0`` statements."""
    entries = re.findall(r"STREAM\[([0-9,\s]+)\]\s*=\s*1\.0", text)
    if not entries:
        raise ValueError("no STREAM entries found")
    idx = [tuple(int(v) for v in e.split(",")) for e in entries]
    dims = len(idx[0]) - 2
    shape = tuple(max(i[d] for i in idx) + 1 for d in range(dims)) + (n, n)
    table = np.zeros(shape, dtype=np.float64)
    for entry in idx:
        table[entry] = 1.0
    return table.tolist()


def parse_lattice(text: str, name: str) -> dict:
    weights = _constant_list(text, "WEIGHTS")
    lveloc = _constant_list(text, "LVELOC")
    bounce = _constant_list(text, "BOUNCE")
    stream = _stream_table(text, len(lveloc))
    return {
        "lattice": name,
        "q": len(lveloc),
        "weights": weights,
        "lveloc": lveloc,
        "bounce": bounce,
        "stream": stream,
    }


def fetch(url: str) -> tuple[bytes, str]:
    with urllib.request.urlopen(url, timeout=180) as fh:
        return fh.read(), fh.headers.get("ETag", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=None, help="upstream checkout (contains LatFlow/)")
    ap.add_argument("--tarball", type=Path, default=None, help="upstream master tarball")
    ap.add_argument("--download", action="store_true", help="download upstream master tarball")
    ap.add_argument("--out", type=Path, default=REF_DIR)
    args = ap.parse_args()

    provenance: dict = {
        "repo": REPO,
        "tarball_url": TARBALL_URL,
        "fetched_utc": None,
        "tarball_sha256": None,
        "extractor": "benchmarks/xval_conv_as_streaming/extract_oss_tables.py",
        "files": {},
    }
    tmpdir = None
    if args.src is not None:
        src = args.src
    else:
        tarball = args.tarball
        if tarball is None:
            if not args.download:
                raise SystemExit("give --src, --tarball or --download")
            tmpdir = tempfile.mkdtemp()
            tarball = Path(tmpdir) / "master.tar.gz"
            data, _etag = fetch(TARBALL_URL)
            tarball.write_bytes(data)
            provenance["fetched_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        provenance["tarball_sha256"] = sha256_bytes(tarball.read_bytes())
        if tmpdir is None:
            tmpdir = tempfile.mkdtemp()
        with tarfile.open(tarball) as tf:
            tf.extractall(tmpdir)
        roots = [p for p in Path(tmpdir).iterdir() if p.is_dir()]
        src = roots[0]

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for name in LATTICES:
        path = src / "LatFlow" / f"{name}.py"
        raw = path.read_bytes()
        provenance["files"][f"LatFlow/{name}.py"] = sha256_bytes(raw)
        tables = parse_lattice(raw.decode("utf-8"), name)
        (out / f"{name}.json").write_text(json.dumps(tables, indent=1) + "\n")
        print(f"{name}: q={tables['q']} stream dims={len(tables['stream'])} ok")

    for rel in ("LatFlow/Domain.py", "LatFlow/utils.py"):
        provenance["files"][rel] = sha256_bytes((src / rel).read_bytes())
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print("provenance ->", out / "provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())