#!/usr/bin/env python3
"""W22 / m14504 — DTRC SUBOFF AFF-8 external anchor: field-by-field machine transcription.

Purpose
-------
Turn every scalar leaf of the anchor-bearing artifacts
  * benchmarks/verified/suboff_sail_resistance/verdict.json
  * benchmarks/verified/suboff_sail_resistance/e2_verdict.json
  * benchmarks/verified/suboff_sail_resistance/tau_ladder_report_m298.md
  * benchmarks/verified/suboff_sail_resistance/tau_ladder_verdict_m298.json
(plus the two upstream anchor tables the k / geometry fields live in)
  * benchmarks/pending/suboff_highre_resistance/refs_lock.json
  * benchmarks/verified/suboff_anchor/anchor.json
into a machine-readable artifact where every value carries
  - an RFC-6901 JSON pointer (JSON sources) or a document-order pointer (markdown),
  - the exact source file sha256 and the git commit,
  - a readback record (independent re-resolution + char-level re-read).

No number in this artifact is typed by hand; all values are parsed from the
sources at build time.  Nothing is written outside this repository.

Output: benchmarks/verified/suboff_anchor_w22/anchor_transcription.json
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT = Path(os.environ.get("W22_TRANSCRIPTION_OUT",
                          str(HERE.parent / "anchor_transcription.json")))

TASK_KEY = "paper-exp-lbm-suboff-tau-w22-anchor8"
MISSION_ID = 14504
PREDECESSOR = "paper-exp-lbm-suboff-tau-4"

# ---------------------------------------------------------------- source list
SOURCES = [
    ("S1", "benchmarks/verified/suboff_sail_resistance/verdict.json", "json",
     "formal mix50 ladder verdict (archived, OVERALL=FAIL)"),
    ("S2", "benchmarks/verified/suboff_sail_resistance/e2_verdict.json", "json",
     "erratum E2 ladder verdict (standard friction estimator, PASS)"),
    ("S3", "benchmarks/verified/suboff_sail_resistance/tau_ladder_report_m298.md", "md",
     "m298 tau-ladder report (all report numbers declared sourced from S4)"),
    ("S4", "benchmarks/verified/suboff_sail_resistance/tau_ladder_verdict_m298.json", "json",
     "m298 tau-ladder formal verdict artifact (named in S3 section 8)"),
    ("S5", "benchmarks/pending/suboff_highre_resistance/refs_lock.json", "json",
     "frozen AFF-8 reference lock (CT ladder, ITTC-57 C_F, (1+k) record)"),
    ("S6", "benchmarks/verified/suboff_anchor/anchor.json", "json",
     "normalized SUBOFF anchor (geometry, AFF-1/AFF-8 (1+k) double entry)"),
    ("S7", "benchmarks/verified/suboff_sail_resistance/result.json", "json",
     "compiled case record (cf_ref register, caliber provenance, which cross-checks "
     "were recorded as not applicable)"),
    ("S8", "benchmarks/verified/suboff_sail_resistance/prereg.md", "md",
     "frozen case prereg (Re, reference formula, hull configuration, frozen parameters)"),
]

# Numeric token grammar for markdown.  A token is accepted only when it is not
# glued to an identifier (L96, AFF-8, m298, GPU5, Re_eff, ...).
NUM = r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"
TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])"          # left boundary: not inside an identifier
    r"(?<![A-Za-z]-)"             # not the tail of AFF-8 / IN-25 style ids
    r"(" + NUM + r")"
    r"(?![A-Za-z0-9_])"           # right boundary
)

FENCE_RE = re.compile(r"^\s*```")
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|[\s:\-|]*$")


# ------------------------------------------------------------------- utilities
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def json_pointer_escape(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def json_pointer_unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def kind_of(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    raise TypeError(f"non-scalar leaf: {type(value)!r}")


def walk_json(node, pointer: str, out: list[dict]) -> None:
    """Depth-first, key-order-preserving walk emitting every scalar leaf."""
    if isinstance(node, dict):
        for key, child in node.items():
            walk_json(child, pointer + "/" + json_pointer_escape(str(key)), out)
    elif isinstance(node, list):
        for idx, child in enumerate(node):
            walk_json(child, pointer + "/" + str(idx), out)
    else:
        out.append({"pointer": pointer, "value": node, "type": kind_of(node)})


def resolve_pointer(doc, pointer: str):
    if pointer == "":
        return doc
    if not pointer.startswith("/"):
        raise ValueError(f"bad pointer {pointer!r}")
    node = doc
    for raw in pointer.split("/")[1:]:
        token = json_pointer_unescape(raw)
        if isinstance(node, list):
            node = node[int(token)]
        elif isinstance(node, dict):
            node = node[token]
        else:
            raise KeyError(f"cannot descend into {type(node)!r} at {token!r}")
    return node


def strip_markup(text: str) -> str:
    return text.replace("**", "").replace("`", "").replace("*", "")


# ------------------------------------------------------- markdown token census
def index_markdown(path: Path) -> dict:
    """Parse a markdown file into an addressable structure.

    Pointers
      /md_tables/<t>/rows/<rowkey>/<col>/tok<j>   numeric tokens inside tables
      /md_prose/tok<j>                            numeric tokens in prose
      /md_code_blocks/<k>/tok<j>                  numeric tokens in fenced code
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    index = {"tables": [], "prose": [], "code_blocks": []}
    table_no = -1
    code_no = -1
    in_fence = False
    cur_table = None

    def emit(bucket: list, line_no: int, col: int, raw: str, value: float) -> None:
        bucket.append({"line": line_no + 1, "col": col + 1, "raw": raw, "value": value})

    for line_no, line in enumerate(lines):
        if FENCE_RE.match(line):
            if not in_fence:
                code_no += 1
                index["code_blocks"].append({"open_line": line_no + 1, "tokens": []})
            in_fence = not in_fence
            continue

        if in_fence:
            for m in TOKEN_RE.finditer(line):
                emit(index["code_blocks"][code_no]["tokens"], line_no, m.start(1),
                     m.group(1), float(m.group(1)))
            continue

        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if TABLE_SEP_RE.match(line):
                continue
            if cur_table is None:
                table_no += 1
                cur_table = {"header_line": line_no + 1, "headers": cells, "rows": []}
                index["tables"].append(cur_table)
                continue
            header = cur_table["headers"]
            key = strip_markup(cells[0]) if cells else f"row{len(cur_table['rows'])}"
            row = {"key": key, "line": line_no + 1, "tokens": {}}
            for ci, cell in enumerate(cells):
                col = strip_markup(header[ci]) if ci < len(header) else f"col{ci}"
                toks = []
                # column offset of this cell inside the raw line
                base = line.index(cell) if cell else 0
                for m in TOKEN_RE.finditer(cell):
                    toks.append({"line": line_no + 1, "col": base + m.start(1) + 1,
                                 "raw": m.group(1), "value": float(m.group(1))})
                row["tokens"][col] = toks
            cur_table["rows"].append(row)
            continue

        # table terminated by a non-table line
        cur_table = None
        for m in TOKEN_RE.finditer(line):
            emit(index["prose"], line_no, m.start(1), m.group(1), float(m.group(1)))

    return index


def md_leaves(path: Path) -> list[dict]:
    idx = index_markdown(path)
    out: list[dict] = []
    for t, table in enumerate(idx["tables"]):
        seen: dict[str, int] = {}
        for row in table["rows"]:
            key = row["key"]
            if key in seen:
                seen[key] += 1
                key = f"{key}~{seen[key]}"
            else:
                seen[key] = 0
            for col, toks in row["tokens"].items():
                for j, tok in enumerate(toks):
                    out.append({
                        "pointer": (f"/md_tables/{t}/rows/{json_pointer_escape(key)}"
                            f"/{json_pointer_escape(col)}/tok{j}"),
                        "value": tok["value"], "type": "float",
                        "locator": {"line": tok["line"], "col": tok["col"], "raw": tok["raw"]},
                    })
    for j, tok in enumerate(idx["prose"]):
        out.append({
            "pointer": f"/md_prose/tok{j}", "value": tok["value"], "type": "float",
            "locator": {"line": tok["line"], "col": tok["col"], "raw": tok["raw"]},
        })
    for k, block in enumerate(idx["code_blocks"]):
        for j, tok in enumerate(block["tokens"]):
            out.append({
                "pointer": f"/md_code_blocks/{k}/tok{j}", "value": tok["value"],
                "type": "float",
                "locator": {"line": tok["line"], "col": tok["col"], "raw": tok["raw"]},
            })
    return out


# ------------------------------------------------------------- readback pathways
def readback_json(path: Path, leaves: list[dict]) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    bad = []
    for leaf in leaves:
        try:
            got = resolve_pointer(doc, leaf["pointer"])
        except Exception as exc:  # pragma: no cover - pointer must resolve
            bad.append({"pointer": leaf["pointer"], "reason": f"unresolvable: {exc}"})
            continue
        if kind_of(got) != leaf["type"] or repr(got) != repr(leaf["value"]):
            bad.append({"pointer": leaf["pointer"], "reason": "value/type mismatch",
                        "recorded": repr(leaf["value"]), "readback": repr(got)})
    return bad


def readback_md(path: Path, leaves: list[dict]) -> list[dict]:
    """Char-level re-read: reopen the file, go to the recorded line/column and
    re-parse the literal token, independent of the index used to write."""
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    bad = []
    for leaf in leaves:
        loc = leaf["locator"]
        line = raw_lines[loc["line"] - 1]
        start = loc["col"] - 1
        raw = line[start:start + len(loc["raw"])]
        if raw != loc["raw"]:
            bad.append({"pointer": leaf["pointer"], "reason": "char-level drift",
                        "recorded": loc["raw"], "readback": raw})
            continue
        if float(raw) != leaf["value"] or repr(float(raw)) != repr(leaf["value"]):
            bad.append({"pointer": leaf["pointer"], "reason": "value mismatch",
                        "recorded": repr(leaf["value"]), "readback": repr(float(raw))})
    return bad


# ------------------------------------------------------------------------ main
def main() -> int:
    generated_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        commit = git("rev-parse", "HEAD")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(git("status", "--porcelain"))
    except Exception:  # pragma: no cover - detached worktree
        commit, branch, dirty = "unknown", "unknown", None

    sources_meta = []
    all_leaves = []
    for sid, rel, kind, purpose in SOURCES:
        path = REPO / rel
        if not path.is_file():
            print(f"FATAL: missing source {rel}", file=sys.stderr)
            return 2
        digest = sha256_file(path)
        if kind == "json":
            leaves = []
            walk_json(json.loads(path.read_text(encoding="utf-8")), "", leaves)
        elif kind == "md":
            leaves = md_leaves(path)
        else:  # pragma: no cover
            raise ValueError(kind)
        for leaf in leaves:
            leaf["source_id"] = sid
            leaf["source_path"] = rel
            leaf["source_sha256"] = digest
        readback = (readback_json if kind == "json" else readback_md)(path, leaves)
        sources_meta.append({
            "source_id": sid, "path": rel, "kind": kind, "purpose": purpose,
            "sha256": digest, "bytes": path.stat().st_size,
            "n_scalar_leaves": len(leaves),
            "n_readback_checked": len(leaves),
            "n_readback_mismatch": len(readback),
            "readback_method": ("rfc6901_pointer_resolution+repr_type_compare"
                                if kind == "json" else
                                "recorded_line_col_char_reread+float_compare"),
            "readback_mismatches": readback[:10],
        })
        all_leaves.extend(leaves)

    # completeness: every enumerated leaf is present exactly once, and no leaf
    # failed readback
    pointers = [f"{leaf['source_id']}{leaf['pointer']}" for leaf in all_leaves]
    dupes = sorted({p for p in pointers if pointers.count(p) > 1})
    missing = [m["pointer"] for s in sources_meta for m in s["readback_mismatches"]]
    n_expected = sum(s["n_scalar_leaves"] for s in sources_meta)
    completeness_pct = 100.0 * (n_expected - len(missing)) / n_expected if n_expected else 0.0

    artifact = {
        "schema": "suboff-anchor-transcription/v1",
        "task_key": TASK_KEY,
        "mission_id": MISSION_ID,
        "domain": "ship-total-performance / resistance",
        "case": "DTRC SUBOFF AFF-8 (Model 4740, with appendages) + LBM case suboff_sail_resistance",
        "title": ("W22 — DTRC SUBOFF AFF-8 外部锚逐字段机读抄录（RFC-6901 指针 + 回读 + sha256）"),
        "predecessor": {
            "task_key": PREDECESSOR,
            "gap": "predecessor produced no report and no artifact (零报告欠账); this task explicitly inherits the gap",
        },
        "generated_by": str(HERE.relative_to(REPO)),
        "generated_utc": generated_utc,
        "git": {"commit": commit, "branch": branch, "worktree_dirty_at_build": dirty},
        "scope": {
            "declared": [
                "all scalar leaves (str/int/float/bool/null) of JSON sources S1,S2,S4,S5,S6,S7",
                "all numeric tokens of markdown sources S3,S8 — table cells, prose and fenced code blocks",
            ],
            "pointer_scheme": {
                "json": "RFC-6901 JSON pointer, empty string = document root",
                "markdown": ("/md_tables/<t>/rows/<row_key>/<column>/tok<j>, "
                             "/md_prose/tok<j>, /md_code_blocks/<k>/tok<j> — "
                             "document-order indices; row_key = first-column text, "
                             "~n suffix disambiguates repeated keys"),
            },
            "exclusions": [],
            "exclusion_note": ("no category is excluded: markdown numeric tokens include "
                               "section numbers and command-line arguments, which are kept "
                               "verbatim rather than filtered, so the census is exhaustive "
                               "rather than chosen"),
            "numeric_token_grammar": NUM,
        },
        "sources": sources_meta,
        "n_leaves_total": len(all_leaves),
        "duplicate_pointers": dupes,
        "completeness": {
            "n_expected": n_expected,
            "n_transcribed": len(all_leaves),
            "n_missing_or_mismatched": len(missing),
            "missing_list": missing,
            "completeness_pct": completeness_pct,
            "completeness_100pct": (completeness_pct == 100.0 and not dupes),
        },
        "leaves": all_leaves,
    }
    OUT.write_text(json.dumps(artifact, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    shown = OUT.relative_to(REPO) if OUT.is_relative_to(REPO) else OUT
    print(f"wrote {shown}: {len(all_leaves)} leaves from "
          f"{len(sources_meta)} sources, completeness={completeness_pct:.4f}%, "
          f"mismatches={len(missing)}")
    return 0 if artifact["completeness"]["completeness_100pct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())