"""Zero-hand-copy gate for the validation-matrix reference anchors.

Every number in ``references.json`` must be re-derivable from a frozen in-repo
source: the file exists, it still contains the literal token that carries the
number, and the typed value equals a float parsed out of that token (or the
frozen artifact field it is claimed to mirror).  The source sha256 is recorded
so a silent edit upstream is detectable instead of being absorbed.

Run standalone::

    python benchmarks/validation_matrix/reference_audit.py [--json]

Exit code 0 = every anchor audited clean, 1 = at least one anchor failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
REFERENCES = HERE / "references.json"

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", text)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dotted(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def load_references() -> dict:
    return json.loads(REFERENCES.read_text(encoding="utf-8"))["references"]


def audit_reference(name: str, spec: dict) -> dict:
    """Audit one anchor; never raises, reports ``pass`` with a reason list."""
    reasons: list[str] = []
    record: dict = {"id": name, "pass": True, "reasons": reasons}

    source = REPO_ROOT / spec["source"]
    record["source"] = spec["source"]
    if not source.exists():
        reasons.append(f"source file missing: {spec['source']}")
        record["pass"] = False
        return record

    text = normalize(source.read_text(encoding="utf-8", errors="replace"))
    token = normalize(spec["token"])
    record["source_sha256"] = sha256_file(source)
    record["token"] = spec["token"]
    record["token_found"] = token in text
    if not record["token_found"]:
        reasons.append(f"token not found in {spec['source']}: {spec['token']!r}")

    value = spec.get("value")
    if value is not None:
        record["value"] = value
        if spec.get("value_token_check", True):
            floats = [float(x) for x in _FLOAT_RE.findall(spec["token"])]
            record["token_floats"] = floats
            rel = [
                abs(f - float(value)) / max(abs(float(value)), 1e-300) for f in floats
            ]
            best = min(rel) if rel else float("inf")
            record["value_matches_token_rel"] = best
            # rounded tokens ("0.041995") legitimately differ in the last
            # digits from the frozen full-precision value.
            if best > 5e-3:
                reasons.append(
                    f"value {value!r} does not match any float in the token "
                    f"(closest relative difference {best:.3e} > 5e-3)"
                )

    art = spec.get("artifact_value_path")
    if art:
        path_str, field = art.split("::", 1)
        art_path = REPO_ROOT / path_str
        if not art_path.exists():
            reasons.append(f"artifact missing: {path_str}")
        else:
            try:
                frozen = float(_dotted(json.loads(art_path.read_text()), field))
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                reasons.append(f"artifact field {field!r} unreadable: {exc}")
            else:
                record["artifact_value"] = frozen
                record["artifact_sha256"] = sha256_file(art_path)
                denom = max(abs(frozen), 1e-300)
                if value is not None and abs(frozen - float(value)) / denom > 5e-3:
                    reasons.append(
                        f"typed value {value!r} disagrees with frozen artifact "
                        f"{frozen!r}"
                    )

    record["pass"] = not reasons
    return record


def audit_all() -> dict:
    refs = load_references()
    records = [audit_reference(name, spec) for name, spec in refs.items()]
    return {
        "registry": str(REFERENCES.relative_to(REPO_ROOT)),
        "n_references": len(records),
        "n_passed": sum(1 for r in records if r["pass"]),
        "pass": all(r["pass"] for r in records),
        "references": records,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="dump the full audit report")
    args = ap.parse_args(argv)
    report = audit_all()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for r in report["references"]:
            mark = "OK  " if r["pass"] else "FAIL"
            extra = "" if r["pass"] else " | " + "; ".join(r["reasons"])
            print(f"[{mark}] {r['id']:<24} <- {r.get('source', '?')}{extra}")
        print(f"\n{report['n_passed']}/{report['n_references']} anchors audited clean")
    return 0 if report["pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())