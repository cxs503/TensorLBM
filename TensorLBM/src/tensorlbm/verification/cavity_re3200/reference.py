"""Reference table for the Re = 3200 lid-driven cavity (Ghia, Ghia & Shin 1982).

Anchor ``REPRO-LBM-HOU1995`` (Hou, Zou, Chen & Doolen, *J. Comput. Phys.* **118**
(1995) 329-347) quotes the Re = 3200 cavity as its highest steady case, so the
reproduction needs the Re = 3200 column of Ghia et al.'s Table I/II.  That column
was previously unusable because one OCR cell lost its decimal point
(``-0.86636`` for ``-0.06636``): the row ``y/L = 0.4531`` sits between
``u = -0.04272`` (y = 0.5) and ``u = -0.24427`` (y = 0.2813), so ``|u| > 0.24`` is
impossible there.  The admission pipeline therefore *quarantines* that single
cell: it is exported with its admissible interval, is **not** used for scoring,
and the verdict must not depend on it (:func:`scoring.quarantine_sensitivity`).

The JSON lives with the benchmark (``benchmarks/verified/cavity/re3200/reference``)
rather than in the package, so this module resolves it by walking up from the
working directory and from this file.  Nothing here is hand-typed: the file is
loaded, its recorded ``sha256`` is re-verified, and a mismatch raises instead of
silently scoring against a mutated table.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REF_NAME = "ghia_re3200_reference.json"
_REL = Path("benchmarks/verified/cavity/re3200/reference") / REF_NAME


class ReferenceError(RuntimeError):
    """Raised when the admitted reference table is missing or fails its hash gate."""


def _candidates() -> Iterable[Path]:
    """Places to look for the table, most specific first."""
    seen = set()
    roots: List[Path] = []
    here = Path(__file__).resolve()
    roots.extend(here.parents)                 # .../TensorLBM/src/... -> repo root
    roots.append(Path.cwd().resolve())
    roots.extend(Path.cwd().resolve().parents)
    for root in roots:
        for cand in (root / _REL, root / "reference" / REF_NAME):
            if cand in seen:
                continue
            seen.add(cand)
            yield cand


def reference_path(explicit: Optional[str | Path] = None) -> Path:
    """Return the path of the admitted Re = 3200 reference table."""
    if explicit is not None:
        p = Path(explicit)
        if not p.is_file():
            raise ReferenceError(f"explicit reference not found: {p}")
        return p
    for cand in _candidates():
        if cand.is_file():
            return cand
    raise ReferenceError(
        f"{REF_NAME} not found; pass an explicit path "
        f"(looked for {_REL} relative to the source tree and cwd)"
    )


def sha256_file(path: str | Path) -> str:
    """sha256 of the bytes on disk (formatting-sensitive, for the artefact ledger)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_sha256(ref: Dict[str, Any]) -> str:
    """sha256 of the canonical table content, exactly as the builder computes it.

    ``benchmarks/verified/cavity/re5000/build_reference.py`` records
    ``sha256(json.dumps(payload, sort_keys=True))`` *before* the ``sha256`` key is
    inserted, so verification must reproduce that: pop the key, dump canonically,
    hash.  Hashing the file bytes instead would fail on any re-indentation and say
    nothing about the numbers.
    """
    body = {k: v for k, v in ref.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def load_reference(path: Optional[str | Path] = None,
                   verify_hash: bool = True) -> Dict[str, Any]:
    """Load and validate the Re = 3200 reference table.

    Validation is deliberately strict: a table that lost its provenance block or
    whose recorded hash no longer matches the bytes on disk is a *different*
    table, and scoring against it would be an unverifiable claim.
    """
    p = reference_path(path)
    ref = json.loads(p.read_text(encoding="utf-8"))
    for key in ("re", "u_re3200", "v_re3200"):
        if key not in ref:
            raise ReferenceError(f"{p} lacks required key {key!r}")
    if float(ref["re"]) != 3200.0:
        raise ReferenceError(f"{p} is Re={ref['re']}, expected 3200")
    recorded = ref.get("sha256")
    actual_content = content_sha256(ref)
    if verify_hash and recorded and recorded != actual_content:
        raise ReferenceError(
            f"reference content hash mismatch for {p}: recorded {recorded}, "
            f"recomputed {actual_content}"
        )
    ref["_path"] = str(p)
    ref["_sha256_content"] = actual_content
    ref["_sha256_file"] = sha256_file(p)
    ref["_hash_verified"] = bool(recorded == actual_content)
    return ref


def admitted(ref: Dict[str, Any]) -> Tuple[List[Tuple[float, float]],
                                           List[Tuple[float, float]]]:
    """Return the *scored* ``(u_re3200, v_re3200)`` rows as float tuples."""
    u = [(float(a), float(b)) for a, b in ref["u_re3200"]]
    v = [(float(a), float(b)) for a, b in ref["v_re3200"]]
    return u, v


def quarantined(ref: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Quarantined (admitted-but-not-scored) cells, empty when there are none."""
    prov = ref.get("provenance") or {}
    return list(prov.get("quarantined") or [])


def scored_coords(ref: Dict[str, Any]) -> Sequence[float]:
    u, v = admitted(ref)
    return tuple(sorted({c for c, _ in u} | {c for c, _ in v}))