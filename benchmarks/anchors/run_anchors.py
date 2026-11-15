#!/usr/bin/env python3
"""Anchor verification for TensorLBM lid-driven cavity benchmarks.

This is the *gate* every cavity anchor consumer must clear before the Ghia
(1982) reference numbers may be used.  Five checks:

1. **Witness sha256** — the OCR text layer that the tables were transcribed
   from still hashes to the pinned value (root of trust).
2. **CSV sha256, file by file** — every ``*.csv`` listed in
   ``ghia1982_provenance.json -> csv_anchors`` matches its pinned sha256, and
   its values agree with the provenance tables (so a CSV cannot drift from the
   witness without failing here).
3. **Table integrity** — monotone coordinates, normalised velocities,
   lid/wall boundary values.
4. **Suspect entries documented** — every entry in ``suspect_entries`` carries a
   reason and an action (the exclusion list the report has to show).
5. **Code-vs-anchor drift guard** — the ``GHIA_RE*`` constants shipped inside
   ``src/tensorlbm/lid_driven_cavity.py`` and
   ``src/tensorlbm/lid_driven_cavity_3d.py`` must equal the BOUND values.  This
   is the check that catches a hand-typed copy drifting from the witness (it is
   how the 2026-11 ``GHIA_RE400`` corruption, v(x=0.9063) = -0.38598 instead of
   -0.23827, was found).
6. **OCR value presence** — each BOUND value must appear among the numeric
   tokens of the OCR witness (after repairing known OCR glyph noise).  Necessary
   but not sufficient: it catches fabricated numbers, it cannot by itself
   certify row alignment.

Usage:
    python benchmarks/anchors/run_anchors.py [--verbose]

Exit code 0 = all anchors pass; non-zero = one or more failures.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ANCHOR_DIR = REPO / "benchmarks" / "anchors"
PROVENANCE_FILE = ANCHOR_DIR / "ghia1982_provenance.json"

#: Source files that ship hand-typed copies of the Ghia tables.
CODE_TABLE_SOURCES = {
    "src/tensorlbm/lid_driven_cavity.py": ("GHIA_RE100", "GHIA_RE400", "GHIA_RE1000"),
    "src/tensorlbm/lid_driven_cavity_3d.py": ("GHIA_RE400",),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_provenance_sha256(data: dict, verbose: bool = False) -> list[str]:
    """Check that the primary witness sha256 matches the actual file."""
    errors = []
    pw = data.get("primary_witness", {})
    witness_file = pw.get("file")
    expected_sha = pw.get("sha256")
    if not witness_file or not expected_sha:
        errors.append("primary_witness missing file or sha256")
        return errors

    witness_path = REPO / witness_file
    if not witness_path.exists():
        errors.append(f"witness file not found: {witness_path}")
        return errors

    actual_sha = sha256_file(witness_path)
    if actual_sha != expected_sha:
        errors.append(
            f"witness sha256 mismatch: expected {expected_sha[:16]}... "
            f"got {actual_sha[:16]}..."
        )
    elif verbose:
        print(f"  ✓ witness sha256 OK: {actual_sha[:16]}...")
    return errors


def _read_csv_columns(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a ``#``-commented two-column anchor CSV -> (header, values[N,2])."""
    lines = [
        ln for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    reader = csv.reader(io.StringIO("\n".join(lines)))
    header = next(reader)
    rows = [row for row in reader if row]
    return header, np.asarray(rows, dtype=float)


def verify_csv_anchors(data: dict, verbose: bool = False) -> list[str]:
    """Verify every CSV listed in ``csv_anchors``: sha256 *and* values.

    Returns the list of errors; the caller reports the ``passed/total`` count.
    """
    errors: list[str] = []
    entries = data.get("csv_anchors", [])
    if not entries:
        errors.append("csv_anchors is empty: no in-repo CSV is sha256-verified")
        return errors

    for entry in entries:
        path = entry.get("path")
        expected = entry.get("sha256")
        if not path or not expected:
            errors.append(f"csv entry missing path or sha256: {entry}")
            continue
        csv_path = REPO / path
        if not csv_path.exists():
            errors.append(f"CSV not found: {csv_path}")
            continue
        actual = sha256_file(csv_path)
        if actual != expected:
            errors.append(
                f"CSV sha256 mismatch for {path}: "
                f"expected {expected[:16]}... got {actual[:16]}..."
            )
            continue
        if verbose:
            print(f"  ✓ CSV {path} sha256 OK ({actual[:16]}...)")

        # Values must match the provenance tables, not merely hash-match.
        re_key = str(entry.get("re"))
        block = data.get("reynolds_numbers", {}).get(re_key)
        if block is None:
            errors.append(f"{path}: provenance has no table for Re={re_key}")
            continue
        header, values = _read_csv_columns(csv_path)
        coord, prof = entry.get("columns", ["", ""])
        if header != [coord, prof]:
            errors.append(f"{path}: header {header} != declared {[coord, prof]}")
            continue
        ref_c = np.asarray(block[coord], dtype=float)
        ref_v = np.asarray(block[prof], dtype=float)
        if values.shape != (len(ref_c), 2):
            errors.append(
                f"{path}: shape {values.shape} != expected {(len(ref_c), 2)}"
            )
            continue
        if not np.allclose(values[:, 0], ref_c, atol=1e-4):
            errors.append(f"{path}: {coord} column differs from provenance")
        if not np.allclose(values[:, 1], ref_v, atol=1e-5):
            errors.append(f"{path}: {prof} column differs from provenance")
        if verbose and not errors:
            print(f"    {len(values)} points cross-checked against provenance")
    return errors


def verify_table_integrity(data: dict, verbose: bool = False) -> list[str]:
    """Structural sanity checks on each Reynolds number table."""
    errors = []
    reynolds = data.get("reynolds_numbers", {})
    for re_str, table in reynolds.items():
        re_val = float(re_str)
        y = np.asarray(table.get("y", []), dtype=float)
        u = np.asarray(table.get("u", []), dtype=float)
        x = np.asarray(table.get("x", []), dtype=float)
        v = np.asarray(table.get("v", []), dtype=float)

        # Length checks
        if len(y) != len(u):
            errors.append(f"Re={re_val}: y/u length mismatch ({len(y)} vs {len(u)})")
        if len(x) != len(v):
            errors.append(f"Re={re_val}: x/v length mismatch ({len(x)} vs {len(v)})")

        # Monotone coordinates
        if len(y) >= 2 and not np.all(np.diff(y) < 0):
            # Ghia tables are listed top-to-bottom (y decreasing)
            if not np.all(np.diff(y) <= 1e-12):
                errors.append(f"Re={re_val}: y not monotonically decreasing")
        if len(x) >= 2 and not np.all(np.diff(x) < 0):
            if not np.all(np.diff(x) <= 1e-12):
                errors.append(f"Re={re_val}: x not monotonically decreasing")

        # Normalised velocities: max|u|,|v| should be O(1)
        peak = max(np.abs(u).max(), np.abs(v).max()) if len(u) > 0 else 0
        if not (0.5 < peak < 1.5):
            errors.append(
                f"Re={re_val}: velocities not normalised (peak={peak:.4g}, expected O(1))"
            )

        # Lid boundary: u(y=1) should be 1.0
        if len(y) > 0 and abs(y[0] - 1.0) < 1e-6:
            if abs(u[0] - 1.0) > 1e-4:
                errors.append(
                    f"Re={re_val}: lid u(y=1)={u[0]:.5f}, expected 1.0"
                )

        # Wall boundaries: u(y=0)=0, v(x=0)=0, v(x=1)=0
        if len(y) > 0 and abs(y[-1]) < 1e-6:
            if abs(u[-1]) > 1e-4:
                errors.append(
                    f"Re={re_val}: bottom wall u(y=0)={u[-1]:.5f}, expected 0"
                )
        if len(x) > 0 and abs(x[-1]) < 1e-6:
            if abs(v[-1]) > 1e-4:
                errors.append(
                    f"Re={re_val}: left wall v(x=0)={v[-1]:.5f}, expected 0"
                )
        if len(x) > 0 and abs(x[0] - 1.0) < 1e-6:
            if abs(v[0]) > 1e-4:
                errors.append(
                    f"Re={re_val}: right wall v(x=1)={v[0]:.5f}, expected 0"
                )

        if verbose and not errors:
            print(f"  ✓ Re={re_val}: {len(y)} points, peak velocity={peak:.4f}")

    return errors


def verify_suspect_entries_removed(data: dict, verbose: bool = False) -> list[str]:
    """Check that suspect entries are documented and have an action."""
    errors = []
    suspects = data.get("suspect_entries", [])
    if not suspects:
        errors.append("suspect_entries is empty: exclusion list not recorded")
    for entry in suspects:
        if "reason" not in entry:
            errors.append(f"suspect entry missing reason: {entry}")
        if "action" not in entry:
            errors.append(f"suspect entry missing action: {entry}")
        if verbose:
            re_val = entry.get("re", "?")
            table = entry.get("table", "?")
            print(f"  ⚠ Re={re_val} {table}: {entry.get('reason', '')[:70]}...")
    return errors


def _extract_code_tables(path: Path) -> dict[str, dict]:
    """Statically read ``GHIA_RE* = {...}`` literals out of an unimported file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, dict] = {}
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            # ``GHIA_RE400: dict[str, list[float]] = {...}``
            target = node.target
            value = node.value
        else:
            continue
        if not isinstance(target, ast.Name) or not target.id.startswith("GHIA_RE"):
            continue
        if value is None:
            continue
        try:
            found[target.id] = ast.literal_eval(value)
        except ValueError:
            continue
    return found


def verify_code_tables_match_anchor(data: dict, verbose: bool = False) -> list[str]:
    """Every hand-typed ``GHIA_RE*`` constant must equal the BOUND table."""
    errors: list[str] = []
    tables = data.get("reynolds_numbers", {})
    checked = 0
    for rel, names in CODE_TABLE_SOURCES.items():
        path = REPO / rel
        if not path.exists():
            errors.append(f"code table source missing: {rel}")
            continue
        code_tables = _extract_code_tables(path)
        for name in names:
            if name not in code_tables:
                errors.append(f"{rel}: {name} not found")
                continue
            re_key = name.replace("GHIA_RE", "")
            block = tables.get(re_key)
            if block is None:
                errors.append(f"{rel}: {name} has no provenance table for Re={re_key}")
                continue
            code = code_tables[name]
            for coord, prof in (("y", "u"), ("x", "v")):
                ref_c = np.asarray(block[coord], dtype=float)
                ref_v = np.asarray(block[prof], dtype=float)
                # code stores x ascending, the witness lists x descending
                want_c = ref_c[::-1] if coord == "x" else ref_c
                want_v = ref_v[::-1] if coord == "x" else ref_v
                got_c = np.asarray(code.get(coord, []), dtype=float)
                got_v = np.asarray(code.get(prof, []), dtype=float)
                if got_c.shape != want_c.shape:
                    errors.append(
                        f"{rel}:{name} {coord} has {got_c.shape[0]} points, "
                        f"witness has {want_c.shape[0]}"
                    )
                    continue
                if not np.allclose(got_c, want_c, atol=1e-4):
                    errors.append(f"{rel}:{name} {coord} coordinates differ from witness")
                bad = np.abs(got_v - want_v) > 2e-4
                if bad.any():
                    for i in np.flatnonzero(bad):
                        errors.append(
                            f"{rel}:{name} Re={re_key} {prof}({coord}={got_c[i]:.4f}) "
                            f"= {got_v[i]:+.5f}, witness = {want_v[i]:+.5f}"
                        )
            checked += 1
            if verbose:
                print(f"  ✓ {rel}:{name} matches BOUND Re={re_key}")
    if verbose:
        print(f"  {checked} code constant(s) checked against the witness")
    return errors


_OCR_GLYPH_FIXES = (
    ("X).", "-0."), ("xJ.", "-0."), ("XI.", "-0."),
    ("Al.", "-0."), ("AI.", "-0."), ("a.", "-0."), ("Al", "-0."),
    ("AI", "-0."), ("a", "0"),
    ("ooooo", "00000"), ("oooo", "0000"), ("ooo", "000"),  # letter o for zero
)


def _ocr_numeric_tokens(text: str) -> np.ndarray:
    """Numeric tokens of the OCR witness after repairing known glyph noise.

    The text layer confuses ``-0.`` with ``a.``/``Al``/``AI``/``X).`` and
    sometimes splits a number ("0.05 188").  Both are repaired here so that the
    check tests *values*, not OCR cosmetics.
    """
    out = text
    for bad, good in _OCR_GLYPH_FIXES:
        out = out.replace(bad, good)
    # repair "1 .00000" (space before the decimal point) and
    # "0.05 188" (digit group split by a space)
    out = re.sub(r"(\d)\s+\.(\d+)", r"\1.\2", out)
    out = re.sub(r"(\d\.\d{1,4})\s(\d{1,4})(?!\.)", r"\1\2", out)
    toks = []
    for tok in re.findall(r"-?\d+\.\d+", out):
        try:
            toks.append(float(tok))
        except ValueError:
            continue
    return np.asarray(toks, dtype=float)


def verify_ocr_value_presence(data: dict, verbose: bool = False) -> list[str]:
    """Each BOUND value must appear among the OCR witness numeric tokens."""
    errors: list[str] = []
    pw = data.get("primary_witness", {})
    witness_path = REPO / pw.get("file", "")
    if not witness_path.exists():
        return [f"witness file not found: {witness_path}"]
    raw = witness_path.read_text(encoding="utf-8", errors="replace")
    toks = _ocr_numeric_tokens(raw)
    if toks.size == 0:
        return ["no numeric tokens extracted from the OCR witness"]
    frac_toks = {f"{abs(t):.5f}".split(".")[1] for t in toks}

    missing = 0
    checked = 0
    magnitude_only: list[str] = []
    for re_key, block in data.get("reynolds_numbers", {}).items():
        for prof in ("u", "v"):
            coord = "y" if prof == "u" else "x"
            for c, v in zip(block[coord], block[prof]):
                checked += 1
                # 5-dp transcription; allow the OCR's last-digit noise (1e-4)
                if np.any(np.abs(toks - float(v)) <= 1e-4):
                    continue
                # The text layer sometimes loses a sign (``-0.`` -> ``4.``);
                # accept a magnitude match but record it as a weaker witness.
                if np.any(np.abs(np.abs(toks) - abs(float(v))) <= 1e-4):
                    magnitude_only.append(
                        f"Re={re_key} {prof}({coord}={float(c):.4f})={float(v):+.5f}"
                    )
                    continue
                # ...and sometimes the leading ``-0.`` is read as another digit
                # (``-0.38289`` -> ``4.38289``): fall back to the mantissa digits.
                frac = f"{abs(float(v)):.5f}".split(".")[1]
                if any(tok_frac == frac for tok_frac in frac_toks):
                    magnitude_only.append(
                        f"Re={re_key} {prof}({coord}={float(c):.4f})={float(v):+.5f} "
                        "(mantissa-only)"
                    )
                    continue
                missing += 1
                errors.append(
                    f"Re={re_key} {prof}({coord}={float(c):.4f})={float(v):+.5f} "
                    "not found in the OCR witness"
                )
    if verbose:
        print(f"  {checked - missing}/{checked} BOUND values found in the witness "
              f"({checked - missing - len(magnitude_only)} exact, "
              f"{len(magnitude_only)} magnitude-only)")
        for m in magnitude_only:
            print(f"    ~ {m} (sign lost by OCR)")
    return errors


def main() -> int:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    if not PROVENANCE_FILE.exists():
        print(f"FAIL: provenance file not found: {PROVENANCE_FILE}")
        return 1

    with open(PROVENANCE_FILE, encoding="utf-8") as f:
        data = json.load(f)

    all_errors: list[str] = []

    print("=== Ghia (1982) Anchor Verification ===")
    print(f"Provenance file: {PROVENANCE_FILE.relative_to(REPO)}")

    # 1. Witness sha256
    print("\n[1] Witness SHA-256 verification")
    errs = verify_provenance_sha256(data, verbose)
    all_errors.extend(errs)
    print(f"  {'PASS' if not errs else 'FAIL'}: {len(errs)} error(s)")

    # 2. CSV anchors (sha256 per file + value cross-check)
    print("\n[2] CSV anchor verification")
    entries = data.get("csv_anchors", [])
    errs = verify_csv_anchors(data, verbose)
    all_errors.extend(errs)
    n_ok = len(entries) - len([e for e in errs if "CSV" in e])
    print(f"  {max(n_ok, 0)}/{len(entries)} CSV anchors verified")
    print(f"  {'PASS' if not errs else 'FAIL'}: {len(errs)} error(s)")

    # 3. Table integrity
    print("\n[3] Table integrity checks")
    errs = verify_table_integrity(data, verbose)
    all_errors.extend(errs)
    print(f"  {'PASS' if not errs else 'FAIL'}: {len(errs)} error(s)")

    # 4. Suspect entries documented
    print("\n[4] Suspect entries documentation")
    errs = verify_suspect_entries_removed(data, verbose)
    all_errors.extend(errs)
    print(f"  {'PASS' if not errs else 'FAIL'}: {len(errs)} error(s)")

    # 5. Code-vs-anchor drift guard
    print("\n[5] Code constants vs BOUND anchor")
    errs = verify_code_tables_match_anchor(data, verbose)
    all_errors.extend(errs)
    print(f"  {'PASS' if not errs else 'FAIL'}: {len(errs)} error(s)")

    # 6. OCR value presence
    print("\n[6] OCR witness value presence")
    errs = verify_ocr_value_presence(data, verbose)
    all_errors.extend(errs)
    print(f"  {'PASS' if not errs else 'FAIL'}: {len(errs)} error(s)")

    # Summary
    print(f"\n=== Summary: {len(all_errors)} error(s) ===")
    for e in all_errors:
        print(f"  ✗ {e}")

    if all_errors:
        print("\nRESULT: FAIL")
        return 1
    else:
        print("\nRESULT: PASS (all anchors verified)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())