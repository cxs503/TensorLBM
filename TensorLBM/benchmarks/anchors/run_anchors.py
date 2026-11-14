#!/usr/bin/env python3
"""Anchor verification for TensorLBM lid-driven cavity benchmarks.

Verifies that:
1. The Ghia (1982) provenance JSON has a valid sha256 for the OCR witness file
2. Any CSV reference files match their recorded sha256 hashes
3. The reference data passes structural sanity checks (normalised velocities,
   monotone coordinates, etc.)

Usage:
    python benchmarks/anchors/run_anchors.py [--verbose]

Exit code 0 = all anchors pass; non-zero = one or more failures.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ANCHOR_DIR = REPO / "benchmarks" / "anchors"
PROVENANCE_FILE = ANCHOR_DIR / "ghia1982_provenance.json"


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


def verify_csv_anchors(data: dict, verbose: bool = False) -> list[str]:
    """Verify any CSV files listed in the provenance against recorded hashes."""
    errors = []
    csv_entries = data.get("csv_anchors", [])
    for entry in csv_entries:
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
        elif verbose:
            print(f"  ✓ CSV {path} sha256 OK")
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
    for entry in suspects:
        if "reason" not in entry:
            errors.append(f"suspect entry missing reason: {entry}")
        if "action" not in entry:
            errors.append(f"suspect entry missing action: {entry}")
        if verbose:
            re_val = entry.get("re", "?")
            print(f"  ⚠ Re={re_val}: {entry['reason'][:80]}...")
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

    # 2. CSV anchors
    print("\n[2] CSV anchor verification")
    errs = verify_csv_anchors(data, verbose)
    all_errors.extend(errs)
    if not errs:
        print("  PASS: no CSV anchors to verify (or all OK)")
    else:
        print(f"  FAIL: {len(errs)} error(s)")

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
