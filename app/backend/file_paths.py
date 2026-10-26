"""Resolved path boundaries for job-owned files.

These checks reject traversal and existing symlinks outside the job root.
They are not an atomic open: output directories must not be writable by an
untrusted actor who can replace symlinks between validation and file access.
"""

from __future__ import annotations

from pathlib import Path


def safe_job_path(root: Path, path: str | Path) -> Path | None:
    """Resolve a path within *root*, or return None when it escapes.

    Absolute paths already inside the root remain supported. Missing paths
    are allowed through so callers can retain their existing 404 handling.
    """
    try:
        resolved_root = root.resolve()
        target = (root / path).resolve()
        return target if target.is_relative_to(resolved_root) else None
    except (OSError, RuntimeError, ValueError):
        # Includes invalid paths and symlink loops; fail closed.
        return None
