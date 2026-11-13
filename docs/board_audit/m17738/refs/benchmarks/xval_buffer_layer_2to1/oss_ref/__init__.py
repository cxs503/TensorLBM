"""Pinned, executable copy of the OSS reference (haimingz/vivsim multigrid).

See ``PROVENANCE.json`` for the upstream commit, file path, licence, and the
sha256 of both the unmodified upstream file and the shimmed copy.  The shimmed
copy changes only the import block; every function body is byte-identical.
"""

from __future__ import annotations

from pathlib import Path

OSS_DIR = Path(__file__).resolve().parent
UPSTREAM_PATH = OSS_DIR / "vivsim_multigrid_upstream.py"
SHIMMED_PATH = OSS_DIR / "vivsim_multigrid.py"

#: sha256 of the unmodified upstream file as vendored on 2026-11-13
UPSTREAM_SHA256 = "d104683cfad3c3114f74e42aebf2678cec1de4d63917cb1d5c13c97a72af6515"
#: upstream commit recorded in oss_inventory (id=11, repo haimingz/vivsim)
UPSTREAM_COMMIT = "fb84b12f02e18f3b9dfdf738d6a1a318d81539f9"
UPSTREAM_FILE = "vivsim/multigrid.py"
UPSTREAM_LICENSE = "MIT"

RELEVANT_LINES = {
    "init_grid": "21-52",
    "fine_to_coarse": "55-95",
    "coarse_to_fine": "98-131",
    "get_omega": "134-148",
    "coord_to_indices": "151-175",
}
