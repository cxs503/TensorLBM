#!/usr/bin/env python3
"""Horizontal OSS survey: how do cached LBM codes encode boundaries?

Round 2 of ``XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND`` has to answer whether
a *partially covered* cell (a wall inside a cell) is an idiom other mature
codes already provide, and if so in which form.  This script machine-extracts
the relevant constructs from the repositories cached on the cluster nodes
(``~/oss-cache``) -- file lines and file sha256, nothing transcribed by hand --
and writes ``artifacts/xval_per_step_hook_mask_blend/oss_ref/oss_cache_survey.json``.

It is a *survey*, not a port: the goal is to decide whether the fractional-mask
question is answered by transplanting another project.

Run: python benchmarks/xval_per_step_hook_mask_blend/survey_oss_cache.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "xval_per_step_hook_mask_blend" / "oss_ref" / "oss_cache_survey.json"
DEFAULT_ROOTS = (Path.home() / "oss-cache",)

# needle -> (capability tag, what finding it evidences)
NEEDLES = {
    "boundary.py": {
        "class BounceBack(": ("binary_bounce_back", "bounce-back method class"),
        "class BFLBounceBack(": ("bfl_bounce_back", "BFL (Ladd-type) bounce-back"),
        "class LinearBounceBack(": ("interpolated_bounce_back", "interpolated (Bouzidi) bounce-back, sub-cell wall distance"),
        "class LinearAntiBounceBack(": ("interpolated_anti_bounce_back", "interpolated anti bounce-back"),
        "def _alpha_bfl(": ("interpolation_weight", "interpolation weight from the wall distance"),
        "class BoundaryMethod(": ("declarative_bc", "BC declared per element/stencil, applied by the solver"),
    },
    "streamcollide.py": {
        "def lbm_streaming(": ("streaming_with_obstacles", "streaming loop with obstacle branch"),
        "else:  # obstacle": ("binary_obstacle", "obstacle handled by a branch, not by a weight"),
        "def get_epsilon(": ("interface_mass", "interface cell fluid fraction (multiphase mass, not solid fraction)"),
        "def interface_mass_exchange(": ("interface_mass", "interface mass exchange"),
        "def lbm_collision(": ("collision", "BGK collision with inlet/obstacle flags"),
    },
}
# capability -> False is asserted when no needle produced it for that repo
CAPABILITY_KEYS = (
    "binary_bounce_back",
    "bfl_bounce_back",
    "interpolated_bounce_back",
    "interpolated_anti_bounce_back",
    "interpolation_weight",
    "declarative_bc",
    "streaming_with_obstacles",
    "binary_obstacle",
    "interface_mass",
    "collision",
)
# constructs that would indicate a *fractional* (float-mask / solid-fraction) wall
FRACTIONAL_PATTERNS = (
    r"solid_fraction",
    r"eps_s\b",
    r"partial(ly)?[_ ]saturat",
    r"volume[_ ]fraction.*(solid|wall)",
    r"mask.*(float|weight)",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def survey_repo(repo: Path) -> dict:
    record: dict = {
        "repo": repo.name,
        "path": str(repo),
        "files": {},
        "snippets": {},
        "capabilities": {},
        "fractional_wall_hits": [],
        "lbm": None,
    }
    for rel, needles in NEEDLES.items():
        hits = sorted(p for p in repo.rglob(rel) if ".git" not in p.parts)
        if not hits:
            continue
        path = hits[0]
        rel = str(path.relative_to(repo))
        raw = path.read_bytes()
        record["files"][rel] = sha256_bytes(raw)
        lines = raw.decode("utf-8", "replace").splitlines()
        for needle, (cap, note) in needles.items():
            for lineno, line in enumerate(lines, 1):
                if needle in line:
                    record["snippets"][f"{rel}::{needle}"] = {
                        "file": rel,
                        "line": lineno,
                        "text": line.strip(),
                        "capability": cap,
                        "note": note,
                    }
                    record["capabilities"][cap] = True
                    break
    caps = {k: bool(record["capabilities"].get(k, False)) for k in CAPABILITY_KEYS}
    record["capabilities"] = caps
    record["lbm"] = any(
        caps[k] for k in ("binary_bounce_back", "binary_obstacle", "streaming_with_obstacles")
    )
    for path in sorted(repo.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix not in (".py", ".rs", ".cl", ".h", ".cpp", ".cuh", ".md"):
            continue
        try:
            text = path.read_text("utf-8", "replace")
        except OSError:
            continue
        for pat in FRACTIONAL_PATTERNS:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                line = text.count("\n", 0, m.start()) + 1
                record["fractional_wall_hits"].append(
                    {"file": str(path.relative_to(repo)), "line": line, "pattern": pat}
                )
                break
    record["fractional_wall_hits"] = record["fractional_wall_hits"][:20]
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=None, help="cache root (repeatable)")
    args = ap.parse_args(argv)
    roots = [Path(r) for r in args.root] if args.root else list(DEFAULT_ROOTS)
    repos: list[dict] = []
    for root in roots:
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir():
                repos.append(survey_repo(child))
    payload = {
        "roots": [str(r) for r in roots],
        "repos": repos,
        "question": "does any cached OSS LBM code express a boundary by a fractional "
        "(solid-fraction) mask field rather than a binary flag or an explicit "
        "off-lattice wall distance?",
        "answer": {
            "fractional_mask_field_found": any(r["fractional_wall_hits"] for r in repos),
            "interpolated_bounce_back_found": any(
                r["capabilities"].get("interpolated_bounce_back") for r in repos
            ),
            "binary_flag_encoding_found": any(
                r["capabilities"].get("binary_obstacle") or r["capabilities"].get("binary_bounce_back")
                for r in repos
            ),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    for r in repos:
        print(f"{r['repo']}: lbm={r['lbm']} caps={[k for k, v in r['capabilities'].items() if v]} "
              f"fractional_hits={len(r['fractional_wall_hits'])}")
    print("answer:", json.dumps(payload["answer"]))
    print("->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())