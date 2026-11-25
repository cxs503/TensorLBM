#!/usr/bin/env python3
"""Check repository hygiene without importing the solver or changing files.

Checks tracked files and untracked, non-ignored files. Historical exceptions
are exact paths, not directory-wide exemptions. Ignored, untracked run outputs
are deliberately outside the check; force-added outputs are checked in CI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path, PurePosixPath

POLICY_PATH = "docs/architecture/repository_layout_policy.json"
MAX_FILE_BYTES = 5 * 1024 * 1024
GENERATED_ROOTS = {"runs", "runs_prop", "outputs"}
GENERATED_PARTS = {"__pycache__", "node_modules", ".venv", ".pytest_cache", ".ruff_cache"}
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".safetensors"}
SOURCE_BACKUP_SUFFIXES = (".bak", ".orig", ".rej", "~")
PATH_LISTS = ("root_files", "legacy_root_files", "legacy_generated_files", "legacy_model_files")
RETIRED_ROOT_DIRECTORIES = {
    ".audit",
    "BB",
    "TensorLBM",
    "_scratch",
    "artifacts",
    "audit",
    "audits",
    "cases",
    "configs",
    "demos",
    "drop_entry",
    "evidence",
    "experiments",
    "figures",
    "frontend-vue",
    "notebooks",
    "papers",
    "raw",
    "references",
    "regression_rules",
    "runs",
    "teaching",
    "tmp",
}
RETIRED_SECOND_LEVEL_DIRECTORIES = {
    ("xval", "XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI"),
    ("scripts", "verification"),
    ("scripts", "audit_m18430"),
    ("scripts", "w26_ghia"),
    ("benchmarks", "blasius_flat_plate_dev"),
    ("development", "idx"),
    ("scripts", "audit"),
    ("docs", "council"),
    ("scripts", "audit_runtest_r6"),
    ("scripts", "xval_yml_xml_pipeline"),
    ("docs", "history"),
    ("scripts", "data"),
    ("docs", "m23777"),
    ("docs", "m23124"),
    ("docs", "m23233"),
    ("docs", "m24569"),
    ("scripts", "physics_ai"),
    ("docs", "smoke"),
    ("scripts", "maintenance"),
    ("scripts", "plotting"),
    ("models", "lbm"),
    ("models", "units"),
    ("xval", "model_matrix_ci_plus_coverage"),
    ("xval", "XVAL-LBM-R4-COEXISTENCE"),
    ("xval", "XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO"),
    ("xval", "model_matrix_wall_axis"),
    ("xval", "mcmp_cross_force"),
    ("xval", "XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY"),
    ("scripts", "sdaa"),
    ("xval", "XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT"),
    ("scripts", "in25_t6"),
    ("xval", "XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST"),
    ("xval", "XVAL-LBM-PURE-FUNCTION-FSI-FIXED-POINT"),
    ("xval", "fluctuating_lbm_ghost_modes"),
    ("scripts", "p387w2_n04_sphere_force"),
    ("xval", "XVAL-LBM-TWO-CONSERVATION-ZOUHE-WITH-CHEMICAL-POT"),
    ("scripts", "p399w2_n04_curved_force"),
    ("xval", "XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO"),
    ("xval", "ludwig_fe_stress"),
    ("xval", "lbmconfig_single_point"),
    ("xval", "XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES"),
    ("xval", "no_mcmp_two_phase_only"),
    ("xval", "component_protocol_codegen"),
    ("xval", "XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING-ROTATION-R6"),
    ("xval", "fused_kernel_type_selector"),
    ("xval", "XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET"),
    ("xval", "life_geometry_config"),
    ("xval", "XVAL-LBM-TERNARY-DOUBLE-WELL-WITH-HARMONIC-LAMBDA"),
    ("xval", "XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN"),
    ("xval", "XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D"),
    ("xval", "XVAL-LBM-LINEAR-INTERPOLATED-BB"),
    ("xval", "XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACT"),
    ("xval", "XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACTOR"),
    ("xval", "XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE"),
    ("benchmarks", "xval_free_energy_binary_equilibrium"),
    ("xval", "XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS-R4"),
    ("xval", "XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS-R5"),
}


def validate_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid policy path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ValueError(f"Policy paths must be normalized repository-relative paths: {value!r}")
    return value


def load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or policy.get("schema_version") not in (1, 2):
        raise ValueError("Expected repository layout policy schema_version 1 or 2")
    path_lists = PATH_LISTS
    if policy["schema_version"] == 2:
        path_lists += ("root_directories",)
        if "historical_checkouts" in policy:
            path_lists += ("historical_checkouts",)
        if "legacy_audit_files" in policy:
            path_lists += ("legacy_audit_files",)
    expected = {"schema_version", "baseline_revision", "large_files", *path_lists}
    if set(policy) != expected:
        raise ValueError(f"Expected policy keys: {sorted(expected)}")
    for key in path_lists:
        values = policy[key]
        if not isinstance(values, list):
            raise ValueError(f"{key} must be a list")
        paths = [validate_path(value) for value in values]
        if len(paths) != len(set(paths)):
            raise ValueError(f"Duplicate paths in {key}")
        if key == "legacy_audit_files" and any(not p.startswith("audits/") for p in paths):
            raise ValueError("Legacy audit paths must be under audits/")
        if key in {"root_files", "legacy_root_files", "root_directories"} and any(
            "/" in p for p in paths
        ):
            raise ValueError(f"{key} may only contain top-level names")
    if set(policy["root_files"]) & set(policy["legacy_root_files"]):
        raise ValueError("Canonical and legacy root files must not overlap")
    if policy["schema_version"] == 2:
        restored = set(policy["root_directories"]) & RETIRED_ROOT_DIRECTORIES
        if restored:
            raise ValueError(
                "Retired root directories may not be restored through policy edits: "
                + ", ".join(sorted(restored))
            )
    if not isinstance(policy["large_files"], dict):
        raise ValueError("large_files must map exact paths to reviewed byte ceilings")
    for name, limit in policy["large_files"].items():
        validate_path(name)
        if type(limit) is not int or limit <= MAX_FILE_BYTES:
            raise ValueError(f"Invalid large-file byte ceiling for {name}")
    if not isinstance(policy["baseline_revision"], str) or not policy["baseline_revision"]:
        raise ValueError("baseline_revision must identify the original inventory")
    return policy


def candidate_files(repo: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return sorted(set(result.stdout.decode("utf-8", errors="surrogateescape").split("\0")) - {""})


def is_generated(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (
        parts[0] in GENERATED_ROOTS
        or bool(set(parts[:-1]) & GENERATED_PARTS)
        or parts[:2] == ("artifacts", "runs")
        or parts[:3] == ("development", "archived", "historical_runs")
        or parts[:3] == ("app", "frontend-vue", "dist")
        or parts[0] in {"build", "dist"}
    )


def is_source_backup(name: str) -> bool:
    path = PurePosixPath(name)
    return path.parts[:1] == ("src",) and path.name.endswith(SOURCE_BACKUP_SUFFIXES)


def check_files(repo: Path, names: list[str], policy: dict) -> list[str]:
    failures = []
    roots = set(policy["root_files"]) | set(policy["legacy_root_files"])
    generated = set(policy["legacy_generated_files"])
    models = set(policy["legacy_model_files"])
    allowed_dirs = set(policy.get("root_directories", ()))
    legacy_audits = set(policy.get("legacy_audit_files", ()))
    if "historical_checkouts" in policy:
        known = set(policy["historical_checkouts"])
        tracked = set(names)
        for name in names:
            suffix = "/.github/workflows/ci.yml"
            if name.endswith(suffix):
                checkout = name.removesuffix(suffix)
                if checkout not in known and f"{checkout}/CITATION.cff" in tracked:
                    failures.append(f"nested-checkout: {checkout}/ — review the source and archive mapping")
    unexpected_dirs = set()
    for name in names:
        path = repo / name
        try:
            # lstat avoids following symlinks outside the checkout.
            size = path.lstat().st_size
        except FileNotFoundError:
            # A tracked deletion in the working tree is not a new violation.
            continue
        if "/" not in name and name not in roots:
            failures.append(
                f"root-file: {name} — choose an owned directory or review the root policy"
            )
        directory, separator, _ = name.partition("/")
        if separator and policy["schema_version"] == 2 and directory not in allowed_dirs:
            unexpected_dirs.add(directory)
        parts = PurePosixPath(name).parts
        if tuple(parts[:2]) in RETIRED_SECOND_LEVEL_DIRECTORIES:
            failures.append(
                f"retired-directory: {'/'.join(parts[:2])}/ — "
                "use the mapped development or canonical documentation path"
            )
        if directory == "audits" and "legacy_audit_files" in policy and name not in legacy_audits:
            failures.append(f"legacy-audit: {name} — place new mission records under development/audit/")
        if is_generated(name) and name not in generated:
            failures.append(f"generated-file: {name} — keep raw outputs out of Git")
        if PurePosixPath(name).suffix.lower() in MODEL_SUFFIXES and name not in models:
            failures.append(
                f"model-file: {name} — use a versioned external artifact with a checksum"
            )
        if is_source_backup(name):
            failures.append(
                f"source-backup: {name} — keep one authoritative source file and archive snapshots under development/"
            )
        ceiling = policy["large_files"].get(name, MAX_FILE_BYTES)
        if size > ceiling:
            failures.append(f"large-file: {name} — {size} bytes exceeds reviewed ceiling {ceiling}")
    for directory in sorted(unexpected_dirs):
        failures.append(f"root-directory: {directory}/ — use an existing area or review the policy")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        policy = load_policy(repo / POLICY_PATH)
        names = candidate_files(repo)
        failures = check_files(repo, names, policy)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Repository layout check could not complete: {exc}")
        return 2
    if failures:
        print("Repository layout violations:")
        for failure in failures:
            print(f"  {failure}")
        print("Do not regenerate exceptions to hide new findings; request a policy review.")
        return 1
    print(f"Repository layout OK ({len(names)} candidate paths; historical exceptions retained).")
    print(
        "This check is not numerical validation, a full artifact inventory, or migration approval."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
