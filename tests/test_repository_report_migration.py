"""Guard historical report relocation without importing a numerical runtime.

Hashes refer to the pre-migration checkout. Passing these checks preserves
provenance; it does not validate the scientific claims in the archived reports.
"""

from __future__ import annotations

import ast
import base64

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO = Path(__file__).resolve().parents[1]
SDAA_SWEEP_DRIVERS = REPO / "development/verification/sdaa-hardware/sweep-drivers"
LEDGER_PATH = REPO / "docs/architecture/migrations/report-assets-2026-10-26.json"
NOTE_START = "<!-- repository-layout-note:start -->"
NOTE_END = "<!-- repository-layout-note:end -->"
NOTE = re.compile(r"\n" + re.escape(NOTE_START) + r"\n.*?" + re.escape(NOTE_END) + r"\n", re.S)


SCRIPT_HANDOFFS = {
    item["old_path"]: item["new_path"]
    for item in json.loads(
        (LEDGER_PATH.parent / "legacy-plotting-maintenance-2026-11-24.json").read_text(encoding="utf-8")
    )["entries"]
}


def current_archive_path(path: str) -> Path:
    if path in SCRIPT_HANDOFFS:
        return REPO / SCRIPT_HANDOFFS[path]
    if path == "audit" or path.startswith("audit/"):
        return REPO / "development" / path
    return REPO / path


class RepositoryReportMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))

    def test_nebb_loose_and_sparse_audits_preserve_unique_originals(self):
        path = LEDGER_PATH.parent / "nebb-early-rounds-and-sparse-audits-2026-11-27.json"
        ledger = json.loads(path.read_text())
        with (path.parent.parent / "migration_manifest.csv").open(newline="") as stream:
            rows = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                    if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(rows, {r["old_path"]: r["new_path"] for r in ledger["entries"]})
        self.assertEqual(len(rows), 13)
        self.assertEqual(len(set(rows.values())), 12)
        self.assertEqual(sum(row["disposition"] == "verified_identical_duplicate_consolidation"
                             for row in ledger["entries"]), 4)
        for retired in ledger["retired_second_level_directories"] + ledger["retired_files"]:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", retired], cwd=REPO))
        for row in ledger["entries"]:
            destination = REPO / row["new_path"]
            self.assert_fingerprint(destination, row["current_sha256"], row["current_size_bytes"])
            mode, blob = subprocess.check_output(
                ["git", "ls-files", "-s", "--", row["new_path"]], cwd=REPO, text=True,
            ).split()[:2]
            self.assertEqual(mode, row["git_mode"])
            if row["byte_identical"]:
                self.assertEqual(blob, row["original_git_blob"])
                self.assertEqual(row["current_sha256"], row["original_sha256"])
            else:
                original = destination.read_bytes()
                for edit in reversed(row["path_only_replacements"]):
                    original = original.replace(edit["new"].encode(), edit["old"].encode())
                self.assertEqual(hashlib.sha256(original).hexdigest(), row["original_sha256"])
        for row in ledger["unchanged_prior_archive"]:
            self.assert_fingerprint(REPO / row["path"], row["sha256"], row["size_bytes"])
        fig = ledger["derived_image"]
        archived = json.loads((REPO / fig["source"]).read_text())
        self.assertEqual(base64.b64decode(archived["png_base64"]), (REPO / fig["target"]).read_bytes())
        self.assert_fingerprint(REPO / fig["target"], fig["sha256"], fig["size_bytes"])
        self.assertFalse(ledger["science"]["full_scientific_replay_performed"])
        for key, owner in ledger["case_ownership"].items():
            raw = subprocess.check_output(
                [sys.executable, "tools/benchmark_registry.py", "--case", owner],
                cwd=REPO, text=True,
            )
            (record,) = json.loads(raw)["cases"]
            self.assertEqual(record["current_acceptance"], "not_evaluated")
            (variant,) = record["variants"]
            registered = set(variant["source"] + variant["documentation"] +
                             variant["config"] + variant["results"] +
                             variant["reference_files"] + variant["evidence"])
            if key == "nebb":
                subset = {r["new_path"] for r in ledger["entries"] if r["old_path"].startswith("xval/")}
                self.assertTrue(subset <= registered)
                self.assertIn(fig["target"], registered)
                self.assertEqual(variant["results"], ["docs/board_audit/m17963/rerun/xval_nebb_bodyforce_driver.json"])
                root = "development/verification/nebb_steady_early_rounds"
            else:
                root = "development/audit/m18748" if key == "sparse" else "development/audit/m17677"
                self.assertTrue({r["new_path"] for r in ledger["entries"]
                                 if r["new_path"].startswith(root + "/")} <= registered)
                if key == "compressed":
                    manifest = json.loads((REPO / owner / "benchmark_case.json").read_text())
                    self.assertEqual({k: manifest[k] for k in ledger["prior_compressed_contract"]},
                                     ledger["prior_compressed_contract"])
            for rel in (f"{root}/README.md", f"{owner}/README.md"):
                doc = REPO / rel
                for link in re.findall(r"\]\(([^)]+)\)", doc.read_text()):
                    self.assertTrue((doc.parent / link).is_file(), link)

    def test_peskin_and_convolution_archive_preserves_original_evidence(self):
        path = LEDGER_PATH.parent / "peskin-and-conv-audit-2026-11-27.json"
        ledger = json.loads(path.read_text())
        with (path.parent.parent / "migration_manifest.csv").open(newline="") as stream:
            rows = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                    if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(rows, {r["old_path"]: r["new_path"] for r in ledger["entries"]})
        self.assertEqual(len(rows), 22)
        for retired in ledger["retired_second_level_directories"]:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", retired], cwd=REPO))
        for row in ledger["entries"]:
            destination = REPO / row["new_path"]
            self.assert_fingerprint(destination, row["current_sha256"], row["current_size_bytes"])
            mode, blob = subprocess.check_output(
                ["git", "ls-files", "-s", "--", row["new_path"]], cwd=REPO, text=True,
            ).split()[:2]
            self.assertEqual(mode, row["git_mode"])
            if row["byte_identical"]:
                self.assertEqual(blob, row["original_git_blob"])
                self.assertEqual(row["current_sha256"], row["original_sha256"])
            else:
                edit = row["path_only_edit"]
                original = destination.read_bytes().replace(edit["new"].encode(), edit["old"].encode())
                self.assertEqual(hashlib.sha256(original).hexdigest(), row["original_sha256"])
        self.assertEqual(sum(row["byte_identical"] for row in ledger["entries"]), 21)
        old_verdict = ledger["prior_verdict"]
        self.assert_fingerprint(REPO / old_verdict["path"], old_verdict["original_sha256"], old_verdict["size_bytes"])
        root = REPO / "development/verification/peskin_tensor_product_stencil_mdf"
        prov = json.loads((root / "oss_ref/PROVENANCE.json").read_text())
        for name, item in prov["files"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), item["sha256_vendored"])
        self.assertFalse(ledger["science"]["full_scientific_replay_performed"])
        for key, owner in ledger["case_ownership"].items():
            raw = subprocess.check_output(
                [sys.executable, "tools/benchmark_registry.py", "--case", owner],
                cwd=REPO, text=True,
            )
            (record,) = json.loads(raw)["cases"]
            self.assertEqual(record["current_acceptance"], "not_evaluated")
            (variant,) = record["variants"]
            registered = set(variant["source"] + variant["config"] + variant["documentation"] +
                             variant["reference_files"] + variant["results"] + variant["evidence"])
            prefix = "development/verification/peskin_tensor_product_stencil_mdf/" if key == "peskin" else "development/audit/m17876/"
            self.assertTrue({row["new_path"] for row in ledger["entries"] if row["new_path"].startswith(prefix)} <= registered)
            if key == "peskin":
                self.assertEqual(record["id"], "xval:peskin_tensor_product_stencil_mdf")
                self.assertEqual(record["cross_validation"]["criteria"], json.loads((root / "predeclared.json").read_text())["predeclared_thresholds"])
                self.assertEqual(len(variant["results"]), 2)
            else:
                self.assertEqual(record["id"], "campaign:xval_conv_as_streaming")
                self.assertEqual(variant["results"], [f"{owner}/artifacts_3d/result.json"])
            for rel in (f"{prefix}README.md", f"{owner}/README.md"):
                doc = REPO / rel
                for link in re.findall(r"\]\(([^)]+)\)", doc.read_text()):
                    self.assertTrue((doc.parent / link).is_file(), link)

    def test_four_audit_doc_archives_keep_original_objects_and_case_navigation(self):
        path = LEDGER_PATH.parent / "amr-survey-les-audit-docs-2026-11-27.json"
        ledger = json.loads(path.read_text())
        with (path.parent.parent / "migration_manifest.csv").open(newline="") as stream:
            rows = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                    if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(rows, {r["old_path"]: r["new_path"] for r in ledger["entries"]})
        self.assertEqual(len(rows), 12)
        self.assertEqual(len(ledger["retired_second_level_directories"]), 4)
        for retired in ledger["retired_second_level_directories"]:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", retired], cwd=REPO))
        for row in ledger["entries"] + ledger["unchanged_prior_archive"]:
            destination = row.get("new_path", row.get("path"))
            self.assert_fingerprint(REPO / destination, row["original_sha256"], row["size_bytes"])
            index = subprocess.check_output(
                ["git", "ls-files", "-s", "--", destination], cwd=REPO, text=True,
            ).split()
            self.assertEqual(index[:2], [row["git_mode"], row["original_git_blob"]])
        self.assertFalse(ledger["scientific_scope"]["full_scientific_replay_performed"])
        for mission, owner in ledger["case_ownership"].items():
            archive = f"development/audit/{mission}/README.md"
            raw = subprocess.check_output(
                [sys.executable, "tools/benchmark_registry.py", "--case", owner],
                cwd=REPO, text=True,
            )
            (record,) = json.loads(raw)["cases"]
            (variant,) = record["variants"]
            self.assertEqual(record["current_acceptance"], "not_evaluated")
            self.assertIn(archive, variant["evidence"])
            for relpath in (archive, f"{owner}/README.md"):
                document = REPO / relpath
                for link in re.findall(r"\]\(([^)]+)\)", document.read_text()):
                    self.assertTrue((document.parent / link).is_file(), link)
            if mission in {"m17631", "m18149"}:
                self.assertEqual(variant["results"], [f"development/audit/{mission}/audit.json"])
            if mission == "m17636":
                (result,) = variant["results"]
                archived = json.loads((REPO / result).read_text())
                self.assertFalse(archived["comparison"]["pass_criterion_15pct"])
            if mission == "m17888":
                self.assertEqual(record["id"], owner)
                self.assertIn("development/audit/m17888/AUDIT_m17888.md", variant["documentation"])

    def test_model_unit_matrix_supports_legacy_roots_and_both_dialects(self):
        command = [sys.executable, "-m", "tensorlbm.model_units"]
        env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
        canonical = subprocess.check_output([*command, "ci-matrix"], cwd=REPO, env=env)
        legacy = subprocess.check_output(
            [*command, "--root", "models/lbm", "--models-root", "models", "ci-matrix"],
            cwd=REPO,
            env=env,
        )
        self.assertEqual(canonical, legacy)
        cells = json.loads(canonical)["include"]
        self.assertEqual({row["kind"] for row in cells}, {"unit_toml", "unit_json"})
        self.assertTrue(all(row["dir"].startswith("benchmarks/model_units/") for row in cells))

    def test_retired_xval_design_directories_and_preserved_assets(self):
        path = LEDGER_PATH.parent / "xval-reference-designs-2026-11-23.json"
        ledger = json.loads(path.read_text(encoding="utf-8"))
        for directory in ledger["retired_paths"]:
            self.assertFalse((REPO / directory).exists(), directory)
        for item in ledger["entries"]:
            with self.subTest(path=item["new_path"]):
                self.assert_fingerprint(
                    REPO / item["new_path"], item["original_sha256"], item["size_bytes"]
                )

    def test_oss_inventory_archives_preserve_evidence_and_navigation(self):
        ledger_path = LEDGER_PATH.parent / "oss-inventory-studies-2026-11-26.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        with (ledger_path.parent.parent / "migration_manifest.csv").open(newline="") as source:
            recorded = [
                (row["old_path"], row["new_path"]) for row in csv.DictReader(source)
                if row["verification"] == ledger_path.relative_to(REPO).as_posix()
            ]
        self.assertCountEqual(recorded, [
            (item["old_path"], item["new_path"]) for item in ledger["entries"]
        ])
        for directory in ledger["retired_paths"]:
            self.assertFalse((REPO / directory).exists(), directory)
        for item in ledger["entries"]:
            if item["byte_identical"]:
                self.assert_fingerprint(
                    REPO / item["new_path"], item["original_sha256"], item["original_size_bytes"]
                )
            else:
                self.assertTrue((REPO / item["new_path"]).is_file())
        archive = REPO / "development/verification/oss_inventory/README.md"
        for report, section in (
            ("docs/board_audit/m18158/AUDIT_m18106.md", "bem"),
            ("docs/REPORT_XVAL_LBM_NO_DEM_REPO_FINAL.md", "dem"),
            ("benchmarks/verified/cavity_natural_convection/README.md", "bem"),
            ("benchmarks/pending/sphere_drag_closure/README.md", "dem"),
        ):
            report_path = REPO / report
            target = os.path.relpath(archive, report_path.parent)
            self.assertIn(f"]({target}#{section})", report_path.read_text(encoding="utf-8"))

    def assert_fingerprint(self, path: Path, sha256: str, size: int):
        raw = path.read_bytes()
        self.assertEqual(len(raw), size, str(path))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), sha256, str(path))

    def test_report_bodies_unchanged_except_declared_links_and_note(self):
        for item in self.ledger["reports"]:
            with self.subTest(report=item["new_path"]):
                text = (REPO / item["new_path"]).read_text(encoding="utf-8")
                self.assertEqual(text.count(NOTE_START), 1)
                self.assertEqual(text.count(NOTE_END), 1)
                text, count = NOTE.subn("", text)
                self.assertEqual(count, 1)
                for old, new in item["link_rewrites"].items():
                    self.assertEqual(text.count(f"]({new})"), 1, new)
                    text = text.replace(f"]({new})", f"]({old})")
                raw = text.encode("utf-8")
                fingerprint = item.get("current_fingerprint", {
                    "bytes": item["original_bytes"], "sha256": item["original_sha256"],
                })
                self.assertEqual(len(raw), fingerprint["bytes"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), fingerprint["sha256"])
                self.assertFalse((REPO / item["old_path"]).exists())

    def test_drop_entry_cmb5_archive_keeps_original_objects(self):
        ledger_path = LEDGER_PATH.parent / "cmb5-hydrorr-stability-2026-11-23.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        with (LEDGER_PATH.parent.parent / "migration_manifest.csv").open(newline="") as source:
            registered_paths = {
                row["old_path"]: row["new_path"] for row in csv.DictReader(source)
                if row["verification"] == ledger_path.relative_to(REPO).as_posix()
            }
        self.assertEqual(registered_paths, {
            entry["old_path"]: entry["new_path"] for entry in ledger["entries"]
        })
        for entry in ledger["entries"]:
            with self.subTest(path=entry["old_path"]):
                self.assertFalse((REPO / entry["old_path"]).exists())
                self.assert_fingerprint(
                    REPO / entry["new_path"], entry["original_sha256"], entry["size_bytes"]
                )

    def test_deduplicated_images_retain_original_bytes(self):
        for item in self.ledger["deduplicated_images"]:
            with self.subTest(image=item["old_path"]):
                self.assert_fingerprint(
                    REPO / item["canonical_path"], item["sha256"], item["bytes"]
                )
                self.assertFalse((REPO / item["old_path"]).exists())

    def test_mcmp_archives_preserve_original_data_and_derived_figures(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "mcmp-force-velocity-xval-2026-11-27.json").read_text()
        )
        self.assertEqual(len(ledger["entries"]), 26)
        self.assertEqual(len(ledger["restored_derived_figures"]), 9)
        for entry in ledger["entries"]:
            with self.subTest(path=entry["new_path"]):
                self.assertFalse((REPO / entry["old_path"]).exists())
                path = REPO / entry["new_path"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 entry["current_sha256"])
                self.assertEqual(
                    subprocess.check_output(
                        ["git", "ls-files", "-s", "--", entry["new_path"]],
                        cwd=REPO, text=True,
                    ).split()[0], entry["git_mode"],
                )
                if path.suffix not in {".py", ".md"}:
                    self.assertTrue(entry["byte_identical"])
                    self.assertEqual(entry["original_sha256"], entry["current_sha256"])
        for figure in ledger["restored_derived_figures"]:
            self.assertEqual(hashlib.sha256((REPO / figure["path"]).read_bytes()).hexdigest(),
                             figure["sha256"])
            for path, sha in figure["inputs"].items():
                self.assertEqual(hashlib.sha256((REPO / path).read_bytes()).hexdigest(), sha)
        self.assertEqual(ledger["validation"]["current_scientific_acceptance"],
                         "not_evaluated")

    def test_runtest_r6_tools_keep_paths_and_fingerprints(self):
        ledger_path = LEDGER_PATH.parent / "runtest-r6-audit-tools-2026-11-23.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        root = REPO / ledger["new_directory"]
        self.assertFalse((REPO / ledger["old_directory"]).exists())
        self.assertEqual(
            {path.name for path in root.iterdir() if path.is_file()},
            {Path(entry["old_path"]).name for entry in ledger["entries"]},
        )
        for entry in ledger["entries"]:
            with self.subTest(path=entry["new_path"]):
                path = REPO / entry["new_path"]
                self.assert_fingerprint(path, entry["new_sha256"], entry["new_size_bytes"])
        for name in ("docs/AUDIT_RUNTEST_6_TENSORLBM.md", "development/audit/m14496/README.md"):
            doc = REPO / name
            links = re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8"))
            self.assertTrue(any((doc.parent / link).resolve() == root for link in links))

    def test_archived_evidence_and_couette_images_are_unchanged(self):
        for item in self.ledger["preserved_files"]:
            with self.subTest(path=item["path"]):
                fingerprint = item.get("current_fingerprint", item)
                self.assert_fingerprint(
                    REPO / item["path"], fingerprint["sha256"], fingerprint["bytes"]
                )

    def test_report_and_case_inline_local_links_resolve(self):
        # Deliberately scoped: this is not a repository-wide Markdown parser,
        # remote URL check, anchor check or a check of external /nfs provenance.
        paths = [item["new_path"] for item in self.ledger["reports"]] + [
            "benchmarks/verified/couette_3d/README.md",
            "benchmarks/pending/capillary_invasion_washburn/README.md",
        ]
        for name in paths:
            path = REPO / name
            text = path.read_text(encoding="utf-8")
            for target in re.findall(r"\]\(([^)]+)\)", text):
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                with self.subTest(document=name, target=target):
                    resolved = (path.parent / unquote(parsed.path)).resolve()
                    self.assertTrue(resolved.is_relative_to(REPO))
                    self.assertTrue(resolved.exists(), f"Missing link: {name} -> {target}")

    def test_loose_xval_duplicate_and_cupy_audit_archive(self):
        ledger_path = LEDGER_PATH.parent / "loose-xval-and-cupy-audit-2026-11-27.json"
        ledger = json.loads(ledger_path.read_text())
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            rows = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                    if row["verification"] == ledger_path.relative_to(REPO).as_posix()}
        self.assertEqual(rows, {entry["old_path"]: entry["new_path"] for entry in ledger["entries"]})
        self.assertEqual(len(rows), 17)
        self.assertFalse(subprocess.check_output(
            ["git", "ls-files", "--", *ledger["retired_second_level_directories"], *ledger["retired_files"]],
            cwd=REPO, text=True,
        ))
        for entry in ledger["entries"]:
            self.assert_fingerprint(REPO / entry["new_path"], entry["original_sha256"], entry["size_bytes"])
            index = subprocess.check_output(
                ["git", "ls-files", "-s", "--", entry["new_path"]], cwd=REPO, text=True,
            ).split()
            self.assertEqual(index[:2], [entry["git_mode"], entry["original_git_blob"]])
        dup = ledger["compatibility"]["source_duplicate"]
        self.assertTrue(all(rows[path] == dup["new_path"] for path in dup["old_paths"]))
        snapshot = REPO / "development/audit/m17877/evidence_snapshot"
        for name, fingerprint in ledger["compatibility"]["frozen_snapshot_checksums"].items():
            self.assertEqual(hashlib.sha256((snapshot / name).read_bytes()).hexdigest(), fingerprint)
        self.assertFalse(ledger["science"]["full_replay_performed"])
        for case_id in ledger["compatibility"]["case_ids"]:
            raw = subprocess.check_output(
                [sys.executable, "tools/benchmark_registry.py", "--case", case_id], cwd=REPO, text=True,
            )
            (case,) = json.loads(raw)["cases"]
            self.assertEqual(case["current_acceptance"], "not_evaluated")
            (variant,) = case["variants"]
            if case_id == "xval:cupy_dropin_no_kernel_fusion":
                self.assertIn(ledger["compatibility"]["prior_artifact_handoff"]["current_destination"],
                              variant["results"])
                self.assertIn("development/verification/cupy_dropin_no_kernel_fusion/artifacts/run_main.json",
                              variant["results"])
            else:
                self.assertIn("development/verification/lbmconfig_poiseuille/m17952_legacy.py",
                              variant["evidence"])
        for name in ("development/audit/m17877/README.md",
                     "benchmarks/cross_validation/cupy_dropin_no_kernel_fusion/README.md",
                     "development/verification/lbmconfig_poiseuille/README.md"):
            path = REPO / name
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                self.assertTrue((path.parent / target).is_file(), target)

    def test_cupy_archived_figures_match_original_payload_and_public_links(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "halo-cupy-archives-2026-11-26.json").read_text(encoding="utf-8")
        )
        report = REPO / "docs/XVAL_LBM_CUPY_DROPIN_NO_KERNEL_FUSION.md"
        self.assertEqual(len(ledger["recovered_figures"]), 5)
        for row in ledger["recovered_figures"]:
            with self.subTest(figure=row["figure_key"]):
                payload_path = REPO / row["source_payload"]
                self.assertEqual(hashlib.sha256(payload_path.read_bytes()).hexdigest(),
                                 row["source_payload_sha256"])
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
                (item,) = [obj for obj in payload["figures"] if obj["name"] == row["figure_key"]]
                decoded = base64.b64decode(item["png_base64"], validate=True)
                figure = REPO / row["path"]
                self.assertEqual(figure.read_bytes(), decoded)
                self.assertEqual(hashlib.sha256(decoded).hexdigest(), row["sha256"])
                self.assertIn(f"../{row['path']}", report.read_text(encoding="utf-8"))

    def test_retired_figure_root_preserves_historical_assets(self):
        ledger_path = LEDGER_PATH.parent / "figures-root-retirement-2026-11-21.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        tracked = subprocess.check_output(
            ["git", "ls-files", "figures"], cwd=REPO, text=True
        )
        self.assertEqual(tracked, "")
        for item in ledger["entries"]:
            with self.subTest(path=item["new_path"]):
                self.assert_fingerprint(
                    REPO / item["new_path"], item["sha256"], item["size_bytes"]
                )

    def test_retired_evidence_root_preserves_its_recorded_assets(self):
        ledger_path = LEDGER_PATH.parent / "evidence-root-retirement-2026-11-21.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        self.assertFalse(subprocess.check_output(
            ["git", "ls-files", "evidence"], cwd=REPO, text=True
        ))
        for entry in ledger["entries"]:
            with self.subTest(path=entry["new_path"]):
                self.assert_fingerprint(
                    REPO / entry["new_path"],
                    entry["destination_sha256"],
                    entry["destination_size_bytes"],
                )

    def test_bulk_root_retirement_accounts_for_all_moved_assets(self):
        ledger_path = LEDGER_PATH.parent / "bulk-root-retirement-2026-11-21.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        followup_path = LEDGER_PATH.parent / "scripts-data-retirement-2026-11-23.json"
        followup = json.loads(followup_path.read_text(encoding="utf-8"))
        handoffs = {item["old_path"]: item for item in followup["entries"]}
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for path, batch in ((ledger_path, ledger), (followup_path, followup)):
            moves = {
                row["old_path"]: row["new_path"]
                for row in rows
                if row["verification"] == path.relative_to(REPO).as_posix()
            }
            self.assertEqual(moves, {item["old_path"]: item["new_path"] for item in batch["entries"]})
        policy = json.loads((REPO / "docs/architecture/repository_layout_policy.json").read_text())
        self.assertTrue(set(ledger["retired_roots"]).isdisjoint(policy["root_directories"]))
        self.assertFalse(subprocess.check_output(["git", "ls-files", "scripts/data"], cwd=REPO))
        self.assertEqual(set(handoffs), {
            item["new_path"] for item in ledger["entries"] if item["new_path"].startswith("scripts/data/")
        })
        for item in ledger["entries"]:
            destination = REPO / item["new_path"]
            if item["new_path"] in handoffs:
                next_move = handoffs[item["new_path"]]
                self.assertEqual(next_move["original_git_blob"], item["git_blob"])
                self.assertEqual(next_move["original_sha256"], item["sha256"])
                self.assertEqual(next_move["size_bytes"], item["size_bytes"])
                self.assertEqual(next_move["git_mode"], item["git_mode"])
                destination = REPO / next_move["new_path"]
            with self.subTest(path=destination):
                self.assertTrue(destination.is_file())
                self.assertFalse((REPO / item["old_path"]).exists())
                if destination.suffix == ".json":
                    self.assert_fingerprint(destination, item["sha256"], item["size_bytes"])

    def test_docs_index_and_benchmark_backlinks_point_to_archived_reports(self):
        index = (REPO / "docs/README.md").read_text(encoding="utf-8")
        cases = ["verified/couette_3d", "pending/capillary_invasion_washburn"]
        for item, case in zip(self.ledger["reports"], cases, strict=True):
            name = Path(item["new_path"]).name
            with self.subTest(report=name):
                self.assertIn(f"](reports/{name})", index)
                case_readme = (REPO / "benchmarks" / case / "README.md").read_text(encoding="utf-8")
                self.assertIn(f"](../../../docs/reports/{name})", case_readme)

    def test_m26816_archive_preserves_root_assets(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "m26816-root-retirement-2026-11-21.json").read_text()
        )
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)}
        for item in ledger["entries"]:
            destination = REPO / item["new_path"]
            with self.subTest(path=destination):
                self.assert_fingerprint(destination, item["sha256"], item["size_bytes"])
                self.assertEqual(manifest[item["old_path"]], item["new_path"])
                self.assertFalse((REPO / item["old_path"]).exists())

    def test_retired_case_wrappers_preserve_original_bytes_and_canonical_entries(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "case-root-retirement-2026-11-21.json").read_text()
        )
        tracked = subprocess.check_output(
            ["git", "ls-files", "--", ledger["root_removed"]], cwd=REPO, text=True,
        )
        self.assertEqual(tracked, "")
        for item in ledger["entries"]:
            if "current_entrypoint" not in item:
                continue
            canonical = REPO / item["current_entrypoint"]
            with self.subTest(runner=canonical):
                archived = REPO / item["new_path"]
                raw = archived.read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item["original_sha256"])
                wrapper = ast.parse(raw)
                target = canonical.relative_to(REPO).with_suffix("")
                self.assertIn(
                    ".".join(target.parts),
                    [node.module for node in wrapper.body if isinstance(node, ast.ImportFrom)],
                )
                index = canonical.with_name("README.md")
                text = index.read_text(encoding="utf-8")
                self.assertIn(canonical.relative_to(REPO).as_posix(), text)
                for link in re.findall(r"\]\(([^)]+)\)", text):
                    self.assertTrue((index.parent / link).is_file(), link)

    def test_development_archive_pilot_preserves_linked_evidence(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "development-archive-first-batch.json").read_text()
        )
        campaign = REPO / "xval/XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER/README.md"
        text = campaign.read_text(encoding="utf-8")
        targets = re.findall(r"\]\(([^)]+)\)", text)
        resolved = {(campaign.parent / target).resolve() for target in targets}
        for target in resolved:
            self.assertTrue(target.is_file(), str(target))
        for item in ledger["entries"]:
            destination = REPO / item["new_path"]
            with self.subTest(path=destination):
                self.assert_fingerprint(destination, item["sha256"], item["size_bytes"])
                self.assertIn(destination, resolved)
                self.assertFalse((REPO / item["old_path"]).exists())

    def test_ibm_epsilon_gram_relocation_preserves_tracked_inputs_and_provenance(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "ibm-epsilon-gram-xval-2026-11-27.json").read_text()
        )
        self.assertEqual(len(ledger["entries"]), 12)
        self.assertEqual(len(ledger["restored_derived_figures"]), 4)
        archive = REPO / "development/verification/ibm_epsilon_gram"
        manifest = json.loads((archive / "manifest.json").read_text())
        for entry in ledger["entries"]:
            with self.subTest(path=entry["new_path"]):
                self.assertFalse((REPO / entry["old_path"]).exists())
                p = REPO / entry["new_path"]
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),
                                 entry["current_sha256"])
                self.assertEqual(
                    subprocess.check_output(
                        ["git", "ls-files", "-s", "--", entry["new_path"]],
                        cwd=REPO, text=True,
                    ).split()[0], entry["git_mode"],
                )
                if p.suffix == ".json":
                    self.assertTrue(entry["byte_identical"])
                    self.assertEqual(entry["original_sha256"], entry["current_sha256"])
        for figure in ledger["restored_derived_figures"]:
            p = REPO / figure["path"]
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), figure["sha256"])
            self.assertEqual(p.parent.name, "derived_figures")
            self.assertFalse(figure["matches_historical_manifest"])
            self.assertEqual(manifest["files"][p.name]["sha256_16"],
                             figure["historical_manifest_sha256_16"])
            for path, sha in figure["inputs"].items():
                self.assertEqual(hashlib.sha256((REPO / path).read_bytes()).hexdigest(), sha)
        self.assertEqual(ledger["validation"]["current_scientific_acceptance"],
                         "not_evaluated")

    def test_sod_figures_and_m26765_publisher_preserve_assets(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "sod-figures-and-m26765-2026-11-21.json").read_text()
        )
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)}
        report_path = REPO / "docs/report_m40.md"
        report = report_path.read_text(encoding="utf-8")
        for item in ledger["entries"]:
            destination = current_archive_path(item["new_path"])
            with self.subTest(path=item["new_path"]):
                self.assert_fingerprint(destination, item["sha256"], item["size_bytes"])
                self.assertEqual(manifest[item["old_path"]], item["new_path"])
                self.assertFalse((REPO / item["old_path"]).exists())
                if destination.suffix == ".png":
                    target = destination.relative_to(report_path.parent).as_posix()
                    self.assertIn(f"]({target})", report)
        index = REPO / "development/audit/m26765/README.md"
        for target in re.findall(r"\]\(([^)]+)\)", index.read_text(encoding="utf-8")):
            with self.subTest(link=target):
                self.assertTrue((index.parent / target).is_file())

    def test_audit_records_retain_bytes_and_sibling_evidence(self):
        ledger = json.loads((LEDGER_PATH.parent / "audit-records-2026-11-21.json").read_text())
        for move in ledger["entries"]:
            with self.subTest(path=move["new_path"]):
                self.assert_fingerprint(current_archive_path(move["new_path"]), move["sha256"], move["size_bytes"])
        report = REPO / "development/audit/m18313/audit_m18313.md"
        self.assertIn("numbers_from_artifact.json", report.read_text())
        self.assertTrue(report.with_name("numbers_from_artifact.json").is_file())

    def test_xval_artifacts_preserved_without_tracked_generated_root(self):
        ledger_path = LEDGER_PATH.parent / "xval-artifacts-retirement-2026-11-23.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        tracked = subprocess.check_output(
            ["git", "ls-files", "--", ledger["retired_path"]], cwd=REPO, text=True,
        )
        self.assertEqual(tracked, "")
        followup = json.loads(
            (LEDGER_PATH.parent / "loose-xval-and-cupy-audit-2026-11-27.json").read_text()
        )
        handoff = followup["compatibility"]["prior_artifact_handoff"]
        for item in ledger["entries"]:
            with self.subTest(path=item["old_path"]):
                destination = item["new_path"]
                if destination == handoff["prior_destination"]:
                    self.assertEqual(item["original_sha256"], handoff["sha256"])
                    destination = handoff["current_destination"]
                self.assert_fingerprint(REPO / destination, item["original_sha256"], item["size_bytes"])

    def test_ci_and_contract_audit_scripts_retain_migration_provenance(self):
        ledger_path = LEDGER_PATH.parent / "ci-contract-evidence-2026-11-23.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        tracked = subprocess.check_output(
            ["git", "ls-files", "--", *ledger["retired_paths"]], cwd=REPO, text=True,
        )
        self.assertEqual(tracked, "")
        for entry in ledger["entries"]:
            with self.subTest(path=entry["new_path"]):
                self.assert_fingerprint(REPO / entry["new_path"], entry["new_sha256"],
                                        entry["new_size_bytes"])

    def test_zou_he_dam_slot_archive_preserves_historical_data_and_derived_figures(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "zou-he-dam-break-slot-xval-2026-11-27.json").read_text()
        )
        self.assertEqual(len(ledger["entries"]), 62)
        for item in ledger["entries"]:
            with self.subTest(path=item["new_path"]):
                raw = (REPO / item["new_path"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item["new_sha256"])
                self.assertFalse((REPO / item["old_path"]).exists())
                index = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", item["new_path"]],
                    cwd=REPO,
                    text=True,
                ).split()
                self.assertEqual(index[:2], [item["git_mode"], item["new_git_blob"]])
                if item["new_path"].endswith((".json", ".csv")):
                    self.assertEqual(item["original_sha256"], item["new_sha256"])
        self.assertEqual(len(ledger["regenerated_figures"]), 8)
        for item in ledger["regenerated_figures"]:
            raw = (REPO / item["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), item["sha256"])
            for src in item["source_data"]:
                self.assertEqual(
                    hashlib.sha256((REPO / src["path"]).read_bytes()).hexdigest(),
                    src["sha256"],
                )

    def test_november_root_migration_destinations(self):
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            moves = {
                row["old_path"]: row["new_path"]
                for row in csv.DictReader(stream)
                if "nov-2026 root migration" in row["verification"]
            }
        self.assertIn("bench_t4_medium.py", moves)
        self.assertIn("audit_m18221", moves)
        self.assertIn("verdict_m15215.sql", moves)
        path = LEDGER_PATH.parent / "t4-physics-ai-drivers-2026-11-24.json"
        followup = json.loads(path.read_text(encoding="utf-8"))
        handoffs = {item["old_path"]: item for item in followup["entries"]}
        self.assertEqual(set(handoffs), {new for new in moves.values() if new.startswith("scripts/physics_ai/")})
        self.assertFalse(subprocess.check_output(["git", "ls-files", "scripts/physics_ai"], cwd=REPO))
        for item in handoffs.values():
            self.assert_fingerprint(REPO / item["new_path"], item["migrated_sha256"], item["migrated_size_bytes"])
        for old, new in moves.items():
            if new in handoffs:
                new = handoffs[new]["new_path"]
            with self.subTest(old=old, new=new):
                # Local runners may recreate ignored scratch directories.
                # Their tracked descendants, rather than filesystem presence,
                # are governed by the repository layout checker.
                self.assertTrue((current_archive_path(new)).exists())
                if (current_archive_path(new)).is_file():
                    self.assertFalse((REPO / old).is_file())

    def test_wave_one_archive_preserves_source_files_and_both_report_versions(self):
        ledger = json.loads((REPO / "docs/architecture/migrations/m26711-2026-11-21.json").read_text())
        self.assertEqual(len(ledger["entries"]), 23)
        for item in ledger["entries"]:
            with self.subTest(path=item["new_path"]):
                blob = (current_archive_path(item["new_path"])).read_bytes()
                self.assertEqual(hashlib.sha256(blob).hexdigest(), item["sha256"])
                self.assertEqual(len(blob), item["size_bytes"])
                self.assertEqual(
                    subprocess.check_output(
                        ["git", "hash-object", str(current_archive_path(item["new_path"]))],
                        cwd=REPO,
                        text=True,
                    ).strip(),
                    item["git_blob"],
                )
                index_entry = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", str(current_archive_path(item["new_path"]))],
                    cwd=REPO,
                    text=True,
                ).split()
                self.assertEqual(index_entry[0], item["git_mode"])
                self.assertFalse((REPO / item["old_path"]).exists())
        for mission in ledger["report_missions"]:
            report = json.loads((REPO / f"development/audit/m{mission}/report.json").read_text())
            self.assertEqual(int(report["mission_id"]), mission)

    def test_trt_beam_archive_preserves_all_original_files(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "trt-beam-xval-2026-11-26.json").read_text()
        )
        self.assertEqual(len(ledger["entries"]), 96)
        for item in ledger["entries"]:
            with self.subTest(path=item["new_path"]):
                raw = (REPO / item["new_path"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item["new_sha256"])
                self.assertFalse((REPO / item["old_path"]).exists())
                index = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", item["new_path"]],
                    cwd=REPO,
                    text=True,
                ).split()
                self.assertEqual(index[:2], [item["git_mode"], item["new_git_blob"]])
                if item["old_path"].endswith((".dat", ".npz", ".json", ".csv")):
                    self.assertEqual(item["original_sha256"], item["new_sha256"])

    def test_reused_tmp_report_preserves_both_mission_versions(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "m26843-root-report-2026-11-21.json").read_text()
        )
        entry, = ledger["entries"]
        self.assert_fingerprint(
            REPO / entry["new_path"], entry["sha256"], entry["size_bytes"]
        )
        self.assertTrue((REPO / "development/audit/m26816/report.json").is_file())
        self.assertFalse((REPO / entry["old_path"]).exists())

    def test_migrations_registered_without_reintroducing_root_exceptions(self):
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        paths = [row["old_path"] for row in rows]
        self.assertEqual(len(paths), len(set(paths)))
        manifest = {row["old_path"]: row for row in rows}
        policy = json.loads((REPO / "docs/architecture/repository_layout_policy.json").read_text())
        roots = set(policy["root_files"]) | set(policy["legacy_root_files"])
        for item in self.ledger["reports"] + self.ledger["deduplicated_images"]:
            old = item["old_path"]
            with self.subTest(old_path=old):
                self.assertNotIn(old, roots)
                expected = item.get("new_path", item.get("canonical_path"))
                self.assertEqual(manifest[old]["new_path"], expected)
                self.assertIn(
                    manifest[old]["status"],
                    {"migrated_pending_review", "deduplicated_pending_review", "accepted"},
                )

    def test_legacy_audits_are_preserved_under_missions(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "legacy-audit-root-2026-11-21.json").read_text()
        )
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)}
        for item in ledger["entries"]:
            destination = item["new_path"]
            with self.subTest(path=destination):
                self.assert_fingerprint(current_archive_path(destination), item["sha256"], item["size_bytes"])
                self.assertEqual(manifest[item["old_path"]], destination)
                self.assertFalse((REPO / item["old_path"]).exists())
        for relative in ("development/audit/m17444/README.md", "development/audit/m18155/README.md"):
            index = REPO / relative
            for target in re.findall(r"\]\(([^)]+)\)", index.read_text(encoding="utf-8")):
                with self.subTest(index=relative, link=target):
                    self.assertTrue((index.parent / target).is_file())

    def test_water_entry_report_layout_preserves_assets_and_index_links(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "water-entry-report-layout-2026-11-21.json").read_text()
        )
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)}
        for item in ledger["entries"]:
            with self.subTest(path=item["new_path"]):
                destination = item["new_path"]
                self.assert_fingerprint(current_archive_path(destination), item["sha256"], item["size_bytes"])
                self.assertEqual(manifest[item["old_path"]], destination)
                self.assertFalse((REPO / item["old_path"]).exists())
        index = REPO / "docs/reports/water_entry/README.md"
        for target in re.findall(r"\]\(([^)]+)\)", index.read_text(encoding="utf-8")):
            with self.subTest(link=target):
                self.assertTrue((index.parent / target).exists())


    def test_retired_audit_root_preserves_original_blobs_and_modes(self):
        ledger_path = LEDGER_PATH.parent / "audit-root-retirement-2026-11-21.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        archive = ledger["archive"]
        self.assertEqual(len(ledger["files"]), archive["tracked_files"])
        self.assertFalse(subprocess.check_output(["git", "ls-files", "--", "audit"], cwd=REPO))
        records = subprocess.check_output(
            ["git", "ls-files", "-s", "-z", "--", archive["new_path"]], cwd=REPO
        )
        indexed = {}
        for entry in filter(None, records.split(b"\0")):
            metadata, path = entry.decode().split("\t", 1)
            mode, oid, stage = metadata.split()
            self.assertEqual(stage, "0")
            indexed[path] = (mode, oid)
        for relative, (mode, original_oid) in ledger["files"].items():
            destination = f"{archive['new_path']}/{relative}"
            with self.subTest(path=destination):
                expected = ledger["path_edits"].get(relative, original_oid)
                self.assertEqual(indexed[destination], (mode, expected))
                raw = (REPO / destination).read_bytes()
                blob = f"blob {len(raw)}\0".encode() + raw
                self.assertEqual(hashlib.sha1(blob).hexdigest(), expected)

    def test_audit_script_directory_preserves_mission_tools_and_provenance(self):
        ledger_path = LEDGER_PATH.parent / "audit-script-directory-retirement-2026-11-24.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            moves = {
                row["old_path"]: row["new_path"]
                for row in csv.DictReader(stream)
                if row["verification"] == ledger_path.relative_to(REPO).as_posix()
            }
        self.assertEqual(moves, {item["old_path"]: item["new_path"] for item in ledger["entries"]})
        self.assertEqual(len(moves), 17)
        self.assertFalse(subprocess.check_output(["git", "ls-files", "scripts/audit"], cwd=REPO))
        for item in ledger["entries"]:
            with self.subTest(path=item["new_path"]):
                self.assert_fingerprint(
                    REPO / item["new_path"], item["migrated_sha256"], item["migrated_size_bytes"]
                )

    def test_concurrent_drop_entry_reintroductions_are_archived_and_pinned(self):
        survey_ledger = LEDGER_PATH.parent / "m26963-bb-root-retirement-2026-11-22.json"
        ledgers = [
            LEDGER_PATH.parent / "m26734-drop-entry-reentry-2026-11-21.json",
            LEDGER_PATH.parent / "m26737-drop-entry-reentry-2026-11-22.json",
            LEDGER_PATH.parent / "m27075-drop-entry-reentry-2026-11-22.json",
            LEDGER_PATH.parent / "m26856-audit-reentry-2026-11-21.json",
            LEDGER_PATH.parent / "m26892-drop-entry-reentry-2026-11-22.json",
            LEDGER_PATH.parent / "m26887-drop-entry-reentry-2026-11-22.json",
            LEDGER_PATH.parent / "m26894-drop-entry-reentry-2026-11-22.json",
            survey_ledger,
            LEDGER_PATH.parent / "m26954-bb-root-retirement-2026-11-22.json",
        ]
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            manifest = {row["old_path"]: row for row in csv.DictReader(stream)}
        self.assertFalse(subprocess.check_output(
            ["git", "ls-files", "--", "audit", "BB", "drop_entry", "send_report.py"],
            cwd=REPO,
            text=True,
        ))
        for ledger_path in ledgers:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            for item in ledger["entries"]:
                with self.subTest(path=item["new_path"]):
                    raw = (REPO / item["new_path"]).read_bytes()
                    expected_sha = item.get("new_sha256", item["original_sha256"])
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), expected_sha)
                    index = subprocess.check_output(
                        ["git", "ls-files", "-s", "--", item["new_path"]],
                        cwd=REPO,
                        text=True,
                    ).split()
                    self.assertEqual(index[0], item["git_mode"])
                    if "new_sha256" not in item:
                        self.assertEqual(index[1], item["original_git_blob"])
                    if item.get("manifest_registration", True):
                        self.assertEqual(
                            manifest[item["old_path"]]["new_path"], item["new_path"]
                        )
                    self.assertFalse((REPO / item["old_path"]).exists())
                    if "published_duplicate" in item:
                        self.assertEqual(
                            (REPO / item["published_duplicate"]).read_bytes(), raw
                        )
        m26734 = json.loads(ledgers[0].read_text(encoding="utf-8"))
        published = m26734["entries"][0]["published_duplicate"]
        self.assertEqual(
            (REPO / published).read_bytes(),
            (REPO / m26734["entries"][0]["new_path"]).read_bytes(),
        )
        m26892 = json.loads(ledgers[4].read_text(encoding="utf-8"))
        published = m26892["entries"][0]["published_duplicate"]
        self.assertEqual(
            (REPO / published).read_bytes(),
            (REPO / m26892["entries"][0]["new_path"]).read_bytes(),
        )
        predecessor = m26892["entries"][2]["duplicate_of"]
        old_path, old_revision = predecessor.split("@", 1)
        snapshot = m26892["entries"][2]
        self.assertEqual(snapshot["duplicate_git_blob"], snapshot["original_git_blob"])
        if subprocess.run(
            ["git", "cat-file", "-e", f"{old_revision}^{{commit}}"],
            cwd=REPO,
            capture_output=True,
        ).returncode == 0:
            self.assertEqual(
                subprocess.check_output(["git", "show", f"{old_revision}:{old_path}"], cwd=REPO),
                (REPO / snapshot["new_path"]).read_bytes(),
            )
        publisher = ast.parse(
            (REPO / "development/audit/drop_entry/BB/m26856/send_report.py").read_text()
        )
        self.assertTrue(any(
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
            and node.value.attr == "environ"
            for node in ast.walk(publisher)
        ))
        m26894_generator = (
            REPO / "benchmarks/water_entry/single_drop/tens1_csf_r2_card.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('os.path.join(ROOT, "drop_entry"', m26894_generator)
        self.assertIn('os.path.join(ROOT, "docs", "drop_entry_bb"', m26894_generator)
        survey = json.loads(survey_ledger.read_text(encoding="utf-8"))["entries"][0]
        for stage in ("probe", "report"):
            script = (REPO / f"scripts/de_x_surv1_mmreview_{stage}.py").read_text(
                encoding="utf-8"
            )
            self.assertIn(survey["published_duplicate"], script)
            self.assertNotIn(survey["old_path"], script)

    def test_sparse_raster_archive_preserves_tracked_evidence(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "sparse-raster-xval-2026-11-26.json").read_text()
        )
        for item in ledger["entries"]:
            with self.subTest(path=item["new_path"]):
                raw = (REPO / item["new_path"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item["new_sha256"])
                self.assertFalse((REPO / item["old_path"]).exists())
                index = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", item["new_path"]],
                    cwd=REPO,
                    text=True,
                ).split()
                self.assertEqual(index[:2], [item["git_mode"], item["new_git_blob"]])
        for item in ledger["recovered_figures"] + ledger["regenerated_figures"]:
            with self.subTest(figure=item["path"]):
                raw = (REPO / item["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item["sha256"])
                self.assertEqual(len(raw), item["size_bytes"])

    def test_papers_root_is_fully_mapped_to_benchmark_and_docs_owners(self):
        ledger_path = LEDGER_PATH.parent / "papers-root-retirement-2026-11-22.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        self.assertEqual(ledger["root_directories_before"], 14)
        self.assertEqual(ledger["root_directories_after"], 13)
        self.assertEqual(len(ledger["entries"]), 35)
        self.assertEqual(sum("new_sha256" in item for item in ledger["entries"]), 15)
        self.assertFalse(subprocess.check_output(
            ["git", "ls-files", "--", "papers"], cwd=REPO, text=True
        ))
        for item in ledger["entries"]:
            with self.subTest(path=item["new_path"]):
                raw = (REPO / item["new_path"]).read_bytes()
                expected_sha = item.get("new_sha256", item["original_sha256"])
                expected_blob = item.get("new_git_blob", item["original_git_blob"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected_sha)
                index = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", item["new_path"]],
                    cwd=REPO,
                    text=True,
                ).split()
                self.assertEqual(index[:2], [item["git_mode"], expected_blob])
                self.assertFalse((REPO / item["old_path"]).exists())

    def test_young_laplace_two_pdf_archive_preserves_original_evidence(self):
        import base64

        ledger = json.loads(
            (LEDGER_PATH.parent / "young-laplace-two-pdf-xval-2026-11-26.json").read_text()
        )
        self.assertEqual(len(ledger["entries"]), 16)
        for item in ledger["entries"]:
            with self.subTest(path=item["new_path"]):
                raw = (REPO / item["new_path"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item["new_sha256"])
                self.assertFalse((REPO / item["old_path"]).exists())
                index = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", item["new_path"]],
                    cwd=REPO,
                    text=True,
                ).split()
                self.assertEqual(index[:2], [item["git_mode"], item["new_git_blob"]])
        for item in ledger["recovered_figures"]:
            with self.subTest(image=item["path"]):
                payload = (REPO / item["source_payload"]).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), item["source_payload_sha256"])
                name = Path(item["path"]).name
                (figure,) = [f for f in json.loads(payload)["figures"] if f["name"] == name]
                raw = (REPO / item["path"]).read_bytes()
                self.assertEqual(raw, base64.b64decode(figure["png_base64"], validate=True))
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item["sha256"])

    def test_aircraft_icing_sdaa_evidence_belongs_to_the_benchmark(self):
        ledger_path = (
            LEDGER_PATH.parent / "aircraft-icing-sdaa-artifacts-2026-11-22.json"
        )
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        self.assertEqual(ledger["root_directories_before"], 13)
        self.assertEqual(ledger["root_directories_after"], 13)
        self.assertEqual(len(ledger["entries"]), 25)
        with (REPO / "docs/architecture/migration_manifest.csv").open(
            encoding="utf-8"
        ) as stream:
            manifest = {
                row["old_path"]: row["new_path"]
                for row in csv.DictReader(stream)
                if row["verification"] == ledger_path.relative_to(REPO).as_posix()
            }
        self.assertEqual(
            manifest,
            {item["old_path"]: item["new_path"] for item in ledger["entries"]},
        )
        for item in ledger["entries"]:
            with self.subTest(path=item["new_path"]):
                raw = (REPO / item["new_path"]).read_bytes()
                self.assertEqual(len(raw), item["size_bytes"])
                self.assertEqual(
                    hashlib.sha256(raw).hexdigest(), item["original_sha256"]
                )
                index = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", item["new_path"]],
                    cwd=REPO,
                    text=True,
                ).split()
                self.assertEqual(
                    index[:2], [item["git_mode"], item["original_git_blob"]]
                )
                self.assertFalse((REPO / item["old_path"]).exists())

        summary = json.loads(
            (
                REPO
                / "benchmarks/verified/aircraft_icing_naca0012_rime/evidence/sdaa_sweep/summary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(summary["gates_summary"]["n_pass"], 6)
        self.assertEqual(summary["gates_summary"]["n_total"], 11)
        self.assertFalse(summary["gates_summary"]["all_pass"])
        self.assertEqual(summary["verdict"]["overall"], "FAIL")
        run_root = (
            REPO
            / "benchmarks/verified/aircraft_icing_naca0012_rime/evidence/sdaa_sweep/runs"
        )
        self.assertFalse(list(run_root.rglob("result.npz")))

        runner = (
            SDAA_SWEEP_DRIVERS / "sdaa_sweep_aircraft_icing_naca0012_rime.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('REPO / "artifacts" / "sdaa_runs"', runner)
        self.assertNotIn("sdaa_aircraft_icing_naca0012_rime.json", runner)
        self.assertIn('evidence_dir / "summary.json"', runner)
        self.assertIn('evidence_dir / "runs"', runner)

    def test_dtmb5415_sdaa_evidence_belongs_to_the_pending_benchmark(self):
        ledger_path = LEDGER_PATH.parent / "dtmb5415-sdaa-artifacts-2026-11-22.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        self.assertEqual(ledger["root_directories_before"], 13)
        self.assertEqual(ledger["root_directories_after"], 13)
        self.assertEqual(len(ledger["entries"]), 15)
        with (REPO / "docs/architecture/migration_manifest.csv").open(
            encoding="utf-8"
        ) as stream:
            manifest = {
                row["old_path"]: row["new_path"]
                for row in csv.DictReader(stream)
                if row["verification"] == ledger_path.relative_to(REPO).as_posix()
            }
        self.assertEqual(
            manifest,
            {item["old_path"]: item["new_path"] for item in ledger["entries"]},
        )
        for item in ledger["entries"]:
            with self.subTest(path=item["new_path"]):
                raw = (REPO / item["new_path"]).read_bytes()
                self.assertEqual(len(raw), item["size_bytes"])
                self.assertEqual(
                    hashlib.sha256(raw).hexdigest(), item["original_sha256"]
                )
                index = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", item["new_path"]],
                    cwd=REPO,
                    text=True,
                ).split()
                self.assertEqual(
                    index[:2], [item["git_mode"], item["original_git_blob"]]
                )
                self.assertFalse((REPO / item["old_path"]).exists())

        case = REPO / "benchmarks/pending/dtmb5415_resistance"
        diagnosis = json.loads(
            (case / "evidence/sdaa_m452/diagnosis.json").read_text(encoding="utf-8")
        )
        self.assertTrue(diagnosis["verdict"]["conclusion"].startswith("HONEST-FAIL"))
        self.assertFalse(
            diagnosis["verdict"]["in_frame_accuracy"]["within_3pct_band"]
        )
        self.assertEqual(
            diagnosis["verdict"]["in_frame_accuracy"]["case_verdict"],
            "WITHHELD_UNSTEADY",
        )
        self.assertIn("pending / honest-FAIL", (case / "README.md").read_text())

        builder = (case / "make_diagnosis.py").read_text(encoding="utf-8")
        probe = (case / "_geometry_probe.py").read_text(encoding="utf-8")
        self.assertNotIn('_REPO_ROOT / "artifacts"', builder)
        self.assertIn('_EVIDENCE_DIR / "records"', builder)
        self.assertIn('_EVIDENCE_DIR / "diagnosis.json"', builder)
        self.assertIn('_EVIDENCE_DIR / "figures"', builder)
        self.assertNotIn('_HERE.parents[2] / "runs"', probe)
        self.assertIn('_HERE / "evidence" / "sdaa_m452"', probe)

    def test_mcmp_knowledge_loop_archive_retains_failure_and_draft_ownership(self):
        ledger_path = LEDGER_PATH.parent / "mcmp-knowledge-loop-archive-2026-11-27.json"
        ledger = json.loads(ledger_path.read_text())
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            paths = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                     if row["verification"] == ledger_path.relative_to(REPO).as_posix()}
        self.assertEqual(len(paths), 10)
        self.assertEqual(paths, {entry["old_path"]: entry["new_path"] for entry in ledger["entries"]})
        for entry in ledger["entries"]:
            self.assert_fingerprint(REPO / entry["new_path"], entry["original_sha256"],
                                    entry["size_bytes"])
            index = subprocess.check_output(
                ["git", "ls-files", "-s", "--", entry["new_path"]], cwd=REPO, text=True,
            ).split()
            self.assertEqual(index[:2], [entry["git_mode"], entry["original_git_blob"]])
            self.assertFalse((REPO / entry["old_path"]).exists())
        self.assertFalse((REPO / "docs/tk-e2e").exists())
        compatibility = ledger["compatibility"]
        original = (REPO / compatibility["frozen_evidence"]).read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(), compatibility["frozen_evidence_sha256"])
        self.assertEqual(json.loads(original)["conclusion"]["verdict"], "Honest-FAIL")
        for path in compatibility["pending_claim_drafts"] + compatibility["later_claim_drafts"]:
            self.assertEqual(json.loads((REPO / path).read_text())["status"], "draft")
        self.assertFalse(ledger["science"]["full_replay_performed"])
        for case_path in (compatibility["owner_pending"], compatibility["owner_later"]):
            output = subprocess.check_output(
                [sys.executable, "tools/benchmark_registry.py", "--case", case_path], cwd=REPO, text=True,
            )
            (case,) = json.loads(output)["cases"]
            self.assertEqual(case["id"], case_path)
            self.assertEqual(case["current_acceptance"], "not_evaluated")
            variant = case["variants"][0]
            if case_path == compatibility["owner_pending"]:
                self.assertEqual(variant["results"], [compatibility["frozen_evidence"]])
                self.assertEqual(set(variant["reference_files"]), set(compatibility["pending_claim_drafts"]))
            else:
                self.assertIn(compatibility["later_case_pointer"], variant["evidence"])
                self.assertNotIn(compatibility["frozen_evidence"], variant["results"])
        indexes = ("development/verification/mcmp_poiseuille_knowledge_loop/README.md",
                   compatibility["owner_pending"] + "/README.md", compatibility["later_case_pointer"])
        for p in indexes:
            index = REPO / p
            for link in re.findall(r"\]\(([^)]+)\)", index.read_text()):
                self.assertTrue((index.parent / link).is_file(), link)

    def test_three_historical_audit_folders_preserve_evidence_and_case_ids(self):
        ledger_path = LEDGER_PATH.parent / "historical-audit-docs-2026-11-27.json"
        ledger = json.loads(ledger_path.read_text())
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            rows = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                    if row["verification"] == ledger_path.relative_to(REPO).as_posix()}
        self.assertEqual(rows, {entry["old_path"]: entry["new_path"] for entry in ledger["entries"]})
        self.assertEqual(len(rows), 9)
        self.assertEqual(len(ledger["retired_second_level_directories"]), 3)
        for path in ledger["retired_second_level_directories"]:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", path], cwd=REPO))
        for entry in ledger["entries"] + ledger["compatibility"]["m21631_prior_archived_evidence"]:
            self.assert_fingerprint(REPO / entry["new_path"], entry["original_sha256"], entry["size_bytes"])
            index = subprocess.check_output(
                ["git", "ls-files", "-s", "--", entry["new_path"]], cwd=REPO, text=True,
            ).split()
            self.assertEqual(index[:2], [entry["git_mode"], entry["original_git_blob"]])
        self.assertFalse(ledger["science"]["full_scientific_replay_performed"])
        owners = ledger["compatibility"]["case_ownership"]
        for mission, path in owners.items():
            output = subprocess.check_output(
                [sys.executable, "tools/benchmark_registry.py", "--case", path], cwd=REPO, text=True,
            )
            (record,) = json.loads(output)["cases"]
            self.assertEqual(record["current_acceptance"], "not_evaluated")
            (variant,) = record["variants"]
            if mission == "m20183":
                self.assertEqual(variant["archive_bucket"], "pending")
                self.assertEqual(variant["results"], ["development/audit/m20183/audit_checks.json"])
                summary = json.loads((REPO / variant["results"][0]).read_text())
                self.assertEqual(summary["delivered_record"]["n_runs_in_dir"], 1)
                self.assertFalse(summary["ledger_criteria"]["publication_target_3pct_met"])
            elif mission == "m18785":
                self.assertIn("development/audit/m18785/AUDIT_m18753.md", variant["documentation"])
                self.assertIn("development/audit/m18785/AUDIT_m18753.md", variant["evidence"])
            else:
                self.assertIn(ledger["compatibility"]["m21631_case_pointer"], variant["evidence"])
        for path in ("development/audit/m20183/README.md", "development/audit/m18785/README.md",
                     ledger["compatibility"]["m21631_case_pointer"]):
            index = REPO / path
            for link in re.findall(r"\]\(([^)]+)\)", index.read_text()):
                self.assertTrue((index.parent / link).is_file(), link)

    def test_m18430_audit_tools_preserve_provenance_and_report_url(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "m18430-audit-tools-2026-11-22.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(ledger["entries"]), 3)
        for item in ledger["entries"]:
            with self.subTest(path=item["old_path"]):
                target = REPO / item["new_path"]
                self.assert_fingerprint(
                    target,
                    item.get("new_sha256", item["original_sha256"]),
                    item.get("new_size_bytes", item["size_bytes"]),
                )
                index = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", item["new_path"]],
                    cwd=REPO,
                    text=True,
                ).split()
                self.assertEqual(
                    index[:2],
                    [item["git_mode"], item.get("new_git_blob", item["original_git_blob"])],
                )
                self.assertFalse((REPO / item["old_path"]).exists())
        report = (REPO / ledger["published_report"]).read_text(encoding="utf-8")
        self.assertIn("](../development/audit/m18430/)", report)

    def test_sphere_d3q27_sdaa_evidence_keeps_its_failure_scope(self):
        ledger_path = (
            LEDGER_PATH.parent / "sphere-re100-d3q27-sdaa-artifacts-2026-11-22.json"
        )
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        self.assertEqual(ledger["root_directories_before"], 13)
        self.assertEqual(ledger["root_directories_after"], 13)
        self.assertEqual(len(ledger["entries"]), 8)
        with (REPO / "docs/architecture/migration_manifest.csv").open(
            encoding="utf-8"
        ) as stream:
            manifest = {
                row["old_path"]: row["new_path"]
                for row in csv.DictReader(stream)
                if row["verification"] == ledger_path.relative_to(REPO).as_posix()
            }
        self.assertEqual(
            manifest,
            {item["old_path"]: item["new_path"] for item in ledger["entries"]},
        )
        for item in ledger["entries"]:
            with self.subTest(path=item["new_path"]):
                raw = (REPO / item["new_path"]).read_bytes()
                self.assertEqual(len(raw), item["size_bytes"])
                self.assertEqual(
                    hashlib.sha256(raw).hexdigest(), item["original_sha256"]
                )
                index = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", item["new_path"]],
                    cwd=REPO,
                    text=True,
                ).split()
                self.assertEqual(
                    index[:2], [item["git_mode"], item["original_git_blob"]]
                )
                self.assertFalse((REPO / item["old_path"]).exists())

        case = REPO / "benchmarks/verified/sphere_re100_d3q27"
        evidence = case / "evidence/sdaa_sweep"
        summary = json.loads((evidence / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["overall_verdict"], "FAIL")
        self.assertEqual(len(list((evidence / "tiers").glob("*.json"))), 5)
        self.assertEqual(len(list((evidence / "logs").glob("*.log"))), 2)
        self.assertFalse(
            subprocess.check_output(
                ["git", "ls-files", "--", "artifacts/logs"], cwd=REPO, text=True
            )
        )
        readme = (case / "README.md").read_text(encoding="utf-8")
        self.assertIn("not extend", readme)
        self.assertIn("sweep remains FAIL", readme)

        runner = (SDAA_SWEEP_DRIVERS / "sdaa_sweep_sphere_re100_d3q27.py").read_text()
        self.assertNotIn('REPO / "artifacts"', runner)
        self.assertIn('EVIDENCE_DIR / "summary.json"', runner)
        self.assertIn('EVIDENCE_DIR / "tiers"', runner)
        self.assertIn('EVIDENCE_DIR / "logs" / "sweep.log"', runner)

    def test_board_audit_and_design_script_handoff_preserves_inputs(self):
        path = LEDGER_PATH.parent / "board-audit-and-design-script-handoff-2026-11-28.json"
        ledger = json.loads(path.read_text())
        with (path.parent.parent / "migration_manifest.csv").open(newline="") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                        if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(manifest, {row["old_path"]: row["new_path"] for row in ledger["entries"]})
        self.assertEqual(len(manifest), 5)
        for old in manifest:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", old], cwd=REPO))
        for row in ledger["entries"] + ledger["consumer_edits"] + ledger["unchanged_prior_archive"]:
            destination = REPO / row["new_path"]
            self.assert_fingerprint(destination, row["current_sha256"], row["current_size_bytes"])
            mode, blob = subprocess.check_output(
                ["git", "ls-files", "-s", "--", row["new_path"]], cwd=REPO, text=True,
            ).split()[:2]
            self.assertEqual(mode, row["git_mode"])
            original = destination.read_bytes()
            if row["byte_identical"]:
                self.assertEqual(blob, row["original_git_blob"])
            else:
                for edit in reversed(row["path_only_replacements"]):
                    self.assertEqual(original.count(edit["new"].encode()), edit["count"])
                    original = original.replace(edit["new"].encode(), edit["old"].encode())
            self.assertEqual(hashlib.sha256(original).hexdigest(), row["original_sha256"])
            self.assertEqual(len(original), row["size_bytes"])
            self.assertEqual(hashlib.sha1(f"blob {len(original)}\0".encode() + original).hexdigest(),
                             row["original_git_blob"])
        for row in ledger["derived_reanalysis"]:
            self.assert_fingerprint(REPO / row["path"], row["sha256"], row["size_bytes"])
            self.assertTrue(row["before_after_byte_identical"])
            self.assertFalse(row["original_historical_artifact_recovered"])
        for name in ledger["compatibility"]["navigation_indexes"]:
            index = REPO / name
            for target in re.findall(r"\]\(([^)]+)\)", index.read_text()):
                self.assertTrue((index.parent / target).is_file(), target)
        for row in ledger["consumer_edits"]:
            report = REPO / row["new_path"]
            for target in re.findall(r"\]\(([^)]+)\)", row["path_only_replacements"][0]["new"]):
                self.assertTrue((report.parent / target).is_file(), target)
        registry = json.loads(subprocess.check_output(
            [sys.executable, "tools/benchmark_registry.py", "--json"], cwd=REPO,
        ))["cases"]
        for owner, pointer in (
            ("benchmarks/case_pool/mcmp_laplace",
             "benchmarks/case_pool/mcmp_laplace/evidence/m15891/README.md"),
            ("benchmarks/verified/mcmp_stratified_poiseuille",
             "benchmarks/verified/mcmp_stratified_poiseuille/evidence/m15044/README.md"),
        ):
            variant = next((row, v) for row in registry for v in row["variants"]
                           if v["path"] == owner)
            self.assertEqual(variant[0]["current_acceptance"], "not_evaluated")
            self.assertIn(pointer, variant[1]["evidence"])
        self.assertFalse(ledger["science"]["full_scientific_replay_performed"])
        self.assertEqual(ledger["science"]["current_acceptance"], "not_evaluated")

    def test_dfg_board_and_drop_design_records_preserve_originals(self):
        path = LEDGER_PATH.parent / "dfg-board-and-drop-design-records-2026-11-27.json"
        ledger = json.loads(path.read_text())
        with (path.parent.parent / "migration_manifest.csv").open(newline="") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                        if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(manifest, {row["old_path"]: row["new_path"] for row in ledger["entries"]})
        self.assertEqual(len(manifest), 6)
        for retired in ledger["retired_second_level_directories"]:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", retired], cwd=REPO))
        for row in ledger["entries"] + ledger["unchanged_prior_archive"]:
            self.assertTrue(row["byte_identical"])
            destination = REPO / row["new_path"]
            self.assert_fingerprint(destination, row["original_sha256"], row["size_bytes"])
            mode, blob = subprocess.check_output(
                ["git", "ls-files", "-s", "--", row["new_path"]], cwd=REPO, text=True,
            ).split()[:2]
            self.assertEqual(mode, row["git_mode"])
            self.assertEqual(blob, row["original_git_blob"])
        for name in ledger["compatibility"]["navigation_indexes"]:
            index = REPO / name
            for target in re.findall(r"\]\(([^)]+)\)", index.read_text()):
                self.assertTrue((index.parent / target).is_file(), target)
        registry = json.loads(subprocess.check_output(
            [sys.executable, "tools/benchmark_registry.py", "--json"], cwd=REPO,
        ))["cases"]
        case = next(row for row in registry if row["id"] == "benchmarks/pending/cylinder_re100_dfg")
        self.assertEqual(case["current_acceptance"], "not_evaluated")
        self.assertEqual(case["variants"][0]["archive_bucket"], "pending")
        self.assertIn("benchmarks/pending/cylinder_re100_dfg/evidence/m15884/README.md",
                      case["variants"][0]["evidence"])
        self.assertTrue(any(row["id"] == "matrix:cylinder_re100_dfg" for row in registry))
        self.assertEqual(ledger["science"]["current_acceptance"], "not_evaluated")

    def test_zero_solver_audit_handoff_preserves_inputs_and_replay_scope(self):
        path = LEDGER_PATH.parent / "zero-solver-audit-handoff-2026-11-28.json"
        ledger = json.loads(path.read_text())
        with (path.parent.parent / "migration_manifest.csv").open(newline="") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                        if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(manifest, {row["old_path"]: row["new_path"] for row in ledger["entries"]})
        self.assertEqual(len(manifest), 2)
        for row in ledger["entries"] + ledger["consumer_edits"]:
            dest = REPO / row["new_path"]
            self.assert_fingerprint(dest, row["current_sha256"], row["current_size_bytes"])
            original = dest.read_bytes()
            for edit in reversed(row["path_only_edits"]):
                old, new = edit["old"].encode(), edit["new"].encode()
                self.assertEqual(original.count(new), edit["count"])
                original = original.replace(new, old)
            self.assertEqual(hashlib.sha256(original).hexdigest(), row["original_sha256"])
            self.assertEqual(len(original), row["size_bytes"])
            if row["old_path"] != row["new_path"]:
                self.assertFalse((REPO / row["old_path"]).exists())
        for row in ledger["unchanged_related_assets_at_migration"]:
            self.assert_fingerprint(REPO / row["new_path"], row["original_sha256"], row["size_bytes"])
        for name in ledger["compatibility"]["navigation_indexes"]:
            index = REPO / name
            for target in re.findall(r"\]\(([^)]+)\)", index.read_text()):
                self.assertTrue((index.parent / target).is_file(), target)
        registry = json.loads(subprocess.check_output(
            [sys.executable, "tools/benchmark_registry.py", "--json"], cwd=REPO,
        ))["cases"]
        case_ids = {row["id"] for row in registry}
        for audit in ledger["compatibility"]["historical_dependencies"]:
            self.assertTrue(set(audit["case_ids"]) <= case_ids)
        for row in ledger["entries"]:
            found = json.loads(subprocess.check_output(
                [sys.executable, "tools/benchmark_registry.py", "--find-asset",
                 Path(row["new_path"]).name], cwd=REPO,
            ))["paths"]
            self.assertIn(row["new_path"], found)
        parity = ledger["compatibility"]["replay_parity"]
        self.assertEqual(len(parity["outputs"]), 5)
        self.assertEqual(len(parity["stdout"]), 2)
        self.assertTrue(all(row["before_after_byte_identical"] for row in parity["outputs"] + parity["stdout"]))
        self.assertEqual(ledger["science"]["current_acceptance"], "not_evaluated")
        self.assertFalse(ledger["science"]["algorithm_changed"])
        self.assertFalse(ledger["science"]["full_scientific_replay_performed"])

    def test_drop_entry_indexes_keep_original_assets_and_case_navigation(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "drop-entry-indexes-2026-11-23.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(ledger["entries"]), 9)
        for item in ledger["entries"]:
            with self.subTest(path=item["old_path"]):
                target = REPO / item["new_path"]
                self.assert_fingerprint(target, item["original_sha256"], item["size_bytes"])
                index = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", item["new_path"]],
                    cwd=REPO,
                    text=True,
                ).split()
                self.assertEqual(index[:2], [item["git_mode"], item["original_git_blob"]])
                self.assertFalse((REPO / item["old_path"]).exists())
        self.assertFalse((REPO / ledger["old_directory"]).exists())
        case = REPO / "benchmarks/water_entry/single_drop/README.md"
        self.assertIn(f"](../../../{ledger['new_directory']}/)", case.read_text(encoding="utf-8"))

    def test_historical_collision_matrix_retires_active_frozen_source_consumers(self):
        path = LEDGER_PATH.parent / "historical-collision-matrix-2026-11-28.json"
        ledger = json.loads(path.read_text())
        with (path.parent.parent / "migration_manifest.csv").open(newline="") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                        if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(manifest, {row["old_path"]: row["new_path"] for row in ledger["entries"]})
        self.assertEqual(len(manifest), 7)
        for directory in ledger["retired_second_level_directories"]:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", directory], cwd=REPO))
        for row in ledger["entries"]:
            dest = REPO / row["new_path"]
            self.assert_fingerprint(dest, row["current_sha256"], row["current_size_bytes"])
            original = dest.read_bytes()
            for edit in reversed(row.get("path_only_edits", [])):
                old, new = edit["old"].encode(), edit["new"].encode()
                self.assertEqual(original.count(new), edit["count"])
                original = original.replace(new, old)
            self.assertEqual(hashlib.sha256(original).hexdigest(), row["original_sha256"])
            self.assertEqual(len(original), row["size_bytes"])
            self.assertFalse((REPO / row["old_path"]).exists())
            mode = subprocess.check_output(
                ["git", "ls-files", "-s", "--", row["new_path"]], cwd=REPO, text=True,
            ).split()[0]
            self.assertEqual(mode, row["git_mode"])
        for row in ledger["unchanged_related_assets_at_migration"]:
            self.assert_fingerprint(REPO / row["new_path"], row["original_sha256"], row["size_bytes"])
        source = ledger["source_authority"]
        policy = json.loads((REPO / source["policy"]).read_text())
        for old in source["removed_exact_allowlist_entries"]:
            self.assertNotIn(old, policy["allowed_active_references"])
        self.assertEqual(policy["historical_tree_oid"], source["unchanged_frozen_tree"])
        for name in ledger["compatibility"]["navigation_indexes"]:
            index = REPO / name
            for target in re.findall(r"\]\(([^)]+)\)", index.read_text()):
                if target.startswith("https://github.com/cxs503/TensorLBM/blob/"):
                    self.assertTrue(target.startswith(
                        "https://github.com/cxs503/TensorLBM/blob/" + ledger["base_main"] + "/"))
                else:
                    self.assertTrue((index.parent / target).is_file(), target)
        for row in ledger["entries"]:
            if row["new_path"].endswith(".py"):
                found = json.loads(subprocess.check_output(
                    [sys.executable, "tools/benchmark_registry.py", "--find-asset",
                     row["new_path"]], cwd=REPO,
                ))["paths"]
                self.assertIn(row["new_path"], found)
        for receipt in ledger["compatibility"]["import_receipts"]:
            self.assertEqual(receipt["actual_imported_package"],
                             policy["historical_root"] + "/src/tensorlbm/__init__.py")
            self.assertTrue(receipt["repo_root_before_after_identical"])
            self.assertFalse(receipt["solver_invoked"])
        self.assertTrue(ledger["compatibility"]["cli_help_parity"]["before_after_byte_identical"])
        self.assertFalse(ledger["science"]["algorithm_changed"])
        self.assertFalse(ledger["science"]["full_scientific_replay_performed"])
        self.assertEqual(ledger["science"]["current_acceptance"], "not_evaluated")

    def test_cavity_sdaa_evidence_preserves_bytes_links_and_scope(self):
        ledger_path = LEDGER_PATH.parent / "cavity-sdaa-artifacts-2026-11-22.json"
        entries = json.loads(ledger_path.read_text(encoding="utf-8"))["entries"]
        self.assertEqual(len(entries), 7)
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            paths = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)}
        evidence = {}
        for entry in entries:
            with self.subTest(path=entry["old_path"]):
                target = REPO / entry["new_path"]
                raw = target.read_bytes()
                self.assertEqual(len(raw), entry["size_bytes"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["original_sha256"])
                indexed = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", entry["new_path"]],
                    cwd=REPO, text=True,
                ).split()
                self.assertEqual(indexed[:2], [entry["git_mode"], entry["original_git_blob"]])
                self.assertEqual(paths[entry["old_path"]], entry["new_path"])
                self.assertFalse((REPO / entry["old_path"]).exists())
                readme = (target.parents[2] / "README.md").read_text(encoding="utf-8")
                self.assertIn(f"](evidence/sdaa/{target.name})", readme)
                if target.suffix == ".json":
                    evidence[target.stem] = json.loads(raw)
        self.assertEqual(evidence["sdaa_cavity_natural_convection"]["overall_verdict"], "FAIL")
        self.assertEqual(evidence["sdaa_thermal_cavity"]["summary"]["passed_cases"], 0)
        re400 = evidence["sdaa_cavity_evidence"]
        self.assertIn("cpu_reference", re400)
        self.assertIn("sdaa_short_run", re400)

    def test_cavity_three_way_archive_preserves_original_objects_and_gap(self):
        ledger_path = LEDGER_PATH.parent / "cavity-three-way-archive-2026-11-27.json"
        ledger = json.loads(ledger_path.read_text())
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            paths = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                     if row["verification"] == ledger_path.relative_to(REPO).as_posix()}
        self.assertEqual(paths, {entry["old_path"]: entry["new_path"] for entry in ledger["entries"]})
        self.assertEqual(len(paths), 14)
        for entry in ledger["entries"]:
            with self.subTest(path=entry["new_path"]):
                self.assert_fingerprint(
                    REPO / entry["new_path"], entry["original_sha256"], entry["size_bytes"],
                )
                index = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", entry["new_path"]], cwd=REPO, text=True,
                ).split()
                self.assertEqual(index[:2], [entry["git_mode"], entry["original_git_blob"]])
                self.assertFalse((REPO / entry["old_path"]).exists())
        self.assertFalse((REPO / ledger["retired_second_level_directories"][0]).exists())
        self.assertEqual(len(ledger["evidence_gaps"]["not_retrievable_as_tracked_objects_in_fetched_refs"]), 7)
        self.assertFalse(ledger["science"]["rerun_performed"])
        self.assertFalse(ledger["archived_comparison"]["verdict"]["inter_tool_pass"])
        case_readme = REPO / "benchmarks/verified/cavity/re100/README.md"
        self.assertIn("benchmarks/verified/cavity/re100/run.py", case_readme.read_text())
        index = REPO / "benchmarks/cross_validation/cavity_three_way/README.md"
        self.assertIn("../../../cross_validation/cavity_three_way/README.md", case_readme.read_text())
        for link in re.findall(r"\]\(([^)]+)\)", index.read_text()):
            self.assertTrue((index.parent / link).is_file(), link)

    def test_p482_report_helpers_keep_frozen_input_and_public_report_contract(self):
        path = LEDGER_PATH.parent / "p482-report-helper-handoff-2026-11-28.json"
        ledger = json.loads(path.read_text())
        with (path.parent.parent / "migration_manifest.csv").open(newline="") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                        if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(manifest, {row["old_path"]: row["new_path"] for row in ledger["entries"]})
        self.assertEqual(len(manifest), 2)
        for row in ledger["entries"] + ledger["consumer_edits"]:
            dest = REPO / row["new_path"]
            self.assert_fingerprint(dest, row["current_sha256"], row["current_size_bytes"])
            original = dest.read_bytes()
            for edit in reversed(row["path_only_edits"]):
                old, new = edit["old"].encode(), edit["new"].encode()
                self.assertEqual(original.count(new), edit["count"])
                original = original.replace(new, old)
            self.assertEqual(hashlib.sha256(original).hexdigest(), row["original_sha256"])
            self.assertEqual(len(original), row["size_bytes"])
            if row["old_path"] != row["new_path"]:
                self.assertFalse((REPO / row["old_path"]).exists())
        for row in ledger["unchanged_related_assets_at_migration"]:
            self.assert_fingerprint(REPO / row["new_path"], row["original_sha256"], row["size_bytes"])
        policy = json.loads((REPO / ledger["source_authority"]["policy"]).read_text())
        for old in ledger["source_authority"]["removed_exact_allowlist_entries"]:
            self.assertNotIn(old, policy["allowed_active_references"])
        self.assertEqual(policy["historical_tree_oid"], ledger["source_authority"]["unchanged_frozen_tree"])
        for name in ledger["compatibility"]["navigation_indexes"]:
            index = REPO / name
            for target in re.findall(r"\]\(([^)]+)\)", index.read_text()):
                self.assertTrue((index.parent / target).is_file(), target)
        for row in ledger["entries"]:
            found = json.loads(subprocess.check_output(
                [sys.executable, "tools/benchmark_registry.py", "--find-asset",
                 row["new_path"]], cwd=REPO,
            ))["paths"]
            self.assertIn(row["new_path"], found)
        receipt = ledger["compatibility"]["historical_manifest_receipt"]
        self.assertEqual(sum(row["status"] == "match" for row in receipt), 2)
        self.assertEqual(sum(row["status"] == "missing" for row in receipt), 1)
        for row in receipt:
            if row["status"] == "match":
                self.assertEqual(hashlib.sha256((REPO / row["path"]).read_bytes()).hexdigest(),
                                 row["expected_sha256"])
            else:
                self.assertIsNone(row["actual_sha256"])
        parity = ledger["compatibility"]["render_parity"]
        self.assertTrue(parity["reports_identical_after_reversing_only_relocated_command"])
        self.assertTrue(parity["payload_markdown_byte_identical"])
        self.assertFalse(parity["solver_invoked"])
        self.assertFalse(parity["publication_invoked"])
        self.assertEqual(ledger["science"]["current_acceptance"], "not_evaluated")
        self.assertFalse(ledger["science"]["algorithm_changed"])
        self.assertFalse(ledger["science"]["full_scientific_replay_performed"])

    def test_poiseuille_sdaa_evidence_preserves_bytes_links_and_scope(self):
        ledger_path = LEDGER_PATH.parent / "poiseuille-channel-duct-pipe-sdaa-2026-11-22.json"
        entries = json.loads(ledger_path.read_text(encoding="utf-8"))["entries"]
        self.assertEqual(len(entries), 4)
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            paths = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)}
        preserved = {}
        for item in entries:
            with self.subTest(path=item["old_path"]):
                target = REPO / item["new_path"]
                raw = target.read_bytes()
                self.assertEqual(len(raw), item["size_bytes"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item["original_sha256"])
                indexed = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", item["new_path"]],
                    cwd=REPO, text=True,
                ).split()
                self.assertEqual(indexed[:2], [item["git_mode"], item["original_git_blob"]])
                self.assertEqual(paths[item["old_path"]], item["new_path"])
                self.assertFalse((REPO / item["old_path"]).exists())
                case = target.parents[2]
                link = f"](evidence/sdaa/{target.name})"
                self.assertIn(link, (case / "README.md").read_text(encoding="utf-8"))
                report = REPO / "docs" / "sdaa_sweep" / f"{case.name}.md"
                self.assertIn(f"](../../{item['new_path']})", report.read_text(encoding="utf-8"))
                preserved[target.stem] = json.loads(raw)
        self.assertEqual(preserved["sdaa_poiseuille_2d"]["verdict"]["status"], "FAIL")
        self.assertEqual(preserved["sdaa_poiseuille_3d_duct"]["overall_verdict"], "FAIL")
        pipe = preserved["sdaa_poiseuille_3d_pipe"]
        self.assertEqual((pipe["verdict"], pipe["config"]["R"]), ("PASS", 20))
        self.assertNotIn("verdict", preserved["sdaa_poiseuille_3d_pipe_case"])

    def test_suboff_evidence_keeps_original_paths_in_the_ledger_and_exact_bytes(self):
        ledger_path = LEDGER_PATH.parent / "suboff-evidence-2026-11-22.json"
        entries = json.loads(ledger_path.read_text(encoding="utf-8"))["entries"]
        self.assertEqual(len(entries), 10)
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            paths = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)}

        for entry in entries:
            with self.subTest(path=entry["old_path"]):
                target = REPO / entry["new_path"]
                raw = target.read_bytes()
                self.assertEqual(len(raw), entry["size_bytes"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["original_sha256"])
                self.assertEqual(paths[entry["old_path"]], entry["new_path"])
                self.assertFalse((REPO / entry["old_path"]).exists())

        case = REPO / "benchmarks/verified/suboff_sail_resistance"
        evidence = case / "evidence/sdaa"
        m288 = evidence / "sdaa_suboff_sail_resistance_m288.json"
        m289 = evidence / "sdaa_suboff_sail_resistance_m289.json"
        self.assertEqual(m288.read_bytes(), m289.read_bytes())
        probe = json.loads((evidence / "sdaa_suboff_sail_resistance_v3.json").read_text())
        self.assertEqual(probe["overall"], "HONEST-FAIL")
        readme = (case / "README.md").read_text(encoding="utf-8")
        for target in evidence.glob("sdaa_suboff_sail_resistance*.json"):
            self.assertIn(f"](evidence/sdaa/{target.name})", readme)

    def test_kpi_aa_and_sparse_audit_archives_keep_original_scope(self):
        path = LEDGER_PATH.parent / "kpi-aa-sparse-audit-docs-2026-11-27.json"
        ledger = json.loads(path.read_text())
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            rows = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                    if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(len(rows), 53)
        self.assertEqual(rows, {row["old_path"]: row["new_path"] for row in ledger["entries"]})
        self.assertEqual(len(ledger["retired_second_level_directories"]), 3)
        for retired in ledger["retired_second_level_directories"]:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", retired], cwd=REPO))
        self.assertEqual(len(ledger["prior_archived_evidence"]), 4)
        for entry in ledger["entries"] + ledger["prior_archived_evidence"]:
            self.assert_fingerprint(REPO / entry["new_path"], entry["original_sha256"], entry["size_bytes"])
            index = subprocess.check_output(
                ["git", "ls-files", "-s", "--", entry["new_path"]], cwd=REPO, text=True,
            ).split()
            self.assertEqual(index[:2], [entry["git_mode"], entry["original_git_blob"]])
        edit, = ledger["consumer_edits"]
        current = (REPO / edit["new_path"]).read_bytes()
        self.assertEqual(current.count(edit["path_only_edit"]["new"].encode()), 1)
        original = current.replace(
            edit["path_only_edit"]["new"].encode(), edit["path_only_edit"]["old"].encode(),
        )
        self.assertEqual(hashlib.sha256(original).hexdigest(), edit["original_sha256"])
        compat = ledger["compatibility"]
        snapshots = compat["snapshot_verification"]
        self.assertEqual(len(snapshots["m17572"]["checked"]), 14)
        self.assertEqual(len(snapshots["m18098"]["checked"]), 8)
        self.assertEqual(snapshots["m17572"]["missing_preexisting"], [])
        for snapshot in snapshots.values():
            for entry in snapshot["checked"]:
                raw = (REPO / entry["candidate_path"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["sha256"])
                if "size_bytes" in entry:
                    self.assertEqual(len(raw), entry["size_bytes"])
        missing, = snapshots["m18098"]["missing_preexisting"]
        self.assertEqual(missing["candidate_path"], "development/audit/m18098/evidence_snapshot/run.log")
        for mission, owner in compat["case_ownership"].items():
            if mission == "m18098":
                self.assertIsNone(owner["id"])
                self.assertIsNone(owner["path"])
                output = subprocess.check_output(
                    [sys.executable, "tools/benchmark_registry.py", "--find-asset", "development/audit/m18098"],
                    cwd=REPO, text=True,
                )
                paths = json.loads(output)["paths"]
                for entry in ledger["entries"]:
                    if entry["new_path"].startswith("development/audit/m18098/"):
                        self.assertIn(entry["new_path"], paths)
                continue
            raw = subprocess.check_output(
                [sys.executable, "tools/benchmark_registry.py", "--case", owner["id"]], cwd=REPO, text=True,
            )
            (record,) = json.loads(raw)["cases"]
            self.assertEqual(record["current_acceptance"], "not_evaluated")
            (variant,) = record["variants"]
            self.assertEqual(variant["path"], owner["path"])
            if mission == "m17085":
                self.assertEqual(variant["archive_bucket"], "verified")
                self.assertIn(owner["pointer"], variant["evidence"])
                self.assertEqual(variant["results"], [owner["path"] + "/result.json"])
            else:
                self.assertIn("benchmarks/xval_aa_pattern_even_odd.py", variant["source"])
                self.assertIn(owner["audit_index"], variant["documentation"])
                for entry in ledger["entries"]:
                    if entry["new_path"].startswith("development/audit/m17572/"):
                        self.assertIn(entry["new_path"], variant["evidence"])
                self.assertIn("development/audit/m17514/verdict_m17514.sql", variant["evidence"])
        for name in compat["navigation_indexes"]:
            index = REPO / name
            for link in re.findall(r"\]\(([^)]+)\)", index.read_text()):
                self.assertTrue((index.parent / link).is_file(), link)
        self.assertFalse(ledger["science"]["full_scientific_replay_performed"])
        self.assertFalse(ledger["science"]["algorithm_changed"])

    def test_aitken_archive_paths_and_original_evidence(self):
        ledger_path = LEDGER_PATH.parent / "aitken-xval-archive-2026-11-27.json"
        ledger = json.loads(ledger_path.read_text())
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            paths = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                     if row["verification"] == ledger_path.relative_to(REPO).as_posix()}
        self.assertEqual(paths, {entry["old_path"]: entry["new_path"] for entry in ledger["entries"]})
        self.assertEqual(len(paths), 11)
        self.assertEqual(sum(entry["byte_identical"] for entry in ledger["entries"]), 9)
        for entry in ledger["entries"]:
            self.assert_fingerprint(REPO / entry["new_path"], entry["current_sha256"],
                                    entry["current_size_bytes"])
            index = subprocess.check_output(
                ["git", "ls-files", "-s", "--", entry["new_path"]], cwd=REPO, text=True,
            ).split()
            self.assertEqual(index[0], entry["git_mode"])
            if entry["byte_identical"]:
                self.assertEqual(index[1], entry["original_git_blob"])
            self.assertFalse((REPO / entry["old_path"]).exists())
        self.assertFalse((REPO / ledger["retired_second_level_directories"][0]).exists())
        self.assertEqual(len(ledger["evidence_gaps"]["missing_original_figures"]), 5)
        self.assertFalse(ledger["science"]["full_replay_performed"])
        # The ledger pins the migration-time hashes; maintained consumers may
        # evolve independently. Enforce path contracts, not permanent solver hashes.
        for entry in ledger["consumer_edits"]:
            consumer = REPO / entry["new_path"]
            self.assertTrue(consumer.is_file())
            self.assertNotIn("xval/aitken_underrelaxation_life", consumer.read_text())
        for path in ("tests/test_fsi_aitken_guard_scale.py", "tests/test_fsi_aitken_life_xval.py"):
            self.assertIn('"development", "verification", "aitken_underrelaxation_life"',
                          (REPO / path).read_text())
        index = REPO / "benchmarks/cross_validation/aitken_underrelaxation_life/README.md"
        for link in re.findall(r"\]\(([^)]+)\)", index.read_text()):
            self.assertTrue((index.parent / link).is_file(), link)
        result = subprocess.check_output(
            [sys.executable, "tools/benchmark_registry.py", "--case", "xval:aitken_underrelaxation_life"],
            cwd=REPO, text=True,
        )
        (case,) = json.loads(result)["cases"]
        self.assertEqual(case["current_acceptance"], "not_evaluated")
        self.assertEqual(case["variants"][0]["results"], [
            "development/verification/aitken_underrelaxation_life/out/results.json",
            "development/verification/aitken_underrelaxation_life/out/scale_guard.json",
        ])

    def test_docs_process_directories_are_retired_with_exact_provenance(self):
        ledger_path = LEDGER_PATH.parent / "docs-process-directories-2026-11-22.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        self.assertEqual(ledger["second_level_directories"]["docs"], {
            "before": 91,
            "after": 88,
        })
        self.assertEqual(len(ledger["entries"]), 8)
        with (REPO / "docs/architecture/migration_manifest.csv").open(
            encoding="utf-8"
        ) as stream:
            manifest = {
                row["old_path"]: row["new_path"]
                for row in csv.DictReader(stream)
                if row["verification"] == ledger_path.relative_to(REPO).as_posix()
            }
        self.assertEqual(
            manifest,
            {item["old_path"]: item["new_path"] for item in ledger["entries"]},
        )
        for item in ledger["entries"]:
            destination = REPO / item["new_path"]
            with self.subTest(path=item["old_path"]):
                self.assert_fingerprint(
                    destination, item["original_sha256"], item["size_bytes"]
                )
                indexed = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", item["new_path"]],
                    cwd=REPO,
                    text=True,
                ).split()
                self.assertEqual(
                    indexed[:2], [item["git_mode"], item["original_git_blob"]]
                )
                self.assertFalse((REPO / item["old_path"]).exists())
        for path in ("docs/council", "docs/history", "docs/smoke"):
            self.assertFalse((REPO / path).exists())

    def test_shan_chen_edm_rounds_keep_original_data_and_derived_figure_hashes(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "shan-chen-edm-rounds-xval-2026-11-27.json").read_text()
        )
        self.assertEqual(len(ledger["entries"]), 43)
        self.assertEqual(len(ledger["restored_derived_figures"]), 6)
        for entry in ledger["entries"]:
            with self.subTest(path=entry["new_path"]):
                self.assertFalse((REPO / entry["old_path"]).exists())
                path = REPO / entry["new_path"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 entry["current_sha256"])
                self.assertEqual(
                    subprocess.check_output(
                        ["git", "ls-files", "-s", "--", entry["new_path"]],
                        cwd=REPO, text=True,
                    ).split()[0], entry["git_mode"],
                )
                if path.suffix == ".json":
                    self.assertTrue(entry["byte_identical"])
                    self.assertEqual(entry["original_sha256"], entry["current_sha256"])
        for figure in ledger["restored_derived_figures"]:
            self.assertEqual(hashlib.sha256((REPO / figure["path"]).read_bytes()).hexdigest(),
                             figure["sha256"])
            for path, sha in figure["input_sha256"].items():
                self.assertEqual(hashlib.sha256((REPO / path).read_bytes()).hexdigest(), sha)
        for consumer in ledger["updated_consumers"]:
            self.assertEqual(
                hashlib.sha256((REPO / consumer["path"]).read_bytes()).hexdigest(),
                consumer["current_sha256"],
            )
        for tool in ledger["frozen_audit_tools"]:
            self.assertEqual(
                hashlib.sha256((REPO / tool["path"]).read_bytes()).hexdigest(),
                tool["sha256"],
            )
        self.assertEqual(ledger["validation"]["current_scientific_acceptance"],
                         "not_evaluated")

    def test_pointer_shift_and_board_audits_preserve_archive_scope(self):
        path = LEDGER_PATH.parent / "pointer-shift-and-board-audits-2026-11-27.json"
        ledger = json.loads(path.read_text())
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            rows = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                    if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(len(rows), 40)
        self.assertEqual(rows, {row["old_path"]: row["new_path"] for row in ledger["entries"]})
        self.assertEqual(len(ledger["retired_second_level_directories"]), 4)
        for retired in ledger["retired_second_level_directories"]:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", retired], cwd=REPO))
        self.assertEqual(len(ledger["prior_archived_evidence"]), 2)
        for entry in ledger["entries"] + ledger["prior_archived_evidence"]:
            self.assert_fingerprint(REPO / entry["new_path"], entry["original_sha256"], entry["size_bytes"])
            index = subprocess.check_output(
                ["git", "ls-files", "-s", "--", entry["new_path"]], cwd=REPO, text=True,
            ).split()
            self.assertEqual(index[:2], [entry["git_mode"], entry["original_git_blob"]])
        edit, = ledger["consumer_edits"]
        raw = (REPO / edit["new_path"]).read_bytes()
        mapping = edit["path_only_edit"]
        self.assertEqual(raw.count(mapping["new"].encode()), 1)
        original = raw.replace(mapping["new"].encode(), mapping["old"].encode())
        self.assertEqual(hashlib.sha256(original).hexdigest(), edit["original_sha256"])
        compat = ledger["compatibility"]
        self.assertEqual(len(compat["snippet_checks"]), 9)
        for entry in compat["snippet_checks"]:
            body = (REPO / entry["path"]).read_bytes().split(b"\n", 1)[1]
            self.assertEqual(hashlib.sha256(body).hexdigest(), entry["sha256_body"])
            self.assertEqual(len(body.splitlines()), entry["n_lines"])
        import base64
        f = compat["encoded_figure"]
        decoded = base64.b64decode((REPO / f["b64"]).read_text())
        self.assertEqual(decoded, (REPO / f["png"]).read_bytes())
        self.assertEqual(hashlib.sha256(decoded).hexdigest(), f["sha256"])
        self.assertEqual(len(decoded), f["size_bytes"])
        for mission, owner in compat["case_ownership"].items():
            if mission != "m18222":
                self.assertIsNone(owner["id"])
                self.assertIsNone(owner["case_path"])
                output = subprocess.check_output(
                    [sys.executable, "tools/benchmark_registry.py", "--find-asset", owner["archive_path"]],
                    cwd=REPO, text=True,
                )
                paths = json.loads(output)["paths"]
                for entry in ledger["entries"]:
                    if entry["new_path"].startswith(owner["archive_path"] + "/"):
                        self.assertIn(entry["new_path"], paths)
                continue
            output = subprocess.check_output(
                [sys.executable, "tools/benchmark_registry.py", "--case", owner["id"]], cwd=REPO, text=True,
            )
            (record,) = json.loads(output)["cases"]
            self.assertEqual(record["current_acceptance"], "not_evaluated")
            (variant,) = record["variants"]
            self.assertEqual(variant["path"], owner["case_path"])
            for entry in ledger["entries"] + ledger["prior_archived_evidence"]:
                if entry["new_path"].startswith(owner["archive_path"] + "/"):
                    self.assertIn(entry["new_path"], variant["evidence"])
            definition = json.loads((REPO / owner["manifest"]).read_text())
            self.assertIn("WITHHELD", definition["scope"])
            self.assertIn("Archive-only", definition["scope"])
        for name in compat["navigation_indexes"]:
            doc = REPO / name
            for link in re.findall(r"\]\(([^)]+)\)", doc.read_text()):
                self.assertTrue((doc.parent / link).is_file(), link)
        self.assertFalse(ledger["science"]["full_scientific_replay_performed"])
        self.assertFalse(ledger["science"]["algorithm_changed"])

    def test_historical_publication_script_handoff_retains_originals(self):
        path = LEDGER_PATH.parent / "historical-publication-script-handoff-2026-11-28.json"
        ledger = json.loads(path.read_text())
        with (path.parent.parent / "migration_manifest.csv").open(newline="") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                        if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(manifest, {row["old_path"]: row["new_path"] for row in ledger["entries"]})
        self.assertEqual(len(manifest), 5)
        for old in manifest:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", old], cwd=REPO))
        for row in ledger["entries"] + ledger["consumer_edits"]:
            destination = REPO / row["new_path"]
            self.assert_fingerprint(destination, row["current_sha256"], row["current_size_bytes"])
            mode, blob = subprocess.check_output(
                ["git", "ls-files", "-s", "--", row["new_path"]], cwd=REPO, text=True,
            ).split()[:2]
            self.assertEqual(mode, row["git_mode"])
            original = destination.read_bytes()
            if row["byte_identical"]:
                self.assertEqual(blob, row["original_git_blob"])
            else:
                edit = row["path_only_edit"]
                self.assertEqual(original.count(edit["new"].encode()), 1)
                original = original.replace(edit["new"].encode(), edit["old"].encode())
            self.assertEqual(hashlib.sha256(original).hexdigest(), row["original_sha256"])
            self.assertEqual(len(original), row["size_bytes"])
            self.assertEqual(hashlib.sha1(f"blob {len(original)}\0".encode() + original).hexdigest(),
                             row["original_git_blob"])
        for name in ledger["compatibility"]["navigation_indexes"]:
            index = REPO / name
            for target in re.findall(r"\]\(([^)]+)\)", index.read_text()):
                self.assertTrue((index.parent / target).is_file(), target)
        report = REPO / ledger["consumer_edits"][0]["new_path"]
        for target in re.findall(r"\]\(([^)]+)\)",
                                 ledger["consumer_edits"][0]["path_only_edit"]["new"]):
            self.assertTrue((report.parent / target).is_file(), target)
        registry = json.loads(subprocess.check_output(
            [sys.executable, "tools/benchmark_registry.py", "--json"], cwd=REPO,
        ))["cases"]
        for mission, owner in ledger["compatibility"]["existing_case_links"].items():
            record, variant = next((row, v) for row in registry for v in row["variants"]
                                   if v["path"] == owner)
            self.assertEqual(record["current_acceptance"], "not_evaluated")
            self.assertIn(f"{owner}/evidence/{mission}/README.md", variant["evidence"])
        self.assertEqual(ledger["compatibility"]["offline_payload_parity"]["figures_present"], 0)
        self.assertEqual(ledger["science"]["current_acceptance"], "not_evaluated")

    def test_xval_preregistrations_keep_original_objects_and_report_links(self):
        ledger_path = LEDGER_PATH.parent / "xval-preregistrations-2026-11-22.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        with (REPO / "docs/architecture/migration_manifest.csv").open(
            encoding="utf-8"
        ) as stream:
            manifest = {
                row["old_path"]: row["new_path"]
                for row in csv.DictReader(stream)
                if row["verification"] == ledger_path.relative_to(REPO).as_posix()
            }
        self.assertEqual(len(ledger["entries"]), 2)
        self.assertEqual(
            manifest,
            {item["old_path"]: item["new_path"] for item in ledger["entries"]},
        )
        for item in ledger["entries"]:
            with self.subTest(path=item["old_path"]):
                target = REPO / item["new_path"]
                self.assert_fingerprint(target, item["original_sha256"], item["size_bytes"])
                index = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", item["new_path"]],
                    cwd=REPO,
                    text=True,
                ).split()
                self.assertEqual(index[:2], [item["git_mode"], item["original_git_blob"]])
                self.assertFalse((REPO / item["old_path"]).exists())
                report = (REPO / item["published_report"]).read_text(encoding="utf-8")
                self.assertIn(f"](../{item['new_path']})", report)

    def test_addedmass_kc_archive_preserves_inputs_and_historical_failure(self):
        path = LEDGER_PATH.parent / "addedmass-kc-xval-2026-11-27.json"
        ledger = json.loads(path.read_text())
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            rows = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                    if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(len(rows), 29)
        self.assertEqual(rows, {row["old_path"]: row["new_path"] for row in ledger["entries"]})
        for retired in ledger["retired_second_level_directories"]:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", retired], cwd=REPO))
        self.assertEqual(sum(not item["byte_identical"] for item in ledger["entries"]), 2)
        for entry in ledger["entries"] + ledger["consumer_edits"]:
            raw = (REPO / entry["new_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["current_sha256"])
            self.assertEqual(len(raw), entry["current_size_bytes"])
            index = subprocess.check_output(
                ["git", "ls-files", "-s", "--", entry["new_path"]], cwd=REPO, text=True,
            ).split()
            self.assertEqual(index[0], entry["git_mode"])
            if entry["byte_identical"]:
                self.assertEqual(index[1], entry["original_git_blob"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["original_sha256"])
            else:
                edit = entry["path_only_edit"]
                self.assertEqual(raw.count(edit["new"].encode()), 1)
                original = raw.replace(edit["new"].encode(), edit["old"].encode())
                self.assertEqual(len(original), entry["size_bytes"])
                self.assertEqual(hashlib.sha256(original).hexdigest(), entry["original_sha256"])
                self.assertEqual(hashlib.sha1(
                    f"blob {len(original)}\0".encode() + original,
                ).hexdigest(), entry["original_git_blob"])
        compat = ledger["compatibility"]
        raw = subprocess.check_output(
            [sys.executable, "tools/benchmark_registry.py", "--case", compat["case_id"]], cwd=REPO, text=True,
        )
        (record,) = json.loads(raw)["cases"]
        self.assertEqual(record["current_acceptance"], "not_evaluated")
        (variant,) = record["variants"]
        self.assertEqual(variant["path"], compat["case_path"])
        for entry in ledger["entries"]:
            self.assertIn(entry["new_path"], variant["evidence"])
        historic = json.loads((REPO / compat["archive_path"] / "results.json").read_text())
        self.assertEqual(historic["verdict"]["verdict"], "HONEST_FAIL")
        self.assertEqual(historic["verdict"]["passing_arms"], [])
        self.assertEqual(ledger["science"]["original_verdict"], "HONEST_FAIL")
        self.assertFalse(ledger["science"]["full_scientific_replay_performed"])
        self.assertFalse(ledger["science"]["algorithm_changed"])
        verify = ledger["verification"]
        self.assertEqual(verify["baseline_render_hashes"], verify["candidate_render_hashes"])
        self.assertEqual(verify["baseline_targeted_tests"], verify["candidate_targeted_tests"])
        self.assertEqual(verify["candidate_targeted_tests"], 4)
        index = REPO / compat["readme"]
        for link in re.findall(r"\]\(([^)]+)\)", index.read_text()):
            self.assertTrue((index.parent / link).is_file(), link)

    def test_case_reporting_tools_keep_source_bytes_and_restore_report_links(self):
        path = LEDGER_PATH.parent / "case-reporting-tools-handoff-2026-11-28.json"
        ledger = json.loads(path.read_text())
        self.assertEqual(ledger["top_level_directories_before"], ledger["top_level_directories_after"])
        self.assertEqual(ledger["scripts_tracked_before"] - ledger["scripts_tracked_after"], 2)
        self.assertEqual(ledger["scientific_acceptance"], "not_evaluated")
        with (path.parent.parent / "migration_manifest.csv").open(newline="") as stream:
            rows = {r["old_path"]: r["new_path"] for r in csv.DictReader(stream)
                    if r["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(rows, {r["old_path"]: r["new_path"] for r in ledger["entries"]})
        for row in ledger["entries"]:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", row["old_path"]], cwd=REPO))
        for row in ledger["entries"] + ledger["path_only_document_updates"]:
            data = (REPO / row["new_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), row["current_sha256"])
            if row.get("format_only_reverse_splices"):
                formatted = data.decode()
                lines = formatted.splitlines(keepends=True)
                for op in reversed(row["format_only_reverse_splices"]):
                    start = op["after_start"]
                    end = start + len(op["after_lines"])
                    self.assertEqual(lines[start:end], op["after_lines"])
                    lines[start:end] = op["before_lines"]
                unformatted = "".join(lines)
                self.assertEqual(ast.dump(ast.parse(formatted)), ast.dump(ast.parse(unformatted)))
                data = unformatted.encode()
                self.assertEqual(hashlib.sha256(data).hexdigest(), row["pre_format_sha256"])
            for edit in reversed(row["path_only_edits"]):
                self.assertEqual(data.count(edit["new"].encode()), edit["count"])
                data = data.replace(edit["new"].encode(), edit["old"].encode())
            self.assertEqual(hashlib.sha256(data).hexdigest(), row["original_sha256"])
        readme = ledger["append_only_document_update"]
        raw = (REPO / readme["new_path"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw.removesuffix(readme["appended_text"].encode())).hexdigest(),
                         readme["original_sha256"])
        for row in ledger["unchanged_historical_assets"]:
            self.assert_fingerprint(REPO / row["new_path"], row["original_sha256"], row["size_bytes"])
        weak = ledger["compatibility"]["weakly"]
        self.assertEqual(weak["exit_codes"], [0, 0])
        self.assertTrue(weak["identical_original_generated_markdown"])
        report = REPO / "docs/verification/lbm_weakly_compressible.md"
        for fig in weak["regenerated_pngs"]:
            self.assertFalse(fig["original_image_recovered"])
            self.assertEqual(fig["sha256_before"], fig["sha256_after"])
            self.assert_fingerprint(REPO / fig["path"], fig["sha256_after"], fig["size_bytes"])
            self.assertIn(os.path.relpath(REPO / fig["path"], report.parent), report.read_text())
        self.assertEqual(ledger["compatibility"]["dam"]["exit_codes"], [0, 0])

    def test_phasefield_case_relocation_preserves_original_figures_and_provenance(self):
        import base64

        ledger = json.loads(
            (LEDGER_PATH.parent / "phasefield-mcmp-published-case-2026-11-27.json").read_text()
        )
        self.assertEqual(len(ledger["entries"]), 2)
        self.assertEqual(len(ledger["decoded_original_figures"]), 4)
        for entry in ledger["entries"]:
            self.assertFalse((REPO / entry["old_path"]).exists())
            p = REPO / entry["new_path"]
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), entry["current_sha256"])
            self.assertEqual(
                subprocess.check_output(["git", "ls-files", "-s", "--", entry["new_path"]],
                                        cwd=REPO, text=True).split()[0], entry["git_mode"])
        for asset in ledger["retained_original_assets"]:
            self.assertEqual(hashlib.sha256((REPO / asset["path"]).read_bytes()).hexdigest(),
                             asset["sha256"])
        for figure in ledger["decoded_original_figures"]:
            original = json.loads((REPO / figure["origin"]).read_text(encoding="utf-8"))
            p = REPO / figure["path"]
            self.assertEqual(p.read_bytes(),
                             base64.b64decode(original[figure["origin_key"]], validate=True))
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), figure["sha256"])
        for consumer in ledger["updated_consumers"]:
            self.assertEqual(
                hashlib.sha256((REPO / consumer["path"]).read_bytes()).hexdigest(),
                consumer["current_sha256"])
        self.assertFalse(ledger["registry"]["new_case"])
        self.assertEqual(ledger["validation"]["current_scientific_acceptance"], "not_evaluated")

    def test_blasius_development_runner_is_archived_not_a_second_case(self):
        ledger_path = (
            LEDGER_PATH.parent / "blasius-dev-runner-retirement-2026-11-22.json"
        )
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        self.assertEqual(ledger["second_level_directories"]["benchmarks"], {
            "before": 103,
            "after": 102,
        })
        self.assertEqual(
            ledger["canonical_runner"],
            "benchmarks/pending/blasius_flat_plate/run.py",
        )
        self.assertTrue((REPO / ledger["canonical_runner"]).is_file())
        self.assertTrue((REPO / "benchmarks/pending/blasius_flat_plate/README.md").is_file())
        self.assertFalse((REPO / "benchmarks/blasius_flat_plate_dev").exists())

        item = ledger["entries"][0]
        destination = REPO / item["new_path"]
        self.assert_fingerprint(
            destination, item["original_sha256"], item["size_bytes"]
        )
        indexed = subprocess.check_output(
            ["git", "ls-files", "-s", "--", item["new_path"]],
            cwd=REPO,
            text=True,
        ).split()
        self.assertEqual(indexed[:2], [item["git_mode"], item["original_git_blob"]])

        with (REPO / "docs/architecture/migration_manifest.csv").open(
            encoding="utf-8"
        ) as stream:
            manifest = {row["old_path"]: row for row in csv.DictReader(stream)}
        self.assertEqual(manifest[item["old_path"]]["new_path"], item["new_path"])
        policy = json.loads(
            (REPO / "docs/architecture/source_authority_policy.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn(item["old_path"], policy["allowed_active_references"])

    def test_oracle_and_norberg_audit_archives_preserve_case_evidence(self):
        path = LEDGER_PATH.parent / "oracle-and-square-cylinder-audit-docs-2026-11-27.json"
        ledger = json.loads(path.read_text())
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            rows = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                    if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(len(rows), 26)
        self.assertEqual(rows, {row["old_path"]: row["new_path"] for row in ledger["entries"]})
        self.assertEqual(len(ledger["retired_second_level_directories"]), 3)
        for retired in ledger["retired_second_level_directories"]:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", retired], cwd=REPO))
        self.assertEqual(len(ledger["prior_archived_evidence"]), 18)
        for entry in ledger["entries"] + ledger["prior_archived_evidence"]:
            self.assert_fingerprint(REPO / entry["new_path"], entry["original_sha256"], entry["size_bytes"])
            index = subprocess.check_output(
                ["git", "ls-files", "-s", "--", entry["new_path"]], cwd=REPO, text=True,
            ).split()
            self.assertEqual(index[:2], [entry["git_mode"], entry["original_git_blob"]])
        frozen = REPO / "development/audit/m18218/evidence_snapshot"
        for line in (frozen / "SHA256SUMS.txt").read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            self.assertEqual(hashlib.sha256((frozen / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(len((frozen / "SHA256SUMS.txt").read_text().splitlines()), 11)
        compat = ledger["compatibility"]
        manifest = json.loads((REPO / compat["oracle_manifest_path"]).read_text())
        self.assertEqual({k: v for k, v in manifest.items() if k not in ("documentation", "evidence")},
                         compat["oracle_protected_fields"])
        mapping = compat["oracle_documentation_mapping"]
        self.assertEqual(manifest["documentation"], [mapping.get(p, p) for p in compat["oracle_old_docs"]])
        self.assertEqual(manifest["evidence"][:len(compat["oracle_old_evidence"])], compat["oracle_old_evidence"])
        for mission, owner in compat["case_ownership"].items():
            raw = subprocess.check_output(
                [sys.executable, "tools/benchmark_registry.py", "--case", owner["id"]], cwd=REPO, text=True,
            )
            (record,) = json.loads(raw)["cases"]
            self.assertEqual(record["current_acceptance"], "not_evaluated")
            (variant,) = record["variants"]
            self.assertEqual(variant["path"], owner["path"])
            if mission == "m16351":
                self.assertEqual(variant["archive_bucket"], "pending")
                self.assertIn(owner["pointer"], variant["evidence"])
                self.assertEqual(variant["results"], [])
            else:
                self.assertIn(owner["audit_index"], variant["evidence"])
                for entry in ledger["entries"] + ledger["prior_archived_evidence"]:
                    if entry["new_path"].startswith(f"development/audit/{mission}/"):
                        self.assertIn(entry["new_path"], variant["evidence"])
                self.assertEqual(variant["results"], compat["oracle_protected_fields"]["results"])
        for name in compat["navigation_indexes"]:
            index = REPO / name
            for link in re.findall(r"\]\(([^)]+)\)", index.read_text()):
                self.assertTrue((index.parent / link).is_file(), link)
        self.assertFalse(ledger["science"]["full_scientific_replay_performed"])
        self.assertFalse(ledger["science"]["algorithm_changed"])

    def test_historical_campaign_report_script_handoff_preserves_provenance(self):
        path = LEDGER_PATH.parent / "historical-campaign-report-script-handoff-2026-11-28.json"
        ledger = json.loads(path.read_text())
        with (path.parent.parent / "migration_manifest.csv").open(newline="") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                        if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(manifest, {row["old_path"]: row["new_path"] for row in ledger["entries"]})
        self.assertEqual(len(manifest), 7)
        for old in manifest:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", old], cwd=REPO))
        for row in ledger["entries"]:
            destination = REPO / row["new_path"]
            self.assert_fingerprint(destination, row["current_sha256"], row["current_size_bytes"])
            mode, blob = subprocess.check_output(
                ["git", "ls-files", "-s", "--", row["new_path"]], cwd=REPO, text=True,
            ).split()[:2]
            self.assertEqual(mode, row["git_mode"])
            original = destination.read_bytes()
            if row["byte_identical"]:
                self.assertEqual(blob, row["original_git_blob"])
            else:
                edit = row["path_only_edit"]
                self.assertEqual(original.count(edit["new"].encode()), 1)
                original = original.replace(edit["new"].encode(), edit["old"].encode())
            self.assertEqual(hashlib.sha256(original).hexdigest(), row["original_sha256"])
            self.assertEqual(len(original), row["size_bytes"])
            self.assertEqual(hashlib.sha1(f"blob {len(original)}\0".encode() + original).hexdigest(),
                             row["original_git_blob"])
        for name in ledger["compatibility"]["navigation_indexes"]:
            index = REPO / name
            for target in re.findall(r"\]\(([^)]+)\)", index.read_text()):
                self.assertTrue((index.parent / target).is_file(), target)
        registry = json.loads(subprocess.check_output(
            [sys.executable, "tools/benchmark_registry.py", "--json"], cwd=REPO,
        ))["cases"]
        for mission, owner in ledger["compatibility"]["related_existing_cases"].items():
            record, variant = next((row, v) for row in registry for v in row["variants"]
                                   if v["path"] == owner)
            self.assertEqual(record["current_acceptance"], "not_evaluated")
            self.assertIn(f"{owner}/evidence/{mission}/README.md", variant["evidence"])
        self.assertEqual(len(ledger["compatibility"]["cli_help_parity"]), 5)
        self.assertTrue(all(row["before_after_byte_identical"]
                            for row in ledger["compatibility"]["cli_help_parity"]))
        self.assertFalse(ledger["science"]["full_scientific_replay_performed"])
        self.assertEqual(ledger["science"]["current_acceptance"], "not_evaluated")

    def test_hotpatch_records_keep_provenance_and_manual_navigation(self):
        ledger_path = LEDGER_PATH.parent / "hotpatch-mission-records-2026-11-22.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        with (REPO / "docs/architecture/migration_manifest.csv").open(
            encoding="utf-8"
        ) as stream:
            entries = {
                row["old_path"]: row["new_path"]
                for row in csv.DictReader(stream)
                if row["verification"] == ledger_path.relative_to(REPO).as_posix()
            }
        self.assertEqual(len(ledger["entries"]), 2)
        self.assertEqual(
            entries,
            {item["old_path"]: item["new_path"] for item in ledger["entries"]},
        )
        for item in ledger["entries"]:
            with self.subTest(path=item["old_path"]):
                self.assert_fingerprint(
                    REPO / item["new_path"], item["original_sha256"], item["size_bytes"]
                )
                index = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", item["new_path"]],
                    cwd=REPO,
                    text=True,
                ).split()
                self.assertEqual(index[:2], [item["git_mode"], item["original_git_blob"]])
                self.assertFalse((REPO / item["old_path"]).exists())

        docs = REPO / "docs"
        catalogue = "benchmarks/tutorials/README.md"
        self.assertIn(f"]({catalogue})", (docs / "README.md").read_text(encoding="utf-8"))
        self.assertTrue((docs / catalogue).is_file())

    def test_stokes_blockforest_ce_audit_migration_keeps_original_assets(self):
        path = LEDGER_PATH.parent / "stokes-blockforest-ce-audits-2026-11-27.json"
        ledger = json.loads(path.read_text())
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            rows = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                    if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(len(rows), 25)
        self.assertEqual(rows, {row["old_path"]: row["new_path"] for row in ledger["entries"]})
        self.assertEqual(len(ledger["retired_second_level_directories"]), 3)
        for retired in ledger["retired_second_level_directories"]:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", retired], cwd=REPO))
        self.assertEqual(len(ledger["prior_archived_evidence"]), 2)
        for entry in ledger["entries"] + ledger["prior_archived_evidence"]:
            self.assert_fingerprint(REPO / entry["new_path"], entry["original_sha256"], entry["size_bytes"])
            index = subprocess.check_output(
                ["git", "ls-files", "-s", "--", entry["new_path"]], cwd=REPO, text=True,
            ).split()
            self.assertEqual(index[:2], [entry["git_mode"], entry["original_git_blob"]])
        self.assertFalse(ledger["science"]["full_scientific_replay_performed"])
        self.assertFalse(ledger["science"]["algorithm_changed"])
        owners = ledger["compatibility"]["case_ownership"]
        for mission, owner in owners.items():
            raw = subprocess.check_output(
                [sys.executable, "tools/benchmark_registry.py", "--case", owner["id"]], cwd=REPO, text=True,
            )
            (record,) = json.loads(raw)["cases"]
            self.assertEqual(record["current_acceptance"], "not_evaluated")
            (variant,) = record["variants"]
            self.assertEqual(variant["path"], owner["path"])
            if mission == "m16256":
                self.assertIn(owner["pointer"], variant["evidence"])
                self.assertEqual(variant["results"], [owner["path"] + "/result.json"])
                self.assertEqual(variant["archive_bucket"], "verified")
            else:
                self.assertIn(owner["audit_index"], variant["documentation"])
                for entry in ledger["entries"]:
                    if entry["new_path"].startswith(f"development/audit/{mission}/"):
                        self.assertIn(entry["new_path"], variant["evidence"])
                self.assertIsNone(variant["archive_bucket"])
        for name in ledger["compatibility"]["navigation_indexes"]:
            index = REPO / name
            for link in re.findall(r"\]\(([^)]+)\)", index.read_text()):
                self.assertTrue((index.parent / link).is_file(), link)
        for published in ledger["compatibility"]["published_paths"]:
            self.assertTrue((REPO / published).is_file(), published)

    def test_dfg_cartesian_mpi_archive_preserves_originals_and_pending_case(self):
        path = LEDGER_PATH.parent / "dfg-and-cartesian-mpi-audits-2026-11-27.json"
        ledger = json.loads(path.read_text())
        with (path.parent.parent / "migration_manifest.csv").open(newline="") as stream:
            rows = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                    if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(rows, {r["old_path"]: r["new_path"] for r in ledger["entries"]})
        self.assertEqual(len(rows), 13)
        self.assertEqual(len(set(rows.values())), 13)
        for retired in ledger["retired_second_level_directories"]:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", retired], cwd=REPO))
        successor = json.loads((LEDGER_PATH.parent /
                                "m28104-dfg-concurrent-path-corrections-2026-11-28.json").read_text())
        corrections = {row["new_path"]: row for row in successor["concurrent_path_corrections"]}
        self.assertEqual(len(corrections), 2)
        self.assertTrue(set(corrections) <= {r["new_path"] for r in ledger["unchanged_prior_archive"]})
        for row in ledger["entries"] + ledger["consumer_edits"] + ledger["unchanged_prior_archive"]:
            destination = REPO / row["new_path"]
            correction = corrections.get(row["new_path"])
            current = correction or row
            self.assert_fingerprint(destination, current["current_sha256"], current["current_size_bytes"])
            mode, blob = subprocess.check_output(
                ["git", "ls-files", "-s", "--", row["new_path"]], cwd=REPO, text=True,
            ).split()[:2]
            self.assertEqual(mode, row["git_mode"])
            original = destination.read_bytes()
            if correction:
                self.assertTrue(row["byte_identical"])
                self.assertEqual(blob, correction["current_git_blob"])
                self.assertEqual(correction["original_git_blob"], row["original_git_blob"])
                self.assertEqual(correction["original_sha256"], row["current_sha256"])
                self.assertEqual(correction["size_bytes"], row["current_size_bytes"])
                for edit in reversed(correction["path_only_edits"]):
                    old, new = edit["old"].encode(), edit["new"].encode()
                    self.assertEqual(original.count(new), edit["count"])
                    original = original.replace(new, old)
            elif row["byte_identical"]:
                self.assertEqual(blob, row["original_git_blob"])
            else:
                replacement = row["path_only_edit"]
                self.assertEqual(original.count(replacement["new"].encode()), 1)
                original = original.replace(replacement["new"].encode(), replacement["old"].encode())
            self.assertEqual(hashlib.sha256(original).hexdigest(), row["original_sha256"])
            self.assertEqual(len(original), row["size_bytes"])
            self.assertEqual(hashlib.sha1(f"blob {len(original)}\0".encode() + original).hexdigest(),
                             row["original_git_blob"])
        for name in ledger["compatibility"]["navigation_indexes"]:
            index = REPO / name
            for target in re.findall(r"\]\(([^)]+)\)", index.read_text()):
                self.assertTrue((index.parent / target).is_file(), target)
        registry = json.loads(subprocess.check_output(
            [sys.executable, "tools/benchmark_registry.py", "--json"], cwd=REPO,
        ))["cases"]
        case = next(entry for entry in registry if entry["id"] == "benchmarks/pending/cylinder_re100_dfg")
        self.assertEqual(case["current_acceptance"], "not_evaluated")
        self.assertEqual(case["variants"][0]["archive_bucket"], "pending")
        self.assertIn("development/audit/m23014/REPORT.md", case["variants"][0]["documentation"])
        self.assertIn("development/audit/m23014/analysis.json", case["variants"][0]["results"])
        self.assertIn("benchmarks/verified/cylinder/re100_dfg/artifacts/level_D40_u0833.series.npz",
                      case["variants"][0]["evidence"])
        self.assertTrue(any(entry["id"] == "matrix:cylinder_re100_dfg" for entry in registry))
        self.assertEqual(ledger["science"]["current_acceptance"], "not_evaluated")

    def test_verification_report_helper_handoff_preserves_originals_and_links(self):
        path = LEDGER_PATH.parent / "verification-report-helper-handoff-2026-11-28.json"
        ledger = json.loads(path.read_text())
        with (path.parent.parent / "migration_manifest.csv").open(newline="") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                        if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(manifest, {row["old_path"]: row["new_path"] for row in ledger["entries"]})
        self.assertEqual(len(manifest), 6)
        for old in manifest:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", old], cwd=REPO))
        for row in ledger["entries"] + ledger["consumer_edits"]:
            destination = REPO / row["new_path"]
            self.assert_fingerprint(destination, row["current_sha256"], row["current_size_bytes"])
            mode, blob = subprocess.check_output(
                ["git", "ls-files", "-s", "--", row["new_path"]], cwd=REPO, text=True,
            ).split()[:2]
            self.assertEqual(mode, row["git_mode"])
            original = destination.read_bytes()
            if row["byte_identical"]:
                self.assertEqual(blob, row["original_git_blob"])
            else:
                edit = row["path_only_edit"]
                self.assertEqual(original.count(edit["new"].encode()), 1)
                original = original.replace(edit["new"].encode(), edit["old"].encode())
            self.assertEqual(hashlib.sha256(original).hexdigest(), row["original_sha256"])
            self.assertEqual(len(original), row["size_bytes"])
            self.assertEqual(hashlib.sha1(f"blob {len(original)}\0".encode() + original).hexdigest(),
                             row["original_git_blob"])
        for name in ledger["compatibility"]["navigation_indexes"]:
            index = REPO / name
            for target in re.findall(r"\]\(([^)]+)\)", index.read_text()):
                self.assertTrue((index.parent / target).is_file(), target)
        for row in ledger["consumer_edits"]:
            report = REPO / row["new_path"]
            for target in re.findall(r"\]\(([^)]+)\)", row["path_only_edit"]["new"]):
                self.assertTrue((report.parent / target).is_file(), target)
        self.assertEqual(len(ledger["compatibility"]["cli_help_parity"]), 5)
        self.assertTrue(all(row["before_after_byte_identical"]
                            for row in ledger["compatibility"]["cli_help_parity"]))
        self.assertEqual(ledger["compatibility"]["new_case_ids"], 0)
        self.assertEqual(ledger["science"]["current_acceptance"], "not_evaluated")
        self.assertFalse(ledger["science"]["full_scientific_replay_performed"])

    def test_packaged_reference_and_user_demo_placement(self):
        path = LEDGER_PATH.parent / "packaged-reference-and-user-demo-2026-11-28.json"
        ledger = json.loads(path.read_text())
        with (path.parent.parent / "migration_manifest.csv").open(newline="") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                        if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(manifest, {row["old_path"]: row["new_path"] for row in ledger["entries"]})
        self.assertEqual(len(manifest), 2)
        for row in ledger["entries"]:
            self.assertFalse((REPO / row["old_path"]).exists())
            self.assertTrue((REPO / row["new_path"]).is_file())
        self.assertIn("python examples/make_demo_gif.py", (REPO / "README.md").read_text())
        source = (REPO / "src/tensorlbm/verification/dam_break_2d_validation.py").read_text()
        self.assertIn("from ._digitize_martin_moyce import main", source)
        self.assertNotIn("from scripts.digitize_martin_moyce", source)
        for row in ledger["unchanged_related_assets_at_migration"]:
            self.assert_fingerprint(REPO / row["new_path"], row["original_sha256"], row["size_bytes"])
        receipt = ledger["compatibility"]["actual_receipt"]
        self.assertEqual(receipt["isolated_wheel_fallback"], "success")
        self.assertTrue(receipt["wheel_contains_helper"])
        self.assertEqual(len(receipt["outputs"]), 2)
        self.assertTrue(all(row["before_after_and_tracked_byte_identical"]
                            for row in receipt["outputs"]))
        self.assertTrue(receipt["demo_help_byte_identical"])
        self.assertFalse(receipt["solver_invoked"])
        self.assertEqual(ledger["science"]["current_acceptance"], "not_evaluated")
        self.assertFalse(ledger["science"]["algorithm_changed"])

    def test_hidden_audit_root_preserves_replay_files(self):
        ledger_path = LEDGER_PATH.parent / "hidden-audit-root-2026-11-21.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            manifest = {
                row["old_path"]: row["new_path"]
                for row in csv.DictReader(stream)
                if row["verification"] == ledger_path.relative_to(REPO).as_posix()
            }
        self.assertEqual(manifest, {item["old_path"]: item["new_path"] for item in ledger["entries"]})
        for item in ledger["entries"]:
            with self.subTest(path=item["new_path"]):
                self.assert_fingerprint(REPO / item["new_path"], item["sha256"], item["size_bytes"])
                self.assertFalse((REPO / item["old_path"]).exists())
        audit_links = {
            "docs/board_audit/m23219/AUDIT-m23126.md":
                "../../../development/audit/m23219/repro/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD/",
            "development/audit/m15893/M15893_BOARD_AUDIT.md":
                "./repro/",
        }
        for name, target in audit_links.items():
            with self.subTest(audit=name):
                doc = REPO / name
                self.assertIn(f"]({target})", doc.read_text(encoding="utf-8"))
                self.assertTrue((doc.parent / target).is_dir())


    def test_bitwise_and_capillary_audit_archives_keep_report_provenance(self):
        path = LEDGER_PATH.parent / "bitwise-capillary-audit-navigation-2026-11-28.json"
        ledger = json.loads(path.read_text())
        with (path.parent.parent / "migration_manifest.csv").open(newline="") as stream:
            manifest = {
                row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                if row["verification"] == path.relative_to(REPO).as_posix()
            }
        self.assertEqual(manifest, {row["old_path"]: row["new_path"] for row in ledger["entries"]})
        for directory in ledger["retired_second_level_directories"]:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", directory], cwd=REPO))
        rows = ledger["entries"] + ledger["consumer_edits"]
        for row in rows:
            with self.subTest(path=row["new_path"]):
                destination = REPO / row["new_path"]
                self.assert_fingerprint(destination, row["current_sha256"], row["current_size_bytes"])
                original = destination.read_bytes()
                for edit in reversed(row["path_only_edits"]):
                    old, new = edit["old"].encode(), edit["new"].encode()
                    self.assertEqual(original.count(new), edit["count"])
                    original = original.replace(new, old)
                self.assertEqual(hashlib.sha256(original).hexdigest(), row["original_sha256"])
                self.assertEqual(len(original), row["size_bytes"])
                blob = hashlib.sha1(f"blob {len(original)}\0".encode() + original).hexdigest()
                self.assertEqual(blob, row["original_git_blob"])
        for row in ledger["unchanged_related_assets_at_migration"]:
            self.assert_fingerprint(REPO / row["new_path"], row["original_sha256"], row["size_bytes"])
        for row in ledger["entries"]:
            self.assertFalse((REPO / row["old_path"]).exists())
        for row in ledger["consumer_edits"]:
            for edit in row["path_only_edits"]:
                for target in re.findall(r"\]\(([^)]+)\)", edit["new"]):
                    if target.startswith("https://github.com/cxs503/TensorLBM/blob/"):
                        self.assertIn("/blob/" + ledger["base_main"] + "/", target)
                    else:
                        self.assertTrue((REPO / row["new_path"]).parent.joinpath(target).exists(), target)
        # Execute the migrated report renderer with real tracked inputs in an
        # isolated directory; never overwrite either archived result/report.
        import shutil
        import tempfile

        directory = Path("benchmarks/xval_bitwise_refdata")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in ("make_report.py", "result.json", "probe_weight_bias.json",
                             "refdata/d2q9_bitwise_v1.json"):
                destination = root / directory / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / directory / relative, destination)
            subprocess.check_output([sys.executable, str(root / directory / "make_report.py")],
                                    cwd=root)
            generated = root / ledger["compatibility"]["replay_output_directory"] / "REPORT.md"
            receipt = ledger["compatibility"]["replay_parity"]
            self.assert_fingerprint(generated, receipt["sha256"], receipt["bytes"])
            self.assertNotEqual(receipt["sha256"], receipt["original_committed_report_sha256"])
            self.assertFalse((root / ledger["entries"][0]["old_path"]).exists())
        self.assertFalse(ledger["compatibility"]["solver_invoked"])
        self.assertEqual(ledger["science"]["current_acceptance"], "not_evaluated")
        self.assertFalse(ledger["science"]["algorithm_changed"])

    def test_m28104_concurrent_archive_keeps_original_evidence_and_case_status(self):
        path = LEDGER_PATH.parent / "m28104-dfg-concurrent-path-corrections-2026-11-28.json"
        ledger = json.loads(path.read_text())
        with (path.parent.parent / "migration_manifest.csv").open(newline="") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)
                        if row["verification"] == path.relative_to(REPO).as_posix()}
        self.assertEqual(manifest, {row["old_path"]: row["new_path"] for row in ledger["entries"]})
        self.assertEqual(len(manifest), 7)
        for directory in ledger["retired_second_level_directories"]:
            self.assertFalse(subprocess.check_output(["git", "ls-files", "--", directory], cwd=REPO))
        for row in ledger["entries"]:
            destination = REPO / row["new_path"]
            self.assert_fingerprint(destination, row["current_sha256"], row["current_size_bytes"])
            original = destination.read_bytes()
            for edit in reversed(row.get("path_only_edits", [])):
                old, new = edit["old"].encode(), edit["new"].encode()
                self.assertEqual(original.count(new), edit["count"])
                original = original.replace(new, old)
            self.assertEqual(hashlib.sha256(original).hexdigest(), row["original_sha256"])
            self.assertEqual(len(original), row["size_bytes"])
            self.assertEqual(hashlib.sha1(f"blob {len(original)}\0".encode() + original).hexdigest(),
                             row["original_git_blob"])
            self.assertFalse((REPO / row["old_path"]).exists())
        for row in ledger["unchanged_related_assets_at_migration"]:
            self.assert_fingerprint(REPO / row["new_path"], row["original_sha256"], row["size_bytes"])
        for name in ledger["compatibility"]["navigation_indexes"]:
            index = REPO / name
            for target in re.findall(r"\]\(([^)]+)\)", index.read_text()):
                if target.startswith("https://github.com/cxs503/TensorLBM/blob/"):
                    self.assertTrue(target.startswith(
                        "https://github.com/cxs503/TensorLBM/blob/" + ledger["base_main"] + "/"))
                else:
                    self.assertTrue((index.parent / target).is_file(), target)
        records = json.loads(subprocess.check_output(
            [sys.executable, "tools/benchmark_registry.py", "--json"], cwd=REPO,
        ))["cases"]
        case = next(row for row in records if row["id"] == "benchmarks/pending/cylinder_re100_dfg")
        self.assertEqual(case["current_acceptance"], "not_evaluated")
        variant = case["variants"][0]
        self.assertEqual(variant["archive_bucket"], "pending")
        self.assertIn("development/audit/m28104/REPORT.md", variant["documentation"])
        self.assertIn("development/audit/m28104/cl_cd_table.json", variant["results"])
        self.assertIn("development/audit/m28104/m28104_dfg_cl_amp.png", variant["evidence"])
        parity = ledger["compatibility"]["replay_parity"]
        self.assertEqual(len(parity["outputs"]), 2)
        self.assertTrue(all(row["before_after_byte_identical"] for row in parity["outputs"]))
        self.assertEqual(parity["regression_tests_before"], 5)
        self.assertEqual(parity["regression_tests_after"], 5)
        self.assertFalse(parity["solver_invoked"])
        self.assertEqual(ledger["science"]["current_acceptance"], "not_evaluated")
        self.assertFalse(ledger["science"]["algorithm_changed"])
        self.assertFalse(ledger["science"]["full_scientific_replay_performed"])

    def test_cac_gravity_mission_archive_keeps_original_objects(self):
        ledger_path = LEDGER_PATH.parent / "cac-gravity-mission-archives-2026-11-27.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        self.assertEqual(len(ledger["entries"]), 5)
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            manifest = {
                row["old_path"]: row["new_path"]
                for row in csv.DictReader(stream)
                if row["verification"] == ledger_path.relative_to(REPO).as_posix()
            }
        self.assertEqual(manifest, {item["old_path"]: item["new_path"] for item in ledger["entries"]})
        for directory in ledger["retired_second_level_directories"]:
            self.assertFalse((REPO / directory).exists())
        for item in ledger["entries"] + ledger["historical_source_snapshots"]:
            with self.subTest(path=item["new_path"]):
                self.assert_fingerprint(
                    REPO / item["new_path"], item["original_sha256"], item["size_bytes"],
                )
                index = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", item["new_path"]], cwd=REPO, text=True,
                ).split()
                self.assertEqual(index[:2], [item["git_mode"], item["original_git_blob"]])
        self.assertFalse(ledger["science"]["m23012_archived_verdict"]["overall_pass"])
        self.assertFalse(ledger["science"]["rerun_performed"])
        self.assertEqual(ledger["science"]["current_acceptance"], "not_evaluated")

    def test_experiment_archive_tree_and_navigation_survive_root_retirement(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "experiment-root-retirement-2026-11-21.json").read_text()
        )
        archive = ledger["archive"]
        tree = subprocess.check_output(
            ["git", "write-tree", f"--prefix={archive['new_path']}/"], cwd=REPO, text=True,
        ).strip()
        self.assertEqual(tree, archive["git_tree"])
        for move in ledger["relocated_files"]:
            if "relocated_sha256" not in move and Path(move["new_path"]).suffix != ".py":
                with self.subTest(path=move["new_path"]):
                    self.assertEqual(
                        hashlib.sha256((REPO / move["new_path"]).read_bytes()).hexdigest(),
                        move["original_sha256"],
                    )
        tracked = subprocess.check_output(
            ["git", "ls-files", "--", ledger["root_removed"]], cwd=REPO, text=True,
        )
        self.assertEqual(tracked, "")
        for index in ("docs/README.md", "docs/architecture/repository_layout.md"):
            doc = REPO / index
            links = re.findall(r"\]\(([^)]+/development/experiments/[^)]*)\)", doc.read_text())
            self.assertTrue(links, index)
            for link in links:
                self.assertTrue((doc.parent / link).is_dir(), link)


if __name__ == "__main__":
    unittest.main()
