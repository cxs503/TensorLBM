"""Exercise the public registry command against tracked repository assets."""
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkRegistryTests(unittest.TestCase):
    def registry(self, *args):
        result = subprocess.check_output(
            [sys.executable, "tools/benchmark_registry.py", *args], cwd=ROOT,
        )
        return json.loads(result)["cases"]

    def test_published_manuals_and_archive_verdicts(self):
        records = self.registry("--json")
        self.assertEqual(len({row["id"] for row in records}), len(records))
        self.assertEqual(sum(bool(row["tutorial"]) for row in records), 35)
        self.assertEqual(sum("pdf" in row["tutorial"] for row in records), 33)
        self.assertTrue(all(row["current_acceptance"] == "not_evaluated" for row in records))

    def test_find_archived_assets_by_path(self):
        result = subprocess.run(
            [sys.executable, "tools/benchmark_registry.py", "--find-asset", "M17444_AKAI2018_REPRO"],
            cwd=ROOT, capture_output=True, check=True,
        )
        paths = json.loads(result.stdout)["paths"]
        self.assertIn("audits/m17444_akai2018_repro/benchmarks/xval_akai2018_wetting/"
                      "analytic_verification.json", paths)
        self.assertEqual(paths, sorted(set(paths)))
        self.assertTrue(all((ROOT / path).is_file() for path in paths))

    def test_matrix_lookup_uses_exact_artifacts_and_preserves_verdict_scope(self):
        (ship,) = self.registry("--case", "matrix:ship_resistance_suboff")
        self.assertEqual(ship["id"], "suboff_re1000")
        entry, = ship["validation_matrix"]
        self.assertEqual(entry["definition"]["measure"]["artifact"], entry["archived_comparison"]["artifact"])
        self.assertTrue(entry["references"])
        self.assertEqual(ship["current_acceptance"], "not_evaluated")

        records = self.registry("--json")
        entries = [e for row in records for e in row["validation_matrix"]]
        matrix = json.loads((ROOT / "benchmarks/validation_matrix/matrix.json").read_text())
        self.assertCountEqual([e["id"] for e in entries], [c["id"] for c in matrix["cases"]])
        report = json.loads((ROOT / "benchmarks/validation_matrix/BENCHMARK_REF_VS_GOT.json").read_text())
        reported = {row["case"]: row["status"] for row in report["cases"]}
        for entry in entries:
            comparison = entry["archived_comparison"]
            self.assertEqual(None if comparison is None else comparison["status"], reported.get(entry["id"]))

    def test_xval_registration_preserves_archived_and_replayed_results(self):
        (case,) = self.registry("--case", "xval:ludwig_fe_stress")
        self.assertEqual(case["current_acceptance"], "not_evaluated")
        variant, = case["variants"]
        self.assertEqual(variant["path"], "xval/ludwig_fe_stress")
        self.assertEqual(case["cross_validation"]["criteria"]["threshold"], 0.15)
        original, replay, ledger = (ROOT / path for path in variant["results"])
        evidence = json.loads(ledger.read_text())
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),
                         evidence["compatibility"]["archive_sha256"])
        self.assertEqual(hashlib.sha256(replay.read_bytes()).hexdigest(),
                         evidence["run"]["result_sha256"])
        summary = json.loads(replay.read_text())["summary"]
        self.assertTrue(summary["pass_15pct"])
        self.assertGreater(summary["worst_force_diff_rel"], 1.0)

    def test_alias_retains_all_variants(self):
        path = "benchmarks/verified/poiseuille_3d_duct/duct_ar05"
        (record,) = self.registry("--case", path)
        self.assertEqual(record["id"], "poiseuille_3d_duct")
        self.assertEqual({variant["path"] for variant in record["variants"]},
                         {path, path.rsplit("/", 1)[0]})


if __name__ == "__main__":
    unittest.main()
