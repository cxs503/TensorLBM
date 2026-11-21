"""Exercise the public registry command against tracked repository assets."""
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

    def test_alias_retains_all_variants(self):
        path = "benchmarks/verified/poiseuille_3d_duct/duct_ar05"
        (record,) = self.registry("--case", path)
        self.assertEqual(record["id"], "poiseuille_3d_duct")
        self.assertEqual({variant["path"] for variant in record["variants"]},
                         {path, path.rsplit("/", 1)[0]})


if __name__ == "__main__":
    unittest.main()
