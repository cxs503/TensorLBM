"""Regression tests for the published 33-case catalogue integrity gate."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools/check_benchmark_manuals.py"
SPEC = importlib.util.spec_from_file_location("manual_integrity", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


class CatalogueTests(unittest.TestCase):
    def test_missing_index_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = module.check(Path(tmp))
            self.assertTrue(result["errors"])

    def test_duplicate_and_missing_assets_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "docs/benchmarks/tutorials"
            directory.mkdir(parents=True)
            row = "| [Demo（demo_case）](tutorials/demo_case.md) | Flow | [md](tutorials/demo_case.md) | — | — | `benchmarks/verified/demo_case` |\n"
            (directory / "README.md").write_text(row * 2, encoding="utf-8")
            result = module.check(root)
            self.assertEqual(result["count"], 2)
            self.assertTrue(any("duplicate case ID" in e for e in result["errors"]))
            self.assertTrue(any("missing manual" in e for e in result["errors"]))
            self.assertTrue(any("missing benchmark target" in e for e in result["errors"]))

    def test_published_catalogue_is_unchanged_and_complete(self):
        result = module.check()
        self.assertEqual(result["count"], 33)
        self.assertEqual(result["errors"], [], "\n".join(result["errors"]))


if __name__ == "__main__":
    unittest.main()
