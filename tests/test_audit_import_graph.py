import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "audit_import_graph.py"
SPEC = importlib.util.spec_from_file_location("audit_import_graph", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(AUDIT)


class ImportGraphAuditTests(unittest.TestCase):
    def test_relative_edges_and_cycle_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sample"
            root.mkdir()
            (root / "__init__.py").write_text("", encoding="utf-8")
            (root / "alpha.py").write_text("from . import beta\n", encoding="utf-8")
            (root / "beta.py").write_text("from . import alpha\n", encoding="utf-8")
            data = AUDIT.scan([(root, "sample")])
            self.assertEqual(data["module_count"], 3)
            self.assertIn({"source": "sample.alpha", "target": "sample.beta"}, data["edges"])
            self.assertIn({"source": "sample.beta", "target": "sample.alpha"}, data["edges"])
            self.assertIn(["sample.alpha", "sample.beta"], data["cycles"])

    def test_dynamic_unknown_target_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pkg"
            root.mkdir()
            (root / "__init__.py").write_text("", encoding="utf-8")
            (root / "loader.py").write_text(
                "import importlib\nname = get_module_name()\nimportlib.import_module(name)\n",
                encoding="utf-8",
            )
            data = AUDIT.scan([(root, "pkg")])
            self.assertEqual(len(data["dynamic_imports"]), 1)
            self.assertEqual(data["dynamic_imports"][0]["status"], "dynamic-target-unknown")

    def test_missing_internal_targets_are_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pkg"
            root.mkdir()
            (root / "__init__.py").write_text("", encoding="utf-8")
            (root / "user.py").write_text(
                "from pkg.missing import value\nimport external.missing\n"
                "from .missing_relative import value\n",
                encoding="utf-8",
            )
            data = AUDIT.scan([(root, "pkg")])
            self.assertEqual(
                {item["target"] for item in data["unresolved_internal_imports"]},
                {"pkg.missing", "pkg.missing_relative"},
            )
            self.assertIn("Unresolved internal imports: **2**", AUDIT.markdown_report(data))
            # A physical alias is visible as an edge, not a missing target.
            (root / "missing.py").write_text("from .user import value\n", encoding="utf-8")
            data = AUDIT.scan([(root, "pkg")])
            self.assertEqual(len(data["unresolved_internal_imports"]), 1)
            self.assertIn({"source": "pkg.user", "target": "pkg.missing"}, data["edges"])

    def test_strict_cli_reports_before_failing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pkg"
            root.mkdir()
            (root / "broken.py").write_text("import pkg.missing\n", encoding="utf-8")
            output = Path(tmp) / "audit.json"
            command = [
                sys.executable,
                str(SCRIPT),
                "--repo",
                tmp,
                "--root",
                "pkg=pkg",
                "--json-out",
                str(output),
            ]
            advisory = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(advisory.returncode, 0, advisory.stderr)
            strict = subprocess.run(
                command + ["--fail-on-unresolved"], capture_output=True, text=True
            )
            self.assertEqual(strict.returncode, 1, strict.stderr)
            self.assertTrue(output.exists())
            self.assertIn("pkg.missing", strict.stdout)

    def test_syntax_error_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pkg"
            root.mkdir()
            (root / "broken.py").write_text("def broken(:\n", encoding="utf-8")
            data = AUDIT.scan([(root, "pkg")])
            self.assertEqual(len(data["parse_errors"]), 1)


if __name__ == "__main__":
    unittest.main()
