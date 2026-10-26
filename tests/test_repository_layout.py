"""Dependency-free tests for the repository hygiene gate."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "check_repository_layout.py"
SPEC = importlib.util.spec_from_file_location("repository_layout", SCRIPT)
assert SPEC and SPEC.loader
LAYOUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAYOUT)


def policy() -> dict:
    return {
        "schema_version": 1,
        "baseline_revision": "test-fixture",
        "root_files": [".gitignore", "README.md"],
        "legacy_root_files": ["old_debug.py"],
        "legacy_generated_files": ["runs/old.log"],
        "legacy_model_files": ["models/old.pt"],
        "large_files": {},
    }


class RepositoryLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)

    def write(self, name: str, text: str = "example") -> Path:
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def check(self, names, config=None):
        return LAYOUT.check_files(self.repo, names, policy() if config is None else config)

    def git(self, *args):
        return subprocess.run(
            ["git", *args], cwd=self.repo, check=True, capture_output=True, text=True
        )

    def test_canonical_and_exact_legacy_root_paths_pass(self):
        for name in ["README.md", "old_debug.py", "experiments/new.py"]:
            self.write(name)
        self.assertEqual(self.check(["README.md", "old_debug.py", "experiments/new.py"]), [])

    def test_new_root_path_fails(self):
        self.write("new_debug.py")
        self.assertIn("root-file:", self.check(["new_debug.py"])[0])

    def test_generated_exception_is_not_a_directory_exemption(self):
        self.write("runs/old.log")
        self.write("runs/new.log")
        self.assertEqual(self.check(["runs/old.log"]), [])
        self.assertIn("generated-file:", self.check(["runs/new.log"])[0])

    def test_generated_directory_boundaries(self):
        for name in [
            "outputs/a.json",
            "artifacts/runs/a.log",
            "frontend-vue/dist/index.html",
            "build/a",
            "dist/a.whl",
            "pkg/__pycache__/a.pyc",
            ".venv/bin/python",
            "frontend-vue/node_modules/a.js",
        ]:
            with self.subTest(name=name):
                self.assertTrue(LAYOUT.is_generated(name))
        for name in [
            "tools/runs.py",
            "docs/runs.md",
            "runs_extra/source.py",
            "tests/fixtures/result.json",
            "benchmarks/verified/case/report.json",
        ]:
            with self.subTest(name=name):
                self.assertFalse(LAYOUT.is_generated(name))

    def test_model_exceptions_are_exact(self):
        for name in ["models/old.pt", "models/new.pt", "tests/fixtures/weights.ONNX"]:
            self.write(name)
        self.assertEqual(self.check(["models/old.pt"]), [])
        for name in ["models/new.pt", "tests/fixtures/weights.ONNX"]:
            self.assertIn("model-file:", self.check([name])[0])

    def test_small_evidence_and_reference_data_are_allowed(self):
        names = [
            "evidence/case/result.json",
            "benchmarks/verified/case/reference.csv",
            "docs/assets/case/figure.png",
            "tests/fixtures/data.npz",
        ]
        for name in names:
            self.write(name)
        self.assertEqual(self.check(names), [])

    def test_file_size_limit_and_exact_legacy_ceiling(self):
        path = self.write("docs/assets/large.json")
        with path.open("wb") as stream:
            stream.truncate(LAYOUT.MAX_FILE_BYTES)
        self.assertEqual(self.check(["docs/assets/large.json"]), [])
        with path.open("ab") as stream:
            stream.write(b"x")
        self.assertIn("large-file:", self.check(["docs/assets/large.json"])[0])
        config = policy()
        config["large_files"]["docs/assets/large.json"] = LAYOUT.MAX_FILE_BYTES + 1
        self.assertEqual(self.check(["docs/assets/large.json"], config), [])
        with path.open("ab") as stream:
            stream.write(b"x")
        self.assertIn("large-file:", self.check(["docs/assets/large.json"], config)[0])

    def test_deleted_legacy_file_does_not_require_a_placeholder(self):
        self.assertEqual(self.check(["old_debug.py", "runs/old.log"]), [])

    def test_symlink_does_not_follow_external_target(self):
        path = self.repo / "docs" / "missing-link"
        path.parent.mkdir()
        path.symlink_to(self.repo / "nonexistent-target")
        self.assertEqual(self.check(["docs/missing-link"]), [])

    def test_policy_validation(self):
        config_path = self.write("policy.json", json.dumps(policy()))
        self.assertEqual(LAYOUT.load_policy(config_path), policy())
        invalid = []
        for name in ["../escape", "/absolute", "docs/../root", "./root", ""]:
            config = policy()
            config["legacy_generated_files"] = [name]
            invalid.append(config)
        config = policy()
        config["root_files"] = ["README.md", "README.md"]
        invalid.append(config)
        config = policy()
        config["large_files"] = {"docs/big.json": -1}
        invalid.append(config)
        invalid.extend([{}, {**policy(), "schema_version": 99}, []])
        for config in invalid:
            with self.subTest(config=config):
                config_path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    LAYOUT.load_policy(config_path)

    def test_git_candidates_include_forced_ignored_files_but_not_local_outputs(self):
        self.git("init", "-q")
        self.write(".gitignore", "runs/\n")
        self.write("runs/local.log")
        self.write("runs/forced.log")
        self.write("experiments/带 空格.py")
        self.git("add", "-f", "runs/forced.log")
        names = LAYOUT.candidate_files(self.repo)
        self.assertIn("experiments/带 空格.py", names)
        self.assertIn("runs/forced.log", names)
        self.assertNotIn("runs/local.log", names)
        self.assertIn("generated-file:", self.check(names)[0])

    def test_cli_pass_failure_and_policy_error_exit_codes(self):
        self.git("init", "-q")
        config_path = self.write(LAYOUT.POLICY_PATH, json.dumps(policy()))

        def run():
            import sys

            return subprocess.run(
                [sys.executable, str(SCRIPT), "--repo", str(self.repo)],
                capture_output=True,
                text=True,
            )

        self.assertEqual(run().returncode, 0)
        self.write("new_debug.py")
        result = run()
        self.assertEqual(result.returncode, 1)
        self.assertIn("root-file", result.stdout)
        config_path.write_text("not JSON")
        self.assertEqual(run().returncode, 2)


if __name__ == "__main__":
    unittest.main()
