"""Tests for the TensorLBM examples framework."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorlbm.examples import registry
from tensorlbm.examples.base import Example, ExampleResult


class TestRegistry:
    def test_list_examples(self):
        names = registry.list_examples()
        assert isinstance(names, list)
        assert "lid_driven_cavity" in names

    def test_get_example(self):
        ex = registry.get("lid_driven_cavity")
        assert ex is not None
        assert ex.name == "lid_driven_cavity"

    def test_get_unknown(self):
        assert registry.get("nonexistent") is None

    def test_run_unknown_raises(self):
        with pytest.raises(KeyError, match="unknown example"):
            registry.run("nonexistent")

    def test_duplicate_register_raises(self):
        from tensorlbm.examples.base import Example

        class DupExample(Example):
            @property
            def name(self):
                return "lid_driven_cavity"  # already registered

            def configure(self):
                return {}

            def run(self, params):
                return ExampleResult(name=self.name, params=params, elapsed_s=0, success=True)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(DupExample())


class TestLidDrivenCavityExample:
    def test_configure(self):
        ex = registry.get("lid_driven_cavity")
        cfg = ex.configure()
        assert "n" in cfg
        assert "re" in cfg
        assert "max_steps" in cfg
        assert cfg["n"] >= 17  # minimum grid size

    def test_run_smoke(self, tmp_path: Path):
        """Quick smoke test with small grid and few steps."""
        result = registry.run(
            "lid_driven_cavity",
            n=17,
            re=10,
            u_lid=0.05,
            max_steps=50,
            output_root=str(tmp_path),
            overwrite=True,
        )
        assert result.success
        assert result.elapsed_s > 0
        assert "run_dir" in result.artifacts
        run_dir = Path(result.artifacts["run_dir"])
        assert run_dir.exists()
        assert (run_dir / "metadata.json").exists()
        assert (run_dir / "u_profile.csv").exists()
        assert (run_dir / "v_profile.csv").exists()

    def test_metadata_content(self, tmp_path: Path):
        result = registry.run(
            "lid_driven_cavity",
            n=17,
            re=10,
            u_lid=0.05,
            max_steps=50,
            output_root=str(tmp_path),
            overwrite=True,
        )
        meta = json.loads((Path(result.artifacts["run_dir"]) / "metadata.json").read_text())
        assert meta["example"] == "lid_driven_cavity"
        assert "metrics" in meta
        assert "elapsed_s" in meta

    def test_run_with_overwrite(self, tmp_path: Path):
        """Running twice with overwrite=True should not raise."""
        params = dict(n=17, re=10, u_lid=0.05, max_steps=50, output_root=str(tmp_path), overwrite=True)
        registry.run("lid_driven_cavity", **params)
        registry.run("lid_driven_cavity", **params)  # should not raise


class TestRunnerCLI:
    def test_list_command(self, capsys):
        from tensorlbm.examples.runner import main
        main(["--list"])
        captured = capsys.readouterr()
        assert "lid_driven_cavity" in captured.out

    def test_json_list(self, capsys):
        from tensorlbm.examples.runner import main
        main(["--list", "--json"])
        captured = capsys.readouterr()
        import json
        names = json.loads(captured.out)
        assert "lid_driven_cavity" in names

    def test_unknown_example(self, capsys):
        from tensorlbm.examples.runner import main
        ret = main(["nonexistent"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "unknown example" in captured.err
