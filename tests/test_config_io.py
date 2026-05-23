"""Tests for config_io.py: load_config from YAML/TOML with env overrides."""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from tensorlbm import load_config

if TYPE_CHECKING:
    from pathlib import Path


@dataclasses.dataclass
class _DemoConfig:
    nx: int = 32
    ny: int = 16
    u_in: float = 0.05
    label: str = "default"
    active: bool = True


class TestLoadConfigYaml:
    def _yaml_file(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "cfg.yaml"
        p.write_text(content, encoding="utf-8")
        return p

    def test_loads_int_field(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        p = self._yaml_file(tmp_path, "nx: 64\nny: 32\n")
        cfg = load_config(_DemoConfig, p)
        assert cfg.nx == 64
        assert cfg.ny == 32

    def test_loads_float_field(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        p = self._yaml_file(tmp_path, "u_in: 0.12\n")
        cfg = load_config(_DemoConfig, p)
        assert cfg.u_in == pytest.approx(0.12)

    def test_loads_string_field(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        p = self._yaml_file(tmp_path, "label: my_run\n")
        cfg = load_config(_DemoConfig, p)
        assert cfg.label == "my_run"

    def test_defaults_preserved_for_missing_keys(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        p = self._yaml_file(tmp_path, "nx: 64\n")
        cfg = load_config(_DemoConfig, p)
        assert cfg.ny == 16  # default preserved
        assert cfg.u_in == pytest.approx(0.05)

    def test_returns_instance_of_config_class(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        p = self._yaml_file(tmp_path, "nx: 8\n")
        cfg = load_config(_DemoConfig, p)
        assert isinstance(cfg, _DemoConfig)

    def test_empty_yaml_uses_defaults(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        p = self._yaml_file(tmp_path, "")
        cfg = load_config(_DemoConfig, p)
        assert cfg.nx == 32

    def test_yml_extension_accepted(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        p = tmp_path / "cfg.yml"
        p.write_text("nx: 10\n", encoding="utf-8")
        cfg = load_config(_DemoConfig, p)
        assert cfg.nx == 10


class TestLoadConfigToml:
    def _toml_file(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "cfg.toml"
        p.write_text(content, encoding="utf-8")
        return p

    def _skip_if_no_toml(self) -> None:
        try:
            import tomllib  # noqa: F401
        except ImportError:
            try:
                import tomli  # noqa: F401
            except ImportError:
                pytest.skip("neither tomllib nor tomli is available")

    def test_loads_int_field(self, tmp_path: Path) -> None:
        self._skip_if_no_toml()
        p = self._toml_file(tmp_path, "nx = 128\nny = 64\n")
        cfg = load_config(_DemoConfig, p)
        assert cfg.nx == 128

    def test_loads_float_field(self, tmp_path: Path) -> None:
        self._skip_if_no_toml()
        p = self._toml_file(tmp_path, "u_in = 0.08\n")
        cfg = load_config(_DemoConfig, p)
        assert cfg.u_in == pytest.approx(0.08)

    def test_returns_instance(self, tmp_path: Path) -> None:
        self._skip_if_no_toml()
        p = self._toml_file(tmp_path, "nx = 16\n")
        cfg = load_config(_DemoConfig, p)
        assert isinstance(cfg, _DemoConfig)


class TestUnsupportedFormat:
    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.json"
        p.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config(_DemoConfig, p)


class TestEnvOverrides:
    def test_env_var_overrides_file_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("yaml")
        p = tmp_path / "cfg.yaml"
        p.write_text("nx: 32\n", encoding="utf-8")
        monkeypatch.setenv("TENSORLBM_NX", "256")
        cfg = load_config(_DemoConfig, p, env_prefix="TENSORLBM")
        assert cfg.nx == 256

    def test_env_var_overrides_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("yaml")
        p = tmp_path / "cfg.yaml"
        p.write_text("", encoding="utf-8")
        monkeypatch.setenv("TENSORLBM_NY", "128")
        cfg = load_config(_DemoConfig, p, env_prefix="TENSORLBM")
        assert cfg.ny == 128

    def test_env_var_case_insensitive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("yaml")
        p = tmp_path / "cfg.yaml"
        p.write_text("", encoding="utf-8")
        monkeypatch.setenv("TENSORLBM_NX", "64")
        cfg = load_config(_DemoConfig, p, env_prefix="tensorlbm")
        assert cfg.nx == 64

    def test_unrelated_env_vars_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("yaml")
        p = tmp_path / "cfg.yaml"
        p.write_text("nx: 32\n", encoding="utf-8")
        monkeypatch.setenv("OTHER_NX", "999")
        cfg = load_config(_DemoConfig, p)
        assert cfg.nx == 32

    def test_float_coercion_from_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env var string for a float field is coerced to float."""
        pytest.importorskip("yaml")
        p = tmp_path / "cfg.yaml"
        p.write_text("", encoding="utf-8")
        monkeypatch.setenv("TENSORLBM_U_IN", "0.15")
        cfg = load_config(_DemoConfig, p, env_prefix="TENSORLBM")
        assert cfg.u_in == pytest.approx(0.15)

    def test_bool_coercion_true_from_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env var string 'true' for a bool field is coerced to True."""
        pytest.importorskip("yaml")
        p = tmp_path / "cfg.yaml"
        p.write_text("", encoding="utf-8")
        monkeypatch.setenv("TENSORLBM_ACTIVE", "true")
        cfg = load_config(_DemoConfig, p, env_prefix="TENSORLBM")
        assert cfg.active is True

    def test_bool_coercion_false_from_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env var string 'false' for a bool field is coerced to False."""
        pytest.importorskip("yaml")
        p = tmp_path / "cfg.yaml"
        p.write_text("", encoding="utf-8")
        monkeypatch.setenv("TENSORLBM_ACTIVE", "false")
        cfg = load_config(_DemoConfig, p, env_prefix="TENSORLBM")
        assert cfg.active is False


class TestNonDataclass:
    def test_plain_class_loaded(self, tmp_path: Path) -> None:
        """load_config should also work with non-dataclass classes."""
        pytest.importorskip("yaml")

        class _PlainConfig:
            def __init__(self, nx: int = 32, ny: int = 16) -> None:
                self.nx = nx
                self.ny = ny

        p = tmp_path / "cfg.yaml"
        p.write_text("nx: 48\nny: 24\n", encoding="utf-8")
        cfg = load_config(_PlainConfig, p)
        assert cfg.nx == 48
        assert cfg.ny == 24
