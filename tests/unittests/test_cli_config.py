"""Tests for esp_research.configs.cli_config module."""

import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from esp_research.configs import CLIConfig


class NestedConfig(BaseModel):
    """Nested config for testing."""

    lr: float = 0.001


class SampleConfig(CLIConfig):
    """Sample config for testing."""

    batch_size: int = 32
    model: NestedConfig = NestedConfig()


@pytest.fixture
def yaml_config(tmp_path: Path) -> Path:
    """Create a sample YAML config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("batch_size: 64\nmodel:\n  lr: 0.01\n")
    return config_file


class TestCLIConfig:
    """Tests for CLIConfig.from_sources()."""

    def test_loads_from_yaml(self, yaml_config: Path) -> None:
        """Config values are loaded from YAML file."""
        config = SampleConfig.from_sources(yaml_file=yaml_config)

        assert config.batch_size == 64
        assert config.model.lr == 0.01

    def test_cli_args_override_yaml(self, yaml_config: Path) -> None:
        """CLI arguments take precedence over YAML values."""
        config = SampleConfig.from_sources(
            yaml_file=yaml_config,
            cli_args=["batch_size=128"],
        )

        assert config.batch_size == 128
        assert config.model.lr == 0.01  # unchanged from YAML

    def test_cli_args_with_dashes(self, yaml_config: Path) -> None:
        """CLI arguments work with or without -- prefix."""
        config = SampleConfig.from_sources(
            yaml_file=yaml_config,
            cli_args=["--batch_size=128"],
        )

        assert config.batch_size == 128

    def test_nested_cli_args(self, yaml_config: Path) -> None:
        """Nested config values can be overridden with dot notation."""
        config = SampleConfig.from_sources(
            yaml_file=yaml_config,
            cli_args=["model.lr=0.1"],
        )

        assert config.model.lr == 0.1

    def test_env_vars_override_yaml(self, yaml_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables take precedence over YAML values."""
        monkeypatch.setenv("BATCH_SIZE", "256")

        config = SampleConfig.from_sources(yaml_file=yaml_config)

        assert config.batch_size == 256

    def test_cli_args_override_env_vars(self, yaml_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI arguments take precedence over environment variables."""
        monkeypatch.setenv("BATCH_SIZE", "256")

        config = SampleConfig.from_sources(
            yaml_file=yaml_config,
            cli_args=["batch_size=512"],
        )

        assert config.batch_size == 512

    def test_nested_env_vars(self, yaml_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nested config values can be set with double underscore env vars."""
        monkeypatch.setenv("MODEL__LR", "0.5")

        config = SampleConfig.from_sources(yaml_file=yaml_config)

        assert config.model.lr == 0.5

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError if YAML file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            SampleConfig.from_sources(yaml_file=tmp_path / "nonexistent.yaml")

    def test_defaults_used_when_not_in_yaml(self, tmp_path: Path) -> None:
        """Default values are used for fields not in YAML."""
        config_file = tmp_path / "partial.yaml"
        config_file.write_text("batch_size: 64\n")

        config = SampleConfig.from_sources(yaml_file=config_file)

        assert config.batch_size == 64
        assert config.model.lr == 0.001  # default value

    def test_triple_underscore_env_var_with_trailing_underscore_parent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Triple underscore env var sets nested field when parent has trailing underscore.

        MODEL___LR is parsed as MODEL_ + __ + LR, setting model_.lr.
        """

        class NestedConfigForTrailing(BaseModel):
            lr: float = 0.001

        class ConfigWithTrailingUnderscore(CLIConfig):
            model_: NestedConfigForTrailing = NestedConfigForTrailing()

        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_:\n  lr: 0.01\n")

        monkeypatch.setenv("MODEL___LR", "0.99")

        config = ConfigWithTrailingUnderscore.from_sources(yaml_file=config_file)

        assert config.model_.lr == 0.99

    def test_triple_underscore_env_var_does_not_set_private_attr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Triple underscore env var does not set PrivateAttr fields.

        When a field is declared with PrivateAttr, it is excluded from
        env var parsing even if the env var name matches.
        """
        from pydantic import PrivateAttr

        class NestedConfigWithPrivate(BaseModel):
            _lr: float = PrivateAttr(default=0.001)
            lr: float = 0.01

        class ConfigWithPrivateNested(CLIConfig):
            model: NestedConfigWithPrivate = NestedConfigWithPrivate()

        config_file = tmp_path / "config.yaml"
        config_file.write_text("model:\n  lr: 0.01\n")

        # This env var will be parsed as model._lr, but _lr is a PrivateAttr
        monkeypatch.setenv("MODEL___LR", "0.99")

        config = ConfigWithPrivateNested.from_sources(yaml_file=config_file)

        # Private attr should remain at default, not be set by env var
        assert config.model._lr == 0.001
        assert config.model.lr == 0.01  # regular field unchanged
