"""Configuration base class with multi-source loading support.

Provides CLIConfig, a Pydantic BaseSettings subclass that loads configuration
from YAML files, environment variables, and CLI arguments with a defined
priority order.
"""

from pathlib import Path
from typing import Self

from pydantic.v1.utils import deep_update
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    EnvSettingsSource,
    YamlConfigSettingsSource,
)


class CLIConfig(BaseSettings, extra="forbid", validate_assignment=True):
    """Base configuration class that supports loading from YAML files, CLI
    arguments, and environment variables.

    Priority (highest to lowest):
    1. CLI arguments
    2. Environment variables
    3. YAML config file
    4. Default values

    Examples
    --------
    Define a config class:

    >>> class MyConfig(CLIConfig):
    ...     learning_rate: float = 0.001
    ...     batch_size: int = 32

    Load from YAML and CLI:

    >>> config = MyConfig.from_sources(yaml_file="config.yaml", cli_args=["batch_size=64"])  # doctest: +SKIP
    """

    @classmethod
    def from_sources(
        cls,
        *,
        yaml_file: str | Path,
        cli_args: list[str] | None = None,
    ) -> Self:
        """Create a config from a YAML file and CLI arguments.

        Priority (highest to lowest): CLI args → Env vars → YAML file.

        For nested config values, use dot notation for CLI args (e.g., "model.lr=0.01")
        and double underscore for env vars (e.g., "MODEL__LR=0.01").

        Parameters
        ----------
        yaml_file
            Path to the YAML configuration file.
        cli_args
            List of CLI arguments in the form "key=value" or "nested.key=value".
            The "--" prefix is optional.

        Returns
        -------
        Self
            The validated configuration object.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        """
        if cli_args is None:
            cli_args = []

        yaml_file = Path(yaml_file)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Config file {yaml_file} does not exist")

        # Build sources in priority order (lowest to highest):
        # YAML -> Env vars -> CLI args
        yaml_values = YamlConfigSettingsSource(cls, yaml_file=yaml_file)
        env_values = EnvSettingsSource(cls, env_nested_delimiter="__")

        # Ensure CLI args have "--" prefix
        prefixed_args = [arg if arg.startswith("--") else f"--{arg}" for arg in cli_args]
        cli_values = CliSettingsSource(cls, cli_parse_args=prefixed_args)

        # Merge in priority order: CLI (highest) > Env > YAML (lowest)
        final_values = deep_update(
            deep_update(yaml_values(), env_values()),
            cli_values(),
        )
        return cls.model_validate(final_values)
