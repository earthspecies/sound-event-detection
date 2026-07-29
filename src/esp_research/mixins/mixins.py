"""Reusable mixins for common class functionality.

This module provides mixin classes that add common functionality to classes
"""


# TODO: it might make sense to break this file into multiples later

from pathlib import Path
from typing import Any, Generic, Self, Type, TypeVar

from alp_data.io import AnyPathT, anypath, read_yaml
from pydantic import BaseModel

from esp_research.types import AnyPathOrStr

# Generic type variable for any Pydantic config (used by FromConfigMixin)
ConfigT = TypeVar("ConfigT", bound=BaseModel)


class FromConfigMixin(Generic[ConfigT]):
    """Mixin providing configuration-based initialization.

    This mixin implements a `from_config` method that enables classes to be
    instantiated from configuration objects or JSON/YAML files. It expects the
    class using this mixin to have a `config_class` attribute that specifies the
    Pydantic model for configuration validation.

    The generic parameter allows type checkers to track the specific config
    type, though at runtime any BaseModel subclass works.

    Example:
        >>> from pydantic import BaseModel
        >>> class MyConfig(BaseModel):
        ...     param1: int
        ...     param2: str
        >>> class MyClass(FromConfigMixin[MyConfig]):
        ...     config_class = MyConfig
        ...     def __init__(self, param1: int, param2: str):
        ...         self.param1 = param1
        ...         self.param2 = param2
        >>> obj = MyClass.from_config(MyConfig(param1=42, param2="value"))
        >>> obj.param1
        42
        >>> obj.param2
        'value'
    """

    config_class: Type[ConfigT]

    @classmethod
    def from_config(cls, config: ConfigT | AnyPathOrStr) -> Self:
        """Create an instance from a config object or file.

        Parameters
        ----------
        config : ConfigT | AnyPathOrStr
            Either a config object (Pydantic BaseModel) or a path to a
            configuration file. Supported file formats: `.json`, `.yaml`, `.yml`.

        Returns
        -------
        Self
            An instance of the class initialized with the config parameters.

        Raises
        ------
        ValueError
            If the config file has an unsupported extension.
        """

        if isinstance(config, (Path, AnyPathT, str)):
            # temp until alp-data is updated
            if isinstance(config, str):
                config = anypath(config)

            if config.suffix == ".json":
                # TODO: we need read_json in alp_data
                # 🚨🚨🚨

                from esp_research.utils.utils import read_json

                data = read_json(config)
            elif config.suffix in [".yaml", ".yml"]:
                data = read_yaml(config)
            else:
                # TODO: raise error for unsupported file extension
                raise ValueError(f"Unsupported config file extension: {config.suffix}. Use .json, .yaml, or .yml")

            config = cls.config_class(**data)

        return cls(**config.model_dump())


class TorchModelEqualityMixin:
    """Mixin providing equality comparison for PyTorch models based on state dictionaries.

    This mixin implements the `__eq__` method required by the TrainableModel
    protocol, comparing two PyTorch models by their state dictionaries. Models
    are considered equal if they are of the same type and have identical state
    dictionaries (weights and buffers).

    Note:
        This implementation compares string representations of state
        dictionaries, which may be computationally expensive for large models.
        It also requires the model to have a `state_dict()` method (standard for
        PyTorch nn.Module).

    Example:
        >>> import torch.nn as nn
        >>> class MyTorchModel(nn.Module, TorchModelEqualityMixin):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = nn.Linear(10, 5)
        >>> model1 = MyTorchModel()
        >>> model2 = MyTorchModel()
        >>> model1 == model2
        False
    """

    def __eq__(self: Self, other: Any) -> bool:  # noqa: ANN401
        if type(self) is not type(other):
            return NotImplemented

        # TODO: do we want to check some attributes as well?

        return str(self.state_dict()) == str(other.state_dict())


# TODO: Add reference implementations for from_checkpoint() and to_checkpoint()
