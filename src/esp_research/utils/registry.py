"""Generic registry for config-based factory classes.

This module provides a Registry class that enables defining multiple variants of
a component (e.g., different optimizers like Adam or SGD, different schedulers,
different model architectures) that can be selected dynamically via configuration.
Each registered class is associated with a Pydantic config that includes a
discriminator field (e.g., "type") to identify which variant to use.

The registry handles:
- Validation of configs using Pydantic's discriminated unions
- Automatic routing to the correct config class based on the discriminator
- Instantiation of the corresponding class via its `from_config` method

This pattern allows users to specify component variants in YAML/JSON configs and
have them automatically validated and instantiated without manual dispatch logic.
"""

from typing import Any, Callable, TypeVar, get_args

from pydantic import BaseModel, TypeAdapter

T = TypeVar("T")


class Registry[T]:
    """A generic registry for config-based factory classes.

    Registered classes must:
    - Provide a config class either via a `config_class` class attribute pointing
      to a Pydantic BaseModel, or by passing it explicitly via `register(config_class=...)`.
    - Have a `from_config` classmethod that accepts the config and returns an instance.
      Consider using `FromConfigMixin` to provide this method.

    The config class must have a discriminator field (default: "type") with a
    single `Literal` value that uniquely identifies the config type.

    Parameters
    ----------
    name : str
        A descriptive name for what this registry contains (used in error messages).
    discriminator : str
        The field name used to distinguish between config types (default: "type").

    Example
    -------
    >>> from typing import Literal, Self
    >>> from pydantic import BaseModel
    >>> from esp_research.utils.registry import Registry
    >>>
    >>> class AdamConfig(BaseModel):
    ...     type: Literal["adam"] = "adam"
    ...     lr: float = 0.001
    ...
    >>> class Adam:
    ...     config_class = AdamConfig
    ...     def __init__(self, lr: float) -> None:
    ...         self.lr = lr
    ...     @classmethod
    ...     def from_config(cls, cfg: AdamConfig) -> Self:
    ...         return cls(lr=cfg.lr)
    ...
    >>> optimizers = Registry[Adam]("optimizer")
    >>> optimizers.register(Adam)
    <class 'esp_research.utils.registry.Adam'>
    >>> obj = optimizers.create({"type": "adam", "lr": 0.01})
    >>> obj.lr
    0.01
    """

    def __init__(self, name: str = "item", discriminator: str = "type") -> None:
        """
        Parameters
        ----------
        name : str
            A descriptive name for what this registry contains (used in error messages).
        discriminator : str
            The field name used to distinguish between config types (default: "type").
        """
        self._name = name
        self._discriminator = discriminator
        self._config_registry: dict[str, type[BaseModel]] = {}
        self._factory_registry: dict[type[BaseModel], type[T]] = {}

    def register(
        self,
        cls: type[T] | None = None,
        *,
        config_class: type[BaseModel] | None = None,
    ) -> type[T] | Callable[[type[T]], type[T]]:
        """Register a class. Can be used as a decorator.

        Parameters
        ----------
        cls : type[T] | None
            The class to register. If None, returns a decorator.
        config_class : type[BaseModel] | None
            The config class to use. If None, uses the class's `config_class` attribute.

        Returns
        -------
        type[T] | Callable[[type[T]], type[T]]
            The registered class, or a decorator if `cls` is None.
        """

        def decorator(cls: type[T]) -> type[T]:
            """Register a class with the registry.

            Returns
            -------
            type[T]
                The registered class (unchanged).

            Raises
            ------
            AttributeError
                If no config_class is provided and the class has no `config_class` attribute.
            TypeError
                If the config_class is not a Pydantic BaseModel, or if the class does not
                have a `from_config` classmethod.
            ValueError
                If the config class has no discriminator field, the discriminator is not a
                single Literal value, or the type name is already registered.
            """
            cfg_cls = config_class or getattr(cls, "config_class", None)

            if cfg_cls is None:
                raise AttributeError(
                    f"Class '{cls.__name__}' must have a 'config_class' attribute or pass config_class to register()."
                )

            if not issubclass(cfg_cls, BaseModel):
                raise TypeError(f"config_class must be a Pydantic BaseModel, got {type(cfg_cls)}")

            if self._discriminator not in cfg_cls.model_fields:
                raise ValueError(f"Config class '{cfg_cls.__name__}' must have a '{self._discriminator}' field.")

            type_vals = get_args(cfg_cls.model_fields[self._discriminator].annotation)
            if len(type_vals) != 1:
                raise ValueError(
                    f"Config '{self._discriminator}' field must be a single Literal value, got: {type_vals}"
                )

            type_name = type_vals[0]

            if type_name in self._config_registry:
                raise ValueError(f"{self._name.title()} type '{type_name}' is already registered.")

            if not hasattr(cls, "from_config") or not callable(cls.from_config):
                raise TypeError(
                    f"Class '{cls.__name__}' must have a 'from_config' classmethod. Consider using FromConfigMixin."
                )

            self._config_registry[type_name] = cfg_cls
            self._factory_registry[cfg_cls] = cls

            return cls

        if cls is not None:
            return decorator(cls)
        return decorator

    def validate(self, data: dict[str, Any] | BaseModel) -> BaseModel:
        """Validate data against registered config types.

        Parameters
        ----------
        data : dict[str, Any] | BaseModel
            The data to validate, either as a dict or an already-instantiated config.

        Returns
        -------
        BaseModel
            The validated config object.

        Raises
        ------
        ValueError
            If the config type is not registered or the discriminator value is unknown.
        """
        if isinstance(data, BaseModel):
            if type(data) not in self._factory_registry:
                raise ValueError(f"{self._name.title()} config '{type(data).__name__}' is not registered.")
            return data

        type_name = data.get(self._discriminator)
        if type_name not in self._config_registry:
            raise ValueError(
                f"Unknown {self._name} {self._discriminator}: '{type_name}'. "
                f"Available: {list(self._config_registry.keys())}"
            )

        config_class = self._config_registry[type_name]
        return TypeAdapter(config_class).validate_python(data)

    def create(self, cfg: dict[str, Any] | BaseModel) -> T:
        """Create an instance from a config dict or object.

        Parameters
        ----------
        cfg : dict[str, Any] | BaseModel
            The config data, either as a dict or an already-instantiated config.

        Returns
        -------
        T
            The created instance.
        """
        validated = self.validate(cfg)
        factory_cls = self._factory_registry[type(validated)]
        return factory_cls.from_config(validated)

    def __getitem__(self, type_name: str) -> type[T]:
        """Look up the registered class by its type name.

        Parameters
        ----------
        type_name : str
            The discriminator value identifying the registered class.

        Returns
        -------
        type[T]
            The registered class.

        Raises
        ------
        KeyError
            If the type name is not registered.
        """
        if type_name not in self._config_registry:
            raise KeyError(f"Unknown {self._name} type: '{type_name}'. Available: {list(self._config_registry.keys())}")
        cfg_cls = self._config_registry[type_name]
        return self._factory_registry[cfg_cls]

    def get_config_class(self, type_name: str) -> type[BaseModel]:
        """Look up the config class by its type name.

        Parameters
        ----------
        type_name : str
            The discriminator value identifying the registered class.

        Returns
        -------
        type[BaseModel]
            The config class for the registered type.

        Raises
        ------
        KeyError
            If the type name is not registered.
        """
        if type_name not in self._config_registry:
            raise KeyError(f"Unknown {self._name} type: '{type_name}'. Available: {list(self._config_registry.keys())}")
        return self._config_registry[type_name]

    @property
    def types(self) -> list[str]:
        """List all registered type names."""
        return list(self._config_registry.keys())

    def __contains__(self, type_name: str) -> bool:
        return type_name in self._config_registry

    def __repr__(self) -> str:
        return f"Registry[{self._name}]({self.types})"
