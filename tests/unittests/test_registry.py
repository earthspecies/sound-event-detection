"""Tests for the Registry class."""

from typing import Literal, Self

import pytest
from pydantic import BaseModel

from esp_research.utils.registry import Registry


class DummyConfig(BaseModel):
    type: Literal["dummy"] = "dummy"
    value: int


class Dummy:
    config_class = DummyConfig

    def __init__(self, value: int) -> None:
        self.value = value

    @classmethod
    def from_config(cls, cfg: DummyConfig) -> Self:
        return cls(value=cfg.value)


class AnotherConfig(BaseModel):
    type: Literal["another"] = "another"
    name: str


class Another:
    config_class = AnotherConfig

    def __init__(self, name: str) -> None:
        self.name = name

    @classmethod
    def from_config(cls, cfg: AnotherConfig) -> Self:
        return cls(name=cfg.name)


class TestRegistryRegister:
    def test_register_with_config_class_attribute(self) -> None:
        registry: Registry[Dummy] = Registry("test")

        @registry.register
        class MyClass(Dummy):
            pass

        assert "dummy" in registry

    def test_register_with_explicit_config_class(self) -> None:
        registry: Registry[Dummy] = Registry("test")

        @registry.register(config_class=DummyConfig)
        class MyClass:
            @classmethod
            def from_config(cls, cfg: DummyConfig) -> Self:
                return cls()

        assert "dummy" in registry

    def test_register_raises_without_config_class(self) -> None:
        registry: Registry[Dummy] = Registry("test")

        with pytest.raises(AttributeError, match="must have a 'config_class' attribute"):

            @registry.register
            class MyClass:
                @classmethod
                def from_config(cls, cfg: DummyConfig) -> Self:
                    return cls()

    def test_register_raises_without_from_config(self) -> None:
        registry: Registry[Dummy] = Registry("test")

        with pytest.raises(TypeError, match="must have a 'from_config' classmethod"):

            @registry.register
            class MyClass:
                config_class = DummyConfig

    def test_register_raises_on_duplicate_type(self) -> None:
        registry: Registry[Dummy] = Registry("test")
        registry.register(Dummy)

        with pytest.raises(ValueError, match="is already registered"):

            @registry.register
            class MyClass(Dummy):
                pass

    def test_register_raises_on_non_basemodel_config(self) -> None:
        registry: Registry[Dummy] = Registry("test")

        with pytest.raises(TypeError, match="must be a Pydantic BaseModel"):

            @registry.register(config_class=dict)  # type: ignore[arg-type]
            class MyClass(Dummy):
                pass


class TestRegistryCreate:
    def test_create_from_dict(self) -> None:
        registry: Registry[Dummy] = Registry("test")
        registry.register(Dummy)

        obj = registry.create({"type": "dummy", "value": 42})

        assert isinstance(obj, Dummy)
        assert obj.value == 42

    def test_create_from_config_object(self) -> None:
        registry: Registry[Dummy] = Registry("test")
        registry.register(Dummy)

        config = DummyConfig(value=42)
        obj = registry.create(config)

        assert isinstance(obj, Dummy)
        assert obj.value == 42

    def test_create_raises_on_unknown_type(self) -> None:
        registry: Registry[Dummy] = Registry("test")
        registry.register(Dummy)

        with pytest.raises(ValueError, match="Unknown test type"):
            registry.create({"type": "unknown", "value": 1})


class TestRegistryTypes:
    def test_types_returns_registered_types(self) -> None:
        registry: Registry[Dummy | Another] = Registry("test")
        registry.register(Dummy)
        registry.register(Another)

        assert set(registry.types) == {"dummy", "another"}

    def test_contains(self) -> None:
        registry: Registry[Dummy] = Registry("test")
        registry.register(Dummy)

        assert "dummy" in registry
        assert "unknown" not in registry


class TestRegistryGetMethods:
    def test_getitem_returns_registered_class(self) -> None:
        registry: Registry[Dummy] = Registry("test")
        registry.register(Dummy)

        assert registry["dummy"] is Dummy

    def test_getitem_raises_on_unknown_type(self) -> None:
        registry: Registry[Dummy] = Registry("test")

        with pytest.raises(KeyError, match="Unknown test type: 'nope'"):
            registry["nope"]

    def test_get_config_class_returns_config(self) -> None:
        registry: Registry[Dummy] = Registry("test")
        registry.register(Dummy)

        assert registry.get_config_class("dummy") is DummyConfig

    def test_get_config_class_raises_on_unknown_type(self) -> None:
        registry: Registry[Dummy] = Registry("test")

        with pytest.raises(KeyError, match="Unknown test type: 'nope'"):
            registry.get_config_class("nope")


class TestRegistryCustomDiscriminator:
    def test_custom_discriminator(self) -> None:
        class KindConfig(BaseModel):
            kind: Literal["my_kind"] = "my_kind"
            value: int

        class KindClass:
            config_class = KindConfig

            def __init__(self, value: int) -> None:
                self.value = value

            @classmethod
            def from_config(cls, cfg: KindConfig) -> Self:
                return cls(value=cfg.value)

        registry: Registry[KindClass] = Registry("test", discriminator="kind")
        registry.register(KindClass)

        obj = registry.create({"kind": "my_kind", "value": 10})

        assert obj.value == 10
