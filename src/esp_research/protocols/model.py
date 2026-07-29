"""Model interface protocols and mixins for ESP research models.

TODO: This module is still under active development. Add more documentation later
"""

from typing import Any, Generic, Protocol, Self, Type, TypeVar, runtime_checkable

from pydantic import BaseModel

from esp_research.types import AnyPathOrStr

from .checkpointing import CheckpointLoadable, CheckpointSaveable

# TODO For now here just so that I can look at their implementations quickly
# from huggingface_hub import PyTorchModelHubMixin
# from transformers import AutoConfig, AutoModel

# TODO: Maybe we can do some sort of mypy test in pre-commit or CI to test if
# people's models match the interface below or not but we can probably also have
# a test suite that all models should pass?

# TODO: Is TrainableModel a good name? ESPModel, ModelInterface, SupportsTraining, etc?

# TODO: maybe it makes more sense to go for something more composbale? like
# forcing the model to have "checkpointer", "config_manager", etc?

# TODO: Maybe we can add support for surgery similar to Flax?
# See https://flax.readthedocs.io/en/latest/guides/surgery.html


class ModelConfig(BaseModel):
    """Base configuration class for ESP models."""

    #
    pass
    # TODO: maybe it's useful to have a to_init_args() method to be more explicit?


ModelConfigT = TypeVar("ModelConfigT", bound=ModelConfig)


@runtime_checkable
class TrainableModel(
    CheckpointSaveable,
    CheckpointLoadable,
    Protocol,
    Generic[ModelConfigT],
):
    """Protocol defining the standard interface for all ESP trainable models.

    This protocol establishes a consistent API for model lifecycle management
    including configuration, initialization, checkpointing, and comparison.
    Models implementing this interface can be seamlessly integrated with ESP's
    training infrastructure.
    """

    # Class attribute specifying the configuration class for this model type.
    # Used to parse and validate configuration files or objects, enabling
    # declarative model instantiation from YAML files or config instances. Each
    # model implementation must define this attribute to reference its specific
    # configuration class
    config_class: Type[ModelConfigT]

    @classmethod
    def from_config(cls, config: ModelConfigT | AnyPathOrStr) -> Self: ...

    # TODO: is this something we actually need and want to enforce?
    def __eq__(self: Self, other: Any) -> bool:  # noqa: ANN401
        ...

    # TODO: define __call__ ?

    # a random list of other methods/integrations I'm thinking about:
    # - get_input_spec(self) -> dict[str, Any]: ...  # Expected input format
    # - get_output_spec(self) -> dict[str, Any]: ...  # Output format
    # - get_hw_requirements() could be useful for slurm?
    # - get_metadata()
    # - to_state() and from_state(): like to_checkpoint() but returns state instead of saving to file
    # - get_data_format()
    # - get_version()
    # - get_checkpoint_metadata()
    # - register()
    # - owner attribute?
    # - from_wandb() or from_mlflow() or ...
    # - def export_to_hf_hub(self, *args: Any, **kwargs: Any) -> Any: ...
    # - def import_from_hf_hub(self, *args: Any, **kwargs: Any) -> Any: ...
    # - get_layer_names() -> list[str]
    # - get_layer_params(layer: str) -> torch.Tensor
    # - integration with _recipes_?


# TODO: this one is still work in progress. Do we want a separate protocol. Probably yesi
class InferenceModel(CheckpointLoadable, Protocol):
    """Base protocol for ESP inference models."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        pass
