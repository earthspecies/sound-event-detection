"""Protocols for evaluation tasks and scoring functions."""

from typing import Any, Protocol, Self, Sequence, TypeVar, runtime_checkable

from pydantic import BaseModel

from esp_research.types import AnyPathOrStr

from .model import InferenceModel


# TODO: Use ApplicationConfig as base?
class EvalConfig(BaseModel):
    """An eval is a collection of tasks."""

    # Is this only need for the type below ?
    # Might be causing import slowdowns
    pass


EvalConfigT = TypeVar("EvalConfigT", bound=EvalConfig)


# Need a dataclass type protocol for EvalResult?
class EvalResult(Protocol):
    """Protocol for eval result objects.

    An eval result object contains the results of evaluating a model on a task.
    Its a container for MetricOutput-like objects.
    """

    name: str
    value: Any
    details: dict[str, Any]

    # can we compare eval results? ordering?
    def __eq__(self: Self, other: Any) -> bool:  # noqa: ANN401
        ...

    def __lt__(self: Self, other: Any) -> bool:  # noqa: ANN401
        ...


@runtime_checkable
class EvaluatesModelOnTasks(Protocol):
    """Protocol for task evaluator classes.

    A task evaluator is an orchestrator that evaluates a model on a collection
    of tasks, called an "eval", like BEANSZero.

    Evaluators must define:
    - `config_class`: The Pydantic config class for this evaluator.
    - `expected_model_output`: A Pydantic BaseModel describing the schema
      that models must return. Used for contract introspection via the CLI.

    Public Methods
    - `evaluate(model)` - Run the eval against a model.
    - `from_config(config)` - Create from a config object or path.

    Future:
    - `save_to_checkpoint_dir(checkpoint_dir)` - Inherited from `CheckpointSaveable`.
    - `from_checkpoint_dir(checkpoint_dir, config)` - Inherited from `CheckpointLoadable`.
    - `list_tasks()` - Return a list of task names in this eval.
    - `publish()` - Publish results to a tracking system and return a URL.
    """

    config_class: type[EvalConfigT]
    expected_model_output: type[BaseModel]

    def evaluate(
        self,
        model: InferenceModel,
    ) -> EvalResult: ...

    @classmethod
    def from_config(
        cls,
        config: EvalConfigT | AnyPathOrStr,
    ) -> Self: ...

    # Nice-to-haves
    # def list_tasks(self) -> list[str]: ...

    # def publish(self) -> str: ...

    # def task_completed(self, task_name: str, wandb_run_id?) -> bool: ...

    # def summarize_results(self) -> dict[str, Any]: ...

    # This could be useful for batch evals where predictions are pre-computed
    # def compute_metrics(self, predictions: dict[str, Sequence[Any]]) -> EvalResult: ...


class ComputesScore(Protocol):
    """Protocol for score functions.

    All scorers receive at least the predictions of a model and optionally the targets.
    They return a scalar float score.

    Predictions are first argument and targets are second argument
    following pytorch's loss convention.
    """

    def __call__(
        self,
        predictions: Sequence[Any],
        targets: Sequence[Any] | None,
        **kwargs: Any,
    ) -> float: ...
