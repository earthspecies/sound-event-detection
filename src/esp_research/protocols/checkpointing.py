"""Protocols and types for checkpoint-capable objects."""

from typing import Protocol, Self, runtime_checkable

from pydantic import BaseModel

from esp_research.types import AnyPathOrStr


@runtime_checkable
class CheckpointSaveable(Protocol):
    """Protocol for objects that can save their state to a checkpoint directory."""

    def save_to_checkpoint_dir(self, checkpoint_dir: AnyPathOrStr) -> None:
        """Save state to checkpoint_dir subdirectory."""
        ...


@runtime_checkable
class CheckpointLoadable(Protocol):
    """Protocol for classes that can be loaded from checkpoints.

    Objects implementing this protocol can be loaded from a checkpoint
    subdirectory given an explicit configuration. The orchestrator
    (`load_checkpoint_dir`) reads the manifest, imports classes, reconstructs
    configs, and calls this method.

    The key simplification is that objects receive their subdirectory path and
    config as explicit parameters - they don't need to know about manifest structure
    or key-based lookups.
    """

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir: AnyPathOrStr, config: BaseModel | AnyPathOrStr) -> Self:
        """Load from checkpoint subdirectory with provided config.

        Parameters
        ----------
        checkpoint_dir : AnyPathOrStr
            Path to this object's subdirectory (e.g., checkpoint_root/model/).
            All state files for this object are in this directory.
        config : BaseModel | AnyPathOrStr
            Configuration for reconstructing this object. Can be a Pydantic
            model (reconstructed from the manifest's stored config dict) or
            a path/string pointing to an external config file.

        Returns
        -------
        Self
            The loaded object instance.

        Raises
        ------
        CheckpointLoadError
            If the object cannot be loaded from the checkpoint directory.
        """
        ...
