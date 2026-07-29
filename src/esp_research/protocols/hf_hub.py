"""Protocols for HuggingFace Hub integration."""

from typing import Protocol, Self, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class HfHubPushable(Protocol):
    """Protocol for objects that can be pushed to HuggingFace Hub."""

    def push_to_hf_hub(
        self,
        repo_id: str,
        *,
        config: BaseModel,
        commit_message: str | None,
        private: bool,
        token: str | None,
        branch: str | None,
    ) -> str:
        """Push model to HuggingFace Hub.

        Parameters
        ----------
        repo_id : str
            The HuggingFace Hub repository ID (e.g. ``"EarthSpeciesProject/model-name"``).
        config : BaseModel
            The model configuration to save alongside the checkpoint.
        commit_message : str | None
            Custom commit message. If ``None``, a default is used.
        private : bool
            Whether to create a private repository.
        token : str | None
            HuggingFace API token. If ``None``, uses the cached token.
        branch : str | None
            Git branch to push to. If ``None``, pushes to the default branch.

        Returns
        -------
        str
            The commit URL on HuggingFace Hub.
        """
        ...


@runtime_checkable
class HfHubLoadable(Protocol):
    """Protocol for classes that can be loaded from HuggingFace Hub."""

    @classmethod
    def from_hf_hub(
        cls,
        repo_id: str,
        *,
        revision: str | None,
        token: str | None,
    ) -> Self:
        """Load a model from HuggingFace Hub.

        Parameters
        ----------
        repo_id : str
            The HuggingFace Hub repository ID (e.g. ``"EarthSpeciesProject/model-name"``).
        revision : str | None
            Git revision to download. If ``None``, uses the default branch.
        token : str | None
            HuggingFace API token. If ``None``, uses the cached token.

        Returns
        -------
        Self
            The loaded model instance.
        """
        ...
