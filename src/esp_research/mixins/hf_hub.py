"""Mixin for HuggingFace Hub push/load via checkpoint infrastructure."""

import tempfile
from pathlib import Path
from typing import Self

from huggingface_hub import HfApi, snapshot_download
from pydantic import BaseModel

from esp_research.protocols.checkpointing import CheckpointLoadable, CheckpointSaveable


class HfHubMixin:
    """Mixin providing HuggingFace Hub push and load through checkpoint infrastructure.

    This mixin bridges the existing checkpoint protocols
    (`CheckpointSaveable` / `CheckpointLoadable`) with HuggingFace Hub using
    the "helpers" approach: `HfApi.upload_folder` for push and
    `snapshot_download` for load. This keeps HF as a pure implementation
    detail — no `ModelHubMixin` inheritance, no HF API surface on your class.

    Classes using this mixin must also implement ``save_to_checkpoint_dir``
    and ``from_checkpoint_dir``, as well as declare a ``config_class``
    attribute (a Pydantic ``BaseModel`` subclass).

    """

    def push_to_hf_hub(
        self,
        repo_id: str,
        *,
        config: BaseModel,
        commit_message: str | None = None,
        private: bool = True,
        token: str | None = None,
        branch: str | None = None,
    ) -> str:
        """Push model to HuggingFace Hub.

        Saves checkpoint files into a ``checkpoint/`` subdirectory and
        ``config.json`` at the root, then uploads via `HfApi.upload_folder`.
        This mirrors the layout used by `save_checkpoint_dir()`.

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

        Raises
        ------
        TypeError
            If the instance does not satisfy `CheckpointSaveable`.
        """
        if not isinstance(self, CheckpointSaveable):
            raise TypeError(f"{type(self).__name__} must implement `CheckpointSaveable` to use `push_to_hf_hub`")

        if commit_message is None:
            commit_message = f"Upload {type(self).__name__} using esp_research"

        api = HfApi(token=token)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Save checkpoint into a subdirectory
            checkpoint_subdir = tmpdir_path / "checkpoint"
            checkpoint_subdir.mkdir()
            self.save_to_checkpoint_dir(checkpoint_subdir)

            # Write config.json at root level
            config_path = tmpdir_path / "config.json"
            config_path.write_text(config.model_dump_json(indent=2))

            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
            commit_info = api.upload_folder(
                repo_id=repo_id,
                folder_path=tmpdir,
                commit_message=commit_message,
                revision=branch,
            )

        return commit_info.commit_url

    @classmethod
    def from_hf_hub(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        token: str | None = None,
    ) -> Self:
        """Load a model from HuggingFace Hub.

        Downloads the full repository snapshot via ``snapshot_download``
        and delegates to ``from_checkpoint_dir`` with the ``checkpoint/``
        subdirectory and the path to ``config.json``.

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

        Raises
        ------
        TypeError
            If the class does not satisfy `CheckpointLoadable`.
        """
        if not issubclass(cls, CheckpointLoadable):
            raise TypeError(f"{cls.__name__} must implement `CheckpointLoadable` to use `from_hf_hub`")

        local_dir = snapshot_download(repo_id=repo_id, revision=revision, token=token)
        local_path = Path(local_dir)

        return cls.from_checkpoint_dir(local_path / "checkpoint", local_path / "config.json")
