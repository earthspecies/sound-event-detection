"""Tests for the checkpoint orchestrator functions."""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alp_data.io import anypath, filesystem_from_path
from pydantic import BaseModel

from esp_research.checkpointing import (
    CheckpointLoadError,
    CheckpointSaveError,
    load_checkpoint_dir,
    load_checkpoint_manifest,
    save_checkpoint_dir,
)

# Public GCS bucket used for exercising the remote (non-local) checkpoint path.
_GCS_TEST_BUCKET = "gs://esp-ci-cd-tests"


class DummyConfig(BaseModel):
    value: int


class DummySaveable:
    """Minimal `CheckpointSaveable`/`CheckpointLoadable` for testing."""

    config_class = DummyConfig

    def __init__(self, value: int) -> None:
        self.value = value

    def save_to_checkpoint_dir(self, checkpoint_dir: Path) -> None:
        (checkpoint_dir / "value.txt").write_text(str(self.value))

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir: Path, config: DummyConfig) -> "DummySaveable":
        return cls(value=int((checkpoint_dir / "value.txt").read_text()))


class TestSaveLoadRoundTrip:
    def test_round_trip_restores_object_and_inline_data(self, tmp_path: Path) -> None:
        """Saving then loading restores the object, its config, and inline data."""
        obj = DummySaveable(value=42)
        config = DummyConfig(value=42)

        save_checkpoint_dir(
            tmp_path / "ckpt",
            objects={"model": (obj, config)},
            inline_data={"step": 7},
        )

        result = load_checkpoint_dir(
            tmp_path / "ckpt",
            class_overrides={"model": DummySaveable},
        )

        assert result.objects["model"].value == 42
        assert result.object_configs["model"] == config
        assert result.manifest.inline_data == {"step": 7}
        # FQCN of the dummy is recorded in the manifest registry.
        assert result.manifest.objects["model"].endswith("DummySaveable")


class TestSaveErrors:
    def test_raises_if_checkpoint_already_exists(self, tmp_path: Path) -> None:
        """A second save into the same directory raises `CheckpointSaveError`."""
        save_checkpoint_dir(tmp_path / "ckpt", inline_data={"step": 1})

        with pytest.raises(CheckpointSaveError, match="already exists"):
            save_checkpoint_dir(tmp_path / "ckpt", inline_data={"step": 2})


class TestLoadErrors:
    def test_manifest_missing_directory_raises(self, tmp_path: Path) -> None:
        """Loading a manifest from a non-existent directory raises `CheckpointLoadError`."""
        with pytest.raises(CheckpointLoadError, match="does not exist"):
            load_checkpoint_manifest(tmp_path / "nope")


@pytest.fixture
def remote_ckpt_dir() -> Iterator[str]:
    """Yield a unique checkpoint directory on the GCS test bucket and clean it up.

    Yields
    ------
    str
        A unique `gs://` directory path that is removed after the test.
    """
    path = f"{_GCS_TEST_BUCKET}/_checkpoint_tests/{uuid.uuid4().hex}"
    yield path
    fs = filesystem_from_path(path)
    if fs.exists(path):
        fs.rm(path, recursive=True)


class TestRemoteRoundTrip:
    def test_save_and_load_on_gcs(self, remote_ckpt_dir: str) -> None:
        """Save/load round-trips against a real GCS path.

        Objects are omitted because the leaf `save_to_checkpoint_dir`
        implementations still assume local paths; this exercises the
        orchestrator's own remote (non-`Path`) read/write code.
        """
        # A gs:// path resolves to a non-local path, so the remote branch runs.
        assert not isinstance(anypath(remote_ckpt_dir), Path)

        save_checkpoint_dir(remote_ckpt_dir, inline_data={"step": 5})

        manifest = load_checkpoint_manifest(remote_ckpt_dir)
        assert manifest.inline_data == {"step": 5}
        assert manifest.esp_research_version

        result = load_checkpoint_dir(remote_ckpt_dir)
        assert result.objects == {}
        assert result.manifest.inline_data == {"step": 5}
