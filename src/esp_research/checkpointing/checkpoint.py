"""Utilities for managing checkpoints.

This module provides functions and types for creating self-contained checkpoint
directories. Each checkpoint directory contains a `manifest.json` file and
one or more state files.

Each stateful object (e.g. model, optimizer) is saved with its state files,
Fully-Qualified Class Name (FQCN), and config, so `load_checkpoint_dir` can
automatically reinstantiate them.

Plain serializable data (e.g. scalars and dicts) is stored inline in the
manifest under `inline_data`.

"""

from __future__ import annotations

import importlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alp_data.io import anypath, exists, filesystem_from_path, read_text
from pydantic import BaseModel

from esp_research import __version__
from esp_research.logging import logger
from esp_research.protocols.checkpointing import CheckpointSaveable
from esp_research.types import AnyPathOrStr

CHECKPOINT_FILENAME = "manifest.json"
SCHEMA_VERSION = 1


class CheckpointError(Exception): ...


class CheckpointLoadError(CheckpointError): ...


class CheckpointSaveError(CheckpointError): ...


class CheckpointManifest(BaseModel):
    """Manifest for a checkpoint directory.

    Attributes
    ----------
    schema_version : int
        Version of the checkpoint schema.
    created_at : datetime
        Timestamp when the checkpoint was created.
    esp_research_version : str
        Version of the esp-research package (read from pyproject.toml at install time).
    git_commit : str | None
        Git commit hash at the time of checkpoint creation.
    uncommitted_changes : str | None
        Filename of the patch file containing uncommitted changes, or None if clean.
    objects : dict[str, str]
        Registry of objects stored in the checkpoint, mapping keys to FQCNs.
        Configuration for each object is stored in a separate `{key}.json` file.
    inline_data : dict[str, Any]
        Plain JSON-serializable data stored inline in the manifest.
    """

    schema_version: int = SCHEMA_VERSION
    created_at: datetime
    esp_research_version: str
    git_commit: str | None
    uncommitted_changes: str | None = None
    objects: dict[str, str]
    inline_data: dict[str, Any] = {}


@dataclass(frozen=True)
class CheckpointResult:
    """Result of loading a checkpoint directory.

    Attributes
    ----------
    manifest : CheckpointManifest
        The checkpoint manifest containing metadata and inline data.
    objects : dict[str, Any]
        Dictionary of loaded objects keyed by their checkpoint key.
    object_configs : dict[str, BaseModel]
        Dictionary of configs used to reconstruct each object, keyed by
        the same keys as `objects`.
    """

    manifest: CheckpointManifest
    objects: dict[str, Any]
    object_configs: dict[str, BaseModel]


def _get_fqcn(cls: type) -> str:
    """Get the fully qualified class name for a class.

    The fully qualified class name includes the module path and the class name,
    e.g. `"esp_research.models.MyModel"`. This is stored in checkpoint metadata
    so that the correct class can be imported and instantiated automatically at
    load time.

    Parameters
    ----------
    cls : type
        The class to get the fully qualified name for.

    Returns
    -------
    str
        The fully qualified class name.
    """
    return f"{cls.__module__}.{cls.__qualname__}"


def _import_via_fqcn(fqcn: str) -> type:
    """Import a class from its fully qualified class name.

    Parameters
    ----------
    fqcn : str
        Fully qualified class name (e.g., `"esp_research.models.MyModel"`).

    Returns
    -------
    type
        The imported class.

    Raises
    ------
    CheckpointLoadError
        If the module or class cannot be imported.
    """
    logger.debug(f"Importing class from '{fqcn}'")
    try:
        module_path, class_name = fqcn.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ValueError, ModuleNotFoundError, AttributeError) as e:
        raise CheckpointLoadError(f"Failed to import class '{fqcn}': {e}") from e


def _get_git_commit() -> str | None:
    """Get the current git commit hash.

    Returns
    -------
    str | None
        The short git commit hash, or None if not in a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _get_uncommitted_changes() -> str | None:
    """Get the unified diff of uncommitted changes (staged + unstaged).

    Returns
    -------
    str | None
        The diff output, or None if there are no changes or not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        diff = result.stdout.strip()
        return diff if diff else None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def save_checkpoint_dir(
    path: AnyPathOrStr,
    objects: dict[str, tuple[CheckpointSaveable, BaseModel]] | None = None,
    inline_data: dict[str, Any] | None = None,
) -> None:
    """Create a self-contained checkpoint directory.

    Each object is saved to its own subdirectory (`path / key`) with its state
    files, Fully-Qualified Class Name (FQCN), and config, so that
    `load_checkpoint_dir` can automatically reinstantiate them.

    The directory may already exist (e.g. a shared prefix for multiple steps),
    but it must not already contain a manifest file.

    Parameters
    ----------
    path : AnyPathOrStr
        Path to the checkpoint directory. Created (with parents) if it does
        not exist.
    objects : dict[str, tuple[CheckpointSaveable, BaseModel]] | None, optional
        Mapping of keys to `(saveable_object, config)` tuples.
    inline_data : dict[str, Any] | None, optional
        Plain JSON-serializable data stored inline in the manifest
        (e.g. step count, loss values).

    Raises
    ------
    CheckpointSaveError
        If a checkpoint already exists at `path` or any object fails to save.
    """
    checkpoint_dir = anypath(path)

    manifest_path = checkpoint_dir / CHECKPOINT_FILENAME
    if exists(manifest_path):
        raise CheckpointSaveError(f"Checkpoint already exists: {checkpoint_dir}")

    logger.info(f"💾 Saving checkpoint to {checkpoint_dir}")

    try:
        if isinstance(checkpoint_dir, Path):
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        object_infos: dict[str, str] = {}

        for key, (obj, config) in (objects or {}).items():
            logger.debug(f"Saving state for '{key}' ({type(obj).__name__})")

            obj_dir = checkpoint_dir / key
            if isinstance(obj_dir, Path):
                obj_dir.mkdir(parents=False, exist_ok=False)
            obj.save_to_checkpoint_dir(obj_dir)

            object_infos[key] = _get_fqcn(type(obj))

            config_path = checkpoint_dir / f"{key}.json"
            with filesystem_from_path(config_path).open(str(config_path), "w") as f:
                f.write(config.model_dump_json(indent=2))

        git_commit = _get_git_commit()
        uncommitted_diff = _get_uncommitted_changes()
        patch_filename: str | None = None

        if uncommitted_diff:
            patch_filename = "uncommitted_changes.patch"
            with filesystem_from_path(checkpoint_dir / patch_filename).open(
                str(checkpoint_dir / patch_filename), "w"
            ) as f:
                f.write(uncommitted_diff)
            logger.debug(f"Uncommitted changes detected, saved to {patch_filename}")

        manifest = CheckpointManifest(
            schema_version=SCHEMA_VERSION,
            created_at=datetime.now(UTC),
            esp_research_version=__version__,
            git_commit=git_commit,
            uncommitted_changes=patch_filename,
            objects=object_infos,
            inline_data=inline_data or {},
        )

        metadata_path = checkpoint_dir / CHECKPOINT_FILENAME
        with filesystem_from_path(metadata_path).open(str(metadata_path), "w") as f:
            f.write(manifest.model_dump_json(indent=2))
        logger.info(f"Checkpoint saved successfully to {checkpoint_dir}")

    except CheckpointSaveError:
        # TODO: handle
        raise
    except Exception as e:
        raise CheckpointSaveError(f"Failed to save checkpoint to {checkpoint_dir}: {e}") from e


def load_checkpoint_manifest(path: AnyPathOrStr) -> CheckpointManifest:
    """Load only the checkpoint manifest without loading any objects.

    This is useful for inspecting checkpoint information (creation time, version,
    git commit, etc.) without the overhead of loading all the saved objects.

    Parameters
    ----------
    path : AnyPathOrStr
        Path to the checkpoint directory.

    Returns
    -------
    CheckpointManifest
        The checkpoint manifest containing information about the checkpoint.

    Raises
    ------
    CheckpointLoadError
        If the checkpoint directory or manifest file does not exist,
        or if the manifest cannot be parsed.
    """
    checkpoint_dir = anypath(path)
    logger.debug(f"Loading checkpoint manifest from {checkpoint_dir}")

    if not exists(checkpoint_dir):
        raise CheckpointLoadError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    metadata_path = checkpoint_dir / CHECKPOINT_FILENAME
    if not exists(metadata_path):
        raise CheckpointLoadError(f"Checkpoint manifest not found: {metadata_path}")

    try:
        metadata = CheckpointManifest.model_validate_json(read_text(metadata_path))
    except Exception as e:
        raise CheckpointLoadError(f"Failed to parse manifest: {e}") from e

    logger.debug(
        f"Checkpoint manifest: schema_version={metadata.schema_version}, "
        f"esp_research_version={metadata.esp_research_version}, "
        f"objects={list(metadata.objects.keys())}"
    )

    return metadata


def load_checkpoint_dir(
    path: AnyPathOrStr,
    class_overrides: dict[str, type] | None = None,
) -> CheckpointResult:
    """Load and instantiate all objects from a checkpoint directory.

    Orchestrates loading by reading the manifest once, then for each object:
    1. Imports the class via FQCN (or uses a class override if provided)
    2. Reconstructs the config from manifest data
    3. Passes the object's subdirectory path and config to `from_checkpoint_dir`

    Parameters
    ----------
    path : AnyPathOrStr
        Path to the checkpoint directory.
    class_overrides : dict[str, type] | None, optional
        Mapping of object keys to class types. If a key is present, the
        provided class is used instead of importing via FQCN. This allows
        loading old checkpoints when classes have been moved or renamed.

    Returns
    -------
    CheckpointResult
        Result containing the manifest and all loaded objects.

    Raises
    ------
    CheckpointLoadError
        If the checkpoint cannot be loaded due to missing files, invalid
        manifest, or import errors.
    """
    checkpoint_dir = anypath(path)
    logger.info(f"Loading checkpoint from {checkpoint_dir}")

    manifest = load_checkpoint_manifest(path)

    objects: dict[str, Any] = {}
    object_configs: dict[str, BaseModel] = {}
    overrides = class_overrides or {}

    for key, fqcn in manifest.objects.items():
        # Use class override if provided, otherwise import via FQCN
        if key in overrides:
            cls = overrides[key]
            logger.debug(f"Using class override for '{key}': {cls.__name__}")
        else:
            cls = _import_via_fqcn(fqcn)

        config_path = checkpoint_dir / f"{key}.json"
        if not exists(config_path):
            raise CheckpointLoadError(f"No config file found for '{key}': {config_path}")

        if not hasattr(cls, "config_class"):
            raise CheckpointLoadError(
                f"Class '{fqcn}' does not have a 'config_class' attribute. Cannot reconstruct config for loading."
            )

        config_data = json.loads(read_text(config_path))
        config = cls.config_class(**config_data)
        object_configs[key] = config

        obj_subdir = checkpoint_dir / key

        logger.debug(f"Loading '{key}' via {cls.__name__}.from_checkpoint_dir")
        objects[key] = cls.from_checkpoint_dir(obj_subdir, config)

    n_objects = len(manifest.objects)
    n_data = len(manifest.inline_data)
    logger.info(f"Checkpoint loaded successfully: {n_objects} object(s), {n_data} data key(s)")

    return CheckpointResult(manifest=manifest, objects=objects, object_configs=object_configs)
