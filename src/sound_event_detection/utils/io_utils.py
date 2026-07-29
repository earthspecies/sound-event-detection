"""I/O utilities for loading models and data."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pandas as pd
import torch.nn as nn
from alp_data.io import AnyPathT, anypath, filesystem_from_path


def load_state_dict_verbose(model: nn.Module, state_dict: dict[str, Any]) -> None:
    """Load a state dict into a model with strict=False, printing any key mismatches.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to load weights into.
    state_dict : dict[str, Any]
        The state dictionary (weights) to load.
    """
    incompatible_keys = model.load_state_dict(state_dict, strict=False)

    if incompatible_keys.missing_keys:
        print("--- Warning: Missing Keys in Model ---")
        print("The following keys were in the model but NOT in the checkpoint:")
        for key in incompatible_keys.missing_keys:
            print(f"  > {key}")

    if incompatible_keys.unexpected_keys:
        print("\n--- Warning: Unexpected (Extra) Keys in Checkpoint ---")
        print("The following keys were in the checkpoint but NOT in the model:")
        for key in incompatible_keys.unexpected_keys:
            print(f"  > {key}")

    if not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys:
        print("--- State dict loaded successfully with no mismatches ---")
    else:
        print("\n--- Load complete (with mismatches noted above) ---")


@contextmanager
def open_anypath(path: str | os.PathLike | AnyPathT, mode: str = "r") -> Generator[Any, None, None]:
    """Open a file from a local or cloud path.

    Parameters
    ----------
    path : str | PathLike | AnyPathT
        Local path or cloud URI (e.g. ``gs://bucket/file``).
    mode : str
        File open mode (default ``"r"``).

    Yields
    ------
    IO
        An open file-like object.
    """
    p = anypath(str(path))
    fs = filesystem_from_path(p)
    with fs.open(str(p), mode) as f:
        yield f


def _load_labels_from_single_source(source: str | list[str]) -> list[str]:
    """Load labels from a single source (path or inline list).

    Parameters
    ----------
    source : str | list[str]
        Either a path to a ``.csv``/``.txt`` file, or a list of label strings.

    Returns
    -------
    list[str]
        List of label strings.

    Raises
    ------
    ValueError
        If `source` is a string that does not end with ``.csv`` or ``.txt``.
    """
    if isinstance(source, list):
        return source

    if source.endswith(".csv"):
        df = pd.read_csv(source)
        return df["species"].tolist() if "species" in df.columns else df.iloc[:, 0].tolist()
    elif source.endswith(".txt"):
        with open_anypath(source) as f:
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Labels must be a list, .csv, or .txt file, got: {source}")


def load_labels(labels_config: str | list | None) -> list[str] | None:
    """Load labels from one or more sources and union them.

    Parameters
    ----------
    labels_config : str | list | None
        Can be:

        - ``None`` — returns ``None`` (for encoder types that supply their own labels)
        - A path to a ``.csv``/``.txt`` file
        - A list of label strings (all items are plain strings)
        - A list of sources where each source is either a path or an inline list;
          all sources are unioned together with order preserved

    Returns
    -------
    list[str] | None
        List of unique label strings (order preserved), or ``None`` if
        `labels_config` is ``None``.

    Raises
    ------
    ValueError
        If any source is not a list, ``.csv``, or ``.txt`` file.
    """
    if labels_config is None:
        return None

    if isinstance(labels_config, str):
        labels = _load_labels_from_single_source(labels_config)
        print(f"Loaded {len(labels)} labels from {labels_config}")
        return labels

    if isinstance(labels_config, list) and len(labels_config) > 0:
        first = labels_config[0]
        is_list_of_sources = isinstance(first, list) or (
            isinstance(first, str) and (first.endswith(".csv") or first.endswith(".txt") or "/" in first)
        )

        if is_list_of_sources:
            all_labels: list[str] = []
            seen: set[str] = set()
            for source in labels_config:
                source_labels = _load_labels_from_single_source(source)
                source_desc = source if isinstance(source, str) else f"inline list ({len(source)} items)"
                print(f"Loaded {len(source_labels)} labels from {source_desc}")
                for label in source_labels:
                    if label not in seen:
                        all_labels.append(label)
                        seen.add(label)
            print(f"Total: {len(all_labels)} unique labels from {len(labels_config)} sources")
            return all_labels
        else:
            print(f"Using {len(labels_config)} labels from config")
            return labels_config

    raise ValueError(f"Invalid labels_config: {labels_config}")
