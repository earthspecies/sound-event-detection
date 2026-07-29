"""On-demand random access to a recording's heavy LSI outputs.

Large-scale inference stores each recording's heavy products — the denoised
waveform, the per-source stem audio, and the framewise predictions — inside
compressed `ItemResult` shards, *not* in the dataset. The
`AttachLSISelectionTables` transform attaches only small strings plus an
``lsi_shard`` pointer column to each dataset row. These helpers are the consumer
side of that pointer: given a row, they open exactly the one shard it points at,
pick out this recording by its id, and decode it back into an `ItemResult`.

This is random access — per recording, on demand: the offline gallery and
ad-hoc analyses load just the recordings they need, when they need them, instead
of holding gigabytes of audio in the dataset. There is no manifest and no lazy
dataframe column; the caller already has the dataset row (required to use the
dataset at all) and reads heavy products explicitly.

    from sound_event_detection.inference.access import load_denoised, load_frame_preds

    row = dataset[i]                  # carries "lsi_shard" + the id column
    wav = load_denoised(row)          # opens only row["lsi_shard"]
    preds = load_frame_preds(row)

`read_item` decodes the whole `ItemResult`; `load_frame_preds` / `load_denoised`
/ `load_stems` are thin accessors for one product each. Each call re-reads the
pointed-to shard, so to read several products for one recording call `read_item`
once and reuse its fields.
"""

from collections.abc import Mapping

import numpy as np

from esp_research.protocols.detector import DetectorOutput
from sound_event_detection.inference.engine import read_shard
from sound_event_detection.inference.result import ItemResult, Stem

__all__ = ["load_denoised", "load_frame_preds", "load_stems", "read_item"]

#: Columns tried, in order, when the caller does not pass `id_column`. Mirrors the
#: LSI producer's shard-key resolution (`str(item[id_column])`).
_DEFAULT_ID_COLUMNS = ("originals_path", "audio_path", "audio_fp", "relative_path")


def _resolve_id(row: Mapping, id_column: str | None) -> str:
    """Return the row's file id, matching the key its shard was written under.

    Parameters
    ----------
    row : Mapping
        A dataset row (e.g. ``dataset[i]``).
    id_column : str or None
        Explicit id column, or ``None`` to try `_DEFAULT_ID_COLUMNS`.

    Returns
    -------
    str
        ``str(row[id_column])`` for the resolved column.

    Raises
    ------
    KeyError
        If an explicit `id_column` is absent from `row`.
    ValueError
        If `id_column` is ``None`` and no default column is present.
    """
    if id_column is not None:
        if id_column not in row:
            raise KeyError(f"id_column '{id_column}' not in row (keys: {list(row.keys())})")
        return str(row[id_column])
    for candidate in _DEFAULT_ID_COLUMNS:
        if candidate in row:
            return str(row[candidate])
    raise ValueError(
        f"could not infer id_column from {_DEFAULT_ID_COLUMNS}; pass id_column. Row keys: {list(row.keys())}"
    )


def read_item(row: Mapping, id_column: str | None = None, shard_column: str = "lsi_shard") -> ItemResult:
    """Open the row's heavy shard and decode its `ItemResult`.

    Parameters
    ----------
    row : Mapping
        A dataset row carrying the `shard_column` pointer (from
        `AttachLSISelectionTables`) and the id column.
    id_column : str or None
        Column holding the file id used to index within the shard. ``None`` tries
        `_DEFAULT_ID_COLUMNS`.
    shard_column : str
        Column holding the heavy-shard path. Default ``"lsi_shard"``.

    Returns
    -------
    ItemResult
        The recording's decoded result (combined preds, and denoised / stem audio
        at the rungs the run stored).

    Raises
    ------
    KeyError
        If `shard_column` is absent, or the id column cannot be resolved, or the
        recording's id is not present in its shard.
    ValueError
        If the pointer is empty (the row had no attached LSI output), or the id
        column cannot be inferred.
    """
    if shard_column not in row:
        raise KeyError(f"shard_column '{shard_column}' not in row; run the attach transform first")
    shard_path = str(row[shard_column])
    if not shard_path:
        raise ValueError(f"row has an empty '{shard_column}' pointer (no LSI output attached for this recording)")

    file_id = _resolve_id(row, id_column)
    recordings = read_shard(shard_path)
    if file_id not in recordings:
        raise KeyError(f"file_id {file_id!r} not found in shard {shard_path!r}")
    return ItemResult.from_arrays(recordings[file_id])


def load_frame_preds(row: Mapping, id_column: str | None = None, shard_column: str = "lsi_shard") -> DetectorOutput:
    """Return the recording's combined framewise predictions.

    Parameters
    ----------
    row : Mapping
        A dataset row (see `read_item`).
    id_column : str or None
        Id column, or ``None`` to infer.
    shard_column : str
        Heavy-shard pointer column.

    Returns
    -------
    DetectorOutput
        Combined predictions of shape ``(1, frames, kept_classes)``.
    """
    return read_item(row, id_column, shard_column).preds


def load_denoised(row: Mapping, id_column: str | None = None, shard_column: str = "lsi_shard") -> np.ndarray | None:
    """Return the recording's denoised waveform, or ``None`` if not stored.

    Parameters
    ----------
    row : Mapping
        A dataset row (see `read_item`).
    id_column : str or None
        Id column, or ``None`` to infer.
    shard_column : str
        Heavy-shard pointer column.

    Returns
    -------
    np.ndarray or None
        The denoised waveform of shape ``(samples,)``, or ``None`` if the run was
        produced at the ``preds`` detail rung.
    """
    return read_item(row, id_column, shard_column).denoised


def load_stems(row: Mapping, id_column: str | None = None, shard_column: str = "lsi_shard") -> tuple[Stem, ...]:
    """Return the recording's per-source stems (empty unless stored).

    Parameters
    ----------
    row : Mapping
        A dataset row (see `read_item`).
    id_column : str or None
        Id column, or ``None`` to infer.
    shard_column : str
        Heavy-shard pointer column.

    Returns
    -------
    tuple[Stem, ...]
        The per-source stems, or an empty tuple unless the run was produced at the
        ``stems`` detail rung.
    """
    return read_item(row, id_column, shard_column).stems
