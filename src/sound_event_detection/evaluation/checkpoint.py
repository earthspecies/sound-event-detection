"""Resumable checkpointing for SED evaluation.

The evaluation accumulates results in a `Scorer` whose state is small, additive
count arrays — so we never store raw frame predictions (which would be huge).
Instead each checkpoint records, per dataset:

- an append-only line in ``progress.jsonl`` (``kind``, ``dataset_key``,
  ``n_completed`` files, ``is_complete``, optional ``state_file`` and ``results``);
  the last line for a key wins, so a crash mid-write at most loses the line in flight, and
- for frame datasets, a `numpy` ``.npz`` snapshot of the scorer's count arrays.

On resume, a complete dataset is skipped (its cached results are reused) and a
partially-finished frame dataset reloads its scorer snapshot and continues from
``n_completed``. Clip datasets are not mid-resumable (the clip scorer needs all
predictions at once), so they checkpoint only on completion.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from esp_research.logging import logger
from sound_event_detection.evaluation.metrics import Scorer

_PROGRESS_FILENAME = "progress.jsonl"
_STATE_SUBDIR = "scorer_state"


@dataclass
class DatasetProgress:
    """Resume state for one dataset.

    Attributes
    ----------
    kind : str
        Either ``"frame"`` or ``"clip"``.
    dataset_key : str
        Dataset name (including split), unique within the run.
    n_completed : int
        Number of files already scored (frame datasets) or total files (clip).
    is_complete : bool
        Whether the whole dataset has been scored.
    state_file : str | None
        Path (relative to the checkpoint dir) of the scorer ``.npz`` snapshot,
        for partially- or fully-completed frame datasets. ``None`` for clip.
    results : dict | None
        Cached results dict, present once ``is_complete`` is True.
    """

    kind: str
    dataset_key: str
    n_completed: int
    is_complete: bool
    state_file: str | None = None
    results: dict | None = None


def _jsonable(obj: Any) -> Any:  # noqa: ANN401
    """Recursively convert numpy scalars/arrays to plain Python for JSON dumps.

    Parameters
    ----------
    obj : Any
        Value possibly containing numpy types, dicts, or lists.

    Returns
    -------
    Any
        A JSON-serialisable copy.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_progress(checkpoint_dir: str | Path, progress: DatasetProgress) -> None:
    """Append one progress record to ``progress.jsonl``.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory holding the checkpoint files.
    progress : DatasetProgress
        Record to append.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "kind": progress.kind,
        "dataset_key": progress.dataset_key,
        "n_completed": progress.n_completed,
        "is_complete": progress.is_complete,
        "state_file": progress.state_file,
        "results": _jsonable(progress.results) if progress.results is not None else None,
    }
    with (checkpoint_dir / _PROGRESS_FILENAME).open("a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(
        "Checkpointed %s/%s (n_completed=%d, complete=%s)",
        progress.kind,
        progress.dataset_key,
        progress.n_completed,
        progress.is_complete,
    )


def load_progress(checkpoint_dir: str | Path) -> dict[str, DatasetProgress]:
    """Load the latest progress record for each dataset.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory holding the checkpoint files.

    Returns
    -------
    dict[str, DatasetProgress]
        Mapping from ``"{kind}/{dataset_key}"`` to its most recent record.
        Empty if no checkpoint exists.
    """
    progress_file = Path(checkpoint_dir) / _PROGRESS_FILENAME
    if not progress_file.exists():
        return {}

    latest: dict[str, DatasetProgress] = {}
    with progress_file.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                key = f"{data['kind']}/{data['dataset_key']}"
                latest[key] = DatasetProgress(
                    kind=data["kind"],
                    dataset_key=data["dataset_key"],
                    n_completed=data["n_completed"],
                    is_complete=data["is_complete"],
                    state_file=data.get("state_file"),
                    results=data.get("results"),
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning("Skipping corrupt progress line: %r", line[:120])
    return latest


def _safe_name(dataset_key: str) -> str:
    """Make a dataset key safe to use in a filename.

    Parameters
    ----------
    dataset_key : str
        Dataset key, possibly containing ``/`` or ``:``.

    Returns
    -------
    str
        Filename-safe variant.
    """
    return dataset_key.replace("/", "_").replace(":", "_")


def save_scorer_state(checkpoint_dir: str | Path, dataset_key: str, n_completed: int, scorer: Scorer) -> str:
    """Write a `Scorer`'s accumulated counts to a fresh ``.npz`` snapshot.

    A new file is written per call (named by ``n_completed``) so writes are never
    in-place — a crash leaves earlier snapshots intact, and `load_progress`
    points at the latest valid one.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory holding the checkpoint files.
    dataset_key : str
        Dataset name (including split).
    n_completed : int
        Number of files scored so far (used in the filename).
    scorer : Scorer
        Scorer whose state to snapshot.

    Returns
    -------
    str
        Path of the snapshot relative to ``checkpoint_dir``.
    """
    state_dir = Path(checkpoint_dir) / _STATE_SUBDIR
    state_dir.mkdir(parents=True, exist_ok=True)

    state = scorer.get_state()
    flat: dict[str, np.ndarray] = {}
    for subkey, arr in state["result_counts"].items():
        if arr is not None:
            flat[f"r::{subkey}"] = arr
    for subkey, arr in state["localization_result_counts"].items():
        if arr is not None:
            flat[f"l::{subkey}"] = arr
    for tkey, counts in state["thresholded_counts"].items():
        for subkey, arr in counts.items():
            if arr is not None:
                flat[f"t::{tkey}::{subkey}"] = arr

    rel = f"{_STATE_SUBDIR}/frame__{_safe_name(dataset_key)}__{n_completed}.npz"
    final_path = Path(checkpoint_dir) / rel
    # Write to a temp file that already ends in ".npz" (so np.savez does not append
    # another suffix), then atomically swap into place.
    tmp_path = final_path.parent / (final_path.name + ".tmp.npz")
    np.savez(tmp_path, **flat)
    os.replace(tmp_path, final_path)
    return rel


def load_scorer_state(checkpoint_dir: str | Path, state_file: str, scorer: Scorer) -> None:
    """Restore a scorer snapshot into `scorer`, in place.

    The snapshot only stores non-``None`` count arrays; missing keys are
    restored as ``None`` using the (freshly constructed) scorer's own key
    layout, so `scorer` must already be configured identically to the run that
    produced the snapshot.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory holding the checkpoint files.
    state_file : str
        Snapshot path relative to ``checkpoint_dir`` (from `DatasetProgress`).
    scorer : Scorer
        Freshly constructed scorer to load the counts into.

    Raises
    ------
    ValueError
        If the snapshot contains count arrays the scorer's configuration does
        not expect, i.e. it was saved under different `frame_eval` settings
        (e.g. `iou_thresholds` or `thresholds_for_thresholded_metrics`).
    """
    loaded = np.load(Path(checkpoint_dir) / state_file)

    expected = (
        {f"r::{sub}" for sub in scorer.result_counts}
        | {f"l::{sub}" for sub in scorer.localization_result_counts}
        | {f"t::{t}::{sub}" for t, counts in scorer.thresholded_counts.items() for sub in counts}
    )
    unexpected = sorted(set(loaded.files) - expected)
    if unexpected:
        raise ValueError(
            f"Scorer snapshot {state_file!r} contains count arrays {unexpected} that the current "
            "scorer configuration does not expect. It was likely saved with different frame_eval "
            "settings (e.g. iou_thresholds or thresholds_for_thresholded_metrics). Resume with the "
            "original eval config, or start a fresh run with a new --checkpoint-dir."
        )

    def _arr(key: str) -> np.ndarray | None:
        return loaded[key] if key in loaded else None

    state = {
        "result_counts": {sub: _arr(f"r::{sub}") for sub in scorer.result_counts},
        "localization_result_counts": {sub: _arr(f"l::{sub}") for sub in scorer.localization_result_counts},
        "thresholded_counts": {
            str(t): {sub: _arr(f"t::{t}::{sub}") for sub in counts} for t, counts in scorer.thresholded_counts.items()
        },
    }
    scorer.load_state(state)
