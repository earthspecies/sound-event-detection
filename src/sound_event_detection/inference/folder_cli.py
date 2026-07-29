"""Run BirdCODE over a folder of audio files -> per-recording selection tables.

Entry point: ``uv run sed-folder``.

A self-contained, server-free alternative to the LSI pipeline for the common
case of "I have a folder of recordings, give me BirdCODE detections". It loads
the detector in-process (from the HuggingFace Hub by default), runs it over every
audio file under a folder — resampling each to the model's rate (32 kHz) as
needed — postprocesses (threshold 0.5, merge
gap 1.0 s, NMS IoU 0.8, optionally geography-filtered), and writes each
recording's selection table next to it: for ``.../x.wav`` it writes
``.../BirdCODE_predictions/x.txt`` (a tab-separated selection table).

The event math and geography filter are reused verbatim from the LSI postprocess
stage (`postprocess_and_convert_detector_output_to_selection_table`), so a table
written here is identical to what the ``run -> postprocess`` pipeline would
produce for the same audio and params.

Geography filtering is off by default and enabled with ``--geo-filter``. Because
a bare folder carries no per-recording metadata, one ``--latitude``/``--longitude``
pair is applied to every file (e.g. all recordings from one site). It reuses the
LSI filter's fail-open semantics: a detection is dropped only when the location
is known **and** the species' range map excludes it, so without coordinates the
filter is a no-op.

Usage::

    # BirdCODE from the Hub over a folder, default postprocessing:
    uv run sed-folder --folder /path/to/audio

    # with geography filtering for a known site:
    uv run sed-folder --folder /path/to/audio \\
        --geo-filter --range-map-dir geography/range_maps \\
        --latitude 42.5 --longitude -72.2
"""

from pathlib import Path

import click
import librosa
import numpy as np
import torch
from alp_data.io import audio_stereo_to_mono, read_audio

from esp_research.logging import logger
from sound_event_detection.inference.lsi_postprocess_cli import (
    _dir_exists,
    _list_range_map_files,
    _load_range_maps,
    postprocess_and_convert_detector_output_to_selection_table,
)
from sound_event_detection.models import FrameDetector

__all__ = ["cli"]

#: Default detector checkpoint pulled from the HuggingFace Hub.
DEFAULT_HF_REPO_ID = "EarthSpeciesProject/sed-birdcode"

#: Folder (created next to each recording) that receives its selection table.
OUTPUT_DIR_NAME = "BirdCODE_predictions"

#: Audio extensions scanned for (matches `alp_data.io.read_audio` support).
_AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3")


def _find_audio_files(folder: Path, recursive: bool) -> list[Path]:
    """Collect the audio files under `folder`.

    Parameters
    ----------
    folder : Path
        Directory to scan.
    recursive : bool
        When ``True`` descend into subdirectories; otherwise only the top level.

    Returns
    -------
    list[Path]
        Sorted paths of files whose suffix is a supported audio extension.
    """
    globber = folder.rglob if recursive else folder.glob
    files = [path for path in globber("*") if path.is_file() and path.suffix.lower() in _AUDIO_EXTENSIONS]
    return sorted(files)


def _load_audio_resampled(path: Path, target_sr: int) -> np.ndarray:
    """Load one recording as mono float32 audio at `target_sr`.

    Reads the file (any format `read_audio` supports), averages channels to mono,
    and resamples to `target_sr` using the same ``kaiser_best`` method the rest of
    the codebase uses.

    Parameters
    ----------
    path : Path
        Audio file to read.
    target_sr : int
        Sample rate to resample to (the model's expected rate).

    Returns
    -------
    np.ndarray
        Mono waveform of shape ``(samples,)`` in float32 at `target_sr`.
    """
    audio, sr = read_audio(path)
    audio = audio_stereo_to_mono(np.asarray(audio, dtype=np.float32), mono_method="average")
    if sr != target_sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr, scale=True, res_type="kaiser_best")
    return np.ascontiguousarray(audio, dtype=np.float32)


def _run_on_folder(
    *,
    folder: Path,
    recursive: bool,
    hf_repo_id: str,
    revision: str | None,
    device: str,
    overlap: float,
    threshold: float,
    merge_max_gap: float,
    nms_iou: float,
    geo_filter: bool,
    range_map_dir: str,
    latitude: float,
    longitude: float,
    overwrite: bool,
) -> None:
    """Run the detector over every audio file under `folder` and write tables.

    Parameters
    ----------
    folder : Path
        Root directory of audio files to process.
    recursive : bool
        Whether to descend into subdirectories.
    hf_repo_id : str
        HuggingFace Hub repo id of the detector checkpoint to load.
    revision : str or None
        Optional Hub revision (branch, tag, or commit) to pin.
    device : str
        Torch device to run inference on (e.g. ``"cuda"`` or ``"cpu"``).
    overlap : float
        Window overlap fraction forwarded to `FrameDetector.run` (``0`` disables).
    threshold : float
        Binarization threshold for event extraction.
    merge_max_gap : float
        Merge same-species events separated by at most this many seconds
        (``0`` disables).
    nms_iou : float
        IoU threshold for cross-class non-maximum suppression.
    geo_filter : bool
        Whether to drop out-of-range species using range maps.
    range_map_dir : str
        Directory of ``*.gpkg`` range maps (used only when `geo_filter`).
    latitude : float
        Latitude (WGS84) applied to every recording for geography filtering;
        ``nan`` if unknown.
    longitude : float
        Longitude (WGS84) applied to every recording; ``nan`` if unknown.
    overwrite : bool
        When ``False`` skip a recording whose selection table already exists.

    Raises
    ------
    click.UsageError
        If `folder` holds no audio files, or geography filtering is enabled but
        `range_map_dir` does not exist or holds no ``*.gpkg`` range maps.
    """
    audio_files = _find_audio_files(folder, recursive)
    if not audio_files:
        raise click.UsageError(f"no audio files ({', '.join(_AUDIO_EXTENSIONS)}) found under {str(folder)!r}")

    # Resolve and validate the range-map directory up front so a misconfigured
    # geography filter fails before any (slow) model load or inference.
    range_map_gdf = None
    if geo_filter:
        if not _dir_exists(range_map_dir):
            raise click.UsageError(f"--range-map-dir does not exist: {range_map_dir!r}")
        range_map_files = _list_range_map_files(range_map_dir)
        if not range_map_files:
            raise click.UsageError(f"no '*.gpkg' range maps found in --range-map-dir: {range_map_dir!r}")
        if np.isnan(latitude) or np.isnan(longitude):
            logger.warning(
                "--geo-filter is on but no --latitude/--longitude given; the filter fails open and drops nothing."
            )
        range_map_gdf = _load_range_maps(range_map_files)

    logger.info(f"loading detector {hf_repo_id!r} (revision={revision}) onto {device}...")
    model = FrameDetector.from_hf_hub(hf_repo_id, revision=revision).eval().to(device)
    target_sr = model.sample_rate

    # Only the enabled steps go into the chain, matching the LSI postprocess CLI.
    pp_config: dict = {}
    if merge_max_gap > 0:
        pp_config["merge_max_gap"] = merge_max_gap
    if nms_iou is not None:
        pp_config["nms"] = {"iou_threshold": nms_iou}

    logger.info(
        f"{len(audio_files)} files | resample->{target_sr}Hz overlap={overlap} "
        f"threshold={threshold} pp={pp_config} geo_filter={geo_filter}"
    )

    done = 0
    errors = 0
    for path in audio_files:
        out_dir = path.parent / OUTPUT_DIR_NAME
        out_path = out_dir / f"{path.stem}.txt"
        if out_path.exists() and not overwrite:
            logger.info(f"skip existing {out_path}")
            done += 1
            continue
        try:
            audio = _load_audio_resampled(path, target_sr)
            preds = model.run(audio[None, :], overlap=overlap)
            tsv = postprocess_and_convert_detector_output_to_selection_table(
                preds,
                threshold,
                pp_config,
                range_map_gdf=range_map_gdf,
                latitude=latitude,
                longitude=longitude,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(tsv)
            done += 1
            logger.info(f"[{done}/{len(audio_files)}] {path.name} -> {out_path}")
        except Exception as exc:  # noqa: BLE001 — isolate one bad file, keep the batch going
            errors += 1
            logger.error(f"error on {path}: {exc}")

    logger.info(f"done: {done} tables written, {errors} errors")


@click.command()
@click.option(
    "--folder",
    "folder",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory of audio files to run BirdCODE over.",
)
@click.option("--recursive/--no-recursive", default=True, help="Scan subdirectories too (default: recursive).")
@click.option("--hf-repo-id", default=DEFAULT_HF_REPO_ID, show_default=True, help="HuggingFace Hub repo id to load.")
@click.option("--revision", default=None, help="Optional Hub revision (branch, tag, or commit) to pin.")
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    show_default=True,
    help="Torch device for inference.",
)
@click.option("--overlap", default=0.5, show_default=True, help="Window overlap fraction (0 disables).")
@click.option("--threshold", default=0.5, show_default=True, help="Detection threshold.")
@click.option("--merge-max-gap", default=1.0, show_default=True, help="Merge same-species events within this gap (s).")
@click.option("--nms-iou", default=0.8, show_default=True, help="IoU threshold for cross-class NMS.")
@click.option("--geo-filter", is_flag=True, default=False, help="Drop out-of-range species using range maps.")
@click.option(
    "--range-map-dir",
    default="geography/range_maps",
    show_default=True,
    help="Directory of *.gpkg range maps (used with --geo-filter).",
)
@click.option("--latitude", default=None, type=float, help="Latitude (WGS84) applied to all files for --geo-filter.")
@click.option("--longitude", default=None, type=float, help="Longitude (WGS84) applied to all files for --geo-filter.")
@click.option("--overwrite", is_flag=True, default=False, help="Rewrite selection tables that already exist.")
def cli(
    folder: Path,
    recursive: bool,
    hf_repo_id: str,
    revision: str | None,
    device: str,
    overlap: float,
    threshold: float,
    merge_max_gap: float,
    nms_iou: float,
    geo_filter: bool,
    range_map_dir: str,
    latitude: float | None,
    longitude: float | None,
    overwrite: bool,
) -> None:
    """Run BirdCODE over a folder of audio and write per-recording selection tables.

    Writes ``<dir>/BirdCODE_predictions/<name>.txt`` next to each ``<dir>/<name>.<ext>``.
    Thin CLI wrapper over `_run_on_folder` (which validates inputs and does the work).
    """
    _run_on_folder(
        folder=folder,
        recursive=recursive,
        hf_repo_id=hf_repo_id,
        revision=revision,
        device=device,
        overlap=overlap,
        threshold=threshold,
        merge_max_gap=merge_max_gap,
        nms_iou=nms_iou,
        geo_filter=geo_filter,
        range_map_dir=range_map_dir,
        latitude=float("nan") if latitude is None else latitude,
        longitude=float("nan") if longitude is None else longitude,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    cli()
