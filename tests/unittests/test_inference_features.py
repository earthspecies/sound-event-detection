"""Tests for the LSI feature stage (enrich selection tables with acoustic features)."""

import io

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from esp_research.protocols.detector import DetectorOutput
from sound_event_detection.inference.engine import read_shard, save_shard
from sound_event_detection.inference.features_v0minimal import FEATURE_COLS
from sound_event_detection.inference.lsi_features_cli import (
    _SourceAudioResolver,
    add_features_to_selection_table,
    cli,
)
from sound_event_detection.inference.result import ItemResult

_SR = 22050


def _preds(values: np.ndarray, class_names: list[str], frame_rate: float = 10.0) -> DetectorOutput:
    """Wrap a (T, C) probability array as a batch-1 DetectorOutput."""
    return DetectorOutput(
        predictions=values[np.newaxis].astype(np.float32), frame_rate=frame_rate, class_names=class_names
    )


def _tone(freq_hz: float, seconds: float, sr: int = _SR, amplitude: float = 0.5) -> np.ndarray:
    """A mono sine tone of the given frequency and duration."""
    t = np.arange(int(seconds * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _selection_table(events: list[tuple[float, float, str, float]]) -> str:
    """Serialize (begin, end, species, score) events as a postprocess-style TSV."""
    df = pd.DataFrame(events, columns=["Begin Time (s)", "End Time (s)", "Species", "Score"])
    return df.to_csv(sep="\t", index=False)


def _parse(tsv: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(tsv), sep="\t")


def test_enrich_appends_feature_columns_with_dominant_freq():
    # A 2 kHz tone over the whole recording; one event covering [0.1, 0.6] s.
    audio = _tone(2000.0, 1.0)
    tsv = _selection_table([(0.1, 0.6, "robin", 0.9)])
    enriched = _parse(add_features_to_selection_table(tsv, audio, _SR))

    # Original columns preserved, feature columns appended in order.
    assert list(enriched.columns) == ["Begin Time (s)", "End Time (s)", "Species", "Score", *FEATURE_COLS]
    assert len(enriched) == 1
    row = enriched.iloc[0]
    # duration_s comes from the span, not the clip length.
    assert abs(row["duration_s"] - 0.5) < 1e-6
    # The dominant frequency recovers the tone within one FFT bin (~21.5 Hz).
    assert abs(row["mean_dominant_freq_hz"] - 2000.0) < 50.0
    assert not np.isnan(row["rms_amplitude"])


def test_enrich_empty_table_gains_empty_feature_columns():
    # Header-only selection table (no events) -> header-only enriched table.
    tsv = _selection_table([])
    enriched = _parse(add_features_to_selection_table(tsv, _tone(2000.0, 1.0), _SR))
    assert list(enriched.columns) == ["Begin Time (s)", "End Time (s)", "Species", "Score", *FEATURE_COLS]
    assert len(enriched) == 0


def test_enrich_short_event_is_nan_except_duration():
    # An event shorter than n_fft (1024 samples ~= 0.046 s) -> spectral features NaN.
    tsv = _selection_table([(0.0, 0.01, "robin", 0.9)])
    enriched = _parse(add_features_to_selection_table(tsv, _tone(2000.0, 1.0), _SR))
    row = enriched.iloc[0]
    assert abs(row["duration_s"] - 0.01) < 1e-6
    assert np.isnan(row["mean_spectral_centroid_hz"])
    assert np.isnan(row["rms_amplitude"])


def test_enrich_clips_event_span_to_audio_bounds():
    # An event whose end runs past the audio still computes (span is clipped).
    tsv = _selection_table([(0.4, 5.0, "robin", 0.9)])  # audio is only 0.5 s long
    enriched = _parse(add_features_to_selection_table(tsv, _tone(2000.0, 0.5), _SR))
    row = enriched.iloc[0]
    # duration_s reflects the requested span; features come from the ~0.1 s slice.
    assert abs(row["duration_s"] - 4.6) < 1e-6
    assert not np.isnan(row["mean_dominant_freq_hz"])


def test_features_main_writes_1to1_enriched_shards(tmp_path):
    """End-to-end: the CLI pairs item + ST shards and writes enriched ST shards 1:1."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    st_dir = run_dir / "postprocessed_thr0.50"
    st_dir.mkdir()

    # Item shard: one recording with a denoised 2 kHz tone.
    preds = _preds(np.full((6, 1), 0.6, dtype=np.float32), ["robin"])
    item = ItemResult(preds=preds, denoised=_tone(2000.0, 1.0)).to_arrays()
    save_shard(str(run_dir / "shard_0000.npz"), [("a.wav", item)], 0)

    # Selection-table shard (1:1) with one event and pooled-quality scalars.
    st_entry = {
        "selection_table": np.array(_selection_table([(0.1, 0.6, "robin", 0.9)]), dtype=np.str_),
        "focal_confidence": np.asarray(np.float32(0.8)),
        "focal_max_stems": np.asarray(np.float32(2.0)),
    }
    save_shard(str(st_dir / "shard_0000.npz"), [("a.wav", st_entry)], 0)

    cfg_path = tmp_path / "feat.yml"
    cfg_path.write_text(
        yaml.dump({"input": {"run_dir": str(run_dir), "postprocessing": "postprocessed_thr0.50"}})
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(cfg_path)])
    assert result.exit_code == 0, result.output

    out_dir = st_dir / "features_v0minimal"
    assert (out_dir / "lineage.yaml").exists()
    assert (out_dir / "shard_0000.npz").exists()  # 1:1 with the ST/item shards
    out = read_shard(str(out_dir / "shard_0000.npz"))["a.wav"]
    enriched = _parse(out["selection_table"].item())
    assert list(enriched.columns)[-len(FEATURE_COLS):] == list(FEATURE_COLS)
    assert abs(enriched.iloc[0]["mean_dominant_freq_hz"] - 2000.0) < 50.0
    # Quality scalars are carried through so the features dir is an attach drop-in.
    assert float(out["focal_confidence"].item()) == np.float32(0.8)
    assert float(out["focal_max_stems"].item()) == 2.0

    # Idempotent: a second run skips the existing shard without error.
    result = runner.invoke(cli, ["--config", str(cfg_path)])
    assert result.exit_code == 0, result.output


class _FakeDataset:
    """Minimal stand-in for an `alp_data` dataset for the source-audio path.

    Mimics the two surfaces `_SourceAudioResolver` uses: a ``_data`` backend of
    undecoded rows (each carrying only the id column) that iterates without
    decoding audio, and integer indexing that yields a decoded item with
    ``audio`` / ``sample_rate``.
    """

    def __init__(self, id_to_audio: dict[str, np.ndarray], sr: int, id_column: str = "relative_path") -> None:
        self._originals_path_column = id_column
        self._id_column = id_column
        self._ids = list(id_to_audio)
        self._audio = id_to_audio
        self._sr = sr
        self._data = [{id_column: fid} for fid in self._ids]

    def __getitem__(self, index: int) -> dict:
        fid = self._ids[index]
        return {"audio": self._audio[fid], "sample_rate": self._sr, self._id_column: fid}

    def __len__(self) -> int:
        return len(self._ids)


def _write_preds_run(tmp_path) -> tuple:
    """Write a preds run (no denoised audio) + a 1:1 selection-table shard.

    Returns ``(run_dir, st_dir)`` for a one-recording ``a.wav`` with one event.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    st_dir = run_dir / "postprocessed_thr0.50"
    st_dir.mkdir()

    # Item shard with predictions only — a preds run stores no audio.
    item = ItemResult(preds=_preds(np.full((6, 1), 0.6, dtype=np.float32), ["robin"])).to_arrays()
    assert "denoised" not in item
    save_shard(str(run_dir / "shard_0000.npz"), [("a.wav", item)], 0)

    st_entry = {
        "selection_table": np.array(_selection_table([(0.1, 0.6, "robin", 0.9)]), dtype=np.str_),
        "focal_confidence": np.asarray(np.float32(0.8)),
        "focal_max_stems": np.asarray(np.float32(2.0)),
    }
    save_shard(str(st_dir / "shard_0000.npz"), [("a.wav", st_entry)], 0)
    return run_dir, st_dir


def test_source_resolver_maps_file_id_to_audio():
    # A backend scan builds the id->index map; indexing decodes just that row.
    audio = _tone(2000.0, 1.0)
    dataset = _FakeDataset({"a.wav": audio, "b.wav": _tone(1000.0, 0.5)}, _SR)
    got, sr = _SourceAudioResolver(dataset, "relative_path").resolve("a.wav")
    assert sr == float(_SR)
    np.testing.assert_array_equal(got, audio)


def test_source_resolver_unknown_id_raises_keyerror():
    dataset = _FakeDataset({"a.wav": _tone(2000.0, 1.0)}, _SR)
    with pytest.raises(KeyError):
        _SourceAudioResolver(dataset, "relative_path").resolve("missing.wav")


def test_features_main_reads_source_audio_for_preds_run(tmp_path, monkeypatch):
    """A preds run enriches from source audio re-read via the run's lineage."""
    run_dir, st_dir = _write_preds_run(tmp_path)

    # Run lineage names the (dummy) source dataset config + id column.
    (run_dir / "lineage.yaml").write_text(
        yaml.dump(
            {
                "stage": "run",
                "run_config": {"dataset": {"config": "dummy.yml", "id_column": "relative_path"}},
                "id_column": "relative_path",
            }
        )
    )
    # The source dataset re-reads a 2 kHz tone for "a.wav".
    fake = _FakeDataset({"a.wav": _tone(2000.0, 1.0)}, _SR)
    monkeypatch.setattr(
        "sound_event_detection.inference.lsi_features_cli.dataset_from_config",
        lambda _cfg: (fake, {}),
    )

    cfg_path = tmp_path / "feat.yml"
    cfg_path.write_text(yaml.dump({"input": {"run_dir": str(run_dir), "postprocessing": "postprocessed_thr0.50"}}))
    result = CliRunner().invoke(cli, ["--config", str(cfg_path)])
    assert result.exit_code == 0, result.output

    out = read_shard(str(st_dir / "features_v0minimal" / "shard_0000.npz"))["a.wav"]
    enriched = _parse(out["selection_table"].item())
    assert list(enriched.columns)[-len(FEATURE_COLS):] == list(FEATURE_COLS)
    assert abs(enriched.iloc[0]["mean_dominant_freq_hz"] - 2000.0) < 50.0
    # Quality scalars are still carried through from the ST shard.
    assert float(out["focal_confidence"].item()) == np.float32(0.8)
    assert float(out["focal_max_stems"].item()) == 2.0


def test_features_main_preds_run_uses_dataset_override(tmp_path, monkeypatch):
    """With no lineage, an explicit input.dataset override drives the source path."""
    run_dir, st_dir = _write_preds_run(tmp_path)  # note: no lineage.yaml written
    fake = _FakeDataset({"a.wav": _tone(2000.0, 1.0)}, _SR)
    monkeypatch.setattr(
        "sound_event_detection.inference.lsi_features_cli.dataset_from_config",
        lambda _cfg: (fake, {}),
    )

    cfg_path = tmp_path / "feat.yml"
    cfg_path.write_text(
        yaml.dump(
            {
                "input": {
                    "run_dir": str(run_dir),
                    "postprocessing": "postprocessed_thr0.50",
                    "dataset": {"config": "dummy.yml", "id_column": "relative_path"},
                }
            }
        )
    )
    result = CliRunner().invoke(cli, ["--config", str(cfg_path)])
    assert result.exit_code == 0, result.output

    out = read_shard(str(st_dir / "features_v0minimal" / "shard_0000.npz"))["a.wav"]
    enriched = _parse(out["selection_table"].item())
    assert abs(enriched.iloc[0]["mean_dominant_freq_hz"] - 2000.0) < 50.0


def test_features_main_preds_run_without_dataset_fails_fast(tmp_path):
    """A preds run with neither lineage nor override fails fast with a clear error."""
    run_dir, _ = _write_preds_run(tmp_path)  # no lineage.yaml, no input.dataset
    cfg_path = tmp_path / "feat.yml"
    cfg_path.write_text(yaml.dump({"input": {"run_dir": str(run_dir), "postprocessing": "postprocessed_thr0.50"}}))
    result = CliRunner().invoke(cli, ["--config", str(cfg_path)])
    assert result.exit_code != 0
    # click prints a UsageError to stderr; depending on the click version that is
    # merged into `output` or kept separate, so check both.
    combined = result.output
    try:
        combined += result.stderr
    except ValueError:
        pass  # stderr was merged into output (older click)
    assert "source dataset could not be located" in combined
