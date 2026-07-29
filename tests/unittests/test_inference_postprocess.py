"""Tests for the LSI postprocessing stage (predictions -> selection-table TSV)."""

import io

import numpy as np
import pandas as pd

from esp_research.protocols.detector import DetectorOutput
from sound_event_detection.inference.lsi_postprocess_cli import (
    _combined_preds,
    _output_dir_name,
    _pool_quality,
    filter_by_geography,
    postprocess_and_convert_detector_output_to_selection_table,
)
from sound_event_detection.inference.result import ItemResult


def _preds(values: np.ndarray, class_names: list[str], frame_rate: float = 10.0) -> DetectorOutput:
    """Wrap a (T, C) probability array as a batch-1 DetectorOutput."""
    return DetectorOutput(predictions=values[np.newaxis].astype(np.float32), frame_rate=frame_rate, class_names=class_names)


def _parse(tsv: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(tsv), sep="\t")


def test_selection_table_extracts_event_with_score():
    # robin is above 0.5 for frames 2,3,4; wren never is.
    values = np.full((6, 2), 0.1, dtype=np.float32)
    values[2:5, 0] = 0.9
    tsv = postprocess_and_convert_detector_output_to_selection_table(_preds(values, ["robin", "wren"]), threshold=0.5, pp_config={})
    df = _parse(tsv)

    assert list(df.columns) == ["Begin Time (s)", "End Time (s)", "Species", "Score"]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["Species"] == "robin"
    assert row["Begin Time (s)"] == 0.2 and row["End Time (s)"] == 0.5  # frames 2..5 at 10 Hz
    assert abs(row["Score"] - 0.9) < 1e-5


def test_empty_predictions_yield_header_only_with_score_column():
    # Nothing crosses the threshold -> header-only table, still with a Score column.
    values = np.full((4, 2), 0.1, dtype=np.float32)
    tsv = postprocess_and_convert_detector_output_to_selection_table(_preds(values, ["robin", "wren"]), threshold=0.5, pp_config={})
    df = _parse(tsv)
    assert list(df.columns) == ["Begin Time (s)", "End Time (s)", "Species", "Score"]
    assert len(df) == 0


def test_zero_class_predictions_yield_header_only():
    # A recording where the codec dropped every class (C == 0) must not crash.
    values = np.zeros((5, 0), dtype=np.float32)
    tsv = postprocess_and_convert_detector_output_to_selection_table(_preds(values, []), threshold=0.5, pp_config={"nms": {"iou_threshold": 0.5}})
    df = _parse(tsv)
    assert len(df) == 0


def test_min_event_duration_filters_short_events():
    values = np.full((6, 1), 0.1, dtype=np.float32)
    values[2:3, 0] = 0.9  # a single-frame (0.1 s) event
    without = _parse(postprocess_and_convert_detector_output_to_selection_table(_preds(values, ["robin"]), threshold=0.5, pp_config={}))
    with_filter = _parse(
        postprocess_and_convert_detector_output_to_selection_table(_preds(values, ["robin"]), threshold=0.5, pp_config={"min_event_duration": 0.5})
    )
    assert len(without) == 1
    assert len(with_filter) == 0


def test_combined_preds_reads_only_prediction_group():
    values = np.full((5, 2), 0.6, dtype=np.float32)
    arrays = ItemResult(preds=_preds(values, ["a", "b"])).to_arrays()
    decoded = _combined_preds(arrays)
    assert decoded.class_names == ["a", "b"]
    assert decoded.predictions.shape == (1, 5, 2)
    assert decoded.frame_rate == 10.0


def test_output_dir_name_only_lists_enabled_steps():
    assert _output_dir_name(0.5, {}) == "postprocessed_thr0.50"
    summary = _output_dir_name(
        0.5, {"merge_max_gap": 1.0, "min_event_duration": 0.01, "nms": {"iou_threshold": 0.8}}
    )
    assert summary == "postprocessed_thr0.50_merge1.00_minDur0.01_nms0.80"


def test_output_dir_name_appends_geo_token_when_filtering():
    # A geography-filtered run lands in its own sibling directory.
    assert _output_dir_name(0.5, {}, geo_filter=True) == "postprocessed_thr0.50_geo"
    assert (
        _output_dir_name(0.5, {"nms": {"iou_threshold": 0.8}}, geo_filter=True)
        == "postprocessed_thr0.50_nms0.80_geo"
    )


def test_allowed_classes_drops_other_species():
    # robin and wren both clear threshold; the allowlist keeps only robin.
    values = np.full((6, 2), 0.1, dtype=np.float32)
    values[2:5, 0] = 0.9  # robin
    values[2:5, 1] = 0.95  # wren, higher score
    kept = _parse(
        postprocess_and_convert_detector_output_to_selection_table(_preds(values, ["robin", "wren"]), threshold=0.5, pp_config={}, allowed_classes={"robin"})
    )
    assert set(kept["Species"]) == {"robin"}
    # Sanity: without the allowlist, both survive (so the allowlist is what dropped wren).
    both = _parse(postprocess_and_convert_detector_output_to_selection_table(_preds(values, ["robin", "wren"]), threshold=0.5, pp_config={}))
    assert set(both["Species"]) == {"robin", "wren"}


def test_allowed_classes_filter_runs_before_nms():
    # robin and wren fully overlap; wren scores higher, so cross-class NMS would
    # keep wren. Filtering wren out FIRST (allowlist) lets robin survive NMS.
    values = np.full((6, 2), 0.1, dtype=np.float32)
    values[2:5, 0] = 0.80  # robin
    values[2:5, 1] = 0.95  # wren, overlapping and higher
    pp = {"nms": {"iou_threshold": 0.1}}
    only_robin = _parse(
        postprocess_and_convert_detector_output_to_selection_table(_preds(values, ["robin", "wren"]), threshold=0.5, pp_config=pp, allowed_classes={"robin"})
    )
    # wren was removed before postprocessing, so it cannot suppress robin in NMS.
    assert set(only_robin["Species"]) == {"robin"}


def test_pool_quality_summarizes_focal_tracks():
    # Detected on frames 0, 2, 3 (nan where the focal species was not gated in).
    detprob = np.array([0.9, np.nan, 0.5, 0.7], dtype=np.float32)
    nstems = np.array([2.0, 0.0, 1.0, 3.0], dtype=np.float32)
    pooled = _pool_quality({"focal_detprob": detprob, "focal_nstems": nstems})
    assert set(pooled) == {"focal_confidence", "focal_max_stems"}
    assert abs(float(pooled["focal_confidence"]) - float(np.nanmean(detprob))) < 1e-6
    assert float(pooled["focal_max_stems"]) == 3.0


def test_pool_quality_all_nan_confidence_is_nan():
    pooled = _pool_quality(
        {
            "focal_detprob": np.array([np.nan, np.nan], dtype=np.float32),
            "focal_nstems": np.array([0.0, 0.0], dtype=np.float32),
        }
    )
    assert np.isnan(float(pooled["focal_confidence"]))
    assert float(pooled["focal_max_stems"]) == 0.0


def test_pool_quality_empty_for_preds_run():
    # A preds run stores no focal tracks, so nothing is pooled.
    values = np.full((5, 2), 0.6, dtype=np.float32)
    arrays = ItemResult(preds=_preds(values, ["a", "b"])).to_arrays()
    assert _pool_quality(arrays) == {}


def test_postprocess_main_writes_1to1_st_shards(tmp_path):
    """End-to-end: the CLI turns items shards into 1:1 selection-table shards."""
    import yaml
    from click.testing import CliRunner

    from sound_event_detection.inference.engine import read_shard, save_shard
    from sound_event_detection.inference.lsi_postprocess_cli import cli

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with_event = np.full((6, 1), 0.1, dtype=np.float32)
    with_event[2:5, 0] = 0.9
    without = np.full((6, 1), 0.1, dtype=np.float32)
    save_shard(str(run_dir / "shard_0000.npz"), [("a.wav", ItemResult(preds=_preds(with_event, ["robin"])).to_arrays())], 0)
    save_shard(str(run_dir / "shard_0001.npz"), [("b.wav", ItemResult(preds=_preds(without, ["robin"])).to_arrays())], 0)

    cfg_path = tmp_path / "pp.yml"
    cfg_path.write_text(yaml.dump({"input": {"run_dir": str(run_dir)}, "postprocessing": {"threshold": 0.5}}))
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(cfg_path)])
    assert result.exit_code == 0, result.output

    pp_dir = run_dir / "postprocessed_thr0.50"
    assert (pp_dir / "lineage.yaml").exists()
    # 1:1 shard naming with the items shards.
    assert (pp_dir / "shard_0000.npz").exists() and (pp_dir / "shard_0001.npz").exists()
    got_a = read_shard(str(pp_dir / "shard_0000.npz"))["a.wav"]["selection_table"].item()
    got_b = read_shard(str(pp_dir / "shard_0001.npz"))["b.wav"]["selection_table"].item()
    assert "robin" in got_a
    assert got_b.strip("\n").count("\n") == 0  # header only, no events

    # Idempotent: a second run skips existing shards without error.
    result = runner.invoke(cli, ["--config", str(cfg_path)])
    assert result.exit_code == 0, result.output


def test_postprocess_main_pools_quality_by_presence(tmp_path):
    """A denoising item's focal tracks are pooled into ST-shard scalars; a preds item's are not."""
    import yaml
    from click.testing import CliRunner

    from sound_event_detection.inference.engine import read_shard, save_shard
    from sound_event_detection.inference.lsi_postprocess_cli import cli

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    values = np.full((6, 1), 0.1, dtype=np.float32)
    values[2:5, 0] = 0.9
    detprob = np.array([np.nan, np.nan, 0.9, 0.8, 0.7, np.nan], dtype=np.float32)
    nstems = np.array([0.0, 0.0, 2.0, 1.0, 3.0, 0.0], dtype=np.float32)
    denoised_item = ItemResult(
        preds=_preds(values, ["robin"]), focal_detprob=detprob, focal_nstems=nstems
    ).to_arrays()
    preds_item = ItemResult(preds=_preds(values, ["robin"])).to_arrays()
    save_shard(str(run_dir / "shard_0000.npz"), [("d.wav", denoised_item), ("p.wav", preds_item)], 0)

    cfg_path = tmp_path / "pp.yml"
    cfg_path.write_text(yaml.dump({"input": {"run_dir": str(run_dir)}, "postprocessing": {"threshold": 0.5}}))
    result = CliRunner().invoke(cli, ["--config", str(cfg_path)])
    assert result.exit_code == 0, result.output

    out = read_shard(str(run_dir / "postprocessed_thr0.50" / "shard_0000.npz"))
    # Denoising item: pooled scalars present and correct.
    assert "focal_confidence" in out["d.wav"] and "focal_max_stems" in out["d.wav"]
    assert abs(float(out["d.wav"]["focal_confidence"].item()) - float(np.nanmean(detprob))) < 1e-6
    assert float(out["d.wav"]["focal_max_stems"].item()) == 3.0
    # Preds item: no quality keys attached.
    assert "focal_confidence" not in out["p.wav"] and "focal_max_stems" not in out["p.wav"]


def _fake_range_maps():
    """A two-species range map: ``in_sp`` covers (10, 10); ``out_sp`` is far away.

    Returns
    -------
    geopandas.GeoDataFrame
        Range maps with a ``gbif_name`` column and polygon geometry (WGS84).
    """
    import geopandas as gpd
    from shapely.geometry import Polygon

    return gpd.GeoDataFrame(
        {"gbif_name": ["in_sp", "out_sp"]},
        geometry=[
            Polygon([(0, 0), (0, 20), (20, 20), (20, 0)]),
            Polygon([(50, 50), (50, 60), (60, 60), (60, 50)]),
        ],
        crs="EPSG:4326",
    )


def test_geo_filter_drops_only_out_of_range_species():
    # 'ghost_sp' has no range map; it must be kept (missing data is not evidence).
    table = pd.DataFrame({"Species": ["in_sp", "out_sp", "ghost_sp"], "Begin Time (s)": [1.0, 2.0, 3.0]})
    filtered = filter_by_geography(table, _fake_range_maps(), latitude=10.0, longitude=10.0)
    assert sorted(filtered["Species"]) == ["ghost_sp", "in_sp"]


def test_geo_filter_skips_when_no_coordinates():
    # No lat/long -> fail open: the table is returned unchanged.
    table = pd.DataFrame({"Species": ["in_sp", "out_sp"], "Begin Time (s)": [1.0, 2.0]})
    filtered = filter_by_geography(table, _fake_range_maps(), latitude=float("nan"), longitude=10.0)
    assert sorted(filtered["Species"]) == ["in_sp", "out_sp"]


def test_geo_filter_empty_table_is_unchanged():
    table = pd.DataFrame({"Species": [], "Begin Time (s)": []})
    filtered = filter_by_geography(table, _fake_range_maps(), latitude=10.0, longitude=10.0)
    assert filtered.empty
