"""Unit tests for sound event detection dataset transforms."""

import polars as pl
import pytest
from alp_data.backends import PolarsBackend

from sound_event_detection.data.transforms import CapPerGroup, SpeciesListFromColumn


class FakeBackend:
    """Minimal DataBackend stand-in backed by a list of row dicts."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    @property
    def columns(self) -> list[str]:
        return list(self._rows[0].keys()) if self._rows else []

    def __iter__(self):
        return iter(self._rows)

    def add_column(self, name: str, values: list) -> "FakeBackend":
        for row, value in zip(self._rows, values, strict=True):
            row[name] = value
        return self


def _run(rows: list[dict], **kwargs) -> list[list[str]]:
    backend, meta = SpeciesListFromColumn(**kwargs)(FakeBackend(rows))
    assert meta == {}
    return [row["species_list"] for row in backend]


def test_parses_json_list_strings_and_sorts_unique():
    rows = [
        {"canonical_name_multispecies": '["Turdus migratorius", "Cardinalis cardinalis", "Turdus migratorius"]'},
        {"canonical_name_multispecies": '["Turdus migratorius"]'},
    ]
    out = _run(rows)
    assert out[0] == ["Cardinalis cardinalis", "Turdus migratorius"]
    assert out[1] == ["Turdus migratorius"]


def test_accepts_actual_lists():
    rows = [{"canonical_name_multispecies": ["b", "a"]}]
    assert _run(rows) == [["a", "b"]]


def test_empty_and_invalid_values_become_empty_lists():
    rows = [
        {"canonical_name_multispecies": ""},
        {"canonical_name_multispecies": None},
        {"canonical_name_multispecies": "not json"},
    ]
    assert _run(rows) == [[], [], []]


def test_custom_columns():
    rows = [{"raw": '["x"]'}]
    out = _run(rows, input_column="raw", output_column="species_list")
    assert out == [["x"]]


def test_missing_input_column_raises_keyerror():
    rows = [{"other": "value"}]
    with pytest.raises(KeyError):
        SpeciesListFromColumn()(FakeBackend(rows))


def test_cap_per_group_caps_large_and_leaves_small():
    df = pl.DataFrame({"sp": ["a"] * 35 + ["b"] * 22 + ["c"] * 8, "x": list(range(65))})
    capped, meta = CapPerGroup(property="sp", count=20)(PolarsBackend(df))
    assert meta == {}
    assert capped.histogram("sp") == {"a": 20, "b": 20, "c": 8}


def test_cap_per_group_missing_property_raises_keyerror():
    df = pl.DataFrame({"other": [1, 2, 3]})
    with pytest.raises(KeyError):
        CapPerGroup(property="sp", count=5)(PolarsBackend(df))


def _write_shard(path, items):
    """Write a shard of (file_id, {key: array}) items via the engine framing."""
    from sound_event_detection.inference.engine import save_shard

    save_shard(str(path), items, job_index=0)


def _st(species_rows):
    """Build a selection-table TSV string with the standard header."""
    lines = ["Begin Time (s)\tEnd Time (s)\tSpecies\tScore"]
    for begin, end, sp, score in species_rows:
        lines.append(f"{begin}\t{end}\t{sp}\t{score}")
    return "\n".join(lines) + "\n"


def _lsi_run(tmp_path):
    """Lay out a minimal LSI run: items shards + 1:1 postprocessed ST shards.

    Returns (run_root, postprocessing_subdir_name).
    """
    import numpy as np

    from sound_event_detection.data.transforms import AttachLSISelectionTables  # noqa: F401 (ensures import path)

    pp = "postprocessed_thr0.50_merge1.00"
    (tmp_path / pp).mkdir()
    # Heavy items shards (contents irrelevant here; only the files must exist as pointer targets).
    _write_shard(tmp_path / "shard_0000.npz", [("a.wav", {"x": np.array(0)}), ("b.wav", {"x": np.array(0)})])
    _write_shard(tmp_path / "shard_0001.npz", [("c.wav", {"x": np.array(0)})])
    # Light ST shards, 1:1 with the items shards (same index).
    _write_shard(
        tmp_path / pp / "shard_0000.npz",
        [
            ("a.wav", {"selection_table": np.array(_st([(0.0, 1.0, "robin", 0.9)]), dtype=np.str_)}),
            ("b.wav", {"selection_table": np.array(_st([]), dtype=np.str_)}),  # empty (header only)
        ],
    )
    _write_shard(
        tmp_path / pp / "shard_0001.npz",
        [("c.wav", {"selection_table": np.array(_st([(2.0, 3.0, "wren", 0.7)]), dtype=np.str_)})],
    )
    return str(tmp_path), pp


def test_attach_lsi_selection_tables_matches_and_points_to_items_shard(tmp_path):
    from sound_event_detection.data.transforms import AttachLSISelectionTables

    run_root, pp = _lsi_run(tmp_path)
    rows = [
        {"audio_path": "a.wav", "canonical_name": "robin"},
        {"audio_path": "c.wav", "canonical_name": "wren"},
        {"audio_path": "z.wav", "canonical_name": "crow"},  # no LSI output
    ]
    backend, meta = AttachLSISelectionTables(run_root=run_root, postprocessing=pp)(FakeBackend(rows))
    out = list(backend)

    assert meta == {"matched": 2, "unmatched": 1}
    # selection_table is the inline TSV; focal species is NOT attached (comes from the join).
    assert "robin" in out[0]["selection_table"]
    assert "lsi_shard" in out[0] and "selection_table" in out[0]
    # lsi_shard points at the 1:1-indexed *items* shard, not the ST shard.
    assert out[0]["lsi_shard"].endswith("shard_0000.npz") and pp not in out[0]["lsi_shard"]
    assert out[1]["lsi_shard"].endswith("shard_0001.npz")
    # Unmatched row gets empty strings for both columns.
    assert out[2]["selection_table"] == "" and out[2]["lsi_shard"] == ""


def test_attach_lsi_uses_explicit_id_column(tmp_path):
    from sound_event_detection.data.transforms import AttachLSISelectionTables

    run_root, pp = _lsi_run(tmp_path)
    rows = [{"my_id": "a.wav"}, {"my_id": "b.wav"}]
    backend, meta = AttachLSISelectionTables(run_root=run_root, postprocessing=pp, id_column="my_id")(FakeBackend(rows))
    # Both are found in the LSI output, so both are "matched" — a header-only table
    # (b.wav, zero events) is still a non-empty string, i.e. a found recording.
    assert meta == {"matched": 2, "unmatched": 0}


def test_attach_lsi_zero_matches_raises(tmp_path):
    # A valid, resolvable id column whose values match no produced recording signals
    # a broken join (wrong run / mismatched id_column), not partial coverage.
    from sound_event_detection.data.transforms import AttachLSISelectionTables

    run_root, pp = _lsi_run(tmp_path)
    rows = [{"audio_path": "nope.wav"}, {"audio_path": "nope2.wav"}]
    with pytest.raises(ValueError, match="matched 0 of 2 rows"):
        AttachLSISelectionTables(run_root=run_root, postprocessing=pp)(FakeBackend(rows))


def test_attach_lsi_partial_match_warns(tmp_path):
    from sound_event_detection.data.transforms import AttachLSISelectionTables

    run_root, pp = _lsi_run(tmp_path)
    rows = [{"audio_path": "a.wav"}, {"audio_path": "z.wav"}]  # z.wav unmatched
    with pytest.warns(UserWarning, match="1 of 2 rows found no LSI output"):
        backend, meta = AttachLSISelectionTables(run_root=run_root, postprocessing=pp)(FakeBackend(rows))
    assert meta == {"matched": 1, "unmatched": 1}


def test_attach_lsi_unknown_explicit_id_column_raises(tmp_path):
    from sound_event_detection.data.transforms import AttachLSISelectionTables

    run_root, pp = _lsi_run(tmp_path)
    with pytest.raises(KeyError):
        AttachLSISelectionTables(run_root=run_root, postprocessing=pp, id_column="missing")(FakeBackend([{"x": 1}]))


def test_attach_lsi_uninferrable_id_column_raises(tmp_path):
    from sound_event_detection.data.transforms import AttachLSISelectionTables

    run_root, pp = _lsi_run(tmp_path)
    with pytest.raises(ValueError, match="could not infer id_column"):
        AttachLSISelectionTables(run_root=run_root, postprocessing=pp)(FakeBackend([{"weird_col": "a.wav"}]))


def test_attach_lsi_missing_postprocessed_dir_raises(tmp_path):
    from sound_event_detection.data.transforms import AttachLSISelectionTables

    with pytest.raises(ValueError, match="no shard_.*found"):
        AttachLSISelectionTables(run_root=str(tmp_path), postprocessing="nonexistent")(
            FakeBackend([{"audio_path": "a.wav"}])
        )


def _lsi_run_with_quality(tmp_path):
    """Like `_lsi_run` but the ST shards also carry pooled focal-quality scalars.

    Returns (run_root, postprocessing_subdir_name).
    """
    import numpy as np

    pp = "postprocessed_thr0.50"
    (tmp_path / pp).mkdir()
    _write_shard(tmp_path / "shard_0000.npz", [("a.wav", {"x": np.array(0)}), ("b.wav", {"x": np.array(0)})])
    _write_shard(
        tmp_path / pp / "shard_0000.npz",
        [
            (
                "a.wav",
                {
                    "selection_table": np.array(_st([(0.0, 1.0, "robin", 0.9)]), dtype=np.str_),
                    "focal_confidence": np.asarray(np.float32(0.8)),
                    "focal_max_stems": np.asarray(np.float32(3.0)),
                },
            ),
            (
                "b.wav",
                {
                    "selection_table": np.array(_st([]), dtype=np.str_),
                    "focal_confidence": np.asarray(np.float32("nan")),
                    "focal_max_stems": np.asarray(np.float32(0.0)),
                },
            ),
        ],
    )
    return str(tmp_path), pp


def test_attach_lsi_attaches_quality_columns_when_present(tmp_path):
    import numpy as np

    from sound_event_detection.data.transforms import AttachLSISelectionTables

    run_root, pp = _lsi_run_with_quality(tmp_path)
    rows = [{"audio_path": "a.wav"}, {"audio_path": "z.wav"}]  # z.wav unmatched
    backend, meta = AttachLSISelectionTables(run_root=run_root, postprocessing=pp)(FakeBackend(rows))
    out = list(backend)

    assert meta.get("quality_attached") is True
    assert abs(out[0]["focal_confidence"] - 0.8) < 1e-6
    assert out[0]["focal_max_stems"] == 3.0
    # Unmatched row gets nan for both quality columns.
    assert np.isnan(out[1]["focal_confidence"]) and np.isnan(out[1]["focal_max_stems"])


def test_attach_lsi_skips_quality_when_shards_lack_it(tmp_path):
    # `_lsi_run` writes ST shards with no quality scalars (a preds run) -> not attached.
    from sound_event_detection.data.transforms import AttachLSISelectionTables

    run_root, pp = _lsi_run(tmp_path)
    backend, meta = AttachLSISelectionTables(run_root=run_root, postprocessing=pp)(
        FakeBackend([{"audio_path": "a.wav"}])
    )
    out = list(backend)
    assert "quality_attached" not in meta
    assert "focal_confidence" not in out[0] and "focal_max_stems" not in out[0]


def test_attach_lsi_quality_opt_out(tmp_path):
    # attach_quality=False never attaches quality even when the shards carry it.
    from sound_event_detection.data.transforms import AttachLSISelectionTables

    run_root, pp = _lsi_run_with_quality(tmp_path)
    backend, meta = AttachLSISelectionTables(run_root=run_root, postprocessing=pp, attach_quality=False)(
        FakeBackend([{"audio_path": "a.wav"}])
    )
    out = list(backend)
    assert "quality_attached" not in meta
    assert "focal_confidence" not in out[0]


def test_cap_per_group_is_deterministic():
    # Same input capped twice must select the exact same rows (regression against
    # the non-deterministic upsample_by_column group ordering).
    df = pl.DataFrame({"sp": ["a"] * 40 + ["b"] * 40, "uid": list(range(80))})
    keep1 = {row["uid"] for row in CapPerGroup(property="sp", count=7)(PolarsBackend(df))[0]}
    keep2 = {row["uid"] for row in CapPerGroup(property="sp", count=7)(PolarsBackend(df))[0]}
    assert keep1 == keep2
    assert len(keep1) == 14  # 7 per group
