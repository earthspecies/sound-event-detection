"""Unit tests for `SedEvalConfig` loading."""

from pathlib import Path

from sound_event_detection.evaluation.config import SedEvalConfig

_FRAME_EVAL_YAML = """\
type: sed
frame_datasets:
  - config: configs/data/powdermill.yml
    species_column: Species
clip_datasets: []
batch_size: 8
inference:
  overlap: 0.5
frame_eval:
  iou_thresholds:
    - 0.5
  discretization_frame_rate: 100.0
  postprocessing:
    merge_max_gap: 1.0
checkpoint_interval: 50
"""


def test_load_yaml_frame_config(tmp_path: Path) -> None:
    config_path = tmp_path / "frame_eval.yml"
    config_path.write_text(_FRAME_EVAL_YAML)

    cfg = SedEvalConfig.from_sources(yaml_file=config_path)

    assert cfg.type == "sed"
    assert len(cfg.frame_datasets) == 1
    assert cfg.frame_datasets[0].config == "configs/data/powdermill.yml"
    assert cfg.frame_datasets[0].species_column == "Species"
    assert cfg.clip_datasets == []
    assert cfg.batch_size == 8
    assert cfg.inference["overlap"] == 0.5
    assert cfg.frame_eval.iou_thresholds == [0.5]
    assert cfg.frame_eval.discretization_frame_rate == 100.0
    assert cfg.frame_eval.postprocessing["merge_max_gap"] == 1.0
    assert cfg.checkpoint_interval == 50


def test_defaults() -> None:
    cfg = SedEvalConfig()

    assert cfg.type == "sed"
    assert cfg.frame_datasets == []
    assert cfg.clip_datasets == []
    assert cfg.batch_size == 32
    assert cfg.frame_eval.n_thresholds == 101
    assert cfg.checkpoint_dir is None
