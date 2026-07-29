"""Validate the migrated configs/birdcode/ configs load and have the expected shape."""

from pathlib import Path

import yaml

from sound_event_detection.evaluation.config import SedEvalConfig

_BIRDCODE = Path(__file__).resolve().parents[2] / "configs" / "birdcode"

_BASELINE_TYPES = {"perch2", "audioprotopnet", "beats_sl_all"}


def test_frame_eval_loads_32k() -> None:
    cfg = SedEvalConfig.from_sources(yaml_file=_BIRDCODE / "frame_eval.yml")

    assert cfg.type == "sed"
    assert len(cfg.frame_datasets) == 70  # 68 WABAD sites + powdermill + xcaj
    assert cfg.clip_datasets == []
    assert cfg.batch_size == 8
    assert cfg.inference["overlap"] == 0.5
    assert cfg.frame_eval.iou_thresholds == [0.2, 0.5]
    assert cfg.frame_eval.postprocessing["merge_max_gap"] == 1.0
    assert all("_16k" not in d.config for d in cfg.frame_datasets)


def test_frame_eval_16k_loads() -> None:
    cfg = SedEvalConfig.from_sources(yaml_file=_BIRDCODE / "frame_eval_16k.yml")

    assert len(cfg.frame_datasets) == 70
    assert cfg.batch_size == 32
    assert all("_16k" in d.config for d in cfg.frame_datasets)


def test_frame_eval_smoke_loads() -> None:
    cfg = SedEvalConfig.from_sources(yaml_file=_BIRDCODE / "frame_eval_smoke.yml")

    assert len(cfg.frame_datasets) == 1
    assert cfg.frame_datasets[0].config == "configs/data/xcaj.yml"
    assert cfg.frame_eval.iou_thresholds == [0.2, 0.5]


def test_birdset_clip_eval_loads() -> None:
    cfg = SedEvalConfig.from_sources(yaml_file=_BIRDCODE / "birdset_clip_eval.yml")

    assert cfg.frame_datasets == []
    assert len(cfg.clip_datasets) == 8
    assert all("birdset/" in d.config for d in cfg.clip_datasets)


def test_httpclient_is_a_pure_client_config() -> None:
    with (_BIRDCODE / "httpclient.yml").open() as f:
        cfg = yaml.safe_load(f)

    assert "type" not in cfg  # pure http-client config; the client kind is auto-detected
    assert cfg["url"] == "http://localhost:8100"


def test_denoising_model_config_loads() -> None:
    with (_BIRDCODE / "models" / "denoising_detector.yml").open() as f:
        cfg = yaml.safe_load(f)

    assert cfg["type"] == "denoising_detector"
    # The wrapped clients' blocks are pure http-client configs (url, no type).
    assert cfg["detector"]["url"].startswith("http://")
    assert "type" not in cfg["detector"]
    assert cfg["separator"]["url"].startswith("http://")
    assert "type" not in cfg["separator"]
