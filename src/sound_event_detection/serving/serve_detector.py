"""Unified detector server driven by a model-config file.

Serves any detector behind the standard contract (``GET /``, ``POST /run``,
``POST /run_as_classifier``), picking what to serve from a model-config YAML
pointed to by the ``SED_MODEL_CONFIG`` environment variable. This is what
`sed-eval`'s run script launches, so every model is described by a single
config file.

The model config dispatches on ``type``:

- ``type: frame`` — a trained checkpoint, loaded either from a local folder or from
  the HuggingFace Hub (``hf_repo_id`` takes precedence over ``model_folder`` when both
  are present; an optional ``revision`` pins a branch, tag, or commit)::

      type: frame
      model_folder: checkpoints/birdcode_esp_research

  or::

      type: frame
      hf_repo_id: EarthSpeciesProject/sed-birdcode

- ``type: perch2 | audioprotopnet | beats_sl_all`` — a baseline classifier wrapped
  in a sliding window (see `create_sliding_window_detector_from_config`)::

      type: audioprotopnet
      addr_file: ~/audioprotopnet-server/server.addr
      window_size: 5.0
      hop_size: 2.0
      analysis_window: 2.0

Deploy with::

    SED_MODEL_CONFIG=configs/birdcode/models/birdcode_esp_research.yml \\
        uv run sed-server --host localhost --port 8100

Environment variables:

- ``SED_MODEL_CONFIG``  path to the model-config YAML (required).
- ``SED_DEVICE``        ``"cpu"`` or ``"cuda"`` for ``type: frame`` (default: cuda if available).
"""

import os
from pathlib import Path

import yaml

from esp_research.protocols.detector import Detector
from sound_event_detection.models.sliding_window_detector import (
    SLIDING_WINDOW_DETECTOR_TYPES,
    create_sliding_window_detector_from_config,
)
from sound_event_detection.serving.server import (
    create_app,
    git_head_commit,
    load_frame_detector,
    load_frame_detector_from_hf,
    state_dict_sha256,
)


def _build_model() -> Detector:
    """Build the detector to serve from the ``SED_MODEL_CONFIG`` file.

    Also attaches a `server_config` identity dict to the built model (surfaced
    by ``GET /`` via `create_app`'s `describe_extras`, so it flows into a
    client's `server_config` and any downstream lineage record):

    - ``type: frame`` -> ``{type, model_folder | hf_repo_id (+ revision), weights_sha256,
      git_commit}``, where `weights_sha256` is a SHA-256 over the loaded checkpoint's
      weights and the source key reflects whether it was loaded from a local folder or
      the HuggingFace Hub.
    - a sliding-window baseline -> the identity the factory composed (its
      wrapped classifier's captured ``GET /`` identity, incl. that classifier's
      `weights_sha256` when its server exposes one), plus this server's
      `git_commit`.

    Returns
    -------
    Detector
        A frame detector (``type: frame``) or a sliding-window baseline, with a
        `server_config` attribute set.

    Raises
    ------
    RuntimeError
        If ``SED_MODEL_CONFIG`` is not set.
    ValueError
        If the config file is not a YAML mapping, or the model config's
        ``type`` is missing or not a supported detector type.
    """
    config_path = os.environ.get("SED_MODEL_CONFIG")
    if not config_path:
        raise RuntimeError("SED_MODEL_CONFIG environment variable is not set.")

    with open(Path(config_path).expanduser(), "r") as f:
        model_config = yaml.safe_load(f)

    if not isinstance(model_config, dict):
        raise ValueError(f"Model config {config_path} must be a YAML mapping, got {type(model_config).__name__}.")

    model_type = model_config.get("type")
    if model_type == "frame":
        repo_id = model_config.get("hf_repo_id")
        if repo_id is not None:
            revision = model_config.get("revision")
            source = repo_id if revision is None else f"{repo_id}@{revision}"
            print(f"Serving frame detector from HuggingFace Hub {source}.", flush=True)
            model = load_frame_detector_from_hf(repo_id, revision=revision)
            model.server_config = {
                "type": "frame",
                "hf_repo_id": repo_id,
                "weights_sha256": state_dict_sha256(model),
            }
            if revision is not None:
                model.server_config["revision"] = revision
        else:
            folder = str(Path(model_config["model_folder"]).expanduser())
            print(f"Serving frame detector from {folder}.", flush=True)
            model = load_frame_detector(folder)
            model.server_config = {
                "type": "frame",
                "model_folder": folder,
                "weights_sha256": state_dict_sha256(model),
            }
        model.server_config["git_commit"] = git_head_commit()
        return model

    if model_type in SLIDING_WINDOW_DETECTOR_TYPES:
        detector = create_sliding_window_detector_from_config(model_config)
        # The factory composed the wrapped classifier's identity into
        # detector.server_config; add this serving process's commit.
        detector.server_config["git_commit"] = git_head_commit()
        print(
            f"Serving sliding-window detector: type={model_type}, {len(detector.labels)} classes, "
            f"sample_rate={detector.sample_rate}, frame_rate={detector.frame_rate}, "
            f"window_duration={detector.window_duration}.",
            flush=True,
        )
        return detector

    raise ValueError(
        f"Unknown model type {model_type!r} in {config_path}. "
        f"Expected 'frame' or one of {sorted(SLIDING_WINDOW_DETECTOR_TYPES)}."
    )


app = create_app(model_factory=_build_model, describe_extras=lambda model: getattr(model, "server_config", {}))
