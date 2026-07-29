"""Configuration schema for the SED evaluation (the ``--eval-config`` YAML).

This describes *what* to evaluate (datasets, metrics, post-processing) and where
to write results — it deliberately says nothing about *which* model or *how to
reach it*. The model lives behind a server and the connection is configured
separately via the ``--httpclient-config`` YAML (`HttpClientConfig`), mirroring
the split used by ``projects/beans-zero``.
"""

from typing import Literal

from pydantic import BaseModel, Field

from esp_research.configs import CLIConfig


class FrameDatasetEntry(BaseModel):
    """A strong-label dataset to evaluate at frame and event level.

    Attributes
    ----------
    config : str
        Path to the `alp_data` dataset config YAML.
    species_column : str
        Column name holding the event label in the selection tables.
    """

    config: str
    species_column: str = "Species"


class ClipDatasetEntry(BaseModel):
    """A weak-label dataset to evaluate at clip level.

    Attributes
    ----------
    config : str
        Path to the `alp_data` dataset config YAML.
    """

    config: str


class FrameEvalConfig(BaseModel):
    """Parameters controlling frame- and event-level scoring.

    Attributes
    ----------
    iou_thresholds : list[float]
        IoU thresholds for event matching.
    n_thresholds : int
        Number of detection thresholds for the mAP sweep.
    min_threshold : float
        Lowest detection threshold in the sweep.
    discretization_frame_rate : float
        Frame rate (Hz) for discretizing predictions/GT for frame metrics.
    thresholds_for_thresholded_metrics : list[float]
        Detection thresholds used for call-rate and call-duration metrics.
    postprocessing : dict | None
        Optional post-processing config (``merge_max_gap``, ``min_event_duration``,
        ``nms``). Passed through to the post-processing helpers unchanged.
    """

    iou_thresholds: list[float] = Field(default_factory=lambda: [0.5])
    n_thresholds: int = 101
    min_threshold: float = 0.0
    discretization_frame_rate: float = 100.0
    thresholds_for_thresholded_metrics: list[float] = Field(default_factory=lambda: [0.5])
    postprocessing: dict | None = None


class SedEvalConfig(CLIConfig):
    """Top-level configuration for a SED evaluation run.

    Attributes
    ----------
    type : Literal["sed"]
        Registry discriminator.
    frame_datasets : list[FrameDatasetEntry]
        Strong-label datasets (frame + event mAP).
    clip_datasets : list[ClipDatasetEntry]
        Weak-label datasets (clip-level classification metrics).
    batch_size : int
        Batch size forwarded to the server per inference call.
    inference : dict
        Extra keyword arguments forwarded to the detector (e.g. ``overlap``).
    frame_eval : FrameEvalConfig
        Frame/event scoring parameters.
    output_dir : str
        Directory for the human-readable ``results.yaml`` summary.
    checkpoint_dir : str | None
        Directory for resumable checkpoints. ``None`` disables checkpointing.
    checkpoint_interval : int | None
        Save a partial checkpoint every N files during a frame dataset. ``None``
        only checkpoints after each completed dataset. Requires `checkpoint_dir`.
    """

    type: Literal["sed"] = "sed"
    frame_datasets: list[FrameDatasetEntry] = Field(default_factory=list)
    clip_datasets: list[ClipDatasetEntry] = Field(default_factory=list)
    batch_size: int = Field(default=32, ge=1)
    inference: dict = Field(default_factory=dict)
    frame_eval: FrameEvalConfig = Field(default_factory=FrameEvalConfig)
    output_dir: str = "results/eval"
    checkpoint_dir: str | None = None
    checkpoint_interval: int | None = Field(default=None, ge=1)
