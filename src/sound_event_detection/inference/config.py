"""Configuration schemas for the large-scale-inference (LSI) pipeline stages.

One config class per stage of the ``run -> postprocess -> features`` pipeline:

- `LsiRunConfig` — the ``sed-lsi`` run config (the ``--run-config`` YAML).
  Describes *what* to run (which `alp_data` dataset) and how/where to write the
  sharded output, and deliberately says nothing about *which* model or *how to
  reach it*. The model lives behind a server and the connection is configured
  separately via the ``--httpclient-config`` YAML (`HttpClientConfig`), mirroring
  the split used by ``sed-eval``.
- `LsiPostprocessConfig` — the ``sed-lsi-postprocess`` config: how to turn a
  run's `ItemResult` shards into selection-table shards.
- `LsiFeaturesConfig` — the ``sed-lsi-features`` config: which acoustic features
  to append to a postprocessed run's selection tables.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from esp_research.configs import CLIConfig
from sound_event_detection.inference.result import DEFAULT_PREDS_THRESHOLD, Detail

#: Default binarization threshold for the postprocess stage when the config
#: sets none (a bare postprocess config only thresholds, at this value).
DEFAULT_DETECTION_THRESHOLD = 0.5

#: Supported acoustic-feature-set versions for the feature stage.
FeatureVersion = Literal["v0minimal"]


class LsiDatasetConfig(BaseModel):
    """The dataset to run inference over.

    Attributes
    ----------
    config : str
        Path to the `alp_data` dataset config YAML.
    id_column : str | None
        Item key naming each file's stable identifier. ``None`` (the default)
        falls back to the dataset's own originals-path column.
    focal_column : str | None
        Item key holding the focal-species label. Required for the ``denoised``
        / ``stems`` detail rungs; unused for ``preds``.
    latitude_column : str
        Item key holding decimal latitude (WGS84). Missing values are stored as
        ``nan``. Default ``"latitudeDecimal"``.
    longitude_column : str
        Item key holding decimal longitude (WGS84); same handling as
        `latitude_column`. Default ``"longitudeDecimal"``.
    """

    config: str
    id_column: str | None = None
    focal_column: str | None = None
    latitude_column: str = "latitudeDecimal"
    longitude_column: str = "longitudeDecimal"


class LsiOutputConfig(BaseModel):
    """Where and at what detail to persist the sharded output.

    Attributes
    ----------
    dir : str
        Output directory (local path or cloud URI) for the ``shard_*.npz`` files
        and the run manifest.
    detail : Detail
        The detail rung to emit: ``preds`` (combined predictions only),
        ``denoised`` (predictions + a denoised waveform), or ``stems`` (the same
        plus every stem). The ``denoised`` / ``stems`` rungs require the
        http-client ``url`` to point at a ``sed.denoising_app`` server. Default
        ``"preds"``.
    files_per_shard : int
        Number of recordings packed into each ``.npz`` shard.
    preds_threshold : float
        Max-probability threshold forwarded to the prediction codec (a class is
        dropped from a stored track if its peak probability is below this).
    """

    dir: str
    detail: Detail = "preds"
    files_per_shard: int = Field(default=8, gt=0)
    preds_threshold: float = DEFAULT_PREDS_THRESHOLD


class LsiRunConfig(CLIConfig):
    """Top-level configuration for a large-scale-inference run.

    Says nothing about the model or how to reach it — that lives in the
    ``--httpclient-config`` YAML (`HttpClientConfig`).

    Attributes
    ----------
    dataset : LsiDatasetConfig
        The dataset to run inference over.
    output : LsiOutputConfig
        Where and at what detail to persist the sharded output.
    max_audio_seconds : float | None
        Skip (and log) any recording longer than this many seconds before it
        reaches the model. Guards the ``denoised`` / ``stems`` rungs against
        ultra-long files that can crash the separator. ``None`` (the default)
        disables the guard.
    """

    dataset: LsiDatasetConfig
    output: LsiOutputConfig
    max_audio_seconds: float | None = None


class PostprocessInputConfig(BaseModel):
    """Where the postprocess stage reads its `ItemResult` shards from.

    Attributes
    ----------
    run_dir : str | None
        Directory (local path or cloud URI) holding the run's ``shard_*.npz``
        `ItemResult` shards. ``None`` (the default) requires ``--run-dir`` on the
        command line (so one config can postprocess several runs).
    """

    run_dir: str | None = None


class PostprocessingConfig(BaseModel):
    """The postprocessing chain applied to each recording's predictions.

    Everything but `threshold` is off unless set, so a bare config yields plain
    threshold-only selection tables (no merge / min-duration / NMS / filtering).

    Attributes
    ----------
    threshold : float
        Binarization threshold applied before event extraction. Default
        `DEFAULT_DETECTION_THRESHOLD`.
    merge_max_gap : float | None
        Merge events of the same class separated by at most this many seconds.
        ``None`` (or a non-positive value) disables merging.
    min_event_duration : float | None
        Drop events shorter than this many seconds. ``None`` (or a non-positive
        value) disables the filter.
    nms : float | None
        IoU threshold for cross-class non-maximum suppression. ``None`` disables
        NMS.
    annotation_col : str
        Name of the label column in the output selection table. Default
        ``"Species"``.
    allowed_classes : list[str] | None
        When given, detections whose `annotation_col` is not in this set are
        dropped (before the merge / min-duration / NMS chain). ``None`` keeps all
        classes.
    geo_filter : bool
        When ``True``, drop detections for species whose range maps exclude the
        recording's location (before the chain). Needs ``geopandas`` / ``shapely``
        and each recording's stored latitude/longitude, plus `range_map_dir`.
        Default ``False``.
    range_map_dir : str | None
        Directory (local path or cloud URI) holding the ``*.gpkg`` range-map
        files consulted by the geography filter. Required when `geo_filter` is
        ``True`` (validated on load), ignored otherwise. ``None`` (the default).

    Raises
    ------
    ValueError
        If `geo_filter` is ``True`` but no `range_map_dir` is set.
    """

    threshold: float = DEFAULT_DETECTION_THRESHOLD
    merge_max_gap: float | None = None
    min_event_duration: float | None = None
    nms: float | None = None
    annotation_col: str = "Species"
    allowed_classes: list[str] | None = None
    geo_filter: bool = False
    range_map_dir: str | None = None

    @model_validator(mode="after")
    def _require_range_map_dir_for_geo_filter(self) -> "PostprocessingConfig":
        """Require `range_map_dir` whenever the geography filter is enabled.

        Returns
        -------
        PostprocessingConfig
            The validated config (unchanged).

        Raises
        ------
        ValueError
            If `geo_filter` is ``True`` but no `range_map_dir` is set.
        """
        if self.geo_filter and not self.range_map_dir:
            raise ValueError("range_map_dir is required when geo_filter is true")
        return self


class LsiPostprocessConfig(CLIConfig):
    """Top-level configuration for the LSI postprocess stage.

    Turns a run's `ItemResult` shards into selection-table shards. The output
    lands in a sibling directory of the run whose name encodes the threshold and
    enabled chain steps, so re-thresholding never overwrites.

    Attributes
    ----------
    input : PostprocessInputConfig
        Where to read the run's `ItemResult` shards from.
    postprocessing : PostprocessingConfig
        The postprocessing chain to apply.
    """

    input: PostprocessInputConfig = Field(default_factory=PostprocessInputConfig)
    postprocessing: PostprocessingConfig = Field(default_factory=PostprocessingConfig)


class FeaturesDatasetConfig(BaseModel):
    """Points the feature stage at the source-audio dataset (optional override).

    Only consulted for ``preds`` runs, whose shards store no audio and so must
    have their features computed on the original source audio re-read at feature
    time. Normally the dataset is recovered automatically from the run's
    ``lineage.yaml``; this override is for when that lineage is missing or its
    recorded (repo-relative) dataset path is unreachable from where the feature
    job runs.

    Attributes
    ----------
    config : str
        Path to the `alp_data` dataset config YAML (the same one the run used).
    id_column : str | None
        Item key whose value keyed each recording's shard. ``None`` (the
        default) falls back to the dataset's own originals-path column, matching
        the run's own default.
    """

    config: str
    id_column: str | None = None


class FeaturesInputConfig(BaseModel):
    """Where the feature stage reads its shards (and source audio) from.

    Attributes
    ----------
    run_dir : str | None
        Directory (local path or cloud URI) holding the run's ``shard_*.npz``
        `ItemResult` shards. These carry the denoised focal audio for a
        ``denoised``/``stems`` run; a ``preds`` run stores no audio and its
        source audio is re-read from `dataset` instead. ``None`` (the default)
        requires ``--run-dir`` on the command line.
    postprocessing : str | None
        Name of the postprocessed-selection-table subdirectory under `run_dir`
        (the event spans to enrich), e.g. ``"postprocessed_thr0.50"``. ``None``
        (the default) requires ``--postprocessing`` on the command line.
    dataset : FeaturesDatasetConfig | None
        Optional override for the source-audio dataset of a ``preds`` run.
        ``None`` (the default) recovers it from the run's ``lineage.yaml``; only
        consulted when a shard has no stored denoised audio.
    """

    run_dir: str | None = None
    postprocessing: str | None = None
    dataset: FeaturesDatasetConfig | None = None


class FeaturesConfig(BaseModel):
    """Which acoustic-feature set to compute.

    Attributes
    ----------
    version : FeatureVersion
        The feature-set version (maps to a module of pure feature math). Default
        ``"v0minimal"``.
    """

    version: FeatureVersion = "v0minimal"


class LsiFeaturesConfig(CLIConfig):
    """Top-level configuration for the LSI feature stage.

    Enriches a postprocessed run's selection tables with per-event acoustic
    features, computed on the stored denoised focal audio when the run stored one
    (``denoised``/``stems``), else on the original source audio re-read at
    feature time (``preds``). The output lands in a ``features_<version>``
    subdirectory of the selection-table directory.

    Attributes
    ----------
    input : FeaturesInputConfig
        Where to read the `ItemResult` shards and the selection-table spans from.
    features : FeaturesConfig
        Which feature set to compute.
    """

    input: FeaturesInputConfig = Field(default_factory=FeaturesInputConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
