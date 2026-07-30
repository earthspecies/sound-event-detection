"""Denoising pipeline: BirdMixIt source separation + focal-species detection.

`BirdMixItClient` talks to the standalone BirdMixIt separation server (and
satisfies the `SourceSeparatorClient` protocol); `DenoisingDetector` is a model
that wraps a separator client and a detector client (`DetectorClient`) to
detect and isolate a focal species. It owns no weights and is served by
`sound_event_detection.serving.serve_denoising_detector` (``sed-denoising-server``);
clients reach it through a `ServedDenoisingDetectorClient`.
"""

from sound_event_detection.denoising.birdmixit_client import BirdMixItClient
from sound_event_detection.denoising.denoising_detector import (
    DenoisingDetector,
    DenoisingDetectorConfig,
    StemDetections,
)
from sound_event_detection.denoising.source_separator import SourceSeparatorClient

__all__ = [
    "BirdMixItClient",
    "DenoisingDetector",
    "DenoisingDetectorConfig",
    "SourceSeparatorClient",
    "StemDetections",
]
