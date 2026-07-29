from esp_research.protocols.encoder import AudioEncoder, AudioEncoderConfig, AudioEncoderOutput
from sound_event_detection.models.encoders.beats import (
    AGGREGATION_STRATEGIES,
    BEATSEncoder,
    BEATSEncoderConfig,
    compute_beats_frame_rate,
)
from sound_event_detection.models.encoders.cnn import CNNEncoder, CNNEncoderConfig
from sound_event_detection.models.encoders.dual_stream import DualStreamEncoder, DualStreamEncoderConfig

__all__ = [
    "AGGREGATION_STRATEGIES",
    "AudioEncoder",
    "AudioEncoderConfig",
    "AudioEncoderOutput",
    "BEATSEncoder",
    "BEATSEncoderConfig",
    "CNNEncoder",
    "CNNEncoderConfig",
    "DualStreamEncoder",
    "DualStreamEncoderConfig",
    "compute_beats_frame_rate",
]
