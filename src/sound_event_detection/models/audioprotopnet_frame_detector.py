"""Frame-level AudioProtoPNet detector.

Uses pretrained AudioProtoPNet weights (ConvNeXt backbone + prototype layer +
classification head) to produce per-frame predictions.  Instead of the original
global max-pool over both frequency and time, this detector max-pools over
frequency only, preserving the time dimension for frame-level SED.

Satisfies the Detector protocol used by the evaluation pipeline.

Weights are loaded from .pt files produced by ``extract_weights.py`` in the
audioprotopnet-server repo.  No dependency on the ``audioprotopnet`` package.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from transformers import ConvNextConfig, ConvNextModel

from esp_research.protocols.encoder import AudioEncoderOutput
from sound_event_detection.models.frame_detector import FrameDetector

# ── Constants matching the original AudioProtoPNet preprocessing ──────────

SAMPLE_RATE = 32_000
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)

# Mel spectrogram parameters (must match hf_repo/processing_protonet.py exactly)
_N_FFT = 2048
_HOP_LENGTH = 256
_N_MELS = 256
_N_STFT = 1025
_POWER = 2.0
_TOP_DB = 80
_NORM_MEAN = -13.369
_NORM_STD = 13.162

# ── Mel spectrogram (matching AudioProtoNetFeatureExtractor exactly) ──────


class _MelSpectrogramTransform(nn.Module):
    """Replicate AudioProtoNetFeatureExtractor from processing_protonet.py.

    Produces identical output to the HuggingFace feature extractor:
    raw float32 PCM → Spectrogram → MelScale → AmplitudeToDB → z-score → [B,1,256,T]
    """

    def __init__(self) -> None:
        super().__init__()
        self.spectrogram = T.Spectrogram(n_fft=_N_FFT, hop_length=_HOP_LENGTH, power=_POWER)
        self.mel_scale = T.MelScale(n_mels=_N_MELS, sample_rate=SAMPLE_RATE, n_stft=_N_STFT)
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=_TOP_DB)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to normalized mel spectrogram.

        Parameters
        ----------
        waveform : torch.Tensor
            Shape ``[B, samples]`` (mono, float32).

        Returns
        -------
        torch.Tensor
            Shape ``[B, 1, n_mels, T]``.
        """
        spec = self.spectrogram(waveform)  # [B, n_stft, T]
        mel = self.mel_scale(spec)  # [B, n_mels, T]
        mel_db = self.amplitude_to_db(mel)  # [B, n_mels, T]
        mel_norm = (mel_db - _NORM_MEAN) / _NORM_STD
        return mel_norm.unsqueeze(1)  # [B, 1, n_mels, T]


# ── Cosine activation (matching ppnet.py / modeling_protonet.py exactly) ──


def _cosine_activation(
    features: torch.Tensor,
    prototype_vectors: torch.Tensor,
    input_vector_length: int = 64,
    n_eps_channels: int = 2,
    epsilon_val: float = 1e-4,
) -> torch.Tensor:
    """Compute cosine similarity between features and prototype vectors.

    Replicates ``AudioProtoNetClassificationHead.cos_activation`` from
    ``hf_repo/modeling_protonet.py`` (inference path only, no margin).

    Parameters
    ----------
    features : torch.Tensor
        Backbone output after add_on_layers, shape ``[B, C, H, W]``.
    prototype_vectors : torch.Tensor
        Learned prototypes, shape ``[num_prototypes, C, 1, 1]``.

    Returns
    -------
    torch.Tensor
        Cosine activations, shape ``[B, num_prototypes, H, W]``.
    """
    proto_h, proto_w = prototype_vectors.shape[2], prototype_vectors.shape[3]
    normalizing_factor = (proto_h * proto_w) ** 0.5

    # Append epsilon channels to features
    eps_x = torch.full(
        (features.shape[0], n_eps_channels, features.shape[2], features.shape[3]),
        epsilon_val,
        device=features.device,
        dtype=features.dtype,
    )
    x = torch.cat((features, eps_x), dim=1)

    # Normalize features
    x_length = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + epsilon_val)
    x_normalized = (input_vector_length * x / x_length) / normalizing_factor

    # Append epsilon channels to prototypes
    eps_p = torch.full(
        (prototype_vectors.shape[0], n_eps_channels, proto_h, proto_w),
        epsilon_val,
        device=prototype_vectors.device,
        dtype=prototype_vectors.dtype,
    )
    appended_protos = torch.cat((prototype_vectors, eps_p), dim=1)

    # Normalize prototypes
    proto_length = torch.sqrt(torch.sum(appended_protos**2, dim=1, keepdim=True) + epsilon_val)
    normalized_protos = appended_protos / (proto_length + epsilon_val)
    normalized_protos = normalized_protos / normalizing_factor

    # Cosine similarity via conv2d
    activations = F.conv2d(x_normalized, normalized_protos)
    activations = activations / (input_vector_length * 1.01)

    # ReLU (relu_on_cos=True in config)
    activations = torch.relu(activations)

    return activations


# ── LinearLayerWithoutNegativeConnections ─────────────────────────────────


class _LinearLayerWithoutNegativeConnections(nn.Module):
    """Per-class linear layer where each class connects only to its own prototypes.

    Replicates ``LinearLayerWithoutNegativeConnections`` from
    ``hf_repo/modeling_protonet.py``.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.features_per_output_class = in_features // out_features
        assert in_features % out_features == 0
        self.weight = nn.Parameter(torch.empty(out_features, self.features_per_output_class))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-class linear layer.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[..., in_features]``.  Any leading dimensions are preserved.

        Returns
        -------
        torch.Tensor
            Shape ``[..., out_features]``.
        """
        leading_shape = x.shape[:-1]
        # Reshape: [..., in_features] → [..., out_features, features_per_class]
        reshaped = x.view(*leading_shape, self.out_features, self.features_per_output_class)
        weight = torch.relu(self.weight)  # non-negative constraint
        output = torch.einsum("...of,of->...o", reshaped, weight)
        if self.bias is not None:
            output = output + self.bias
        return output


# ── AudioEncoder-compatible wrapper (for fine-tuning) ─────────────────────


class AudioProtoPNetEncoder(nn.Module):
    """Wraps AudioProtoPNet as an AudioEncoder for use with FrameDetector.

    forward([B, samples]) → [B, T, num_prototypes]
    by applying mel → backbone → add_on_layers → cosine_sim → max_pool_freq.

    Satisfies the AudioEncoder protocol used by the training pipeline.
    freeze() freezes the backbone only (prototypes remain trainable).
    unfreeze() unfreezes the backbone.
    """

    def __init__(
        self,
        backbone: nn.Module,
        add_on_layers: nn.Module,
        prototype_vectors: nn.Parameter,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.add_on_layers = add_on_layers
        self.prototype_vectors = prototype_vectors
        self.mel_transform = _MelSpectrogramTransform()
        self._device = device

        # Compute output frame_rate by running a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 1, _N_MELS, 626, device=device)
            backbone_out = self.backbone(dummy).last_hidden_state
            after_addon = self.add_on_layers(backbone_out)
            self._activation_W = after_addon.shape[3]

    # ── AudioEncoder protocol ──────────────────────────────────────────────

    @property
    def output_dim(self) -> int:
        return self.prototype_vectors.shape[0]  # num_prototypes

    @property
    def output_frame_rate(self) -> float:
        return self._activation_W / WINDOW_SECONDS

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def window_duration(self) -> float:
        return WINDOW_SECONDS

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw waveform to per-frame prototype activations.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, samples]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, T, num_prototypes]``.
        """
        mel = self.mel_transform(x)  # [B, 1, n_mels, T]
        backbone_out = self.backbone(mel).last_hidden_state  # [B, C, H, W]
        features = self.add_on_layers(backbone_out)  # [B, C, 2H, 2W]
        activations = _cosine_activation(features, self.prototype_vectors)  # [B, num_protos, H', W']
        pooled = activations.max(dim=2).values  # [B, num_protos, W']
        return pooled.permute(0, 2, 1)  # [B, W', num_protos]

    # ── Freeze / unfreeze (backbone only) ─────────────────────────────────

    def freeze(self) -> None:
        """Freeze backbone parameters; prototypes remain trainable."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        """Unfreeze backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    def encode(self, waveform: torch.Tensor, padding_mask: torch.Tensor) -> AudioEncoderOutput:
        """Encode audio waveform, satisfying the `AudioEncoder` protocol.

        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform of shape ``(batch, samples)``.
        padding_mask : torch.Tensor
            Input padding mask. Not used internally, accepted for protocol compatibility.

        Returns
        -------
        AudioEncoderOutput
            Encoder output with ``embeddings`` of shape ``(batch, frames, num_prototypes)``
            and an all-false ``padding_mask``.
        """
        embeddings = self.forward(waveform)
        out_mask = torch.zeros(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
        return AudioEncoderOutput(embeddings=embeddings, padding_mask=out_mask)


# ── Factory function ──────────────────────────────────────────────────────


def create_audioprotopnet_frame_detector(
    num_classes: int,
    num_prototypes: int,
    labels: list[str],
) -> FrameDetector:
    """Create an uninitialised AudioProtoPNet-based FrameDetector ready for weight loading.

    Builds the full architecture (ConvNeXt backbone + upsample add-on layers +
    prototype vectors + last layer) with random weights. The caller is responsible
    for loading trained weights afterwards via the checkpointing system.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    num_prototypes : int
        Total number of prototype vectors (must be divisible by `num_classes`).
    labels : list[str]
        Class labels in output order. Length must equal `num_classes`.

    Returns
    -------
    FrameDetector
        Model with `classifier` set to a
        `_LinearLayerWithoutNegativeConnections` instance, ready for
        weight loading.

    Raises
    ------
    ValueError
        If ``len(labels) != num_classes`` or ``num_prototypes % num_classes != 0``.
    """
    if len(labels) != num_classes:
        raise ValueError(f"len(labels)={len(labels)} does not match num_classes={num_classes}")
    if num_prototypes % num_classes != 0:
        raise ValueError(f"num_prototypes={num_prototypes} must be divisible by num_classes={num_classes}")

    device = torch.device("cpu")

    backbone_cfg = ConvNextConfig.from_pretrained("facebook/convnext-base-224-22k", num_channels=1)
    backbone = ConvNextModel(backbone_cfg)
    backbone_output_dim = backbone_cfg.hidden_sizes[-1]

    add_on_layers = nn.Upsample(scale_factor=2, mode="bilinear")
    prototype_vectors = nn.Parameter(torch.empty(num_prototypes, backbone_output_dim, 1, 1))

    encoder = AudioProtoPNetEncoder(
        backbone=backbone,
        add_on_layers=add_on_layers,
        prototype_vectors=prototype_vectors,
        device=device,
    )

    last_layer = _LinearLayerWithoutNegativeConnections(
        in_features=num_prototypes,
        out_features=num_classes,
        bias=True,
    )

    model = FrameDetector(encoder=encoder, labels=labels, head="linear")
    model.classifier = last_layer
    return model
