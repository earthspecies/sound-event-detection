"""Build a connected detector client from an `HttpClientConfig`.

`detector_client_from_config` takes an `HttpClientConfig` (its ``url`` names a
running server; ``timeout`` / ``retries`` / ``auth`` are optional) and connects
a client to the server it names. The kind of client is auto-detected from the
server's ``GET /`` metadata:

- a standard detector server (``sed.app``: frame or sliding-window baseline)
  -> `ServedDetectorClient`;
- a denoising detector server (``sed.denoising_app``, whose metadata carries
  ``type: denoising_detector``) -> `ServedDenoisingDetectorClient`, which adds
  the denoising surface (`separate_and_detect`, `threshold`, `n_stems`).

The URL alone says which model you are talking to; no model weights are loaded
here. This is the shared entry point of `sed-eval` and large-scale inference.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from esp_research.adapters.client_config import HttpClientConfig
from esp_research.protocols.classifier import MultiLabelClassifierOutput
from esp_research.protocols.detector import DetectorOutput


@runtime_checkable
class DetectorClient(Protocol):
    """Client surface for running a detector that lives on a server.

    This is the full surface the evaluation harness, the eval CLI, and
    large-scale inference rely on; every dispatched client provides it.

    Attributes
    ----------
    labels : list[str]
        Output class labels.
    sample_rate : int
        Expected input audio sample rate in Hz.
    frame_rate : float
        Output frame rate in frames per second.
    window_duration : float
        Detector input window duration in seconds.
    server_config : dict
        JSON-serialisable identity of the model(s) behind the client, used to
        detect a changed model when resuming an evaluation.
    """

    labels: list[str]
    sample_rate: int
    frame_rate: float
    window_duration: float
    server_config: dict

    def run(self, audio: np.ndarray, *args: object, **kwargs: object) -> DetectorOutput:
        """Run frame-level inference on a batch of audio.

        Parameters
        ----------
        audio : np.ndarray
            Batched waveform of shape ``(batch, n_samples)`` at `self.sample_rate`.
        *args : object
            Extra positional arguments for the implementing client.
        **kwargs : object
            Extra keyword arguments for the implementing client (e.g. ``batch_size``).

        Returns
        -------
        DetectorOutput
            Frame-level predictions of shape ``(batch, time, classes)``.
        """
        ...

    def run_as_classifier(self, audio: np.ndarray, *args: object, **kwargs: object) -> MultiLabelClassifierOutput:
        """Run clip-level classification on a batch of audio.

        Parameters
        ----------
        audio : np.ndarray
            Batched waveform of shape ``(batch, n_samples)`` at `self.sample_rate`.
        *args : object
            Extra positional arguments for the implementing client.
        **kwargs : object
            Extra keyword arguments for the implementing client (e.g. ``batch_size``).

        Returns
        -------
        MultiLabelClassifierOutput
            Clip-level predictions of shape ``(batch, classes)``.
        """
        ...

    def describe_summary(self) -> dict:
        """Return a compact, serialisable summary of the model's metadata.

        Returns
        -------
        dict
            Summary recorded in results files to identify the model.
        """
        ...

    def close(self) -> None:
        """Release the client's network resources."""
        ...


__all__ = [
    "DetectorClient",
    "detector_client_from_config",
]


def detector_client_from_config(config: HttpClientConfig, labels: list[str] | None = None) -> DetectorClient:
    """Connect a detector client to the server named by an `HttpClientConfig`.

    Fetches the server's metadata once from ``GET /`` and connects the matching
    client: a `ServedDenoisingDetectorClient` when the metadata identifies a
    denoising detector server (``type: denoising_detector``), else a
    `ServedDetectorClient`. This is the same path `sed-eval` takes (a ``url`` in
    its httpclient config), so LSI and eval reach a server identically. Each
    served client base64-encodes the request audio (forcing ``audio_key="audio"``)
    to match the server's contract.

    Parameters
    ----------
    config : HttpClientConfig
        Connection settings for the running server. ``url`` (base URL, e.g.
        ``http://host:port``) is required; ``timeout`` (per-request seconds),
        ``retries``, and ``auth`` are optional. ``route`` / ``audio_key`` /
        ``audio_format`` are ignored — the served clients set the routes and
        the audio key from the wire contract.
    labels : list[str] or None
        Optional output label list. The authoritative labels come from the
        server; when provided and it does not match, a `ValueError` is raised.

    Returns
    -------
    DetectorClient
        A connected client ready for `run`; a `ServedDenoisingDetectorClient`
        (adding `separate_and_detect` / `threshold` / `n_stems`) when the
        server is a denoising detector.

    Raises
    ------
    ValueError
        If `labels` is provided and differs from the server's.
    """
    from sound_event_detection.adapters import served_client
    from sound_event_detection.adapters.served_client import ServedDetectorClient
    from sound_event_detection.adapters.served_denoising_client import ServedDenoisingDetectorClient
    from sound_event_detection.denoising.denoising_detector import DENOISING_DETECTOR_TYPE

    # The probe reaches HttpClient through the served_client module attribute
    # (not a direct import) so tests that monkeypatch `served_client.HttpClient`
    # intercept the probe together with the clients from_config builds.
    probe = served_client.HttpClient.from_config(config)
    try:
        meta = probe.describe()
    finally:
        probe.close()

    if meta.get("type") == DENOISING_DETECTOR_TYPE:
        detector: DetectorClient = ServedDenoisingDetectorClient.from_config(config, meta=meta)
    else:
        detector = ServedDetectorClient.from_config(config, meta=meta)

    if labels is not None and list(labels) != detector.labels:
        detector.close()  # release the route-bound HTTP clients opened by from_config
        raise ValueError("Provided labels do not match the server's labels.")

    return detector
