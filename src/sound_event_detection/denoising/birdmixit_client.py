"""HTTP client for the standalone BirdMixIt source-separation server.

`BirdMixItClient` mirrors `ServedDetectorClient`: it connects to a server URL,
reuses `esp_research.adapters.HttpClient`, and exposes `separate` /
`separate_file`. It returns plain ndarrays (no Output wrapper) — the
`DenoisingDetector` consumes the stems directly.

The wire contract matches the BirdMixIt server (`birdmixit-server/serve.py`). By
default (`binary=True`) audio is POSTed as a raw ``application/octet-stream``
body to ``/separate_binary`` / ``/separate_file_binary`` with the layout as query
parameters, and stems come back as a raw float32 body with ``x-shape``/``x-dtype``
headers — avoiding the base64 inflation and JSON serialize/parse that dominate the
cost (the stem response is ``n_stems``x the input audio). With `binary=False` it
uses the base64-in-JSON ``/separate`` / ``/separate_file`` endpoints instead.
"""

import base64

import httpx
import numpy as np

from esp_research.adapters import HttpClient


class BirdMixItClient:
    """Client for a remote BirdMixIt separation server.

    On construction it queries the server's ``GET /`` endpoint at `url` for
    `sample_rate`, `n_stems`, and `max_batch_size`. `separate` POSTs
    equal-length audio blocks to ``/separate`` and returns the stems.

    Blocking is the caller's responsibility; the server separates one block per
    forward pass. Batches larger than `max_batch_size` are transparently split
    into several requests and re-concatenated.

    Attributes
    ----------
    sample_rate : int
        Sample rate the server expects and returns (Hz). Audio must be
        resampled to this rate before calling `separate`.
    n_stems : int
        Number of separated stems produced per block.
    max_batch_size : int
        Maximum number of blocks the server accepts per request.
    server_config : dict
        The server's raw ``GET /`` payload, retained so a wrapping model can
        surface the separator's identity (e.g. a `weights_sha256`, when the
        server exposes one) in its own `server_config`.
    binary : bool
        When ``True`` (default) audio is uploaded as a raw
        ``application/octet-stream`` body to the ``/separate_binary`` /
        ``/separate_file_binary`` endpoints and stems are returned as a raw body
        with shape/dtype headers, avoiding the base64+JSON tax on both the request
        and the (n_stems-times-larger) stem response. When ``False`` the
        base64-in-JSON ``/separate`` / ``/separate_file`` endpoints are used.
    """

    def __init__(self, url: str, timeout: float = 300.0, binary: bool = True) -> None:
        """Connect to a BirdMixIt server at `url`.

        Parameters
        ----------
        url : str
            Base URL of the running server, e.g. ``http://host:port``. Discovered
            the same way as any other served client (see the ``url`` field of the
            LSI/eval configs); the server's host:port is not resolved here.
        timeout : float
            Per-request timeout in seconds.
        binary : bool
            When ``True`` (default) use the raw-binary endpoints; when ``False``
            use the base64-in-JSON endpoints. See the class docstring.
        """
        self.binary = binary
        base_url = url
        self._client = HttpClient(base_url, route="separate", audio_key="audio", timeout=timeout)
        # Whole-file stitching endpoint shares the base URL but a different route.
        self._file_client = HttpClient(base_url, route="separate_file", audio_key="audio", timeout=timeout)
        self._separate_binary_url = f"{base_url}/separate_binary"
        self._separate_file_binary_url = f"{base_url}/separate_file_binary"
        self._binary_client = httpx.Client(timeout=httpx.Timeout(timeout))

        config = self._client.describe()
        self.server_config: dict = dict(config)
        self.sample_rate: int = int(config["sample_rate"])
        self.n_stems: int = int(config["n_stems"])
        self.max_batch_size: int = int(config["max_batch_size"])

    def separate(self, audio: np.ndarray) -> np.ndarray:
        """Separate equal-length audio blocks into stems.

        Parameters
        ----------
        audio : np.ndarray
            Blocks of shape ``(batch, samples)`` at `self.sample_rate`. All
            blocks must be the same length.

        Returns
        -------
        np.ndarray
            Stems of shape ``(batch, n_stems, samples)``, float32.

        Raises
        ------
        ValueError
            If `audio` is not a 2-D array.
        """
        if audio.ndim != 2:
            raise ValueError(f"Expected 2D audio array [batch, samples], got shape {audio.shape}")

        blocks = np.ascontiguousarray(audio, dtype=np.float32)
        stems = [
            self._separate_request(blocks[start : start + self.max_batch_size])
            for start in range(0, len(blocks), self.max_batch_size)
        ]
        return np.concatenate(stems, axis=0)

    def separate_file(self, audio: np.ndarray) -> np.ndarray:
        """Separate a whole recording into stitched whole-file stems.

        Unlike `separate` (which takes pre-blocked audio and returns per-block
        stems), this sends the whole recording and the server does the
        windowing, batched separation, and cross-block stitching, returning
        coherent whole-file stems continuous across block boundaries.

        Parameters
        ----------
        audio : np.ndarray
            Mono waveform of shape ``(samples,)`` at `self.sample_rate`.

        Returns
        -------
        np.ndarray
            Stitched stems of shape ``(n_stems, samples)``, float32, at
            `self.sample_rate`.

        Raises
        ------
        ValueError
            If `audio` is not a 1-D array.
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D audio array [samples], got shape {audio.shape}")

        mono = np.ascontiguousarray(audio, dtype=np.float32)
        if self.binary:
            return self._post_binary(self._separate_file_binary_url, mono.tobytes(), {"samples": int(mono.shape[0])})

        response = self._file_client({"audio": mono.tobytes(), "samples": int(mono.shape[0])})
        wire_dtype = np.dtype(response.get("dtype", "float32"))
        return (
            np.frombuffer(base64.b64decode(response["stems"]), dtype=wire_dtype)
            .reshape(response["shape"])
            .astype(np.float32)
        )

    def _separate_request(self, blocks: np.ndarray) -> np.ndarray:
        """POST a single sub-batch of blocks and decode the returned stems.

        Parameters
        ----------
        blocks : np.ndarray
            Contiguous float32 blocks of shape ``(batch, samples)`` with
            ``batch <= max_batch_size``.

        Returns
        -------
        np.ndarray
            Stems of shape ``(batch, n_stems, samples)``, float32.
        """
        batch, samples = blocks.shape
        if self.binary:
            return self._post_binary(self._separate_binary_url, blocks.tobytes(), {"batch": batch, "samples": samples})

        response = self._client({"audio": blocks.tobytes(), "batch": batch, "samples": samples})
        wire_dtype = np.dtype(response.get("dtype", "float32"))
        return (
            np.frombuffer(base64.b64decode(response["stems"]), dtype=wire_dtype)
            .reshape(response["shape"])
            .astype(np.float32)
        )

    def close(self) -> None:
        """Close the underlying HTTP clients (both JSON routes and the binary client)."""
        self._client.close()
        self._file_client.close()
        self._binary_client.close()

    def _post_binary(self, url: str, audio_bytes: bytes, params: dict) -> np.ndarray:
        """POST raw audio bytes to a binary endpoint and decode the raw stems.

        Sends `audio_bytes` as the raw ``application/octet-stream`` request body
        with `params` as query parameters, and rebuilds the stems from the raw
        response body using its ``x-shape``/``x-dtype`` headers. Avoids the
        base64 inflation and JSON serialize/parse on both the request and the
        (n_stems-times-larger) stem response.

        Parameters
        ----------
        url : str
            Full URL of the binary endpoint (``/separate_binary`` or
            ``/separate_file_binary``).
        audio_bytes : bytes
            Contiguous little-endian float32 PCM to send as the body.
        params : dict
            Query parameters describing the layout (``batch``/``samples`` or just
            ``samples``).

        Returns
        -------
        np.ndarray
            Stems reshaped to the response's ``x-shape`` header, float32. Raises
            `httpx.HTTPStatusError` via `raise_for_status` on a non-2xx status.
        """
        response = self._binary_client.post(
            url, params=params, content=audio_bytes, headers={"Content-Type": "application/octet-stream"}
        )
        response.raise_for_status()
        shape = tuple(int(dim) for dim in response.headers["x-shape"].split(","))
        wire_dtype = np.dtype(response.headers.get("x-dtype", "float32"))
        return np.frombuffer(response.content, dtype=wire_dtype).reshape(shape).astype(np.float32)
