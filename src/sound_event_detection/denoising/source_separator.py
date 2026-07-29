"""Protocol for a source-separation client, mirroring the `Detector` protocol.

`DenoisingDetector` depends on this structural interface rather than a concrete
client, so any separation backend behind the same wire contract can be dropped
in. `BirdMixItClient` satisfies it today; the protocol defines the wire contract
the denoising detector relies on (the input rate, the stem count, and whole-file
separation).
"""

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["SourceSeparatorClient"]


@runtime_checkable
class SourceSeparatorClient(Protocol):
    """A client for a source-separation server.

    Attributes
    ----------
    sample_rate : int
        Sample rate the separator expects and returns, in Hz. Audio must be
        resampled to this rate before separation.
    n_stems : int
        Number of stems the separator returns per recording.
    """

    sample_rate: int
    n_stems: int

    def separate_file(self, audio: np.ndarray) -> np.ndarray:
        """Separate a whole recording into stitched, whole-file stems.

        Parameters
        ----------
        audio : np.ndarray
            Mono waveform of shape ``(samples,)`` at `sample_rate`.

        Returns
        -------
        np.ndarray
            Stems of shape ``(n_stems, samples)``, float32, at `sample_rate`,
            coherent across the whole recording.
        """
        ...
