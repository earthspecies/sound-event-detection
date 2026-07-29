"""Utility functions for ESP Research."""

from typing import Literal, Tuple

import torch
import torch.nn.functional as F
from alp_data.io.paths import AnyPathT


def pad_or_crop(
    wav: torch.Tensor,
    target_len: int,
    window_selection: Literal["random", "center", "start"] = "random",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad or crop a waveform to a target length.

    Parameters
    ----------
    wav : torch.Tensor
        Input waveform tensor with shape `(T,)` or `(C, T)`, where `T` is the time
        dimension (number of samples) and `C` is the number of channels.
        Does not expect a batch dimension; this function processes single samples.
    target_len : int
        Target length to pad or window to
    window_selection : Literal["random", "center", "start"]
        How to select the window if cropping is needed

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple of (processed waveform, padding mask).

        The processed waveform has shape `(target_len,)` if input was `(T,)`, or
        `(C, target_len)` if input was `(C, T)`. The padding mask has shape
        `(target_len,)` and dtype `bool`, where `True` indicates padded regions
        and `False` indicates real audio.

        When cropping (input longer than `target_len`), the padding mask is all
        `False` since no padding was applied. When padding (input shorter than
        `target_len`), the mask is `False` for the original audio samples and
        `True` for the padded regions.

    Raises
    ------
    ValueError
        If ``window_selection`` is not one of ``\"random\"``, ``\"center\"``,
        or ``\"start\"``.
    """
    wav_len = wav.size(-1)
    padding_mask = torch.zeros(target_len, dtype=torch.bool)
    processed_wav = wav

    if wav_len > target_len:  # crop
        if window_selection == "random":
            start = torch.randint(0, wav_len - target_len + 1, ()).item()
            end = start + target_len
            processed_wav = wav[..., start:end]
        elif window_selection == "center":
            start = (wav_len - target_len) // 2
            end = start + target_len
            processed_wav = wav[..., start:end]
        elif window_selection == "start":
            processed_wav = wav[..., :target_len]
        else:
            raise ValueError(f"Unknown window selection: {window_selection}")
    else:  # pad
        pad_len = target_len - wav_len
        processed_wav = F.pad(wav, (0, pad_len))
        padding_mask[wav_len:] = True

    return processed_wav, padding_mask


# TODO: temporary function here until #229 in alp-data is merged
def read_json(path: str | AnyPathT) -> object:
    """Read a JSON file and return its contents.

    Parameters
    ----------
    path : str or AnyPathT
        The path string or path object pointing to the JSON file.

    Returns
    -------
    object
        The contents of the JSON file.

    Raises
    ------
    json.JSONDecodeError
        If there is an error parsing the JSON file.
    ValueError
        If the JSON file is empty.
    """

    import json

    from alp_data.io import filesystem_from_path

    try:
        with filesystem_from_path(path).open(str(path), "r") as fp:
            result = json.load(fp)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error parsing JSON file '{path}': {e.msg}", e.doc, e.pos) from e

    if result is None:
        raise ValueError(f"JSON file '{path}' is empty")

    return result
