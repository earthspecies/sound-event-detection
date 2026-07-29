"""Pooling utilities for aggregating frame-level predictions."""

import torch


def tempered_pooling(frame_probs: torch.Tensor, temperature: float = 1.0, dim: int = 1) -> torch.Tensor:
    """Tempered pooling over frame-level probabilities.

    Generalises the linear softmax pooling function from:
    "A Comparison of Five Multiple Instance Learning Pooling Functions for
    Sound Event Detection with Weak Labeling"
    https://arxiv.org/pdf/1810.09050

    Formula: y = sum(y_i^{1+t}) / sum(y_i^t)

    At t=1 this recovers linear softmax pooling: sum(y_i^2) / sum(y_i).
    As t → infinity this approaches max pooling.

    Parameters
    ----------
    frame_probs : torch.Tensor
        Frame-level probabilities, shape (batch, time, n_classes).
    temperature : float
        Pooling temperature. t=1 is linear softmax, higher values approach max.
    dim : int
        The time dimension to pool over.

    Returns
    -------
    torch.Tensor
        Clip-level probabilities, shape (batch, n_classes).
    """
    epsilon = 1e-7
    numerator = torch.sum(frame_probs ** (1 + temperature), dim=dim)
    denominator = torch.sum(frame_probs**temperature, dim=dim) + epsilon
    return numerator / denominator
