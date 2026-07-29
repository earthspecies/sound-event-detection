"""Helpers for clip-level multilabel classification evaluation.

Combines multilabel metric computation, label-space remapping, and target
ontology / multi-hot target construction for clip-level classification
outputs. The dataset loop lives in `SedEvaluator._evaluate_clip_dataset`,
which batches fixed-length clips through a detector client's
``run_as_classifier``; the helpers here derive the target ontology from the
dataset's weak labels and build the multi-hot target matrix.
"""

from __future__ import annotations

import torch
from alp_data import Dataset
from torchmetrics.classification import MultilabelAveragePrecision

from sound_event_detection.evaluation.classification_metrics import (
    TopKAccuracy,
    cmAP,
    cmAP5,
    mAP,
    mAUROC,
    pcmAP,
)


def _to_named(values: torch.Tensor, class_names: list[str]) -> dict[str, float]:
    """Convert a 1-D tensor to a ``{class_name: value}`` dict.

    Parameters
    ----------
    values : torch.Tensor
        1-D tensor of length ``len(class_names)``.
    class_names : list[str]
        Class names in the same order as ``values``.

    Returns
    -------
    dict[str, float]
        Mapping from class name to scalar float value.
    """
    return {name: values[i].item() for i, name in enumerate(class_names)}


def compute_multilabel_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    class_names: list[str],
) -> dict[str, float | dict[str, float]]:
    """Compute all multilabel evaluation metrics from pre-collected predictions.

    Parameters
    ----------
    preds : torch.Tensor
        Class probability scores in [0, 1], shape (N, num_classes).
    targets : torch.Tensor
        Binary ground-truth labels (int), shape (N, num_classes).
    num_classes : int
        Number of classes.
    class_names : list[str]
        Class names used as keys for per-class metric dicts.

    Returns
    -------
    dict[str, float | dict[str, float]]
        Keys: ``cmAP``, ``cmAP5``, ``mAP``, ``pcmAP``, ``MultilabelAUROC``,
        ``T1Accuracy``, ``T3Accuracy``, ``class_AP``, ``class_AP_masked``.
    """
    results: dict[str, float | dict[str, float]] = {}

    results["cmAP"] = cmAP(num_labels=num_classes)(preds, targets).item()

    metric_cmAP5 = cmAP5(num_labels=num_classes, sample_threshold=5)
    metric_cmAP5.update(preds, targets)
    results["cmAP5"] = metric_cmAP5.compute().item()

    results["mAP"] = mAP(num_labels=num_classes)(preds, targets).item()
    results["pcmAP"] = pcmAP(num_labels=num_classes, padding_factor=5)(preds, targets).item()
    results["MultilabelAUROC"] = mAUROC(num_labels=num_classes)(preds, targets).item()

    t1 = TopKAccuracy(topk=1)
    t1.update(preds, targets)
    results["T1Accuracy"] = t1.compute().item()

    t3 = TopKAccuracy(topk=3)
    t3.update(preds, targets)
    results["T3Accuracy"] = t3.compute().item()

    ap_per_class = MultilabelAveragePrecision(num_labels=num_classes, average=None)(preds, targets)
    results["class_AP"] = _to_named(ap_per_class, class_names)

    mask = targets.sum(dim=0) >= 5
    masked_ap = torch.where(mask, ap_per_class, torch.tensor(float("nan")))
    results["class_AP_masked"] = _to_named(masked_ap, class_names)

    return results


def remap_to_target_labels(
    class_predictions: torch.Tensor, source_labels: list[str], target_labels: list[str]
) -> torch.Tensor:
    """Reorder prediction columns from one label ontology to another.

    Columns present in both ontologies are copied; columns in `target_labels`
    that are absent from `source_labels` are left as zero.

    Parameters
    ----------
    class_predictions : torch.Tensor
        Class probabilities in the source label space, shape (batch, n_source_classes).
    source_labels : list[str]
        Labels naming the columns of `class_predictions`.
    target_labels : list[str]
        Labels naming the columns of the result.

    Returns
    -------
    torch.Tensor
        Class probabilities aligned to `target_labels`, shape (batch, n_target_classes).

    Examples
    --------
    >>> import torch
    >>> preds = torch.tensor([[0.1, 0.2, 0.3]])
    >>> remap_to_target_labels(preds, ["A", "B", "C"], ["C", "A", "D"])
    tensor([[0.3000, 0.1000, 0.0000]])
    """
    source_idx = {s: i for i, s in enumerate(source_labels)}

    result = torch.zeros(class_predictions.shape[0], len(target_labels), device=class_predictions.device)
    src = [source_idx[s] for s in target_labels if s in source_idx]
    dst = [i for i, s in enumerate(target_labels) if s in source_idx]

    if src:
        result[:, dst] = class_predictions[:, src]
    return result


def normalize_species_list(species_list: list[str] | str | None) -> list[str]:
    """Coerce a clip's species label into a list of strings.

    Parameters
    ----------
    species_list : list[str] or str or None
        Per-clip weak labels as produced by the dataset transform.

    Returns
    -------
    list[str]
        Species names as a list (empty if there were none).
    """
    if species_list is None:
        return []
    if isinstance(species_list, str):
        return [species_list]
    return list(species_list)


def clip_dataset_target_labels(dataset: Dataset) -> list[str]:
    """Build the sorted set of target labels from a clip dataset's weak labels.

    BirdSet (and other clip datasets) do not expose ``get_available_labels``,
    so the target ontology is derived as the sorted union of every clip's
    ``species_list``.

    Parameters
    ----------
    dataset : Dataset
        alp-data Dataset yielding items with a ``species_list`` field.

    Returns
    -------
    list[str]
        Sorted unique species names across the dataset.
    """
    labels: set[str] = set()
    for item in dataset:
        labels.update(normalize_species_list(item.get("species_list", [])))
    return sorted(labels)


def multi_hot_targets(species_lists: list[list[str]], gt_labels: list[str]) -> torch.Tensor:
    """Build a multi-hot target matrix from per-clip species lists.

    Parameters
    ----------
    species_lists : list[list[str]]
        Per-clip weak labels.
    gt_labels : list[str]
        Target class ontology (column order). Species outside it are ignored.

    Returns
    -------
    torch.Tensor
        Binary targets of shape ``(n_clips, n_gt_labels)``, dtype int32.
    """
    gt_idx = {label: i for i, label in enumerate(gt_labels)}
    targets = torch.zeros(len(species_lists), len(gt_labels), dtype=torch.int32)
    for row, species_list in enumerate(species_lists):
        for species in species_list:
            if species in gt_idx:
                targets[row, gt_idx[species]] = 1
    return targets
