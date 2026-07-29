"""Multilabel classification metrics."""

from __future__ import annotations

from typing import Any

import torch
import torchmetrics
from torchmetrics import Metric
from torchmetrics.classification import MultilabelAUROC
from torchmetrics.classification.average_precision import MultilabelAveragePrecision


class cmAP5(Metric):
    """Class-wise Mean Average Precision with a minimum-sample-count threshold.

    Classes with fewer positive samples than ``sample_threshold`` are excluded
    from the macro mean (replaced with NaN and ignored by nanmean).

    Parameters
    ----------
    num_labels : int
        Number of classes.
    sample_threshold : int
        Minimum number of positive samples required for a class to be included.
    thresholds : int or list[float] or torch.Tensor or None
        Passed to ``MultilabelAveragePrecision``.
    dist_sync_on_step : bool
        Synchronise metric state across processes at each ``update`` call.
    """

    def __init__(
        self,
        num_labels: int,
        sample_threshold: int,
        thresholds: int | list[float] | torch.Tensor | None = None,
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_labels = num_labels
        self.sample_threshold = sample_threshold
        self.thresholds = thresholds

        self.multilabel_ap = MultilabelAveragePrecision(
            average="macro", num_labels=self.num_labels, thresholds=self.thresholds
        )

        self.add_state("accumulated_predictions", default=[], dist_reduce_fx="cat")
        self.add_state("accumulated_labels", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate a batch of predictions and targets.

        Parameters
        ----------
        logits : torch.Tensor
            Class probability scores, shape (B, num_labels).
        labels : torch.Tensor
            Binary ground-truth labels, shape (B, num_labels).
        """
        self.accumulated_predictions.append(logits)
        self.accumulated_labels.append(labels)

    def compute(self) -> torch.Tensor:
        """Compute macro cmAP over all accumulated batches, excluding rare classes.

        Returns
        -------
        torch.Tensor
            Scalar macro AP ignoring classes below ``sample_threshold``.
        """
        if not isinstance(self.accumulated_predictions, list):
            self.accumulated_predictions = [self.accumulated_predictions]
        if not isinstance(self.accumulated_labels, list):
            self.accumulated_labels = [self.accumulated_labels]

        all_predictions = torch.cat(self.accumulated_predictions, dim=0)
        all_labels = torch.cat(self.accumulated_labels, dim=0)

        class_aps = self.multilabel_ap(all_predictions, all_labels)

        if self.sample_threshold > 1:
            mask = all_labels.sum(axis=0) >= self.sample_threshold
            class_aps = torch.where(mask, class_aps, torch.nan)

        return torch.nanmean(class_aps)


class cmAP(MultilabelAveragePrecision):
    """Macro-averaged class-wise Mean Average Precision.

    Wraps ``MultilabelAveragePrecision(average="macro")``.

    Parameters
    ----------
    num_labels : int
        Number of classes.
    thresholds : int or list[float] or torch.Tensor or None
        Threshold values for the precision-recall curve.
    """

    def __init__(self, num_labels: int, thresholds: int | list[float] | torch.Tensor | None = None) -> None:
        super().__init__(num_labels=num_labels, average="macro", thresholds=thresholds)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return super().__call__(logits, labels)


class mAP(MultilabelAveragePrecision):
    """Micro-averaged Mean Average Precision.

    Wraps ``MultilabelAveragePrecision(average="micro")``.

    Parameters
    ----------
    num_labels : int
        Number of classes.
    thresholds : int or list[float] or torch.Tensor or None
        Threshold values for the precision-recall curve.
    """

    def __init__(self, num_labels: int, thresholds: int | list[float] | torch.Tensor | None = None) -> None:
        super().__init__(num_labels=num_labels, average="micro", thresholds=thresholds)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return super().__call__(logits, labels)


class pcmAP(MultilabelAveragePrecision):
    """Padded class-wise Mean Average Precision (BirdClef 2023 evaluation metric).

    Appends ``padding_factor`` all-positive rows to both predictions and targets
    before computing macro AP.

    Reference: https://www.kaggle.com/competitions/birdclef-2023/overview/evaluation

    Parameters
    ----------
    num_labels : int
        Number of classes.
    padding_factor : int
        Number of all-positive synthetic rows to append. Default 5.
    average : str
        Averaging strategy. Default ``"macro"``.
    thresholds : int or list[float] or torch.Tensor or None
        Threshold values for the precision-recall curve.
    """

    def __init__(
        self,
        num_labels: int,
        padding_factor: int = 5,
        average: str = "macro",
        thresholds: int | list[float] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_labels=num_labels, average=average, thresholds=thresholds, **kwargs)
        self.padding_factor = padding_factor

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute padded cmAP.

        Parameters
        ----------
        logits : torch.Tensor
            Class probability scores, shape (B, num_labels).
        targets : torch.Tensor
            Binary ground-truth labels, shape (B, num_labels).

        Returns
        -------
        torch.Tensor
            Scalar macro AP after padding.
        """
        ones = torch.ones(self.padding_factor, logits.shape[1])
        logits = torch.cat((logits, ones), dim=0)
        targets = torch.cat((targets, ones.int()), dim=0)
        return super().__call__(logits, targets, **kwargs)


class mAUROC(MultilabelAUROC):
    """Macro-averaged multilabel Area Under the ROC Curve.

    Wraps ``torchmetrics.classification.MultilabelAUROC(average="macro")``.

    Parameters
    ----------
    num_labels : int
        Number of classes.
    thresholds : int or list[float] or torch.Tensor or None
        Threshold values for computing the ROC curve.
    """

    def __init__(self, num_labels: int, thresholds: int | list[float] | torch.Tensor | None = None) -> None:
        super().__init__(num_labels=num_labels, average="macro", thresholds=thresholds)


class TopKAccuracy(torchmetrics.Metric):
    """Top-K accuracy for multilabel classification with optional no-call handling.

    A prediction is correct if at least one of the top-K predicted classes
    appears in the ground-truth label set.

    Parameters
    ----------
    topk : int
        Number of top predictions to consider. Default 1.
    include_nocalls : bool
        If True, all-negative instances are counted: a no-call instance is
        correct when the maximum prediction score is below ``threshold``.
        Default False.
    threshold : float
        Decision threshold for no-call correctness. Only used when
        ``include_nocalls=True``. Default 0.5.
    **kwargs
        Additional keyword arguments forwarded to ``torchmetrics.Metric``.
    """

    def __init__(self, topk: int = 1, include_nocalls: bool = False, threshold: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.topk = topk
        self.include_nocalls = include_nocalls
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate a batch of predictions and targets.

        Parameters
        ----------
        preds : torch.Tensor
            Class probability scores, shape (B, num_classes).
        targets : torch.Tensor
            Binary ground-truth labels, shape (B, num_classes).
        """
        _, topk_pred_indices = preds.topk(self.topk, dim=1, largest=True, sorted=True)
        targets = targets.to(preds.device)
        no_call_targets = targets.sum(dim=1) == 0

        if self.include_nocalls:
            no_positive_predictions = preds.topk(self.topk, dim=1, largest=True).values < self.threshold
            correct_all_negative = no_call_targets & no_positive_predictions.all(dim=1)
        else:
            correct_all_negative = torch.tensor(0).to(targets.device)

        expanded_targets = targets.unsqueeze(1).expand(-1, self.topk, -1)
        correct_positive = expanded_targets.gather(2, topk_pred_indices.unsqueeze(-1)).any(dim=1)

        self.correct += correct_positive.sum() + correct_all_negative.sum()
        if not self.include_nocalls:
            self.total += targets.size(0) - no_call_targets.sum()
        else:
            self.total += targets.size(0)

    def compute(self) -> torch.Tensor:
        """Compute top-K accuracy over all accumulated batches.

        Returns
        -------
        torch.Tensor
            Scalar accuracy in [0, 1].
        """
        return self.correct.float() / self.total
