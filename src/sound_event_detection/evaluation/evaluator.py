"""`SedEvaluator` — runs SED evaluation against a detector client.

This implements the shared `EvaluatesModelOnTasks` protocol and is driven by a
data/eval config (`SedEvalConfig`). The model it evaluates is any
`DetectorClient` (e.g. a `ServedDetectorClient` connected to a detector
server).
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from alp_data import Dataset, dataset_from_config

import sound_event_detection.data.transforms  # noqa: F401 - registers custom transforms
from esp_research.evals.base import evals_registry
from esp_research.logging import logger
from esp_research.metrics import MetricOutput
from esp_research.mixins import FromConfigMixin
from esp_research.utils import pad_or_crop
from sound_event_detection.adapters.dispatch import DetectorClient
from sound_event_detection.adapters.served_client import SedRunResponse
from sound_event_detection.evaluation.checkpoint import (
    DatasetProgress,
    load_progress,
    load_scorer_state,
    save_progress,
    save_scorer_state,
)
from sound_event_detection.evaluation.classification_eval_helpers import (
    clip_dataset_target_labels,
    compute_multilabel_metrics,
    multi_hot_targets,
    normalize_species_list,
    remap_to_target_labels,
)
from sound_event_detection.evaluation.config import ClipDatasetEntry, FrameDatasetEntry, FrameEvalConfig, SedEvalConfig
from sound_event_detection.evaluation.evaluation import score_file
from sound_event_detection.evaluation.metrics import Scorer


@dataclass
class SedEvalResult:
    """Container for SED evaluation results (satisfies the `EvalResult` protocol).

    Attributes
    ----------
    name : str
        Evaluation name (``"sed"``).
    value : float
        Aggregate score: the mean of the headline per-dataset metrics
        (frame mAP for frame datasets, cmAP for clip datasets).
    details : dict[str, Any]
        Full per-dataset results, keyed under ``frame_datasets`` / ``clip_datasets``.
    metric_outputs : list[MetricOutput]
        Headline metric per evaluated dataset.
    """

    name: str
    value: float
    details: dict[str, Any]
    metric_outputs: list[MetricOutput]

    def __eq__(self, other: object) -> bool:
        """Compare two results by aggregate value.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            ``True`` if both are `SedEvalResult` with equal `value`.
        """
        if not isinstance(other, SedEvalResult):
            return NotImplemented
        return self.value == other.value

    def __lt__(self, other: object) -> bool:
        """Order two results by aggregate value.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            ``True`` if this result's `value` is smaller.
        """
        if not isinstance(other, SedEvalResult):
            return NotImplemented
        return self.value < other.value


@evals_registry.register
class SedEvaluator(FromConfigMixin[SedEvalConfig]):
    """Evaluate a detector client on frame- and clip-level SED datasets.

    Attributes
    ----------
    config_class : type[SedEvalConfig]
        Configuration class for this evaluator.
    expected_model_output : type[SedRunResponse]
        Pydantic schema describing the server's ``/run`` response contract.
    config : SedEvalConfig
        The resolved evaluation configuration.
    """

    config_class = SedEvalConfig
    expected_model_output = SedRunResponse

    def __init__(
        self,
        *,
        type: Literal["sed"] = "sed",
        frame_datasets: list[FrameDatasetEntry] | None = None,
        clip_datasets: list[ClipDatasetEntry] | None = None,
        batch_size: int = 32,
        inference: dict | None = None,
        frame_eval: FrameEvalConfig | dict | None = None,
        output_dir: str = "results/eval",
        checkpoint_dir: str | None = None,
        checkpoint_interval: int | None = None,
    ) -> None:
        """Build the evaluator from config fields.

        Accepts the fields of `SedEvalConfig` (so `FromConfigMixin.from_config`
        can splat a dumped config), re-validating them into ``self.config``.

        Parameters
        ----------
        type : Literal["sed"]
            Registry discriminator.
        frame_datasets : list[FrameDatasetEntry] | None
            Strong-label datasets.
        clip_datasets : list[ClipDatasetEntry] | None
            Weak-label datasets.
        batch_size : int
            Batch size forwarded to the server.
        inference : dict | None
            Extra detector keyword arguments (e.g. ``overlap``).
        frame_eval : FrameEvalConfig | dict | None
            Frame/event scoring parameters.
        output_dir : str
            Directory for the ``results.yaml`` summary.
        checkpoint_dir : str | None
            Directory for resumable checkpoints, or ``None`` to disable.
        checkpoint_interval : int | None
            Files between partial checkpoints within a frame dataset.
        """
        self.config = SedEvalConfig(
            type=type,
            frame_datasets=frame_datasets or [],
            clip_datasets=clip_datasets or [],
            batch_size=batch_size,
            inference=inference or {},
            frame_eval=frame_eval if frame_eval is not None else FrameEvalConfig(),
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
        )

    def evaluate(self, model: DetectorClient) -> SedEvalResult:
        """Evaluate `model` on all configured datasets.

        Parameters
        ----------
        model : DetectorClient
            Connected detector client. The caller owns its lifecycle
            (construction and `close`).

        Returns
        -------
        SedEvalResult
            Aggregate and per-dataset results.
        """
        progress = load_progress(self.config.checkpoint_dir) if self.config.checkpoint_dir else {}

        fe = self.config.frame_eval
        results: dict[str, Any] = {
            "model": model.describe_summary(),
            "frame_eval": {
                "iou_thresholds": fe.iou_thresholds,
                "discretization_frame_rate": fe.discretization_frame_rate,
                "postprocessing": fe.postprocessing,
            },
            "frame_datasets": {},
            "clip_datasets": {},
        }

        for entry in self.config.frame_datasets:
            name, ds_results = self._evaluate_frame_dataset(model, entry, progress)
            results["frame_datasets"][name] = ds_results
            self._write_results_yaml(results)

        for entry in self.config.clip_datasets:
            name, ds_results = self._evaluate_clip_dataset(model, entry, progress)
            results["clip_datasets"][name] = ds_results
            self._write_results_yaml(results)

        self._write_results_yaml(results)
        return self._build_result(results)

    def _evaluate_frame_dataset(
        self,
        client: DetectorClient,
        entry: FrameDatasetEntry,
        progress: dict[str, DatasetProgress],
    ) -> tuple[str, dict]:
        """Evaluate one strong-label dataset, resuming from a checkpoint if present.

        Parameters
        ----------
        client : DetectorClient
            Connected detector client.
        entry : FrameDatasetEntry
            Dataset config entry.
        progress : dict[str, DatasetProgress]
            Loaded checkpoint progress, keyed ``"frame/{dataset_name}"``.

        Returns
        -------
        tuple[str, dict]
            ``(dataset_name, results_dict)``.

        Raises
        ------
        ValueError
            If the dataset sample rate does not match the server's.
        """
        dataset, _ = dataset_from_config(entry.config)
        dataset_name = self._dataset_name(dataset)
        key = f"frame/{dataset_name}"

        prog = progress.get(key)
        if prog is not None and prog.is_complete:
            logger.info("Skipping %s — complete checkpoint found.", dataset_name)
            return dataset_name, prog.results or {}

        if dataset.sample_rate != client.sample_rate:
            raise ValueError(
                f"Dataset {dataset_name} sample rate ({dataset.sample_rate}) "
                f"!= model sample rate ({client.sample_rate})"
            )

        gt_labels = self._gt_labels(dataset, entry.species_column)
        self._log_label_overlap(dataset_name, gt_labels, client.labels)

        fe = self.config.frame_eval
        scorer = Scorer(
            dataset_ontology=gt_labels,
            n_thresholds=fe.n_thresholds,
            min_threshold=fe.min_threshold,
            annotation_col=entry.species_column,
            discretization_frame_rate=fe.discretization_frame_rate,
            iou_thresholds=fe.iou_thresholds,
            thresholds_for_thresholded_metrics=fe.thresholds_for_thresholded_metrics,
        )

        n_done = 0
        if prog is not None and prog.state_file is not None:
            load_scorer_state(self.config.checkpoint_dir, prog.state_file, scorer)
            n_done = prog.n_completed
            logger.info("Resuming %s from file %d/%d.", dataset_name, n_done, len(dataset))

        n_files = len(dataset)
        files_since_ckpt = 0
        for i, item in enumerate(dataset):
            if i < n_done:
                continue
            output = client.run(
                item["audio"][np.newaxis, :], batch_size=self.config.batch_size, **self.config.inference
            )
            score_file(
                preds=output.predictions[0],
                frame_rate=output.frame_rate,
                pred_labels=client.labels,
                gt_selection_table=item["selection_table"],
                scorer=scorer,
                gt_labels=gt_labels,
                species_column=entry.species_column,
                postprocessing_config=fe.postprocessing,
            )
            n_done = i + 1
            files_since_ckpt += 1

            if (
                self.config.checkpoint_dir
                and self.config.checkpoint_interval
                and files_since_ckpt >= self.config.checkpoint_interval
            ):
                state_file = save_scorer_state(self.config.checkpoint_dir, dataset_name, n_done, scorer)
                save_progress(
                    self.config.checkpoint_dir,
                    DatasetProgress("frame", dataset_name, n_done, is_complete=False, state_file=state_file),
                )
                files_since_ckpt = 0

        ds_results = {**scorer.get_results(), "n_files": n_files, "n_gt_classes": len(gt_labels)}
        self._log_frame_summary(dataset_name, ds_results, fe.iou_thresholds)

        if self.config.checkpoint_dir:
            state_file = save_scorer_state(self.config.checkpoint_dir, dataset_name, n_done, scorer)
            save_progress(
                self.config.checkpoint_dir,
                DatasetProgress(
                    "frame", dataset_name, n_done, is_complete=True, state_file=state_file, results=ds_results
                ),
            )
        return dataset_name, ds_results

    def _evaluate_clip_dataset(
        self,
        client: DetectorClient,
        entry: ClipDatasetEntry,
        progress: dict[str, DatasetProgress],
    ) -> tuple[str, dict]:
        """Evaluate one weak-label dataset (no mid-dataset resume).

        Clips are padded/cropped to the client's window length and batched
        through `client.run_as_classifier`; the clip predictions are remapped
        to the dataset's target ontology and scored with the multilabel
        classification metric suite.

        Parameters
        ----------
        client : DetectorClient
            Connected detector client.
        entry : ClipDatasetEntry
            Dataset config entry.
        progress : dict[str, DatasetProgress]
            Loaded checkpoint progress, keyed ``"clip/{dataset_name}"``.

        Returns
        -------
        tuple[str, dict]
            ``(dataset_name, results_dict)``.

        Raises
        ------
        ValueError
            If the dataset sample rate does not match the model's, or the
            dataset yields no items.
        """
        dataset, _ = dataset_from_config(entry.config)
        dataset_name = self._dataset_name(dataset, drop_split_all=True)
        key = f"clip/{dataset_name}"

        prog = progress.get(key)
        if prog is not None and prog.is_complete:
            logger.info("Skipping %s — complete checkpoint found.", dataset_name)
            return dataset_name, prog.results or {}

        if dataset.sample_rate != client.sample_rate:
            raise ValueError(
                f"Dataset {dataset_name} sample rate ({dataset.sample_rate}) "
                f"!= model sample rate ({client.sample_rate})"
            )

        gt_labels = clip_dataset_target_labels(dataset)
        self._log_label_overlap(dataset_name, gt_labels, client.labels)

        n_files = len(dataset)
        logger.info("Clip evaluation on %s: %d files (%d target classes)", dataset_name, n_files, len(gt_labels))

        target_samples = int(client.window_duration * client.sample_rate)
        model_labels_set = set(client.labels)
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        audio_buffer: list[np.ndarray] = []
        species_buffer: list[list[str]] = []
        n_species_total = 0
        n_species_in_model = 0

        def flush() -> None:
            if not audio_buffer:
                return
            output = client.run_as_classifier(
                np.stack(audio_buffer), batch_size=self.config.batch_size, **self.config.inference
            )
            clip_probs = torch.from_numpy(output.predictions).float()  # (B, C)
            all_preds.append(remap_to_target_labels(clip_probs, client.labels, gt_labels))  # (B, n_gt)
            all_targets.append(multi_hot_targets(species_buffer, gt_labels))
            audio_buffer.clear()
            species_buffer.clear()

        start = time.time()
        for i, item in enumerate(dataset):
            wav = torch.from_numpy(np.ascontiguousarray(item["audio"], dtype=np.float32))
            wav, _ = pad_or_crop(wav, target_samples, window_selection="start")
            audio_buffer.append(wav.numpy())

            species_list = normalize_species_list(item.get("species_list", []))
            species_buffer.append(species_list)
            n_species_total += len(species_list)
            n_species_in_model += sum(1 for species in species_list if species in model_labels_set)

            if len(audio_buffer) == self.config.batch_size:
                flush()
            if i < 5 or i % 100 == 0:
                logger.info("%s: [%d/%d] elapsed=%.1fs", dataset_name, i, n_files, time.time() - start)
        flush()

        if not all_preds:
            raise ValueError(f"Clip dataset {dataset_name} yielded no items; check its config/split/filters.")

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        ds_results = compute_multilabel_metrics(preds, targets, num_classes=len(gt_labels), class_names=gt_labels)

        gt_coverage = n_species_in_model / n_species_total if n_species_total > 0 else 0.0
        ds_results.update({"n_files": n_files, "n_classes": len(gt_labels), "gt_coverage": gt_coverage})
        logger.info(
            "%s: cmAP=%.4f, mAP=%.4f, T1=%.3f, T3=%.3f, gt_coverage=%.1f%%",
            dataset_name,
            ds_results["cmAP"],
            ds_results["mAP"],
            ds_results["T1Accuracy"],
            ds_results["T3Accuracy"],
            100.0 * gt_coverage,
        )

        if self.config.checkpoint_dir:
            save_progress(
                self.config.checkpoint_dir,
                DatasetProgress("clip", dataset_name, n_files, is_complete=True, results=ds_results),
            )
        return dataset_name, ds_results

    @staticmethod
    def _dataset_name(dataset: Dataset, drop_split_all: bool = False) -> str:
        """Build a display/checkpoint name for a dataset.

        Parameters
        ----------
        dataset : Dataset
            The loaded dataset.
        drop_split_all : bool
            If ``True``, omit the split suffix when the split is ``"all"``
            (matches the clip-evaluation naming).

        Returns
        -------
        str
            ``"{name}"`` or ``"{name}:{split}"``.
        """
        name = dataset.info.name
        split = getattr(dataset, "split", None)
        if split and not (drop_split_all and split == "all"):
            name = f"{name}:{split}"
        return name

    @staticmethod
    def _gt_labels(dataset: Dataset, species_column: str) -> list[str]:
        """Get the dataset's canonical GT labels.

        Parameters
        ----------
        dataset : Dataset
            The loaded dataset.
        species_column : str
            Column name for the event label.

        Returns
        -------
        list[str]
            Available GT labels.
        """
        try:
            return dataset.get_available_labels(species_column)
        except TypeError:
            return dataset.get_available_labels()

    @staticmethod
    def _log_label_overlap(dataset_name: str, gt_labels: list[str], model_labels: list[str]) -> None:
        """Warn about GT labels missing from the model and report extra model labels.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset, for logging.
        gt_labels : list[str]
            Ground-truth class labels.
        model_labels : list[str]
            Model output class labels.
        """
        missing = set(gt_labels) - set(model_labels)
        extra = set(model_labels) - set(gt_labels)
        logger.info("%s: GT classes=%d, model classes=%d", dataset_name, len(gt_labels), len(model_labels))
        if missing:
            logger.warning("%s: %d GT classes not in model: %s...", dataset_name, len(missing), sorted(missing)[:5])
        if extra:
            logger.info("%s: %d model classes not in GT (ignored).", dataset_name, len(extra))

    @staticmethod
    def _log_frame_summary(dataset_name: str, ds_results: dict, iou_thresholds: list[float]) -> None:
        """Log frame and event mAP for a finished frame dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        ds_results : dict
            Scorer results for the dataset.
        iou_thresholds : list[float]
            IoU thresholds present in the results.
        """
        frame_map = ds_results["frame_map"]["mAP"]
        event_str = ", ".join(f"iou{iou}={ds_results[f'event_map_iou_{iou}']['mAP']:.4f}" for iou in iou_thresholds)
        logger.info("%s: frame_mAP=%.4f, event_mAP: %s", dataset_name, frame_map, event_str)

    def _write_results_yaml(self, results: dict) -> None:
        """Write the human-readable results summary to ``output_dir/results.yaml``.

        Parameters
        ----------
        results : dict
            The accumulated results dict.
        """
        output_dir = Path(self.config.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "results.yaml").open("w") as f:
            yaml.dump(results, f, default_flow_style=False)

    @staticmethod
    def _build_result(results: dict) -> SedEvalResult:
        """Assemble the `SedEvalResult` from accumulated per-dataset results.

        Parameters
        ----------
        results : dict
            Accumulated results with ``frame_datasets`` and ``clip_datasets``.

        Returns
        -------
        SedEvalResult
            Aggregate result. ``value`` is the mean of the headline metrics
            (``frame_map`` for frame datasets, ``cmAP`` for clip datasets);
            ``nan`` if no datasets were evaluated.
        """
        metric_outputs: list[MetricOutput] = []
        for name, res in results["frame_datasets"].items():
            metric_outputs.append(MetricOutput(name=f"{name}/frame_mAP", value=res["frame_map"]["mAP"]))
        for name, res in results["clip_datasets"].items():
            metric_outputs.append(MetricOutput(name=f"{name}/cmAP", value=res["cmAP"]))

        values = [m.value for m in metric_outputs]
        value = float(np.mean(values)) if values else float("nan")
        return SedEvalResult(name="sed", value=value, details=results, metric_outputs=metric_outputs)
