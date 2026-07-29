"""Large-scale inference: a model-agnostic sharded engine plus a result codec.

`run_sharded` runs any producer over a dataset and writes ``.npz`` shards;
`read_shard` reads them back. `ItemResult` (+ `Stem`) is the per-recording
persistence DTO, and its `to_arrays` / `from_arrays` codec owns the shard key
convention. The pipeline's config-driven entry points are the ``sed-lsi`` CLI
(`sound_event_detection.inference.cli`), the ``sed-lsi-postprocess`` stage
(`sound_event_detection.inference.lsi_postprocess_cli`), and the
``sed-lsi-features`` stage (`sound_event_detection.inference.lsi_features_cli`);
each stamps a chained ``lineage.yaml`` into its output directory (see
`write_lineage`). `load_frame_preds` / `load_denoised` / `load_stems` read a
recording's heavy output back from a dataset row that carries an attached
``lsi_shard`` pointer.
"""

from sound_event_detection.inference.access import load_denoised, load_frame_preds, load_stems, read_item
from sound_event_detection.inference.engine import read_shard, run_sharded
from sound_event_detection.inference.result import Detail, ItemResult, Stem

__all__ = [
    "Detail",
    "ItemResult",
    "Stem",
    "load_denoised",
    "load_frame_preds",
    "load_stems",
    "read_item",
    "read_shard",
    "run_sharded",
]
