# Sound Event Detection

Pretrained sound event detection models focused on bioacoustics. Supports three main functions:

- Inference with pre-trained models: Within python, via a script, or via the large-scale inference (LSI) pipeline.
- Evaluation of model performance on detection datasets.
- Load pre-computed model detections for datasets like Xeno-Canto and iNaturalist.

## Installation

Requires [`uv`](https://docs.astral.sh/uv/). Installation may take several minutes. GPU is not required but will improve speed.

Required packages are listed in `pyproject.toml`. To install them, run:

```bash
uv sync --group gpu   # omit --group gpu for CPU-only
```

All commands run through `uv run`. It may be necessary to include `--group gpu` if using a GPU. Evaluation and LSI also need the dataset storage referenced by `configs/data/*.yml`.

Large-scale inference and using precomputed selection tables both require [`alp-data`](https://github.com/earthspecies/alp-data/), which is already included in `pyproject.toml`.

## Quick start — BirdCODE over a folder of audio

Run the pretrained BirdCODE detector (loaded from the Hub) over every audio file in a folder — any sample rate, resampled to 32 kHz as needed — and write a selection table next to each recording: `dir/x.wav` → `dir/BirdCODE_predictions/x.txt`. Currently supports wav, flac, ogg, and mp3.

```bash
uv run sed-folder --folder /path/to/audio
```

Postprocessing is applied: By default, per-frame detections are thresholded at 0.5, boxes with the same label are merged if separated by less than 1 second, and non-maximal suppression is applied with an IoU threshold of 0.8. Geography filtering is off by default; enable it with `--geo-filter`, a directory of `*.gpkg` range maps, and the recording site's coordinates (applied to every file):

```bash
uv run sed-folder --folder /path/to/audio \
    --geo-filter --range-map-dir geography/range_maps \
    --latitude 42.5 --longitude -72.2
```

## Official models

| Model | Publication | Checkpoint | Summary |
|---|---|---|---|
| BirdCODE | TODO | [EarthSpeciesProject/sed-birdcode](https://huggingface.co/EarthSpeciesProject/sed-birdcode) | Bird Communication Detector |

## CLI entry points

| Command | Purpose | Assumes running |
|---|---|---|
| `sed-folder` | Run BirdCODE over a folder of audio → selection tables | — (loads the model in-process) |
| `sed-server` | Serve a frame detector or sliding-window detector | backing classifier server (sliding-window only) |
| `sed-denoising-server` | Serve the denoising detector | a detector server + a separator server |
| `sed-eval` | Run an evaluation against a served model | a `sed-server` / `sed-denoising-server` server |
| `sed-lsi` | Large-scale inference over a dataset | a `sed-server` (`preds`) or `sed-denoising-server` (`denoised`/`stems`) server |
| `sed-lsi-postprocess` | Turn LSI predictions into selection tables | — (reads shards) |
| `sed-lsi-features` | Add per-event acoustic features to selection tables | — (reads shards) |

Every CLI has a `describe` subcommand that prints its config schema(s), e.g. `uv run sed-eval describe`.

## Using BirdCODE in Python

`FrameDetector` loads a trained detector in-process, either from the HuggingFace Hub by repo id or from a checkpoint directory (local, `gs://…`, or `r2://…`):

```python
from sound_event_detection.models import FrameDetector

# From the HuggingFace Hub (downloads the snapshot, then rebuilds the model);
birdcode = FrameDetector.from_hf_hub("EarthSpeciesProject/sed-birdcode").eval().to("cuda")

# Or from a checkpoint directory: weights from best_model.pt, labels from
# labels.txt, architecture from config.yaml.
ckpt = "checkpoints/birdcode_esp_research"
birdcode = FrameDetector.from_checkpoint_dir(ckpt, f"{ckpt}/config.yaml").to("cuda")

out = birdcode.run(audio, overlap=0.5)   # audio: np.ndarray [batch, samples] at 32 kHz
out.predictions                          # [batch, time, classes] probabilities in [0, 1]
out.class_names                          # list[str] labels aligned to the classes axis
```

## Serving models

For large-scale inference and evaluation, we serve the model over HTTP, then point a client CLI at it via an http-client config.

A **model config** YAML tells the server what to load, dispatching on `type`. The unified server (`sed-server`) reads its path from the `SED_MODEL_CONFIG` environment variable.

### Frame detectors — `type: frame`

Trained detectors (BirdCODE and ablations) loaded either from the HuggingFace Hub or from a local checkpoint directory. All current checkpoints run at 32 kHz.

Set `hf_repo_id` to download and serve a checkpoint from the Hub — this is how the example config loads BirdCODE. An optional `revision` pins a branch, tag, or commit (defaults to the repo's default branch):

```yaml
type: frame
hf_repo_id: EarthSpeciesProject/sed-birdcode
# revision: main   # optional
```

Alternatively, `model_folder` serves a local checkpoint directory (expects `config.yaml`, `best_model.pt`, and `labels.txt`).

Serve either config the same way:

```bash
SED_MODEL_CONFIG=configs/birdcode/models/birdcode_esp_research.yml \
    uv run sed-server --host 0.0.0.0 --port 8100
```

`sed-server` accepts `--host` (default `localhost`), `--port` (default `8100`), `--workers`, `--reload`, and `--log-level`. `SED_DEVICE=cpu|cuda` selects the device (default: cuda if available).

Ablation checkpoints use the same `type: frame` shape:
`configs/birdcode/models/ablations/`.

### Sliding-window detectors — `type: perch2 | audioprotopnet | beats_sl_all`

Clip classifiers wrapped in a `SlidingWindowDetector` to produce frame-level predictions. Each needs a **backing classifier server** already running, discovered through `addr_file` (a text file containing `host:port`):

```yaml
type: audioprotopnet
addr_file: ~/audioprotopnet-server/server.addr
window_size: 5.0        # seconds
hop_size: 2.0           # seconds
analysis_window: 2.0    # optional; defaults to window_size
```

| Type | Backing server | Sample rate |
|---|---|---|
| `perch2` | [earthspecies/perch2-server](https://github.com/earthspecies/perch2-server) | 32 kHz |
| `audioprotopnet` | [earthspecies/audioprotopnet-server](https://github.com/earthspecies/audioprotopnet-server) | 32 kHz |
| `beats_sl_all` | in-repo (below) | 16 kHz |

The external servers write their own `server.addr`; point the config's `addr_file` at it. Serve the wrapper the same way as a frame detector:

```bash
SED_MODEL_CONFIG=configs/birdcode/models/baselines/audioprotopnet_2s.yml \
    uv run sed-server --port 8100
```

`beats_sl_all` runs at 16 kHz — evaluate it with `frame_eval_16k.yml` (frame detection) or `birdset_clip_eval_16k.yml` (clip classification). Its backing classifier is served in-repo:

```bash
# 1. backing classifier (16 kHz), then record its host:port
SED_DEVICE=cuda uv run uvicorn \
    sound_event_detection.serving.sl_beats_all_server:app --host 0.0.0.0 --port 8200
echo "HOST:8200" > .server_addrs/beats_sl_all.addr   # path the config's addr_file points at

# 2. the sliding-window wrapper
SED_MODEL_CONFIG=configs/birdcode/models/baselines/beats_sl_all_2s.yml \
    uv run sed-server --port 8100
```

### Denoising detector — `type: denoising_detector`

NOTE: This requires a separator server to be running. Separator server code will be provided at a later date.

Wraps a detector client and a source-separator client, adding `POST /separate_and_detect` (used by LSI) to the standard contract. Both backing servers must be up when it starts. Its model config names them as pure http-client configs:

```yaml
type: denoising_detector
detector:  {url: http://localhost:8100, timeout: 300}   # a sed-server detector server
separator: {url: http://localhost:8200, timeout: 300}   # a separator server
threshold: 0.5
resampling_method: torchaudio_kaiser_fast
```

```bash
# with a detector server and a separator server already running:
SED_MODEL_CONFIG=configs/birdcode/models/denoising_detector.yml \
    uv run sed-denoising-server --host 0.0.0.0 --port 8110
```

`sed-denoising-server` takes the same options as `sed-server` (default port `8110`).

### HTTP contract

- `GET /` — model metadata: `{labels, sample_rate, frame_rate, window_duration}`
- `GET /health` — `{status: "ok"}` once the model is loaded
- `GET /labels` — ordered label list
- `POST /run` — frame-level inference; response `{predictions, shape [batch, time, classes], frame_rate}`
- `POST /run_as_classifier` — clip-level pooled inference; response shape `[batch, classes]`
- `POST /separate_and_detect` — denoising server only; per-stem audio + predictions

## Evaluation — `sed-eval`

Serve a model, then run `sed-eval` against it with an **eval config** (*what* to evaluate) and an **http-client config** (*how* to reach the model — a `url` plus optional `timeout`/`retries`/`auth`; the client kind is auto-detected from the server).

```bash
# write an http-client config pointing at the running server, e.g.:
#   url: http://HOST:8100
uv run sed-eval --eval-config configs/birdcode/frame_eval.yml \
    --httpclient-config configs/birdcode/httpclient.yml \
    [--checkpoint-dir <dir>] [--output-dir <dir>]
```

- `--checkpoint-dir` — resumable checkpoint directory (auto-generated under `checkpoints/sed/` if omitted).
- `--output-dir` — override the eval config's `output_dir`.
- `sed-eval --resume <checkpoint-dir>` — resume a run; configs are reloaded from the checkpoint.

### Eval configs

| Config | Pathway | Datasets | Sample rate |
|---|---|---|---|
| `configs/birdcode/frame_eval.yml` | frame (detection) | 68 WABAD sites + Powdermill + XC-AJ | 32 kHz |
| `configs/birdcode/birdset_clip_eval.yml` | clip (classification) | 8 BirdSet test splits | 32 kHz |

An eval config selects the pathway through its dataset lists: `frame_datasets` (strong labels, with `species_column`) go through detection; `clip_datasets` (weak labels) through classification.

### Metrics

- **Frame pathway**: frame mAP, event mAP per IoU threshold, thresholded precision/recall/F1.
- **Clip pathway**: cmAP (headline), cmAP5, mAP, pcmAP, MultilabelAUROC, top-1/top-3 accuracy, per-class AP, and `gt_coverage`.

### Results

Each eval run writes `<output_dir>/results.yaml`, updated after every dataset:

- `model` — the served model's metadata (`GET /` response)
- `frame_eval` — the scoring parameters used
- `frame_datasets.<name>` — per-dataset detection metrics
- `clip_datasets.<name>` — per-dataset classification metrics

## Large-scale inference (LSI)

Run a served detector over a dataset, persist per-recording results as compressed `.npz` shards, then postprocess (and optionally enrich) them into selection tables. Three stages: **run → postprocess → features**. Each stage takes `--job-index N --num-jobs M` to split the work across an array of parallel jobs, and writes a `lineage.yaml` chaining back to the stage that produced its input.

The LSI configs (`configs/inference/lsi_birdcode_*.yml`) run the BirdCODE frame detector over the full Xeno-Canto and iNaturalist training splits; they read their datasets from `configs/data/inference/`.

### Run — `sed-lsi`

Builds a dataset from a **run config** (*what* to run) and a detector client from an **http-client config** (*how* to reach the model), then runs the sharded engine over this job's slice.

```bash
# with the appropriate server running (see below):
uv run sed-lsi --run-config configs/inference/lsi_birdcode_xc.yml \
    --httpclient-config <httpclient.yml> [--job-index N --num-jobs M] [--output-dir DIR]
```

The run config's `output.detail` selects what is stored per recording — and which server the `url` must reach:

| `detail` | Stored | Server |
|---|---|---|
| `preds` | combined framewise predictions | a `sed-server` detector server |
| `denoised` | predictions + a threshold-gated denoised waveform | a `sed-denoising-server` server |
| `stems` | the above + every separated stem (audio + preds) | a `sed-denoising-server` server |

### Postprocess — `sed-lsi-postprocess`

Reads the combined predictions in each shard and writes a per-recording selection table (1:1 with the input shards). Re-postprocessing is a cheap re-run into a sibling directory.

```bash
uv run sed-lsi-postprocess --config configs/inference/lsi_birdcode_xc_postprocess.yml \
    --run-dir <run_dir> [--job-index N --num-jobs M]
```

`--run-dir` overrides the config's `input.run_dir` (postprocess several runs
with one config).

#### Geography filtering

Setting `postprocessing.geo_filter: true` drops detections for species whose range maps exclude a recording's location (using the latitude/longitude stored in each shard). It requires `postprocessing.range_map_dir` — a directory (local path or cloud URI) of `*.gpkg` range-map files, globbed at startup and checked to exist before any shards are processed:

```yaml
postprocessing:
  geo_filter: true
  range_map_dir: geography/range_maps   # dir of *.gpkg range maps
```

To use geography filtering, download the open range-map dataset from iNaturalist (<https://www.inaturalist.org/pages/range_maps>) into `range_map_dir`. Each range map's species `name` is resolved to a GBIF canonical name to match the detector's labels. The filter fails open: a detection is dropped only on positive out-of-range evidence (valid coordinates **and** a range map that excludes the point); recordings without coordinates, or species without a range map, are left untouched.

### Features — `sed-lsi-features`

Enriches a postprocessed selection table with per-event `v0minimal` acoustic features. Writes enriched selection tables 1:1 with the postprocess shards.

```bash
uv run sed-lsi-features --config configs/inference/lsi_birdcode_xc_features.yml \
    --run-dir <run_dir> --postprocessing postprocessed_thr0.50_merge1.00_nms0.80_geo \
    [--job-index N --num-jobs M]
```

## Loading a dataset with attached selection tables (Python)

Public GCS buckets hold BirdCODE detections as selection tables for a subset of **Xeno-Canto** and **iNaturalist** recordings. Two data configs load each corpus with those tables attached via the `attach_lsi_selection_tables` transform:

- `configs/data/inference/xeno_canto_selection_tables.yml`
- `configs/data/inference/inaturalist_selection_tables.yml`

Load either with `alp_data.dataset_from_config`, importing the transforms module first so the custom transform is registered:

```python
import io
import pandas as pd
from alp_data import dataset_from_config
import sound_event_detection.data.transforms  # noqa: F401 — registers attach_lsi_selection_tables

dataset, meta = dataset_from_config("configs/data/inference/xeno_canto_selection_tables.yml")
print(meta["attach_lsi_selection_tables"])  # {'matched': ..., 'unmatched': ...}

# The attached `selection_table` column lives on the metadata backend
# (`dataset._data`), so you can read it without decoding audio. It is a TSV
# string (empty for unmatched rows); parse it into a DataFrame of events:
for row in dataset._data:
    if row["selection_table"]:
        events = pd.read_csv(io.StringIO(row["selection_table"]), sep="\t")
        break
```

Each row of a parsed `selection_table` is one detection event, with columns:

- `Begin Time (s)`, `End Time (s)` — the event's span within the recording
- `Species` — predicted class label
- `Score` — mean BirdCODE probability over the event
- 13 `v0minimal` acoustic-feature columns (see `sound_event_detection.inference.features_v0minimal.FEATURE_COLS`)
