"""
Transforms for sound event detection datasets.

Registers transforms with the alp_data transform registry so that dataset
configs reference them by ``type`` when loaded via
`alp_data.dataset_from_config`:

* `SpeciesListFromSelectionTable` (``species_list_from_selection_table``) —
  derive a per-file species list from a selection-table column.
* `SpeciesListFromColumn` (``species_list_from_column``) — derive a per-file
  species list from a JSON-list metadata column (e.g. BirdSet's
  ``canonical_name_multispecies``).
* `CapPerGroup` (``cap_per_group``) — cap every category of a column to at most
  a fixed number of rows (a fixed-count balance, unlike `balanced_sample`'s
  min/median/mean strategies).
* `AttachLSISelectionTables` (``attach_lsi_selection_tables``) — attach the
  selection-table TSV and heavy-shard pointer produced by large-scale inference.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
from alp_data.backends.protocol import DataBackend
from alp_data.io import anypath, filesystem_from_path
from pydantic import BaseModel

#: Regex capturing the zero-padded index of a ``shard_NNNN.npz`` filename.
_SHARD_RE = re.compile(r"shard_(\d+)\.npz$")

#: Columns tried, in order, when `AttachLSISelectionTables.id_column` is not given.
#: These are the common `alp_data` originals-path columns the LSI producer keys
#: its shards by (``str(item[id_column])``).
_DEFAULT_ID_COLUMNS = ("originals_path", "audio_path", "audio_fp", "relative_path")


class SpeciesListFromSelectionTableConfig(BaseModel):
    """Configuration for SpeciesListFromSelectionTable transform."""

    type: Literal["species_list_from_selection_table"] = "species_list_from_selection_table"
    species_column: str = "Species"
    output_column: str = "species_list"
    selection_table_column: str = "selection_table"


class SpeciesListFromSelectionTable:
    """Extract a list of unique species from selection tables.

    Parses the selection table (TSV format stored as string) for each row and
    creates a new column containing the list of unique species found in that
    selection table. Each selection table may contain multiple annotations
    (rows), but only unique species are returned. For example, a table with 10
    Cardinal and 2 Robin annotations yields ``['Cardinal', 'Robin']``.

    Parameters
    ----------
    species_column : str
        Column name in the selection table that contains species labels.
        Default is ``"Species"`` (Raven/Powdermill format).
    output_column : str
        Name of the new column to store the species list.
        Default is ``"species_list"``.
    selection_table_column : str
        Column name containing the selection table TSV string.
        Default is ``"selection_table"``.
    """

    def __init__(
        self,
        *,
        species_column: str = "Species",
        output_column: str = "species_list",
        selection_table_column: str = "selection_table",
    ) -> None:
        self.species_column = species_column
        self.output_column = output_column
        self.selection_table_column = selection_table_column

    @classmethod
    def from_config(cls, cfg: SpeciesListFromSelectionTableConfig) -> SpeciesListFromSelectionTable:
        return cls(**cfg.model_dump(exclude={"type"}))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """Extract species list from selection tables.

        Parameters
        ----------
        backend : DataBackend
            Backend wrapping DataFrame with a column containing selection table TSV strings.

        Returns
        -------
        tuple[DataBackend, dict]
            The backend with a new species_list column, and empty metadata dict.

        Raises
        ------
        KeyError
            If the `selection_table_column` is not found in the DataFrame.
        """
        if self.selection_table_column not in backend.columns:
            raise KeyError(
                f"Column '{self.selection_table_column}' not found in DataFrame. "
                f"Available columns: {list(backend.columns)}"
            )

        species_lists = []
        for row in backend:
            selection_table_str = row[self.selection_table_column]
            if not selection_table_str or not isinstance(selection_table_str, str):
                species_lists.append([])
                continue

            try:
                # Use strip('\n\r') instead of strip() to preserve leading tabs.
                # Some datasets have an unnamed index column that starts with a tab.
                lines = selection_table_str.strip("\n\r").split("\n")
                if len(lines) < 2:
                    species_lists.append([])
                    continue

                header = lines[0].split("\t")
                species_idx = header.index(self.species_column)

                species_set = set()
                for line in lines[1:]:
                    fields = line.split("\t")
                    if len(fields) > species_idx:
                        species = fields[species_idx].strip()
                        if species:
                            species_set.add(species)

                species_lists.append(sorted(species_set))
            except Exception:
                species_lists.append([])

        return backend.add_column(self.output_column, species_lists), {}


class SpeciesListFromColumnConfig(BaseModel):
    """Configuration for SpeciesListFromColumn transform."""

    type: Literal["species_list_from_column"] = "species_list_from_column"
    input_column: str = "canonical_name_multispecies"
    output_column: str = "species_list"


class SpeciesListFromColumn:
    """Extract a list of unique species from a JSON-list metadata column.

    For each row, parses ``input_column`` (a JSON-encoded list of species names,
    e.g. BirdSet's ``canonical_name_multispecies``) into a list of unique species
    stored in ``output_column``. Values that are already lists are used as-is;
    empty or unparsable values yield an empty list.

    Parameters
    ----------
    input_column : str
        Column containing the species names as a JSON list string (or an actual
        list). Default is ``"canonical_name_multispecies"`` (BirdSet format).
    output_column : str
        Name of the new column to store the species list. Default is
        ``"species_list"``.
    """

    def __init__(
        self,
        *,
        input_column: str = "canonical_name_multispecies",
        output_column: str = "species_list",
    ) -> None:
        self.input_column = input_column
        self.output_column = output_column

    @classmethod
    def from_config(cls, cfg: SpeciesListFromColumnConfig) -> SpeciesListFromColumn:
        return cls(**cfg.model_dump(exclude={"type"}))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """Extract species lists from a JSON-list metadata column.

        Parameters
        ----------
        backend : DataBackend
            Backend wrapping a DataFrame with the ``input_column``.

        Returns
        -------
        tuple[DataBackend, dict]
            The backend with a new species-list column, and an empty metadata dict.

        Raises
        ------
        KeyError
            If the `input_column` is not found in the DataFrame.
        """
        if self.input_column not in backend.columns:
            raise KeyError(
                f"Column '{self.input_column}' not found in DataFrame. Available columns: {list(backend.columns)}"
            )

        species_lists = []
        for row in backend:
            value = row[self.input_column]
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except (ValueError, TypeError):
                    value = []
            if not isinstance(value, (list, tuple)):
                value = []

            species_set = {str(species).strip() for species in value if str(species).strip()}
            species_lists.append(sorted(species_set))

        return backend.add_column(self.output_column, species_lists), {}


class CapPerGroupConfig(BaseModel):
    """Configuration for the CapPerGroup transform."""

    type: Literal["cap_per_group"] = "cap_per_group"
    property: str
    count: int
    seed: int = 42


class CapPerGroup:
    """Cap every category of a column to at most `count` rows.

    A fixed-count balance: unlike `alp_data`'s ``balanced_sample`` (whose target
    is derived from the data via a min/median/mean strategy), this caps each
    category at an explicit `count`. Categories already at or below the cap are
    left untouched; larger ones are downsampled (without replacement, seeded) to
    exactly `count`. Used to carve a small, species-balanced slice (e.g. 20 files
    per species) out of a larger dataset for the LSI example.

    Parameters
    ----------
    property : str
        Column to group by (e.g. ``"canonical_name"``).
    count : int
        Maximum number of rows to keep per category.
    seed : int
        Random seed for the downsample, for reproducibility. Defaults to ``42``.
    """

    def __init__(self, property: str, count: int, seed: int = 42) -> None:
        self.property = property
        self.count = count
        self.seed = seed

    @classmethod
    def from_config(cls, cfg: CapPerGroupConfig) -> CapPerGroup:
        return cls(**cfg.model_dump(exclude={"type"}))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """Cap each category to at most `count` rows.

        Parameters
        ----------
        backend : DataBackend
            Backend wrapping the DataFrame to cap.

        Returns
        -------
        tuple[DataBackend, dict]
            The capped backend (same type as input) and an empty metadata dict.

        Raises
        ------
        KeyError
            If `property` is not found in the DataFrame columns.
        """
        if self.property not in backend.columns:
            raise KeyError(
                f"Property '{self.property}' not found in DataFrame. Available columns: {list(backend.columns)}"
            )
        # Deterministic downsample done here rather than via
        # `backend.upsample_by_column`: that sampler shares one RNG across groups
        # but visits the groups in a non-deterministic order, so even with a fixed
        # `seed` and a stably ordered input it selects a different subset on every
        # load. That silently breaks any run-then-reload workflow keyed on row
        # identity (e.g. attaching LSI outputs onto a capped dataset). We instead
        # group row indices in the backend's (stable) order, visit groups in sorted
        # key order, and draw each group's sample with its own seeded RNG — fully
        # reproducible given the same input.
        groups: dict[object, list[int]] = {}
        for index, row in enumerate(backend):
            groups.setdefault(row[self.property], []).append(index)

        keep: list[int] = []
        for position, key in enumerate(sorted(groups, key=repr)):
            indices = groups[key]
            if len(indices) <= self.count:
                keep.extend(indices)
            else:
                # Seed by sorted-key position (not hash(key): Python's str hash is
                # per-process salted, which would defeat reproducibility).
                rng = np.random.default_rng([self.seed, position])
                chosen = rng.choice(len(indices), size=self.count, replace=False)
                keep.extend(indices[i] for i in sorted(int(c) for c in chosen))

        return backend[sorted(keep)], {}


class AttachLSISelectionTablesConfig(BaseModel):
    """Configuration for the AttachLSISelectionTables transform."""

    type: Literal["attach_lsi_selection_tables"] = "attach_lsi_selection_tables"
    run_root: str
    postprocessing: str
    id_column: str | None = None
    output_column: str = "selection_table"
    shard_column: str = "lsi_shard"
    attach_quality: bool = True
    confidence_column: str = "focal_confidence"
    max_stems_column: str = "focal_max_stems"


class AttachLSISelectionTables:
    """Attach large-scale-inference outputs alongside each dataset row.

    Adds two small string columns, mirroring the old ``AddPrecomputedSelectionTables``
    but also emitting a pointer to the heavy shard:

    * `output_column` (default ``"selection_table"``) — the recording's
      selection-table TSV string, read from the
      postprocessing stage's ST shards. Rows with no match get ``""``.
    * `shard_column` (default ``"lsi_shard"``) — the path to the recording's
      heavy `ItemResult` shard under `run_root`, so a consumer can load denoised
      / stem audio and framewise predictions on demand (see
      `sound_event_detection.inference.access`). Rows with no match get ``""``.

    When the postprocessing stage recorded per-recording quality (denoised/stems
    runs only), two more columns are attached **by presence** — skipped entirely
    for a preds run that has no quality:

    * `confidence_column` (default ``"focal_confidence"``) — mean focal detection
      probability, and `max_stems_column` (default ``"focal_max_stems"``) — the
      most stems combined at any frame. Rows with no match get ``nan``. Set
      `attach_quality=False` to never attach them.

    The columns attached are otherwise flexible: the selection-table TSV keeps a
    ``Score`` per event by default, but that schema is owned by the postprocessing
    stage, not this transform, and nothing here requires it.

    The pointer is resolved without any manifest by relying on the postprocessing
    stage writing ST shards 1:1 with the items shards: a recording found in
    ``{run_root}/{postprocessing}/shard_0007.npz`` has its heavy data in
    ``{run_root}/shard_0007.npz``.

    Focal species is deliberately NOT attached — it comes from the source
    dataset's own column via the id join, which is the point of LSI.

    Parameters
    ----------
    run_root : str
        Directory (local or cloud) holding the LSI `ItemResult` ``shard_*.npz``.
    postprocessing : str
        Name of the postprocessing subdirectory of `run_root` holding the
        selection-table shards to attach.
    id_column : str or None
        Column whose stringified value is the shard file id. When ``None``, the
        first of `_DEFAULT_ID_COLUMNS` present in the data is used; it must match
        the ``id_column`` the producer keyed its shards by.
    output_column : str
        Name of the attached selection-table column. Default ``"selection_table"``.
    shard_column : str
        Name of the attached heavy-shard pointer column. Default ``"lsi_shard"``.
    attach_quality : bool
        When ``True`` (default) attach the pooled quality columns if the ST
        shards carry them; when ``False`` never attach them.
    confidence_column : str
        Name of the attached mean-confidence column. Default ``"focal_confidence"``.
    max_stems_column : str
        Name of the attached max-stem-count column. Default ``"focal_max_stems"``.
    """

    def __init__(
        self,
        *,
        run_root: str,
        postprocessing: str,
        id_column: str | None = None,
        output_column: str = "selection_table",
        shard_column: str = "lsi_shard",
        attach_quality: bool = True,
        confidence_column: str = "focal_confidence",
        max_stems_column: str = "focal_max_stems",
    ) -> None:
        self.run_root = run_root.rstrip("/")
        self.postprocessing = postprocessing
        self.id_column = id_column
        self.output_column = output_column
        self.shard_column = shard_column
        self.attach_quality = attach_quality
        self.confidence_column = confidence_column
        self.max_stems_column = max_stems_column

    @classmethod
    def from_config(cls, cfg: AttachLSISelectionTablesConfig) -> AttachLSISelectionTables:
        return cls(**cfg.model_dump(exclude={"type"}))

    def _resolve_id_column(self, backend: DataBackend) -> str:
        """Return the id column to key rows by, validating it exists.

        Parameters
        ----------
        backend : DataBackend
            The backend being transformed.

        Returns
        -------
        str
            The resolved id column name.

        Raises
        ------
        KeyError
            If an explicit `id_column` is not among the backend columns.
        ValueError
            If `id_column` is ``None`` and none of `_DEFAULT_ID_COLUMNS` is present.
        """
        columns = list(backend.columns)
        if self.id_column is not None:
            if self.id_column not in columns:
                raise KeyError(f"id_column '{self.id_column}' not found. Available columns: {columns}")
            return self.id_column
        for candidate in _DEFAULT_ID_COLUMNS:
            if candidate in columns:
                return candidate
        raise ValueError(
            f"could not infer id_column from {_DEFAULT_ID_COLUMNS}; pass id_column explicitly. "
            f"Available columns: {columns}"
        )

    def _build_index(self) -> tuple[dict[str, str], dict[str, str], dict[str, float], dict[str, float]]:
        """Load ST shards into per-file-id maps.

        Returns
        -------
        tuple[dict[str, str], dict[str, str], dict[str, float], dict[str, float]]
            The selection-table strings, the heavy-shard pointers, and the pooled
            quality scalars (``focal_confidence`` and ``focal_max_stems``), each
            keyed by file id. The two quality maps are empty for a preds run whose
            ST shards carry no quality.

        Raises
        ------
        ValueError
            If no ``shard_*.npz`` are found under the postprocessing directory.
        """
        from sound_event_detection.inference.engine import read_shard

        pp_dir = f"{self.run_root}/{self.postprocessing}"
        if not isinstance(anypath(pp_dir), Path):
            fs = filesystem_from_path(pp_dir)
            scheme = pp_dir.split("://", 1)[0]
            paths = [f"{scheme}://{match}" for match in fs.glob(f"{fs._strip_protocol(pp_dir)}/shard_*.npz")]
        else:
            paths = [str(path) for path in Path(pp_dir).expanduser().glob("shard_*.npz")]
        if not paths:
            raise ValueError(f"no shard_*.npz found under {pp_dir!r}")

        tables: dict[str, str] = {}
        shards: dict[str, str] = {}
        confidence: dict[str, float] = {}
        max_stems: dict[str, float] = {}
        for path in paths:
            match = _SHARD_RE.search(path)
            if not match:
                continue
            items_shard = f"{self.run_root}/shard_{int(match.group(1)):04d}.npz"
            for file_id, arrays in read_shard(path).items():
                tables[file_id] = str(arrays["selection_table"].item())
                shards[file_id] = items_shard
                if "focal_confidence" in arrays:
                    confidence[file_id] = float(arrays["focal_confidence"].item())
                if "focal_max_stems" in arrays:
                    max_stems[file_id] = float(arrays["focal_max_stems"].item())
        return tables, shards, confidence, max_stems

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """Attach the selection-table and heavy-shard columns to `backend`.

        Parameters
        ----------
        backend : DataBackend
            Backend whose rows carry the id column keyed by the LSI producer.

        Returns
        -------
        tuple[DataBackend, dict]
            The backend with the `output_column` and `shard_column` added (plus
            the quality columns when `attach_quality` is set and the run recorded
            them), and a ``{"matched", "unmatched"}`` count of rows that found a
            selection table (``"quality_attached"`` is added when the quality
            columns were attached).

        Warns
        -----
        UserWarning
            When some — but not all — rows found no LSI output (partial coverage):
            their attached columns are left empty. This is expected when the
            dataset covers recordings the run did not process.

        Raises
        ------
        ValueError
            When the backend is non-empty yet *no* row matched any LSI output,
            which signals a broken join (wrong `run_root`/`postprocessing`, or an
            `id_column` that disagrees with the one the producer keyed its shards
            by) rather than mere partial coverage. `_resolve_id_column` and
            `_build_index` may also raise (see their own docstrings) before the
            join is attempted.
        """
        id_column = self._resolve_id_column(backend)
        tables, shards, confidence, max_stems = self._build_index()

        selection_tables: list[str] = []
        pointers: list[str] = []
        confidences: list[float] = []
        max_stems_col: list[float] = []
        matched = 0
        for row in backend:
            file_id = str(row[id_column])
            tsv = tables.get(file_id, "")
            selection_tables.append(tsv)
            pointers.append(shards.get(file_id, ""))
            confidences.append(confidence.get(file_id, float("nan")))
            max_stems_col.append(max_stems.get(file_id, float("nan")))
            if tsv:
                matched += 1

        total = len(selection_tables)
        unmatched = total - matched
        pp_dir = f"{self.run_root}/{self.postprocessing}"
        if total and matched == 0:
            raise ValueError(
                f"AttachLSISelectionTables matched 0 of {total} rows against the LSI output under "
                f"{pp_dir!r} (id_column={id_column!r}). Every row's {id_column!r} value failed to match a "
                f"produced recording id: check that run_root/postprocessing point at the right run and that "
                f"id_column matches the one the producer keyed its shards by."
            )
        if unmatched:
            warnings.warn(
                f"AttachLSISelectionTables: {unmatched} of {total} rows found no LSI output under {pp_dir!r} "
                f"(id_column={id_column!r}); their {self.output_column!r}/{self.shard_column!r} columns are "
                f"empty. This is expected if the dataset covers recordings the run did not process.",
                stacklevel=2,
            )

        backend = backend.add_column(self.output_column, selection_tables)
        backend = backend.add_column(self.shard_column, pointers)
        meta = {"matched": matched, "unmatched": unmatched}
        # Attach quality by presence: only when enabled and the run actually
        # recorded it (denoised/stems runs populate `confidence`; preds runs leave
        # it empty, so the columns are skipped entirely).
        if self.attach_quality and confidence:
            backend = backend.add_column(self.confidence_column, confidences)
            backend = backend.add_column(self.max_stems_column, max_stems_col)
            meta["quality_attached"] = True
        return backend, meta


try:
    from alp_data.transforms import register_transform

    register_transform(SpeciesListFromSelectionTableConfig, SpeciesListFromSelectionTable)
    register_transform(SpeciesListFromColumnConfig, SpeciesListFromColumn)
    register_transform(CapPerGroupConfig, CapPerGroup)
    register_transform(AttachLSISelectionTablesConfig, AttachLSISelectionTables)
except ImportError:
    pass
