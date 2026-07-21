"""Hydrologic stage-event detection using HydroEventDetector.

This module wraps :class:`hydro_event_detector.HydroEventDetector` so the
USGS gage QC package and a future graphical application can use the event
workflow through a stable, file-oriented API.

The workflow intentionally preserves three event collections:

1. ``detected_events``: every event identified from crossings between the
   observed stage and the Lyne-Hollick baseflow estimate.
2. ``filtered_events``: events retained by
   ``HydroEventDetector.filter_events(event_filter_percentile)``.
3. ``selected_events``: filtered events ordered using one of three user-
   selectable ranking methods and optionally limited to ``top_n_events``:

   - ``flow_peak``: largest absolute peak stage.
   - ``peak_quick_stage``: largest rise above baseflow.
   - ``combined``: weighted combination of percentile-rank-normalized absolute
     peak stage and rise above baseflow.

The original HydroEventDetector event columns are retained so the outputs can
also be consumed by ``mrms_usgs_events`` and other existing workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    from hydro_event_detector import HydroEventDetector
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    raise ImportError(
        "stage_event_detection requires the 'hydro-event-detector' package. "
        "Install it with: pip install hydro-event-detector"
    ) from exc


DEFAULT_BASEFLOW_ALPHA = 0.987
DEFAULT_EVENT_FILTER_PERCENTILE = 50.0
DEFAULT_TOP_N_EVENTS = 60
DEFAULT_RANKING_METHOD = "combined"
DEFAULT_FLOW_PEAK_WEIGHT = 0.5
DEFAULT_QUICK_STAGE_WEIGHT = 0.5

VALID_RANKING_METHODS = frozenset(
    {"flow_peak", "peak_quick_stage", "combined"}
)

_REQUIRED_INPUT_COLUMNS = ("datetime", "Stage_ft")
_HYDROEVENTDETECTOR_EVENT_COLUMNS = (
    "date_start",
    "flow_start",
    "baseflow_start",
    "date_peak",
    "flow_peak",
    "baseflow_peak",
    "date_end",
    "flow_end",
    "baseflow_end",
    "month_num",
    "month_name",
    "event_volume",
    "baseflow_volume",
)
_INDEX_COLUMNS = ("i_start", "i_peak", "i_end")


@dataclass(frozen=True)
class StageEventDetectionResult:
    """Results and saved paths produced by :func:`detect_stage_events`."""

    site_id: str | None
    output_directory: Path
    baseflow_alpha: float
    event_filter_percentile: float
    top_n_events: int | None
    ranking_method: str
    flow_peak_weight: float
    quick_stage_weight: float

    number_of_observations: int
    number_of_detected_events: int
    number_of_filtered_events: int
    number_of_selected_events: int

    baseflow: pd.DataFrame
    detected_events: pd.DataFrame
    filtered_events: pd.DataFrame
    selected_events: pd.DataFrame

    baseflow_path: Path
    detected_events_path: Path
    filtered_events_path: Path
    selected_events_path: Path
    manifest_path: Path


def _validate_parameters(
    *,
    baseflow_alpha: float,
    event_filter_percentile: float,
    top_n_events: int | None,
    ranking_method: str,
    flow_peak_weight: float,
    quick_stage_weight: float,
) -> tuple[float, float, int | None, str, float, float]:
    """Validate and normalize user-facing event-detection parameters."""

    try:
        alpha = float(baseflow_alpha)
    except (TypeError, ValueError) as exc:
        raise TypeError("baseflow_alpha must be a numeric value.") from exc

    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("baseflow_alpha must be greater than 0 and less than 1.")

    try:
        percentile = float(event_filter_percentile)
    except (TypeError, ValueError) as exc:
        raise TypeError("event_filter_percentile must be a numeric value.") from exc

    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError(
            "event_filter_percentile must be between 0 and 100, inclusive."
        )

    normalized_top_n: int | None
    if top_n_events is None:
        normalized_top_n = None
    else:
        if isinstance(top_n_events, bool):
            raise TypeError("top_n_events must be a positive integer or None.")
        try:
            numeric_top_n = float(top_n_events)
        except (TypeError, ValueError) as exc:
            raise TypeError("top_n_events must be a positive integer or None.") from exc
        if not np.isfinite(numeric_top_n) or not numeric_top_n.is_integer():
            raise ValueError("top_n_events must be a positive integer or None.")
        normalized_top_n = int(numeric_top_n)
        if normalized_top_n <= 0:
            raise ValueError("top_n_events must be greater than 0 or None.")

    normalized_ranking_method = str(ranking_method).strip().lower()
    if normalized_ranking_method not in VALID_RANKING_METHODS:
        expected = ", ".join(sorted(VALID_RANKING_METHODS))
        raise ValueError(
            f"ranking_method must be one of: {expected}."
        )

    try:
        flow_weight = float(flow_peak_weight)
        quick_weight = float(quick_stage_weight)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "flow_peak_weight and quick_stage_weight must be numeric values."
        ) from exc

    if not np.isfinite(flow_weight) or flow_weight < 0.0:
        raise ValueError("flow_peak_weight must be finite and non-negative.")
    if not np.isfinite(quick_weight) or quick_weight < 0.0:
        raise ValueError("quick_stage_weight must be finite and non-negative.")

    total_weight = flow_weight + quick_weight
    if total_weight <= 0.0:
        raise ValueError(
            "At least one ranking weight must be greater than zero."
        )

    # Store normalized weights so the combined score always remains on 0-1.
    flow_weight /= total_weight
    quick_weight /= total_weight

    return (
        alpha,
        percentile,
        normalized_top_n,
        normalized_ranking_method,
        flow_weight,
        quick_weight,
    )


def _prepare_stage_input(stage: pd.DataFrame) -> pd.DataFrame:
    """Return a validated HydroEventDetector input dataframe.

    The returned dataframe always contains exactly ``datetime`` and
    ``Stage_ft``, is sorted by datetime, uses timezone-naive timestamps, and
    contains no duplicate, missing, or non-finite observations.
    """

    if not isinstance(stage, pd.DataFrame):
        raise TypeError("stage must be a pandas DataFrame.")

    missing = [column for column in _REQUIRED_INPUT_COLUMNS if column not in stage]
    if missing:
        raise ValueError(
            "Stage input is missing required columns: " + ", ".join(missing)
        )

    prepared = stage.loc[:, list(_REQUIRED_INPUT_COLUMNS)].copy()
    prepared["datetime"] = pd.to_datetime(prepared["datetime"], errors="coerce", utc=True)
    prepared["Stage_ft"] = pd.to_numeric(prepared["Stage_ft"], errors="coerce")

    invalid_datetime = prepared["datetime"].isna()
    invalid_stage = prepared["Stage_ft"].isna() | ~np.isfinite(prepared["Stage_ft"])
    invalid = invalid_datetime | invalid_stage
    if invalid.any():
        raise ValueError(
            "HydroEventDetector input contains "
            f"{int(invalid.sum())} invalid datetime or Stage_ft observation(s). "
            "Run stage_download/stage_gap_processing first or clean the input."
        )

    prepared["datetime"] = prepared["datetime"].dt.tz_localize(None)
    prepared = prepared.sort_values("datetime", kind="mergesort").reset_index(drop=True)

    duplicate_mask = prepared["datetime"].duplicated(keep=False)
    if duplicate_mask.any():
        raise ValueError(
            "HydroEventDetector input contains duplicated timestamps. "
            "Use stage_hydroeventdetector.parquet generated by stage_download.py."
        )

    if len(prepared) < 3:
        raise ValueError("At least three valid observations are required.")

    return prepared


def _empty_event_dataframe() -> pd.DataFrame:
    """Create a stable empty event dataframe for files and application code."""

    columns = [
        "site_id",
        "event_id",
        "peak_rank",
        "selected_rank",
        *_INDEX_COLUMNS,
        *_HYDROEVENTDETECTOR_EVENT_COLUMNS,
        "peak_quick_stage_ft",
        "flow_peak_normalized",
        "peak_quick_stage_normalized",
        "combined_normalized_score",
        "flow_peak_rank",
        "peak_quick_stage_rank",
        "combined_rank",
        "flow_peak_weight",
        "quick_stage_weight",
        "ranking_method",
        "event_filter_percentile",
        "is_selected",
    ]
    return pd.DataFrame(columns=columns)


def _combine_event_outputs(
    event_dataframe: pd.DataFrame | None,
    event_indices: pd.DataFrame | None,
    *,
    site_id: str | None,
    event_filter_percentile: float,
    selected: bool,
) -> pd.DataFrame:
    """Combine HydroEventDetector statistics and index positions by row order."""

    statistics = (
        event_dataframe.copy().reset_index(drop=True)
        if isinstance(event_dataframe, pd.DataFrame)
        else pd.DataFrame()
    )
    indices = (
        event_indices.copy().reset_index(drop=True)
        if isinstance(event_indices, pd.DataFrame)
        else pd.DataFrame(columns=list(_INDEX_COLUMNS))
    )

    if statistics.empty and indices.empty:
        return _empty_event_dataframe()

    if len(statistics) != len(indices):
        raise RuntimeError(
            "HydroEventDetector returned inconsistent event statistics and "
            f"indices ({len(statistics)} versus {len(indices)} rows)."
        )

    for column in _INDEX_COLUMNS:
        if column not in indices:
            raise RuntimeError(
                f"HydroEventDetector event indices are missing '{column}'."
            )

    combined = pd.concat(
        [indices.loc[:, list(_INDEX_COLUMNS)], statistics],
        axis=1,
    )

    for column in ("date_start", "date_peak", "date_end"):
        if column in combined:
            combined[column] = pd.to_datetime(combined[column], errors="coerce")

    if {"flow_peak", "baseflow_peak"}.issubset(combined.columns):
        combined["peak_quick_stage_ft"] = (
            pd.to_numeric(combined["flow_peak"], errors="coerce")
            - pd.to_numeric(combined["baseflow_peak"], errors="coerce")
        )
    else:
        combined["peak_quick_stage_ft"] = np.nan

    combined.insert(0, "event_id", np.arange(1, len(combined) + 1, dtype=int))
    combined.insert(0, "site_id", site_id)
    combined["peak_rank"] = pd.Series(pd.NA, index=combined.index, dtype="Int64")
    combined["selected_rank"] = pd.Series(
        pd.NA, index=combined.index, dtype="Int64"
    )
    combined["ranking_method"] = pd.NA
    combined["event_filter_percentile"] = float(event_filter_percentile)
    combined["is_selected"] = bool(selected)

    return combined


def _percentile_rank_0_1(values: pd.Series) -> pd.Series:
    """Return percentile-rank normalization on 0-1 while preserving missingness."""

    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=values.index, dtype=float)
    valid = numeric.notna() & np.isfinite(numeric)
    if valid.any():
        output.loc[valid] = numeric.loc[valid].rank(
            method="average",
            pct=True,
            ascending=True,
        )
    return output


def add_event_ranking_metrics(
    events: pd.DataFrame,
    *,
    flow_peak_weight: float = DEFAULT_FLOW_PEAK_WEIGHT,
    quick_stage_weight: float = DEFAULT_QUICK_STAGE_WEIGHT,
) -> pd.DataFrame:
    """Add all three app-ready event scores and rankings.

    Normalized values use percentile ranks and therefore lie on 0-1. The
    combined score is a weighted average of the two normalized metrics.
    Rank 1 always represents the largest or most important event.
    """

    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame.")

    missing = [
        column
        for column in ("flow_peak", "baseflow_peak")
        if column not in events.columns
    ]
    if missing:
        raise ValueError(
            "Event data are missing required columns: " + ", ".join(missing)
        )

    try:
        flow_weight = float(flow_peak_weight)
        quick_weight = float(quick_stage_weight)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "flow_peak_weight and quick_stage_weight must be numeric values."
        ) from exc

    if not np.isfinite(flow_weight) or flow_weight < 0.0:
        raise ValueError("flow_peak_weight must be finite and non-negative.")
    if not np.isfinite(quick_weight) or quick_weight < 0.0:
        raise ValueError("quick_stage_weight must be finite and non-negative.")

    total_weight = flow_weight + quick_weight
    if total_weight <= 0.0:
        raise ValueError(
            "At least one ranking weight must be greater than zero."
        )

    flow_weight /= total_weight
    quick_weight /= total_weight

    ranked = events.copy()
    ranked["flow_peak"] = pd.to_numeric(
        ranked["flow_peak"], errors="coerce"
    )
    ranked["baseflow_peak"] = pd.to_numeric(
        ranked["baseflow_peak"], errors="coerce"
    )
    ranked["peak_quick_stage_ft"] = (
        ranked["flow_peak"] - ranked["baseflow_peak"]
    )

    ranked["flow_peak_normalized"] = _percentile_rank_0_1(
        ranked["flow_peak"]
    )
    ranked["peak_quick_stage_normalized"] = _percentile_rank_0_1(
        ranked["peak_quick_stage_ft"]
    )
    ranked["combined_normalized_score"] = (
        flow_weight * ranked["flow_peak_normalized"]
        + quick_weight * ranked["peak_quick_stage_normalized"]
    )

    ranked["flow_peak_rank"] = ranked["flow_peak"].rank(
        method="min", ascending=False, na_option="bottom"
    ).astype("Int64")
    ranked["peak_quick_stage_rank"] = ranked["peak_quick_stage_ft"].rank(
        method="min", ascending=False, na_option="bottom"
    ).astype("Int64")
    ranked["combined_rank"] = ranked["combined_normalized_score"].rank(
        method="min", ascending=False, na_option="bottom"
    ).astype("Int64")

    ranked["flow_peak_weight"] = flow_weight
    ranked["quick_stage_weight"] = quick_weight
    return ranked


def select_ranked_events(
    events: pd.DataFrame,
    *,
    ranking_method: str = DEFAULT_RANKING_METHOD,
    top_n_events: int | None = DEFAULT_TOP_N_EVENTS,
) -> pd.DataFrame:
    """Order and optionally limit events using a user-selectable ranking."""

    method = str(ranking_method).strip().lower()
    if method not in VALID_RANKING_METHODS:
        expected = ", ".join(sorted(VALID_RANKING_METHODS))
        raise ValueError(f"ranking_method must be one of: {expected}.")

    if top_n_events is not None:
        if isinstance(top_n_events, bool) or not isinstance(top_n_events, int):
            raise TypeError("top_n_events must be a positive integer or None.")
        if top_n_events <= 0:
            raise ValueError("top_n_events must be greater than 0 or None.")

    if events.empty:
        selected = events.copy()
        selected["ranking_method"] = method
        return selected

    sort_definitions = {
        "flow_peak": (
            ["flow_peak", "peak_quick_stage_ft", "date_peak"],
            [False, False, True],
        ),
        "peak_quick_stage": (
            ["peak_quick_stage_ft", "flow_peak", "date_peak"],
            [False, False, True],
        ),
        "combined": (
            [
                "combined_normalized_score",
                "flow_peak",
                "peak_quick_stage_ft",
                "date_peak",
            ],
            [False, False, False, True],
        ),
    }
    sort_columns, ascending = sort_definitions[method]
    missing = [column for column in sort_columns if column not in events.columns]
    if missing:
        raise RuntimeError(
            "Ranked event data are missing required columns: "
            + ", ".join(missing)
        )

    selected = events.sort_values(
        sort_columns,
        ascending=ascending,
        kind="mergesort",
        na_position="last",
    )
    if top_n_events is not None:
        selected = selected.head(top_n_events)

    selected = selected.copy().reset_index(drop=True)
    rank_values = pd.Series(
        np.arange(1, len(selected) + 1, dtype=int),
        dtype="Int64",
    )
    # peak_rank is retained for backward compatibility.
    selected["selected_rank"] = rank_values
    selected["peak_rank"] = rank_values
    selected["ranking_method"] = method
    selected["is_selected"] = True
    return selected


def _build_baseflow_dataframe(
    stage: pd.DataFrame,
    baseflow: Any,
    *,
    site_id: str | None,
) -> pd.DataFrame:
    """Create a visualization-ready stage and baseflow time series."""

    baseflow_array = np.asarray(baseflow, dtype=float)
    if baseflow_array.ndim != 1 or len(baseflow_array) != len(stage):
        raise RuntimeError(
            "HydroEventDetector returned a baseflow array with an unexpected shape."
        )

    output = stage.copy()
    output.insert(0, "site_id", site_id)
    output["baseflow_ft"] = baseflow_array
    output["stage_above_baseflow_ft"] = output["Stage_ft"] - output["baseflow_ft"]
    return output


def _package_version(distribution_name: str) -> str | None:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(manifest), indent=2, default=str),
        encoding="utf-8",
    )


def detect_stage_events_from_dataframe(
    stage: pd.DataFrame,
    output_directory: str | Path,
    *,
    site_id: str | None = None,
    baseflow_alpha: float = DEFAULT_BASEFLOW_ALPHA,
    event_filter_percentile: float = DEFAULT_EVENT_FILTER_PERCENTILE,
    top_n_events: int | None = DEFAULT_TOP_N_EVENTS,
    ranking_method: str = DEFAULT_RANKING_METHOD,
    flow_peak_weight: float = DEFAULT_FLOW_PEAK_WEIGHT,
    quick_stage_weight: float = DEFAULT_QUICK_STAGE_WEIGHT,
    input_path: str | Path | None = None,
) -> StageEventDetectionResult:
    """Detect, filter, rank, save, and return water-stage events.

    Parameters
    ----------
    stage:
        DataFrame containing ``datetime`` and ``Stage_ft``.
    output_directory:
        Directory where processed Parquet files and the manifest are written.
    site_id:
        Optional USGS site identifier. Keep it as a string to preserve leading
        zeros.
    baseflow_alpha:
        Lyne-Hollick recursive-filter coefficient passed unchanged to
        ``HydroEventDetector.baseflow_lyne_hollick``.
    event_filter_percentile:
        Percentile passed unchanged to ``HydroEventDetector.filter_events``.
        HydroEventDetector applies it to peak quick flow.
    top_n_events:
        Maximum number of filtered events retained. Use ``None`` to retain
        every filtered event.
    ranking_method:
        Event ordering exposed to the future app: ``flow_peak``,
        ``peak_quick_stage``, or ``combined``.
    flow_peak_weight, quick_stage_weight:
        Non-negative weights for the combined score. They are normalized to
        sum to 1.0. They do not affect the two single-metric rankings.
    input_path:
        Optional source path recorded in the manifest.
    """

    (
        alpha,
        percentile,
        normalized_top_n,
        normalized_ranking_method,
        normalized_flow_weight,
        normalized_quick_weight,
    ) = _validate_parameters(
        baseflow_alpha=baseflow_alpha,
        event_filter_percentile=event_filter_percentile,
        top_n_events=top_n_events,
        ranking_method=ranking_method,
        flow_peak_weight=flow_peak_weight,
        quick_stage_weight=quick_stage_weight,
    )
    prepared = _prepare_stage_input(stage)

    normalized_site_id = None if site_id is None else str(site_id).strip()
    if normalized_site_id == "":
        normalized_site_id = None

    output_dir = Path(output_directory).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = HydroEventDetector(
        date_range=prepared["datetime"].to_numpy(),
        streamflow=prepared["Stage_ft"].to_numpy(dtype=float),
    )

    detector.baseflow_lyne_hollick(alpha=alpha)
    detector.detect_events()
    detector.create_events_dataframe()

    detected_events = _combine_event_outputs(
        detector.dataframe,
        detector.events,
        site_id=normalized_site_id,
        event_filter_percentile=percentile,
        selected=False,
    )

    # filter_events mutates detector.events. Recreate the statistics dataframe
    # afterward exactly as recommended by HydroEventDetector.
    detector.filter_events(percentile)
    detector.create_events_dataframe()

    filtered_events = _combine_event_outputs(
        detector.dataframe,
        detector.events,
        site_id=normalized_site_id,
        event_filter_percentile=percentile,
        selected=False,
    )
    # Ranking metrics are calculated after HydroEventDetector filtering so the
    # app compares the same eligible event population for all three methods.
    filtered_events = add_event_ranking_metrics(
        filtered_events,
        flow_peak_weight=normalized_flow_weight,
        quick_stage_weight=normalized_quick_weight,
    )
    selected_events = select_ranked_events(
        filtered_events,
        ranking_method=normalized_ranking_method,
        top_n_events=normalized_top_n,
    )
    baseflow = _build_baseflow_dataframe(
        prepared,
        detector.baseflow,
        site_id=normalized_site_id,
    )

    baseflow_path = output_dir / "stage_baseflow.parquet"
    detected_events_path = output_dir / "stage_events_detected.parquet"
    filtered_events_path = output_dir / "stage_events_filtered.parquet"
    selected_events_path = output_dir / "stage_events_selected.parquet"
    manifest_path = output_dir / "event_detection_manifest.json"

    baseflow.to_parquet(baseflow_path, index=False)
    detected_events.to_parquet(detected_events_path, index=False)
    filtered_events.to_parquet(filtered_events_path, index=False)
    selected_events.to_parquet(selected_events_path, index=False)

    manifest = {
        "module": "hydro_event_detector",
        "site_id": normalized_site_id,
        "input_path": str(Path(input_path).expanduser()) if input_path else None,
        "output_directory": str(output_dir),
        "baseflow_method": "Lyne-Hollick",
        "baseflow_alpha": alpha,
        "event_definition": "Stage_ft > baseflow_ft",
        "event_filter_method": "HydroEventDetector.filter_events",
        "event_filter_metric": "peak quick flow",
        "event_filter_percentile": percentile,
        "ranking_method": normalized_ranking_method,
        "available_ranking_methods": sorted(VALID_RANKING_METHODS),
        "ranking_normalization": "percentile_rank_0_1",
        "flow_peak_weight": normalized_flow_weight,
        "quick_stage_weight": normalized_quick_weight,
        "combined_score_formula": (
            "flow_peak_weight * flow_peak_normalized + "
            "quick_stage_weight * peak_quick_stage_normalized"
        ),
        "event_sort_order": "descending",
        "top_n_events": normalized_top_n,
        "number_of_observations": int(len(prepared)),
        "number_of_detected_events": int(len(detected_events)),
        "number_of_filtered_events": int(len(filtered_events)),
        "number_of_selected_events": int(len(selected_events)),
        "files": {
            "baseflow": baseflow_path.name,
            "detected_events": detected_events_path.name,
            "filtered_events": filtered_events_path.name,
            "selected_events": selected_events_path.name,
        },
        "schemas": {
            "baseflow_columns": baseflow.columns.tolist(),
            "detected_event_columns": detected_events.columns.tolist(),
            "filtered_event_columns": filtered_events.columns.tolist(),
            "selected_event_columns": selected_events.columns.tolist(),
        },
        "interoperability": {
            "hydroeventdetector_columns_preserved": [
                column
                for column in _HYDROEVENTDETECTOR_EVENT_COLUMNS
                if column in selected_events.columns
            ],
            "mrms_usgs_events_peak_columns": ["date_peak", "flow_peak"],
        },
        "software": {
            "hydro_event_detector_version": _package_version(
                "hydro-event-detector"
            ),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        },
    }
    _write_manifest(manifest_path, manifest)

    return StageEventDetectionResult(
        site_id=normalized_site_id,
        output_directory=output_dir,
        baseflow_alpha=alpha,
        event_filter_percentile=percentile,
        top_n_events=normalized_top_n,
        ranking_method=normalized_ranking_method,
        flow_peak_weight=normalized_flow_weight,
        quick_stage_weight=normalized_quick_weight,
        number_of_observations=len(prepared),
        number_of_detected_events=len(detected_events),
        number_of_filtered_events=len(filtered_events),
        number_of_selected_events=len(selected_events),
        baseflow=baseflow,
        detected_events=detected_events,
        filtered_events=filtered_events,
        selected_events=selected_events,
        baseflow_path=baseflow_path,
        detected_events_path=detected_events_path,
        filtered_events_path=filtered_events_path,
        selected_events_path=selected_events_path,
        manifest_path=manifest_path,
    )


def detect_stage_events(
    input_path: str | Path,
    output_directory: str | Path | None = None,
    *,
    site_id: str | None = None,
    baseflow_alpha: float = DEFAULT_BASEFLOW_ALPHA,
    event_filter_percentile: float = DEFAULT_EVENT_FILTER_PERCENTILE,
    top_n_events: int | None = DEFAULT_TOP_N_EVENTS,
    ranking_method: str = DEFAULT_RANKING_METHOD,
    flow_peak_weight: float = DEFAULT_FLOW_PEAK_WEIGHT,
    quick_stage_weight: float = DEFAULT_QUICK_STAGE_WEIGHT,
) -> StageEventDetectionResult:
    """Run event detection from ``stage_hydroeventdetector.parquet``.

    When ``output_directory`` is omitted and the input file is inside a
    ``downloaded_data`` directory, outputs are written to the sibling
    ``processed_data`` directory. Otherwise, outputs are written to a
    ``processed_data`` directory beside the input file.
    """

    path = Path(input_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Stage input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Stage input path is not a file: {path}")

    if output_directory is None:
        if path.parent.name == "downloaded_data":
            output_dir = path.parent.parent / "processed_data"
        else:
            output_dir = path.parent / "processed_data"
    else:
        output_dir = Path(output_directory).expanduser()

    stage = pd.read_parquet(path)
    return detect_stage_events_from_dataframe(
        stage,
        output_dir,
        site_id=site_id,
        baseflow_alpha=baseflow_alpha,
        event_filter_percentile=event_filter_percentile,
        top_n_events=top_n_events,
        ranking_method=ranking_method,
        flow_peak_weight=flow_peak_weight,
        quick_stage_weight=quick_stage_weight,
        input_path=path,
    )


def process_stage_events_from_files(
    input_path: str | Path,
    output_directory: str | Path | None = None,
    **kwargs: Any,
) -> StageEventDetectionResult:
    """Backward-friendly alias for file-based stage-event processing."""

    return detect_stage_events(
        input_path=input_path,
        output_directory=output_directory,
        **kwargs,
    )


__all__ = [
    "DEFAULT_BASEFLOW_ALPHA",
    "DEFAULT_EVENT_FILTER_PERCENTILE",
    "DEFAULT_TOP_N_EVENTS",
    "DEFAULT_RANKING_METHOD",
    "DEFAULT_FLOW_PEAK_WEIGHT",
    "DEFAULT_QUICK_STAGE_WEIGHT",
    "VALID_RANKING_METHODS",
    "StageEventDetectionResult",
    "add_event_ranking_metrics",
    "select_ranked_events",
    "detect_stage_events",
    "detect_stage_events_from_dataframe",
    "process_stage_events_from_files",
]