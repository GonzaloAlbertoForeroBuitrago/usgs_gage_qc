"""Continuity and gap processing for USGS water-stage observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StageGapProcessingResult:
    site_id: str
    state: str
    expected_interval_minutes: float
    gap_threshold_minutes: float
    observation_count: int
    valid_observation_count: int
    gap_count: int
    segment_count: int
    stage_processed_path: Path
    continuity_gaps_path: Path
    continuous_segments_path: Path


REQUIRED_OBSERVATION_COLUMNS = [
    "site_id", "datetime", "value", "parameter_code", "parameter_name",
    "unit", "qualifier", "approval_status", "time_series_id", "source",
]


def preserve_site_id(site_id: object) -> str:
    value = str(site_id).strip()
    if not value:
        raise ValueError("site_id cannot be empty.")
    return value


def preserve_state(state: object) -> str:
    value = str(state).strip().upper()
    if not value:
        raise ValueError("state cannot be empty.")
    return value


def _validate_observation_schema(observations: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_OBSERVATION_COLUMNS if c not in observations.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def infer_expected_interval_minutes(datetimes: pd.Series) -> float:
    parsed = pd.to_datetime(datetimes, utc=True, errors="coerce")
    diffs = (
        parsed.dropna().sort_values().diff().dt.total_seconds().div(60.0).dropna()
    )
    diffs = diffs[diffs > 0].round(6)
    if diffs.empty:
        raise ValueError("Cannot infer interval from fewer than two valid timestamps.")

    modes = diffs.mode(dropna=True)
    expected = float(modes.iloc[0] if not modes.empty else diffs.median())
    if not np.isfinite(expected) or expected <= 0:
        raise ValueError(f"Invalid inferred interval: {expected}")
    return expected


def process_stage_gaps(
    observations: pd.DataFrame,
    *,
    state: str,
    site_id: str,
    expected_interval_minutes: Optional[float] = None,
    gap_factor: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float]:
    state = preserve_state(state)
    site_id = preserve_site_id(site_id)
    if gap_factor <= 1:
        raise ValueError("gap_factor must be greater than 1.")

    _validate_observation_schema(observations)
    data = observations.copy()
    data["original_row_id"] = np.arange(len(data), dtype=np.int64)
    data["site_id"] = data["site_id"].map(preserve_site_id)

    unexpected = sorted(set(data["site_id"]) - {site_id})
    if unexpected:
        raise ValueError(f"Input contains unexpected site IDs: {unexpected}")

    data["datetime"] = pd.to_datetime(data["datetime"], utc=True, errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.sort_values(
        ["datetime", "original_row_id"], kind="mergesort", na_position="last"
    ).reset_index(drop=True)

    if expected_interval_minutes is None:
        expected_interval_minutes = infer_expected_interval_minutes(data["datetime"])
    else:
        expected_interval_minutes = float(expected_interval_minutes)
        if not np.isfinite(expected_interval_minutes) or expected_interval_minutes <= 0:
            raise ValueError("expected_interval_minutes must be positive and finite.")

    gap_threshold_minutes = expected_interval_minutes * float(gap_factor)

    data["previous_datetime"] = data["datetime"].shift(1)
    data["previous_value"] = data["value"].shift(1)
    data["observed_interval_minutes"] = (
        data["datetime"] - data["previous_datetime"]
    ).dt.total_seconds().div(60.0)

    data["expected_interval_minutes"] = expected_interval_minutes
    data["gap_threshold_minutes"] = gap_threshold_minutes
    data["is_invalid_datetime"] = data["datetime"].isna()
    data["is_missing_value"] = data["value"].isna()
    data["is_duplicate_timestamp"] = (
        data["datetime"].notna() & data["datetime"].duplicated(keep=False)
    )
    data["is_non_increasing_timestamp"] = (
        data["observed_interval_minutes"].notna()
        & (data["observed_interval_minutes"] <= 0)
    )
    data["is_temporal_gap"] = (
        data["observed_interval_minutes"].notna()
        & (data["observed_interval_minutes"] > gap_threshold_minutes)
    )

    previous_missing = data["value"].shift(1).isna()
    previous_invalid_datetime = data["datetime"].shift(1).isna()

    data["starts_new_segment"] = (
        data["datetime"].notna()
        & data["value"].notna()
        & (
            data["previous_datetime"].isna()
            | data["is_temporal_gap"]
            | previous_missing
            | previous_invalid_datetime
            | data["is_non_increasing_timestamp"]
        )
    )

    segment_counter = data["starts_new_segment"].cumsum().astype("Int64")
    processable = (
        data["datetime"].notna()
        & data["value"].notna()
        & ~data["is_non_increasing_timestamp"]
    )
    data["segment_id"] = segment_counter.where(processable, pd.NA)

    data["baseflow"] = np.nan
    data["quickflow"] = np.nan
    data["baseflow_status"] = "not_calculated"

    gap_rows = data.loc[data["is_temporal_gap"]].copy()
    if gap_rows.empty:
        continuity_gaps = pd.DataFrame(columns=[
            "site_id", "previous_datetime", "current_datetime", "previous_value",
            "current_value", "observed_interval_minutes",
            "expected_interval_minutes", "gap_threshold_minutes",
            "estimated_missing_intervals", "previous_segment_id",
            "current_segment_id", "gap_reason",
        ])
    else:
        continuity_gaps = pd.DataFrame({
            "site_id": gap_rows["site_id"],
            "previous_datetime": gap_rows["previous_datetime"],
            "current_datetime": gap_rows["datetime"],
            "previous_value": gap_rows["previous_value"],
            "current_value": gap_rows["value"],
            "observed_interval_minutes": gap_rows["observed_interval_minutes"],
            "expected_interval_minutes": gap_rows["expected_interval_minutes"],
            "gap_threshold_minutes": gap_rows["gap_threshold_minutes"],
            "estimated_missing_intervals": np.maximum(
                np.rint(
                    gap_rows["observed_interval_minutes"]
                    / gap_rows["expected_interval_minutes"]
                ).astype("Int64") - 1,
                1,
            ),
            "previous_segment_id": gap_rows["segment_id"] - 1,
            "current_segment_id": gap_rows["segment_id"],
            "gap_reason": "timestamp_gap",
        }).reset_index(drop=True)

    valid_segments = data.loc[
        data["segment_id"].notna()
        & data["datetime"].notna()
        & data["value"].notna()
    ].copy()

    if valid_segments.empty:
        continuous_segments = pd.DataFrame(columns=[
            "site_id", "segment_id", "start_datetime", "end_datetime",
            "start_value", "end_value", "observation_count",
            "valid_value_count", "duration_minutes",
            "expected_interval_minutes", "segment_status",
        ])
    else:
        continuous_segments = (
            valid_segments.groupby(["site_id", "segment_id"], sort=True)
            .agg(
                start_datetime=("datetime", "first"),
                end_datetime=("datetime", "last"),
                start_value=("value", "first"),
                end_value=("value", "last"),
                observation_count=("value", "size"),
                valid_value_count=("value", "count"),
                expected_interval_minutes=("expected_interval_minutes", "first"),
            )
            .reset_index()
        )
        continuous_segments["duration_minutes"] = (
            continuous_segments["end_datetime"]
            - continuous_segments["start_datetime"]
        ).dt.total_seconds().div(60.0)
        continuous_segments["segment_status"] = np.where(
            continuous_segments["valid_value_count"] >= 2,
            "continuous",
            "single_observation",
        )
        continuous_segments = continuous_segments[[
            "site_id", "segment_id", "start_datetime", "end_datetime",
            "start_value", "end_value", "observation_count",
            "valid_value_count", "duration_minutes",
            "expected_interval_minutes", "segment_status",
        ]]

    data = data.sort_values("original_row_id", kind="mergesort").reset_index(drop=True)

    processed_columns = REQUIRED_OBSERVATION_COLUMNS + [
        "observed_interval_minutes", "expected_interval_minutes",
        "gap_threshold_minutes", "is_invalid_datetime", "is_missing_value",
        "is_duplicate_timestamp", "is_non_increasing_timestamp",
        "is_temporal_gap", "starts_new_segment", "segment_id",
        "baseflow", "quickflow", "baseflow_status",
    ]
    stage_processed = data[processed_columns].copy()

    return (
        stage_processed,
        continuity_gaps,
        continuous_segments,
        expected_interval_minutes,
        gap_threshold_minutes,
    )


def process_stage_gaps_from_files(
    *,
    state: str,
    site_id: str,
    data_root: str | Path = "data",
    expected_interval_minutes: Optional[float] = None,
    gap_factor: float = 1.5,
) -> StageGapProcessingResult:
    state = preserve_state(state)
    site_id = preserve_site_id(site_id)
    data_root = Path(data_root)

    site_directory = data_root / state / site_id
    observations_path = (
        site_directory / "downloaded_data" / "stage_observations.parquet"
    )
    processed_directory = site_directory / "processed_data"

    if not observations_path.exists():
        raise FileNotFoundError(f"Stage observations not found: {observations_path}")

    observations = pd.read_parquet(observations_path)
    (
        stage_processed,
        continuity_gaps,
        continuous_segments,
        inferred_interval,
        gap_threshold,
    ) = process_stage_gaps(
        observations,
        state=state,
        site_id=site_id,
        expected_interval_minutes=expected_interval_minutes,
        gap_factor=gap_factor,
    )

    processed_directory.mkdir(parents=True, exist_ok=True)
    stage_processed_path = processed_directory / "stage_processed.parquet"
    continuity_gaps_path = processed_directory / "continuity_gaps.parquet"
    continuous_segments_path = processed_directory / "continuous_segments.parquet"

    stage_processed.to_parquet(stage_processed_path, index=False)
    continuity_gaps.to_parquet(continuity_gaps_path, index=False)
    continuous_segments.to_parquet(continuous_segments_path, index=False)

    return StageGapProcessingResult(
        site_id=site_id,
        state=state,
        expected_interval_minutes=float(inferred_interval),
        gap_threshold_minutes=float(gap_threshold),
        observation_count=len(stage_processed),
        valid_observation_count=int(
            (stage_processed["datetime"].notna() & stage_processed["value"].notna()).sum()
        ),
        gap_count=len(continuity_gaps),
        segment_count=len(continuous_segments),
        stage_processed_path=stage_processed_path,
        continuity_gaps_path=continuity_gaps_path,
        continuous_segments_path=continuous_segments_path,
    )
