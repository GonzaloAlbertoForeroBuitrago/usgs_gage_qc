"""Unified USGS water-stage download from modern and legacy services."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PARAMETER_CODE_STAGE = "00065"

MODERN_BASE_URL = "https://api.waterdata.usgs.gov/ogcapi/v0"
LEGACY_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"


OBSERVATION_COLUMNS = [
    "site_id",
    "datetime",
    "value",
    "parameter_code",
    "parameter_name",
    "unit",
    "qualifier",
    "approval_status",
    "time_series_id",
    "source",
]

STATION_COLUMNS = [
    "site_id",
    "agency_code",
    "station_name",
    "state_code",
    "state_name",
    "county_code",
    "site_type_code",
    "hydrologic_unit_code",
    "latitude",
    "longitude",
    "altitude",
    "vertical_datum",
    "time_zone",
    "uses_daylight_savings",
    "source",
]

TIME_SERIES_COLUMNS = [
    "site_id",
    "time_series_id",
    "parameter_code",
    "parameter_name",
    "parameter_description",
    "unit",
    "statistic_id",
    "computation_identifier",
    "computation_period_identifier",
    "start_datetime",
    "end_datetime",
    "data_gap_interval",
    "web_description",
    "source",
]


@dataclass
class StageDownloadResult:
    """Files and normalized data produced by one download."""

    state: str
    site_id: str
    source: str
    output_directory: Path
    observations: pd.DataFrame
    hydroeventdetector_input: pd.DataFrame
    excluded_observations: pd.DataFrame
    station_metadata: pd.DataFrame
    time_series_metadata: pd.DataFrame
    observations_path: Path
    hydroeventdetector_path: Path
    excluded_observations_path: Path
    station_metadata_path: Path
    time_series_metadata_path: Path
    manifest_path: Path


def preserve_site_id(site_id: str) -> str:
    """
    Preserve the station identifier exactly as supplied.

    The value is converted to string and stripped. It is never converted
    to integer and never padded with zfill.
    """

    value = str(site_id).strip()

    if not value:
        raise ValueError("site_id cannot be empty")

    return value


def preserve_state(state: str) -> str:
    """Return a clean state directory name."""

    value = str(state).strip()

    if not value:
        raise ValueError("state cannot be empty")

    return value.upper()


def request_json(
    url: str,
    *,
    params: dict[str, Any],
    timeout: int = 120,
) -> dict[str, Any]:
    """Perform an HTTP GET request and return a JSON object."""

    response = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={
            "Accept": "application/json",
            "User-Agent": "usgs-gage-qc/0.1",
        },
    )
    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object from {response.url}")

    return payload


def empty_observations() -> pd.DataFrame:
    return pd.DataFrame(columns=OBSERVATION_COLUMNS)


def empty_station_metadata() -> pd.DataFrame:
    return pd.DataFrame(columns=STATION_COLUMNS)


def empty_time_series_metadata() -> pd.DataFrame:
    return pd.DataFrame(columns=TIME_SERIES_COLUMNS)


def normalize_qualifier(value: Any) -> str | None:
    """Convert a qualifier or qualifier list to a stable string."""

    if value is None:
        return None

    if isinstance(value, list):
        cleaned = [
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        ]
        return "|".join(cleaned) if cleaned else None

    text = str(value).strip()
    return text or None


def normalize_approval_status(value: Any) -> str | None:
    """Normalize equivalent modern and legacy approval values."""

    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    mapping = {
        "A": "Approved",
        "APPROVED": "Approved",
        "P": "Provisional",
        "PROVISIONAL": "Provisional",
    }

    return mapping.get(text.upper(), text)


def normalize_parameter_name(
    parameter_code: Any,
    parameter_name: Any,
) -> str | None:
    """Return one common parameter name for equivalent USGS series."""

    code = None if parameter_code is None else str(parameter_code).strip()

    if code == PARAMETER_CODE_STAGE:
        return "Gage height"

    if parameter_name is None or pd.isna(parameter_name):
        return None

    text = str(parameter_name).strip()
    return text or None


def normalize_daylight_savings(value: Any) -> bool | None:
    """Normalize Y/N and boolean daylight-saving indicators."""

    if value is None or pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    text = str(value).strip().upper()

    if text in {"Y", "YES", "TRUE", "1"}:
        return True

    if text in {"N", "NO", "FALSE", "0"}:
        return False

    return None


def normalize_county_code(
    state_code: Any,
    county_code: Any,
) -> str | None:
    """Normalize county codes to the county portion used by both sources."""

    if county_code is None or pd.isna(county_code):
        return None

    county = str(county_code).strip()
    state = None if state_code is None else str(state_code).strip()

    if state and len(county) == 5 and county.startswith(state):
        return county[len(state):]

    return county or None


def finalize_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the shared observation schema and data types."""

    if df.empty:
        return empty_observations()

    result = df.copy()

    for column in OBSERVATION_COLUMNS:
        if column not in result.columns:
            result[column] = None

    result["site_id"] = result["site_id"].astype("string")
    result["datetime"] = pd.to_datetime(
        result["datetime"],
        utc=True,
        errors="coerce",
    )
    result["value"] = pd.to_numeric(
        result["value"],
        errors="coerce",
    )

    result["parameter_code"] = (
        result["parameter_code"]
        .astype("string")
        .str.strip()
    )

    result["parameter_name"] = [
        normalize_parameter_name(code, name)
        for code, name in zip(
            result["parameter_code"],
            result["parameter_name"],
        )
    ]

    result["approval_status"] = (
        result["approval_status"]
        .map(normalize_approval_status)
        .astype("string")
    )

    result = result.dropna(subset=["datetime", "value"])

    result = (
        result[OBSERVATION_COLUMNS]
        .sort_values(
            ["datetime", "time_series_id"],
            na_position="last",
        )
        .drop_duplicates(
            subset=[
                "site_id",
                "datetime",
                "time_series_id",
                "parameter_code",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return result


def prepare_hydroeventdetector_input(
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the minimal, auditable input used by HydroEventDetector.

    The returned input contains one observation per timestamp with the
    historical column names expected by ``mrms_usgs_events``:
    ``datetime`` and ``Stage_ft``. Datetimes are timezone-naive
    ``datetime64[ns]`` values and stage values are ``float64``.

    Any invalid or duplicate rows excluded from that interoperable input are
    returned separately with an ``exclusion_reason`` column. The full
    normalized observations remain unchanged in ``stage_observations.parquet``.
    """

    required = {"datetime", "value"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(
            "Cannot prepare HydroEventDetector input; missing columns: "
            f"{sorted(missing)}"
        )

    working = observations.copy()
    working["_datetime"] = pd.to_datetime(
        working["datetime"], errors="coerce", utc=True
    )
    working["_stage"] = pd.to_numeric(working["value"], errors="coerce")

    invalid_datetime = working["_datetime"].isna()
    invalid_stage = working["_stage"].isna()
    invalid = invalid_datetime | invalid_stage

    invalid_rows = working.loc[invalid].copy()
    invalid_rows["exclusion_reason"] = "invalid_datetime_and_stage"
    invalid_rows.loc[invalid_datetime & ~invalid_stage, "exclusion_reason"] = (
        "invalid_datetime"
    )
    invalid_rows.loc[~invalid_datetime & invalid_stage, "exclusion_reason"] = (
        "invalid_stage"
    )

    valid = working.loc[~invalid].sort_values(
        ["_datetime", "time_series_id"],
        kind="stable",
        na_position="last",
    )
    duplicate = valid.duplicated(subset="_datetime", keep="last")

    duplicate_rows = valid.loc[duplicate].copy()
    duplicate_rows["exclusion_reason"] = "duplicate_datetime"

    selected = valid.loc[~duplicate].reset_index(drop=True)
    hydroeventdetector_input = pd.DataFrame(
        {
            "datetime": selected["_datetime"]
            .dt.tz_localize(None)
            .astype("datetime64[ns]"),
            "Stage_ft": selected["_stage"].astype("float64"),
        }
    )

    audit_columns = [*OBSERVATION_COLUMNS, "exclusion_reason"]
    excluded_observations = pd.concat(
        [invalid_rows, duplicate_rows], ignore_index=True
    )
    excluded_observations = excluded_observations.reindex(columns=audit_columns)

    return hydroeventdetector_input, excluded_observations


def finalize_station_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the shared station metadata schema."""

    if df.empty:
        return empty_station_metadata()

    result = df.copy()

    for column in STATION_COLUMNS:
        if column not in result.columns:
            result[column] = None

    result["site_id"] = result["site_id"].map(preserve_site_id)

    result["state_code"] = (
        result["state_code"]
        .astype("string")
        .str.strip()
    )

    result["county_code"] = [
        normalize_county_code(state_code, county_code)
        for state_code, county_code in zip(
            result["state_code"],
            result["county_code"],
        )
    ]

    result["uses_daylight_savings"] = result[
        "uses_daylight_savings"
    ].map(normalize_daylight_savings)

    return result[STATION_COLUMNS].reset_index(drop=True)


def finalize_time_series_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the shared time-series metadata schema."""

    if df.empty:
        return empty_time_series_metadata()

    result = df.copy()

    for column in TIME_SERIES_COLUMNS:
        if column not in result.columns:
            result[column] = None

    result["start_datetime"] = pd.to_datetime(
        result["start_datetime"],
        utc=True,
        errors="coerce",
    )
    result["end_datetime"] = pd.to_datetime(
        result["end_datetime"],
        utc=True,
        errors="coerce",
    )

    result["site_id"] = result["site_id"].map(preserve_site_id)

    result["parameter_code"] = (
        result["parameter_code"]
        .astype("string")
        .str.strip()
    )

    result["parameter_name"] = [
        normalize_parameter_name(code, name)
        for code, name in zip(
            result["parameter_code"],
            result["parameter_name"],
        )
    ]

    return result[TIME_SERIES_COLUMNS].reset_index(drop=True)


def modern_monitoring_location(
    site_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download and normalize modern monitoring-location metadata."""

    site_id = preserve_site_id(site_id)

    url = (
        f"{MODERN_BASE_URL}/collections/"
        "monitoring-locations/items"
    )

    payload = request_json(
        url,
        params={
            "f": "json",
            "id": f"USGS-{site_id}",
            "limit": 10,
        },
    )

    features = payload.get("features") or []

    if not features:
        return empty_station_metadata(), payload

    feature = features[0]
    properties = feature.get("properties") or {}
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates") or [None, None]

    longitude = coordinates[0] if len(coordinates) >= 1 else None
    latitude = coordinates[1] if len(coordinates) >= 2 else None

    row = {
        "site_id": site_id,
        "agency_code": properties.get("agency_code"),
        "station_name": (
            properties.get("monitoring_location_name")
            or properties.get("name")
        ),
        "state_code": properties.get("state_code"),
        "state_name": properties.get("state_name"),
        "county_code": properties.get("county_code"),
        "site_type_code": properties.get("site_type_code"),
        "hydrologic_unit_code": properties.get(
            "hydrologic_unit_code"
        ),
        "latitude": latitude,
        "longitude": longitude,
        "altitude": properties.get("altitude"),
        "vertical_datum": properties.get("vertical_datum"),
        "time_zone": (
            properties.get("time_zone_abbreviation")
            or properties.get("time_zone")
        ),
        "uses_daylight_savings": properties.get(
            "uses_daylight_savings"
        ),
        "source": "modern_ogc",
    }

    return finalize_station_metadata(pd.DataFrame([row])), payload


def modern_time_series(
    site_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download all modern stage time-series metadata for a station."""

    site_id = preserve_site_id(site_id)

    url = (
        f"{MODERN_BASE_URL}/collections/"
        "time-series-metadata/items"
    )

    payload = request_json(
        url,
        params={
            "f": "json",
            "monitoring_location_id": f"USGS-{site_id}",
            "parameter_code": PARAMETER_CODE_STAGE,
            "limit": 10000,
        },
    )

    rows: list[dict[str, Any]] = []

    for feature in payload.get("features") or []:
        properties = feature.get("properties") or {}

        rows.append(
            {
                "site_id": site_id,
                "time_series_id": (
                    properties.get("id")
                    or feature.get("id")
                ),
                "parameter_code": properties.get(
                    "parameter_code"
                ),
                "parameter_name": properties.get(
                    "parameter_name"
                ),
                "parameter_description": properties.get(
                    "parameter_description"
                ),
                "unit": properties.get("unit_of_measure"),
                "statistic_id": properties.get("statistic_id"),
                "computation_identifier": properties.get(
                    "computation_identifier"
                ),
                "computation_period_identifier": properties.get(
                    "computation_period_identifier"
                ),
                "start_datetime": (
                    properties.get("start")
                    or properties.get("begin")
                ),
                "end_datetime": (
                    properties.get("end")
                    or properties.get("latest")
                ),
                "data_gap_interval": properties.get(
                    "data_gap_interval"
                ),
                "web_description": properties.get(
                    "web_description"
                ),
                "source": "modern_ogc",
            }
        )

    return (
        finalize_time_series_metadata(pd.DataFrame(rows)),
        payload,
    )


def modern_observations(
    site_id: str,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Download stage observations from the modern OGC API."""

    site_id = preserve_site_id(site_id)

    start = pd.Timestamp(start_date)

    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")

    end = pd.Timestamp(end_date)

    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    if end <= start:
        raise ValueError("end_date must be later than start_date")

    datetime_filter = (
        f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}/"
        f"{end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )

    url = f"{MODERN_BASE_URL}/collections/continuous/items"

    params: dict[str, Any] = {
        "f": "json",
        "monitoring_location_id": f"USGS-{site_id}",
        "parameter_code": PARAMETER_CODE_STAGE,
        "datetime": datetime_filter,
        "limit": 20000,
    }

    payloads: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    while url:
        payload = request_json(url, params=params)
        payloads.append(payload)

        for feature in payload.get("features") or []:
            properties = feature.get("properties") or {}

            rows.append(
                {
                    "site_id": site_id,
                    "datetime": properties.get("time"),
                    "value": properties.get("value"),
                    "parameter_code": (
                        properties.get("parameter_code")
                        or PARAMETER_CODE_STAGE
                    ),
                    "parameter_name": properties.get(
                        "parameter_name"
                    ),
                    "unit": properties.get("unit_of_measure"),
                    "qualifier": normalize_qualifier(
                        properties.get("qualifier")
                    ),
                    "approval_status": normalize_qualifier(
                        properties.get("approval_status")
                    ),
                    "time_series_id": properties.get(
                        "time_series_id"
                    ),
                    "source": "modern_ogc",
                }
            )

        next_url = None

        for link in payload.get("links") or []:
            if link.get("rel") == "next":
                next_url = link.get("href")
                break

        url = next_url
        params = {}

    return finalize_observations(pd.DataFrame(rows)), payloads


def legacy_iv(
    site_id: str,
    start_date: str,
    end_date: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """Download and normalize stage data from legacy NWIS IV."""

    site_id = preserve_site_id(site_id)

    payload = request_json(
        LEGACY_IV_URL,
        params={
            "format": "json",
            "sites": site_id,
            "parameterCd": PARAMETER_CODE_STAGE,
            "startDT": str(start_date),
            "endDT": str(end_date),
            "siteStatus": "all",
        },
    )

    time_series_entries = (
        payload.get("value", {}).get("timeSeries", [])
    )

    observation_rows: list[dict[str, Any]] = []
    series_rows: list[dict[str, Any]] = []
    station_row: dict[str, Any] | None = None

    for series_index, series in enumerate(time_series_entries):
        source_info = series.get("sourceInfo") or {}
        variable = series.get("variable") or {}

        site_codes = source_info.get("siteCode") or []
        returned_site_id = (
            site_codes[0].get("value")
            if site_codes
            else site_id
        )
        returned_site_id = preserve_site_id(returned_site_id)

        variable_codes = variable.get("variableCode") or []
        parameter_code = (
            variable_codes[0].get("value")
            if variable_codes
            else PARAMETER_CODE_STAGE
        )

        unit_info = variable.get("unit") or {}
        unit = (
            unit_info.get("unitCode")
            or unit_info.get("unitAbbreviation")
        )

        series_name = str(series.get("name") or "")
        time_series_id = (
            series_name
            if series_name
            else f"legacy_{returned_site_id}_{series_index}"
        )

        if station_row is None:
            geo = (
                source_info
                .get("geoLocation", {})
                .get("geogLocation", {})
            )
            site_properties = {
                item.get("name"): item.get("value")
                for item in source_info.get("siteProperty") or []
            }
            time_zone_info = (
                source_info.get("timeZoneInfo") or {}
            )
            default_time_zone = (
                time_zone_info.get("defaultTimeZone") or {}
            )

            station_row = {
                "site_id": returned_site_id,
                "agency_code": (
                    site_codes[0].get("agencyCode")
                    if site_codes
                    else "USGS"
                ),
                "station_name": source_info.get("siteName"),
                "state_code": site_properties.get("stateCd"),
                "state_name": None,
                "county_code": site_properties.get("countyCd"),
                "site_type_code": site_properties.get("siteTypeCd"),
                "hydrologic_unit_code": site_properties.get(
                    "hucCd"
                ),
                "latitude": geo.get("latitude"),
                "longitude": geo.get("longitude"),
                "altitude": None,
                "vertical_datum": None,
                "time_zone": (
                    default_time_zone.get("zoneAbbreviation")
                    or default_time_zone.get("zoneOffset")
                ),
                "uses_daylight_savings": time_zone_info.get(
                    "siteUsesDaylightSavingsTime"
                ),
                "source": "legacy_nwis_iv",
            }

        series_rows.append(
            {
                "site_id": returned_site_id,
                "time_series_id": time_series_id,
                "parameter_code": parameter_code,
                "parameter_name": variable.get("variableName"),
                "parameter_description": variable.get(
                    "variableDescription"
                ),
                "unit": unit,
                "statistic_id": None,
                "computation_identifier": "Instantaneous",
                "computation_period_identifier": "Points",
                "start_datetime": None,
                "end_datetime": None,
                "data_gap_interval": None,
                "web_description": None,
                "source": "legacy_nwis_iv",
            }
        )

        for values_block in series.get("values") or []:
            for observation in values_block.get("value") or []:
                qualifiers = observation.get("qualifiers") or []

                approval_status = None
                other_qualifiers: list[str] = []

                for qualifier in qualifiers:
                    qualifier_text = str(qualifier).strip()

                    if qualifier_text in {"A", "P"}:
                        approval_status = qualifier_text
                    elif qualifier_text:
                        other_qualifiers.append(qualifier_text)

                observation_rows.append(
                    {
                        "site_id": returned_site_id,
                        "datetime": observation.get("dateTime"),
                        "value": observation.get("value"),
                        "parameter_code": parameter_code,
                        "parameter_name": variable.get(
                            "variableName"
                        ),
                        "unit": unit,
                        "qualifier": normalize_qualifier(
                            other_qualifiers
                        ),
                        "approval_status": approval_status,
                        "time_series_id": time_series_id,
                        "source": "legacy_nwis_iv",
                    }
                )

    station_df = (
        finalize_station_metadata(pd.DataFrame([station_row]))
        if station_row
        else empty_station_metadata()
    )

    return (
        finalize_observations(pd.DataFrame(observation_rows)),
        station_df,
        finalize_time_series_metadata(pd.DataFrame(series_rows)),
        payload,
    )


def write_json(path: Path, payload: Any) -> None:
    """Write readable JSON for audit and debugging."""

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def download_stage_data(
    state: str,
    site_id: str,
    start_date: str,
    end_date: str,
    *,
    output_root: str | Path = "data",
) -> StageDownloadResult:
    """
    Download and store stage data using a modern-first fallback.

    Directory:
        output_root / STATE / site_id / downloaded_data
    """

    state = preserve_state(state)
    site_id = preserve_site_id(site_id)

    station_directory = (
        Path(output_root)
        / state
        / site_id
    )

    downloaded_directory = (
        station_directory
        / "downloaded_data"
    )

    processed_directory = (
        station_directory
        / "processed_data"
    )

    downloaded_directory.mkdir(parents=True, exist_ok=True)
    processed_directory.mkdir(parents=True, exist_ok=True)

    modern_station, modern_station_raw = (
        modern_monitoring_location(site_id)
    )
    modern_series, modern_series_raw = modern_time_series(site_id)

    modern_data: pd.DataFrame
    modern_observations_raw: list[dict[str, Any]]

    try:
        modern_data, modern_observations_raw = modern_observations(
            site_id=site_id,
            start_date=start_date,
            end_date=end_date,
        )
    except requests.RequestException as exc:
        modern_data = empty_observations()
        modern_observations_raw = [
            {
                "error": str(exc),
                "source": "modern_ogc",
            }
        ]

    if not modern_data.empty:
        selected_source = "modern_ogc"
        observations = modern_data
        station_metadata = modern_station
        time_series_metadata = modern_series

        write_json(
            downloaded_directory
            / "modern_monitoring_location_raw.json",
            modern_station_raw,
        )
        write_json(
            downloaded_directory
            / "modern_time_series_metadata_raw.json",
            modern_series_raw,
        )
        write_json(
            downloaded_directory
            / "modern_observations_raw.json",
            modern_observations_raw,
        )

        fallback_attempted = False
    else:
        fallback_attempted = True

        (
            observations,
            legacy_station,
            legacy_series,
            legacy_raw,
        ) = legacy_iv(
            site_id=site_id,
            start_date=start_date,
            end_date=end_date,
        )

        selected_source = "legacy_nwis_iv"

        station_metadata = (
            modern_station
            if not modern_station.empty
            else legacy_station
        )

        time_series_metadata = (
            modern_series
            if not modern_series.empty
            else legacy_series
        )

        write_json(
            downloaded_directory
            / "modern_monitoring_location_raw.json",
            modern_station_raw,
        )
        write_json(
            downloaded_directory
            / "modern_time_series_metadata_raw.json",
            modern_series_raw,
        )
        write_json(
            downloaded_directory
            / "modern_observations_raw.json",
            modern_observations_raw,
        )
        write_json(
            downloaded_directory
            / "legacy_iv_raw.json",
            legacy_raw,
        )

    hydroeventdetector_input, excluded_observations = (
        prepare_hydroeventdetector_input(observations)
    )

    observations_path = (
        downloaded_directory / "stage_observations.parquet"
    )
    hydroeventdetector_path = (
        downloaded_directory / "stage_hydroeventdetector.parquet"
    )
    excluded_observations_path = (
        downloaded_directory / "excluded_observations.parquet"
    )
    station_metadata_path = (
        downloaded_directory / "station_metadata.parquet"
    )
    time_series_metadata_path = (
        downloaded_directory / "time_series_metadata.parquet"
    )
    manifest_path = (
        downloaded_directory / "download_manifest.json"
    )

    observations.to_parquet(observations_path, index=False)
    hydroeventdetector_input.to_parquet(
        hydroeventdetector_path,
        index=False,
    )
    excluded_observations.to_parquet(
        excluded_observations_path,
        index=False,
    )
    station_metadata.to_parquet(
        station_metadata_path,
        index=False,
    )
    time_series_metadata.to_parquet(
        time_series_metadata_path,
        index=False,
    )

    manifest = {
        "state": state,
        "site_id": site_id,
        "parameter_code": PARAMETER_CODE_STAGE,
        "requested_start": str(start_date),
        "requested_end": str(end_date),
        "selected_source": selected_source,
        "fallback_attempted": fallback_attempted,
        "observation_rows": int(len(observations)),
        "hydroeventdetector_rows": int(len(hydroeventdetector_input)),
        "excluded_observation_rows": int(len(excluded_observations)),
        "station_metadata_rows": int(len(station_metadata)),
        "time_series_metadata_rows": int(
            len(time_series_metadata)
        ),
        "observation_columns": OBSERVATION_COLUMNS,
        "hydroeventdetector_columns": ["datetime", "Stage_ft"],
        "excluded_observation_columns": [
            *OBSERVATION_COLUMNS,
            "exclusion_reason",
        ],
        "files": {
            "observations": observations_path.name,
            "hydroeventdetector_input": hydroeventdetector_path.name,
            "excluded_observations": excluded_observations_path.name,
            "station_metadata": station_metadata_path.name,
            "time_series_metadata": time_series_metadata_path.name,
        },
        "station_metadata_columns": STATION_COLUMNS,
        "time_series_metadata_columns": TIME_SERIES_COLUMNS,
    }

    write_json(manifest_path, manifest)

    if observations.empty:
        raise RuntimeError(
            f"No stage observations were found for site {site_id} "
            f"between {start_date} and {end_date} using either "
            "USGS source."
        )

    return StageDownloadResult(
        state=state,
        site_id=site_id,
        source=selected_source,
        output_directory=downloaded_directory,
        observations=observations,
        hydroeventdetector_input=hydroeventdetector_input,
        excluded_observations=excluded_observations,
        station_metadata=station_metadata,
        time_series_metadata=time_series_metadata,
        observations_path=observations_path,
        hydroeventdetector_path=hydroeventdetector_path,
        excluded_observations_path=excluded_observations_path,
        station_metadata_path=station_metadata_path,
        time_series_metadata_path=time_series_metadata_path,
        manifest_path=manifest_path,
    )
