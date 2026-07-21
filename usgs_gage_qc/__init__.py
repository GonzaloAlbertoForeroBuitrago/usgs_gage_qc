"""
USGS Water Stage Quality Control.

Tools for downloading, processing, analyzing, and visualizing
USGS water-stage time series.
"""

__version__ = "0.1.0"

# ---------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------

from .stage_download import (
    StageDownloadResult,
    download_stage_data,
)

# ---------------------------------------------------------------------
# Hydro Event Detector
# ---------------------------------------------------------------------

from .hydro_event_detector import (
    DEFAULT_BASEFLOW_ALPHA,
    DEFAULT_EVENT_FILTER_PERCENTILE,
    DEFAULT_TOP_N_EVENTS,
    DEFAULT_RANKING_METHOD,
    DEFAULT_FLOW_PEAK_WEIGHT,
    DEFAULT_QUICK_STAGE_WEIGHT,
    VALID_RANKING_METHODS,
    StageEventDetectionResult,
    add_event_ranking_metrics,
    select_ranked_events,
    detect_stage_events,
    detect_stage_events_from_dataframe,
    process_stage_events_from_files,
)

# ---------------------------------------------------------------------
# Gap processing
# ---------------------------------------------------------------------

from .stage_gap_processing import (
    StageGapProcessingResult,
    infer_expected_interval_minutes,
    process_stage_gaps,
    process_stage_gaps_from_files,
)

__all__ = [

    # -----------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------

    "StageDownloadResult",
    "download_stage_data",

    # -----------------------------------------------------------------
    # Hydro Event Detector
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Gap processing
    # -----------------------------------------------------------------

    "StageGapProcessingResult",
    "infer_expected_interval_minutes",
    "process_stage_gaps",
    "process_stage_gaps_from_files",
]