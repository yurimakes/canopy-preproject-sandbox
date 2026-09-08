"""Minimal single-Trip preprocessing and prediction contracts for CANOPY P0."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from typing import Sequence


EARTH_RADIUS_M = 6_371_008.8
STRICT_POINT_COUNT = 201
STRICT_SPEED_COUNT = 200
STRICT_GAP_MS = 1_000
STRICT_PREPROCESSING_VERSION = "strict-speed200-candidate-v1"
MODEL_ARTIFACT_NOT_AVAILABLE = "MODEL_ARTIFACT_NOT_AVAILABLE"


class TransportMode(str, Enum):
    """Candidate five-class model label contract."""

    WALK = "WALK"
    BIKE = "BIKE"
    CAR = "CAR"
    BUS = "BUS"
    SUBWAY = "SUBWAY"


class PreprocessingStatus(str, Enum):
    READY = "READY"
    INSUFFICIENT_POINTS = "INSUFFICIENT_POINTS"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    NON_CONTIGUOUS = "NON_CONTIGUOUS"
    INVALID_COORDINATE = "INVALID_COORDINATE"


class PredictionStatus(str, Enum):
    PREDICTED = "PREDICTED"
    MODEL_NOT_RUN = "MODEL_NOT_RUN"
    MODEL_ERROR = "MODEL_ERROR"


@dataclass(frozen=True)
class LocationSample:
    """One transient location observation; timestamp is integer Unix milliseconds."""

    timestamp_ms: int
    latitude: float
    longitude: float


@dataclass(frozen=True)
class StrictPreprocessingResult:
    status: PreprocessingStatus
    preprocessing_version: str
    speed_values_kmh: tuple[float, ...] | None
    selected_point_count: int
    selected_start_timestamp_ms: int | None
    failure_reason: str | None


@dataclass(frozen=True)
class PredictionObject:
    """Storage-neutral prediction object; raw locations are intentionally absent."""

    trip_id: str
    preprocessing_status: PreprocessingStatus
    prediction_status: PredictionStatus
    model_prediction: TransportMode | None
    model_confidence: float | None
    user_confirmed_mode: TransportMode | None
    model_version: str | None
    preprocessing_version: str
    prediction_timestamp: str | None
    failure_reason: str | None


def _failed_result(
    status: PreprocessingStatus,
    reason: str,
    selected_point_count: int = 0,
) -> StrictPreprocessingResult:
    return StrictPreprocessingResult(
        status=status,
        preprocessing_version=STRICT_PREPROCESSING_VERSION,
        speed_values_kmh=None,
        selected_point_count=selected_point_count,
        selected_start_timestamp_ms=None,
        failure_reason=reason,
    )


def _valid_timestamp(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_coordinate(latitude: object, longitude: object) -> bool:
    if (
        not isinstance(latitude, Real)
        or isinstance(latitude, bool)
        or not isinstance(longitude, Real)
        or isinstance(longitude, bool)
    ):
        return False
    return (
        math.isfinite(float(latitude))
        and math.isfinite(float(longitude))
        and -90.0 <= float(latitude) <= 90.0
        and -180.0 <= float(longitude) <= 180.0
    )


def _haversine_m(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    phi_1 = math.radians(latitude_1)
    phi_2 = math.radians(latitude_2)
    delta_phi = math.radians(latitude_2 - latitude_1)
    delta_lambda = math.radians(longitude_2 - longitude_1)
    value = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_1)
        * math.cos(phi_2)
        * math.sin(delta_lambda / 2.0) ** 2
    )
    value = min(1.0, max(0.0, value))
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(value))


def _exact_gap_runs(timestamps: list[int]) -> list[list[int]]:
    if not timestamps:
        return []
    runs: list[list[int]] = []
    current = [timestamps[0]]
    for previous, timestamp in zip(timestamps, timestamps[1:]):
        if timestamp - previous == STRICT_GAP_MS:
            current.append(timestamp)
        else:
            runs.append(current)
            current = [timestamp]
    runs.append(current)
    return runs


def preprocess_strict_trip(
    samples: Sequence[LocationSample],
) -> StrictPreprocessingResult:
    """Return the first clean 201-point exact-1s window and 200 actual-dt speeds.

    Identical duplicate timestamps are collapsed. Conflicting duplicates cannot be
    part of an accepted window. No point or speed value is fabricated or modified.
    """

    if any(not _valid_timestamp(sample.timestamp_ms) for sample in samples):
        return _failed_result(
            PreprocessingStatus.INVALID_TIMESTAMP,
            "TIMESTAMP_MUST_BE_INTEGER_MILLISECONDS",
        )

    points: dict[int, tuple[object, object]] = {}
    conflicts: set[int] = set()
    for sample in samples:
        value = (sample.latitude, sample.longitude)
        if sample.timestamp_ms in points and points[sample.timestamp_ms] != value:
            conflicts.add(sample.timestamp_ms)
        else:
            points[sample.timestamp_ms] = value

    timestamps = sorted(points)
    if len(timestamps) < STRICT_POINT_COUNT:
        return _failed_result(
            PreprocessingStatus.INSUFFICIENT_POINTS,
            "FEWER_THAN_201_UNIQUE_TIMESTAMPS",
            selected_point_count=len(timestamps),
        )

    runs = _exact_gap_runs(timestamps)
    candidate_run_found = False
    for run in runs:
        if len(run) < STRICT_POINT_COUNT:
            continue
        candidate_run_found = True
        for start in range(len(run) - STRICT_POINT_COUNT + 1):
            selected = run[start : start + STRICT_POINT_COUNT]
            if any(
                timestamp in conflicts
                or not _valid_coordinate(*points[timestamp])
                for timestamp in selected
            ):
                continue

            speeds: list[float] = []
            for first, second in zip(selected, selected[1:]):
                delta_seconds = (second - first) / 1_000.0
                if delta_seconds != 1.0:
                    raise AssertionError("Selected STRICT window is not exact-1s")
                latitude_1, longitude_1 = points[first]
                latitude_2, longitude_2 = points[second]
                speed_kmh = (
                    _haversine_m(
                        float(latitude_1),
                        float(longitude_1),
                        float(latitude_2),
                        float(longitude_2),
                    )
                    / delta_seconds
                    * 3.6
                )
                if not math.isfinite(speed_kmh):
                    return _failed_result(
                        PreprocessingStatus.INVALID_COORDINATE,
                        "NON_FINITE_DERIVED_SPEED",
                        selected_point_count=len(selected),
                    )
                speeds.append(speed_kmh)

            if len(speeds) != STRICT_SPEED_COUNT:
                raise AssertionError("STRICT window did not produce 200 speeds")
            return StrictPreprocessingResult(
                status=PreprocessingStatus.READY,
                preprocessing_version=STRICT_PREPROCESSING_VERSION,
                speed_values_kmh=tuple(speeds),
                selected_point_count=len(selected),
                selected_start_timestamp_ms=selected[0],
                failure_reason=None,
            )

    if not candidate_run_found:
        return _failed_result(
            PreprocessingStatus.NON_CONTIGUOUS,
            "NO_201_POINT_EXACT_1S_RUN",
            selected_point_count=max(map(len, runs), default=0),
        )
    return _failed_result(
        PreprocessingStatus.INVALID_COORDINATE,
        "NO_CANDIDATE_WINDOW_WITH_201_VALID_COORDINATES",
    )


def prediction_without_model_artifact(
    trip_id: str,
    preprocessing: StrictPreprocessingResult,
) -> PredictionObject:
    """Create the P0 prediction object when no deployable model artifact exists."""

    if not isinstance(trip_id, str) or not trip_id.strip():
        raise ValueError("trip_id must be a non-empty opaque string")
    failure_reason = (
        MODEL_ARTIFACT_NOT_AVAILABLE
        if preprocessing.status is PreprocessingStatus.READY
        else f"PREPROCESSING_{preprocessing.status.value}"
    )
    return PredictionObject(
        trip_id=trip_id,
        preprocessing_status=preprocessing.status,
        prediction_status=PredictionStatus.MODEL_NOT_RUN,
        model_prediction=None,
        model_confidence=None,
        user_confirmed_mode=None,
        model_version=None,
        preprocessing_version=preprocessing.preprocessing_version,
        prediction_timestamp=None,
        failure_reason=failure_reason,
    )
