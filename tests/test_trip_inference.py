from dataclasses import asdict
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from canopy_ml.trip_inference import (  # noqa: E402
    LocationSample,
    MODEL_ARTIFACT_NOT_AVAILABLE,
    PredictionStatus,
    PreprocessingStatus,
    TransportMode,
    prediction_without_model_artifact,
    preprocess_strict_trip,
)


def synthetic_samples(count: int = 201) -> list[LocationSample]:
    """Generate non-personal deterministic observations near the equator."""

    return [
        LocationSample(
            timestamp_ms=index * 1_000,
            latitude=0.0,
            longitude=index * 0.00001,
        )
        for index in range(count)
    ]


class StrictTripPreprocessingTests(unittest.TestCase):
    def test_first_qualifying_window_from_longer_trip_is_ready(self) -> None:
        result = preprocess_strict_trip(synthetic_samples(250))

        self.assertEqual(result.status, PreprocessingStatus.READY)
        self.assertEqual(result.selected_point_count, 201)
        self.assertEqual(result.selected_start_timestamp_ms, 0)
        self.assertIsNotNone(result.speed_values_kmh)
        self.assertEqual(len(result.speed_values_kmh or ()), 200)
        self.assertTrue(all(math.isfinite(value) for value in result.speed_values_kmh or ()))

    def test_fewer_than_201_points_is_rejected(self) -> None:
        result = preprocess_strict_trip(synthetic_samples(200))

        self.assertEqual(result.status, PreprocessingStatus.INSUFFICIENT_POINTS)
        self.assertIsNone(result.speed_values_kmh)

    def test_two_second_gap_is_non_contiguous(self) -> None:
        samples = synthetic_samples()
        samples = [
            LocationSample(
                timestamp_ms=(
                    sample.timestamp_ms
                    if index <= 100
                    else sample.timestamp_ms + 1_000
                ),
                latitude=sample.latitude,
                longitude=sample.longitude,
            )
            for index, sample in enumerate(samples)
        ]

        result = preprocess_strict_trip(samples)

        self.assertEqual(result.status, PreprocessingStatus.NON_CONTIGUOUS)
        self.assertIsNone(result.speed_values_kmh)

    def test_invalid_timestamp_is_rejected(self) -> None:
        samples = synthetic_samples()
        samples[100] = LocationSample(100.5, 0.0, 0.001)  # type: ignore[arg-type]

        result = preprocess_strict_trip(samples)

        self.assertEqual(result.status, PreprocessingStatus.INVALID_TIMESTAMP)
        self.assertIsNone(result.speed_values_kmh)

    def test_invalid_coordinate_is_rejected(self) -> None:
        samples = synthetic_samples()
        samples[100] = LocationSample(100_000, math.nan, 0.001)

        result = preprocess_strict_trip(samples)

        self.assertEqual(result.status, PreprocessingStatus.INVALID_COORDINATE)
        self.assertIsNone(result.speed_values_kmh)


class PredictionContractTests(unittest.TestCase):
    def test_candidate_label_contract_is_exact(self) -> None:
        self.assertEqual(
            [mode.value for mode in TransportMode],
            ["WALK", "BIKE", "CAR", "BUS", "SUBWAY"],
        )

    def test_ready_preprocessing_without_artifact_does_not_run_model(self) -> None:
        preprocessing = preprocess_strict_trip(synthetic_samples())

        prediction = prediction_without_model_artifact(
            "synthetic-trip",
            preprocessing,
        )

        self.assertEqual(prediction.prediction_status, PredictionStatus.MODEL_NOT_RUN)
        self.assertEqual(prediction.failure_reason, MODEL_ARTIFACT_NOT_AVAILABLE)
        self.assertIsNone(prediction.model_prediction)
        self.assertIsNone(prediction.model_confidence)
        self.assertIsNone(prediction.user_confirmed_mode)
        self.assertIsNone(prediction.model_version)
        self.assertIsNone(prediction.prediction_timestamp)

    def test_preprocessing_failure_does_not_run_model(self) -> None:
        preprocessing = preprocess_strict_trip(synthetic_samples(200))

        prediction = prediction_without_model_artifact(
            "synthetic-trip",
            preprocessing,
        )

        self.assertEqual(prediction.prediction_status, PredictionStatus.MODEL_NOT_RUN)
        self.assertEqual(
            prediction.failure_reason,
            "PREPROCESSING_INSUFFICIENT_POINTS",
        )

    def test_prediction_object_contains_no_raw_location_or_source_ids(self) -> None:
        preprocessing = preprocess_strict_trip(synthetic_samples())
        prediction = prediction_without_model_artifact("synthetic-trip", preprocessing)
        stored_fields = set(asdict(prediction))

        self.assertTrue(
            stored_fields.isdisjoint(
                {"latitude", "longitude", "timestamp_ms", "uid", "tid"}
            )
        )


if __name__ == "__main__":
    unittest.main()
