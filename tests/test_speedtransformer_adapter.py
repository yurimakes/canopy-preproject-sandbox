import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from canopy_ml.speedtransformer_adapter import (  # noqa: E402
    ARTIFACT_STATUS,
    CONFIDENCE_SEMANTICS,
    EXPECTED_HYPERPARAMETERS,
    LABEL_MAPPING,
    MODEL_ARCHITECTURE,
    REFERENCE_COMMIT,
    SCALER_FORMAT,
    STRICT_LARGE_SHA256,
    SUPPORTED_ARTIFACT_VERSION,
    ArtifactValidationError,
    predict_with_optional_artifact,
    validate_artifact_files,
)
from canopy_ml.trip_inference import (  # noqa: E402
    LocationSample,
    MODEL_ARTIFACT_NOT_AVAILABLE,
    PredictionStatus,
    STRICT_PREPROCESSING_VERSION,
    preprocess_strict_trip,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def ready_preprocessing():
    return preprocess_strict_trip(
        [
            LocationSample(index * 1_000, 0.0, index * 0.00001)
            for index in range(201)
        ]
    )


def write_test_artifact(directory: Path) -> dict:
    model_bytes = b"TEST-ONLY-NOT-A-TORCH-MODEL"
    (directory / "model.pt").write_bytes(model_bytes)
    scaler = {
        "format": SCALER_FORMAT,
        "feature_names": ["speed_kmh"],
        "with_mean": True,
        "with_std": True,
        "mean": [10.0],
        "scale": [2.0],
        "var": [4.0],
        "n_features_in": 1,
        "n_samples_seen": 300000,
    }
    scaler_bytes = (json.dumps(scaler, indent=2, sort_keys=True) + "\n").encode()
    (directory / "scaler.json").write_bytes(scaler_bytes)
    manifest = {
        "artifact_version": SUPPORTED_ARTIFACT_VERSION,
        "artifact_status": ARTIFACT_STATUS,
        "model_architecture": MODEL_ARCHITECTURE,
        "reference_commit": REFERENCE_COMMIT,
        "dataset_sha256": STRICT_LARGE_SHA256,
        "preprocessing_version": STRICT_PREPROCESSING_VERSION,
        "label_mapping": LABEL_MAPPING,
        "label_mapping_status": "candidate",
        "seed": 316,
        "seed_selection_status": "fixed_baseline_not_selected_after_performance",
        "train_counts": {mode: 300 for mode in LABEL_MAPPING.values()},
        "validation_counts": {mode: 30 for mode in LABEL_MAPPING.values()},
        "internal_test_used": False,
        "hyperparameters": EXPECTED_HYPERPARAMETERS,
        "checkpoint_criterion": "validation_loss",
        "best_epoch": 1,
        "stop_epoch": 8,
        "validation_loss": 1.0,
        "validation_accuracy": 0.2,
        "validation_macro_f1": 0.1,
        "python_version": "test-only",
        "torch_version": "test-only",
        "sklearn_version": "test-only",
        "created_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_file": "model.pt",
        "scaler_file": "scaler.json",
        "model_sha256": sha256_bytes(model_bytes),
        "scaler_sha256": sha256_bytes(scaler_bytes),
        "confidence_semantics": CONFIDENCE_SEMANTICS,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


class ArtifactContractTests(unittest.TestCase):
    def test_candidate_label_mapping_is_exact(self) -> None:
        self.assertEqual(
            LABEL_MAPPING,
            {"0": "WALK", "1": "BIKE", "2": "CAR", "3": "BUS", "4": "SUBWAY"},
        )

    def test_metadata_schema_and_hashes_validate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_test_artifact(directory)
            files = validate_artifact_files(directory)
            self.assertEqual(files.manifest["artifact_status"], "preproject_candidate")

    def test_model_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_test_artifact(directory)
            (directory / "model.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ArtifactValidationError, "Model SHA256 mismatch"):
                validate_artifact_files(directory)

    def test_preprocessing_version_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = write_test_artifact(directory)
            changed = copy.deepcopy(manifest)
            changed["preprocessing_version"] = "incompatible-test-version"
            (directory / "manifest.json").write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactValidationError, "Preprocessing version mismatch"):
                validate_artifact_files(directory)

    def test_missing_artifact_keeps_model_not_run_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            absent = Path(temporary) / "absent"
            result = predict_with_optional_artifact(
                "synthetic-trip",
                ready_preprocessing(),
                absent,
                Path(temporary) / "unused-reference",
            )
            self.assertEqual(result.prediction.prediction_status, PredictionStatus.MODEL_NOT_RUN)
            self.assertEqual(result.prediction.failure_reason, MODEL_ARTIFACT_NOT_AVAILABLE)
            self.assertEqual(result.logits, ())


if __name__ == "__main__":
    unittest.main()
