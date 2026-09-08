"""Validated SpeedTransformer artifact loading and single-Trip inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .trip_inference import (
    PredictionObject,
    PredictionStatus,
    PreprocessingStatus,
    STRICT_PREPROCESSING_VERSION,
    StrictPreprocessingResult,
    TransportMode,
    prediction_without_model_artifact,
)


REFERENCE_COMMIT = "2c0e7ac2aa52813c0899b2f8ba7c6394ace2e959"
MODEL_ARCHITECTURE = "models/transformer/model_utils.py:TrajectoryTransformer"
SUPPORTED_ARTIFACT_VERSION = "speedtransformer-strict-large-seed316-v1"
STRICT_LARGE_SHA256 = "5CC827DAB88D17F68C5EF7F1A09127A81E8B5F683990EDAB0CC7D00302C275FA"
ARTIFACT_STATUS = "preproject_candidate"
SCALER_FORMAT = "sklearn_standard_scaler_json_v1"
CONFIDENCE_SEMANTICS = "uncalibrated_softmax_probability_of_predicted_class"
LABEL_MAPPING = {
    "0": "WALK",
    "1": "BIKE",
    "2": "CAR",
    "3": "BUS",
    "4": "SUBWAY",
}
EXPECTED_HYPERPARAMETERS = {
    "window_size": 200,
    "feature_size": 1,
    "num_classes": 5,
    "d_model": 128,
    "nhead": 8,
    "num_layers": 4,
    "kv_heads": 4,
    "dropout": 0.1,
    "batch_size": 64,
    "learning_rate": 0.0002,
    "weight_decay": 0.0001,
    "gradient_clip": 1.0,
    "max_epochs": 50,
    "patience": 7,
    "criterion": "CrossEntropyLoss",
    "optimizer": "AdamW",
}
REQUIRED_MANIFEST_FIELDS = {
    "artifact_version",
    "artifact_status",
    "model_architecture",
    "reference_commit",
    "dataset_sha256",
    "preprocessing_version",
    "label_mapping",
    "label_mapping_status",
    "seed",
    "seed_selection_status",
    "train_counts",
    "validation_counts",
    "internal_test_used",
    "hyperparameters",
    "checkpoint_criterion",
    "best_epoch",
    "stop_epoch",
    "validation_loss",
    "validation_accuracy",
    "validation_macro_f1",
    "python_version",
    "torch_version",
    "sklearn_version",
    "created_timestamp",
    "model_file",
    "scaler_file",
    "model_sha256",
    "scaler_sha256",
    "confidence_semantics",
}
REQUIRED_SCALER_FIELDS = {
    "format",
    "feature_names",
    "with_mean",
    "with_std",
    "mean",
    "scale",
    "var",
    "n_features_in",
    "n_samples_seen",
}


class ArtifactValidationError(RuntimeError):
    """Raised before inference when artifact integrity or compatibility fails."""


@dataclass(frozen=True)
class ArtifactFiles:
    artifact_dir: Path
    manifest: dict[str, Any]
    scaler: dict[str, Any]
    model_path: Path


@dataclass(frozen=True)
class ArtifactInferenceResult:
    prediction: PredictionObject
    logits: tuple[float, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"Could not read JSON artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"JSON artifact must contain an object: {path.name}")
    return value


def _validate_sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789ABCDEF" for character in value)
    ):
        raise ArtifactValidationError(f"{field_name} must be uppercase SHA256")
    return value


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest)
    if missing:
        raise ArtifactValidationError(f"Manifest missing fields: {sorted(missing)}")
    if manifest["artifact_version"] != SUPPORTED_ARTIFACT_VERSION:
        raise ArtifactValidationError("Artifact version mismatch")
    if manifest["artifact_status"] != ARTIFACT_STATUS:
        raise ArtifactValidationError("Artifact status is not a preproject candidate")
    if manifest["model_architecture"] != MODEL_ARCHITECTURE:
        raise ArtifactValidationError("Model architecture mismatch")
    if manifest["reference_commit"] != REFERENCE_COMMIT:
        raise ArtifactValidationError("Reference commit mismatch")
    _validate_sha256(manifest["dataset_sha256"], "dataset_sha256")
    if manifest["dataset_sha256"] != STRICT_LARGE_SHA256:
        raise ArtifactValidationError("Dataset SHA256 mismatch")
    if manifest["preprocessing_version"] != STRICT_PREPROCESSING_VERSION:
        raise ArtifactValidationError("Preprocessing version mismatch")
    if manifest["label_mapping"] != LABEL_MAPPING:
        raise ArtifactValidationError("Label mapping mismatch")
    if manifest["label_mapping_status"] != "candidate":
        raise ArtifactValidationError("Label mapping status mismatch")
    if manifest["seed"] != 316:
        raise ArtifactValidationError("Seed mismatch")
    if manifest["seed_selection_status"] != "fixed_baseline_not_selected_after_performance":
        raise ArtifactValidationError("Seed selection status mismatch")
    expected_counts = {mode: 300 for mode in LABEL_MAPPING.values()}
    expected_validation = {mode: 30 for mode in LABEL_MAPPING.values()}
    if manifest["train_counts"] != expected_counts:
        raise ArtifactValidationError("Train counts mismatch")
    if manifest["validation_counts"] != expected_validation:
        raise ArtifactValidationError("Validation counts mismatch")
    if manifest["internal_test_used"] is not False:
        raise ArtifactValidationError("Internal Test must not be used")
    if manifest["hyperparameters"] != EXPECTED_HYPERPARAMETERS:
        raise ArtifactValidationError("Hyperparameter mismatch")
    if manifest["checkpoint_criterion"] != "validation_loss":
        raise ArtifactValidationError("Checkpoint criterion mismatch")
    if manifest["confidence_semantics"] != CONFIDENCE_SEMANTICS:
        raise ArtifactValidationError("Confidence semantics mismatch")
    for field_name in ("best_epoch", "stop_epoch"):
        if not isinstance(manifest[field_name], int) or manifest[field_name] < 1:
            raise ArtifactValidationError(f"{field_name} must be a positive integer")
    if manifest["best_epoch"] > manifest["stop_epoch"]:
        raise ArtifactValidationError("best_epoch cannot exceed stop_epoch")
    for field_name in (
        "validation_loss",
        "validation_accuracy",
        "validation_macro_f1",
    ):
        value = manifest[field_name]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ArtifactValidationError(f"{field_name} must be finite")
    for field_name in ("validation_accuracy", "validation_macro_f1"):
        if not 0.0 <= float(manifest[field_name]) <= 1.0:
            raise ArtifactValidationError(f"{field_name} must be in [0, 1]")
    for field_name in ("python_version", "torch_version", "sklearn_version"):
        if not isinstance(manifest[field_name], str) or not manifest[field_name]:
            raise ArtifactValidationError(f"{field_name} must be recorded")
    try:
        created = datetime.fromisoformat(manifest["created_timestamp"])
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError("created_timestamp must be ISO 8601") from error
    if created.tzinfo is None:
        raise ArtifactValidationError("created_timestamp must include a timezone")
    for field_name in ("model_file", "scaler_file"):
        value = manifest[field_name]
        if not isinstance(value, str) or Path(value).name != value:
            raise ArtifactValidationError(f"{field_name} must be a plain filename")
    _validate_sha256(manifest["model_sha256"], "model_sha256")
    _validate_sha256(manifest["scaler_sha256"], "scaler_sha256")


def validate_scaler(scaler: Mapping[str, Any]) -> None:
    missing = REQUIRED_SCALER_FIELDS - set(scaler)
    if missing:
        raise ArtifactValidationError(f"Scaler missing fields: {sorted(missing)}")
    if scaler["format"] != SCALER_FORMAT:
        raise ArtifactValidationError("Scaler format mismatch")
    if scaler["feature_names"] != ["speed_kmh"]:
        raise ArtifactValidationError("Scaler feature mapping mismatch")
    if scaler["with_mean"] is not True or scaler["with_std"] is not True:
        raise ArtifactValidationError("Scaler must use mean and standard deviation")
    if scaler["n_features_in"] != 1:
        raise ArtifactValidationError("Scaler must contain exactly one feature")
    if not isinstance(scaler["n_samples_seen"], int) or scaler["n_samples_seen"] < 1:
        raise ArtifactValidationError("Invalid scaler sample count")
    for field_name in ("mean", "scale", "var"):
        values = scaler[field_name]
        if not isinstance(values, list) or len(values) != 1:
            raise ArtifactValidationError(f"Scaler {field_name} must have one value")
        if not isinstance(values[0], (int, float)) or not math.isfinite(float(values[0])):
            raise ArtifactValidationError(f"Scaler {field_name} must be finite")
    if float(scaler["scale"][0]) <= 0.0 or float(scaler["var"][0]) < 0.0:
        raise ArtifactValidationError("Invalid scaler variance or scale")


def validate_artifact_files(artifact_dir: Path) -> ArtifactFiles:
    artifact_dir = artifact_dir.resolve()
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactValidationError("manifest.json is missing")
    manifest = _read_json_object(manifest_path)
    validate_manifest(manifest)
    model_path = artifact_dir / manifest["model_file"]
    scaler_path = artifact_dir / manifest["scaler_file"]
    if not model_path.is_file():
        raise ArtifactValidationError("Model file is missing")
    if not scaler_path.is_file():
        raise ArtifactValidationError("Scaler file is missing")
    if sha256_file(model_path) != manifest["model_sha256"]:
        raise ArtifactValidationError("Model SHA256 mismatch")
    if sha256_file(scaler_path) != manifest["scaler_sha256"]:
        raise ArtifactValidationError("Scaler SHA256 mismatch")
    scaler = _read_json_object(scaler_path)
    validate_scaler(scaler)
    return ArtifactFiles(
        artifact_dir=artifact_dir,
        manifest=manifest,
        scaler=scaler,
        model_path=model_path,
    )


def _load_reference_model(reference_repo: Path) -> type[nn.Module]:
    reference_repo = reference_repo.resolve()
    model_path = reference_repo / "models" / "transformer" / "model_utils.py"
    if not model_path.is_file():
        raise ArtifactValidationError("Reference model module is missing")
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={reference_repo.as_posix()}",
            "-C",
            str(reference_repo),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != REFERENCE_COMMIT:
        raise ArtifactValidationError("Reference repository commit mismatch")
    spec = importlib.util.spec_from_file_location(
        "canopy_speedtransformer_artifact_model",
        model_path,
    )
    if spec is None or spec.loader is None:
        raise ArtifactValidationError("Could not construct reference model module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, "TrajectoryTransformer", None)
    if model_class is None:
        raise ArtifactValidationError("TrajectoryTransformer is missing")
    return model_class


class SpeedTransformerPredictor:
    def __init__(
        self,
        files: ArtifactFiles,
        model: nn.Module,
        device: torch.device,
    ) -> None:
        self.files = files
        self.model = model
        self.device = device

    @property
    def model_version(self) -> str:
        return (
            f"{self.files.manifest['artifact_version']}@sha256:"
            f"{self.files.manifest['model_sha256']}"
        )

    def predict(
        self,
        trip_id: str,
        preprocessing: StrictPreprocessingResult,
    ) -> ArtifactInferenceResult:
        if preprocessing.status is not PreprocessingStatus.READY:
            return ArtifactInferenceResult(
                prediction=prediction_without_model_artifact(trip_id, preprocessing),
                logits=(),
            )
        values = preprocessing.speed_values_kmh
        if values is None or len(values) != EXPECTED_HYPERPARAMETERS["window_size"]:
            raise ArtifactValidationError("READY preprocessing must contain 200 speeds")
        features = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        mean = float(self.files.scaler["mean"][0])
        scale = float(self.files.scaler["scale"][0])
        scaled = ((features - mean) / scale).astype(np.float32).reshape(1, 200, 1)
        tensor = torch.from_numpy(scaled).to(self.device)
        with torch.no_grad():
            logits_tensor = self.model(
                tensor,
                None,
                src_key_padding_mask=None,
            )
        if logits_tensor.shape != (1, 5) or not torch.isfinite(logits_tensor).all():
            raise ArtifactValidationError("Model produced invalid logits")
        probabilities = torch.softmax(logits_tensor, dim=1)
        predicted_index = int(logits_tensor.argmax(dim=1).item())
        model_score = float(probabilities[0, predicted_index].item())
        if not math.isfinite(model_score) or not 0.0 <= model_score <= 1.0:
            raise ArtifactValidationError("Model produced an invalid softmax probability")
        mode = TransportMode(self.files.manifest["label_mapping"][str(predicted_index)])
        prediction = PredictionObject(
            trip_id=trip_id,
            preprocessing_status=preprocessing.status,
            prediction_status=PredictionStatus.PREDICTED,
            model_prediction=mode,
            model_confidence=model_score,
            user_confirmed_mode=None,
            model_version=self.model_version,
            preprocessing_version=preprocessing.preprocessing_version,
            prediction_timestamp=datetime.now(timezone.utc).isoformat(),
            failure_reason=None,
        )
        return ArtifactInferenceResult(
            prediction=prediction,
            logits=tuple(float(value) for value in logits_tensor[0].cpu().tolist()),
        )


def load_predictor(
    artifact_dir: Path,
    reference_repo: Path,
    device: str | torch.device = "cpu",
) -> SpeedTransformerPredictor:
    files = validate_artifact_files(artifact_dir)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ArtifactValidationError("CUDA was requested but is unavailable")
    model_class = _load_reference_model(reference_repo)
    hyperparameters = files.manifest["hyperparameters"]
    model = model_class(
        feature_size=hyperparameters["feature_size"],
        num_classes=hyperparameters["num_classes"],
        d_model=hyperparameters["d_model"],
        nhead=hyperparameters["nhead"],
        num_layers=hyperparameters["num_layers"],
        window_size=hyperparameters["window_size"],
        dropout=hyperparameters["dropout"],
        kv_heads=hyperparameters["kv_heads"],
    ).to(torch_device)
    try:
        state_dict = torch.load(
            files.model_path,
            map_location=torch_device,
            weights_only=True,
        )
        model.load_state_dict(state_dict, strict=True)
    except Exception as error:
        raise ArtifactValidationError("Could not load model state_dict") from error
    model.eval()
    return SpeedTransformerPredictor(files=files, model=model, device=torch_device)


def predict_with_optional_artifact(
    trip_id: str,
    preprocessing: StrictPreprocessingResult,
    artifact_dir: Path | None,
    reference_repo: Path,
    device: str | torch.device = "cpu",
) -> ArtifactInferenceResult:
    if artifact_dir is None or not artifact_dir.exists():
        return ArtifactInferenceResult(
            prediction=prediction_without_model_artifact(trip_id, preprocessing),
            logits=(),
        )
    predictor = load_predictor(artifact_dir, reference_repo, device=device)
    return predictor.predict(trip_id, preprocessing)
