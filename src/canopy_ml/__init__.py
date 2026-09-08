"""CANOPY preproject ML contracts."""

from .trip_inference import (
    LocationSample,
    MODEL_ARTIFACT_NOT_AVAILABLE,
    PredictionObject,
    PredictionStatus,
    PreprocessingStatus,
    STRICT_PREPROCESSING_VERSION,
    StrictPreprocessingResult,
    TransportMode,
    prediction_without_model_artifact,
    preprocess_strict_trip,
)

try:
    from .speedtransformer_adapter import (
        ArtifactInferenceResult,
        ArtifactValidationError,
        SpeedTransformerPredictor,
        load_predictor,
        predict_with_optional_artifact,
        validate_artifact_files,
        validate_manifest,
    )
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "torch"}:
        raise

__all__ = [
    "LocationSample",
    "MODEL_ARTIFACT_NOT_AVAILABLE",
    "PredictionObject",
    "PredictionStatus",
    "PreprocessingStatus",
    "STRICT_PREPROCESSING_VERSION",
    "StrictPreprocessingResult",
    "TransportMode",
    "prediction_without_model_artifact",
    "preprocess_strict_trip",
]

if "ArtifactValidationError" in globals():
    __all__ += [
        "ArtifactInferenceResult",
        "ArtifactValidationError",
        "SpeedTransformerPredictor",
        "load_predictor",
        "predict_with_optional_artifact",
        "validate_artifact_files",
        "validate_manifest",
    ]
