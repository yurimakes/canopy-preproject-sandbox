# P0 single-Trip STRICT preprocessing checkpoint

Date: 2026-09-08

Status: preprocessing path implemented and locally tested; model inference is not available.

This is a preproject P0 checkpoint. It is not an E2E service implementation, production interface, model-performance result, or team-approved preprocessing decision. No training was run and Internal Test was not used.

## Artifact audit

The current repository was searched across tracked and ignored local files, excluding environment packages and raw datasets. Existing local Colab ZIP entry names were also checked.

### Model artifact status

`NOT_AVAILABLE`

- No `.pt`, `.pth`, `.ckpt`, `.onnx`, `.safetensors`, pickle, or joblib model artifact was found.
- No model metadata file associated with deployable weights was found.
- `scripts/17_compare_strict_vs_mild_release.py` copies the best `state_dict` in memory and restores it for evaluation, but contains no `torch.save` or artifact export.
- Existing experiment logs and training configuration reports do not contain deployable weights.

### Scaler artifact status

`NOT_AVAILABLE`

- No fitted `StandardScaler` artifact was found.
- Script 17 fits a scaler in memory on each arm's Train data and transforms Validation data; it does not persist the scaler.

### Model version source

`NOT_AVAILABLE`

The pinned reference commit and runner commit describe code provenance but do not identify trained weights. Without an artifact digest and a compatible scaler/preprocessing manifest, no deployable `model_version` can be assigned.

## Implemented P0 scope

The new `src/canopy_ml/trip_inference.py` module implements:

- a candidate five-class `TransportMode` enum;
- a typed transient `LocationSample` input;
- reusable single-Trip STRICT preprocessing;
- typed preprocessing success/failure results;
- a storage-neutral prediction object with separate model and user-confirmed fields;
- an explicit no-artifact path that returns `MODEL_NOT_RUN` without fabricating a prediction.

The module uses only the Python standard library.

## Input contract

`preprocess_strict_trip` accepts a sequence of `LocationSample` values:

| Field | Type | Meaning |
|---|---|---|
| `timestamp_ms` | integer | Timestamp in Unix milliseconds |
| `latitude` | finite number | Decimal degrees in `[-90, 90]` |
| `longitude` | finite number | Decimal degrees in `[-180, 180]` |

Input location data is transient and is not present in `PredictionObject`.

## STRICT preprocessing contract

The implementation follows the current personal preproject baseline candidate:

- sort by timestamp;
- collapse identical duplicate timestamp/coordinate observations;
- mark conflicting duplicate timestamps as unusable in a candidate window;
- find exact-1-second timestamp runs;
- search deterministically for the first clean 201-point window;
- validate all 201 coordinates in that window;
- calculate 200 pairwise Haversine speeds using actual `dt`;
- return speed in km/h;
- do not interpolate, impute, apply class-specific cutoffs, clip, or smooth.

A Trip may contain more than 201 observations. The implementation selects the first qualifying window rather than requiring the whole Trip to contain exactly 201 points.

Preprocessing version: `strict-speed200-candidate-v1`. The word `candidate` is intentional; this is not a team-approved production version.

## Window policy

`FIRST_CLEAN_201_POINT_EXACT_1S_WINDOW`

This matches the deterministic first-window convention in `scripts/12_make_speedtransformer_smoke_dataset.py`. Multi-window aggregation or voting is not implemented.

## Preprocessing states

| State | Meaning |
|---|---|
| `READY` | First clean 201-point exact-1s window produced 200 finite speeds |
| `INSUFFICIENT_POINTS` | Fewer than 201 unique timestamps remain |
| `INVALID_TIMESTAMP` | A timestamp is not an integer millisecond value |
| `NON_CONTIGUOUS` | At least 201 unique timestamps exist but no exact-1s run reaches 201 points |
| `INVALID_COORDINATE` | Candidate runs exist but no 201-point window has valid, non-conflicting coordinates, or a derived speed is non-finite |

No failure state silently relaxes STRICT or fills missing data.

## Prediction object

Implemented fields:

- `trip_id`
- `preprocessing_status`
- `prediction_status`
- `model_prediction`
- `model_confidence`
- `user_confirmed_mode`
- `model_version`
- `preprocessing_version`
- `prediction_timestamp`
- `failure_reason`

`model_prediction` and `user_confirmed_mode` are separate fields.

Because no model artifact exists, `prediction_without_model_artifact` always returns:

- `prediction_status=MODEL_NOT_RUN`;
- `model_prediction=null`;
- `model_confidence=null`;
- `model_version=null`;
- `prediction_timestamp=null`;
- `failure_reason=MODEL_ARTIFACT_NOT_AVAILABLE` after successful preprocessing, or a typed preprocessing failure reason otherwise.

No confidence is calculated. Current experiment code exposes logits and argmax only; confidence semantics and calibration are not implemented.

## Candidate label contract

Exact order:

1. `WALK`
2. `BIKE`
3. `CAR`
4. `BUS`
5. `SUBWAY`

The implemented enum contains only the five labels above. `ETC` is excluded, and `SUBWAY` is not renamed.

## Local test result

Command:

```text
python3 -m unittest discover -s tests -v
```

Result: `Ran 9 tests ... OK`

Verified cases:

- a synthetic 250-point exact-1s Trip selects the first 201 points;
- `READY` produces exactly 200 finite speed values;
- 200 points return `INSUFFICIENT_POINTS`;
- one 2-second gap prevents a 201-point strict run and returns `NON_CONTIGUOUS`;
- a non-integer timestamp returns `INVALID_TIMESTAMP`;
- a non-finite coordinate returns `INVALID_COORDINATE`;
- the label enum is exactly `WALK, BIKE, CAR, BUS, SUBWAY`;
- successful preprocessing without an artifact returns `MODEL_NOT_RUN` and no prediction/confidence/version;
- failed preprocessing also prevents model execution;
- the prediction object contains no location coordinates, timestamps, or source UID/TID fields.

The fixtures are generated in test code from deterministic synthetic equatorial points. They contain no copied AI Hub trajectory, raw UID/TID, personal route, home, school, or workplace location.

## Not implemented

- trained model artifact loading;
- fitted scaler artifact loading;
- actual model inference;
- confidence calculation;
- model version assignment;
- Trip-ending event or application adapter;
- persistence, API, UI, confirmation handling, carbon calculation, or Azure integration.

## Blocker and next P0 step

Actual prediction is blocked by the absence of a compatible deployable SpeedTransformer weights artifact and fitted-scaler/version manifest.

The next P0 step is to identify an approved artifact source and bind weights, scaler, class order, model configuration, reference commit, preprocessing version, and artifact digest. Only then can an inference-only adapter be implemented and a single synthetic Trip be run through preprocessing to an actual prediction object.
