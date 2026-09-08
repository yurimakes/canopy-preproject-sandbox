# ML Trip inference interface working contract

Date: 2026-09-08

Status: audit-backed preproject contract proposal. This is not a team-approved production interface and does not represent completed E2E implementation.

## 1. Purpose

Define the minimum boundary for this intended flow:

1. a Trip ends;
2. its transient location samples enter inference preprocessing;
3. preprocessing either produces a 200-speed model input or a typed rejection;
4. the model either produces a transportation-mode prediction or is not run;
5. the prediction record can later be persisted;
6. a user can confirm or correct the prediction;
7. the confirmed mode can later be persisted separately;
8. a future carbon calculation can consume the confirmed mode.

This document fixes names and invariants for the next implementation step. It does not add an API, database, user interface, Azure resource, carbon calculator, or production model service.

Internal Test remains sealed and is not used to define or validate this contract.

## 2. Current implementation audit

The audit used tracked repository files. Ignored raw/processed data and ZIP bundles are not counted as application implementation.

| Area | Status | Verified repository evidence | Missing boundary |
|---|---|---|---|
| Trip | Planned | `reports/daily/CANOPY_DAILY_20260908.md` names Trip inference design as the next P0 | No CANOPY Trip type, lifecycle, identifier contract, or Trip-ending event was found |
| GPS/location | Partially Implemented | Offline AI Hub scripts parse timestamps and coordinates, validate coordinates, and compute Haversine distance; for example `scripts/12_make_speedtransformer_smoke_dataset.py` | No service/iPhone location input adapter, permission flow, collection lifecycle, or operational location schema was found |
| Preprocessing | Partially Implemented | Scripts 12, 16, 18, and 21 build experimental speed datasets; the STRICT working decision is recorded in `reports/decisions/GPS_CONTINUITY_PREPROCESSING_WORKING_DECISION_20260908.md` | Logic is script-oriented and tied to offline data files; no reusable single-Trip preprocessing interface exists |
| SpeedTransformer | Partially Implemented | `scripts/17_compare_strict_vs_mild_release.py` imports the pinned reference `TrajectoryTransformer`, constructs it, and runs Train/Validation experiments | No tracked deployable weights, production model package, or serving adapter was found |
| Inference/prediction | Partially Implemented | Script 17 evaluates batches and obtains a class index with `logits.argmax(dim=1)` | No single-Trip prediction object, confidence output, model artifact loader, or inference-only entry point exists |
| API/backend | Not Found | No application source files, web framework, endpoint, router, or controller were found | Target backend repository and technology stack need verification |
| Prediction storage | Not Found | No database code, migration, repository, or prediction schema was found | Persistence ownership and schema need verification |
| User confirmation/correction | Not Found | No confirmation domain type, UI, endpoint, or storage implementation was found | Product flow, authentication, and update rules need verification |
| Carbon calculation | Not Found | No carbon calculation code or mode-to-factor contract was found | Carbon method, factors, and ownership need verification |
| Database/schema | Not Found | No SQL, ORM model, migration, or application schema file was found | Target database and migration process need verification |
| Azure | Not Found | No Azure configuration, SDK integration, deployment manifest, or resource definition was found | Hosting and cloud-resource choices need verification |

The external SpeedTransformer reference repository and its reports are evidence for ML experiments, not evidence that a CANOPY application interface is implemented.

## 3. Trip inference input contract

This is a proposed in-memory boundary. There is no existing CANOPY Trip schema to reuse.

### `TripInferenceRequest`

| Field | Type | Required | Contract |
|---|---|---:|---|
| `trip_id` | opaque string | yes | Stable identifier supplied by the Trip owner; must not encode a user identifier or location |
| `trip_end_timestamp` | UTC timestamp | yes | Time at which inference was requested after Trip completion |
| `location_samples` | ordered sequence of `LocationSample` | yes | Transient input to preprocessing; not part of the operational prediction-record example |

### `LocationSample`

| Field | Type | Required | Current use |
|---|---|---:|---|
| `timestamp` | UTC timestamp with subsecond precision preserved | yes | Ordering and adjacent-gap validation |
| `latitude` | finite decimal degrees | yes | Haversine distance |
| `longitude` | finite decimal degrees | yes | Haversine distance |
| `horizontal_accuracy` | finite distance or null | no | Not used by the current experimental preprocessing; acceptance behavior needs verification before adoption |

Input invariants:

- Samples are sorted by `timestamp` inside preprocessing; caller ordering is not trusted.
- Duplicate timestamps with conflicting coordinates are rejected for the candidate window.
- Latitude and longitude must be finite and within valid geographic ranges.
- No transportation label, AI Hub label, user confirmation, or carbon value is accepted as preprocessing input.
- Location samples are handled as transient sensitive data. This document does not authorize persistence or retention.

## 4. Preprocessing result contract

The current working baseline candidate is STRICT, not a team-approved production policy.

### STRICT candidate invariants

- exactly 201 valid GPS points in the selected window;
- every adjacent timestamp gap is exactly 1 second;
- exactly 200 speed values;
- speed is Haversine distance divided by the actual timestamp difference and expressed in km/h;
- no interpolation;
- no missing-value imputation;
- no class-specific cutoff;
- no clipping;
- no smoothing.

The offline implementation in script 12 selects the first qualifying clean window. Whether a production Trip may yield one window, several windows, or an aggregated prediction is not implemented and needs a separate team decision. P0 should initially require exactly one selected candidate window and record the selection rule explicitly.

### Proposed `PreprocessingResult`

| Field | Type | Required | Contract |
|---|---|---:|---|
| `trip_id` | opaque string | yes | Echoes the request identifier |
| `preprocessing_status` | enum | yes | `READY`, `INSUFFICIENT_DATA`, or `PREPROCESSING_REJECTED` |
| `preprocessing_version` | immutable string | yes | Identifies algorithm and parameter set; candidate name must not imply team approval |
| `speed_values_kmh` | array of 200 finite numbers or null | conditional | Present only for `READY`; transient model input, not proposed persistence data |
| `selected_point_count` | integer | yes | `201` for `READY`; otherwise the accepted count used to explain failure |
| `failure_reason` | stable code or null | conditional | Required when status is not `READY`; must not contain coordinates or user data |

State meanings:

- `READY`: all STRICT candidate invariants passed and 200 model features exist.
- `INSUFFICIENT_DATA`: no candidate run contains 201 valid points with exact 1-second adjacent gaps.
- `PREPROCESSING_REJECTED`: malformed timestamps, conflicting duplicate observations, invalid coordinates, non-finite speed, or another explicit integrity failure prevented preprocessing.

No fallback may interpolate, impute, clip, smooth, or silently relax the gap threshold. When preprocessing is not `READY`, prediction state is `MODEL_NOT_RUN`.

## 5. Model prediction output contract

### Candidate model labels

The current experimental runner defines this exact order:

| Model label | Mode |
|---:|---|
| 0 | `WALK` |
| 1 | `BIKE` |
| 2 | `CAR` |
| 3 | `BUS` |
| 4 | `SUBWAY` |

AI Hub raw labels and detail labels are source-data metadata, not model labels. The tracked label audit observed multiple detail codes inside some folder classes and excluded the `ETC` folder from the candidate five-class dataset. `ETC` is therefore not a candidate model output.

No tracked team decision renames `SUBWAY` to `RAIL`; this contract retains `SUBWAY`.

### Proposed `ModelPredictionResult`

| Field | Type | Required | Contract |
|---|---|---:|---|
| `trip_id` | opaque string | yes | Joins the result to the Trip |
| `prediction_status` | enum | yes | `PREDICTED`, `MODEL_NOT_RUN`, or `MODEL_ERROR` |
| `model_prediction` | five-class mode or null | conditional | Required only for `PREDICTED`; derived from the selected class index |
| `model_confidence` | number or null | no | Must remain null until score semantics and calibration are implemented and verified |
| `model_version` | immutable string or null | conditional | Required for `PREDICTED`; identifies the exact weights, not only source code |
| `preprocessing_version` | immutable string | yes | Copied from preprocessing |
| `prediction_timestamp` | UTC timestamp or null | conditional | Set when model execution finishes; null for `MODEL_NOT_RUN` |
| `failure_reason` | stable code or null | conditional | Required for non-`PREDICTED` results; contains no sensitive input |

Current code produces logits and an argmax class index during experimental evaluation, but does not emit confidence. Although a numerical score could be derived from logits, it must not be called calibrated confidence without a defined transformation and validation. Therefore `model_confidence` is nullable in the contract.

`MODEL_NOT_RUN` is used when preprocessing is not `READY` or no compatible model artifact is available. `MODEL_ERROR` is reserved for an attempted model invocation that fails.

## 6. User confirmation contract

Model output and user-confirmed mode are separate facts and must never share one field.

### Proposed `TripModeConfirmation`

| Field | Type | Required | Contract |
|---|---|---:|---|
| `trip_id` | opaque string | yes | Target Trip |
| `model_prediction` | five-class mode or null | yes | Immutable snapshot of what the model predicted; null if the model was not run |
| `user_confirmed_mode` | supported product mode or null | conditional | Null before user action; populated only by an explicit user action |
| `confirmation_status` | enum | yes | `PENDING`, `CONFIRMED_AS_PREDICTED`, `CORRECTED_BY_USER`, or `MANUALLY_PROVIDED` |
| `confirmation_timestamp` | UTC timestamp or null | conditional | Required after confirmation or correction |

State rules:

- `PENDING`: prediction may exist, but no confirmed user value exists.
- `CONFIRMED_AS_PREDICTED`: the user explicitly accepted the prediction; both mode fields are present and equal.
- `CORRECTED_BY_USER`: the user explicitly selected a different mode; both fields remain present and different.
- `MANUALLY_PROVIDED`: no model prediction exists and the user explicitly supplied a mode.

A correction is not automatically a training label. Reuse for training requires a separate consent, quality-review, provenance, sampling, and dataset-versioning process that is not implemented here.

## 7. Proposed operational storage fields

No database implementation exists. The following is a storage-neutral record proposal and intentionally contains no raw GPS or speed sequence.

| Field | Nullability | Notes |
|---|---|---|
| `trip_id` | non-null | Opaque join key |
| `preprocessing_status` | non-null | Result of STRICT candidate preprocessing |
| `prediction_status` | non-null | Whether model execution occurred |
| `model_prediction` | nullable | Immutable model output |
| `model_confidence` | nullable | Null until implemented and verified |
| `user_confirmed_mode` | nullable | Separate explicit user result |
| `confirmation_status` | non-null | Starts at `PENDING` when a prediction awaits review |
| `model_version` | nullable | Required whenever `model_prediction` is non-null |
| `preprocessing_version` | non-null | Required even for preprocessing failure records |
| `prediction_timestamp` | nullable | UTC |
| `confirmation_timestamp` | nullable | UTC |
| `failure_reason` | nullable | Stable non-sensitive code |

Suggested constraints:

- `model_prediction` and `user_confirmed_mode` are distinct columns and independently immutable/auditable facts.
- A non-null `model_prediction` requires `prediction_status=PREDICTED`, `model_version`, and `prediction_timestamp`.
- A non-null `user_confirmed_mode` requires a non-`PENDING` confirmation status and `confirmation_timestamp`.
- A carbon consumer may use `user_confirmed_mode` only when confirmation status records an explicit user action.
- Behavior for `PENDING` predictions in carbon calculation is not defined here and needs product verification; prediction must not be silently represented as user-confirmed.

## 8. Failure and fallback states

| Stage | State | Minimum behavior |
|---|---|---|
| Trip input | `INVALID_REQUEST` | Reject missing/invalid `trip_id`, end time, or location payload without model execution |
| Preprocessing | `INSUFFICIENT_DATA` | Record a non-sensitive reason; do not fabricate points or speeds |
| Preprocessing | `PREPROCESSING_REJECTED` | Record integrity failure; do not relax STRICT automatically |
| Prediction | `MODEL_NOT_RUN` | Keep `model_prediction` and `model_confidence` null |
| Prediction | `MODEL_ERROR` | Keep prediction fields null and expose a stable error code, not internal stack data |
| Confirmation | `PENDING` | Do not populate `user_confirmed_mode` |
| Confirmation | `MANUALLY_PROVIDED` | Allow explicit user selection when no prediction exists, subject to future product/API rules |

Retry, timeout, idempotency, and event-delivery semantics are not implemented and need verification in the target backend.

## 9. Model and preprocessing versioning

`model_version` must identify exact deployable weights. A reference repository commit or architecture name alone is insufficient. The future model manifest should bind at least:

- model artifact digest;
- architecture/reference commit;
- training runner commit;
- training dataset digest;
- class order;
- compatible preprocessing version.

No tracked model artifact exists, so no current production `model_version` can be assigned.

`preprocessing_version` must identify the algorithm and parameters, including point count, exact gap policy, feature definition, units, and window-selection rule. The STRICT policy is only a baseline candidate, so any initial identifier must retain candidate status until team approval.

Persist both versions on every prediction attempt so later confirmation and carbon consumers can interpret historical records without using mutable defaults.

## 10. Privacy and security notes

- Raw location samples are transient sensitive input and are omitted from the operational storage example.
- `trip_id` must be opaque and must not embed user identity or location.
- Logs and failure reasons must not include coordinates, raw identifiers, access credentials, or full request payloads.
- Location retention, encryption, access control, deletion, and consent policies are not defined in this repository and need verification before implementation.
- User corrections must not flow automatically into a training dataset.

## 11. Implemented versus not implemented

Implemented for offline experiments:

- AI Hub GPS parsing and validity checks;
- strict and mild candidate-window generation scripts;
- Haversine-derived speed generation;
- candidate five-class ordering;
- Train-only scaling in experiment runners;
- pinned reference-model construction and batch evaluation;
- argmax prediction during Validation evaluation.

Not implemented as a CANOPY E2E path:

- Trip-ending event or Trip domain object;
- service/iPhone location adapter;
- reusable single-Trip preprocessor;
- inference-only model artifact loading;
- prediction/confidence object returned to an application;
- API/backend integration;
- prediction persistence or database migration;
- user confirmation/correction UI and endpoint;
- confirmed-mode persistence;
- carbon calculation using confirmed mode;
- Azure deployment or resource integration.

Needs verification before implementation:

- target application/backend repository and stack;
- canonical Trip and location schema;
- deployable trained model artifact and ownership;
- model confidence semantics;
- supported product-mode enum and whether it exactly matches the candidate model enum;
- location privacy/retention policy;
- actual iPhone GPS sampling characteristics;
- carbon-calculation contract.

## 12. Next implementation order

### P0: one Trip to a prediction object

1. Verify the target application repository, language, Trip type, and location-sample type.
2. Extract the audited STRICT candidate logic into a side-effect-free single-Trip preprocessor with typed results and no automatic fallback relaxation.
3. Define a privacy-safe test fixture and tests for `READY`, `INSUFFICIENT_DATA`, and `PREPROCESSING_REJECTED` without using Internal Test.
4. Obtain and version an approved compatible model artifact; implement an inference-only adapter that accepts exactly 200 speed values.
5. Return a `ModelPredictionResult` without database, UI, or Azure assumptions. Keep confidence null unless its semantics are separately implemented and verified.
6. Demonstrate one fixture through preprocessing to prediction-object creation and record exact versions.

P0 is blocked from a real model prediction until a compatible deployable model artifact is identified. The repository currently supports model construction and experiment-time in-memory checkpoints, not loading a persisted production artifact.

### P1: persistence and user confirmation

1. In the verified target backend, add prediction and confirmation persistence with separate fields.
2. Add idempotent prediction-write and explicit confirmation/correction operations.
3. Connect the product UI only after authentication, authorization, retry, and audit behavior are specified.
4. Preserve model output when a user corrects it; do not turn corrections into training labels automatically.

### P2: confirmed mode downstream

1. Define the carbon-calculation interface and factor provenance.
2. Consume only explicitly confirmed mode where product policy requires confirmation.
3. Add Behavior Change or other downstream features after the carbon and privacy contracts are verified.

Azure deployment is not placed in P0 because no Azure implementation or selected hosting architecture exists in this repository.
