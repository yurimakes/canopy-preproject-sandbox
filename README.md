# CANOPY Pre-project Test

Personal pre-project sandbox for validating CANOPY data and ML assumptions before work moves into a team production repository. It records reproducible audits, controlled diagnostics, and a minimal Trip inference contract—not a production service.

## Purpose

- Audit the structure, pairing, content quality, and leakage risks of AI Hub transportation-mode data.
- Build and verify a user-disjoint development split.
- Evaluate GPS continuity and speed preprocessing candidates without label-dependent filtering.
- Check whether speed-only inputs contain useful transportation-mode signal and whether the pinned SpeedTransformer reference can consume them.
- Run lightweight, controlled Validation diagnostics rather than claim final model performance.
- Validate a reusable single-Trip preprocessing and inference boundary before backend, storage, UI, or cloud integration.

## Current status

| Area | Status | Verified state |
|---|---|---|
| AI Hub data audit | Verified | Manifest, pairing, content, sequence, and leakage audits exist. The public Training/Validation data has no audited trajectory overlap but has 849 overlapping UIDs, so a separate UID-disjoint split was created. Its Internal Test is a CANOPY development split, not the official AI Hub Test set. |
| GPS preprocessing | Candidate | STRICT uses 201 valid points with exact 1-second adjacent gaps to produce 200 actual-time speed values, without interpolation, imputation, clipping, smoothing, or class-specific cutoffs. MILD permits a limited number of 2-second gaps as a coverage candidate; it is not adopted as a performance improvement. |
| Speed signal diagnostic | Verified | Random Forest statistics and controlled Transformer runs show that speed contains class-discrimination signal. These diagnostics do not establish final model superiority. |
| SpeedTransformer | Candidate | Experiments pin `TrajectoryTransformer` to reference commit `2c0e7ac2aa52813c0899b2f8ba7c6394ace2e959`. Paper results, reference reproduction, AI Hub diagnostics, and CANOPY service performance remain separate claims. |
| Trip inference P0 | Implemented (preproject) | Reusable single-Trip STRICT preprocessing, structured failure states, a prediction object, and `MODEL_NOT_RUN` behavior are implemented. An artifact loader/adapter is prepared, but no compatible trained model/scaler artifact exists locally, so real prediction has not run. |
| Artifact export | Prepared | The STRICT_LARGE seed-316 Colab exporter, manifest contract, hash validation, independent reload path, and CPU `--validate-only` check are prepared. Actual GPU training and artifact generation have not run. |
| Application E2E | Not yet implemented | Trip lifecycle, API/backend, persistence, user confirmation UI, carbon calculation, and Azure integration are not implemented in this repository. |

## Key findings

- The audited public AI Hub Training/Validation split has zero same-trajectory overlap but 849 overlapping user UIDs; controlled experiments therefore use the separate UID-disjoint development split.
- STRICT exact-1-second preprocessing is the current personal preproject baseline candidate, not a team-approved production policy.
- In the equal-size 645-Train, five-seed Validation diagnostic, MILD was higher in Accuracy and Macro F1 for 5/5 predeclared seeds.
- That direction did not reproduce in the equal-size 1,500-Train diagnostic: MILD was higher on both metrics in 1/5 seeds, with mean deltas of `-0.006667` Accuracy and `-0.006420` Macro F1. No statistical significance is claimed.
- Internal Test is not used for the current preprocessing or model-selection path. Validation has 30 samples per class, so one sample changes class recall by about `0.033333`.
- Single-Trip preprocessing works, but artifact-backed inference remains pending a real GPU-trained `model.pt`, fitted scaler, manifest, hash check, and independent reload verification.

## Repository structure

```text
scripts/           Data audits, dataset builders, controlled ML runners, artifact exporter
src/canopy_ml/     Reusable Trip preprocessing and SpeedTransformer artifact adapter
tests/             Privacy-safe synthetic contract and validation tests
reports/
  daily/           Dated project checkpoints
  data/            Data-quality and preprocessing evidence
  ml/              Controlled diagnostic logs and reports
  decisions/       Preproject working decisions and interface contracts
data/processed/    Generated/ignored processed datasets
splits/            Development split material; sensitive/generated files are ignored
```

Raw and private data is not part of the tracked repository.

## Important reports

- [Daily checkpoint](reports/daily/CANOPY_DAILY_20260908.md)
- [AI Hub data audit](reports/data/AIHUB_DATA_AUDIT_CHECKPOINT_20260907.md)
- [GPS preprocessing working decision](reports/decisions/GPS_CONTINUITY_PREPROCESSING_WORKING_DECISION_20260908.md)
- [ML Trip inference interface](reports/decisions/ML_TRIP_INFERENCE_INTERFACE_20260908.md)
- [Speed signal diagnostic](reports/ml/SPEED_SIGNAL_DIAGNOSTIC_20260908.md)
- [SpeedTransformer reference-capacity diagnostic](reports/ml/SPEEDTRANSFORMER_REFERENCE_CAPACITY_DIAGNOSTIC_20260908.md)
- [645-Train strict/mild robustness](reports/ml/MULTISEED_RELEASE_ROBUSTNESS_20260908.md)
- [1,500-Train strict/mild robustness](reports/ml/LARGE_STRICT_MILD_ROBUSTNESS_20260908.md)
- [Trip inference P0 checkpoint](reports/ml/P0_TRIP_STRICT_INFERENCE_20260908.md)
- [STRICT artifact export plan](reports/ml/STRICT_ARTIFACT_EXPORT_PLAN_20260908.md)

## Data and privacy policy

- Do not commit raw GPS data, raw UID/TID values, downloaded AI Hub archives, or reversible user/trajectory mappings.
- Keep generated processed datasets ignored unless a separately reviewed release explicitly requires them.
- Do not commit model checkpoints, fitted scaler artifacts, artifact ZIPs, or other generated model outputs.
- Do not commit secrets, API keys, credentials, `.env` files, private URLs, or personal local paths.
- Use deterministic synthetic or otherwise privacy-safe fixtures for repository tests.

## ML evaluation policy

- Validation is not Test; the current Internal Test remains outside preprocessing and model selection.
- Do not directly compare metrics produced from different datasets, splits, representations, or evaluation protocols.
- Keep the metric at the minimum-Validation-loss checkpoint separate from the highest observed Macro F1 diagnostic.
- Keep paper results, reference reproduction, AI Hub experiments, and CANOPY service results explicitly separate.
- All current metrics are preproject diagnostics, not production KPIs or claims of statistical significance.

## Next steps

P0 is to run the fixed STRICT_LARGE seed-316 export on a T4 GPU, generate `model.pt`, the fitted scaler and manifest, verify their hashes and fresh-process reload, and connect single-Trip preprocessing to a real prediction object.

P1 is to connect prediction storage and the separate user confirmation/correction contract. P2 is to let a later carbon-calculation flow consume the confirmed mode. P1 and P2 are planned work, not completed implementations.
