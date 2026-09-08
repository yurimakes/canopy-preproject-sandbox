# STRICT_LARGE candidate artifact export plan

Date: 2026-09-08

Status: export pipeline prepared and locally validated; GPU artifact not generated.

This checkpoint prepares a reproducible Colab GPU path for a preproject deployable-candidate artifact. It does not designate a production model, establish AI Hub final performance, or measure CANOPY service performance. No local training was run, and Internal Test was not used.

## Purpose

The next P0 step is to turn the existing single-Trip STRICT preprocessing result into an actual five-class prediction using a versioned model, fitted scaler, integrity manifest, and independent reload check. The resulting artifact remains a candidate until its Colab run succeeds and the downloaded output is verified locally.

## Fixed training input

- Dataset: `data/processed/speedtransformer_large_control_v1/speed200_strict_large_v1.csv`
- SHA256: `5CC827DAB88D17F68C5EF7F1A09127A81E8B5F683990EDAB0CC7D00302C275FA`
- Train: 1,500 samples, 300 per class
- Validation: 150 samples, 30 per class
- Source policy: all 1,650 samples are `strict_1s`
- Internal Test: absent and unused

The runner rejects any unknown split, unexpected class count, non-200-row sample, label mismatch, policy mismatch, or dataset hash mismatch before training.

## Seed decision

Seed `316` is fixed. It is the baseline seed repeatedly used from the project's earlier experiments; it is not the highest-performing seed selected after viewing the five-seed robustness results. No seed is selected by Validation performance in this export plan.

## Reference and protocol

- Reference repository: `https://github.com/othmaneechc/SpeedTransformer.git`
- Pinned commit: `2c0e7ac2aa52813c0899b2f8ba7c6394ace2e959`
- Architecture: `models/transformer/model_utils.py:TrajectoryTransformer`
- Window/feature/classes: `200 / 1 / 5`
- Transformer: `d_model=128`, `nhead=8`, `num_layers=4`, `kv_heads=4`, `dropout=0.1`
- Training: `batch_size=64`, `AdamW`, `learning_rate=2e-4`, `weight_decay=1e-4`, gradient clip `1.0`
- Stopping: maximum 50 epochs, patience 7
- Criterion: cross-entropy
- Checkpoint selection: minimum Validation loss

`scripts/23_train_export_strict_artifact.py` imports script 17 and asserts these constants before any training. The scaler is fitted only on the 300,000 scalar Train speed observations. Validation is transformed with that fitted scaler and is never used to fit it.

## Candidate label contract

| Index | Candidate model label |
|---:|---|
| 0 | WALK |
| 1 | BIKE |
| 2 | CAR |
| 3 | BUS |
| 4 | SUBWAY |

This mapping is a candidate model contract and is distinct from AI Hub raw/detail labels.

## Artifact contents

The Colab runner produces `speedtransformer_strict_v1_artifact.zip` with exactly:

- `model.pt`: CPU `state_dict` from the minimum-Validation-loss checkpoint;
- `scaler.json`: fitted StandardScaler semantics (`mean`, `scale`, `var`, `n_samples_seen`, feature metadata);
- `manifest.json`: compatibility, provenance, metrics, runtime versions, and model/scaler hashes;
- `training.log`: epoch metrics, fixed conditions, and independent reload verification output.

The artifact ZIP SHA256 is printed only after successful packaging. No ZIP hash is recorded here because the GPU output does not yet exist.

## Manifest compatibility contract

The loader requires:

- artifact version `speedtransformer-strict-large-seed316-v1` and status `preproject_candidate`;
- the exact pinned reference commit and architecture;
- the exact STRICT_LARGE dataset hash;
- preprocessing version `strict-speed200-candidate-v1`;
- the candidate five-class mapping and fixed seed status;
- exact Train/Validation counts and `internal_test_used=false`;
- the complete fixed hyperparameter set and `checkpoint_criterion=validation_loss`;
- finite Validation metrics and runtime versions;
- plain model/scaler filenames and matching uppercase SHA256 values;
- confidence semantics `uncalibrated_softmax_probability_of_predicted_class`.

An unknown artifact version, preprocessing mismatch, reference mismatch, label mismatch, hash mismatch, or malformed scaler is rejected before inference.

## Independent integrity check

After writing the real trained checkpoint, the exporter starts a fresh Python process. That process:

1. loads and validates the manifest and scaler;
2. recomputes model and scaler SHA256;
3. verifies the reference repository commit;
4. recreates `TrajectoryTransformer` with the manifest hyperparameters;
5. loads the saved `state_dict` strictly and switches to eval mode;
6. runs the existing STRICT preprocessor on a deterministic privacy-safe synthetic 201-point fixture;
7. performs one actual artifact-backed forward pass and requires finite logits of shape `(1, 5)`.

The exporter stops without packaging if this fresh-process verification fails. It does not use a random initialized model as a substitute.

## Inference adapter

`src/canopy_ml/speedtransformer_adapter.py` connects a `READY` `StrictPreprocessingResult` to the validated artifact. It applies the persisted one-feature StandardScaler, obtains logits, maps argmax through the candidate label contract, and writes the prediction separately from `user_confirmed_mode`.

The existing `model_confidence` field contains the predicted-class softmax probability. Its manifest semantics explicitly call it uncalibrated; calibration has not been evaluated. When the artifact path is missing, the existing `MODEL_NOT_RUN` / `MODEL_ARTIFACT_NOT_AVAILABLE` behavior remains unchanged.

## Local verification completed

Local checks cover CLI parsing, full dataset structure and SHA256, reference import and pinned commit, protocol equality with script 17, manifest/scaler schema, exact label mapping, missing-artifact behavior, model hash rejection, and preprocessing-version rejection. Temporary test-only bytes are created only inside an automatically removed temporary directory and are never represented as a trained model.

GPU training was deliberately not run locally. The real `model.pt`, fitted `scaler.json`, runtime manifest metrics/hashes, training log, artifact ZIP, and artifact ZIP SHA256 remain pending the Colab T4 run.
