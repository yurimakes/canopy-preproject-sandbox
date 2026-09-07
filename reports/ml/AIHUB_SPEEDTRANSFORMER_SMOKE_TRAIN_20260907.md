# AI Hub SpeedTransformer Smoke Training - 2026-09-07

## 목적
AI Hub 기반 speed-only 데이터가 CANOPY UID-disjoint split을 유지한 채
SpeedTransformer의 실제 train / validation / internal-test pipeline을
끝까지 실행할 수 있는지 확인.

## 데이터
- Source: AI Hub 교통수단 판별 GPS data
- Dataset artifact: speed200_smoke_v1
- Generation script: scripts/12_make_speedtransformer_smoke_dataset.py
- Classes:
  - 0 WALK
  - 1 BIKE
  - 2 CAR
  - 3 BUS
  - 4 SUBWAY
- Train: 500 samples
  - class별 100
- Validation: 150 samples
  - class별 30
- Internal Test: 150 samples
  - class별 30
- Total: 800 samples
- Each sample: 200 speed values
- Input shape: (N, 200, 1)

Important:
- internal_test is a CANOPY internal split.
- It is NOT the AI Hub official Test set.

## Split
Existing CANOPY UID-disjoint split preserved.

The SpeedTransformer reference DataProcessor random traj_id split was not used.

## Preprocessing
- GPS 201 continuous 1-second observations -> 200 derived speed values
- Haversine distance
- speed unit: km/h
- no label-dependent speed threshold
- no missing-value imputation
- no speed clipping
- no smoothing
- one clean window per TID
- StandardScaler fit on Train only

## Reference Model
SpeedTransformer author-linked implementation.

Reference commit:
2c0e7ac2aa52813c0899b2f8ba7c6394ace2e959

## Smoke Configuration
- feature_size: 1
- num_classes: 5
- window_size: 200
- d_model: 64
- nhead: 8
- num_layers: 2
- kv_heads: 4
- dropout: 0.1
- optimizer: AdamW
- learning_rate: 2e-4
- batch_size: 64
- epochs: 3
- seed: 316
- runtime: Google Colab
- accelerator observed: Tesla T4

This configuration is for smoke testing only.
It is NOT a paper reproduction configuration or final hyperparameter set.

## Execution Result

### Epoch 1
- train_loss: 1.784906
- val_loss: 1.593141
- val_accuracy: 0.2600
- val_macro_f1: 0.1455

### Epoch 2
- train_loss: 1.529665
- val_loss: 1.433871
- val_accuracy: 0.4333
- val_macro_f1: 0.3585

### Epoch 3
- train_loss: 1.439258
- val_loss: 1.347264
- val_accuracy: 0.4333
- val_macro_f1: 0.3283

Among the three observed smoke epochs, epoch 2 had the highest validation
Macro F1. This must NOT be interpreted as a confirmed best epoch or
early-stopping result.

## Internal Test Smoke Diagnostic
- loss: 1.410795
- accuracy: 0.3800
- macro_f1: 0.2839

Confusion Matrix:
[[29,  1,  0,  0,  0],
 [10, 19,  1,  0,  0],
 [ 2, 17,  8,  3,  0],
 [ 6, 22,  1,  1,  0],
 [13,  2, 13,  2,  0]]

Predicted classes:
- 0
- 1
- 2
- 3

Class 4 (SUBWAY) was not predicted during this smoke run.

This is a failure signal to investigate, but it is NOT evidence that
SpeedTransformer cannot classify SUBWAY because this was only a small
3-epoch smoke experiment.

## Confirmed Result
AI Hub-derived speed-only data successfully passed:
1. data loading
2. Train-only scaling
3. SpeedTransformer forward
4. loss calculation
5. backward propagation
6. optimizer update
7. validation inference
8. internal-test inference
9. 5-class output generation

Result:
AIHUB_SPEEDTRANSFORMER_SMOKE_TRAIN_OK

## Not Established
- final AI Hub model performance
- paper reproduction
- GeoLife reproduction
- CANOPY service performance
- optimal window size
- optimal epoch
- optimal hyperparameters
- superiority over V3
- comparison with AI Hub official baseline

## Next
1. Controlled 100 / 120 / 200 window experiment on identical sample IDs and UID split
2. Inspect per-class Precision / Recall / F1, especially SUBWAY
3. Verify paper/reference training hyperparameters before larger experiment
4. Evaluate dense 1-second AI Hub GPS vs CANOPY service sampling-domain gap
5. Compare SpeedTransformer against V3 only under identical evaluation conditions
