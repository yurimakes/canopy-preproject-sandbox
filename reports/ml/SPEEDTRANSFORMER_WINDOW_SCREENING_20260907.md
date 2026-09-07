# SpeedTransformer Window Size Controlled Screening - 2026-09-07

## 목적

AI Hub 기반 SpeedTransformer smoke dataset에서
window size 60 / 100 / 120 / 200의 영향을 동일 조건으로 짧게 screening한다.

이 실험은 최종 성능 비교나 최적 window 확정 실험이 아니다.

---

## Dataset

Source:
- AI Hub 교통수단 판별 GPS data

Smoke dataset:
- speed200_smoke_v1

Classes:
- 0 WALK
- 1 BIKE
- 2 CAR
- 3 BUS
- 4 SUBWAY

Split:
- Train: 500 samples
  - class별 100
- Validation: 150 samples
  - class별 30
- Internal Test: 150 samples
  - 이번 window screening에서는 사용하지 않음

UID policy:
- CANOPY uid_split_v1 사용
- Train / Validation / Internal Test UID overlap = 0
- reference DataProcessor의 random traj_id split은 사용하지 않음

---

## Controlled Comparison Design

모든 window에서 동일하게 유지:

- 동일 Train sample IDs
- 동일 Validation sample IDs
- 동일 UID-disjoint split
- 동일 5-class label mapping
- 동일 seed: 316
- 동일 optimizer: AdamW
- 동일 learning rate: 2e-4
- 동일 batch size: 64
- 동일 epochs: 3
- 동일 model configuration
  - feature_size: 1
  - num_classes: 5
  - d_model: 64
  - nhead: 8
  - num_layers: 2
  - kv_heads: 4
  - dropout: 0.1
- StandardScaler는 각 실험의 Train 값에만 fit
- Validation은 평가에만 사용
- Internal Test는 window 선택에 사용하지 않음

변경 변수:
- window_size = 60 / 100 / 120 / 200

100보다 짧은 window도 별도 sample을 재추출하지 않고,
동일한 clean 200-speed sample의 앞부분 N개를 사용했다.

따라서 이번 결과는 동일 sample에서 window 길이만 변경한
controlled smoke screening이다.

---

## Results

### Window 60

Observed validation results:

Epoch 1:
- train_loss: 1.788340
- val_loss: 1.597642
- val_accuracy: 0.2133
- val_macro_f1: 0.1228

Epoch 2:
- train_loss: 1.545312
- val_loss: 1.450542
- val_accuracy: 0.4267
- val_macro_f1: 0.3293

Epoch 3:
- train_loss: 1.465500
- val_loss: 1.394938
- val_accuracy: 0.4133
- val_macro_f1: 0.3004

Highest observed validation Macro F1:
- epoch: 2
- accuracy: 0.4267
- macro_f1: 0.3293

Per-class at that observed epoch:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| WALK | 0.5263 | 1.0000 | 0.6897 |
| BIKE | 0.4074 | 0.7333 | 0.5238 |
| CAR | 0.3226 | 0.3333 | 0.3279 |
| BUS | 0.2500 | 0.0667 | 0.1053 |
| SUBWAY | 0.0000 | 0.0000 | 0.0000 |

Confusion Matrix:

[[30,  0,  0,  0,  0],
 [ 7, 22,  1,  0,  0],
 [ 6, 12, 10,  2,  0],
 [10, 15,  3,  2,  0],
 [ 4,  5, 17,  4,  0]]

---

### Window 100

Highest observed validation Macro F1:
- epoch: 2
- accuracy: 0.4400
- macro_f1: 0.3333

Per-class:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| WALK | 0.5882 | 1.0000 | 0.7407 |
| BIKE | 0.4561 | 0.8667 | 0.5977 |
| CAR | 0.3226 | 0.3333 | 0.3279 |
| BUS | 0.0000 | 0.0000 | 0.0000 |
| SUBWAY | 0.0000 | 0.0000 | 0.0000 |

Confusion Matrix:

[[30,  0,  0,  0,  0],
 [ 3, 26,  1,  0,  0],
 [ 6, 11, 10,  3,  0],
 [ 9, 17,  4,  0,  0],
 [ 3,  3, 16,  8,  0]]

---

### Window 120

Highest observed validation Macro F1:
- epoch: 2
- accuracy: 0.4333
- macro_f1: 0.3342

Per-class:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| WALK | 0.6250 | 1.0000 | 0.7692 |
| BIKE | 0.4500 | 0.9000 | 0.6000 |
| CAR | 0.3478 | 0.2667 | 0.3019 |
| BUS | 0.0000 | 0.0000 | 0.0000 |
| SUBWAY | 0.0000 | 0.0000 | 0.0000 |

Confusion Matrix:

[[30,  0,  0,  0,  0],
 [ 2, 27,  1,  0,  0],
 [ 6, 11,  8,  5,  0],
 [ 7, 19,  4,  0,  0],
 [ 3,  3, 10, 14,  0]]

---

### Window 200

Highest observed validation Macro F1:
- epoch: 2
- accuracy: 0.4400
- macro_f1: 0.3539

Per-class:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| WALK | 0.7317 | 1.0000 | 0.8451 |
| BIKE | 0.4590 | 0.9333 | 0.6154 |
| CAR | 0.3000 | 0.2000 | 0.2400 |
| BUS | 0.0714 | 0.0667 | 0.0690 |
| SUBWAY | 0.0000 | 0.0000 | 0.0000 |

Confusion Matrix:

[[30,  0,  0,  0,  0],
 [ 1, 28,  0,  1,  0],
 [ 4, 10,  6, 10,  0],
 [ 3, 22,  3,  2,  0],
 [ 3,  1, 11, 15,  0]]

---

## Screening Summary

| Window | Highest observed epoch | Validation Accuracy | Validation Macro F1 |
|---:|---:|---:|---:|
| 60 | 2 | 0.4267 | 0.3293 |
| 100 | 2 | 0.4400 | 0.3333 |
| 120 | 2 | 0.4333 | 0.3342 |
| 200 | 2 | 0.4400 | 0.3539 |

Within this specific 3-epoch controlled screening,
window 200 produced the highest observed Validation Macro F1.

However, this difference is not sufficient to establish that
window 200 is the optimal window size.

---

## Failure Cases

### SUBWAY

All four window sizes produced:

- SUBWAY Recall = 0
- SUBWAY F1 = 0

The model did not predict SUBWAY in the selected validation checkpoint
for any tested window.

This is an important failure signal requiring investigation.

It is NOT evidence that SpeedTransformer is inherently incapable of
classifying subway transportation.

### BUS

BUS performance was also unstable and weak.

Observed Recall:

- window 60: 0.0667
- window 100: 0.0000
- window 120: 0.0000
- window 200: 0.0667

CAR / BUS / SUBWAY confusion therefore requires further analysis.

---

## Interpretation

Confirmed:
- SpeedTransformer can train with window sizes 60 / 100 / 120 / 200.
- Controlled validation screening was completed under the same sample IDs and split.
- Among the four observed runs, window 200 had the highest Validation Macro F1.
- BUS and especially SUBWAY are major failure cases in this smoke configuration.
- Internal Test was not used for window selection.

Not confirmed:
- optimal window size
- final AI Hub performance
- paper reproduction
- superiority over V3
- CANOPY service performance
- optimal epoch or early-stopping policy

The epoch-2 values are only the highest observed values among three smoke epochs.
They must not be described as confirmed best epochs.

---

## Comparison Caution

Team experiments using different:
- window construction
- preprocessing
- speed filtering
- split
- sample counts
- epochs
- label naming

cannot be directly compared numerically with this screening.

Especially, AI Hub train / subway label semantics must be aligned before
rail-class metrics are compared.

---

## Next Priority

P0:
- investigate BUS / SUBWAY failure
- verify SpeedTransformer reference hyperparameters and training conditions
- determine whether the failure comes from:
  - speed-only feature limitation
  - preprocessing
  - insufficient smoke training
  - class distribution / sampling
  - dense GPS dataset characteristics

P1:
- construct a realistic short-window dataset where 60-speed sequences do not
  require the TID to first qualify for a clean 200-speed sequence

P2:
- compare against V3 only after identical dataset / UID split / labels /
  metrics / postprocessing conditions are prepared
