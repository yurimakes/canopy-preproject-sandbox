# Three-Arm Release Comparison - 2026-09-08

## 목적

STRICT_BASE 500, STRICT_EXPANDED 645, MILD_EXPANDED 645를 동일한
Validation 150 samples에서 비교하여 Train sample 수와 release policy의 영향을
구분하는 단일-seed controlled diagnostic이다.

## 실행 환경과 Provenance

- Python: 3.13.15
- PyTorch: 2.11.0+cu128
- CUDA available: True
- GPU: Tesla T4
- device: CUDA
- reference model: `models/transformer/model_utils.py:TrajectoryTransformer`
- reference commit: `2c0e7ac2aa52813c0899b2f8ba7c6394ace2e959`
- raw log: `reports/ml/19_three_arm_release_comparison_20260908.txt`

로그 시작 시 CUBLASLT 요청 workspace가 설정된 CUBLAS workspace보다 크다는
PyTorch warning이 1회 기록되었다. 학습은 세 arm 모두 완료되어
`TRAINING_STATUS=COMPLETED`를 출력했다.

## Dataset / Split 검증

| Arm | Train | Per class | Validation | Per class |
|---|---:|---:|---:|---:|
| STRICT_BASE | 500 | 100 | 150 | 30 |
| STRICT_EXPANDED | 645 | 129 | 150 | 30 |
| MILD_EXPANDED | 645 | 129 | 150 | 30 |

- 세 arm Validation sample ID, feature, label exact equality: True
- STRICT_EXPANDED mild-release samples: 0
- StandardScaler: 각 arm의 Train에만 fit
- Internal Test: 미사용

Dataset SHA256:

- STRICT_BASE: `0BEE39C8C9FEAEF6E1F4C66CAAAD25A064D9CB4128C4F602FBEEF51FF021F9DC`
- STRICT_EXPANDED: `54CC016782D14343578E738C84B0CC2AC3CBFFE1871C3A1D005F87B272229A8B`
- MILD_EXPANDED: `2C3A968D209C41ED23C89935E65D441946E2FE344E862B29E4954444B82F43AC`

## 고정 학습 설정

- seed: 316
- window_size: 200
- feature_size: 1
- num_classes: 5
- d_model: 128
- nhead: 8
- num_layers: 4
- kv_heads: 4
- dropout: 0.1
- model parameters: 728,494
- batch_size: 64
- learning_rate: 2e-4
- weight_decay: 1e-4
- gradient_clip: 1.0
- max_epochs: 50
- patience: 7
- criterion: CrossEntropyLoss
- optimizer: AdamW
- checkpoint / early-stopping criterion: Validation loss

각 arm 시작 전 Python, NumPy, PyTorch, CUDA 및 Train DataLoader generator seed를
reset하는 deterministic runner를 사용했다.

## Best Validation-Loss Checkpoints

| Arm | Best epoch | Stop epoch | Validation loss | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|
| STRICT_BASE | 14 | 21 | 0.933921 | 0.693333 | 0.684480 |
| STRICT_EXPANDED | 13 | 20 | 0.889174 | 0.700000 | 0.694504 |
| MILD_EXPANDED | 24 | 31 | 0.842050 | 0.726667 | 0.720097 |

## STRICT_BASE Class Metrics

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| WALK | 0.743590 | 0.966667 | 0.840580 |
| BIKE | 0.880000 | 0.733333 | 0.800000 |
| CAR | 0.440000 | 0.366667 | 0.400000 |
| BUS | 0.720000 | 0.600000 | 0.654545 |
| SUBWAY | 0.666667 | 0.800000 | 0.727273 |

Confusion matrix, rows=true and columns=predicted, class order
WALK / BIKE / CAR / BUS / SUBWAY:

```text
[[29, 0, 1, 0, 0],
 [ 2,22, 3, 3, 0],
 [ 3, 2,11, 4,10],
 [ 2, 1, 7,18, 2],
 [ 3, 0, 3, 0,24]]
```

## STRICT_EXPANDED Class Metrics

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| WALK | 0.763158 | 0.966667 | 0.852941 |
| BIKE | 0.920000 | 0.766667 | 0.836364 |
| CAR | 0.461538 | 0.400000 | 0.428571 |
| BUS | 0.633333 | 0.633333 | 0.633333 |
| SUBWAY | 0.709677 | 0.733333 | 0.721311 |

```text
[[29, 0, 1, 0, 0],
 [ 1,23, 3, 3, 0],
 [ 3, 1,12, 7, 7],
 [ 2, 1, 6,19, 2],
 [ 3, 0, 4, 1,22]]
```

## MILD_EXPANDED Class Metrics

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| WALK | 0.763158 | 0.966667 | 0.852941 |
| BIKE | 0.888889 | 0.800000 | 0.842105 |
| CAR | 0.565217 | 0.433333 | 0.490566 |
| BUS | 0.617647 | 0.700000 | 0.656250 |
| SUBWAY | 0.785714 | 0.733333 | 0.758621 |

```text
[[29, 0, 1, 0, 0],
 [ 1,24, 1, 4, 0],
 [ 3, 2,13, 7, 5],
 [ 2, 1, 5,21, 1],
 [ 3, 0, 3, 2,22]]
```

## Highest Observed Validation Macro F1

아래 값은 checkpoint selection metric이 아니라 epoch 중 관측된 diagnostic이다.
checkpoint는 계속 Validation loss로 선택했다.

| Arm | Epoch | Accuracy | Macro F1 |
|---|---:|---:|---:|
| STRICT_BASE | 7 | 0.726667 | 0.724061 |
| STRICT_EXPANDED | 20 | 0.733333 | 0.727674 |
| MILD_EXPANDED | 12 | 0.753333 | 0.746257 |

## Comparisons

### STRICT_EXPANDED - STRICT_BASE

- Accuracy: +0.006667
- Macro F1: +0.010025

| Class | Recall delta | F1 delta |
|---|---:|---:|
| WALK | +0.000000 | +0.012361 |
| BIKE | +0.033333 | +0.036364 |
| CAR | +0.033333 | +0.028571 |
| BUS | +0.033333 | -0.021212 |
| SUBWAY | -0.066667 | -0.005961 |

### MILD_EXPANDED - STRICT_BASE

- Accuracy: +0.033333
- Macro F1: +0.035617

| Class | Recall delta | F1 delta |
|---|---:|---:|
| WALK | +0.000000 | +0.012361 |
| BIKE | +0.066667 | +0.042105 |
| CAR | +0.066667 | +0.090566 |
| BUS | +0.100000 | +0.001705 |
| SUBWAY | -0.066667 | +0.031348 |

### MILD_EXPANDED - STRICT_EXPANDED: Equal-Size Comparison

두 arm은 모두 Train 645 = 129/class이며 Validation도 정확히 동일하다.

- Accuracy: +0.026667
- Macro F1: +0.025592

| Class | Recall delta | F1 delta |
|---|---:|---:|
| WALK | +0.000000 | +0.000000 |
| BIKE | +0.033333 | +0.005742 |
| CAR | +0.033333 | +0.061995 |
| BUS | +0.066667 | +0.022917 |
| SUBWAY | +0.000000 | +0.037309 |

## 해석

단일 seed, 소규모 동일 Validation diagnostic에서 MILD_EXPANDED가 같은 크기의
STRICT_EXPANDED보다 높은 Validation metric을 보였다.

이 결과는 mild release가 최종적으로 우수하거나 통계적으로 유의함을 의미하지 않는다.
AI Hub 최종 성능, CANOPY 서비스 성능 또는 논문 재현 성능도 아니다.

## 제한사항

- seed 316 한 번의 결과다.
- Validation은 class별 30 samples, 총 150 samples다.
- class Recall 0.033333 차이는 약 1 sample 차이에 해당한다.
- 동일 Validation을 반복 관찰하므로 후속 판단도 diagnostic 범위로 제한한다.
- CUBLAS workspace warning이 기록되었으며 결과 로그는 완료됐지만, warning이 없는
  별도 환경과의 bitwise equivalence는 확인하지 않았다.
- Internal Test는 사용하지 않았다.

## Next

STRICT_EXPANDED와 MILD_EXPANDED만 사전 고정한 5개 seed에서 재실행하여
equal-size 비교의 방향과 변동성을 기술 통계로 확인한다.
