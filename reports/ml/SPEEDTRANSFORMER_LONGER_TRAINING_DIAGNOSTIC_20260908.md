# SpeedTransformer Longer Training Diagnostic - 2026-09-08

## 목적

초기 3-epoch SpeedTransformer smoke run에서 BUS/SUBWAY가 거의 예측되지 않은 원인 중
짧은 학습 시간(undertraining)의 영향을 분리하여 확인한다.

## Controlled Conditions

기존 window=200 smoke run과 동일하게 유지:

- Dataset: speed200_smoke_v1
- Train: 500 samples
- Validation: 150 samples
- UID-disjoint split
- Internal Test: 사용하지 않음
- window_size: 200
- feature_size: 1
- num_classes: 5
- d_model: 64
- nhead: 8
- num_layers: 2
- kv_heads: 4
- dropout: 0.1
- batch_size: 64
- learning_rate: 2e-4
- seed: 316
- Train-only StandardScaler

변경한 주요 변수:
- 최대 epoch: 3 -> 50
- patience: 7
- checkpoint criterion: Validation loss

## Reproducibility Check

이번 longer run의 epoch 1~3 결과가 기존 window=200 controlled screening과 일치했다.

Epoch 1:
- val_loss: 1.592431
- val_accuracy: 0.3067
- val_macro_f1: 0.1774

Epoch 2:
- val_loss: 1.430112
- val_accuracy: 0.4400
- val_macro_f1: 0.3539

Epoch 3:
- val_loss: 1.347026
- val_accuracy: 0.4267
- val_macro_f1: 0.3240

따라서 동일한 초기 조건에서 학습을 연장한 진단으로 볼 수 있다.

## Longer Training Observation

Epoch 7:
- val_accuracy: 0.7200
- val_macro_f1: 0.7141

이는 관측된 epoch 중 높은 Macro F1 값이지만,
reference checkpoint policy 기준 best checkpoint를 의미하지 않는다.

## Reference-style Validation-loss Checkpoint

Best validation-loss checkpoint:
- epoch: 34
- val_loss: 0.862642
- val_accuracy: 0.6933
- val_macro_f1: 0.6836

Per-class:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| WALK | 0.8667 | 0.8667 | 0.8667 |
| BIKE | 0.9231 | 0.8000 | 0.8571 |
| CAR | 0.5625 | 0.3000 | 0.3913 |
| BUS | 0.5676 | 0.7000 | 0.6269 |
| SUBWAY | 0.5854 | 0.8000 | 0.6761 |

Confusion Matrix:

[[26, 0, 1, 0, 3],
 [1, 24, 0, 4, 1],
 [0, 1, 9, 10, 10],
 [1, 1, 4, 21, 3],
 [2, 0, 2, 2, 24]]

Early stopping:
- stop epoch: 41
- patience: 7
- best validation-loss epoch: 34

Best epoch와 stop epoch는 서로 다르다.

## Interpretation

Confirmed:
- 초기 3-epoch 결과는 longer run의 첫 3 epoch와 재현됨.
- 학습을 연장하자 BUS와 SUBWAY 예측이 회복됨.
- Validation-loss checkpoint 기준 BUS Recall 0.70, SUBWAY Recall 0.80이 관측됨.
- 따라서 초기 BUS/SUBWAY failure에는 짧은 3-epoch 학습이 크게 영향을 준 것으로 판단할 근거가 있음.

Not confirmed:
- 34 epoch가 최적 epoch
- 50 epoch가 최적 학습 길이
- SpeedTransformer의 최종 AI Hub 성능
- RF보다 우수하거나 열등함
- V3보다 우수함
- CANOPY 서비스 성능

## Important Metric Distinction

Highest observed Macro F1과 reference checkpoint는 다름.

- observed high Macro F1:
  - epoch 7
  - 0.7141

- reference-style best Validation-loss checkpoint:
  - epoch 34
  - Macro F1 0.6836

모델 선택 기준을 혼동하지 않는다.

## Next

P0:
- 동일 데이터/UID split에서 reference-capacity architecture
  (d_model=128, num_layers=4) 진단

P1:
- lightweight architecture와 reference-capacity architecture 비교
- Validation loss / Macro F1 / class별 metric 함께 기록

Internal Test는 모델/설정 선택 전까지 사용하지 않는다.
