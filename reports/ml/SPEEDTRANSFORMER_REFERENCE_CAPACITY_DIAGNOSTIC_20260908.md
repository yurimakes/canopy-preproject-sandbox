# SpeedTransformer Reference-Capacity Diagnostic - 2026-09-08

## 목적
AI Hub speed-only smoke dataset에서 reference에 가까운 모델 용량 및
학습 설정을 적용했을 때 Validation 성능과 클래스별 failure가 어떻게 변하는지 확인한다.

## Dataset / Split
- speed200_smoke_v1
- Train: 500
- Validation: 150
- Internal Test: 사용하지 않음
- CANOPY UID-disjoint split 유지
- 5 classes: WALK / BIKE / CAR / BUS / SUBWAY

## Configuration
- window_size: 200
- d_model: 128
- nhead: 8
- num_layers: 4
- kv_heads: 4
- dropout: 0.1
- batch_size: 64
- learning_rate: 2e-4
- weight_decay: 1e-4
- gradient_clip: 1.0
- max_epochs: 50
- patience: 7
- checkpoint criterion: Validation loss
- seed: 316
- GPU: Tesla T4

Reference commit:
2c0e7ac2aa52813c0899b2f8ba7c6394ace2e959

## Result

Early stop:
- epoch 38

Best Validation-loss checkpoint:
- epoch: 31
- val_loss: 0.667003
- val_accuracy: 0.7400
- val_macro_f1: 0.7427

Per-class:
| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| WALK | 0.8621 | 0.8333 | 0.8475 |
| BIKE | 0.9231 | 0.8000 | 0.8571 |
| CAR | 0.5429 | 0.6333 | 0.5846 |
| BUS | 0.6923 | 0.6000 | 0.6429 |
| SUBWAY | 0.7353 | 0.8333 | 0.7812 |

Confusion Matrix:
[[25, 0, 1, 1, 3],
 [1, 24, 3, 1, 1],
 [0, 1, 19, 6, 4],
 [0, 1, 10, 18, 1],
 [3, 0, 2, 0, 25]]

Highest observed Validation Macro F1:
- epoch: 30
- val_accuracy: 0.7667
- val_macro_f1: 0.7628

이 값은 진단용 관측값이며 checkpoint 선택 기준은 Validation loss였다.

## Comparison with Lightweight Longer Training

Lightweight:
- d_model 64
- 2 layers
- best val-loss checkpoint Macro F1: 0.6836

Reference-capacity/config:
- d_model 128
- 4 layers
- best val-loss checkpoint Macro F1: 0.7427

주의:
모델 크기 외에도 weight decay와 gradient clipping 조건이 함께 변경되었으므로
순수 capacity ablation으로 해석하지 않는다.

## Interpretation

Confirmed:
- 3-epoch smoke에서 보이지 않던 BUS/SUBWAY 예측이 longer/reference-style training에서 회복됨.
- speed-only 입력에도 Transformer가 학습할 수 있는 구분 신호가 존재함.
- reference-capacity/config에서 Validation metric이 추가 개선됨.

Not confirmed:
- 최종 AI Hub 성능
- paper reproduction
- V3 대비 우위
- RF 대비 우위
- CANOPY 서비스 성능
- 최적 architecture / epoch

Internal Test는 사용하지 않았다.

## Next
P0:
- strict 1-second continuous GPS 조건 release feasibility audit

P1:
- gap distribution 확인 후 released-window training 여부 결정

P2:
- stride=50 overlapping window 실험은 leakage/중복 영향 검토 후 진행
