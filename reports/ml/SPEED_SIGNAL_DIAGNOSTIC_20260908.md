# Speed-only Signal Diagnostic - 2026-09-08

## 목적
SpeedTransformer smoke run에서 BUS/SUBWAY 성능이 낮은 원인이
speed 데이터 자체의 정보 부족인지 우선 진단한다.

## 조건
- 기존 CANOPY UID-disjoint split 유지
- Train: 500 samples
- Validation: 150 samples
- Internal Test: 사용하지 않음
- 동일 speed200 smoke dataset 사용
- 5 classes: WALK / BIKE / CAR / BUS / SUBWAY

## Speed Distribution Audit

Train median speed (km/h):
- WALK: 4.070
- BIKE: 13.717
- BUS: 6.349
- CAR: 23.663
- SUBWAY: 34.844

클래스별 speed 분포에 상당한 overlap은 존재하지만,
speed 정보에 클래스 구분 신호가 전혀 없는 상태는 아님.

### Extreme speed observations
200 km/h 초과:
- Train SUBWAY: 6 / 20,000 rows
- Validation SUBWAY: 3 / 6,000 rows
- 다른 Train/Validation class: 0

200 km/h 초과 값은 존재하지만 매우 희소하므로,
현재 SUBWAY Recall=0의 주원인이라고 확정할 근거는 없음.

## Random Forest Diagnostic Baseline

Speed sequence에서 다음 13개 sample-level 통계 feature 사용:
- mean
- median
- std
- p90
- p95
- p99
- max
- zero percentage
- >30 / >50 / >80 / >100 / >200 km/h percentage

Model:
- RandomForestClassifier
- n_estimators: 300
- random_state: 316

Validation result:
- Accuracy: 0.7600
- Macro F1: 0.7568

Per-class Recall:
- WALK: 0.8333
- BIKE: 0.8000
- CAR: 0.5333
- BUS: 0.8000
- SUBWAY: 0.8333

Confusion Matrix:
[[25, 1, 1, 0, 3],
 [1, 24, 1, 4, 0],
 [1, 1, 16, 6, 6],
 [0, 2, 2, 24, 2],
 [3, 0, 2, 0, 25]]

Top observed feature importances:
- p95_speed: 0.1341
- p90_speed: 0.1326
- p99_speed: 0.1300
- mean_speed: 0.1178
- over_30_pct: 0.1027
- std_speed: 0.1018

## Interpretation

Confirmed:
- 같은 speed dataset에서 simple statistical features만으로도
  BUS와 SUBWAY를 포함한 클래스 구분 신호가 관측됨.
- 따라서 현재 SpeedTransformer smoke failure를
  "speed-only 데이터에 정보가 없기 때문"이라고 단정할 수 없음.

Current hypothesis:
- Transformer smoke run의 짧은 학습량
- lightweight model configuration
- sequence representation / optimization
등을 우선 점검할 가치가 있음.

Not confirmed:
- Random Forest가 SpeedTransformer보다 최종적으로 우수함
- 최종 AI Hub 성능
- CANOPY 서비스 성능
- 최적 preprocessing
- 최적 model

RF와 Transformer는 feature representation이 다르므로
현재 metric을 최종 모델 비교 수치로 직접 사용하지 않는다.

## Next
1. SpeedTransformer reference 학습 hyperparameter 확인
2. smoke configuration과 reference configuration 차이 정리
3. 동일 UID split에서 longer controlled training 설계
4. Validation으로만 학습 상태 판단
5. Internal Test는 모델/설정 선택 완료 전까지 사용하지 않음
