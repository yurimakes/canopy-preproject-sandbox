# GPS Continuity Release Feasibility Audit - 2026-09-08

## 목적
SpeedTransformer용 200-speed 입력에서 기존 strict 조건
(201 valid GPS points, 모든 인접 gap = 1초)을 완화할 가치가 있는지 확인한다.

## 조건
- 5 classes: WALK / BIKE / CAR / BUS / SUBWAY
- CANOPY UID-disjoint split 유지
- 결측 보간 없음
- label-dependent speed filtering 없음
- released window 학습은 아직 수행하지 않음

## Valid-point gap characteristics

인접 valid GPS edge 중 정확히 1초인 비율:
- WALK: 98.9764%
- BIKE: 98.9752%
- CAR: 99.0575%
- BUS: 99.0250%
- SUBWAY: 98.4505%

대부분의 edge는 1초이지만,
201-point 연속 run 조건에서는 일부 gap 때문에 TID 전체 eligibility가 크게 감소한다.

## Train feasibility

| Policy | Eligible TIDs |
|---|---:|
| strict 1s | 4,990 |
| max gap 2s | 5,808 |
| max gap 5s | 5,921 |
| max gap 10s | 6,009 |
| max gap 30s | 6,163 |
| max gap 60s | 6,236 |
| unbounded | 7,935 |

strict -> 2s:
- +818 TIDs
- 약 +16.4%

2s -> 5s:
- +113 TIDs

## Validation feasibility

| Policy | Eligible TIDs |
|---|---:|
| strict 1s | 631 |
| max gap 2s | 725 |
| max gap 5s | 734 |
| max gap 10s | 748 |
| max gap 30s | 762 |
| max gap 60s | 767 |
| unbounded | 1,031 |

strict -> 2s:
- +94 TIDs
- 약 +14.9%

2s -> 5s:
- +9 TIDs

## SUBWAY

Train:
- strict: 489
- 2s: 598
- 5s: 630

Validation:
- strict: 59
- 2s: 71
- 5s: 75

SUBWAY 역시 strict 1초 조건 완화로 추가 확보 가능성이 확인됐다.

## Decision

첫 release 실험 후보:
- max adjacent valid-point gap = 2 seconds

근거:
- strict 대비 Train/Validation 확보량 증가가 의미 있음
- 5초 이상으로 추가 완화했을 때 전체 추가 이득은 상대적으로 작음
- 큰 GPS gap까지 허용할 근거는 아직 없음

Internal Test는 release cap 선택 근거로 사용하지 않는다.

## Not confirmed
- 2초 release가 모델 성능을 개선함
- 2초가 최종 preprocessing 기준임
- 큰 gap을 연결해도 안전함
- CANOPY 실제 iPhone sampling 조건에 적합함

## Next
1. strict에서는 탈락하지만 2초 release에서 새로 확보되는 TID 특성 확인
2. 실제 selected window의 2초 gap 개수와 wall-clock span 확인
3. 이상 없을 경우 max_gap=2s released smoke dataset 생성
4. 동일 UID split에서 Validation 기반 비교
